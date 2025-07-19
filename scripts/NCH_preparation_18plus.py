#!/usr/bin/env python3

import argparse
import os
import numpy as np
import multiprocessing as mp
from tqdm.auto import tqdm
from mne.io import read_raw_edf
import sleep_study as ss
import pandas as pd
import pyarrow as pa
from collections import Counter

EPOCH_SEC_SIZE = ss.data.EPOCH_SEC_SIZE      # 30 seconds
TARGET_SAMPLING_RATE = ss.info.REFERENCE_FREQ  # 100Hz

# ── Load demographics and compute which patients are ≥ 18 ──
DEMOGRAPHIC_CSV = "/srv/scratch/z5298768/AttnSleep_data/Health_Data/DEMOGRAPHIC.csv"
demo_df = pd.read_csv(DEMOGRAPHIC_CSV)
demo_df["BIRTH_DATE"] = pd.to_datetime(
    demo_df["BIRTH_DATE"], format="%m/%d/%Y", errors="coerce"
)
today = pd.Timestamp.now().normalize()
demo_df["AGE_YEARS"] = (today - demo_df["BIRTH_DATE"]).dt.days / 365.25
adult_ids = set(
    demo_df.loc[demo_df["AGE_YEARS"] >= 18, "STUDY_PAT_ID"].astype(str)
)

def extract_patient_id(fname):
    """E.g. '12160_4150.edf' → '4150'. Adjust if your naming is different."""
    return os.path.splitext(fname)[0].split("_")[-1]

def split_into_epochs(eeg_signal, sampling_rate, epoch_length_s=30):
    """Return only full 30 s chunks."""
    epoch_len = epoch_length_s * sampling_rate
    return [
        eeg_signal[i : i + epoch_len]
        for i in range(0, len(eeg_signal), epoch_len)
        if len(eeg_signal[i : i + epoch_len]) == epoch_len
    ]

def load_all_files(data_dir):
    fns = [f for f in os.listdir(data_dir) if f.endswith(".edf")]
    print(f"Found {len(fns)} EDF files in {data_dir}")
    return fns

def load_annotation_mapping(data_dir):
    mapping = {
        fn.replace(".tsv", ".edf"): os.path.join(data_dir, fn)
        for fn in os.listdir(data_dir) if fn.endswith(".tsv")
    }
    print(f"Automatically mapped {len(mapping)} annotation files.")
    return mapping

def extract_labels(annotation_mapping, file_name, num_epochs=None):
    """Read the TSV and return a list of stage labels (one per 30 s)."""
    if file_name not in annotation_mapping:
        print(f"Annotation file not found for {file_name}.")
        return []
    tsv_path = annotation_mapping[file_name]
    if not os.path.exists(tsv_path):
        print(f"Annotation TSV missing: {tsv_path}.")
        return []

    df = pd.read_csv(tsv_path,
                     sep="\t", header=0,
                     names=["start_time","duration","annotation"],
                     skiprows=1)
    df["start_time"] = pd.to_numeric(df["start_time"], errors="coerce")
    df["duration"]   = pd.to_numeric(df["duration"], errors="coerce")
    stages = df[df["annotation"].str.contains("Sleep stage", na=False)]

    epoch_dur = 30.0
    if num_epochs is None:
        max_end = (stages["start_time"] + stages["duration"]).max()
        num_epochs = int(max_end // epoch_dur)

    labels = ["unknown"] * num_epochs
    for _, row in stages.iterrows():
        st = row["start_time"]
        dur = row["duration"]
        lab = row["annotation"].replace("Sleep stage ", "").strip()
        if pd.isna(st) or pd.isna(dur):
            continue
        start_epoch = int(st // epoch_dur)
        count = max(1, int(np.round(dur / epoch_dur)))
        for i in range(count):
            idx = start_epoch + i
            if 0 <= idx < num_epochs:
                labels[idx] = lab

    return labels

def process_single_file(args):
    fname, data_dir, select_ch, annotation_mapping = args

    # 1) filter out minors
    pid = extract_patient_id(fname)
    if pid not in adult_ids:
        print(f"Skipping {fname}: patient {pid} is under 18")
        return None

    psg_path = os.path.join(data_dir, fname)
    try:
        raw = read_raw_edf(psg_path, preload=False)

        # 2) ensure the channel exists
        if select_ch not in raw.info["ch_names"]:
            print(f"Channel {select_ch} not in {fname}.")
            raw.close()
            return None

        # pick the channel by index
        ch_idx = raw.ch_names.index(select_ch)
        eeg = raw.get_data(picks=[ch_idx], units="uV")[0]
        raw.close()

        # 3) epoch and label
        epochs = split_into_epochs(eeg, TARGET_SAMPLING_RATE)
        labels_full = extract_labels(annotation_mapping, fname)
        n = min(len(epochs), len(labels_full))
        if n < 240:
            print(f"{fname}: only {n} epochs (<240).")
            return None

        start = (n - 240) // 2
        end   = start + 240
        eeg_epochs = epochs[start:end]
        sliced     = labels_full[start:end]

        # 4) map N1/N2/N3 → non-REM
        labels = [
            "non-REM" if l in {"N1","N2","N3"} else l
            for l in sliced
        ]

        if len(eeg_epochs) != len(labels):
            print(f"{fname}: mismatch {len(eeg_epochs)} vs {len(labels)}.")
            return None

        valid = [l for l in labels if l not in {"unknown","?"}]
        print(f"{fname}: {len(valid)}/{len(labels)} valid labels.")

        if not valid:
            print(f"{fname}: all labels unknown/?.")
            return None

        return {
            "file_name": fname,
            "eeg_epochs": [[float(np.float16(x)) for x in ep] for ep in eeg_epochs],
            "labels": labels
        }

    except Exception as e:
        print(f"Error on {fname}: {e}")
        return None

def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]

def wrapper(args):
    file_list, data_dir, select_ch, annotation_mapping, worker_id, output_dir = args
    batch_size = 20
    saved_paths = []

    # pair each filename with its original index (optional)
    items = [(i,f) for i,f in enumerate(file_list)]
    batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]

    for bi, batch in enumerate(batches):
        out_path = os.path.join(output_dir, f"nch_worker_{worker_id}_batch_{bi}.arrow")
        if os.path.exists(out_path):
            print(f"[W{worker_id}] Skipping batch {bi}")
            saved_paths.append(out_path)
            continue

        results = []
        for _, fname in batch:
            print(f"[W{worker_id}] → {fname}")
            res = process_single_file((fname, data_dir, select_ch, annotation_mapping))
            if res:
                results.append(res)

        if results:
            save_to_arrow(results, out_path)
            print(f"[W{worker_id}] Saved batch {bi} ({len(results)} files).")
            saved_paths.append(out_path)
        else:
            print(f"[W{worker_id}] No valid files in batch {bi}.")

    return saved_paths

def process_nch_data(all_files, data_dir, select_ch, annotation_mapping, output_dir, num_workers=2):
    print(f"Launching {num_workers} workers for {len(all_files)} files…")
    chunks = chunkify(all_files, num_workers)
    args_list = [
        (chunks[i], data_dir, select_ch, annotation_mapping, i, output_dir)
        for i in range(num_workers)
    ]

    with mp.Pool(num_workers, maxtasksperchild=1) as pool:
        lists_of_paths = list(tqdm(
            pool.imap(wrapper, args_list),
            total=num_workers,
            desc="Parallel batches"
        ))

    # flatten
    all_paths = [p for sub in lists_of_paths for p in sub]
    print("Done. Arrow files written:")
    for p in all_paths:
        print(" -", p)
    return all_paths

def save_to_arrow(data_list, file_path):
    try:
        fnames, eepo, labs = [], [], []
        for e in data_list:
            fnames.append(e["file_name"])
            eepo.append(e["eeg_epochs"])
            labs.append(e["labels"])
            print(f"  • {e['file_name']}: {len(e['eeg_epochs'])}×{len(e['eeg_epochs'][0])}")

        nested = pa.list_(pa.list_(pa.float32()))
        tbl = pa.table({
            "file_name": fnames,
            "eeg_epochs": pa.array(eepo, type=nested),
            "labels": pa.array(labs, type=pa.list_(pa.string()))
        })

        with pa.OSFile(file_path, "wb") as sink:
            with pa.ipc.new_file(sink, tbl.schema) as writer:
                writer.write(tbl)

    except Exception as ex:
        print(f"Error saving {file_path}: {ex}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   type=str, default="/srv/scratch/speechdata/sleep_data/NCH/nch")
    p.add_argument("--output_dir", type=str, default="/srv/scratch/z5298768/chronos_classification_local/prepare_time_series/C4-M1_18plus")
    p.add_argument("--select_ch",  type=str, default="EEG C4-M1")
    p.add_argument("--num_workers",type=int, default=8)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    files = load_all_files(args.data_dir)
    mapping = load_annotation_mapping(args.data_dir)
    if not files:
        print("No EDFs found. Exiting.")
        return

    written = process_nch_data(files,
                               args.data_dir,
                               args.select_ch,
                               mapping,
                               args.output_dir,
                               args.num_workers)

    if not written:
        print("No batches written. Exiting.")

if __name__ == "__main__":
    main()
