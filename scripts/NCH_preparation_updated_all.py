import argparse
import os
import numpy as np
import multiprocessing as mp
from gluonts.dataset.arrow import ArrowWriter
from tqdm.auto import tqdm
from mne.io import read_raw_edf
import sleep_study as ss
import gc
import pandas as pd
import pyarrow as pa
from collections import Counter

EPOCH_SEC_SIZE = ss.data.EPOCH_SEC_SIZE  # 30 seconds
TARGET_SAMPLING_RATE = ss.info.REFERENCE_FREQ  # 100Hz


def split_into_epochs(eeg_signal, sampling_rate, epoch_length_s=30):
    """Split EEG signal into 30-second epochs."""
    epoch_length = epoch_length_s * sampling_rate
    return [
        eeg_signal[i: i + epoch_length]
        for i in range(0, len(eeg_signal), epoch_length)
        if len(eeg_signal[i: i + epoch_length]) == epoch_length
    ]


def load_all_files(data_dir):
    """Load all EDF files in the specified directory."""
    all_files = [f for f in os.listdir(data_dir) if f.endswith(".edf")]
    print(f"Found {len(all_files)} EDF files in {data_dir}")
    return all_files


def load_annotation_mapping(data_dir):
    """Automatically find .tsv annotation files in the dataset directory."""
    mapping = {
        file.replace(".tsv", ".edf"): os.path.join(data_dir, file)
        for file in os.listdir(data_dir) if file.endswith(".tsv")
    }
    print(f"Automatically mapped {len(mapping)} annotation files.")
    return mapping


# def extract_labels(annotation_mapping, file_name, num_epochs):
#     """Extract sleep stage labels from the corresponding TSV file."""
#     if file_name not in annotation_mapping:
#         print(f"Annotation file not found for {file_name}. Skipping labels.")
#         return ["unknown"] * num_epochs

#     annotation_file = annotation_mapping[file_name]
#     if not os.path.exists(annotation_file):
#         print(f"Annotation file missing: {annotation_file}. Skipping labels.")
#         return ["unknown"] * num_epochs

#     tsv_data = pd.read_csv(
#         annotation_file, sep="\t", header=0,
#         names=["start_time", "duration", "annotation"], skiprows=1
#     )

#     tsv_data["start_time"] = pd.to_numeric(tsv_data["start_time"], errors="coerce")
#     tsv_data["duration"] = pd.to_numeric(tsv_data["duration"], errors="coerce")

#     sleep_stage_data = tsv_data[tsv_data["annotation"].str.contains("Sleep stage", na=False)]
#     labels = ["unknown"] * num_epochs
#     epoch_duration = 30.0  # seconds

#     for _, row in sleep_stage_data.iterrows():
#         start_time, duration, sleep_stage = row["start_time"], row["duration"], row["annotation"].replace("Sleep stage ", "").strip()
#         if pd.isna(start_time) or pd.isna(duration):
#             continue

#         start_epoch = int(start_time // epoch_duration)
#         num_epochs_for_row = max(1, int(np.round(duration / epoch_duration)))

#         for i in range(num_epochs_for_row):
#             current_epoch = start_epoch + i
#             if 0 <= current_epoch < num_epochs:
#                 labels[current_epoch] = sleep_stage

#     return labels

def extract_labels(annotation_mapping, file_name, num_epochs=None):
    """Extract sleep stage labels from .tsv, correctly sized based on annotation range."""
    if file_name not in annotation_mapping:
        print(f"Annotation file not found for {file_name}. Skipping labels.")
        return []

    annotation_file = annotation_mapping[file_name]
    if not os.path.exists(annotation_file):
        print(f"Annotation file missing: {annotation_file}. Skipping labels.")
        return []

    tsv_data = pd.read_csv(
        annotation_file, sep="\t", header=0,
        names=["start_time", "duration", "annotation"], skiprows=1
    )

    tsv_data["start_time"] = pd.to_numeric(tsv_data["start_time"], errors="coerce")
    tsv_data["duration"] = pd.to_numeric(tsv_data["duration"], errors="coerce")

    sleep_stage_data = tsv_data[tsv_data["annotation"].str.contains("Sleep stage", na=False)]
    
    # Infer number of epochs only from annotations if not provided
    epoch_duration = 30.0
    if num_epochs is None:
        max_end_time = (sleep_stage_data["start_time"] + sleep_stage_data["duration"]).max()
        num_epochs = int(max_end_time // epoch_duration)

    labels = ["unknown"] * num_epochs

    for _, row in sleep_stage_data.iterrows():
        start_time = row["start_time"]
        duration = row["duration"]
        label = row["annotation"].replace("Sleep stage ", "").strip()

        if pd.isna(start_time) or pd.isna(duration):
            continue

        start_epoch = int(start_time // epoch_duration)
        num_epochs_for_row = max(1, int(np.round(duration / epoch_duration)))

        for i in range(num_epochs_for_row):
            idx = start_epoch + i
            if 0 <= idx < num_epochs:
                labels[idx] = label

    return labels



def process_single_file(args):
    fname, data_dir, select_ch, annotation_mapping = args
    psg_path = os.path.join(data_dir, fname)

    try:
        raw = read_raw_edf(psg_path, preload=False)
        if select_ch not in raw.info['ch_names']:
            print(f"Channel {select_ch} not found in {fname}. Skipping.")
            return None

        eeg_signal = raw.get_data(picks=[select_ch], units='uV')[0]
        all_epochs = split_into_epochs(eeg_signal, TARGET_SAMPLING_RATE)
        total_eeg_epochs = len(all_epochs)

        labels_full = extract_labels(annotation_mapping, fname)
        total_label_epochs = len(labels_full)

        n_epochs = min(len(all_epochs), len(labels_full))
        if n_epochs < 240:
            print(f"{fname}: Not enough data ({n_epochs} epochs). Skipping.")
            return None

        start = (n_epochs - 240) // 2
        end = start + 240

        eeg_epochs = all_epochs[start:end]
        labels = labels_full[start:end]

        # DEBUG ZONE STARTS
        # if fname == "12160_4150.edf":
        #     print("=" * 50)
        #     print(f"DEBUG for {fname}")
        #     print(f"Total EEG epochs: {total_eeg_epochs}")
        #     print(f"Total label epochs: {total_label_epochs}")
        #     print(f"Min aligned length: {n_epochs}")
        #     print(f"Slice indices: start={start}, end={end}")
        #     print(f"Full label counts: {Counter(labels_full)}")
        #     print(f"Mid 2-hr label counts: {Counter(labels)}")
        #     print("=" * 50)
        # DEBUG ZONE ENDS

        if len(eeg_epochs) != len(labels):
            print(f"{fname}: Epoch-label mismatch ({len(eeg_epochs)} vs {len(labels)}). Skipping.")
            return None

        valid_labels = [l for l in labels if l not in ["unknown", "?"]]
        print(f"{fname}: valid labels in mid 2hrs = {len(valid_labels)}/{len(labels)}")

        if not valid_labels:
            print(f"{fname}: All labels are 'unknown' or '?'. Skipping.")
            return None

        entry = {
            "file_name": fname,
            "eeg_epochs": [[float(np.float16(x)) for x in epoch] for epoch in eeg_epochs],
            "labels": labels,
        }

        raw.close()
        return entry

    except Exception as e:
        print(f"Error processing {fname}: {e}")
        return None



    
def chunkify(lst, n):
    """Split list `lst` into `n` approximately equal parts."""
    return [lst[i::n] for i in range(n)]


def wrapper(args):
    file_list, data_dir, select_ch, annotation_mapping, worker_id, output_dir = args
    batch_size = 20
    results = []
    batch_index = 0

    for i, fname in enumerate(file_list):
        results.append((i, fname))  # store index + name

    total_files = len(results)
    file_batches = [results[i:i+batch_size] for i in range(0, total_files, batch_size)]

    for batch_index, batch in enumerate(file_batches):
        out_path = os.path.join(output_dir, f"nch_worker_{worker_id}_batch_{batch_index}.arrow")
        
        if os.path.exists(out_path):
            print(f"[Worker {worker_id}] ⏭️ Skipping batch {batch_index} (already saved)")
            continue

        batch_results = []
        for i, fname in batch:
            print(f"[Worker {worker_id}] Processing {fname}", flush=True)
            result = process_single_file((fname, data_dir, select_ch, annotation_mapping))
            if result:
                batch_results.append(result)

        if batch_results:
            save_to_arrow(batch_results, out_path)
            print(f"[Worker {worker_id}] Saved batch {batch_index} ({len(batch_results)} files) to {out_path}")
        else:
            print(f"[Worker {worker_id}] No valid results in batch {batch_index}")





def process_nch_data(all_files, data_dir, select_ch, annotation_mapping, output_dir, num_workers=2):
    """Distribute EEG files across workers and process in parallel."""
    print(f"Processing {len(all_files)} files with {num_workers} CPU cores...", flush=True)

    file_chunks = chunkify(all_files, num_workers)

    args_list = [
        (file_chunks[i], data_dir, select_ch, annotation_mapping, i, output_dir)
        for i in range(num_workers)
    ]

    with mp.Pool(num_workers, maxtasksperchild=1) as pool:
        output_files = list(tqdm(
            pool.imap(wrapper, args_list),
            total=num_workers,
            desc="Processing in parallel"
        ))

    print("All workers finished.")
    print("Output Arrow files:")
    for path in output_files:
        if path:
            print(f" - {path}")

    return output_files



def save_to_arrow(data_list, file_path):
    """Save a list of EEG entries to an Arrow file."""
    try:
        # Flatten to columns
        file_names = []
        eeg_epochs = []
        labels = []

        for entry in data_list:
            file_names.append(entry["file_name"])
            # Force all epochs to be lists of floats
            eeg = [[float(x) for x in epoch] for epoch in entry["eeg_epochs"]]
            eeg_epochs.append(eeg)
            labels.append(entry["labels"])
            
            # Debug
            print(f"eeg shape: {len(eeg)} x {len(eeg[0])} from file {entry['file_name']}")

        # Define a nested list-of-floats type explicitly
        list_of_floats = pa.list_(pa.float32())
        nested_list_type = pa.list_(list_of_floats)

        table = pa.table({
            "file_name": file_names,
            "eeg_epochs": pa.array(eeg_epochs, type=nested_list_type),
            "labels": pa.array(labels, type=pa.list_(pa.string()))
        })

        with pa.OSFile(file_path, 'wb') as sink:
            with pa.ipc.new_file(sink, table.schema) as writer:
                writer.write(table)

    except Exception as e:
        print(f"Error saving {file_path}: {e}")






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/srv/scratch/speechdata/sleep_data/NCH/nch",
                        help="Directory with PSG files.")
    parser.add_argument("--output_dir", type=str, default="/srv/scratch/z5298768/chronos_classification_local/prepare_time_series/C4-M1_updated_all",
                        help="Directory to save the arrow file.")
    parser.add_argument("--select_ch", type=str, default="EEG C4-M1",
                        help="EEG channel to select")
    parser.add_argument("--num_workers", type=int, default=8,  # Safe limit
                        help="Number of CPU workers for parallel processing.")

    args = parser.parse_args()

    print("Loading all EDF files...")
    all_files = load_all_files(args.data_dir)
    annotation_mapping = load_annotation_mapping(args.data_dir)

    if not all_files:
        print("Error: No EDF files found.")
        return

    final_data_list = process_nch_data(all_files, args.data_dir, args.select_ch, annotation_mapping, args.output_dir, args.num_workers)

    if not final_data_list:
        print("No data prepared. Exiting.")
        return

    # save_to_arrow(final_data_list, args.output_dir)



if __name__ == "__main__":
    main()
