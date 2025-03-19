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


def extract_labels(annotation_mapping, file_name, num_epochs):
    """Extract sleep stage labels from the corresponding TSV file."""
    if file_name not in annotation_mapping:
        print(f"Annotation file not found for {file_name}. Skipping labels.")
        return ["unknown"] * num_epochs

    annotation_file = annotation_mapping[file_name]
    if not os.path.exists(annotation_file):
        print(f"Annotation file missing: {annotation_file}. Skipping labels.")
        return ["unknown"] * num_epochs

    tsv_data = pd.read_csv(
        annotation_file, sep="\t", header=0,
        names=["start_time", "duration", "annotation"], skiprows=1
    )

    tsv_data["start_time"] = pd.to_numeric(tsv_data["start_time"], errors="coerce")
    tsv_data["duration"] = pd.to_numeric(tsv_data["duration"], errors="coerce")

    sleep_stage_data = tsv_data[tsv_data["annotation"].str.contains("Sleep stage", na=False)]
    labels = ["unknown"] * num_epochs
    epoch_duration = 30.0  # seconds

    for _, row in sleep_stage_data.iterrows():
        start_time, duration, sleep_stage = row["start_time"], row["duration"], row["annotation"].replace("Sleep stage ", "").strip()
        if pd.isna(start_time) or pd.isna(duration):
            continue

        start_epoch = int(start_time // epoch_duration)
        num_epochs_for_row = max(1, int(np.round(duration / epoch_duration)))

        for i in range(num_epochs_for_row):
            current_epoch = start_epoch + i
            if 0 <= current_epoch < num_epochs:
                labels[current_epoch] = sleep_stage

    return labels


def process_single_file(args):
    """Process a single EEG file with proper memory management."""
    fname, data_dir, select_ch, annotation_mapping = args
    psg_path = os.path.join(data_dir, fname)

    if not os.path.exists(psg_path):
        print(f"File {fname} not found. Skipping.")
        return None

    try:
        raw = read_raw_edf(psg_path, preload=False)  # Avoid loading full file into memory
        if select_ch not in raw.info['ch_names']:
            print(f"Channel {select_ch} not found in {fname}. Skipping.")
            return None

        # raw.pick([select_ch])
        eeg_signal = raw.get_data(picks=[select_ch], units='uV')[0]  # Load only one channel
        epochs = split_into_epochs(eeg_signal, TARGET_SAMPLING_RATE)

        if not epochs:
            print(f"Skipping {fname} due to lack of valid data.")
            return None

        labels = extract_labels(annotation_mapping, fname, len(epochs))
        labels = labels if labels else ["unknown"] * len(epochs)

        # Convert EEG data to lists before returning (avoid large numpy arrays)
        entry = {
            # "eeg_epochs": [epoch.tolist() for epoch in epochs],  
            "eeg_epochs": [np.array(epoch, dtype=np.float32).astype(np.float16).tolist() for epoch in epochs],
            "file_name": fname,
            "labels": labels,
        }
        
        if not isinstance(entry["eeg_epochs"], list) or not isinstance(entry["labels"], list):
            print(f"Invalid data format in {fname}. Skipping.")
            return None


        raw.close()
        del raw, eeg_signal, epochs
        gc.collect()  # Explicitly free memory

        return entry

    except Exception as e:
        print(f"Error processing {fname}: {e}")
        return None

def wrapper(args):
    """Worker function to process and save data directly to disk."""
    fname = args[0]
    print(f"Worker started for {fname}", flush=True)  # Debug start
    try:
        result = process_single_file(args)
        print(f"Worker finished for {fname}", flush=True)  # Debug end
        return result
    except Exception as e:
        print(f"Worker failed on {fname}: {e}", flush=True)
        return None



def process_nch_data(all_files, data_dir, select_ch, annotation_mapping, output_dir, num_workers=2):
    """Process EEG files in parallel in small batches."""
    print(f"Processing {len(all_files)} files with {num_workers} CPU cores...", flush=True)

    batch_size = max(2, len(all_files) // num_workers)  # Reduce batch size 5->2
    results = []
    
    with mp.Pool(num_workers, maxtasksperchild=5) as pool:
        for i in range(0, len(all_files), batch_size):
            batch = all_files[i:i+batch_size]
            batch_results = list(tqdm(
                pool.imap(wrapper, [(fname, data_dir, select_ch, annotation_mapping) for fname in batch]),
                total=len(batch),
                desc=f"Processing batch {i//batch_size + 1}"
            ))
            batch_output_path = os.path.join(output_dir, f"nch_sleep_data_batch_{i}.arrow")
            batch_results = [r for r in batch_results if r is not None]

            if batch_results:
                print(f"Sample batch entry before saving: {batch_results[0]}")
                save_to_arrow(batch_results, batch_output_path)  # Save each batch separately


            gc.collect()  # Free memory after each batch


    final_data = [r for r in results if r is not None]
    return final_data





def save_to_arrow(data_list, output_dir):
    """Save the prepared data to an Arrow file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Remove None entries
    data_list = [d for d in data_list if d is not None]

    if not data_list:
        print("Warning: No valid data to save. Skipping.")
        return

    try:
        path = os.path.join(output_dir, 'nch_sleep_data_all.arrow')
        ArrowWriter(compression="lz4").write_to_file(data_list, path=path)
        print(f"Data saved to {path}")
    except Exception as e:
        print(f"Error while saving to Arrow: {e}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/srv/scratch/speechdata/sleep_data/NCH/nch",
                        help="Directory with PSG files.")
    parser.add_argument("--output_dir", type=str, default="/srv/scratch/z5298768/chronos_classification/prepare_time_series/C4-M1_updated",
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

    save_to_arrow(final_data_list, args.output_dir)



if __name__ == "__main__":
    main()
