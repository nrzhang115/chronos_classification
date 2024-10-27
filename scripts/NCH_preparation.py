# NCH Dataset Preparation 
# Extract the sleep stages in NCH dataset and convert into GluontTS-compatible arrow file 

import argparse
import glob
import os
import numpy as np
import pandas as pd
from gluonts.dataset.arrow import ArrowWriter
from tqdm.auto import tqdm
from mne.io import read_raw_edf
from datetime import datetime
import sleep_study as ss
import gc 

EPOCH_SEC_SIZE = ss.data.EPOCH_SEC_SIZE  # Use the predefined epoch size (30 seconds)
TARGET_SAMPLING_RATE = ss.info.REFERENCE_FREQ  # Use the predefined target sampling rate (100Hz)

# Sleep stage mappings
ann2label = {
    "Sleep stage W": 0,   # Wake
    "Sleep stage N1": 1,  # N1
    "Sleep stage N2": 2,  # N2
    "Sleep stage N3": 3,  # N3
    "Sleep stage R": 4,   # REM
}

def extract_sleep_stages(ann_file):
    """Extract sleep stages from the annotation file without segmenting into epochs."""
    df = pd.read_csv(ann_file, sep="\t")

    # Filter sleep stage rows
    # Filter only known sleep stages
    valid_stages = set(ann2label.keys())
    sleep_stages = df[df['description'].isin(valid_stages)]

    # Convert to labeled format, directly using known mappings
    labels = [(row['onset'], ann2label[row['description']]) for _, row in sleep_stages.iterrows()]

    return labels

def extract_start_time(ann_file):
    """Extract the start time of the sleep study from the 'Lights Off' event or the first event if 'Lights Off' is missing."""
    df = pd.read_csv(ann_file, sep="\t")

    # Look for the "Lights Off" event to define the start time
    lights_off_event = df[df['description'] == 'Lights Off']

    if not lights_off_event.empty:
        # Extract the onset time for the "Lights Off" event
        start_time = lights_off_event['onset'].values[0]
        return start_time
    else:
        # If no "Lights Off" event is found, use the very first event in the annotation file
        first_event = df['onset'].min()
        print(f"No 'Lights Off' event found in {ann_file}. Using the very beginning (onset={first_event}) as start time.")
        return first_event

def split_into_epochs(eeg_signal, sampling_rate, epoch_length_s=30):
    """
    Split EEG signal into 30-second epochs.
    """
    epoch_length = epoch_length_s * sampling_rate

    # Split the EEG signal into epochs
    epochs = [
        eeg_signal[i: i + epoch_length]
        for i in range(0, len(eeg_signal), epoch_length)
        if len(eeg_signal[i: i + epoch_length]) == epoch_length
    ]
    return epochs

def process_nch_data(psg_fnames, ann_fnames, select_ch, chunk_size=10):
    """
    Process the NCH dataset in chunks to save memory.
    """
    final_data_list = []  # Accumulate all chunks

    for chunk_start in range(0, len(psg_fnames), chunk_size):
        psg_chunk = psg_fnames[chunk_start:chunk_start + chunk_size]
        ann_chunk = ann_fnames[chunk_start:chunk_start + chunk_size]

        data_list = []  # Reset data_list for each chunk

        for psg_fname, ann_fname in zip(psg_chunk, ann_chunk):
            # Read EEG data using MNE
            raw = read_raw_edf(psg_fname, preload=True)

            # Skip if the selected channel does not exist
            if select_ch not in raw.info['ch_names']:
                print(f"Channel {select_ch} not found in {psg_fname}. Skipping this file.")
                continue

            # Pick the specific EEG channel
            raw.pick([select_ch])

            # Extract start time
            start_time_sec = extract_start_time(ann_fname)
            start_time = np.datetime64(int(start_time_sec), 's')

            # Extract sleep stage labels
            labels = extract_sleep_stages(ann_fname)

            if len(labels) == 0:
                print(f"Skipping {psg_fname} due to lack of valid data.")
                continue

            # Prepare the time series entry
            entry = {
                "start": start_time,
                "target": [label[1] for label in labels]
            }
            data_list.append(entry)

            # Clear memory for each study processed
            raw.close()
            del raw
            gc.collect()  # Explicitly free memory

        # Add chunk data to the final list
        final_data_list.extend(data_list)

    return final_data_list


def save_to_arrow(data_list, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define the path for the arrow file
    path = os.path.join(output_dir, 'nch_sleep_data.arrow')

    # Write the data list to an Arrow file
    ArrowWriter(compression="lz4").write_to_file(
        data_list,   # List of data entries (time series)
        path=path    # Output path for the arrow file
    )

    print(f"Data saved to {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/srv/scratch/speechdata/sleep_data/NCH/nch",
                        help="Directory with PSG and annotation files.")
    parser.add_argument("--output_dir", type=str, default="/srv/scratch/z5298768/chronos_classification/prepare_time_seires/C4-M1",
                        help="Directory to save the arrow file.")
    parser.add_argument("--select_ch", type=str, default="EEG C4-M1",
                        help="EEG channel to select")
    args = parser.parse_args()

    # Get all PSG (.edf) and annotation (.tsv) files
    psg_fnames = glob.glob(os.path.join(args.data_dir, "*.edf"))
    ann_fnames = glob.glob(os.path.join(args.data_dir, "*.tsv"))

    # Check if files are found
    print(f"Found {len(psg_fnames)} .edf files")
    print(f"Found {len(ann_fnames)} .tsv files")

    if len(psg_fnames) == 0 or len(ann_fnames) == 0:
        print("Error: No .edf or .tsv files found in the specified directory.")
        return

    psg_fnames.sort()
    ann_fnames.sort()

    # Process the NCH dataset in chunks and accumulate all data
    final_data_list = process_nch_data(psg_fnames, ann_fnames, args.select_ch)

    # Save as Arrow file once at the end
    save_to_arrow(final_data_list, args.output_dir)


if __name__ == "__main__":
    main()