import argparse
import glob
import os
import numpy as np
from gluonts.dataset.arrow import ArrowWriter
from tqdm.auto import tqdm
from mne.io import read_raw_edf
import sleep_study as ss
import gc

EPOCH_SEC_SIZE = ss.data.EPOCH_SEC_SIZE  # Use the predefined epoch size (30 seconds)
TARGET_SAMPLING_RATE = ss.info.REFERENCE_FREQ  # Use the predefined target sampling rate (100Hz)

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

def process_nch_data(selected_files, data_dir, select_ch):
    """
    Process only the selected files from the NCH dataset.
    """
    data_list = []  # Accumulate all processed data

    for fname in tqdm(selected_files, desc="Processing selected files"):
        psg_path = os.path.join(data_dir, fname)

        if not os.path.exists(psg_path):
            print(f"File {fname} not found in directory {data_dir}. Skipping this file.")
            continue

        # Read EEG data using MNE
        raw = read_raw_edf(psg_path, preload=True)

        # Skip if the selected channel does not exist
        if select_ch not in raw.info['ch_names']:
            print(f"Channel {select_ch} not found in {fname}. Skipping this file.")
            continue

        # Pick the specific EEG channel
        raw.pick([select_ch])

        # Extract EEG signal and split into 30-second epochs
        eeg_signal = raw.get_data()[0]  # single-channel data
        epochs = split_into_epochs(eeg_signal, TARGET_SAMPLING_RATE)

        if len(epochs) == 0:
            print(f"Skipping {fname} due to lack of valid data.")
            continue

        # Prepare the time series entry
        entry = {
            "eeg_epochs": epochs,  # Save the segmented EEG epochs
            "file_name": fname,  # Keep track of the source file
        }
        data_list.append(entry)

        # Clear memory for each study processed
        raw.close()
        del raw
        gc.collect()  # Explicitly free memory

    return data_list

def save_to_arrow(data_list, output_dir):
    """
    Save the prepared data to an Arrow file.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define the path for the Arrow file
    path = os.path.join(output_dir, 'nch_sleep_data_selected.arrow')

    # Write the data list to an Arrow file
    ArrowWriter(compression="lz4").write_to_file(
        data_list,   # List of data entries (time series)
        path=path    # Output path for the arrow file
    )

    print(f"Data saved to {path}")

def load_selected_files(file_path):
    """
    Load the list of selected files from a text file.
    """
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/srv/scratch/speechdata/sleep_data/NCH/nch",
                        help="Directory with PSG files.")
    parser.add_argument("--output_dir", type=str, default="/srv/scratch/z5298768/chronos_classification/prepare_time_seires/C4-M1_updated",
                        help="Directory to save the arrow file.")
    parser.add_argument("--select_ch", type=str, default="EEG C4-M1",
                        help="EEG channel to select")
    parser.add_argument("--selected_files", type=str, default="/home/z5298768/chronos_classification/scripts/Selected_Files",
                    help="Path to text file containing list of selected files.")
    args = parser.parse_args()

    # Load the selected file names
    selected_files = load_selected_files(args.selected_files)

    if len(selected_files) == 0:
        print("Error: No selected files found in the specified file.")
        return

    # Process only the selected files
    final_data_list = process_nch_data(selected_files, args.data_dir, args.select_ch)

    # Save as Arrow file once at the end
    save_to_arrow(final_data_list, args.output_dir)

if __name__ == "__main__":
    main()
