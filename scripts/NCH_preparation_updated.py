import argparse
import glob
import os
import numpy as np
from gluonts.dataset.arrow import ArrowWriter
from tqdm.auto import tqdm
from mne.io import read_raw_edf
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


def process_nch_data(psg_fnames, select_ch, chunk_size=10):
    """
    Process the NCH dataset in chunks to save memory.
    """
    final_data_list = []  # Accumulate all chunks

    for chunk_start in range(0, len(psg_fnames), chunk_size):
        psg_chunk = psg_fnames[chunk_start:chunk_start + chunk_size]

        data_list = []  # Reset data_list for each chunk

        for psg_fname in tqdm(psg_chunk, desc="Processing files"):
            # Read EEG data using MNE
            raw = read_raw_edf(psg_fname, preload=True)

            # Skip if the selected channel does not exist
            if select_ch not in raw.info['ch_names']:
                print(f"Channel {select_ch} not found in {psg_fname}. Skipping this file.")
                continue

            # Pick the specific EEG channel
            raw.pick([select_ch])

            # Extract EEG signal and split into 30-second epochs
            eeg_signal = raw.get_data()[0]  # single-channel data
            epochs = split_into_epochs(eeg_signal, TARGET_SAMPLING_RATE)

            if len(epochs) == 0:
                print(f"Skipping {psg_fname} due to lack of valid data.")
                continue

            # Prepare the time series entry
            entry = {
                "eeg_epochs": epochs,  # Save the segmented EEG epochs
                "file_name": os.path.basename(psg_fname),  # Keep track of the source file
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
    """
    Save the prepared data to an Arrow file.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define the path for the Arrow file
    path = os.path.join(output_dir, 'nch_sleep_data_updated.arrow')

    # Write the data list to an Arrow file
    ArrowWriter(compression="lz4").write_to_file(
        data_list,   # List of data entries (time series)
        path=path    # Output path for the arrow file
    )

    print(f"Data saved to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/srv/scratch/speechdata/sleep_data/NCH/nch",
                        help="Directory with PSG files.")
    parser.add_argument("--output_dir", type=str, default="/srv/scratch/z5298768/chronos_classification/prepare_time_seires/C4-M1",
                        help="Directory to save the arrow file.")
    parser.add_argument("--select_ch", type=str, default="EEG C4-M1",
                        help="EEG channel to select")
    args = parser.parse_args()

    # Get all PSG (.edf) files
    psg_fnames = glob.glob(os.path.join(args.data_dir, "*.edf"))

    # Check if files are found
    # print(f"Found {len(psg_fnames)} .edf files")

    if len(psg_fnames) == 0:
        print("Error: No .edf files found in the specified directory.")
        return

    psg_fnames.sort()

    # Process the NCH dataset in chunks and accumulate all data
    final_data_list = process_nch_data(psg_fnames, args.select_ch)

    # Save as Arrow file once at the end
    save_to_arrow(final_data_list, args.output_dir)


if __name__ == "__main__":
    main()
