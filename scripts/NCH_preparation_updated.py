import argparse
import os
import numpy as np
from gluonts.dataset.arrow import ArrowWriter
from tqdm.auto import tqdm
from mne.io import read_raw_edf
import sleep_study as ss
import gc
import pandas as pd

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

def load_all_files(data_dir):
    """
    Load all EDF files in the specified directory.
    """
    all_files = [f for f in os.listdir(data_dir) if f.endswith(".edf")]
    print(f"Found {len(all_files)} EDF files in {data_dir}")
    return all_files

def load_annotation_mapping(data_dir):
    """
    Automatically find .tsv annotation files in the dataset directory.
    """
    mapping = {}
    for file in os.listdir(data_dir):
        if file.endswith(".tsv"):
            edf_name = file.replace(".tsv", ".edf")
            mapping[edf_name] = os.path.join(data_dir, file)
    print(f"Automatically mapped {len(mapping)} annotation files.")
    return mapping

# def load_annotation_mapping(annotation_file, data_dir):
#     """
#     Load the mapping of .edf files to their corresponding .tsv files.
#     """
#         # Check if the annotation file exists
#     if not os.path.exists(annotation_file):
#         print(f"Annotation file not found: {annotation_file}")
#         return {}
    
#     mapping = {}
#     print(f"Loading annotations from {annotation_file}")
#     with open(annotation_file, 'r') as f:
#         for line in f:
#             print(f"Raw line from file: {line.strip()}")  # Debug: Print raw line
#             line = line.strip()
#             if line:
#                 # Construct full path to .tsv file in the dataset directory
#                 tsv_path = os.path.join(data_dir, line)
#                 edf_name = os.path.basename(tsv_path).replace('.tsv', '.edf')
#                 mapping[edf_name] = tsv_path
#     print("Annotation Mapping Debug:")
#     for edf, tsv in mapping.items():
#         print(f"{edf} -> {tsv} (Exists: {os.path.exists(tsv)})")
#     return mapping



# def extract_labels(annotation_mapping, file_name, num_epochs):
#     """
#     Extract sleep stage labels from the corresponding TSV file and pad missing epochs.
#     """
#     if file_name not in annotation_mapping:
#         print(f"Annotation file not listed for {file_name}. Skipping labels.")
#         return None

#     annotation_file = annotation_mapping[file_name]
#     if not os.path.exists(annotation_file):
#         print(f"Annotation file not found: {annotation_file}. Skipping labels.")
#         return None

#     # Debug: Print the file being processed
#     print(f"Processing annotation file: {annotation_file}")

#     # Read the TSV file, skipping the first row
#     tsv_data = pd.read_csv(
#         annotation_file,
#         sep="\t",
#         header=0,  # Use the first row as headers
#         names=["start_time", "duration", "annotation"],  # Define column names
#         skiprows=1,  # Skip the header row already in the file
#     )

#     # Ensure 'start_time' and 'duration' are numeric
#     tsv_data["start_time"] = pd.to_numeric(tsv_data["start_time"], errors="coerce")
#     tsv_data["duration"] = pd.to_numeric(tsv_data["duration"], errors="coerce")

#     # Filter rows for sleep stage annotations
#     sleep_stage_data = tsv_data[tsv_data["annotation"].str.contains("Sleep stage", na=False)]

#     # Debug: Show filtered rows
#     print(f"Filtered Sleep Stages from {annotation_file}:\n{sleep_stage_data.head()}")

#     # Initialize all epochs with 'unknown'
#     labels = ['unknown'] * num_epochs
#     epoch_duration = 30.0  # Duration of each epoch in seconds

#     for _, row in sleep_stage_data.iterrows():
#         start_time = row["start_time"]
#         duration = row["duration"]
#         sleep_stage = row["annotation"].replace("Sleep stage ", "").strip()

#         # Skip rows with invalid numeric values
#         if pd.isna(start_time) or pd.isna(duration):
#             continue

#         # Calculate start and end epochs
#         start_epoch = int(start_time // epoch_duration)
#         num_epochs_for_row = int(duration // epoch_duration)

#         # Assign the sleep stage to the corresponding epochs
#         for i in range(num_epochs_for_row):
#             current_epoch = start_epoch + i
#             if current_epoch < num_epochs:  # Ensure we don't exceed the total number of epochs
#                 labels[current_epoch] = sleep_stage

#     # Debug: Print the first few labels
#     print(f"Extracted and padded labels for {file_name}: {labels[:10]}...")  # Show first 10 labels

#     return labels

def extract_labels(annotation_mapping, file_name, num_epochs):
    """
    Extract sleep stage labels from the corresponding TSV file and pad missing epochs.
    """
    if file_name not in annotation_mapping:
        print(f"Annotation file not listed for {file_name}. Skipping labels.")
        return None

    annotation_file = annotation_mapping[file_name]
    if not os.path.exists(annotation_file):
        print(f"Annotation file not found: {annotation_file}. Skipping labels.")
        return None

    print(f"Processing annotation file: {annotation_file}")

    # Read the TSV file
    tsv_data = pd.read_csv(
        annotation_file,
        sep="\t",
        header=0,
        names=["start_time", "duration", "annotation"],
        skiprows=1,
    )

    # Debugging: Check if TSV file structure is valid
    print(f"TSV file head for {file_name}:\n{tsv_data.head()}")

    # Ensure numeric values
    tsv_data["start_time"] = pd.to_numeric(tsv_data["start_time"], errors="coerce")
    tsv_data["duration"] = pd.to_numeric(tsv_data["duration"], errors="coerce")

    # Filter only sleep stages
    sleep_stage_data = tsv_data[tsv_data["annotation"].str.contains("Sleep stage", na=False)]
    print(f"Filtered Sleep Stages from {annotation_file}:\n{sleep_stage_data}")

    # Initialize labels with 'unknown'
    labels = ['unknown'] * num_epochs
    epoch_duration = 30.0  # 30 seconds per epoch

    for _, row in sleep_stage_data.iterrows():
        start_time = row["start_time"]
        duration = row["duration"]
        sleep_stage = row["annotation"].replace("Sleep stage ", "").strip()

        if pd.isna(start_time) or pd.isna(duration):
            continue

        # Calculate start epoch index
        start_epoch = int(np.floor(start_time / epoch_duration))
        num_epochs_for_row = max(1, int(np.round(duration / epoch_duration)))
        # print(f"Checking: start_epoch={start_epoch}, num_epochs={num_epochs} for {file_name}")
        # print(f"DEBUG: File {file_name} - num_epochs = {num_epochs}")

        for i in range(num_epochs_for_row):
            current_epoch = start_epoch + i
            if 0 <= current_epoch < num_epochs:
                labels[current_epoch] = sleep_stage
                # print(f"Assigned {sleep_stage} to epoch {current_epoch} in {file_name}")
            else:
                print(f"Warning: Epoch index {current_epoch} is out of range (0-{num_epochs-1})")
    
    print(f"Final assigned labels for {file_name} (400-420): {labels[400:420]}")
    return labels

def process_nch_data(all_files, data_dir, select_ch, annotation_mapping):
    """
    Process only the selected files from the NCH dataset.
    """
    data_list = []  # Accumulate all processed data

    for fname in tqdm(all_files, desc="Processing all files"):
        psg_path = os.path.join(data_dir, fname)

        if not os.path.exists(psg_path):
            print(f"File {fname} not found in directory {data_dir}. Skipping this file.")
            continue

        # Read EEG data using MNE
        # raw = read_raw_edf(psg_path, preload=True)
        raw = read_raw_edf(psg_path, preload=False) # Save memory

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

        # Extract labels from the corresponding TSV file
        labels = extract_labels(annotation_mapping, fname, len(epochs))
        labels = labels if labels else ['unknown'] * len(epochs)  # Avoid None values

        # # Prepare the time series entry
        # entry = {
        #     "eeg_epochs": epochs,  # Save the segmented EEG epochs
        #     "file_name": fname,  # Keep track of the source file
        #     "labels": labels,  # Add extracted labels
        # }
        # data_list.append(entry)
        
        # Prepare the time series entry
        entry = {
            "eeg_epochs": [epoch.tolist() for epoch in epochs],  # Convert np arrays to lists
            "file_name": fname,  # Keep track of the source file
            "labels": labels,
        }
        print(f"Labels just before saving for {fname} (400-420): {labels[400:420]}")
        data_list.append(entry)

        # # Clear memory for each study processed
        # raw.close()
        # del raw
        # gc.collect()  # Explicitly free memory
        # Clear memory for each study processed
        raw.close()
        del eeg_signal, epochs, labels, raw
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
    path = os.path.join(output_dir, 'nch_sleep_data_all.arrow')

    # Write the data list to an Arrow file
    ArrowWriter(compression="lz4").write_to_file(
        data_list,   # List of data entries (time series)
        path=path    # Output path for the arrow file
    )

    print(f"Data saved to {path}")

# def load_selected_files(file_path):
#     """
#     Load the list of selected files from a text file.
#     """
#     with open(file_path, 'r') as f:
#         return [line.strip() for line in f if line.strip()]

def main():
    print("Starting main function...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/srv/scratch/speechdata/sleep_data/NCH/nch",
                        help="Directory with PSG files.")
    parser.add_argument("--output_dir", type=str, default="/srv/scratch/z5298768/chronos_classification/prepare_time_seires/C4-M1_updated",
                        help="Directory to save the arrow file.")
    parser.add_argument("--select_ch", type=str, default="EEG C4-M1",
                        help="EEG channel to select")
    # parser.add_argument("--selected_files", type=str, default="/home/z5298768/chronos_classification/scripts/Selected_Files",
    #                     help="Path to text file containing list of selected files.")
    # parser.add_argument("--annotation_file", type=str, default="/home/z5298768/chronos_classification/scripts/Selected_annotations",
    #                     help="Path to text file containing list of annotation file paths.")
    args = parser.parse_args()

    # # Load the selected file names
    # selected_files = load_selected_files(args.selected_files)

    # # Load the mapping of annotation files
    # print("Calling load_annotation_mapping...")
    # annotation_mapping = load_annotation_mapping(args.annotation_file, args.data_dir)


    # if len(selected_files) == 0:
    #     print("Error: No selected files found in the specified file.")
    #     return
    
    # Load all EDF files in the directory
    print("Loading all EDF files in the data directory...")
    all_files = load_all_files(args.data_dir)

    # Load annotation mapping automatically
    print("Scanning for annotation files...")
    annotation_mapping = load_annotation_mapping(args.data_dir)

    if len(all_files) == 0:
        print("Error: No EDF files found in the specified directory.")
        return
    

    # Process only the selected files
    # final_data_list = process_nch_data(selected_files, args.data_dir, args.select_ch, annotation_mapping)
    final_data_list = process_nch_data(all_files, args.data_dir, args.select_ch, annotation_mapping)

    # Debugging: Print first entry before saving
    print("First entry sample in final_data_list:")
    print(final_data_list[0])
    
    if not final_data_list:
        print("No data prepared. Exiting.")
        return

    # Save as Arrow file once at the end
    save_to_arrow(final_data_list, args.output_dir)

if __name__ == "__main__":
    main()
