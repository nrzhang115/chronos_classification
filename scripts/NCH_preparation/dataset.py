import os
import pandas as pd
from time import time
import random
from gluonts.dataset.arrow import ArrowWriter
import sleep_study as ss

def check_annotations(df):
    event_dict = {k.lower(): 0 for k in ss.info.EVENT_DICT.keys()}

    for x in df.description:
        try:
            event_dict[x.lower()] += 1
        except:
            pass

    return any(event_dict.values())

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


def create_dataset(output_dir='~/sleep_study_dataset', percentage=100):
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    total = len(ss.data.study_list)
    sample_size = int(total * percentage / 100)
    sample_study_list = ss.data.study_list[:sample_size]

    all_data = []  # Collect all data entries here for Chronos

    for i, name in enumerate(sample_study_list):
        if i % 100 == 0:
            print(f'Processing {i} of {sample_size}')
        
        # Process each study and collect Chronos-compatible data
        data_list = ss.data.get_raw_eeg_and_labels_for_chronos(name, ss.data_dir, "EEG C4-M1")

        if data_list:
            print(f"Processed {name}: {len(data_list)} entries")
            all_data.extend(data_list)
        else:
            print(f"Skipping {name} due to no valid data")

    # Save the collected data in Chronos-compatible Arrow format
    save_to_arrow(all_data, output_dir)
    
    



def get_studies_by_patient_age(low, high, txt_path='age_file.csv'):
    study_list = []
    ages = []
    df = pd.read_csv(txt_path, sep=",", dtype={'FILE_NAME': 'str', 'AGE_AT_SLEEP_STUDY_DAYS': 'int'})
    
    df = df[(df.AGE_AT_SLEEP_STUDY_DAYS >= low*365) & (df.AGE_AT_SLEEP_STUDY_DAYS < high*365)]
    print("found", len(df), "patients between", low, "(incl.) and", high, "(excl.) years old.")

    return df.FILE_NAME.tolist(), df.AGE_AT_SLEEP_STUDY_DAYS.tolist()

def annotation_stats(percentage=100):  # Default percentage to 10 for consistency
    output_dir = './'

    broken = []
    total = len(ss.data.study_list)
    sample_size = int(total * percentage / 100)  # Calculate the percentage of the total dataset
    sample_study_list = ss.data.study_list[:sample_size]

    for i, name in enumerate(sample_study_list):

        if i % 100 == 0:
            print('Processing %d of %d' % (i, sample_size))

        path = os.path.join(ss.data_dir, 'Sleep_Data', name + '.tsv')
        df = pd.read_csv(path, sep='\t')

        if not check_annotations(df):
            broken.append(name)

    print('Processed %d files' % i)
    print('%d files have no labeled sleeping stage' % len(broken))

    # Check if the correct percentage of data is processed
    expected_files_count = sample_size
    processed_files_count = len(sample_study_list) - len(broken)

    print(f"Expected number of processed files: {expected_files_count}")
    print(f"Actual number of processed files: {processed_files_count}")

    if processed_files_count == expected_files_count:
        print(f"The processing of {percentage}% of the dataset is verified.")
    else:
        print(f"The processing of {percentage}% of the dataset is not accurate.")

