import mne
import os
import pandas as pd
import numpy as np
from dateutil import parser
from datetime import timezone
from datetime import datetime
import pywt
from scipy import signal, interpolate
from scipy.signal import resample

import sleep_study as ss


EPOCH_SEC_SIZE = 30  # Define the epoch size in seconds
TARGET_SAMPLING_RATE = ss.info.REFERENCE_FREQ   # The target sampling rate is defined in info.py
study_list = None # Will be initialized after the module is initialized.

def clean_ch_names(ch_names):
    return [x.upper() for x in ch_names]

# Initializes a list of study names by reading .edf files in the Sleep_Data directory.
def init_study_list(percentage=100):
    path = ss.data_dir
    all_studies = [x[:-4] for x in os.listdir(path) if x.endswith('edf')]
    sample_size = int(len(all_studies) * percentage / 100)
    return all_studies[:sample_size]

def init_age_file():
    new_fn = 'age_file.csv'
    age_path = os.path.join('/srv','scratch','z5298768','AttnSleep_data','Health_Data', ss.info.SLEEP_STUDY)

    df = pd.read_csv(age_path, sep=',', dtype='str')
    df['FILE_NAME'] = df["STUDY_PAT_ID"].str.cat(df["SLEEP_STUDY_ID"], sep='_')

    df.to_csv(new_fn, columns=["FILE_NAME", "AGE_AT_SLEEP_STUDY_DAYS"], index=False)
    return os.path.abspath(new_fn)

# Loads EEG data from an .edf file, sets the measurement date, loads annotations from a .tsv file, 
# and renames channels to uppercase.
def load_study(name, preload=False, exclude=[], verbose='CRITICAL'):
    path = os.path.join(ss.data_dir, name + '.edf')
    path = os.path.abspath(path)
    # file_size = os.stat(path).st_size / 1024 / 1024

    raw = mne.io.read_raw_edf(input_fname=path, exclude=exclude, preload=preload,
                                verbose=verbose)

    patient_id, study_id = name.split('_')

    tmp = ss.info.SLEEP_STUDY
    date = tmp[(tmp.STUDY_PAT_ID == int(patient_id))
             & (tmp.SLEEP_STUDY_ID == int(study_id))] \
                     .SLEEP_STUDY_START_DATETIME.values[0] \
                     .split()[0]

    time = str(raw.info['meas_date']).split()[1][:-6]

    new_datetime = parser.parse(date + ' ' + time + ' UTC') \
                         .replace(tzinfo=timezone.utc)

    raw.set_meas_date(new_datetime)
    # raw._raw_extras[0]['meas_date'] = new_datetime

    annotation_path = os.path.join(ss.data_dir, name + '.tsv')
    df = pd.read_csv(annotation_path, sep='\t')
    annotations = mne.Annotations(df.onset, df.duration, df.description,
                                  orig_time=new_datetime)

    raw.set_annotations(annotations)

    raw.rename_channels({name: name.upper() for name in raw.info['ch_names']})

    return raw

# Check if the file contains specified events
def contains_specified_events(annotations, event_dict):
    return any(event in event_dict for event in annotations.description)

# Extract Raw EEG Signals with Downsampling
def get_raw_eeg_and_labels(name, data_dir, select_ch, target_sampling_rate=TARGET_SAMPLING_RATE):
    raw = load_study(name)
    current_sampling_rate = int(raw.info['sfreq'])
    
    raw_ch_df = raw.to_data_frame(time_format=None)[select_ch]
    raw_ch_df = raw_ch_df.to_frame()

    annotation_path = os.path.join(data_dir, name + '.tsv')
    df = pd.read_csv(annotation_path, sep='\t', usecols=['onset', 'duration', 'description'])

    annotations = mne.Annotations(onset=df['onset'].values, duration=df['duration'].values, description=df['description'].values, orig_time=raw.info['meas_date'])
    raw.set_annotations(annotations)

    events, _ = mne.events_from_annotations(raw, event_id=ss.info.EVENT_DICT)

    data_list = []
    labels = []
    
    for event in events:
        onset, label = event[[0, 2]]  # onset and event label
        start = np.datetime64(datetime.fromtimestamp(onset / current_sampling_rate))  # Convert onset to timestamp
        data_list.append({
            "start": start,
            "target": label  # Ensure this aligns with Chronos' target field
        })

    return data_list
###############################################################################
def channel_stats(verbose=True):
    study_ch_names = {}

    for i, study in enumerate(study_list):

        if (i % 10 == 0) and verbose:
            print('Processing %d of %d' % (i, len(study_list)))

        raw = load_study(study)
        study_ch_names[study] = raw.ch_names

    names = {}
    for y in study_ch_names.values():
        for x in y:
            x = x.upper()
            try:
                names[x] += 1
            except:
                names[x] = 1

    names = {k: v for k, v in sorted(names.items(), key=lambda item: item[1], reverse=True)}

    print('\n'.join('%-20s %4d &  %.2f%%\\' % (k, v, v / len(study_list) * 100) for k, v in names.items()))
    return names


def sleep_stage_stats(studies=[]):
    res = {k.lower(): 0 for k in ss.info.EVENT_DICT}
    if len(studies)<1:
        studies = study_list
        
    for i, study in enumerate(studies):

        if (i + 1) % 100 == 0:
            print('Processed %d of %d' % (i + 1, len(studies)))

        annotation_path = os.path.join(ss.data_dir, study+ '.tsv')
        df = pd.read_csv(annotation_path, sep='\t')

        for event in df.description.tolist():
            try:
                res[event.lower()] += 1
            except:
                pass

    total = sum(res.values())

    print('Stage  Observations  Percentage')
    print('\n'.join(['%-2s     %7d        %6.3f%%' % (k.split()[-1], v, v / total * 100) for (k, v) in res.items()]))

    return res
