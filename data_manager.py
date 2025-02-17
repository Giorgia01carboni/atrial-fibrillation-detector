import wfdb
import numpy as np


def load_ecg(record_name):

    """
        Using data from MIT-BIH AFDB
        Returns: ecg signal, RR intervals, sampling frequency, true labels from annotations
    """

    # read file
    path_to_folder = "../afdb/"
    record = wfdb.rdrecord(f"{path_to_folder}{record_name}")  #ECG record
    annotation = wfdb.rdann(f"{path_to_folder}{record_name}", "atr")  #ECG annotations, used to find RR intervals

    # Signal and sampling frequency extraction
    signal = record.p_signal[:, 0]
    fs = record.fs

    # RR intervals
    rr_intervals = np.diff(annotation.sample) / fs  # In seconds

    # Ground truth from actual annotations (used to check my results against this)
    ground_truth = np.array([1 if "AFIB" in label else 0 for label in annotation.aux_note])

    return signal, fs, rr_intervals, ground_truth

