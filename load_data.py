"""
Load the data from mne files into numpy array to later be used for classification.
The data is already pre-processed when loaded. The steps that are followed are:
    - Remove exterior channels
    - Apply [-0.2 0] baseline
    - Resample to 64 Hz
    - Get first 5 s
    - Reorder channels into 2D matrix (+ time)
This results in a 4D matrix (nSamples, d1, d2, time) that is saved alongside the labels of each sample.
"""
import mne
from copy import copy
import numpy as np
import dill

mne.set_log_level('WARNING')

# #############################################################################


def load_single(fname: str, classes: list[str]) -> mne.Epochs:
    """
    Loads a single subject, removes the exterior channels, applies a baseline, and resample the data to 64 Hz
    :param fname: File name to load (mne format)
    :param classes: List of valid classes
    :return: Valid epochs of the given file.
    """
    epochs = mne.read_epochs(fname, verbose=False)
    epochs.drop_channels(ch_names=['Iz','F7', 'F8', 'FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8', 'P7', 'P8', 'P9', 'P10', 'Status'])
    epochs.apply_baseline(baseline=(-0.2, 0))
    epochs.resample(64)
    # If you only want two classes, you can get them like this: (otherwise you have the three classes in 'epochs')
    if len(classes) < 3:
        epochs = mne.concatenate_epochs([epochs[c] for c in classes])
    return epochs


def load_all(subjects: list[str], classes: list[str]) -> tuple[mne.Epochs, np.ndarray]:
    """
    Loads the subject file in subjects, and extract the valid classes. Then it crops the data to use only 0-5 s
    :param subjects: List of subject file IDs
    :param classes: List of classes to be extracted
    :return: Epochs of every subject and list of subject IDs
    """
    epochs_all = []
    subj = []
    for i, subject in enumerate(subjects):
        print('*' * 20)
        print(f'Subject {subject}({i+1}/{len(subjects)})')
        single = load_single(f'data/epochs_subj{subject}-epo.fif', classes)
        epochs_all.append(single)
        subj.append(np.zeros((len(single), 1)) + i)
    subj = np.concatenate(subj)
    epochs = mne.concatenate_epochs(epochs_all)
    epochs.crop(tmin=0, tmax=5, include_tmax=True)
    return epochs, subj


def reshape_data(epochs: mne.epochs, subj: np.ndarray) -> dict[str: np.ndarray, str:np.ndarray]:
    """
    Transforms the data from mne epochs (nTrials, nChannels, time) to a numpy array with shape (nTrials, d1, d2, time)
    where d1=9, and d2=7. These correspond to place each electrode into a 2D grid.
    :param epochs: MNE epochs for all subjects and classes
    :param subj: Subjects ID for each epoch (to be saved)
    :return: A dictionary with the reshaped data, the label for each trial, and the subject for each trial.
    """
    # To extract data from a specific class:
    x = epochs.get_data(copy=True)
    y = np.expand_dims(epochs.events[:, -1] // 2, -1)
    channel = copy(epochs.ch_names)
    rename = {'AF7': 'AF5', 'AF8': 'AF6', 'PO7': 'PO5', 'PO8': 'PO6'}
    for k,v in rename.items():
        channel[channel.index(k)] = v

    row = {'Fp': 0, 'AF': 1, 'F': 2, 'FC': 3, 'C': 4, 'CP': 5, 'P': 6, 'PO': 7, 'O': 8}
    col = {'5': 0, '3': 1, '1': 2, 'z': 3, '2': 4, '4': 5, '6': 6}
    tr, ch, tlen = x.shape
    xImage = np.zeros((tr, len(row), len(col), tlen))
    for i, ch in enumerate(channel):
        if len(ch) == 2:
            r = row[ch[:1]]
        else:
            r = row[ch[:2]]
        c = col[ch[-1]]
        xImage[:, r, c, :] = x[:, i, :]

    data = {'x': xImage, 'y': y, 'subj': subj}
    with open('data/full_classReady.dill', 'wb') as f:
        dill.dump(data, f)
    return data


if __name__ == '__main__':
    subjects = [6, 7, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19,  # 2019 cohort
                2401, 2402, 2403, 2404, 2406, 2407, 2408, 2409, 2415, 2416, 2417, 2418, 2420, 2422, 2423]  # 2024 cohort
    classes = ['not_recognised', 'remembered']
    epochs, subj = load_all(subjects, classes)
    data = reshape_data(epochs, subj)
