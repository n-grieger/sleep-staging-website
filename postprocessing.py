import numpy as np
from scipy.signal import iirfilter, sosfiltfilt

stage_map = {'Wake': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 5, 'Unknown': -1}
stage_map_inv = {val: key for key, val in stage_map.items()}

DECIMALS = 3

assert "spindle_vect" in globals(), "name `spindle_vect` not defined. Should be set in postprocessing_sumo.py"
assert "sample_rate" in globals(), "name `resample_rate` not defined. Should be set in preprocessing.py"
assert "resample_rate" in globals(), "name `resample_rate` not defined. Should be set in ort_webworker.js"
assert "channels" in globals(), "name `channels` not defined. Should be set in ort_webworker.js"
assert "data_raw" in globals(), "name `channels` not defined. Should be set in preprocessing.py"

assert "rsn" in globals(), "name `rsn` not defined. Should be set in ort_webworker.js"
assert "remove_spindles" in globals(), "name `remove_spindles` not defined. Should be set in ort_webworker.js"

# if sleepstages are unknown
if not rsn:
    # foreach epoch set value Unknown
    sleepstages = np.array([stage_map["Unknown"]] * (spindle_vect.shape[1] // resample_rate // 30))
else:
    assert "sleepstages" in globals(), "name `sleepstages` not defined. Should be set in postprocessing_rsn.py"


def spindle_vect_to_indices(x):
    # be able to detect spindles at the start and end of vector
    diff = np.diff(np.r_[0, x, 0])
    return np.c_[np.argwhere(diff == 1), np.argwhere(diff == -1)]


def overlap(spindles, sleepstages):
    sleepstages = np.repeat(sleepstages, 2)          # transform sleepstages to half epoches
    next_epoch = np.hstack([sleepstages[1:], [-1]])  # overlap last half of current epoch with the next epoch

    sleepstages = np.logical_or(                     # there can only be a spindle if
        sleepstages == stage_map["N2"],              # a) current epoch is "N2"
        sleepstages == stage_map["N3"],              # b) current epoch is "N3"
        np.logical_and(                              # c) current epoch is "N1" and next epoch is "N2" or "N3"
            sleepstages == stage_map["N1"],
            np.logical_or(next_epoch == stage_map["N2"], next_epoch == stage_map["N3"])
        )
    )

    for i, channel in enumerate(spindles):
        idx = [sleepstages[center // 15] for center in (channel.sum(axis=1) // 2).astype(int)]
        spindles[i] = channel[idx]

    return spindles


def analyze(channel, sleepstages, sample_rate, channel_raw):
    # get center of spindle
    centers = (channel.sum(axis=1) // 60).astype(int)

    # get key of stage_map foreach center of spindle
    labels = [stage_map_inv[stage] for stage in sleepstages[centers]]

    # get duration of spindle [s]: end - start
    duration = (channel * [-1, 1]).sum(axis=1)

    frequencies = []
    amplitudes = []
    # get properties from raw input
    for spindle in channel:
        idx = np.rint(spindle * sample_rate).astype(int)
        sample = channel_raw[idx[0]:idx[1]]

        # TODO
        if sample.shape[0] < 16:
            sample = np.resize(sample, 16)

        # filter signal between 10 and 16 Hz
        bandpass_frequencies = [10, 16]
        sos = iirfilter(2, [bandpass_frequency * 2.0 / sample_rate for bandpass_frequency in bandpass_frequencies], btype="bandpass", ftype="butter", output="sos")
        sample_filtered = sosfiltfilt(sos, sample, 0)

        # calculate every zero crossings of sample_filtered (1/2 pulse)
        zero_crossings = np.where(np.diff(np.sign(sample_filtered)))[0]

        # calculate instantaneous frequencies between zero crossings # TODO
        instantaneous_frequency = sample_rate / np.diff(zero_crossings) / 2

        # get spindle frequency
        frequencies.append(instantaneous_frequency.mean() if instantaneous_frequency.shape[0] > 0 else 0)

        # get spindle amplitude
        amplitudes.append(sample.max() - sample.min())

    return list(zip(labels, *channel.T, np.round(duration, DECIMALS), np.round(frequencies, DECIMALS), np.round(amplitudes, DECIMALS)))


def analyze_characteristics(analysis, sleepstages):
    labels = np.array([spindle[0] for spindle in analysis])
    analysis = np.array([spindle[1:] for spindle in analysis])
    # get count of analysis
    count = analysis.shape[0]

    # get spindles per minute [spm]: "spindles in N2" / ("epoches in N2" * 30 / 60) # one epoch equals 30 seconds (*30), one minute equals 60 seconds (/60)
    spm = (labels == "N2").sum() / ((sleepstages == stage_map["N2"]).sum() / 2) if count > 0 and (sleepstages == stage_map["N2"]).sum() > 0 else 0

    sleepstages_duration = [(sleepstages == val).sum() / 2 for val in stage_map.values()]

    if count > 0:
        statistics = np.vstack([
            analysis[:, 2:].mean(axis=0),
            analysis[:, 2:].std(axis=0),
            analysis[:, 2:].min(axis=0),
            np.quantile(analysis[:, 2:], .25, axis=0),
            np.median(analysis[:, 2:], axis=0),
            np.quantile(analysis[:, 2:], .75, axis=0),
            analysis[:, 2:].max(axis=0)
        ]).T.round(DECIMALS)
    else:
        statistics = np.empty((3, 5))
        statistics.fill(np.nan)

    return count, np.round(spm, DECIMALS), *sleepstages_duration, *statistics.flatten()


def export(spindles, sleepstages, sample_rate, resample_rate, channels, data_raw):
    analyses = [analyze(channel, sleepstages, sample_rate, channel_raw) for channel, channel_raw in zip(spindles, data_raw)]

    # characteristics of each spindle
    headers = "phase start end duration frequency amplitude"
    results = ["\n".join([headers, *[
        " ".join(map(str, spindle))
        for spindle in analysis]])
        for analysis in analyses]

    # characteristics of each channel
    headers = "channel count spm"
    for key in stage_map.keys():
        headers += f" {key}_duration"
    for column in ["duration", "frequency", "amplitude"]:
        for stat in ["mean", "std", "min", ".25", "median", ".75", "max"]:
            headers += f" {column}_{stat}"
    characteristics = {"characteristics.txt": "\n".join([headers, *[
        " ".join(map(str, [name.replace(" ", "_"), *analyze_characteristics(analysis, sleepstages)]))
        for name, analysis in zip(channels, analyses)]])
    }

    return *results, characteristics


# remove spindles whose sleep phase is not known
spindle_vect = spindle_vect[:, :sleepstages.shape[0] * 30 * resample_rate]
# get start and end of each spindle
spindles = [spindle_vect_to_indices(channel_vect) / resample_rate for channel_vect in spindle_vect]
# remove impossible spindles (e. g. "Wake")
if rsn and remove_spindles:
    spindles = overlap(spindles, sleepstages)

# export spindle characteristics
results = export(spindles, sleepstages, sample_rate, resample_rate, channels, data_raw)

del stage_map
del stage_map_inv
del DECIMALS
del iirfilter
del sosfiltfilt
if "spindle_vect" in globals(): del spindle_vect
if "resample_rate" in globals(): del resample_rate
del spindle_vect_to_indices
del overlap
del export
del analyze
del analyze_characteristics

if "rsn" in globals(): del rsn
if "remove_spindles" in globals(): del remove_spindles
