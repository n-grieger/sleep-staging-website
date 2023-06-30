import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

assert "data_raw" in globals(), "name `channels` not defined. Should be set in ort_webworker.js"
assert "sample_rate" in globals(), "name `sample_rate` not defined. Should be set in postprocessing.py"
assert "resample_rate" in globals(), "name `resample_rate` not defined. Should be set in ort_webworker.js"


def butter_bandpass_filter(data, sample_rate):
    lowcut = 0.3
    highcut = 30.0
    order = 10

    sos_high = butter(order, lowcut, btype='hp', fs=sample_rate, output='sos')
    sos_low = butter(order, highcut, btype='lp', fs=sample_rate, output='sos')

    return sosfiltfilt(sos_low, sosfiltfilt(sos_high, data, padlen=3 * order), padlen=3 * order)


def downsample(data, sample_rate, resampling_frequency):
    gcd = np.gcd(sample_rate, resampling_frequency)

    if sample_rate == resampling_frequency:
        return data

    up = resampling_frequency // gcd
    down = sample_rate // gcd

    return resample_poly(data, up, down)


def zscore(data):
    return (data - data.mean()) / data.std() if data.std() != 0 else 0


def preprocess(data, sample_rate, resample_rate):
    # preprocessing
    data = np.array([
        downsample(butter_bandpass_filter(x, sample_rate), sample_rate, resample_rate) for x in data
    ], dtype=np.float32)

    # onnx model requires an input shape that is a multiple of 16
    if data.shape[1] % 16 != 0:
        data = np.concatenate((data, np.zeros((data.shape[0], 16 - data.shape[1] % 16))), axis=1)

    # get zscore
    data = np.array([zscore(x) for x in data], dtype=np.float32)

    # flatten() will be reversed by new ort.Tensor in ort_webworker.js
    return data.shape, data.flatten()


input_shape, data = preprocess(data_raw.copy(), sample_rate, resample_rate)

del butter_bandpass_filter
del downsample
del zscore
del preprocess
del butter, resample_poly, sosfiltfilt
