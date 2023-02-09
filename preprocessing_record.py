import numpy as np
from scipy.signal import butter as _butter
from scipy.signal import sosfiltfilt as _sosfiltfilt, resample_poly
from sklearn.preprocessing import RobustScaler


def preprocess(channel_data, sampling_rates, preprocessing_sampling_rate):
    data = np.array(channel_data.to_py(), dtype='float32')
    assert all([sampling_rates[0] == s for s in sampling_rates])
    sampling_rate = sampling_rates[0]
    n_channels = data.shape[0]

    data_new = np.zeros((n_channels, preprocessing_sampling_rate * 30 * int(data.shape[1] / sampling_rate / 30)))
    for i in range(n_channels):
        # bandpass_freqs = 2 * np.array([0.3, 35]) / sampling_rate
        bandpass_freqs = 2 * np.array([0.2, 30]) / sampling_rate  # FIXME make this adaptable
        sos = _butter(4, bandpass_freqs, btype='band', output='sos')
        x = _sosfiltfilt(sos, data[i], axis=-1, padtype='constant')

        # Resample to new sampling rate
        if not sampling_rate == preprocessing_sampling_rate:
            x = resample_poly(x, int(preprocessing_sampling_rate), int(sampling_rate))

        data_new[i] = x

    data: np.ndarray = data_new.reshape([n_channels, -1, preprocessing_sampling_rate * 30]).transpose([1, 0, 2]).astype(
        'float32')
    del data_new

    # normalization à la Defossez
    # shape should be (rec-len,2,preprocessing_sampling_rate*30) and we normalize over the first and last dimensions
    robust_scaler = RobustScaler(unit_variance=True)
    clamp_value = 20

    a, b, c = data.shape
    data = data.transpose([0, 2, 1]).reshape([a * c, b])
    data = robust_scaler.fit_transform(data)
    data[data < -clamp_value] = -clamp_value
    data[data > clamp_value] = clamp_value
    data = data.reshape([a, c, b]).transpose([0, 2, 1])
    return data.tolist()


# params are set in global context by webworker
samples = preprocess(channel_data, sampling_rates, preprocessing_sampling_rate)
