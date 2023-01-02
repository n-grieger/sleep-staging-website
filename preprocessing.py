import numpy as np
from sklearn.preprocessing import RobustScaler
from scipy.signal import butter as _butter
from scipy.signal import sosfiltfilt as _sosfiltfilt


def preprocess(channel_data, sampling_rates, preprocessing_sampling_rate):
    data = np.array(channel_data.to_py(), dtype='float32')
    assert all([sampling_rates[0] == s for s in sampling_rates])
    sampling_rate = sampling_rates[0]
    n_channels = data.shape[0]

    data_new = np.zeros((n_channels, preprocessing_sampling_rate * 30 * int(data.shape[1] / sampling_rate / 30)))
    for i in range(n_channels):
        bandpass_freqs = 2 * np.array([0.3, 35]) / sampling_rate
        sos = _butter(4, bandpass_freqs, btype='band', output='sos')
        x = _sosfiltfilt(sos, data[i], axis=-1, padtype='constant')

        # Resample to new sampling rate
        if not sampling_rate == preprocessing_sampling_rate:
            t_old = np.linspace(0, x.size / sampling_rate, x.size)
            num_samples_new = np.round(x.size / sampling_rate * preprocessing_sampling_rate).astype('int')
            t_new = np.linspace(0, num_samples_new / sampling_rate, num_samples_new)
            x = np.interp(t_new, t_old, x)

        data_new[i] = x

    data = data_new.reshape([n_channels, -1, preprocessing_sampling_rate * 30]).transpose(1, 0, 2)

    # normalization à la Defossez
    # shape should be (rec-len,2,preprocessing_sampling_rate*30) and we normalize over the first and last dimensions
    robust_scaler = RobustScaler(unit_variance=True)
    clamp_value = 20

    a, b, c = data.shape
    reshaped_data = data.transpose(0, 2, 1).reshape(a * c, b)
    scaled_data = robust_scaler.fit_transform(reshaped_data)
    scaled_data[scaled_data < -clamp_value] = -clamp_value
    scaled_data[scaled_data > clamp_value] = clamp_value
    scaled_data = scaled_data.reshape(a, c, b).transpose(0, 2, 1)

    def_samples = []
    for i in range(scaled_data.shape[0]):
        idx_start = i - 5
        idx_end = i + 5 + 1
        left_epochs = np.empty((0, n_channels, preprocessing_sampling_rate * 30))
        right_epochs = np.empty((0, n_channels, preprocessing_sampling_rate * 30))
        if idx_start < 0:
            left_epochs = np.zeros((-idx_start, n_channels, preprocessing_sampling_rate * 30))
            idx_start = 0
        if idx_end > scaled_data.shape[0]:
            right_epochs = np.zeros((idx_end - scaled_data.shape[0], n_channels, preprocessing_sampling_rate * 30))
            idx_end = scaled_data.shape[0]
        x = scaled_data[idx_start:idx_end]
        x = np.concatenate([left_epochs, x, right_epochs], axis=0, dtype='float32')
        def_samples.append(x.flatten())

    return def_samples


# params are set in global context by webworker
samples = preprocess(channel_data, sampling_rates, preprocessing_sampling_rate)
