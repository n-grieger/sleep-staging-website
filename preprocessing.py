import numpy as np
from scipy.signal import butter as _butter
from scipy.signal import sosfiltfilt as _sosfiltfilt, resample_poly
from sklearn.preprocessing import RobustScaler


def preprocess_record(data, sampling_rates, preprocessing_sampling_rate):
    # global data, sampling_rates, preprocessing_sampling_rate
    data = np.array(data.to_py(), dtype='float32')
    assert all([sampling_rates[0] == s for s in sampling_rates])
    sampling_rate = sampling_rates[0]
    n_channels = data.shape[0]
    if (data.shape[1] / sampling_rate) % 30 != 0:
        print(f'signal cannot be divided into 30s epochs, it will be shortened by '
              f'{(data.shape[1] / sampling_rate) % 30} seconds')
        data = data[:, :int(data.shape[1] / sampling_rate / 30) * sampling_rate * 30]

    data_new = np.zeros((int(data.shape[1] / sampling_rate / 30), n_channels, preprocessing_sampling_rate * 30),
                        dtype='float32')
    for i in range(n_channels):
        # bandpass_freqs = 2 * np.array([0.3, 35]) / sampling_rate
        bandpass_freqs = 2 * np.array([0.2, 30]) / sampling_rate  # FIXME make this adaptable
        sos = _butter(4, bandpass_freqs, btype='band', output='sos').astype('float32')
        x = _sosfiltfilt(sos, data[i], axis=-1, padtype='constant').astype('float32')

        # Resample to new sampling rate
        if not sampling_rate == preprocessing_sampling_rate:
            x = resample_poly(x, int(preprocessing_sampling_rate), int(sampling_rate))

        data_new[:, i, :] = x.reshape([-1, preprocessing_sampling_rate * 30])

    data = data_new
    del data_new

    # normalization à la Defossez
    # shape should be (rec_len,2,preprocessing_sampling_rate*30) and we normalize over the first and last dimensions
    robust_scaler = RobustScaler(unit_variance=True)
    clamp_value = 20

    a, b, c = data.shape
    data = data.transpose([0, 2, 1]).reshape([a * c, b])
    data = robust_scaler.fit_transform(data)
    data[data < -clamp_value] = -clamp_value
    data[data > clamp_value] = clamp_value
    data = data.reshape([a, c, b]).transpose([0, 2, 1])
    return len(data), data


def standardize_online(x, eps=1e-15, axis=(0, -1)):
    # temporal_context, n_channels, f_fft, l_fft = x.shape
    mu = np.mean(x, axis=axis, keepdims=True)
    sigma = np.std(x, axis=axis, keepdims=True, ddof=1)
    return (x - mu) / (sigma + eps)


def preprocess_gen():
    global data, n_blocks, seq_len
    n_channels, n_datapoints = data.shape[1:]
    num_side_epochs = int((seq_len - 1) // 2)

    sample_lists = np.array_split(range(len(data)), n_blocks)

    for block in range(n_blocks):
        def_samples = []

        for i in sample_lists[block]:
            idx_start = i - num_side_epochs
            idx_end = i + num_side_epochs + 1
            left_epochs = np.empty((0, n_channels, n_datapoints))
            right_epochs = np.empty((0, n_channels, n_datapoints))
            if idx_start < 0:
                left_epochs = np.zeros((-idx_start, n_channels, n_datapoints))
                idx_start = 0
            if idx_end > data.shape[0]:
                right_epochs = np.zeros((idx_end - data.shape[0], n_channels, n_datapoints))
                idx_end = data.shape[0]
            x = data[idx_start:idx_end]
            x = np.concatenate([left_epochs, x, right_epochs], axis=0, dtype='float32')
            # sequence_length, channels, data_points

            def_samples.append(x.flatten().astype('float32'))

        yield [len(sample_lists[block]), seq_len, n_channels, n_datapoints], def_samples


# params are set in global context by webworker
n_samples, data = preprocess_record(data, sampling_rates, preprocessing_sampling_rate)
preprocess_gen()
