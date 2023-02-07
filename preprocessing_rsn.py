import numpy as np
from scipy.signal import butter as _butter
from scipy.signal import sosfiltfilt as _sosfiltfilt
from scipy.signal import stft, resample_poly
from sklearn.preprocessing import RobustScaler


def standardize_online(x, eps=1e-15, axis=(0, -1)):
    # temporal_context, n_channels, f_fft, l_fft = x.shape
    mu = np.mean(x, axis=axis, keepdims=True)
    sigma = np.std(x, axis=axis, keepdims=True, ddof=1)
    return (x - mu) / (sigma + eps)


def preprocess(channel_data, sampling_rates, preprocessing_sampling_rate):
    data = np.array(channel_data.to_py(), dtype='float32')
    assert all([sampling_rates[0] == s for s in sampling_rates])
    sampling_rate = sampling_rates[0]
    n_channels = data.shape[0]

    data_new = np.zeros((n_channels, preprocessing_sampling_rate * 30 * int(data.shape[1] / sampling_rate / 30)))
    for i in range(n_channels):
        # bandpass_freqs = 2 * np.array([0.3, 35]) / sampling_rate
        bandpass_freqs = 2 * np.array([0.2, 30]) / sampling_rate
        sos = _butter(4, bandpass_freqs, btype='band', output='sos')
        x = _sosfiltfilt(sos, data[i], axis=-1, padtype='constant')

        # Resample to new sampling rate
        if not sampling_rate == preprocessing_sampling_rate:
            x = resample_poly(x, int(preprocessing_sampling_rate), int(sampling_rate))

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
        # sequence_length, channels, data_points

        # convert to stft
        window_length = 2
        window_stride = 1
        n_fft = 2 ** int(np.log2(preprocessing_sampling_rate * window_length) + 1)
        n_stride = int(preprocessing_sampling_rate * window_stride)
        window = np.hanning
        window = window(n_fft) / window(n_fft).sum()
        # _, _, x_stft = stft(x, preprocessing_sampling_rate, nperseg=n_fft, noverlap=n_fft - n_stride, nfft=n_fft,
        #                     scaling='spectrum', return_onesided=True, padded=False, window=window, boundary=None)
        _, _, x_stft = stft(x, preprocessing_sampling_rate, nperseg=n_fft, noverlap=n_fft - n_stride, nfft=n_fft,
                            return_onesided=True, padded=False, window=window, boundary=None)
        x_stft = x_stft.squeeze()
        # sequence_length, channels, f_fft, l_fft
        x_stft = (x_stft ** 2)
        x_stft = np.abs(x_stft)
        x_stft = np.clip(x_stft, 1e-20, 1e32)
        x_stft = np.log10(x_stft)
        x_stft = standardize_online(x_stft)
        f_fft, l_fft = x_stft.shape[-2:]

        def_samples.append(x_stft.flatten())

    return [11, n_channels, f_fft, l_fft], def_samples


# params are set in global context by webworker
input_shape, samples = preprocess(channel_data, sampling_rates, preprocessing_sampling_rate)
