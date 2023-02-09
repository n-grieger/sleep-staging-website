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


def preprocess(channel_data, start_index, n_samples):
    data = np.array(channel_data.to_py(), dtype='float32')
    n_channels, n_datapoints = data.shape[1:]
    sr = int(n_datapoints / 30)
    def_samples = []

    for i in range(start_index, start_index + n_samples):
        idx_start = i - 5
        idx_end = i + 5 + 1
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

        # convert to stft
        window_length = 2
        window_stride = 1
        n_fft = 2 ** int(np.log2(sr * window_length) + 1)
        n_stride = int(sr * window_stride)
        window = np.hanning
        window = window(n_fft) / window(n_fft).sum()
        # _, _, x_stft = stft(x, preprocessing_sampling_rate, nperseg=n_fft, noverlap=n_fft - n_stride, nfft=n_fft,
        #                     scaling='spectrum', return_onesided=True, padded=False, window=window, boundary=None)
        _, _, x_stft = stft(x, sr, nperseg=n_fft, noverlap=n_fft - n_stride, nfft=n_fft,
                            return_onesided=True, padded=False, window=window, boundary=None)
        x_stft = x_stft.squeeze()
        # sequence_length, channels, f_fft, l_fft
        x_stft = x_stft ** 2
        x_stft = np.abs(x_stft)
        x_stft = np.clip(x_stft, 1e-20, 1e32)
        x_stft = np.log10(x_stft)
        x_stft = standardize_online(x_stft)
        f_fft, l_fft = x_stft.shape[-2:]

        def_samples.append(x_stft.flatten())

    return [11, n_channels, f_fft, l_fft], def_samples


# params are set in global context by webworker
input_shape, samples = preprocess(channel_data, start_index, n_samples)
