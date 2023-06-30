import numpy as np
from scipy.signal import iirfilter, filtfilt
from scipy.signal import resample_poly
from scipy.signal import stft

def preprocess_record(data, sample_rate, preprocessing_sampling_rate):
    n_channels = data.shape[0]
    if (data.shape[1] / sample_rate) % 30 != 0:
        print(f'signal cannot be divided into 30s epochs, it will be shortened by '
              f'{(data.shape[1] / sample_rate) % 30} seconds')
        data = data[:, :int(data.shape[1] / sample_rate / 30) * sample_rate * 30]

    data_new = np.zeros((int(data.shape[1] / sample_rate / 30), n_channels, preprocessing_sampling_rate * 30),
                        dtype='float32')
    for i in range(n_channels):
        # first resample to 100Hz; Why? I don't know. (s. edf_to_h5.py l:215)
        if not int(sample_rate) == 100:
            x = resample_poly(data[i], 100, int(sample_rate))
        else:
            x = data[i]

        # bandpass_freqs = 2 * np.array([0.3, 35]) / sample_rate
        # bandpass_freqs = 2 * np.array([0.2, 30]) / sample_rate  # FIXME make this adaptable
        # sos = _butter(4, bandpass_freqs, btype='band', output='sos').astype('float32')
        # x = _sosfiltfilt(sos, data[i], axis=-1, padtype='constant').astype('float32')
        b, a = iirfilter(
            2, [ff * 2.0 / 100 for ff in [0.2, 30]], btype="bandpass", ftype='butter',
        )
        x = filtfilt(b, a, x, 0)

        # Resample to new sampling rate
        if not 100 == preprocessing_sampling_rate:
            x = resample_poly(x, int(preprocessing_sampling_rate), 100)

        data_new[:, i, :] = x.reshape([-1, preprocessing_sampling_rate * 30])

    data = data_new
    del data_new

    # normalization à la Defossez
    # shape should be (rec_len,2,preprocessing_sampling_rate*30) and we normalize over the first and last dimensions
    # robust_scaler = RobustScaler()
    # clamp_value = 20
    #
    # a, b, c = data.shape
    # data = data.transpose([0, 2, 1]).reshape([a * c, b])
    # data = robust_scaler.fit_transform(data)
    # data[data < -clamp_value] = -clamp_value
    # data[data > clamp_value] = clamp_value
    # data = data.reshape([a, c, b]).transpose([0, 2, 1])
    a, b, c = data.shape
    data = data.transpose([0, 2, 1]).reshape([a * c, b])
    data = normalize_signal_IQR(data)
    data = data.reshape([a, c, b]).transpose([0, 2, 1])

    # pad 900 seconds of signal left and right, 900s correspond to 30 epochs
    data = np.r_['0', np.zeros((30, b, c)), data, np.zeros((30, b, c))]

    return len(data), data


def standardize_online(x, eps=1e-15, axis=(0, -1)):
    # temporal_context, n_channels, f_fft, l_fft = x.shape
    mu = np.mean(x, axis=axis, keepdims=True)
    sigma = np.std(x, axis=axis, keepdims=True, ddof=1)
    return (x - mu) / (sigma + eps)


def preprocess_gen():
    global data, n_blocks, seq_len
    n_channels, n_datapoints = data.shape[1:]
    sr = int(n_datapoints / 30)
    num_side_epochs = int((seq_len - 1) // 2)

    # ignore first and last num_side_epochs epochs, they are padded anyway
    sample_lists = np.array_split(range(num_side_epochs, len(data) - num_side_epochs), n_blocks)

    for block in range(n_blocks):
        def_samples = []

        for i in sample_lists[block]:
            x = data[i - num_side_epochs:i + num_side_epochs + 1]
            # sequence_length, channels, data_points

            x = np.clip(x, -20, 20) / 20

            # convert to stft
            window_length = 2
            window_stride = 1
            n_fft = 2 ** int(np.log2(sr * window_length) + 1)
            n_stride = int(sr * window_stride)
            window = np.hanning
            window = window(n_fft) / window(n_fft).sum()
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

            def_samples.append(x_stft.flatten().astype('float32'))

        yield [len(sample_lists[block]), seq_len, n_channels, f_fft, l_fft], def_samples


# params are set in global context by webworker
n_samples, data = preprocess_record(data_raw.copy(), sample_rate, preprocessing_sampling_rate)
preprocess_gen()