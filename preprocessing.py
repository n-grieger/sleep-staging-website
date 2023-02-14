import numpy as np


def preprocess(channel_data, start_index, n_samples, seq_len):
    data = np.array(channel_data.to_py(), dtype='float32')
    n_channels, n_datapoints = data.shape[1:]
    def_samples = []
    num_side_epochs = int((seq_len - 1) // 2)

    for i in range(start_index, start_index + n_samples):
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
        def_samples.append(x.flatten())

    return [seq_len, n_channels, n_datapoints], def_samples


# params are set in global context by webworker
input_shape, samples = preprocess(channel_data, start_index, n_samples, seq_len)
