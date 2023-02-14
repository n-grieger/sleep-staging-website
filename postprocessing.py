import numpy as np


def postprocess(labels, probabilities_in, seq_len):
    labels = np.array(labels.to_py(), dtype='long')
    return labels


# params are set in global context by webworker
labels_out = postprocess(labels_in, probabilities_in, seq_len)
