import numpy as np


def postprocess(labels):
    labels = np.array(labels.to_py(), dtype='long')
    labels = labels.reshape([-1, 11])[:, 5]
    return labels


# params are set in global context by webworker
labels_out = postprocess(labels_in)
