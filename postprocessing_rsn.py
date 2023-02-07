import numpy as np


def postprocess(labels: np.ndarray):
    labels = labels.reshape([-1, 11])[:, 5]
    return labels


# params are set in global context by webworker
labels_out = postprocess(labels_in)
