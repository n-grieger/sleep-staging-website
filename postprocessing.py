import numpy as np


def postprocess(labels: np.ndarray):
    return labels


# params are set in global context by webworker
labels_out = postprocess(labels_in)
