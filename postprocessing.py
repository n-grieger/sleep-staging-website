import numpy as np

model_stages = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
output_stages = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'N3', 5: 'REM'}
stage_to_output_map = {'Wake': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 5}


def postprocess(labels, probabilities_in, seq_len):
    labels = np.array(labels.to_py(), dtype='long')
    labels = np.array([stage_to_output_map[model_stages[lab]] for lab in labels])
    return labels


# params are set in global context by webworker
labels_out = postprocess(labels_in, probabilities_in, seq_len)
