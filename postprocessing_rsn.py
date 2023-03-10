import numpy as np

model_stages = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
output_stages = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'N3', 5: 'REM'}
stage_to_output_map = {'Wake': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 5}


def softmax(x, axis=-1):
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)


def postprocess(labels, probabilities, seq_len):
    try:
        # labels = np.array(labels.to_py(), dtype='long')
        probabilities = np.array(probabilities.to_py(), dtype='float32')
        half_seq_len = int((seq_len - 1) // 2)
        # labels_out = []
        # labels_out.append(labels.reshape([-1, seq_len])[:, half_seq_len])

        # aggregate labels and probabilities
        probabilities = softmax(probabilities)
        # for p in probabilities.reshape([-1, 5]): print(p)
        # return np.array(labels.to_py(), dtype='long').reshape([-1, seq_len])[:, half_seq_len]

        probabilities = probabilities.reshape([-1, seq_len, 5])
        predicted_labels_geo = np.zeros(probabilities.shape[0])
        # ignore first and last 30-half_seq_len epochs because of padding
        for i in range(30 - half_seq_len, probabilities.shape[0] - (30 - half_seq_len)):
            pred_to_agg = []
            pred_to_agg += [probabilities[i - j, half_seq_len + j]
                            for j in range(half_seq_len, 0, -1)
                            if i - j >= 0]
            pred_to_agg += [probabilities[i, half_seq_len]]
            pred_to_agg += [probabilities[i + j + 1, half_seq_len - (1 + j)]
                            for j in range(half_seq_len)
                            if i + j + 1 < probabilities.shape[0]]
            # normal average
            # predicted_labels_avg[i] = np.argmax(np.array(pred_to_agg).mean(axis=0))
            # geometric average
            predicted_labels_geo[i] = np.argmax(np.exp(np.log(np.array(pred_to_agg)).mean(axis=0)))
            # print(np.exp(np.log(np.array(pred_to_agg)).mean(axis=0)))
        # remove first and last 30-half_seq_len epochs because of padding
        predicted_labels_geo = predicted_labels_geo[30 - half_seq_len:-(30 - half_seq_len)]
    except Exception as e:
        print(e)

    labels = np.array([stage_to_output_map[model_stages[lab]] for lab in predicted_labels_geo])
    return labels


# params are set in global context by webworker
labels_out = postprocess(labels_in, probabilities_in, seq_len)
