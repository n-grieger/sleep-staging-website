import numpy as np


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
        probabilities = probabilities.reshape([-1, seq_len, 5])
        # predicted_labels_avg = np.zeros(probabilities.shape[0])
        predicted_labels_geo = np.zeros(probabilities.shape[0])
        for i in range(probabilities.shape[0]):
            pred_to_agg = []
            pred_to_agg += [probabilities[i - j, half_seq_len + j]
                            for j in range(half_seq_len, 0)
                            if i - j >= 0]
            pred_to_agg += [probabilities[i, half_seq_len]]
            pred_to_agg += [probabilities[i + j + 1, half_seq_len - (1 + j)]
                            for j in range(half_seq_len)
                            if i + j + 1 < probabilities.shape[0]]
            # normal average
            # predicted_labels_avg[i] = np.argmax(np.array(pred_to_agg).mean(axis=0))
            # geometric average
            predicted_labels_geo[i] = np.argmax(np.exp(np.log(np.array(pred_to_agg)).mean(axis=0)))

        # labels_out.append(predicted_labels_avg)
        # labels_out.append(predicted_labels_geo)
    except Exception as e:
        print(e)
    return predicted_labels_geo


# params are set in global context by webworker
labels_out = postprocess(labels_in, probabilities_in, seq_len)
