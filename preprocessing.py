import numpy as np

assert "data_raw" in globals(), "name `channels` not defined. Should be set in ort_webworker.js"
assert "sample_rates" in globals(), "name `sample_rates` not defined. Should be set in ort_webworker.js"

# convert input array to numpy
data_raw = np.array(data_raw.to_py(), dtype=np.float32)

# check if the sampling rate of all channels is equal
sample_rates = sample_rates.to_py()
assert all([sample_rates[0] == s for s in sample_rates])
sample_rate = int(sample_rates[0])

del sample_rates
