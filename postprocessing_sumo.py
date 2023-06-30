import numpy as np

assert "onnx_value_map" in globals(), "name `onnx_value_map` not defined. Should be set in ort_webworker.js"
assert "input_shape" in globals(), "name `input_shape` not defined. Should be set in preprocessing_sumo.py"

# get most possible value of onnx_value_map (1 or 0 -> spindle or not)
onnx_value_map = np.array(onnx_value_map.to_py(), dtype='float32').reshape((input_shape[0], 2, -1))
spindle_vect = onnx_value_map.argmax(axis=1)

del onnx_value_map
if "data" in globals(): del data
