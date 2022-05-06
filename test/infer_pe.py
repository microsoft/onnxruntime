import onnxruntime as ort
import os
import numpy as np

input_ = np.random.rand(10, 36, 36, 528)
onnx_path = "bug.onnx"

import pdb
#pdb.set_trace()
sess_opt = ort.SessionOptions()
#sess_opt.inter_op_num_threads = 2
sess_opt.execution_mode = ort.ExecutionMode.ORT_PARALLEL
sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
model = ort.InferenceSession(onnx_path, sess_opt, providers=['CUDAExecutionProvider'])
input_name = model.get_inputs()[0].name
output_name = model.get_outputs()[0].name
input_ = input_.astype('float32')
output = model.run(None, {input_name: input_})
print("When running with parallel executor: ", output[0][0])


sess_opt2 = ort.SessionOptions()
#sess_opt.inter_op_num_threads = 2
sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
model2 = ort.InferenceSession(onnx_path, sess_opt2, providers=['CPUExecutionProvider'])
input_name = model2.get_inputs()[0].name
output_name = model2.get_outputs()[0].name
output2 = model2.run(None, {input_name: input_})
print("When running with sequential executor: ", output2[0][0])
assert np.allclose(output[0], output2[0], atol=1e-5)
print("Test Passed!")