import onnxruntime as ort
import os
import numpy as np

input_ = np.random.rand(10, 36, 36, 528)
onnx_path = "bug.onnx"

#import pdb
#pdb.set_trace()
_ = input(os.getpid())
sess_opt = ort.SessionOptions()
sess_opt.inter_op_num_threads = 1
sess_opt.execution_mode = ort.ExecutionMode.ORT_PARALLEL
sess_opt.optimized_model_filepath = "bug.opt.onnx"
sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
# e.g.for "op1,op2,op3;op4,op5", [op1,op2,op3],[op4,op5] will occupy separate streams exclusively#
# grouped_ops has priority over streams_per_ep, which will only be applied to ops not refered in grouped_ops
sess_opt.add_session_config_entry('session.node_partition_config_file', 'dummpy_config.cfg')
sess_opt.log_severity_level = 0
#sess_opt.log_verbosity_level = 255
model = ort.InferenceSession(onnx_path, sess_opt, providers=['CUDAExecutionProvider'])
input_name = model.get_inputs()[0].name
output_name = model.get_outputs()[0].name
input_ = input_.astype('float32')
output = model.run(None, {input_name: input_})
# print("When running with parallel executor: ", output[0][0])

sess_opt2 = ort.SessionOptions()
#sess_opt.inter_op_num_threads = 2
sess_opt2.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
model2 = ort.InferenceSession(onnx_path, sess_opt2, providers=['CPUExecutionProvider'])
input_name = model2.get_inputs()[0].name
output_name = model2.get_outputs()[0].name
output2 = model2.run(None, {input_name: input_})
# print("When running with sequential executor: ", output2[0][0])
assert np.allclose(output[0], output2[0], atol=1e-5)

print("Test Passed!")
