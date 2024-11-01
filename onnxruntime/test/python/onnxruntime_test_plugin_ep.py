import numpy

import onnxruntime as ort

ort.register_plugin_execution_provider_library("outTreeEp", "/home/leca/code/onnxruntime/samples/outTreeEp/build/liboutTreeEp.so")
#ort.register_plugin_execution_provider_library("tensorrtEp", "/home/leca/code/onnxruntime/samples/tensorRTEp/build/libTensorRTEp.so")

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
#session = ort.InferenceSession("/home/leca/code/onnxruntime/samples/c_test/Relu.onnx", sess_options, providers=[("CPUExecutionProvider")])
#session = ort.InferenceSession("/home/leca/code/onnxruntime/samples/c_test/Relu.onnx", sess_options, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
session = ort.InferenceSession("/home/leca/code/onnxruntime/samples/c_test/Relu.onnx", sess_options, providers=["outTreeEp", "CPUExecutionProvider"], provider_options=[{"int_property":"3", "str_property":"strvalue"}, {}])

# runtime error for tensorrtEp for using two different map in libonnx.a, as onnxruntime_pybind_state.so is linked to static ORT libs
#session = ort.InferenceSession("/home/leca/code/onnxruntime/samples/c_test/Relu.onnx", sess_options, providers=["tensorrtEp", "CPUExecutionProvider"], provider_options=[{"device_id":"0", "str_property":"strvalue"}, {}])

y = session.run(None, {'x': numpy.array([-3.0, 5.0, -2.0, 4.0]).astype(numpy.float32)})
print(y)

ort.unregister_plugin_execution_provider("outTreeEp")
