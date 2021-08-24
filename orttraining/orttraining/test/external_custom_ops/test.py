import onnxruntime

import faulthandler;
faulthandler.enable()

import orttraining_external_custom_ops

orttraining_external_custom_ops.register_custom_ops()
so = onnxruntime.SessionOptions()
sess = onnxruntime.InferenceSession("testdata/model.onnx", so)
