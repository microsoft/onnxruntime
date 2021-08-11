import os
import sys
sys.setdlopenflags(os.RTLD_GLOBAL|os.RTLD_NOW)
import onnxruntime
import custom_ops
so = onnxruntime.SessionOptions()
sess = onnxruntime.InferenceSession("testdata/model.onnx", so)
