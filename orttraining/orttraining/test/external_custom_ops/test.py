import os
import sys
sys.setdlopenflags(os.RTLD_GLOBAL|os.RTLD_NOW|os.RTLD_DEEPBIND)
import onnxruntime
sys.setdlopenflags(os.RTLD_LOCAL|os.RTLD_NOW|os.RTLD_DEEPBIND)
import orttraining_external_custom_ops
so = onnxruntime.SessionOptions()
sess = onnxruntime.InferenceSession("testdata/model.onnx", so)
input = np.random.rand(2, 2).astype(np.float32)
output  = sess.run(None, {"input1" : input})[0]
np.testing.assert_equal(input, output)