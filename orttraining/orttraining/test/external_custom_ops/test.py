import onnxruntime
import orttraining_external_custom_ops
so = onnxruntime.SessionOptions()
sess = onnxruntime.InferenceSession("testdata/model.onnx", so)
