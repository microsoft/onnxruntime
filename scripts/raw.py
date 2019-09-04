import onnx
from onnx import numpy_helper
from onnx import external_data_helper
from onnx import numpy_helper
from onnx import helper
from onnx import utils
from onnx import AttributeProto, TensorProto, GraphProto


onnx_model = onnx.load('/home/chasun/src/onnxruntime/cmake/models/opset9/tf_resnet_v1_50/model.onnx')

external_data_helper.convert_model_to_external_data(onnx_model, True)
onnx.save(onnx_model, '/tmp/model.onnx')
