import numpy
import onnxruntime
from onnxruntime.capi import _pybind_state as C
from onnxruntime.capi.onnxruntime_pybind11_state import RunOptions

#model_path = '/bert_ort/leca/models/CustomOpTwo.onnx'
#model_path = '/bert_ort/leca/models/Detection/model.onnx'
model_path = '/bert_ort/leca/models/Relu.onnx'
shared_lib_path = '/bert_ort/leca/code/onnxruntime2/samples/customEP2/build/libcustomep2.so'

session_options =C.get_default_session_options()
sess = C.InferenceSession(session_options, model_path, True, True)  # custom Op doesn't work because Graph::Resolve() is invoked here but registry info is the line below
sess.initialize_session(['customEp2'], [{'shared_lib_path': shared_lib_path, 'int_property':'3', 'str_property':'strval'}], set())
print('Create custom EP success!')

x = numpy.zeros(4, dtype=numpy.float32)
x[0], x[1], x[2], x[3] = -3, 5, -2, 4
x_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(x)

y = sess.run_with_ort_values({'x':x_ortvalue._get_c_value()}, ['graphOut'], RunOptions())[0].numpy()
print('y:')
print(y)
