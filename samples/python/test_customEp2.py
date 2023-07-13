from onnxruntime.capi import _pybind_state as C

#model_path = '/bert_ort/leca/models/CustomOpTwo.onnx'
model_path = '/bert_ort/leca/models/Detection/model.onnx'
shared_lib_path = '/bert_ort/leca/code/onnxruntime2/samples/customEP2/build/libcustomep2.so'

session_options =C.get_default_session_options()
sess = C.InferenceSession(session_options, model_path, True, True)
sess.initialize_session(['customEp2'], [{'shared_lib_path': shared_lib_path}], set())

print('Create custom EP success!')
