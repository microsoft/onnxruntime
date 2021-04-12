import onnxruntime as ort
from onnxruntime.capi import _pybind_state as C

session_options = C.get_default_session_options()
sess = C.InferenceSession(session_options, "model.onnx", True, True)
sess.initialize_session(['my_ep'], 
                        [{'shared_lib_path':'C:/git/onnxruntime/build/Windows/Debug/Debug/my_execution_provider.dll',
                                     'device_id':'1', 'some_config':'val'}], 
                        set())
print("OK")