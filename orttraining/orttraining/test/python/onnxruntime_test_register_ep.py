import unittest
import onnxruntime_pybind11_state as C
import os

class EPRegistrationTests(unittest.TestCase):
  def get_test_execution_provider_path(self):
      return os.path.join('.', 'libtest_execution_provider.so')

  def test_register_custom_eps(self):
    C._register_provider_lib('TestExecutionProvider', self.get_test_execution_provider_path(), {'some_config':'val'})
    
    assert 'TestExecutionProvider' in C.get_available_providers()
    
    this = os.path.dirname(__file__)
    custom_op_model = os.path.join(this, "testdata", "custom_execution_provider_library", "test_model.onnx")
    if not os.path.exists(custom_op_model):
        raise FileNotFoundError("Unable to find '{0}'".format(custom_op_model))

    session_options = C.get_default_session_options()
    sess = C.InferenceSession(session_options, custom_op_model, True, True)
    sess.initialize_session(['TestExecutionProvider'], 
                        [{'device_id':'0'}], 
                        set())
    print("Created session with customize execution provider successfully!")


if __name__ == '__main__':
  unittest.main()

