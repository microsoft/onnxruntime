import subprocess
import onnxruntime
import os
import sys

class BaseModel(object): 
    def __init__(self, model_name, providers):
        self.model_name_ = model_name 
        self.providers_ = providers
        self.model_zoo_source_ = 'onnx'
        self.model_path_ = None
        self.cvs_model_path_ = None
        self.session_ = None
        self.session_options_ = onnxruntime.SessionOptions()
        self.onnx_zoo_test_data_dir_ = None
        self.cvs_model_test_data_dir_ = None
        self.inputs_ = None
        self.outputs_ = []
        self.validate_decimal_ = 4
        self.cleanup_files = []

    def get_model_name(self):
        return self.model_name_

    def get_session(self):
        return self.session_

    def set_model_zoo_source(self, source):
        self.model_zoo_source_ = source

    def get_onnx_zoo_test_data_dir(self):
        return self.onnx_zoo_test_data_dir_

    def get_cvs_model_test_data_dir(self):
        return self.cvs_model_test_data_dir_

    def get_cvs_model_path(self):
        return self.cvs_model_path_

    def get_outputs(self):
        return self.outputs_

    def set_inputs(self, inputs):
        self.inputs_ = inputs

    def get_decimal(self):
        return self.validate_decimal_

    def get_session_options(self):
        return self.session_options_

    def set_session_options(self, session_options):
        self.session_options_ = session_options 

    def get_input_nodes(self):
        if self.session_:
            return self.session_.get_inputs()
        return None

    def get_output_nodes(self):
        if self.session_:
            return self.session_.get_outputs()
        return None

    def get_ort_inputs(self, inputs):
        data = {
            self.session_.get_inputs()[0].name: inputs[0] 
        }

        return data

    def get_ort_outputs(self):
        return None

    def convert_model_from_float_to_float16(self, model_path=None):
        # from onnxmltools.utils.float16_converter import convert_float_to_float16
        from onnxmltools.utils import load_model, save_model
        from float16 import convert_float_to_float16

        if not model_path:
            model_path = self.model_path_

        onnx_model = load_model(model_path)
        new_onnx_model = convert_float_to_float16(onnx_model)
        save_model(new_onnx_model, 'new_fp16_model.onnx')

        self.model_path_ = os.path.join(os.getcwd(), "new_fp16_model.onnx")

    def create_session(self, model_path=None):
        if not model_path:
            model_path = self.model_path_
        
        print(model_path)
        try: 
            self.session_ = onnxruntime.InferenceSession(model_path, providers=self.providers_, sess_options=self.session_options_)
            return
        except:
            print("Use symbolic_shape_infer.py")
        
        try:
            new_model_path = model_path[:].replace(".onnx", "_new.onnx")
            subprocess.run("python3 ../symbolic_shape_infer.py --input " + model_path + " --output " + new_model_path + " --auto_merge", shell=True, check=True)     
            self.session_ = onnxruntime.InferenceSession(new_model_path, providers=self.providers_, sess_options=self.session_options_)
            return
        except Exception as e:
            self.session_ = None
            print(e)
            raise
