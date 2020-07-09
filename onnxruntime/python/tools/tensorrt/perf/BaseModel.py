import subprocess
import onnxruntime
import os

class BaseModel(object): 
    def __init__(self, model_name, providers):
        self.model_name_ = model_name 
        self.providers_ = providers
        self.session_ = None
        self.onnx_zoo_test_data_dir_ = None
        self.test_data_num_ = 1
        self.outputs_ = []
        self.validate_decimal_ = 4
        self.cleanup_files = []

    def get_model_name(self):
        return self.model_name_

    def get_session(self):
        return self.session_

    def get_onnx_zoo_test_data_dir(self):
        return self.onnx_zoo_test_data_dir_

    def get_outputs(self):
        return self.outputs_

    def get_decimal(self):
        return self.validate_decimal_

    def create_session(self, model_path):
        try: 
            self.session_ = onnxruntime.InferenceSession(model_path, providers=self.providers_)
        except:
            model_new_path = model_path[:].replace(".onnx", "_new.onnx")
            subprocess.run("python3 ../symbolic_shape_infer.py --input " + model_path + " --output " + model_new_path + " --auto_merge", shell=True, check=True)     
            self.session_ = onnxruntime.InferenceSession(model_new_path, providers=self.providers_)
