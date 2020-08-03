import os
import sys
import numpy as np
import onnxruntime
import subprocess
import onnx
from onnx import numpy_helper
from BaseModel import *

class ShufflenetV2(BaseModel):
    def __init__(self, model_name='Shufflenet-v2', providers=None): 
        BaseModel.__init__(self, model_name, providers)
        self.inputs_ = []
        self.ref_outputs_ = []
        # self.validate_decimal_ = 3 

        self.model_path_ = os.path.join(os.getcwd(), "model", "test_shufflenetv2", "model.onnx")

        if not os.path.exists(self.model_path_):
            subprocess.run("wget https://github.com/onnx/models/raw/master/vision/classification/shufflenet/model/shufflenet-v2-10.tar.gz", shell=True, check=True)
            subprocess.run("tar zxf shufflenet-v2-10.tar.gz", shell=True, check=True)

        self.onnx_zoo_test_data_dir_ = os.path.join(os.getcwd(), "model/test_shufflenetv2") 

        self.cvs_model_path_ = os.path.join(os.getcwd(), "..", "cvs_models", "shufflenetv2_general_AnnaSoda", "shufflenetv2_general.onnx")
        self.cvs_model_test_data_dir_ = os.path.join(os.getcwd(), "..", "cvs_models", "shufflenetv2_general_AnnaSoda") 


    def preprocess(self):
        return

    def get_ort_inputs(self, inputs):
        if self.model_zoo_source_ == 'onnx':
            data = {
                self.session_.get_inputs()[0].name: inputs 
            }
        else:
            data = {
                self.session_.get_inputs()[0].name: inputs[0]
            }

        return data

    def inference(self):
        self.outputs_ = []

        if self.model_zoo_source_ == 'onnx':
            for test_data in self.inputs_:
                img_data = test_data
                output = self.session_.run(None, {
                    self.session_.get_inputs()[0].name: img_data
                })
                self.outputs_.append(output[0])
        else:
            for test_data in self.inputs_:
                img_data = test_data[0]
                output = self.session_.run(None, {
                    self.session_.get_inputs()[0].name: img_data
                })
                self.outputs_.append(output)

    def postprocess(self):
        return

