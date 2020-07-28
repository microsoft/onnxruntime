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


    def preprocess(self):
        return

    def get_ort_inputs(self, inputs):
        data = {
            self.session_.get_inputs()[0].name: inputs 
        }

        return data

    def inference(self):
        # session = self.session_
        # if input_list:
            # outputs = []
            # for test_data in input_list:
                # img_data = test_data
                # output = session.run(None, {
                    # session.get_inputs()[0].name: img_data
                # })
                # outputs.append(output[0])
            # self.outputs_ = outputs

        self.outputs_ = []
        for test_data in self.inputs_:
            img_data = test_data
            output = self.session_.run(None, {
                self.session_.get_inputs()[0].name: img_data
            })
            self.outputs_.append(output[0])

    def postprocess(self):
        return

