import os
import sys
import numpy as np
import onnxruntime
import subprocess
import onnx
from onnx import numpy_helper
from BaseModel import *

class Resnet152v2(BaseModel):
    def __init__(self, model_name='Resnet-152-v2', providers=None): 
        BaseModel.__init__(self, model_name, providers)
        self.inputs_ = []
        self.ref_outputs_ = []

        self.model_path_ = os.path.join(os.getcwd(), "resnet152v2", "resnet152-v2-7.onnx")

        if not os.path.exists(self.model_path_):
            subprocess.run("wget https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet152-v2-7.tar.gz", shell=True, check=True)
            subprocess.run("tar zxf resnet152-v2-7.tar.gz", shell=True, check=True)

        self.onnx_zoo_test_data_dir_ = os.path.join(os.getcwd(), "resnet152v2") 

    def preprocess(self):
        return

    def inference(self):
        # if input_list:
            # inputs = input_list
            # test_data_num = len(input_list) 
        # else:
            # inputs = self.inputs_
            # test_data_num = 3

        # session = self.session_

        # # get the name of the first input of the model
        # input_name = session.get_inputs()[0].name
        # self.outputs_ = [[session.run([], {input_name: inputs[i][0]})[0]] for i in range(test_data_num)]


        test_data_num = len(self.inputs_)
        input_name = self.session_.get_inputs()[0].name
        self.outputs_ = [[self.session_.run([], {input_name: self.inputs_[i][0]})[0]] for i in range(test_data_num)]

    def postprocess(self):
        return

