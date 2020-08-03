import os
import sys
import numpy as np
import onnxruntime
import subprocess
import onnx
from onnx import numpy_helper
from BaseModel import *

class YoloV3PyTorch(BaseModel):
    def __init__(self, model_name='yolov3-pytorch', providers=None): 
        BaseModel.__init__(self, model_name, providers)
        self.inputs_ = []
        self.ref_outputs_ = []
        self.validate_decimal_ = 3 

        self.cvs_model_path_ = os.path.join(os.getcwd(), "..", "cvs_models", "Yolov3_Pytorch", "yolov3-ms-openimages_40000_0627_608.batch.onnx")
        self.cvs_model_test_data_dir_ = os.path.join(os.getcwd(), "..", "cvs_models", "Yolov3_Pytorch") 


    def preprocess(self):
        return

    def inference(self):
        self.outputs_ = []
        for test_data in self.inputs_:
            img_data = test_data[0]
            output = self.session_.run(None, {
                self.session_.get_inputs()[0].name: img_data
            })
            self.outputs_.append([output[0]])
        session = self.session_

        # if input_list:
            # outputs = []
            # for test_data in input_list:
                # img_data = test_data[0]
                # output = session.run(None, {
                    # session.get_inputs()[0].name: img_data
                # })
                # outputs.append([output[0]])
            # self.outputs_ = outputs

    def postprocess(self):
        return
