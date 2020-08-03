import os
import sys
import numpy as np
import onnxruntime
import subprocess
import onnx
from onnx import numpy_helper
from BaseModel import *

class YoloV3(BaseModel):
    def __init__(self, model_name='yolov3', providers=None): 
        BaseModel.__init__(self, model_name, providers)
        self.inputs_ = []
        self.ref_outputs_ = []
        self.validate_decimal_ = 3 

        self.model_path_ = os.path.join(os.getcwd(), "yolov3", "yolov3.onnx")

        if not os.path.exists(self.model_path_):
            subprocess.run("wget https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/yolov3/model/yolov3-10.tar.gz", shell=True, check=True)
            subprocess.run("tar zxf yolov3-10.tar.gz", shell=True, check=True)

        self.onnx_zoo_test_data_dir_ = os.path.join(os.getcwd(), "yolov3") 
        
        self.cvs_model_path_ = os.path.join(os.getcwd(), "..", "cvs_models", "Yolov3_Pytorch", "yolov3-ms-openimages_40000_0627_608.batch.onnx")
        self.cvs_model_test_data_dir_ = os.path.join(os.getcwd(), "..", "cvs_models", "Yolov3_Pytorch") 


    def get_ort_inputs(self, ort_inputs):

        data = {}
        for i in range(len(ort_inputs)):
            ort_input = ort_inputs[i]
            name = self.session_.get_inputs()[i].name 
            data[name] = ort_input
        return data


    def inference(self):
        self.outputs_ = []

        for ort_inputs in self.inputs_:

            data = {}
            for i in range(len(ort_inputs)):
                ort_input = ort_inputs[i]
                name = self.session_.get_inputs()[i].name 
                data[name] = ort_input

            output = self.session_.run(None, data) 

            self.outputs_.append([output[0]])

    def postprocess(self):
        return
