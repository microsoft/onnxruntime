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


    def preprocess(self):
        return

    def inference(self, input_list=None):
        session = self.session_
        if input_list:
            outputs = []
            for test_data in input_list:
                img_data = test_data[0]
                img_data_2 = test_data[1]
                output = session.run(None, {
                    session.get_inputs()[0].name: img_data,
                    session.get_inputs()[1].name: img_data_2,
                })
                outputs.append([output[0]])
            self.outputs_ = outputs

    def postprocess(self):
        return
