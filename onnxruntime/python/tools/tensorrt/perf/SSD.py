import os
import sys
import numpy as np
import onnx
import onnxruntime
import subprocess
from PIL import Image
from BaseModel import *
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

class SSD(BaseModel):
    def __init__(self, model_name='SSD', providers=None): 
        BaseModel.__init__(self, model_name, providers)
        self.inputs_ = []
        self.ref_outputs_ = []

        if not os.path.exists("model.onnx"):
            subprocess.run("wget https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/ssd/model/ssd-10.tar.gz", shell=True, check=True)
            subprocess.run("tar zxf ssd-10.tar.gz", shell=True, check=True)

        self.onnx_zoo_test_data_dir_ = os.getcwd() 

        self.create_session('model.onnx')

    def preprocess(self):
        return

    def inference(self, input_list=None):
        session = self.session_
        if input_list:
            outputs = []
            for test_data in input_list:
                img_data = test_data[0]
                output = session.run(None, {
                    session.get_inputs()[0].name: img_data
                })
                outputs.append([output[0]])
            self.outputs_ = outputs

    def postprocess(self):
        return



