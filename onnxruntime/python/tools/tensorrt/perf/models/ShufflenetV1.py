import os
import sys
import numpy as np
import onnxruntime
import subprocess
import onnx
from onnx import numpy_helper
from BaseModel import *

class ShufflenetV1(BaseModel):
    def __init__(self, model_name='Shufflenet-v1', providers=None):
        BaseModel.__init__(self, model_name, providers)
        self.inputs_ = []
        self.ref_outputs_ = []
        # self.validate_decimal_ = 3

        self.model_path_ = os.path.join(os.getcwd(), "shufflenet", "model.onnx")

        if not os.path.exists(self.model_path_):
            subprocess.run("wget https://github.com/onnx/models/raw/master/vision/classification/shufflenet/model/shufflenet-9.tar.gz", shell=True, check=True)
            subprocess.run("tar zxf shufflenet-9.tar.gz", shell=True, check=True)

        self.onnx_zoo_test_data_dir_ = os.path.join(os.getcwd(), "shufflenet")

    def inference(self):
        self.outputs_ = []
        for test_data in self.inputs_:
            img_data = test_data[0]
            output = self.session_.run(None, {
                self.session_.get_inputs()[0].name: img_data
            })
            self.outputs_.append([output[0]])

