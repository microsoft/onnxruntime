import os
import subprocess
from BaseModel import *

class Resnet18v1(BaseModel):
    def __init__(self, model_name='Resnet18-v1', providers=None):
        BaseModel.__init__(self, model_name, providers)
        self.inputs_ = []
        self.ref_outputs_ = []

        self.model_path_ = os.path.join(os.getcwd(), "resnet18-v1-7", "resnet18-v1-7.onnx")

        if not os.path.exists(self.model_path_):
            subprocess.run("wget https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet18-v1-7.tar.gz", shell=True, check=True)
            subprocess.run("tar zxf resnet18-v1-7.tar.gz", shell=True, check=True)

        self.onnx_zoo_test_data_dir_ = os.path.join(os.getcwd(), "resnet18-v1-7")

    def inference(self):
        test_data_num = len(self.inputs_)
        input_name = self.session_.get_inputs()[0].name
        self.outputs_ = [[self.session_.run([], {input_name: self.inputs_[i][0]})[0]] for i in range(test_data_num)]

