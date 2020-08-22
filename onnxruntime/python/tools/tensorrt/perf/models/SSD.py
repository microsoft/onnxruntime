import os
import subprocess
from BaseModel import *

class SSD(BaseModel):
    def __init__(self, model_name='SSD', providers=None):
        BaseModel.__init__(self, model_name, providers)
        self.inputs_ = []
        self.ref_outputs_ = []

        self.model_path_ = os.path.join(os.getcwd(), "model.onnx")

        if not os.path.exists(self.model_path_):
            subprocess.run("wget https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/ssd/model/ssd-10.tar.gz", shell=True, check=True)
            subprocess.run("tar zxf ssd-10.tar.gz", shell=True, check=True)

        self.onnx_zoo_test_data_dir_ = os.getcwd()

    def inference(self):
        self.outputs_ = []
        for test_data in self.inputs_:
            img_data = test_data[0]
            output = self.session_.run(None, {
                self.session_.get_inputs()[0].name: img_data
            })
            self.outputs_.append([output[0]])
