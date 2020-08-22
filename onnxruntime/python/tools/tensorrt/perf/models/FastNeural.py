import os
import subprocess
from BaseModel import *

class FastNeural(BaseModel):
    def __init__(self, model_name='Fast-Neural', providers=None):
        BaseModel.__init__(self, model_name, providers)
        self.inputs_ = []
        self.ref_outputs_ = []
        self.validate_decimal_ = 3

        self.model_path_ = os.path.join(os.getcwd(), "mosaic", "mosaic.onnx")

        if not os.path.exists(self.model_path_):
            subprocess.run("wget https://github.com/onnx/models/raw/master/vision/style_transfer/fast_neural_style/model/mosaic-9.tar.gz", shell=True, check=True)
            subprocess.run("tar zxf mosaic-9.tar.gz", shell=True, check=True)

        self.onnx_zoo_test_data_dir_ = os.path.join(os.getcwd(), "mosaic")

    def get_ort_inputs(self, inputs):
        data = {
            self.session_.get_inputs()[0].name: inputs[0]
        }

        return data

    def get_ort_outputs(self):
        return None

    def inference(self):
        self.outputs_ = []
        for test_data in self.inputs_:
            img_data = test_data[0]
            output = self.session_.run(None, {
                self.session_.get_inputs()[0].name: img_data
            })
            self.outputs_.append([output[0]])
