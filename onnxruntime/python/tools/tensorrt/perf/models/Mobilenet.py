import os
import subprocess
from BaseModel import *

class Mobilenet(BaseModel):
    def __init__(self, model_name='Mobilenet-v2-1.0', providers=None):
        BaseModel.__init__(self, model_name, providers)
        self.inputs_ = []
        self.ref_outputs_ = []
        self.validate_decimal_ = 3

        self.model_path_ = os.path.join(os.getcwd(), "mobilenetv2-1.0", "mobilenetv2-1.0.onnx")

        if not os.path.exists(self.model_path_):
            subprocess.run("wget https://github.com/onnx/models/raw/master/vision/classification/mobilenet/model/mobilenetv2-7.tar.gz", shell=True, check=True)
            subprocess.run("tar zxf mobilenetv2-7.tar.gz", shell=True, check=True)

        self.onnx_zoo_test_data_dir_ = os.path.join(os.getcwd(), "mobilenetv2-1.0")

        self.cvs_model_path_ = os.path.join(os.getcwd(), "..", "cvs_models", "mobilenetv2_ssdlite_v2_general_flickr32", "mobilenetv2_ssdlite_v2_general.onnx")
        self.cvs_model_test_data_dir_ = os.path.join(os.getcwd(), "..", "cvs_models", "mobilenetv2_ssdlite_v2_general_flickr32")

    def inference(self):
        test_data_num = len(self.inputs_)
        input_name = self.session_.get_inputs()[0].name
        self.outputs_ = [[self.session_.run([], {input_name: self.inputs_[i][0]})[0]] for i in range(test_data_num)]

    # not use for perf
    def inference_old(self, input_list=None):
        if input_list:
            inputs = input_list
            test_data_num = len(input_list)
        else:
            inputs = self.inputs_
            test_data_num = 3

        session = self.session_

        # get the name of the first input of the model
        input_name = session.get_inputs()[0].name
        self.outputs_ = [[session.run([], {input_name: inputs[i][0]})[0]] for i in range(test_data_num)]

