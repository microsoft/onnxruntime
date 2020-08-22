import os
import subprocess
from BaseModel import *

class TinyYolov3(BaseModel):
    def __init__(self, model_name='tiny_yolov3', providers=None):
        BaseModel.__init__(self, model_name, providers)
        self.inputs_ = []
        self.ref_outputs_ = []
        self.validate_decimal_ = 3

        self.model_path_ = os.path.join(os.getcwd(), "yolov3-tiny.onnx")

        if not os.path.exists(self.model_path_):
            subprocess.run("wget https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/tiny-yolov3/model/tiny-yolov3-11.tar.gz", shell=True, check=True)
            subprocess.run("tar zxf tiny-yolov3-11.tar.gz", shell=True, check=True)

        self.onnx_zoo_test_data_dir_ = os.path.join(os.getcwd())

    # override
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

