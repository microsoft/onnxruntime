import os
import subprocess
from BaseModel import *

class TinyYolov2(BaseModel):
    def __init__(self, model_name='tiny_yolov2', providers=None): 
        BaseModel.__init__(self, model_name, providers)
        self.inputs_ = []
        self.ref_outputs_ = []
        self.validate_decimal_ = 3 

        self.model_path_ = os.path.join(os.getcwd(), "tiny_yolov2", "model.onnx")

        if not os.path.exists(self.model_path_):
            subprocess.run("wget https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-7.tar.gz", shell=True, check=True)
            subprocess.run("tar zxf tinyyolov2-7.tar.gz", shell=True, check=True)

        self.onnx_zoo_test_data_dir_ = os.path.join(os.getcwd(), "tiny_yolov2") 

        self.cvs_model_path_ = os.path.join(os.getcwd(), "..", "cvs_models", "tinyyolov2_general_Flickr32", "tinyyolov2_general.onnx")
        self.cvs_model_test_data_dir_ = os.path.join(os.getcwd(), "..", "cvs_models", "tinyyolov2_general_Flickr32") 

    def inference(self):
        self.outputs_ = []
        for test_data in self.inputs_:
            img_data = test_data[0]
            output = self.session_.run(None, {
                self.session_.get_inputs()[0].name: img_data
            })
            self.outputs_.append([output[0]])
        session = self.session_

