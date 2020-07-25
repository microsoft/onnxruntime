import os
import sys
import numpy as np
import onnxruntime
import subprocess
import onnx
from onnx import numpy_helper
from BaseModel import *
import cv2
# from matplotlib.pyplot import imshow

class YoloV4(BaseModel):
    def __init__(self, model_name='yolov4', providers=None): 
        BaseModel.__init__(self, model_name, providers)
        self.inputs_ = []
        self.ref_outputs_ = []
        self.validate_decimal_ = 3 

        self.model_path_ = os.path.join(os.getcwd(), "yolov4", "yolov4.onnx")

        if not os.path.exists(self.model_path_):
            subprocess.run("wget https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/yolov4/model/yolov4.tar.gz", shell=True, check=True)
            subprocess.run("tar zxf yolov4.tar.gz", shell=True, check=True)

        self.onnx_zoo_test_data_dir_ = os.path.join(os.getcwd(), "custom_test_data") 

        
        # input_size = 416
        # original_image = cv2.imread("data/kite.jpg")
        # original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        # original_image_size = original_image.shape[:2]

        # image_data = self.image_preprocess(np.copy(original_image), [input_size, input_size])
        # self.image_data_ = image_data[np.newaxis, ...].astype(np.float32)

        # print("Preprocessed image shape:",self.image_data_.shape) # shape of the preprocessed input


    def get_ort_inputs(self, ort_inputs):

        data = {}
        for i in range(len(ort_inputs)):
            ort_input = ort_inputs[i]
            name = self.session_.get_inputs()[i].name 
            data[name] = ort_input
        return data

    def image_preprocess(self, image, target_size, gt_boxes=None):

        ih, iw = target_size
        h, w, _ = image.shape

        scale = min(iw/w, ih/h)
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        image_padded = np.full(shape=[ih, iw, 3], fill_value=128.0)
        dw, dh = (iw - nw) // 2, (ih-nh) // 2
        image_padded[dh:nh+dh, dw:nw+dw, :] = image_resized
        image_padded = image_padded / 255.

        if gt_boxes is None:
            return image_padded

        else:
            gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
            gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
            return image_padded, gt_boxes


    def inference(self):
        output_name = self.session_.get_outputs()[0].name
        input_name = self.session_.get_inputs()[0].name

        for test_data in self.inputs_:
            img_data = test_data[0]
            self.outputs_.append(self.session_.run([output_name], {input_name: img_data}))

    def inference_for_predicted_image(self):
        output_name = self.session_.get_outputs()[0].name
        input_name = self.session_.get_inputs()[0].name

        self.outputs_ = self.session_.run([output_name], {input_name: self.image_data_})[0]
        self.save_output_to_pb()
        self.postprocess()

    def save_output_to_pb(self):
        from onnx import numpy_helper
        import onnx
        import os

        data_dir = "./test_data_set"
        test_data_dir = os.path.join(data_dir)
        if not os.path.exists(test_data_dir):
            os.makedirs(test_data_dir)

        # Convert the NumPy array to a TensorProto
        tensor = numpy_helper.from_array(np.asarray(self.outputs_))
        # print('TensorProto:\n{}'.format(tensor))

        # Save the TensorProto
        with open(os.path.join(test_data_dir, "output_kite.pb"), 'wb') as f:
            f.write(tensor.SerializeToString())

    def postprocess(self):
        sys.path.append('.')
        from core.config import cfg
        import core.utils as utils
        from PIL import Image

        original_image = cv2.imread("./data/kite.jpg")
        original_image_size = original_image.shape[:2]
        input_size = 416

        ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS)
        STRIDES = np.array(cfg.YOLO.STRIDES)
        XYSCALE = cfg.YOLO.XYSCALE

        pred_bbox = utils.postprocess_bbbox(np.expand_dims(self.outputs_, axis=0), ANCHORS, STRIDES, XYSCALE)
        # pred_bbox = utils.postprocess_bbbox(np.asanyarray(self.outputs_), ANCHORS, STRIDES, XYSCALE)
        bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.25)
        bboxes = utils.nms(bboxes, 0.213, method='nms')
        image = utils.draw_bbox(original_image, bboxes)

        cv2.imwrite("result.jpg", image)
