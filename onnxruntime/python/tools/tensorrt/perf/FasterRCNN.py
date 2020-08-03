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

class FasterRCNN(BaseModel):
    def __init__(self, model_name='FasterRCNN-10', providers=None): 
        BaseModel.__init__(self, model_name, providers)
        self.inputs_ = []
        self.ref_outputs_ = []
        self.validate_decimal_ = 3

        self.model_path_ = os.path.join(os.getcwd(), "faster_rcnn_R_50_FPN_1x.onnx")

        if not os.path.exists(self.model_path_):
            subprocess.run("wget https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-10.tar.gz", shell=True, check=True)
            subprocess.run("tar zxf FasterRCNN-10.tar.gz", shell=True, check=True)

        self.onnx_zoo_test_data_dir_ = os.getcwd()

        self.cvs_model_path_ = os.path.join(os.getcwd(), "..", "cvs_models", "fasterRCNN","faster_rcnn_general_Animal661", "faster_rcnn_general_Animal661.onnx")
        self.cvs_model_test_data_dir_ = os.path.join(os.getcwd(), "..", "cvs_models", "fasterRCNN", "faster_rcnn_general_Animal661") 

    
    def convert_inputs_for_cvs_model(self, inputs): 
        # inputs: [[test_data_0_input_0.pb, test_data_0_input_1.pb ...], [test_data_1_input_0.pb, test_data_1_input_1.pb ...] ...]

        new_inputs = []

        for input_data in inputs:
            input_data_pb = [] 
            for data in input_data:
                input_data_pb.append(np.expand_dims(data, axis=0))
            new_inputs.append(input_data_pb)
        return new_inputs

    def convert_outputs_from_cvs_model(self, outputs): 
        new_outputs = []

        for output_data in outputs:
            output_data_pb = [] 
            for data in output_data:
                print(data)
                output_data_pb.append(data[:4])
            new_outputs.append(output_data_pb)
        return new_outputs

    def preprocess(self):
        self.image_ = Image.open('dependencies/demo.jpg')
        image = self.image_

        # Resize
        ratio = 800.0 / min(image.size[0], image.size[1])
        image = image.resize((int(ratio * image.size[0]), int(ratio * image.size[1])), Image.BILINEAR)

        # Convert to BGR
        image = np.array(image)[:, :, [2, 1, 0]].astype('float32')

        # HWC -> CHW
        image = np.transpose(image, [2, 0, 1])

        # Normalize
        mean_vec = np.array([102.9801, 115.9465, 122.7717])
        for i in range(image.shape[0]):
            image[i, :, :] = image[i, :, :] - mean_vec[i]

        # Pad to be divisible of 32
        import math
        padded_h = int(math.ceil(image.shape[1] / 32) * 32)
        padded_w = int(math.ceil(image.shape[2] / 32) * 32)

        padded_image = np.zeros((3, padded_h, padded_w), dtype=np.float32)
        padded_image[:, :image.shape[1], :image.shape[2]] = image
        image = padded_image

        self.image_ = image

    def inference(self):
        self.outputs_ = []
        for test_data in self.inputs_:
            img_data = test_data[0]
            output = self.session_.run(None, {
                self.session_.get_inputs()[0].name: img_data
            })
            # self.outputs_.append([output[0]])
            self.outputs_.append(output)

        '''
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
        else:
            img_data = self.image_

            boxes, labels, scores = session.run(None, {
                session.get_inputs()[0].name: img_data
            })

            self.boxes_ = boxes
            self.labels_ = labels
            self.scores_ = scores
        '''

    def postprocess(self):
        import matplotlib as mpl
        mpl.use('Agg')

        classes = [line.rstrip('\n') for line in open('dependencies/coco_classes.txt')]
        classes[0:5]

        image = Image.open('dependencies/demo.jpg')
        boxes = self.boxes_
        labels = self.labels_
        scores = self.scores_
        score_threshold=0.7

        # Resize boxes
        ratio = 800.0 / min(image.size[0], image.size[1])
        boxes = boxes / ratio

        fig = plt.figure(figsize=(12,9))
        ax = fig.add_subplot(1,1,1)
        ax.imshow(image)

        # Showing boxes with score > 0.7
        for box, label, score in zip(boxes, labels, scores):
            if score > score_threshold:
                rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='b', facecolor='none')
                ax.annotate(classes[label] + ':' + str(np.round(score, 2)), (box[0], box[1]), color='w', fontsize=12)
                ax.add_patch(rect)
        fig.savefig('result.png')


        '''
        pred = boxes
        import cv2
        viz = cv2.imread('dependencies/demo.jpg')
        for i in range(pred.shape[0]):
            # skip low confidence detection
            if scores[i] < score_threshold:
                continue
            cv2.rectangle(viz, (pred[i, 0], pred[i, 1]), (pred[i, 2], pred[i, 3]), (0, 255, 0), 2)
            print('%d %d %d %d %f' % (
                round(pred[i, 0]), round(pred[i, 1]),
                round(pred[i, 2] - pred[i, 0]), round(pred[i, 3] - pred[i, 1]),
                scores[i]))

        cv2.imwrite('result.jpg', viz)
        '''



