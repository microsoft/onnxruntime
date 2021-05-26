#!/usr/bin/env python
# coding: utf-8
# -------------------------------------------------------------------------
# Copyright (c) Microsoft, Intel Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import onnxruntime
from onnxruntime.quantization.calibrate import CalibrationDataReader
import numpy as np

import torch
import torchvision

class YoloV3Evaluator:
    def __init__(self,
                 model_path,
                 data_reader: CalibrationDataReader,
                 width=416,
                 height=416,
                 providers=["CUDAExecutionProvider"],
                 ground_truth_object_class_file="./coco-object-categories-2017.json",
                 onnx_object_class_file="./onnx_coco_classes.txt"):
        '''
        :param model_path: ONNX model to validate 
        :param data_reader: user implemented object to read in and preprocess calibration dataset
                            based on CalibrationDataReader Interface

        '''
        self.model_path = model_path
        self.data_reader = data_reader
        self.width = width
        self.height = height
        self.providers = providers
        self.class_to_id = {}  # object class -> id
        self.onnx_class_list = []
        self.prediction_result_list = []
        self.identical_class_map = {
            "motorbike": "motorcycle",
            "aeroplane": "airplane",
            "sofa": "couch",
            "pottedplant": "potted plant",
            "diningtable": "dining table",
            "tvmonitor": "tv"
        }

        f = open(onnx_object_class_file, 'r')
        lines = f.readlines()
        for c in lines:
            self.onnx_class_list.append(c.strip('\n'))

        self.generate_class_to_id(ground_truth_object_class_file)
        print(self.class_to_id)

        self.session = onnxruntime.InferenceSession(model_path, providers=providers)

    def generate_class_to_id(self, ground_truth_object_class_file):
        with open(ground_truth_object_class_file) as f:
            import json
            classes = json.load(f)

        for c in classes:
            self.class_to_id[c["name"]] = c["id"]

    def set_data_reader(self, data_reader):
        self.data_reader = data_reader

    def get_result(self):
        return self.prediction_result_list

    def set_bbox_prediction(self, boxes, scores, indices, is_batch, image_id, image_id_batch):
        out_boxes, out_scores, out_classes, out_batch_index = [], [], [], []

        for idx_ in indices:
            out_classes.append(idx_[1])
            out_batch_index.append(idx_[0])
            out_scores.append(scores[tuple(idx_)])
            idx_1 = (idx_[0], idx_[2])
            out_boxes.append(boxes[idx_1])

        for i in range(len(out_classes)):
            out_class = out_classes[i]
            class_name = self.onnx_class_list[int(out_class)]
            if class_name in self.identical_class_map:
                class_name = self.identical_class_map[class_name]
            id = self.class_to_id[class_name]

            bbox = [out_boxes[i][1], out_boxes[i][0], out_boxes[i][3], out_boxes[i][2]]
            bbox_yxhw = [
                out_boxes[i][1], out_boxes[i][0], out_boxes[i][3] - out_boxes[i][1], out_boxes[i][2] - out_boxes[i][0]
            ]
            bbox_yxhw_str = [
                str(out_boxes[i][1]),
                str(out_boxes[i][0]),
                str(out_boxes[i][3] - out_boxes[i][1]),
                str(out_boxes[i][2] - out_boxes[i][0])
            ]
            score = str(out_scores[i])
            coor = np.array(bbox[:4], dtype=np.int32)
            c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])

            if is_batch:
                image_id = image_id_batch[out_batch_index[i]]
            self.prediction_result_list.append({
                "image_id": int(image_id),
                "category_id": int(id),
                "bbox": bbox_yxhw,
                "score": out_scores[i]
            })

    def predict(self):
        session = self.session

        outputs = []

        # If you decide to run batch inference, please make sure all input images must be re-sized to the same shape.
        # Which means the bounding boxes from groun truth annotation must to be adjusted accordingly, otherwise you will get very low mAP results.
        # Here we simply choose to run serial inference.
        if self.data_reader.get_batch_size() > 1:
            # batch inference
            print("Doing batch inference...")

            image_id_list = []
            image_id_batch = []
            while True:
                inputs = self.data_reader.get_next()
                if not inputs:
                    break
                image_id_list = inputs["image_id"]
                del inputs["image_id"]
                image_id_batch.append(image_id_list)
                outputs.append(session.run(None, inputs))

                for index in range(len(outputs)):
                    output = outputs[index]
                    boxes = output[0]
                    scores = output[1]
                    indices = output[2]

                    self.set_bbox_prediction(boxes, scores, indices, True, None, image_id_batch[index])
        else:
            # serial inference
            while True:
                inputs = self.data_reader.get_next()
                if not inputs:
                    break

                image_id = inputs["image_id"]
                del inputs["image_id"]

                output = session.run(None, inputs)

                boxes = output[0]
                scores = output[1]
                indices = output[2]

                self.set_bbox_prediction(boxes, scores, indices, False, image_id, None)

    def evaluate(self, prediction_result, annotations):
        # calling coco api
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        annFile = annotations
        cocoGt = COCO(annFile)

        resFile = prediction_result
        cocoDt = cocoGt.loadRes(resFile)

        imgIds = sorted(cocoGt.getImgIds())

        # running evaluation
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.imgIds = imgIds
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

class YoloV3VariantEvaluator(YoloV3Evaluator): 
    def __init__(self, model_path,
                       data_reader: CalibrationDataReader,
                       width=608,
                       height=384,
                       providers=["CUDAExecutionProvider"],
                       ground_truth_object_class_file="./coco-object-categories-2017.json",
                       onnx_object_class_file="./onnx_coco_classes.txt"):

        YoloV3Evaluator.__init__(self, model_path, data_reader,width, height, providers, ground_truth_object_class_file, onnx_object_class_file)

    def predict(self):
        from postprocessing import PostprocessYOLOWrapper 
        session = self.session
        outputs = []

        image_id_list = []
        image_id_batch = []
        image_size_list = []
        image_size_batch = []

        postprocess_yolo = PostprocessYOLOWrapper('yolov3', (608, 608))

        while True:
            inputs = self.data_reader.get_next()
            if not inputs:
                break
            image_size_list = inputs["image_size"]
            image_id_list = inputs["image_id"]
            del inputs["image_size"]
            del inputs["image_id"]

            # in the case of batch size is 1
            if type(image_id_list) == int:
                image_size_list = [image_size_list]
                image_id_list = [image_id_list]


            image_size_batch.append(image_size_list)
            image_id_batch.append(image_id_list)
            outputs.append(session.run(None, inputs))

        for i in range(len(outputs)):
            output = outputs[i]
            
            for batch_i in range(self.data_reader.get_batch_size()):

                if batch_i > len(image_size_batch[i])-1 or batch_i > len(image_id_batch[i])-1:
                    continue

                image_height = image_size_batch[i][batch_i][0]
                image_width= image_size_batch[i][batch_i][1]
                image_id = image_id_batch[i][batch_i]

                boxes, classes, scores = postprocess_yolo.postprocessor.process(
                output, (image_width, image_height), 0.01)

                for j in range(len(boxes)):
                    box = boxes[j]
                    class_name = self.onnx_class_list[int(classes[j])]
                    if class_name in self.identical_class_map:
                        class_name = self.identical_class_map[class_name]
                    id = self.class_to_id[class_name]
                    x = float(box[0])
                    y = float(box[1])
                    w = float(box[2] - box[0] + 1)
                    h = float(box[3] - box[1] + 1)
                    self.prediction_result_list.append({"image_id":int(image_id), "category_id":int(id), "bbox":[x,y,w,h], "score":scores[j]})

class YoloV3Variant2Evaluator(YoloV3Evaluator):
    def __init__(self,
                 model_path,
                 data_reader: CalibrationDataReader,
                 width=608,
                 height=384,
                 providers=["CUDAExecutionProvider"],
                 ground_truth_object_class_file="./coco-object-categories-2017.json",
                 onnx_object_class_file="./onnx_coco_classes.txt"):

        YoloV3Evaluator.__init__(self, model_path, data_reader, width, height, providers,
                                 ground_truth_object_class_file, onnx_object_class_file)

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            # gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[[0, 2]] -= pad[0]  # x padding
        coords[[1, 3]] -= pad[1]  # y padding
        coords[:4] /= gain
        return coords

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.zeros_like(x)
        y[0] = x[0] - x[2] / 2  # top left x
        y[1] = x[1] - x[3] / 2  # top left y
        y[2] = x[0] + x[2] / 2  # bottom right x
        y[3] = x[1] + x[3] / 2  # bottom right y
        return y

    def set_bbox_prediction(self, bboxes, scores, image_height, image_width, image_id):

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            bbox[0] *= self.width  #x
            bbox[1] *= self.height  #y
            bbox[2] *= self.width  #w
            bbox[3] *= self.height  #h

            img0_shape = (image_height, image_width)
            img1_shape = (self.height, self.width)
            bbox = self.xywh2xyxy(bbox)
            bbox = self.scale_coords(img1_shape, bbox, img0_shape)

            class_name = 'person'
            if class_name in self.identical_class_map:
                class_name = self.identical_class_map[class_name]
            id = self.class_to_id[class_name]

            bbox[2] = bbox[2] - bbox[0]
            bbox[3] = bbox[3] - bbox[1]

            self.prediction_result_list.append({
                "image_id": int(image_id),
                "category_id": int(id),
                "bbox": list(bbox),
                "score": scores[i][0]
            })

    def predict(self):
        session = self.session
        outputs = []

        image_id_list = []
        image_id_batch = []
        image_size_list = []
        image_size_batch = []
        while True:
            inputs = self.data_reader.get_next()
            if not inputs:
                break
            image_size_list = inputs["image_size"]
            image_id_list = inputs["image_id"]
            del inputs["image_size"]
            del inputs["image_id"]

            # in the case of batch size is 1
            if type(image_id_list) == int:
                image_size_list = [image_size_list]
                image_id_list = [image_id_list]

            image_size_batch.append(image_size_list)
            image_id_batch.append(image_id_list)
            outputs.append(session.run(None, inputs))

        for i in range(len(outputs)):
            output = outputs[i]
            for batch_i in range(self.data_reader.get_batch_size()):
                batch_idx = output[0][:, 0] == batch_i
                bboxes = output[1][batch_idx, :]
                scores = output[2][batch_idx, :]

                if batch_i > len(image_size_batch[i]) - 1 or batch_i > len(image_id_batch[i]) - 1:
                    continue

                image_height = image_size_batch[i][batch_i][0]
                image_width = image_size_batch[i][batch_i][1]
                image_id = image_id_batch[i][batch_i]
                self.set_bbox_prediction(bboxes, scores, image_height, image_width, image_id)

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        # gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    return coords


def letterbox(img, new_shape=(416, 416), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def post_process_without_nms(opts):
    final_output = []
    for batch_i in range(opt.batch_size):
        batch_idx = opts[0][:, 0] == batch_i
        bbox = opts[1][batch_idx, :]
        score = opts[2][batch_idx, :]
        bbox[:, 0] *= opt.input_w  #x
        bbox[:, 1] *= opt.input_h  #y
        bbox[:, 2] *= opt.input_w  #w
        bbox[:, 3] *= opt.input_h  #h
        bbox = xywh2xyxy(bbox)
        bbox0 = scale_coords(img.shape[2:], bbox, img0.shape[0:2])
        if bbox0.shape[0] == 0:
            final_output.append(torch.empty(0, 5).numpy())
            continue

        output = np.concatenate((bbox, score), axis=1)
        final_output.append(output)

    return final_output


def post_process_with_nms(predictions, image_height, image_width, conf_thres=0.35, nms_thres=0.35):
    """Performs NMS and score thresholding
        """
    final_output = []
    batch_size = 1
    input_w = 512
    input_h = 288
    for batch_i in range(batch_size):
        scores = predictions[0][batch_i, :, 0]
        keep_idx = scores >= conf_thres
        boxes_ = predictions[1][batch_i, keep_idx, :]
        boxes_[:, 0] *= input_w  #x
        boxes_[:, 1] *= input_h  #y
        boxes_[:, 2] *= input_w  #w
        boxes_[:, 3] *= input_h  #h
        boxes_ = xywh2xyxy(boxes_)
        img0_shape = (image_height, image_width)
        img1_shape = (input_h, input_w)
        # bbox = self.scale_coords(img1_shape, bbox, img0_shape)
        boxes_ = scale_coords(img1_shape, boxes_, img0_shape)
        # boxes_ = scale_coords(img.shape[2:], boxes_, img0.shape[0:2])
        boxes_ = torch.from_numpy(boxes_)
        scores = torch.from_numpy(scores[keep_idx])
        if scores.dim() == 0:
            final_output.append(torch.empty(0, 5).numpy())
            continue
        keep_idx = torchvision.ops.nms(boxes_, scores, nms_thres)
        scores = scores[keep_idx].view(-1, 1)
        boxes_ = boxes_[keep_idx].view(-1, 4)
        output = torch.cat((boxes_, scores), dim=-1)
        final_output.append(output.numpy())
    return final_output

class YoloV3Variant3Evaluator(YoloV3Evaluator):
    def __init__(self,
                 model_path,
                 data_reader: CalibrationDataReader,
                 width=512,
                 height=288,
                 providers=["CUDAExecutionProvider"],
                 ground_truth_object_class_file="./coco-object-categories-2017.json",
                 onnx_object_class_file="./onnx_coco_classes.txt"):

        YoloV3Evaluator.__init__(self, model_path, data_reader, width, height, providers,
                                 ground_truth_object_class_file, onnx_object_class_file)

    def predict(self):
        session = onnxruntime.InferenceSession(self.model_path, providers=self.providers)
        outputs = []

        image_id_list = []
        image_id_batch = []
        image_size_list = []
        image_size_batch = []
            
        class_name = 'person'
        id = self.class_to_id[class_name]

        while True:
            inputs = self.data_reader.get_next()
            if not inputs:
                break
            image_size_list = inputs["image_size"]
            image_id_list = inputs["image_id"]
            del inputs["image_size"]
            del inputs["image_id"]

            # in the case of batch size is 1
            if type(image_id_list) == int:
                image_size_list = [image_size_list]
                image_id_list = [image_id_list]

            image_size_batch.append(image_size_list)
            image_id_batch.append(image_id_list)
            outputs.append(session.run(None, inputs))

        for j in range(len(outputs)):
            output = outputs[j]
            image_id = image_id_batch[j][0]
            image_height = image_size_batch[j][0][0]
            image_width = image_size_batch[j][0][1]
            dets = post_process_with_nms(output, image_height, image_width)[0]

            for i in range(dets.shape[0]):
                x1 = dets[i, 0]
                y1 = dets[i, 1]
                x2 = dets[i, 2]
                y2 = dets[i, 3]
                score = dets[i, 4]

                bbox = [x1, y1, x2-x1, y2-y1]
                self.prediction_result_list.append({
                    "image_id": int(image_id),
                    "category_id": int(id),
                    "bbox": list(bbox),
                    "score": score 
                })

