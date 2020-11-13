#!/usr/bin/env python
# coding: utf-8
# -------------------------------------------------------------------------
# Copyright (c) Microsoft, Intel Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from .calibrate import CalibrationDataReader, calibrate
import onnxruntime
import numpy as np

class YoloV3VisionValidator: 
    def __init__(self, model_path,
                       data_reader: CalibrationDataReader,
                       providers=["CUDAExecutionProvider"],
                       ground_truth_object_class_file="./coco-object-categories-2017.json",
                       onnx_object_class_file="./onnx_coco_classes.txt",
                       save_bbox_to_image=False):
        '''
        :param model_path: ONNX model to validate 
        :param data_reader: user implemented object to read in and preprocess calibration dataset
                            based on CalibrationDataReader Interface

        '''
        self.model_path = model_path
        self.data_reader = data_reader
        self.providers = providers 
        self.class_to_id = {} # object class -> id
        self.onnx_class_list = []
        self.prediction_result_list = []
        self.prediction_result_list_str = []
        self.save_bbox_to_image = save_bbox_to_image
        self.identical_class_map = {"motorbike": "motorcycle", "aeroplane": "airplane", "sofa": "couch", "pottedplant": "potted plant", "diningtable": "dining table", "tvmonitor": "tv"}

        f = open(onnx_object_class_file, 'r')
        lines = f.readlines()
        for c in lines:
            self.onnx_class_list.append(c.strip('\n'))

        self.generate_class_to_id(ground_truth_object_class_file)
        print(self.class_to_id)

    def generate_class_to_id(self, ground_truth_object_class_file):
        with open(ground_truth_object_class_file) as f:
            import json
            classes = json.load(f)

        for c in classes:
            self.class_to_id[c["name"]] = c["id"]

    def get_result(self):
        return self.prediction_result_list, self.prediction_result_list_str

    def predict(self):
        session = onnxruntime.InferenceSession(self.model_path, providers=self.providers)

        outputs = []
        while True:
            inputs = self.data_reader.get_next()
            if not inputs:
                break

            image_id = inputs["image_id"]
            del inputs["image_id"]

            print(inputs)

            output = session.run(None, inputs)
            outputs.append(output)

            out_boxes, out_scores, out_classes = [], [], []

            # boxes = output[1]
            # classes = output[2]
            # indices = output[0]

            print(output)
            print(output[0])
            print(output[1])
            print(output[2])

            # for idx_ in indices:
                # out_classes.append(idx_[1])
                # out_scores.append(scores[tuple(idx_)])
                # idx_1 = (idx_[0], idx_[2])
                # out_boxes.append(boxes[idx_1])

            for i in range(len(out_classes)):
                out_class = out_classes[i]
                class_name = self.onnx_class_list[int(out_class)]
                if class_name in self.identical_class_map:
                    class_name = self.identical_class_map[class_name]
                id = self.class_to_id[class_name]

                # box = [str(out_boxes[i][1]), str(out_boxes[i][0]), str(out_boxes[i][3]-out_boxes[i][1]), str(out_boxes[i][2]-out_boxes[i][0])]
                bbox = [out_boxes[i][1], out_boxes[i][0], out_boxes[i][3], out_boxes[i][2]]
                bbox_yxhw = [out_boxes[i][1], out_boxes[i][0], out_boxes[i][3]-out_boxes[i][1], out_boxes[i][2]-out_boxes[i][0]]
                bbox_yxhw_str = [str(out_boxes[i][1]), str(out_boxes[i][0]), str(out_boxes[i][3]-out_boxes[i][1]), str(out_boxes[i][2]-out_boxes[i][0])]
                score = str(out_scores[i])
                coor = np.array(bbox[:4], dtype=np.int32)
                c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])

                self.prediction_result_list.append({"image_id":int(image_id), "category_id":int(id), "bbox":bbox_yxhw, "score":out_scores[i]})
                # self.prediction_result_list_str.append({"image_id":image_id, "category_id":id, "bbox":bbox_yxhw_str, "score":score})


class YoloV3Validator: 
    def __init__(self, model_path,
                       data_reader: CalibrationDataReader,
                       providers=["CUDAExecutionProvider"],
                       ground_truth_object_class_file="./coco-object-categories-2017.json",
                       onnx_object_class_file="./onnx_coco_classes.txt",
                       save_bbox_to_image=False):
        '''
        :param model_path: ONNX model to validate 
        :param data_reader: user implemented object to read in and preprocess calibration dataset
                            based on CalibrationDataReader Interface

        '''
        self.model_path = model_path
        self.data_reader = data_reader
        self.providers = providers 
        self.class_to_id = {} # object class -> id
        self.onnx_class_list = []
        self.prediction_result_list = []
        self.prediction_result_list_str = []
        self.save_bbox_to_image = save_bbox_to_image
        self.identical_class_map = {"motorbike": "motorcycle", "aeroplane": "airplane", "sofa": "couch", "pottedplant": "potted plant", "diningtable": "dining table", "tvmonitor": "tv"}

        f = open(onnx_object_class_file, 'r')
        lines = f.readlines()
        for c in lines:
            self.onnx_class_list.append(c.strip('\n'))

        self.generate_class_to_id(ground_truth_object_class_file)
        print(self.class_to_id)


    def generate_class_to_id(self, ground_truth_object_class_file):
        with open(ground_truth_object_class_file) as f:
            import json
            classes = json.load(f)

        for c in classes:
            self.class_to_id[c["name"]] = c["id"]

    def get_result(self):
        return self.prediction_result_list, self.prediction_result_list_str

    def predict(self):
        session = onnxruntime.InferenceSession(self.model_path, providers=self.providers)

        outputs = []
        while True:
            inputs = self.data_reader.get_next()
            if not inputs:
                break

            image_id = inputs["image_id"]
            del inputs["image_id"]

            print(inputs)

            output = session.run(None, inputs)
            outputs.append(output)

            out_boxes, out_scores, out_classes = [], [], []
            boxes = output[0]
            scores = output[1]
            indices = output[2]

            print(boxes)
            print(scores)
            print(indices)

            for idx_ in indices:
                out_classes.append(idx_[1])
                out_scores.append(scores[tuple(idx_)])
                idx_1 = (idx_[0], idx_[2])
                out_boxes.append(boxes[idx_1])

            print("----")
            print(out_boxes)
            print(out_scores)
            print(out_classes)


            for i in range(len(out_classes)):
                out_class = out_classes[i]
                class_name = self.onnx_class_list[int(out_class)]
                if class_name in self.identical_class_map:
                    class_name = self.identical_class_map[class_name]
                id = self.class_to_id[class_name]

                # box = [str(out_boxes[i][1]), str(out_boxes[i][0]), str(out_boxes[i][3]-out_boxes[i][1]), str(out_boxes[i][2]-out_boxes[i][0])]
                bbox = [out_boxes[i][1], out_boxes[i][0], out_boxes[i][3], out_boxes[i][2]]
                bbox_yxhw = [out_boxes[i][1], out_boxes[i][0], out_boxes[i][3]-out_boxes[i][1], out_boxes[i][2]-out_boxes[i][0]]
                bbox_yxhw_str = [str(out_boxes[i][1]), str(out_boxes[i][0]), str(out_boxes[i][3]-out_boxes[i][1]), str(out_boxes[i][2]-out_boxes[i][0])]
                score = str(out_scores[i])
                coor = np.array(bbox[:4], dtype=np.int32)
                c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])

                self.prediction_result_list.append({"image_id":int(image_id), "category_id":int(id), "bbox":bbox_yxhw, "score":out_scores[i]})
                # self.prediction_result_list_str.append({"image_id":image_id, "category_id":id, "bbox":bbox_yxhw_str, "score":score})


class ONNXValidator_ori: 
    def __init__(self, model_path,
                       data_reader: CalibrationDataReader,
                       providers=["CUDAExecutionProvider"],
                       ground_truth_object_class_file="./coco-object-categories-2017.json",
                       onnx_object_class_file="./onnx_coco_classes.txt",
                       save_bbox_to_image=False):
        '''
        :param model_path: ONNX model to validate 
        :param data_reader: user implemented object to read in and preprocess calibration dataset
                            based on CalibrationDataReader Interface

        '''
        self.model_path = model_path
        self.data_reader = data_reader
        self.providers = providers 
        self.class_to_id = {} # object class -> id
        self.onnx_class_list = []
        self.prediction_result_list = []
        self.prediction_result_list_str = []
        self.save_bbox_to_image = save_bbox_to_image
        self.identical_class_map = {"motorbike": "motorcycle", "aeroplane": "airplane", "sofa": "couch", "pottedplant": "potted plant", "diningtable": "dining table", "tvmonitor": "tv"}

        f = open(onnx_object_class_file, 'r')
        lines = f.readlines()
        for c in lines:
            self.onnx_class_list.append(c.strip('\n'))

        self.generate_class_to_id(ground_truth_object_class_file)


    def generate_class_to_id(self, ground_truth_object_class_file):
        import json
        with open(ground_truth_object_class_file) as f:
            classes = json.load(f)

        for c in classes:
            self.class_to_id[c["name"]] = c["id"]

        print(self.class_to_id)

    def get_result(self):
        return self.prediction_result_list, self.prediction_result_list_str

    def predict(self):
        session = onnxruntime.InferenceSession(self.model_path, providers=self.providers)

        outputs = []
        while True:
            inputs = self.data_reader.get_next()
            if not inputs:
                break

            image_id = inputs["image_id"]
            del inputs["image_id"]

            if self.save_bbox_to_image and "file_name" in inputs:
                file_name = inputs["file_name"]
                del inputs["file_name"]

            output = session.run(None, inputs)
            outputs.append(output)

            out_boxes, out_scores, out_classes = [], [], []
            boxes = output[0]
            scores = output[1]
            indices = output[2]

            for idx_ in indices:
                out_classes.append(idx_[1])
                out_scores.append(scores[tuple(idx_)])
                idx_1 = (idx_[0], idx_[2])
                out_boxes.append(boxes[idx_1])


            for i in range(len(out_classes)):
                out_class = out_classes[i]
                class_name = self.onnx_class_list[int(out_class)]
                if class_name in self.identical_class_map:
                    class_name = self.identical_class_map[class_name]
                id = self.class_to_id[class_name]

                # box = [str(out_boxes[i][1]), str(out_boxes[i][0]), str(out_boxes[i][3]-out_boxes[i][1]), str(out_boxes[i][2]-out_boxes[i][0])]
                bbox = [out_boxes[i][1], out_boxes[i][0], out_boxes[i][3], out_boxes[i][2]]
                bbox_yxhw = [out_boxes[i][1], out_boxes[i][0], out_boxes[i][3]-out_boxes[i][1], out_boxes[i][2]-out_boxes[i][0]]
                bbox_yxhw_str = [str(out_boxes[i][1]), str(out_boxes[i][0]), str(out_boxes[i][3]-out_boxes[i][1]), str(out_boxes[i][2]-out_boxes[i][0])]
                score = str(out_scores[i])
                coor = np.array(bbox[:4], dtype=np.int32)
                c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])

                self.prediction_result_list.append({"image_id":int(image_id), "category_id":int(id), "bbox":bbox_yxhw, "score":out_scores[i]})
                # self.prediction_result_list_str.append({"image_id":image_id, "category_id":id, "bbox":bbox_yxhw_str, "score":score})





            '''
            idx_list = []
            coco_out_classes = []
            coco_out_class_names = []
            for idx in range(len(out_classes)):
                class_idx = out_classes[idx]
                class_name = self.onnx_class_list[int(class_idx)]
                
                if class_name in self.identical_class_map:
                    class_name = self.identical_class_map[class_name]

                coco_out_class_names.append(class_name)

                if class_name not in self.class_to_id:
                    print("-----------")
                    print(class_name)
                    print("-----------")
                    coco_out_classes.append(-1)
                    continue

                id = self.class_to_id[class_name]
                coco_out_classes.append(id)
                idx_list.append(idx)
                

            
            import cv2
            from PIL import Image
            import numpy as np
            if self.save_bbox_to_image:
                images_folder = "val2017"
                image_filepath = images_folder + '/' + file_name 
                img = Image.open(image_filepath) 
                image = np.array(img)
                image_h, image_w, _ = image.shape
                bbox_thick = int(0.6 * (image_h + image_w) / 600)
                fontScale = 0.5

                for i in idx_list:
                    id = coco_out_classes[i]
                    # box = [str(out_boxes[i][1]), str(out_boxes[i][0]), str(out_boxes[i][3]-out_boxes[i][1]), str(out_boxes[i][2]-out_boxes[i][0])]
                    bbox = [out_boxes[i][1], out_boxes[i][0], out_boxes[i][3], out_boxes[i][2]]
                    bbox_yxhw = [out_boxes[i][1], out_boxes[i][0], out_boxes[i][3]-out_boxes[i][1], out_boxes[i][2]-out_boxes[i][0]]
                    bbox_yxhw_str = [str(out_boxes[i][1]), str(out_boxes[i][0]), str(out_boxes[i][3]-out_boxes[i][1]), str(out_boxes[i][2]-out_boxes[i][0])]
                    score = str(out_scores[i])
                    coor = np.array(bbox[:4], dtype=np.int32)
                    c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])

                    self.prediction_result_list.append({"image_id":int(image_id), "category_id":int(id), "bbox":bbox_yxhw, "score":out_scores[i]})
                    self.prediction_result_list_str.append({"image_id":image_id, "category_id":id, "bbox":bbox_yxhw_str, "score":score})

                    cv2.rectangle(image, c1, c2, (255,0,0), bbox_thick)
                    show_label = True
                    if show_label:
                        bbox_mess = '%s: %.2f' % (coco_out_class_names[i], out_scores[i])
                        t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0]
                        cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), (255,0,0), -1)
                        cv2.putText(image, bbox_mess, (c1[0], c1[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)

                cv2.imwrite("bbox_"+file_name, image)
            else:
                for i in idx_list:
                    id = coco_out_classes[i]
                    # box = [str(out_boxes[i][1]), str(out_boxes[i][0]), str(out_boxes[i][3]-out_boxes[i][1]), str(out_boxes[i][2]-out_boxes[i][0])]
                    bbox = [out_boxes[i][1], out_boxes[i][0], out_boxes[i][3], out_boxes[i][2]]
                    bbox_yxhw = [out_boxes[i][1], out_boxes[i][0], out_boxes[i][3]-out_boxes[i][1], out_boxes[i][2]-out_boxes[i][0]]
                    bbox_yxhw_str = [str(out_boxes[i][1]), str(out_boxes[i][0]), str(out_boxes[i][3]-out_boxes[i][1]), str(out_boxes[i][2]-out_boxes[i][0])]
                    score = str(out_scores[i])
                    coor = np.array(bbox[:4], dtype=np.int32)
                    c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])

                    self.prediction_result_list.append({"image_id":int(image_id), "category_id":int(id), "bbox":bbox_yxhw, "score":out_scores[i]})
                    self.prediction_result_list_str.append({"image_id":image_id, "category_id":id, "bbox":bbox_yxhw_str, "score":score})

            '''
