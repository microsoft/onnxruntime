#!/usr/bin/env python
# coding: utf-8
# -------------------------------------------------------------------------
# Copyright (c) Microsoft, Intel Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from .calibrate import CalibrationDataReader, calibrate
import onnxruntime

class ONNXValidator: 
    def __init__(self, model_path, data_reader: CalibrationDataReader, providers=["CUDAExecutionProvider"], coco_classes="./coco-object-categories-2017.json", onnx_coco_classes="./onnx_coco_classes.txt"):
        '''
        :param model_path: ONNX model to validate 
        :param data_reader: user implemented object to read in and preprocess calibration dataset
                            based on CalibrationDataReader Interface

        '''
        self.model_path = model_path
        self.data_reader = data_reader
        self.providers = providers 
        self.coco_class_id_map = {} # class -> id
        self.class_list = []
        self.coco_result_list = []

        f = open(onnx_coco_classes, 'r')
        lines = f.readlines()
        for c in lines:
            self.class_list.append(c.strip('\n'))
        print(self.class_list)

        self.generate_class_id_map(coco_classes, onnx_coco_classes)


    def generate_class_id_map(self, coco_classes, onnx_coco_classes):
        import json
        with open(coco_classes) as f:
            classes = json.load(f)

        for c in classes:
            self.coco_class_id_map[c["name"]] = c["id"]

        print(self.coco_class_id_map)

    def get_result(self):
        return self.coco_result_list

    def generate(self):
        session = onnxruntime.InferenceSession(self.model_path, providers=self.providers)

        outputs = []
        while True:
            inputs = self.data_reader.get_next()
            if not inputs:
                break

            image_id = inputs["id"]
            del inputs["id"]
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

            
            idx_list = []
            idx = 0

            coco_out_classes = []
            for idx in range(len(out_classes)):
                class_idx = out_classes[idx]
                class_name = self.class_list[int(class_idx)]
                print(class_name)

                if class_name not in self.coco_class_id_map:
                    coco_out_classes.append(-1)
                    continue

                id = self.coco_class_id_map[class_name]
                coco_out_classes.append(id)
                idx_list.append(idx)
                

            print(out_boxes)
            print(out_scores)
            print(out_classes)
            print(image_id)
            print(coco_out_classes)
            print("----------------------")

            for i in idx_list:
                id = coco_out_classes[i]
                box = [str(out_boxes[i][1]), str(out_boxes[i][0]), str(out_boxes[i][3]), str(out_boxes[i][2])]
                score = str(out_scores[i])
                # box = [out_boxes[i][1], out_boxes[i][0], out_boxes[i][3], out_boxes[i][2]]
                # score = out_scores[i]
                self.coco_result_list.append({"image_id":image_id, "category_id":id, "bbox":box, "score":score})


