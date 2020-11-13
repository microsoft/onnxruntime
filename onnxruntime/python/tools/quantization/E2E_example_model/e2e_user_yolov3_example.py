import os
import sys
import numpy as np
import re
import abc
import subprocess
import json
from PIL import Image

import onnx
import onnxruntime
from onnx import helper, TensorProto, numpy_helper
from onnxruntime.quantization import quantize_static, calibrate, CalibrationDataReader, calculate_calibration_data, get_calibrator, YoloV3OnnxModelZooDataReader, YoloV3VisionDataReader, YoloV3VisionValidator, YoloV3Validator
import cv2

def generate_coco_list(json_file, plain_text_file):
    import json
    data = {}
    with open(json_file, 'r') as file:
        items = json.load(file)

    with open(plain_text_file, 'w') as file:
        for item in items:
            file.write(item["name"])
            file.write('\n')


def json_to_plain_text(json_file, plain_text_file):
    import json
    data = {}
    with open(json_file, 'r') as file:
        data = json.load(file)

    with open(plain_text_file, 'w') as file:
        for key, value in data.items():
            s = key + ' ' + str(max(abs(value[0]), abs(value[1]))) 
            file.write(s)
            file.write('\n')

'''
def sort_dynamic_range_file(a_file, b_file):
    file1 = open(a_file, 'r')
    Lines = file1.readlines()

    with open(b_file, 'w') as file:
        for s in sorted(Lines):
            file.write(s)
'''

def prediction_validation(prediction_result, annotations):
    # calling coco api
    import matplotlib.pyplot as plt
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import numpy as np
    import skimage.io as io
    import pylab
    pylab.rcParams['figure.figsize'] = (10.0, 8.0)


    annType = ['segm','bbox','keypoints']
    annType = annType[1]      #specify type here
    prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
    print('Running demo for *%s* results.'%(annType))

    # annFile = './annotations/instances_val2017.json'
    annFile = annotations
    cocoGt=COCO(annFile)

    # resFile = 'instances_val2014_fakebbox100_results.json'
    resFile = prediction_result 
    cocoDt=cocoGt.loadRes(resFile)

    imgIds=sorted(cocoGt.getImgIds())
    imgIds=imgIds[0:100]
    imgId = imgIds[np.random.randint(100)]


    # running evaluation
    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    cocoEval.params.imgIds  = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    
def get_prediction_validation():
    # input_model_path = './yolov3_new.onnx'
    input_model_path = './yolov3_merge_coco_openimage_500200_288x512_batch_nms_obj_300_score_0p35_iou_0p35_shape.onnx'
    # input_model_path = './yolov3_merge_coco_openimage_500200.384x608_batch_shape.onnx'
    validation_dataset = './val2017'
    # validation_dataset = './val2017short'
    annotations = './annotations/instances_val2017.json'
    image_names = os.listdir(validation_dataset)
    stride = 1000

    # image_names[:] = image_names[0:1]
    
    results = []
    for i in range(0, len(image_names), stride):
        print("Total %s images. Start to process from %s ..." % (str(len(image_names)), str(i)))
        # dr = YoloV3OnnxModelZooDataReader(validation_dataset, augmented_model_path=input_model_path, start_index=i, size_limit=stride, is_validation=True)
        # validator = YoloV3Validator(input_model_path, dr, providers=["CUDAExecutionProvider"])

        dr = YoloV3VisionDataReader(validation_dataset, augmented_model_path=input_model_path, start_index=i, size_limit=stride, is_validation=True)
        validator = YoloV3VisionValidator(input_model_path, dr, providers=["CPUExecutionProvider"])

        validator.predict()
        results += validator.get_result()[0]

    print("Total %s bounding boxes." % (len(results)))

    prediction_validation(results, annotations)


def generate_calibration_table():

    input_model_path = './yolov3_new.onnx'
    calibration_dataset = './test2017'
    augmented_model_path = 'augmented_model.onnx'
    image_names = os.listdir(calibration_dataset)


    if os.path.exists(augmented_model_path):
        os.remove(augmented_model_path)
    calibrator = None 

    stride = 1000 
    for i in range(0, len(image_names), stride):
        print("Total %s images. Start to process from %s ..." % (str(len(image_names)), str(i)))
        dr = YoloV3OnnxModelZooDataReader(calibration_dataset, start_index=i, size_limit=stride)

        if not calibrator:
            calibrator = get_calibrator(input_model_path, dr)
        else:
            calibrator.set_data_reader(dr)

        calculate_calibration_data(input_model_path, calibrator, implicitly_quantize_all_ops=True)

    if calibrator:
        calibrator.write_calibration_table()

    print('Calibration table saved.')



if __name__ == '__main__':
    # generate_calibration_table()
    get_prediction_validation()
    # generate_coco_list("coco-object-categories-2017.json", "logic.txt")
