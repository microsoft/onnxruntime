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
from onnxruntime.quantization import quantize_static, calibrate, CalibrationDataReader, calculate_calibration_data, get_calibrator, YoloV3DataReader, YoloV3VisionDataReader, YoloV3VisionValidator, YoloV3Validator
import cv2

def prediction_evaluation(prediction_result, annotations):

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

    
def get_prediction_evaluation(model_path, augmented_model_path, validation_dataset, providers):
    image_list = os.listdir(validation_dataset)
    stride = 1000
    
    results = []
    for i in range(0, len(image_list), stride):
        print("Total %s images.\nStart to process from %s ..." % (str(len(image_list)), str(i)))
        dr = YoloV3DataReader(validation_dataset, augmented_model_path=model_path, start_index=i, size_limit=stride, is_validation=True)
        validator = YoloV3Validator(model_path, dr, providers=providers)

        # dr = YoloV3VisionDataReader(validation_dataset, augmented_model_path=model_path, start_index=i, size_limit=stride, is_validation=True)
        # validator = YoloV3VisionValidator(model_path, dr, providers=providers)

        validator.predict()
        results += validator.get_result()[0]

    print("Total %s bounding boxes." % (len(results)))

        
    annotations = './annotations/instances_val2017.json'
    prediction_evaluation(results, annotations)


def generate_calibration_table(model_path, augmented_model_path, calibration_dataset):

    image_list = os.listdir(calibration_dataset)

    if os.path.exists(augmented_model_path):
        os.remove(augmented_model_path)
    calibrator = None 

    stride = 1000 
    for i in range(0, len(image_list), stride):
        print("Total %s images.\nStart to process from %s ..." % (str(len(image_list)), str(i)))
        dr = YoloV3DataReader(calibration_dataset, start_index=i, size_limit=stride)

        if not calibrator:
            calibrator = get_calibrator(model_path, dr)
        else:
            calibrator.set_data_reader(dr)

        calculate_calibration_data(model_path, calibrator, implicitly_quantize_all_ops=True)

    if calibrator:
        calibrator.write_calibration_table()

    print('Calibration table saved.')


if __name__ == '__main__':

    model_path = './yolov3_new.onnx'
    # model_path = './yolov3_merge_coco_openimage_500200_288x512_batch_nms_obj_300_score_0p35_iou_0p35_shape.onnx'
    augmented_model_path = 'augmented_model.onnx'
    calibration_dataset = './test2017'
    validation_dataset = './val2017'

    generate_calibration_table(model_path, augmented_model_path, calibration_dataset)
    get_prediction_evaluation()




