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
from onnxruntime.quantization import quantize_static, calibrate, CalibrationDataReader, calculate_calibration_data, get_calibrator, ONNXValidator, YoloV3OnnxModelZooDataReader
import cv2

class YoloV3OnnxModelZooDataReader_(CalibrationDataReader):
    def __init__(self, calibration_image_folder, start_index=0, size_limit=0, augmented_model_path='augmented_model.onnx', is_validation=False, save_bbox_to_image=False, annotations='./annotations/instances_val2017.json'):
        self.image_folder = calibration_image_folder
        self.augmented_model_path = augmented_model_path
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.datasize = 0
        self.start_index = start_index
        self.size_limit = size_limit
        self.is_validation = is_validation
        self.save_bbox_to_image = save_bbox_to_image
        self.annotations = annotations

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            session = onnxruntime.InferenceSession(self.augmented_model_path, None)
            print(session.get_inputs()[0].shape)
            width = 416
            height = 416
            nchw_data_list, filename_list, image_size_list = yolov3_preprocess_func(self.image_folder, height, width, self.start_index, self.size_limit)
            input_name = session.get_inputs()[0].name
            self.datasize = len(nchw_data_list)

            data = []
            if self.is_validation:
                img_name_to_img_id = parse_annotations(self.annotations)
                for i in range(len(nchw_data_list)):
                    nhwc_data = nchw_data_list[i]
                    file_name = filename_list[i]
                    if self.save_bbox_to_image:
                        data.append({input_name: nhwc_data, "image_shape": image_size_list[i], "image_id": img_name_to_img_id[file_name], "file_name": file_name})
                    else:
                        data.append({input_name: nhwc_data, "image_shape": image_size_list[i], "image_id": img_name_to_img_id[file_name]})

            else:
                for i in range(len(nchw_data_list)):
                    nhwc_data = nchw_data_list[i]
                    file_name = filename_list[i]
                    data.append({input_name: nhwc_data, "image_shape": image_size_list[i]})
                    # self.enum_data_dicts = iter([{input_name: nhwc_data, "image_shape": arr} for nhwc_data in nchw_data_list])

            self.enum_data_dicts = iter(data)

            # annotations = './annotations/instances_val2017.json'
            # img_name_to_img_id = parse_annotations(annotations)
            # data = []
            # for i in range(len(nchw_data_list)):
                # nhwc_data = nchw_data_list[i]
                # file_name = filename_list[i]
                # if self.is_validation:
                    # data.append({input_name: nhwc_data, "image_shape": image_size_list[i], "file_name": file_name, "id": img_name_to_img_id[file_name]})
                # else:
                    # data.append({input_name: nhwc_data, "image_shape": image_size_list[i]})
            # self.enum_data_dicts = iter(data)

        return next(self.enum_data_dicts, None)

class YoloV3DataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder, start_index=0, size_limit=0, augmented_model_path='augmented_model.onnx', is_validation=False):
        self.image_folder = calibration_image_folder
        self.augmented_model_path = augmented_model_path
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.datasize = 0
        self.start_index = start_index
        self.size_limit = size_limit
        self.is_validation = is_validation

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            session = onnxruntime.InferenceSession(self.augmented_model_path, None)
            (_, _, height, width) = session.get_inputs()[0].shape
            print(session.get_inputs()[0].shape)
            nchw_data_list, filename_list, _ = yolov3_preprocess_func(self.image_folder, height, width, self.start_index, self.size_limit)
            input_name = session.get_inputs()[0].name
            self.datasize = len(nchw_data_list)

            if self.is_validation:
                annotations = './annotations/instances_val2017.json'
                img_name_to_img_id = parse_annotations(annotations)
                data = []
                for i in range(len(nchw_data_list)):
                    nhwc_data = nchw_data_list[i]
                    file_name = filename_list[i]
                    data.append({input_name: nhwc_data, "id": img_name_to_img_id[file_name], "width_height": img_name_to_width_height[file_name]})
                self.enum_data_dicts = iter(data)
            else:
                self.enum_data_dicts = iter([{input_name: nhwc_data} for nhwc_data in nchw_data_list])
        return next(self.enum_data_dicts, None)

def parse_annotations(filename):
    import json
    annotations = {}
    with open(filename, 'r') as f:
        annotations = json.load(f)


    img_name_to_img_id = {}
    for image in annotations["images"]:
        file_name = image["file_name"]
        img_name_to_img_id[file_name] = image["id"]

    return img_name_to_img_id

def yolov3_preprocess_func(images_folder, height, width, start_index=0, size_limit=0):
    '''
    Loads a batch of images and preprocess them
    parameter images_folder: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    return: list of matrices characterizing multiple images
    '''
    # this function is from yolo3.utils.letterbox_image
    def letterbox_image(image, size):
        '''resize image with unchanged aspect ratio using padding'''
        iw, ih = image.size
        w, h = size
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        return new_image

    image_names = os.listdir(images_folder)
    print(len(image_names))
    if size_limit > 0 and len(image_names) >= size_limit:
        end_index = start_index + size_limit
        if end_index > len(image_names):
            end_index = len(image_names)

        batch_filenames = [image_names[i] for i in range(start_index, end_index)]
    else:
        batch_filenames = image_names


    unconcatenated_batch_data = []
    image_size_list = []

    print(batch_filenames)

    for image_name in batch_filenames:
        image_filepath = images_folder + '/' + image_name
        img = Image.open(image_filepath) 
        model_image_size = (height, width)
        boxed_image = letterbox_image(img, tuple(reversed(model_image_size)))
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.transpose(image_data, [2, 0, 1])
        image_data = np.expand_dims(image_data, 0)
        unconcatenated_batch_data.append(image_data)
        image_size_list.append(np.array([img.size[1], img.size[0]], dtype=np.float32).reshape(1, 2))

        # print(image_filepath)
        # if image_filepath == "./val2017/000000157098.jpg":
            # # cv2.rectangle(img, (186.0152, 15.640884), (186.0152+367.55603, 15.640884+ 337.237), (255,0,0), 2)
            # # cv2.rectangle(img, (int(186), int(15)), (int(553), int(352)), (255,0,0), 2)

            # boxed_image = np.array(img)
            
            # # cv2.rectangle(boxed_image, (1, 2), (6, 12), (0, 0, 255), 2)
            # cv2.rectangle(boxed_image, (int(186), int(15)), (int(367), int(337)), (255,0,0), 2)
            # cv2.imwrite("my.png",boxed_image)


    batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
    return batch_data, batch_filenames, image_size_list

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

def prediction_validation(prediction_result, annotations)
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

    
def generate_prediction_result_for_validation():
    input_model_path = './yolov3_new.onnx'
    validate_dataset = './val2017'
    annotations = './annotations/instances_val2017.json'
    image_names = os.listdir(validate_dataset)
    stride = 500 
    
    results = []
    for i in range(0, len(image_names), stride):
        print("Total %s images. Start to process from %s ..." % (str(len(image_names)), str(i)))
        dr = YoloV3OnnxModelZooDataReader(validate_dataset, augmented_model_path=input_model_path, start_index=i, size_limit=stride, is_validation=True)
        validator = ONNXValidator(input_model_path, dr, providers=["TensorrtExecutionProvider", "CUDAExecutionProvider"])
        validator.predict()
        results += validator.get_result()[0]

    # print(results)
    # with open('prediction.json', 'w') as file:
        # file.write(json.dumps(results)) # use `json.loads` to do the reverse


    print("Total %s bounding boxes." % (len(results)))

    prediction_validation(results, annotations)


def generate_calibration_table():

    # input_model_path = './yolov3_merge_coco_openimage_500200.384x608_batch_shape.onnx'
    input_model_path = './yolov3_new.onnx'
    calibration_dataset = './test2017'
    # calibration_dataset = './val2017short'
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
    # generate_prediction_result_for_validation()
    generate_calibration_table()
    # sort_dynamic_range_file("final_dynamic_range", "final_dynamic_range_sort")
    # generate_coco_list("coco-object-categories-2017.json", "logic.txt")
