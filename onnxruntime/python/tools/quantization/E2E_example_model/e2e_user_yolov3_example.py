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
from onnxruntime.quantization import quantize_static, calibrate, CalibrationDataReader, get_dynamic_range_table, ONNXValidator 

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

class YoloV3OnnxModelZooDataReader(CalibrationDataReader):
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
            print(session.get_inputs()[0].shape)
            width = 416
            height = 416
            nhwc_data_list, filename_list, image_size_list = yolov3_preprocess_func(self.image_folder, height, width, self.start_index, self.size_limit)
            input_name = session.get_inputs()[0].name
            self.datasize = len(nhwc_data_list)

            # if self.is_validation:
                # annotations = './annotations/instances_val2017.json'
                # img_name_to_img_id = parse_annotations(annotations)
                # data = []
                # for i in range(len(nhwc_data_list)):
                    # nhwc_data = nhwc_data_list[i]
                    # file_name = filename_list[i]
                    # data.append({input_name: nhwc_data, "image_shape": arr, "id": img_name_to_img_id[file_name]})
                # self.enum_data_dicts = iter(data)
            # else:
                # self.enum_data_dicts = iter([{input_name: nhwc_data, "image_shape": arr} for nhwc_data in nhwc_data_list])

            annotations = './annotations/instances_val2017.json'
            img_name_to_img_id = parse_annotations(annotations)
            data = []
            for i in range(len(nhwc_data_list)):
                nhwc_data = nhwc_data_list[i]
                file_name = filename_list[i]
                data.append({input_name: nhwc_data, "image_shape": image_size_list[i], "id": img_name_to_img_id[file_name]})
            self.enum_data_dicts = iter(data)
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
            nhwc_data_list, filename_list, _ = yolov3_preprocess_func(self.image_folder, height, width, self.start_index, self.size_limit)
            input_name = session.get_inputs()[0].name
            self.datasize = len(nhwc_data_list)

            if self.is_validation:
                annotations = './annotations/instances_val2017.json'
                img_name_to_img_id, img_name_to_width_height = parse_annotations(annotations)
                data = []
                for i in range(len(nhwc_data_list)):
                    nhwc_data = nhwc_data_list[i]
                    file_name = filename_list[i]
                    data.append({input_name: nhwc_data, "id": img_name_to_img_id[file_name], "width_height": img_name_to_width_height[file_name]})
                self.enum_data_dicts = iter(data)
            else:
                self.enum_data_dicts = iter([{input_name: nhwc_data} for nhwc_data in nhwc_data_list])
        return next(self.enum_data_dicts, None)


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


    batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
    return batch_data, batch_filenames, image_size_list

def json_to_plain_text(json_file, plain_text_file):
    import json
    data = {}
    with open(json_file, 'r') as file:
        data = json.load(file)

    with open(plain_text_file, 'w') as file:
        for key, value in data.items():
            s = key + ': ' + str(max(abs(value[0]), abs(value[1]))) 
            file.write(s)
            file.write('\n')

# def sort_dynamic_range_file(a_file, b_file):
    # file1 = open(a_file, 'r')
    # Lines = file1.readlines()

    # with open(b_file, 'w') as file:
        # for s in sorted(Lines):
            # file.write(s)


    
def start_to_generate_prediction_result_for_validation():
    input_model_path = './yolov3_new.onnx'
    validate_dataset = './val2017'
    annotations = './annotations/instances_val2017.json'
    image_names = os.listdir(validate_dataset)
    length = len(image_names)
    length = 3
    stride = 3000
    stride = 3
    end_index = stride
    
    results = []
    for i in range(0, length, stride):
        dr = YoloV3OnnxModelZooDataReader(validate_dataset, augmented_model_path=input_model_path, start_index=i, size_limit=stride, is_validation=True)
        validator = ONNXValidator(input_model_path, dr)
        validator.generate()
        results += validator.get_result()

    print(results)
    with open('prediction.json', 'w') as file:
        file.write(json.dumps(results)) # use `json.loads` to do the reverse



def start_to_generate_cal_table():

    # sort_dynamic_range_file("final_dynamic_range", "final_dynamic_range_sort")

    # input_model_path = './yolov3_merge_coco_openimage_500200.384x608_batch_shape.onnx'
    input_model_path = './yolov3_new.onnx'
    output_model_path = './calibrated_quantized_model.onnx'
    calibration_dataset_path = './test2017'
    # calibration_dataset_path = './test2017short'
    image_names = os.listdir(calibration_dataset_path)
    stride = 3000
    end_index = stride
    for i in range(0, len(image_names), stride):
        # dr = YoloV3DataReader(calibration_dataset_path, start_index=i, size_limit=stride)
        dr = YoloV3OnnxModelZooDataReader(calibration_dataset_path, start_index=i, size_limit=stride)
        end_index = i + stride if (i + stride) <= len(image_names) else len(image_names)
        get_dynamic_range_table(input_model_path, output_model_path, dr, batch_end_index=end_index, implicitly_quantize_all_ops=True)

    json_to_plain_text("table/"+"dynamic_range_"+str(end_index)+".json", "final_dynamic_range")
    
    print('Calibrated and quantized model saved.')


if __name__ == '__main__':
    start_to_generate_prediction_result_for_validation()
    # start_to_generate_cal_table()
