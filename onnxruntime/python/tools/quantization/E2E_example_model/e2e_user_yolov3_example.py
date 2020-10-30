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
from onnxruntime.quantization import quantize_static, calibrate, CalibrationDataReader, get_dynamic_range_table

class YoloV3OnnxModelZooDataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder, start_index=0, size_limit=0, augmented_model_path='augmented_model.onnx'):
        self.image_folder = calibration_image_folder
        self.augmented_model_path = augmented_model_path
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.datasize = 0
        self.start_index = start_index
        self.size_limit = size_limit

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            session = onnxruntime.InferenceSession(self.augmented_model_path, None)
            print(session.get_inputs()[0].shape)
            height = 416
            width = 416
            nhwc_data_list = yolov3_preprocess_func(self.image_folder, height, width, self.start_index, self.size_limit)
            input_name = session.get_inputs()[0].name
            self.datasize = len(nhwc_data_list)
            arr = np.full((1,2), 0, dtype="float32")
            arr[0][0] = 506.0
            arr[0][1] = 640.0
            # self.enum_data_dicts["image_shape"] = arr
            self.enum_data_dicts = iter([{input_name: nhwc_data, "image_shape": arr} for nhwc_data in nhwc_data_list])
        return next(self.enum_data_dicts, None)

class YoloV3DataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder, start_index=0, size_limit=0, augmented_model_path='augmented_model.onnx'):
        self.image_folder = calibration_image_folder
        self.augmented_model_path = augmented_model_path
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.datasize = 0
        self.start_index = start_index
        self.size_limit = size_limit

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            session = onnxruntime.InferenceSession(self.augmented_model_path, None)
            (_, _, height, width) = session.get_inputs()[0].shape
            print(session.get_inputs()[0].shape)
            nhwc_data_list = yolov3_preprocess_func(self.image_folder, height, width, self.start_index, self.size_limit)
            input_name = session.get_inputs()[0].name
            self.datasize = len(nhwc_data_list)
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


    batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
    return batch_data

def preprocess_func(images_folder, height, width, size_limit=0):
    '''
    Loads a batch of images and preprocess them
    parameter images_folder: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    return: list of matrices characterizing multiple images
    '''
    image_names = os.listdir(images_folder)
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []

    for image_name in batch_filenames:
        image_filepath = images_folder + '/' + image_name
        pillow_img = Image.new("RGB", (width, height))
        pillow_img.paste(Image.open(image_filepath).resize((width, height)))
        input_data = np.float32(pillow_img) - \
        np.array([123.68, 116.78, 103.94], dtype=np.float32)
        nhwc_data = np.expand_dims(input_data, axis=0)
        unconcatenated_batch_data.append(nhwc_data)
    batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
    return batch_data

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
    

def main():

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
    main()
