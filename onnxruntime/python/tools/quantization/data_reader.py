from onnxruntime.quantization import CalibrationDataReader 
from .preprocessing import yolov3_preprocess_func, yolov3_vision_preprocess_func
import onnxruntime
from argparse import Namespace

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                               TensorDataset)
from tqdm import tqdm
from transformers import (BertConfig, BertForSequenceClassification, BertTokenizer,)
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features

import os
import logging
import numpy as np
import os
import random
import sys
import time
import torch

# Setup logging level to WARN. Change it accordingly
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

logging.getLogger("transformers.modeling_utils").setLevel(
   logging.INFO)  # Reduce logging

logger = logging.getLogger(__name__)


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

class ObejctDetectionDataReader(CalibrationDataReader):
    def __init__(self, model_path='augmented_model.onnx'):
        self.model_path = model_path
        self.preprocess_flag = None
        self.start_index = 0 
        self.end_index = 0 
        self.stride = 1 
        self.batch_size = 1 
        self.support_batch = False
        self.enum_data_dicts = iter([])
        self.batches = []

    def set_start_index(self, i):
        self.start_index = i

    def set_size_limit(self, limit):
        self.size_limit = limit

    def set_batch_size(self, batch_size):
        self.batches = []
        self.batch_size = batch_size

    def get_batch_size(self):
        return self.batch_size

    def set_preprocess_flag(self, flag):
        self.preprocess_flag = flag

class YoloV3DataReader(ObejctDetectionDataReader):
    def __init__(self, calibration_image_folder,
                       width=416,
                       height=416,
                       start_index=0,
                       stride=1,
                       batch_size=1,
                       model_path='augmented_model.onnx',
                       is_evaluation=False,
                       annotations='./annotations/instances_val2017.json'):
        ObejctDetectionDataReader.__init__(self, model_path)
        self.image_folder = calibration_image_folder
        self.model_path = model_path
        self.preprocess_flag = True
        self.enum_data_dicts = iter([])
        self.width = width
        self.height = height
        self.start_index = start_index
        self.end_index = len(os.listdir(calibration_image_folder)) 
        self.stride = stride if stride >= 1 else 1 # stride must > 0
        self.batch_size = batch_size
        self.is_evaluation = is_evaluation

        session = onnxruntime.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
        self.input_name = session.get_inputs()[0].name
        self.img_name_to_img_id = parse_annotations(annotations)

    def get_next(self):
        input_data = next(self.enum_data_dicts, None)
        if input_data:
            return input_data

        if self.start_index < self.end_index:
            if self.batch_size == 1:
                data = self.load_serial()
            else:
                data = self.load_batches()

            self.start_index += self.stride
            self.enum_data_dicts = iter(data)

            return next(self.enum_data_dicts, None)
        else:
            return None

    def load_serial(self):
        width = self.width 
        height = self.width 
        nchw_data_list, filename_list, image_size_list = yolov3_preprocess_func(self.image_folder, height, width, self.start_index, self.stride)
        input_name = self.input_name

        data = []
        if self.is_evaluation:
            img_name_to_img_id = self.img_name_to_img_id 
            for i in range(len(nchw_data_list)):
                nhwc_data = nchw_data_list[i]
                file_name = filename_list[i]
                data.append({input_name: nhwc_data, "image_shape": image_size_list[i], "image_id": img_name_to_img_id[file_name]})

        else:
            for i in range(len(nchw_data_list)):
                nhwc_data = nchw_data_list[i]
                file_name = filename_list[i]
                data.append({input_name: nhwc_data, "image_shape": image_size_list[i]})
                # self.enum_data_dicts = iter([{input_name: nhwc_data, "image_shape": arr} for nhwc_data in nchw_data_list])
        return data

    def load_batches(self):
        width = self.width 
        height = self.height 
        batch_size = self.batch_size
        stride = self.stride
        input_name = self.input_name

        for index in range(0, stride, batch_size):
            start_index = self.start_index + index 
            print("Load batch from index %s ..." % (str(start_index)))
            nchw_data_list, filename_list, image_size_list = yolov3_preprocess_func(self.image_folder, height, width, start_index, batch_size)

            if nchw_data_list.size == 0:
                break

            nchw_data_batch = []
            image_id_batch = []
            batches = []
            if self.is_evaluation:
                img_name_to_img_id = self.img_name_to_img_id 
                for i in range(len(nchw_data_list)):
                    nhwc_data = np.squeeze(nchw_data_list[i], 0)
                    nchw_data_batch.append(nhwc_data)
                    img_name = filename_list[i]
                    image_id = img_name_to_img_id[img_name]
                    image_id_batch.append(image_id)
                batch_data = np.concatenate(np.expand_dims(nchw_data_batch, axis=0), axis=0)
                batch_id = np.concatenate(np.expand_dims(image_id_batch, axis=0), axis=0)
                print(batch_data.shape)
                data = {input_name: batch_data, "image_id": batch_id, "image_shape": np.asarray([[416, 416]], dtype=np.float32)}
            else:
                for i in range(len(nchw_data_list)):
                    nhwc_data = np.squeeze(nchw_data_list[i], 0)
                    nchw_data_batch.append(nhwc_data)
                batch_data = np.concatenate(np.expand_dims(nchw_data_batch, axis=0), axis=0)
                print(batch_data.shape)
                data = {input_name: batch_data, "image_shape": np.asarray([[416, 416]], dtype=np.float32)}

            batches.append(data)

        return batches


class YoloV3VisionDataReader(YoloV3DataReader):
    def __init__(self, calibration_image_folder,
                       width=608,
                       height=384,
                       start_index=0,
                       stride=0,
                       batch_size=1,
                       model_path='augmented_model.onnx',
                       is_evaluation=False,
                       annotations='./annotations/instances_val2017.json'):
        YoloV3DataReader.__init__(self, calibration_image_folder, width, height, start_index, stride, batch_size, model_path, is_evaluation, annotations)


    def load_serial(self):
        width = self.width 
        height = self.height 
        input_name = self.input_name
        nchw_data_list, filename_list, image_size_list = yolov3_vision_preprocess_func(self.image_folder, height, width, self.start_index, self.stride)

        data = []
        if self.is_evaluation:
            img_name_to_img_id = self.img_name_to_img_id
            for i in range(len(nchw_data_list)):
                nhwc_data = nchw_data_list[i]
                file_name = filename_list[i]
                data.append({input_name: nhwc_data, "image_id": img_name_to_img_id[file_name], "image_size": image_size_list[i]})

        else:
            for i in range(len(nchw_data_list)):
                nhwc_data = nchw_data_list[i]
                file_name = filename_list[i]
                data.append({input_name: nhwc_data})
                # self.enum_data_dicts = iter([{input_name: nhwc_data, "image_shape": arr} for nhwc_data in nchw_data_list])
        return data

    def load_batches(self):
        width = self.width 
        height = self.height 
        stride = self.stride
        batch_size = self.batch_size
        input_name = self.input_name

        batches = []
        for index in range(0, stride, batch_size):
            start_index = self.start_index + index 
            print("Load batch from index %s ..." % (str(start_index)))
            nchw_data_list, filename_list, image_size_list = yolov3_vision_preprocess_func(self.image_folder, height, width, start_index, batch_size)

            if nchw_data_list.size == 0:
                break

            nchw_data_batch = []
            image_id_batch = []
            if self.is_evaluation:
                img_name_to_img_id = self.img_name_to_img_id
                for i in range(len(nchw_data_list)):
                    nhwc_data = np.squeeze(nchw_data_list[i], 0)
                    nchw_data_batch.append(nhwc_data)
                    img_name = filename_list[i]
                    image_id = img_name_to_img_id[img_name]
                    image_id_batch.append(image_id)
                batch_data = np.concatenate(np.expand_dims(nchw_data_batch, axis=0), axis=0)
                batch_id = np.concatenate(np.expand_dims(image_id_batch, axis=0), axis=0)
                print(batch_data.shape)
                data = {input_name: batch_data, "image_size": image_size_list, "image_id": batch_id}
            else:
                for i in range(len(nchw_data_list)):
                    nhwc_data = np.squeeze(nchw_data_list[i], 0)
                    nchw_data_batch.append(nhwc_data)
                batch_data = np.concatenate(np.expand_dims(nchw_data_batch, axis=0), axis=0)
                print(batch_data.shape)
                data = {input_name: batch_data}

            batches.append(data)

        return batches

