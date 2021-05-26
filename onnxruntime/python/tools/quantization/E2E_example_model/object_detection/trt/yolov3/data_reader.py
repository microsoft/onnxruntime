from onnxruntime.quantization import CalibrationDataReader
from preprocessing import yolov3_preprocess_func, yolov3_preprocess_func_2, yolov3_variant_preprocess_func, yolov3_variant_preprocess_func_2
import onnxruntime
from argparse import Namespace
import os
import numpy as np


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
        self.enum_data_dicts = iter([])
        self.input_name = None
        self.get_input_name()

    def get_batch_size(self):
        return self.batch_size

    def get_input_name(self):
        if self.input_name:
            return
        session = onnxruntime.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
        self.input_name = session.get_inputs()[0].name


class YoloV3DataReader(ObejctDetectionDataReader):
    def __init__(self,
                 calibration_image_folder,
                 width=416,
                 height=416,
                 start_index=0,
                 end_index=0,
                 stride=1,
                 batch_size=1,
                 model_path='augmented_model.onnx',
                 is_evaluation=False,
                 annotations='./annotations/instances_val2017.json',
                 preprocess_func=yolov3_preprocess_func):
        ObejctDetectionDataReader.__init__(self, model_path)
        self.image_folder = calibration_image_folder
        self.model_path = model_path
        self.preprocess_flag = True
        self.enum_data_dicts = iter([])
        self.width = width
        self.height = height
        self.start_index = start_index
        self.end_index = len(os.listdir(calibration_image_folder)) if end_index == 0 else end_index
        self.stride = stride if stride >= 1 else 1  # stride must > 0
        self.batch_size = batch_size
        self.is_evaluation = is_evaluation

        # self.input_name = 'input_1'
        self.img_name_to_img_id = parse_annotations(annotations)
        self.preprocess_func = preprocess_func

    def get_dataset_size(self):
        return len(os.listdir(self.image_folder))

    def get_next(self):
        iter_data = next(self.enum_data_dicts, None)
        if iter_data:
            return iter_data

        self.enum_data_dicts = None
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
        nchw_data_list, filename_list, image_size_list = preprocess_func(self.image_folder, height, width,
                                                                                self.start_index, self.stride)
        input_name = self.input_name

        print("Start from index %s ..." % (str(self.start_index)))
        data = []
        if self.is_evaluation:
            img_name_to_img_id = self.img_name_to_img_id
            for i in range(len(nchw_data_list)):
                nhwc_data = nchw_data_list[i]
                file_name = filename_list[i]
                data.append({
                    input_name: nhwc_data,
                    "image_shape": image_size_list[i],
                    "image_id": img_name_to_img_id[file_name]
                })

        else:
            for i in range(len(nchw_data_list)):
                nhwc_data = nchw_data_list[i]
                file_name = filename_list[i]
                data.append({input_name: nhwc_data, "image_shape": image_size_list[i]})
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
            nchw_data_list, filename_list, image_size_list = preprocess_func(self.image_folder, height, width,
                                                                                    start_index, batch_size)

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
                data = {
                    input_name: batch_data,
                    "image_id": batch_id,
                    "image_shape": np.asarray([[416, 416]], dtype=np.float32)
                }
            else:
                for i in range(len(nchw_data_list)):
                    nhwc_data = np.squeeze(nchw_data_list[i], 0)
                    nchw_data_batch.append(nhwc_data)
                batch_data = np.concatenate(np.expand_dims(nchw_data_batch, axis=0), axis=0)
                print(batch_data.shape)
                data = {input_name: batch_data, "image_shape": np.asarray([[416, 416]], dtype=np.float32)}

            batches.append(data)

        return batches


class YoloV3VariantDataReader(YoloV3DataReader):
    def __init__(self,
                 calibration_image_folder,
                 width=608,
                 height=384,
                 start_index=0,
                 end_index=0,
                 stride=1,
                 batch_size=1,
                 model_path='augmented_model.onnx',
                 is_evaluation=False,
                 annotations='./annotations/instances_val2017.json',
                 preprocess_func=yolov3_variant_preprocess_func):
        YoloV3DataReader.__init__(self, calibration_image_folder, width, height, start_index, end_index, stride,
                                  batch_size, model_path, is_evaluation, annotations, preprocess_func)
        # # self.input_name = '000_net'
        # self.input_name = 'images'

    def load_serial(self):
        width = self.width
        height = self.height
        input_name = self.input_name
        nchw_data_list, filename_list, image_size_list = self.preprocess_func(
            self.image_folder, height, width, self.start_index, self.stride)

        print("Start from index %s ..." % (str(self.start_index)))
        data = []
        if self.is_evaluation:
            img_name_to_img_id = self.img_name_to_img_id
            for i in range(len(nchw_data_list)):
                nhwc_data = nchw_data_list[i]
                file_name = filename_list[i]
                data.append({
                    input_name: nhwc_data,
                    "image_id": img_name_to_img_id[file_name],
                    "image_size": image_size_list[i]
                })

        else:
            for i in range(len(nchw_data_list)):
                nhwc_data = nchw_data_list[i]
                file_name = filename_list[i]
                data.append({input_name: nhwc_data})
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
            nchw_data_list, filename_list, image_size_list = preprocess_func(
                self.image_folder, height, width, start_index, batch_size)

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
