#!/usr/bin/env python
# coding: utf-8
# -------------------------------------------------------------------------
# Copyright (c) Microsoft, Intel Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import sys
import numpy as np
import re
import subprocess
import json

import abc

import onnx
import onnxruntime
from onnx import helper, TensorProto, numpy_helper
from quantize import quantize, QuantizationMode
from calibrate import calibrate

#user-implement preprocess func
from data_preprocess import preprocess_func


class CalibrationDataReaderInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls,subclass):
        return (hasattr(subclass,'get_next') and callable(subclass.get_next) or NotImplemented)

    @abc.abstractmethod
    def get_next(self) -> dict:
        """generate the input data dict for ONNXinferenceSession run"""
        raise NotImplementedError

class CalibrationDataReader(CalibrationDataReaderInterface):
    def __init__(self,calibration_image_folder,augmented_model_path='augmented_model.onnx'): 
        self.image_folder = calibration_image_folder
        self.augmented_model_path = augmented_model_path
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.datasize = 0

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            session = onnxruntime.InferenceSession(self.augmented_model_path, None)
            (_,height,width,_) = session.get_inputs()[0].shape
            nhwc_data_list = preprocess_func(self.image_folder,height,width,size_limit = 0)
            input_name = session.get_inputs()[0].name
            self.datasize = len(nhwc_data_list)           
            self.enum_data_dicts = iter([{input_name:nhwc_data_list[i]} for i in range(self.datasize)])
        return next(self.enum_data_dicts,None)

def main():
    model_path = 'path/to/model.onnx'
    calibration_dataset_path = 'path/to/calibration_data_set'
    dr = CalibrationDataReader(calibration_dataset_path)
    #call calibrate to generate quantization dictionary containing the zero point and scale values
    quantization_params_dict = calibrate(model_path,dr)
    calibrated_quantized_model = quantize(onnx.load(model_path),
                                          quantization_mode=QuantizationMode.QLinearOps,
                                          force_fusions=False,
                                          quantization_params=quantization_params_dict)
    output_model_path = 'path/to/output_model.onnx'
    onnx.save(calibrated_quantized_model, output_model_path)
    print('Calibrated and quantized model saved.')

if __name__ == '__main__':
   main()
