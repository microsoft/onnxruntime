import os
import sys
import numpy as np
import re
import abc
import subprocess
import json
import argparse
import time

import onnx
import onnxruntime
from onnx import helper, TensorProto, numpy_helper
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType


class GenerateRandomCalibrationData(CalibrationDataReader):
    def __init__(self, input_tensor_shape, augmented_model_path='augmented_model.onnx'):
        self.augmented_model_path = augmented_model_path
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.datasize = 0
        self.model_input_shape = input_tensor_shape

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            session = onnxruntime.InferenceSession(self.augmented_model_path, None)
            nhwc_data_list = []
            nhwc_data_list.append(np.random.rand(*(self.model_input_shape)).astype('f'))
            input_name = session.get_inputs()[0].name
            self.datasize = len(nhwc_data_list)
            self.enum_data_dicts = iter([{input_name: nhwc_data} for nhwc_data in nhwc_data_list])
        return next(self.enum_data_dicts, None)

def benchmark(model_path, model_input_shape):
    session = onnxruntime.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    total = 0.0
    runs = 10
    input_data = np.random.rand(*(model_input_shape)).astype('f')
    # Warming up
    _ = session.run([], {input_name: input_data})
    for i in range(runs):
        start = time.perf_counter()
        _ = session.run([], {input_name: input_data})
        end = (time.perf_counter() - start) * 1000
        total += end
        print(f"{end:.2f}ms")
    total /= runs
    print(f"Avg: {total:.2f}ms")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", required=True, help="input model")
    parser.add_argument("--output_model", required=True, help="output model")
    parser.add_argument("--input_tensor_shape", required=True, help="input shape of the model")
    parser.add_argument("--quant_format",
                        default=QuantFormat.QOperator,
                        type=QuantFormat.from_string,
                        choices=list(QuantFormat))
    parser.add_argument("--per_channel", default=False, type=bool)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    input_model_path = args.input_model
    ops_to_quantize=["Conv", "MatMul", "Add"]
    onnx_model = onnx.load(input_model_path)
    op_names_to_quantize = []
    for op in onnx_model.graph.node:
        if op.op_type in ops_to_quantize:
            print(op.op_type, op.name)
            op_names_to_quantize.append(op.name)

    output_model_path = args.output_model
    input_tensor_shape = args.input_tensor_shape.split(',')
    input_tensor_shape = [int(i) for i in input_tensor_shape] 
    dr = GenerateRandomCalibrationData(input_tensor_shape)
    quantize_static(input_model_path,
                    output_model_path,
                    dr,
                    quant_format=args.quant_format,
                    per_channel=args.per_channel,
                    weight_type=QuantType.QInt8,
                    nodes_to_quantize=op_names_to_quantize)
    print('Calibrated and quantized model saved.')

    print('benchmarking fp32 model...')
    benchmark(input_model_path, input_tensor_shape)

    print('benchmarking int8 model...')
    benchmark(output_model_path, input_tensor_shape)


if __name__ == '__main__':
    main()
