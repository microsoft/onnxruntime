#!/usr/bin/env python
# coding: utf-8
# -------------------------------------------------------------------------
# Copyright (c) Microsoft, Intel Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import sys
import argparse
import numpy as np
from PIL import Image
import onnx
import onnxruntime
from onnx import helper, TensorProto, numpy_helper
from quantize import quantize, QuantizationMode
from data_preprocess import load_batch

import re
import subprocess
import json


def augment_graph(model, quantization_candidates=['Conv', 'MatMul'], black_nodes=[], white_nodes=[]):
    '''
    Adds ReduceMin and ReduceMax nodes to all quantization_candidates op type nodes in
    model and ensures their outputs are stored as part of the graph output
        parameter model: loaded FP32 ONNX model to quantize
        parameter quantization_candidates: node op types for nodes to be quantized.
                                           Calibraton will be done for them.
        parameter black_nodes: nodes with these names will be force ignored by this
                               calibration augmentation, no mather what's their op type.
        parameter white_nodes: nodes with these names will be force to be calibration augmented.
        return: augmented ONNX model
    '''

    added_nodes = []
    added_outputs = []
    for node in model.graph.node:
        should_be_calibrate = ((node.op_type in quantization_candidates) and
                               (node.name not in black_nodes)) or (node.name in white_nodes)
        if should_be_calibrate:
            input_name = node.output[0]
            # Adding ReduceMin nodes
            reduce_min_name = ''
            if node.name != '':
                reduce_min_name = node.name + '_ReduceMin'
            reduce_min_node = onnx.helper.make_node('ReduceMin', [input_name], [input_name + '_ReduceMin'],
                                                    reduce_min_name,
                                                    keepdims=0)
            added_nodes.append(reduce_min_node)
            added_outputs.append(helper.make_tensor_value_info(reduce_min_node.output[0], TensorProto.FLOAT, ()))

            # Adding ReduceMax nodes
            reduce_max_name = ''
            if node.name != '':
                reduce_max_name = node.name + '_ReduceMax'
            reduce_max_node = onnx.helper.make_node('ReduceMax', [input_name], [input_name + '_ReduceMax'],
                                                    reduce_max_name,
                                                    keepdims=0)
            added_nodes.append(reduce_max_node)
            added_outputs.append(helper.make_tensor_value_info(reduce_max_node.output[0], TensorProto.FLOAT, ()))
    model.graph.node.extend(added_nodes)
    model.graph.output.extend(added_outputs)
    return model


# Using augmented outputs to generate inputs to quantize.py


def get_intermediate_outputs(model_path, session, inputs, calib_mode='naive'):
    '''
    Gather intermediate model outputs after running inference
        parameter model_path: path to augmented FP32 ONNX model
        parameter inputs: list of loaded test inputs (or image matrices)
        parameter calib_mode: type 'naive' gives (ReduceMin, ReduceMax) pairs
                                for each augmented node across test data sets, where
                                the first element is a minimum of all ReduceMin values
                                and the second element is a maximum of all ReduceMax
                                values; more techniques can be added based on further experimentation
                                to improve the selection of the min max values. For example: some kind
                                of noise reduction can be applied before taking the min and max values.
        return: dictionary mapping added node names to (ReduceMin, ReduceMax) pairs
    '''
    model = onnx.load(model_path)
    # number of outputs in original model
    num_model_outputs = len(model.graph.output)
    num_inputs = len(inputs)
    input_name = session.get_inputs()[0].name
    intermediate_outputs = [session.run([], {input_name: inputs[i]}) for i in range(num_inputs)]

    # Creating dictionary with output results from multiple test inputs
    node_output_names = [session.get_outputs()[i].name for i in range(len(intermediate_outputs[0]))]
    output_dicts = [dict(zip(node_output_names, intermediate_outputs[i])) for i in range(num_inputs)]
    merged_dict = {}
    for d in output_dicts:
        for k, v in d.items():
            merged_dict.setdefault(k, []).append(v)
    added_node_output_names = node_output_names[num_model_outputs:]
    node_names = [added_node_output_names[i].rpartition('_')[0]
                  for i in range(0, len(added_node_output_names), 2)]  # output names

    # Characterizing distribution of a node's values across test data sets
    clean_merged_dict = dict((i, merged_dict[i]) for i in merged_dict if i != list(merged_dict.keys())[0])
    if calib_mode == 'naive':
        pairs = [
            tuple([
                float(min(clean_merged_dict[added_node_output_names[i]])),
                float(max(clean_merged_dict[added_node_output_names[i + 1]]))
            ]) for i in range(0, len(added_node_output_names), 2)
        ]
    else:
        raise ValueError('Unknown value for calib_mode. Currently only naive mode is supported.')

    final_dict = dict(zip(node_names, pairs))
    return final_dict


def calculate_scale_zeropoint(node, next_node, rmin, rmax):
    zp_and_scale = []
    # adjust rmin and rmax such that 0 is included in the range. This is required
    # to make sure zero can be uniquely represented.
    rmin = min(rmin, 0)
    rmax = max(rmax, 0)

    # We update the output range min and max when next node is clip or relu
    # With this technique we can remove these 2 ops and
    # reduce the output range which in turn helps to improve accuracy
    if next_node.op_type == 'Clip':
        clip_min = next_node.attribute[0].f
        clip_max = next_node.attribute[1].f
        if rmin < clip_min:
            rmin = clip_min
        if rmax > clip_max:
            rmax = clip_max
    if next_node.op_type == 'Relu':
        if rmin < 0:
            rmin = 0

    scale = np.float32((rmax - rmin) / 255 if rmin != rmax else 1)
    initial_zero_point = (0 - rmin) / scale
    zero_point = np.uint8(round(max(0, min(255, initial_zero_point))))

    zp_and_scale.append(zero_point)
    zp_and_scale.append(scale)
    return zp_and_scale


def calculate_quantization_params(model, quantization_thresholds):
    '''
        Given a model and quantization thresholds, calculates the quantization params.
    :param model: ModelProto to quantize
    :param quantization_thresholds:
        Dictionary specifying the min and max values for outputs of conv and matmul nodes.
        The quantization_thresholds should be specified in the following format:
            {
                "param_name": [min, max]
            }
        example:
            {
                'Conv_3:0': [np.float32(0), np.float32(0.5)],
                'Conv_4:0': [np.float32(1), np.float32(3.5)]
            }
    :return: Dictionary containing the zero point and scale values for outputs of conv and matmul nodes.
        The dictionary format is
            {
                "param_name": [zero_point, scale]
            }
    '''
    if quantization_thresholds is None:
        raise ValueError('quantization thresholds is required to calculate quantization params (zero point and scale)')

    quantization_params = {}
    for index, node in enumerate(model.graph.node):
        node_output_name = node.output[0]
        if node_output_name in quantization_thresholds:
            node_thresholds = quantization_thresholds[node_output_name]
            node_params = calculate_scale_zeropoint(node, model.graph.node[index + 1], node_thresholds[0],
                                                    node_thresholds[1])
            quantization_params[node_output_name] = node_params

    return quantization_params


def load_pb_file(data_file_name, size_limit, samples, channels, height, width):
    '''
    Load tensor data from pb files.
    :param data_file_name: path to the pb file
    :param dataset_size: number of image-data in the pb file. Default is 0 which means all samples from .pb file.
    :param samples: number of samples 'N'
    :param channels: number of channels in the image 'C'
    :param height: image height for data size check 'H'
    :param width: image width for data size check 'W'
    :return input data for the model
    '''
    tensor = onnx.TensorProto()
    inputs = np.empty(0)
    with open(data_file_name, 'rb') as fin:
        tensor.ParseFromString(fin.read())
        inputs = numpy_helper.to_array(tensor)
        try:
            shape = inputs.shape
            dataset_size = 1
            if len(shape) == 5 and (shape[0] <= size_limit or size_limit == 0):
                dataset_size = shape[0]
            elif len(shape) == 5 and shape[0] > size_limit:
                inputs = inputs[:size_limit]
                dataset_size = size_limit

            inputs = inputs.reshape(dataset_size, samples, channels, height, width)
        except:
            sys.exit(
                "Input .pb file contains incorrect input size. \nThe required size is: (%s). The real size is: (%s)" %
                ((dataset_size, samples, channels, height, width), shape))

    return inputs


def main():
    # Parsing command-line arguments
    parser = argparse.ArgumentParser(description='parsing model and test data set paths')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--force_fusions', default=False, action='store_true')
    parser.add_argument('--op_types',
                        type=str,
                        default='Conv,MatMul',
                        help='comma delimited operator types to be calibrated and quantized')
    parser.add_argument('--black_nodes',
                        type=str,
                        default='',
                        help='comma delimited operator names that should not be quantized')
    parser.add_argument('--white_nodes',
                        type=str,
                        default='',
                        help='comma delimited operator names force to be quantized')
    parser.add_argument('--augmented_model_path',
                        type=str,
                        default='augmented_model.onnx',
                        help='save augmented model to this file for verification purpose')
    parser.add_argument('--output_model_path', type=str, default='calibrated_quantized_model.onnx')
    parser.add_argument('--dataset_size',
                        type=int,
                        default=0,
                        help="Number of images or tensors to load. Default is 0 which means all samples")
    parser.add_argument('--data_preprocess',
                        type=str,
                        required=True,
                        choices=['preprocess_method1', 'preprocess_method2', 'None'],
                        help="Refer to Readme.md for guidance on choosing this option.")
    args = parser.parse_args()
    calibrate_op_types = args.op_types.split(',')
    black_nodes = args.black_nodes.split(',')
    black_nodes = [x for x in black_nodes if x]
    white_nodes = args.white_nodes.split(',')
    white_nodes = [x for x in white_nodes if x]
    model_path = args.model_path
    output_model_path = args.output_model_path
    images_folder = args.dataset_path
    calib_mode = "naive"
    size_limit = args.dataset_size

    # Generating augmented ONNX model
    model = onnx.load(model_path)
    augmented_model = augment_graph(model, calibrate_op_types, black_nodes, white_nodes)
    onnx.save(augmented_model, args.augmented_model_path)

    # Conducting inference
    session = onnxruntime.InferenceSession(args.augmented_model_path, None)
    (samples, channels, height, width) = session.get_inputs()[0].shape

    # Generating inputs for quantization
    if args.data_preprocess == "None":
        inputs = load_pb_file(images_folder, args.dataset_size, samples, channels, height, width)
    else:
        inputs = load_batch(images_folder, height, width, args.data_preprocess, size_limit)
    print(inputs.shape)
    dict_for_quantization = get_intermediate_outputs(model_path, session, inputs, calib_mode)
    quantization_params_dict = calculate_quantization_params(model, quantization_thresholds=dict_for_quantization)
    calibrated_quantized_model = quantize(onnx.load(model_path),
                                          quantization_mode=QuantizationMode.QLinearOps,
                                          force_fusions=args.force_fusions,
                                          quantization_params=quantization_params_dict)
    onnx.save(calibrated_quantized_model, output_model_path)

    print("Calibrated, quantized model saved.")


if __name__ == '__main__':
    main()
