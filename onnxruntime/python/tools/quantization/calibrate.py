#!/usr/bin/env python
# coding: utf-8
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import argparse
import numpy as np
from PIL import Image
import onnx
import onnxruntime
from onnx import helper, TensorProto
#from smooth_average import smooth_average

# Adding ability to record intermediate graph outputs
selected_types = ['Conv', 'MatMul'] # node types to extend

def augment_graph(model):
    '''
    Adds ReduceMin and ReduceMax nodes to all Conv and MatMul nodes in
    model and ensures their outputs are stored as part of the graph output
        parameter model: loaded FP32 ONNX model to quantize
        return: augmented ONNX model
    '''
    added_nodes = []
    added_outputs = []
    for node in model.graph.node:
        if node.op_type in selected_types:
            input_name = node.output[0]
            # Adding ReduceMin nodes
            reduce_min_name = node.name + '_ReduceMin'
            reduce_min_node = onnx.helper.make_node('ReduceMin', [input_name],
                            [input_name + '_ReduceMin'], reduce_min_name, keepdims=0)
            added_nodes.append(reduce_min_node)
            added_outputs.append(helper.make_tensor_value_info(reduce_min_node.output[0], TensorProto.FLOAT, ()))

            # Adding ReduceMax nodes
            reduce_max_name = node.name + '_ReduceMax'
            reduce_max_node = onnx.helper.make_node('ReduceMax', [input_name],
                            [input_name + '_ReduceMax'], reduce_max_name, keepdims=0)
            added_nodes.append(reduce_max_node)
            added_outputs.append(helper.make_tensor_value_info(reduce_max_node.output[0], TensorProto.FLOAT, ()))
    model.graph.node.extend(added_nodes)
    model.graph.output.extend(added_outputs)
    return model

# Reading user input (images)
def load_and_resize_image(image_filepath, height, width):
    '''
    Resizes image to NCHW format
        parameter image_filepath: path to image files
        parameter height: image height in pixels
        parameter width: image width in pixels
        return: matrix characterizing image
    '''
    pillow_img = Image.open(image_filepath).resize((width, height))
    input_data = np.float32(pillow_img)/127.5 - 1.0 # normalization
    input_data -= np.mean(input_data) # normalization
    nhwc_data = np.expand_dims(input_data, axis=0)
    nchw_data = nhwc_data.transpose(0, 3, 1, 2) # ONNX Runtime standard
    return nchw_data

def load_batch(images_folder, height, width, size_limit=30):
    '''
    Loads a batch of images
    parameter images_folder: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images used to run inference
    return: list of matrices characterizing multiple images
    '''
    image_names = os.listdir(images_folder)
    if len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []
    for image_name in batch_filenames:
        image_filepath = images_folder + '/' + image_name
        nchw_data = load_and_resize_image(image_filepath, height, width)
        unconcatenated_batch_data.append(nchw_data)
    batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
    return batch_data

# Using augmented outputs to generate inputs to quantize.py
def get_intermediate_outputs(model_path, session, inputs, calib_mode='naive'):
    '''
    Gather intermediate model outputs after running inference
        parameter model: path to augmented FP32 ONNX model
        parameter inputs: list of loaded test inputs (or image matrices)
        parameter average_mode: type 'naive' gives (ReduceMin, ReduceMax) pairs
                                for each augmented node across test data sets, where
                                the first element is a minimum of all ReduceMin values
                                and the second element is a maximum of all ReduceMax
                                values; the type 'smooth' yields a smooth average
                                with anomalous values removed
        return: dictionary mapping added node names to (ReduceMin, ReduceMax) pairs
    '''
    model = onnx.load(model_path)
    num_model_outputs = len(model.graph.output) # number of outputs in original model
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
    node_names = [added_node_output_names[i].rpartition('_')[0] for i in range(0, len(added_node_output_names), 2)] # output names

    # Characterizing distribution of a node's values across test data sets
    clean_merged_dict = dict((i, merged_dict[i]) for i in merged_dict if i != list(merged_dict.keys())[0])
    if calib_mode == 'naive':
        pairs = [tuple([float(min(clean_merged_dict[added_node_output_names[i]])),
                float(max(clean_merged_dict[added_node_output_names[i+1]]))])
                for i in range(0, len(added_node_output_names), 2)]
    elif calib_mode == 'smooth':
        # Calls smoooth averaging script (number of bootstraps and confidence threshold are adjustable)
        pairs = [tuple([float(smooth_average(sorted(clean_merged_dict[added_node_output_names[i]]))),
                float(smooth_average(sorted(clean_merged_dict[added_node_output_names[i+1]])))])
                for i in range(0, len(added_node_output_names), 2)]
    final_dict = dict(zip(node_names, pairs))
    return final_dict

def main():
    # Parsing command-line arguments
    parser = argparse.ArgumentParser(description='parsing model and test data set paths')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--calib_mode', default='naive')
    parser.add_argument('--dataset_size', type=int, default=30)
    args = parser.parse_args()
    model_path = args.model_path
    images_folder = args.dataset_path
    calib_mode = args.calib_mode
    size_limit = args.dataset_size

    # Generating augmented ONNX model
    augmented_model_path = 'augmented_model.onnx'
    model = onnx.load(model_path)
    augmented_model = augment_graph(model)
    onnx.save(augmented_model, augmented_model_path)

    # Conducting inference
    session = onnxruntime.InferenceSession(augmented_model_path, None)
    (samples, channels, height, width) = session.get_inputs()[0].shape

    # Generating inputs for quantization
    inputs = load_batch(images_folder, height, width, size_limit)
    dict_for_quantization = get_intermediate_outputs(model_path, session, inputs, calib_mode)
    print(dict_for_quantization)
    return dict_for_quantization

if __name__ == '__main__':
    main()
