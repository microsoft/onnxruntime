#!/usr/bin/env python
# coding: utf-8
# -------------------------------------------------------------------------
# Copyright (c) Microsoft, Intel Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import numpy as np
import onnx
import onnxruntime
from onnx import helper, TensorProto
from onnx import onnx_pb as onnx_proto

from .quant_utils import QuantType
from .registry import QLinearOpsRegistry

import abc
import itertools


class CalibrationDataReader(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'get_next') and callable(subclass.get_next) or NotImplemented)

    @abc.abstractmethod
    def get_next(self) -> dict:
        """generate the input data dict for ONNXinferenceSession run"""
        raise NotImplementedError

class ONNXCalibrater:
    def __init__(self, model_path, data_reader: CalibrationDataReader, calibrate_op_types, black_nodes, white_nodes,
                 augmented_model_path):
        '''
        :param model_path: ONNX model to calibrate
        :param data_reader: user implemented object to read in and preprocess calibration dataset
                            based on CalibrationDataReader Interface
        :param op_types: operator types to be calibrated and quantized, default = 'Conv,MatMul'
        :param black_nodes: operator names that should not be quantized, default = ''
        :param white_nodes: operator names that force to be quantized, default = ''
        :param augmented_model_path: save augmented_model to this path

        '''
        self.model_path = model_path
        self.data_reader = data_reader
        self.calibrate_op_types = calibrate_op_types
        self.black_nodes = black_nodes
        self.white_nodes = white_nodes
        self.augmented_model_path = augmented_model_path
        self.input_name_to_nodes = {}
        self.calibration_cache = {} # save temporary calibration table


    def set_data_reader(self, data_reader):
        self.data_reader = data_reader

    def get_calibration_cache(self):
        return self.calibration_cache

    def augment_graph(self, augment_all_ops=False):
        '''
        Adds ReduceMin and ReduceMax nodes to all quantization_candidates op type nodes in
        model and ensures their outputs are stored as part of the graph output
        :return: augmented ONNX model
        '''

        model = onnx.load(self.model_path)
        model = onnx.shape_inference.infer_shapes(model)
        value_infos = {vi.name: vi for vi in model.graph.value_info}
        value_infos.update({ot.name: ot for ot in model.graph.output})
        value_infos.update({it.name: it for it in model.graph.input})

        added_nodes = []
        added_outputs = []
        tensors_to_calibrate = set()

        for node in model.graph.node:
            if augment_all_ops:
                should_be_calibrate = True
            else:
                should_be_calibrate = ((node.op_type in self.calibrate_op_types) and
                                       (node.name not in self.black_nodes)) or (node.name in self.white_nodes)
            if should_be_calibrate:
                for tensor_name in itertools.chain(node.input, node.output):
                    if tensor_name in value_infos.keys():
                        vi = value_infos[tensor_name]
                        if vi.type.HasField('tensor_type') and vi.type.tensor_type.elem_type == TensorProto.FLOAT and (
                                tensor_name not in model.graph.initializer):
                            tensors_to_calibrate.add(tensor_name)
        
        # If augmenting all ops, it's possible that some nodes' input value are 0.
        # Can't reduce on dim with value of 0 if 'keepdims' is false, therefore set keepdims to 1.
        if augment_all_ops:
            keepdims_value = 1
        else:
            keepdims_value = 0

        for tensor in tensors_to_calibrate:
            # Adding ReduceMin nodes
            reduce_min_name = tensor + '_ReduceMin'
            reduce_min_node = onnx.helper.make_node('ReduceMin', [tensor], [tensor + '_ReduceMin'],
                                                    reduce_min_name,
                                                    keepdims=keepdims_value)

            added_nodes.append(reduce_min_node)
            added_outputs.append(helper.make_tensor_value_info(reduce_min_node.output[0], TensorProto.FLOAT, ()))

            # Adding ReduceMax nodes
            reduce_max_name = tensor + '_ReduceMax'
            reduce_max_node = onnx.helper.make_node('ReduceMax', [tensor], [tensor + '_ReduceMax'],
                                                    reduce_max_name,
                                                    keepdims=keepdims_value)

            added_nodes.append(reduce_max_node)
            added_outputs.append(helper.make_tensor_value_info(reduce_max_node.output[0], TensorProto.FLOAT, ()))

        model.graph.node.extend(added_nodes)
        model.graph.output.extend(added_outputs)

        return model

    #Using augmented outputs to generate inputs for quantization
    def get_intermediate_outputs(self, calib_mode='naive', providers=None):
        ''' 
            Gather intermediate model outputs after running inference
            parameter calib_mode: type 'naive' gives (ReduceMin, ReduceMax) pairs
                                for each augmented node across test data sets, where
                                the first element is a minimum of all ReduceMin values
                                and the second element is a maximum of all ReduceMax
                                values;
            :return: dictionary mapping: {added node names: (ReduceMin, ReduceMax) pairs }
        '''

        #conduct inference session and get intermediate outputs
        if providers:
            sess_options = onnxruntime.SessionOptions()
            sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL #ORT_ENABLE_BASIC
            session = onnxruntime.InferenceSession(self.augmented_model_path, sess_options=sess_options, providers=providers)
        else:
            session = onnxruntime.InferenceSession(self.augmented_model_path, None)

        #number of outputs in original model
        model = onnx.load(self.model_path)
        num_model_outputs = len(model.graph.output)
        model_original_outputs = set(output.name for output in model.graph.output)

        intermediate_outputs = []

        while True:
            inputs = self.data_reader.get_next()
            if not inputs:
                break
            intermediate_outputs.append(session.run(None, inputs))


        node_output_names = [session.get_outputs()[i].name for i in range(len(intermediate_outputs[0]))]
        output_dicts_list = [
            dict(zip(node_output_names, intermediate_output)) for intermediate_output in intermediate_outputs
        ]


        merged_dict = {}
        for d in output_dicts_list:
            for k, v in d.items():
                merged_dict.setdefault(k, []).append(v)
        added_node_output_names = node_output_names[num_model_outputs:]
        node_names = [added_node_output_names[i].rpartition('_')[0]
                      for i in range(0, len(added_node_output_names), 2)]  #output names

        # Characterizing distribution of a node's values across test data sets
        clean_merged_dict = dict((i, merged_dict[i]) for i in merged_dict if i not in model_original_outputs)


        if calib_mode == 'naive':

            pairs = []
            for i in range(0, len(added_node_output_names), 2):
                min_value = 0
                max_value = 0
                min_value_array = min(clean_merged_dict[added_node_output_names[i]]) 
                max_value_array = max(clean_merged_dict[added_node_output_names[i + 1]])
                if type(min_value_array) == int or min_value_array.size > 0:
                    min_value = float(min_value_array)
                if type(max_value_array) == int or max_value_array.size > 0:
                    max_value = float(max_value_array)

                pairs.append(tuple([min_value, max_value]))

        else:
            raise ValueError('Unknown value for calib_mode. Currently only naive mode is supported.')

        final_dict = dict(zip(node_names, pairs))


        # merge new calibration data with previous calibration data
        if len(self.calibration_cache) > 0:
            for key, value in self.calibration_cache.items():
                min_value = min(value[0], final_dict[key][0])
                max_value = max(value[1], final_dict[key][1])
                final_dict[key] = (min_value, max_value)

        self.calibration_cache = final_dict

        return final_dict

    def _get_input_name_to_nodes(self, model):
        '''
            Helper function to get input_name_to_nodes dictionary
        '''

        for node in model.graph.node:
            for input_name in node.input:
                if input_name not in self.input_name_to_nodes:
                    self.input_name_to_nodes[input_name] = [node]
                else:
                    self.input_name_to_nodes[input_name].append(node)

    def calculate_scale_zeropoint(self, next_node, rmin, rmax):

        zp_and_scale = []
        # adjust rmin and rmax such that 0 is included in the range. This is required
        # to make sure zero can be uniquely represented.
        rmin = min(rmin, 0)
        rmax = max(rmax, 0)

        # We update the output range min and max when next node is clip or relu
        # With this technique we can remove these 2 ops and
        # reduce the output range which in turn helps to improve accuracy
        if next_node:
            if next_node.op_type == 'Clip':
                clip_min = next_node.attribute[0].f
                clip_max = next_node.attribute[1].f
                if rmin < clip_min:
                    rmin = clip_min
                if rmax > clip_max:
                    rmax = clip_max
            elif next_node.op_type == 'Relu':
                if rmin < 0:
                    rmin = 0

        scale = np.float32((rmax - rmin) / 255 if rmin != rmax else 1)
        initial_zero_point = (0 - rmin) / scale
        zero_point = np.uint8(round(max(0, min(255, initial_zero_point))))

        zp_and_scale.append(zero_point)
        zp_and_scale.append(scale)

        return zp_and_scale

    def calculate_quantization_params(self, quantization_thresholds):
        '''
            Given quantization thresholds, calculate the quantization params.
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
            raise ValueError(
                'quantization thresholds is required to calculate quantization params (zero point and scale)')

        quantization_params = {}
        model = onnx.load(self.model_path)

        self._get_input_name_to_nodes(model)

        for tensor_name in quantization_thresholds.keys():
            child = None
            if tensor_name in self.input_name_to_nodes:
                children = self.input_name_to_nodes[tensor_name]
                if (len(children) == 1):
                    child = children[0]
            node_thresholds = quantization_thresholds[tensor_name]
            node_params = self.calculate_scale_zeropoint(child, node_thresholds[0], node_thresholds[1])
            quantization_params[tensor_name] = node_params

        return quantization_params

def get_calibrator(model_path,
                   data_reader: CalibrationDataReader,
                   op_types=['Conv', 'MatMul'],
                   black_nodes=[],
                   white_nodes=[],
                   augmented_model_path='augmented_model.onnx'):

    calibrator = ONNXCalibrater(model_path, data_reader, op_types, black_nodes, white_nodes, augmented_model_path)

    return calibrator

def calculate_calibration_data(model_path,
                               calibrator=None,
                               calibration_data_reader: CalibrationDataReader=None,
                               op_types_to_quantize=[],
                               activation_type=QuantType.QUInt8,
                               nodes_to_quantize=[],
                               nodes_to_exclude=[],
                               augmented_model_path='augmented_model.onnx'):

    if activation_type != QuantType.QUInt8:
        raise ValueError("Static quantization only support uint8 for activation now.")

    if not op_types_to_quantize or len(op_types_to_quantize) == 0:
        op_types_to_quantize = list(QLinearOpsRegistry.keys())

    print("model path: %s" % model_path)
    print("augmented model path: %s" % augmented_model_path)

    if not calibrator:
        calibrator = get_calibrator(model_path, calibration_data_reader, op_types_to_quantize, nodes_to_quantize, nodes_to_exclude, augmented_model_path=augmented_model_path)

    if not os.path.exists(augmented_model_path):
        augmented_model = calibrator.augment_graph(augment_all_ops=True)
        onnx.save(augmented_model, augmented_model_path)

    calibrator.get_intermediate_outputs(providers=["CUDAExecutionProvider"])

def generate_calibration_table(calibrator, model_path, augmented_model_path, remove_previous_flag, data_reader, calibration_dataset=None, stride=5000, batch_size=20):

    if remove_previous_flag and os.path.exists(augmented_model_path):
        os.remove(augmented_model_path)
        print("remove previously generated %s and start to generate a new one." % (augmented_model_path))

    if not calibrator:
        calibrator = get_calibrator(model_path, data_reader, augmented_model_path=augmented_model_path)
    calculate_calibration_data(model_path, calibrator, augmented_model_path=augmented_model_path)

    return calibrator.get_calibration_cache()

def calibrate(model_path,
              data_reader: CalibrationDataReader,
              op_types=['Conv', 'MatMul'],
              black_nodes=[],
              white_nodes=[],
              augmented_model_path='augmented_model.onnx'):
    '''
        Given an onnx model, augment and run the augmented model on calibration data set, aggregate and calculate the quantization parameters.
    :param model_path: ONNX model to calibrate
    :param data_reader: user implemented object to read in and preprocess calibration dataset based on CalibrationDataReader interface
    :param op_types: operator types to be calibrated and quantized, default = 'Conv,MatMul'
    :param black_nodes: operator names that should not be quantized, default = ''
    :param white_nodes: operator names that force to be quantized, default = ''
    :param augmented_model_path: save augmented_model to this path
    '''
    #1. initialize a calibrater
    calibrater = ONNXCalibrater(model_path, data_reader, op_types, black_nodes, white_nodes, augmented_model_path)
    #2. augment
    augmented_model = calibrater.augment_graph()
    onnx.save(augmented_model, augmented_model_path)
    #3. generate quantization thresholds
    dict_for_quantization = calibrater.get_intermediate_outputs()
    #4. generate quantization parameters dict
    quantization_params_dict = calibrater.calculate_quantization_params(dict_for_quantization)

    print("Calibrated,quantized parameters calculated and returned.")
    return quantization_params_dict
