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
from onnx import helper, TensorProto, ModelProto
from onnx import onnx_pb as onnx_proto
from six import string_types
from enum import Enum

from .quant_utils import QuantType
from .registry import QLinearOpsRegistry

import abc
import itertools


class CalibrationMethod(Enum):
    MinMax = 0


class CalibrationDataReader(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'get_next') and callable(subclass.get_next) or NotImplemented)

    @abc.abstractmethod
    def get_next(self) -> dict:
        """generate the input data dict for ONNXinferenceSession run"""
        raise NotImplementedError


class CalibraterBase:
    def __init__(self, model, op_types_to_calibrate=[], augmented_model_path='augmented_model.onnx'):
        '''
        :param model: ONNX model to calibrate. It can be a ModelProto or a model path
        :param op_types_to_calibrate: operator types to calibrate. By default, calibrate all the float32/float16 tensors.
        :param augmented_model_path: save augmented model to this path.
        '''
        if isinstance(model, string_types):
            self.model = onnx.load(model)
        elif isinstance(model, ModelProto):
            self.model = model
        else:
            raise ValueError('model should be either model path or onnx.ModelProto.')

        self.op_types_to_calibrate = op_types_to_calibrate
        self.augmented_model_path = augmented_model_path

        # augment graph
        self.augment_model = None
        self.augment_graph()

        # Create InferenceSession
        self.infer_session = None
        self.execution_providers = ['CPUExecutionProvider']
        self._create_inference_session()

    def set_execution_providers(self, execution_providers=['CPUExecutionProvider']):
        '''
        reset the execution providers to execute the collect_data. It triggers to re-creating inference session.
        '''
        self.execution_providers = execution_providers
        self._create_inference_session()

    def _create_inference_session(self):
        '''
        create an OnnxRuntime InferenceSession.
        '''
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        self.infer_session = onnxruntime.InferenceSession(self.augmented_model_path,
                                                          sess_options=sess_options,
                                                          providers=self.execution_providers)

    def get_augment_model(self):
        '''
        return: augmented onnx model
        '''
        return self.augment_model

    def augment_graph(self):
        '''
        abstract method: augment the input model to prepare for collecting data. It will:
            1. save augmented model to augmented_model_path.
            2. set the self.augment_model
        '''
        raise NotImplementedError

    def collect_data(self, data_reader: CalibrationDataReader):
        '''
        abstract method: collect the tensors that will be used for range computation. It can be called multiple times.
        '''
        raise NotImplementedError

    def compute_range(self, data_reader: CalibrationDataReader):
        '''
        abstract method: compute the [min, max] range for the tensors to calibrate based on the collected data.
        '''
        raise NotImplementedError


class MinMaxCalibrater(CalibraterBase):
    def __init__(self, model, op_types_to_calibrate=[], augmented_model_path='augmented_model.onnx'):
        '''
        :param model: ONNX model to calibrate. It can be a ModelProto or a model path
        :param op_types_to_calibrate: operator types to calibrate. By default, calibrate all the float32/float16 tensors.
        :param augmented_model_path: save augmented model to this path.
        '''
        super(MinMaxCalibrater, self).__init__(model, op_types_to_calibrate, augmented_model_path)
        self.intermediate_outputs = []
        self.calibrate_tensors_range = None
        self.num_model_outputs = len(self.model.graph.output)
        self.model_original_outputs = set(output.name for output in self.model.graph.output)

    def augment_graph(self):
        '''
        Adds ReduceMin and ReduceMax nodes to all quantization_candidates op type nodes in
        model and ensures their outputs are stored as part of the graph output
        :return: augmented ONNX model
        '''
        model = onnx_proto.ModelProto()
        model.CopyFrom(self.model)
        model = onnx.shape_inference.infer_shapes(model)
        value_infos = {vi.name: vi for vi in model.graph.value_info}
        value_infos.update({ot.name: ot for ot in model.graph.output})
        value_infos.update({it.name: it for it in model.graph.input})
        initializer = set(init.name for init in model.graph.initializer)

        added_nodes = []
        added_outputs = []
        tensors_to_calibrate = set()
        tensor_type_to_calibrate = set([TensorProto.FLOAT, TensorProto.FLOAT16])

        for node in model.graph.node:
            if len(self.op_types_to_calibrate) == 0 or node.op_type in self.op_types_to_calibrate:
                for tensor_name in itertools.chain(node.input, node.output):
                    if tensor_name in value_infos.keys():
                        vi = value_infos[tensor_name]
                        if vi.type.HasField('tensor_type') and (
                                vi.type.tensor_type.elem_type in tensor_type_to_calibrate) and (
                                    tensor_name not in initializer):
                            tensors_to_calibrate.add(tensor_name)

        for tensor in tensors_to_calibrate:
            # Adding ReduceMin nodes
            reduce_min_name = tensor + '_ReduceMin'
            reduce_min_node = onnx.helper.make_node('ReduceMin', [tensor], [tensor + '_ReduceMin'],
                                                    reduce_min_name,
                                                    keepdims=0)

            added_nodes.append(reduce_min_node)
            added_outputs.append(helper.make_tensor_value_info(reduce_min_node.output[0], TensorProto.FLOAT, ()))

            # Adding ReduceMax nodes
            reduce_max_name = tensor + '_ReduceMax'
            reduce_max_node = onnx.helper.make_node('ReduceMax', [tensor], [tensor + '_ReduceMax'],
                                                    reduce_max_name,
                                                    keepdims=0)

            added_nodes.append(reduce_max_node)
            added_outputs.append(helper.make_tensor_value_info(reduce_max_node.output[0], TensorProto.FLOAT, ()))

        model.graph.node.extend(added_nodes)
        model.graph.output.extend(added_outputs)
        onnx.save(model, self.augmented_model_path)
        self.augment_model = model

    def clear_collected_data(self):
        self.intermediate_outputs = []

    def collect_data(self, data_reader: CalibrationDataReader):
        while True:
            inputs = data_reader.get_next()
            if not inputs:
                break
            self.intermediate_outputs.append(self.infer_session.run(None, inputs))

    def compute_range(self):
        ''' 
        Compute the min-max range of tensor
        :return: dictionary mapping: {added node names: (ReduceMin, ReduceMax) pairs }
        '''

        if len(self.intermediate_outputs) == 0:
            raise ValueError("No data is collected.")

        output_names = [self.infer_session.get_outputs()[i].name for i in range(len(self.intermediate_outputs[0]))]
        output_dicts_list = [
            dict(zip(output_names, intermediate_output)) for intermediate_output in self.intermediate_outputs
        ]

        merged_output_dict = {}
        for d in output_dicts_list:
            for k, v in d.items():
                merged_output_dict.setdefault(k, []).append(v)
        added_output_names = output_names[self.num_model_outputs:]
        calibrate_tensor_names = [
            added_output_names[i].rpartition('_')[0] for i in range(0, len(added_output_names), 2)
        ]  #output names

        merged_added_output_dict = dict(
            (i, merged_output_dict[i]) for i in merged_output_dict if i not in self.model_original_outputs)

        pairs = []
        for i in range(0, len(added_output_names), 2):
            min_value = 0
            max_value = 0
            min_value_array = min(merged_added_output_dict[added_output_names[i]])
            max_value_array = max(merged_added_output_dict[added_output_names[i + 1]])
            if type(min_value_array) == int or min_value_array.size > 0:
                min_value = float(min_value_array)
            if type(max_value_array) == int or max_value_array.size > 0:
                max_value = float(max_value_array)

            pairs.append(tuple([min_value, max_value]))

        self.calibrate_tensors_range = dict(zip(calibrate_tensor_names, pairs))

        return self.calibrate_tensors_range


def create_calibrator(model,
                      op_types_to_calibrate=[],
                      augmented_model_path='augmented_model.onnx',
                      calibrate_method=CalibrationMethod.MinMax):
    if calibrate_method == CalibrationMethod.MinMax:
        return MinMaxCalibrater(model, op_types_to_calibrate, augmented_model_path)

    raise ValueError('Unsupported calibration method {}'.format(calibrate_method))
