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

from .quant_utils import QuantType
from .registry import QLinearOpsRegistry

import abc
import itertools
import copy

def smooth_distribution(p, eps=0.0001):
    """Given a discrete distribution (may have not been normalized to 1),
    smooth it by replacing zeros with eps multiplied by a scaling factor
    and taking the corresponding amount off the non-zero values.
    Ref: http://web.engr.illinois.edu/~hanj/cs412/bk3/KL-divergence.pdf
         https://github.com//apache/incubator-mxnet/blob/master/python/mxnet/contrib/quantization.py
    """
    is_zeros = (p == 0).astype(np.float32)
    is_nonzeros = (p != 0).astype(np.float32)
    n_zeros = is_zeros.sum()
    n_nonzeros = p.size - n_zeros

    if not n_nonzeros:
        # raise ValueError('The discrete probability distribution is malformed. All entries are 0.')
        return -1
    eps1 = eps * float(n_zeros) / float(n_nonzeros)
    assert eps1 < 1.0, 'n_zeros=%d, n_nonzeros=%d, eps1=%f' % (n_zeros, n_nonzeros, eps1)

    hist = p.astype(np.float32)
    hist += eps * is_zeros + (-eps1) * is_nonzeros
    assert (hist <= 0).sum() == 0

    return hist

class CalibrationDataCollector(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def collect(self, name_to_arr):
        """
        Get histogram based on given data.
            name_to_arr : dict 
                tensor name to data as NDArray 
        """

class HistogramCollector(CalibrationDataCollector):
    def __init__(self, num_quantized_bins=128):
        self.histogram_dict = {}
        self.num_quantized_bins= num_quantized_bins

    def get_histogram_dict(self):
        return self.histogram_dict

    def collect(self, name_to_arr):
        for tensor, data_arr in name_to_arr.items():
            data_arr = np.asarray(data_arr)
            data_arr = data_arr.flatten()

            if data_arr.size > 0:
                min_value = np.min(data_arr)
                max_value = np.max(data_arr)
            else:
                min_value = 0
                max_value = 0

            threshold = max(abs(min_value), abs(max_value))

            if tensor in self.histogram_dict:
                old_histogram = self.histogram_dict[tensor]
                self.histogram_dict[tensor] = self.merge_histogram(old_histogram, data_arr, min_value, max_value, threshold)
            else:
                # hist, hist_edges = np.histogram(data_arr, self.num_quantized_bins, range=(min_value, max_value))
                hist, hist_edges = np.histogram(data_arr, self.num_quantized_bins, range=(-threshold, threshold))
                self.histogram_dict[tensor] = (hist, hist_edges, min_value, max_value, threshold)

    def merge_histogram(self, old_histogram, data_arr, new_min, new_max, new_threshold):

        (old_hist, old_hist_edges, old_min, old_max, old_threshold) = old_histogram

        if new_threshold <= old_threshold:
            new_hist, _ = np.histogram(data_arr, len(old_hist), range=(-old_threshold, old_threshold))
            return (new_hist + old_hist, old_hist_edges, min(old_min, new_min), max(old_max, new_max), old_threshold)
        else:
            old_num_bins = len(old_hist)
            old_stride = 2 * old_threshold / old_num_bins
            half_increased_bins = 0 if old_stride == 0 else int((new_threshold - old_threshold) // old_stride + 1) 
            new_num_bins = old_num_bins + 2 * half_increased_bins
            new_threshold = half_increased_bins * old_stride + old_threshold
            hist, hist_edges = np.histogram(data_arr, new_num_bins, range=(-new_threshold, new_threshold))
            hist[half_increased_bins:new_num_bins-half_increased_bins] += old_hist
            return (hist, hist_edges, min(old_min, new_min), max(old_max, new_max), new_threshold)

    def get_optimal_collection_result(self):
        return self.get_optimal_thresholds(self.histogram_dict, self.num_quantized_bins)


    def get_optimal_thresholds(self, histogram_dict, num_quantized_bins=128):
        thresholds_dict = {} # per tensor thresholds

        for tensor, histogram in histogram_dict.items():
            optimal_threshold = self.get_optimal_threshold(histogram, num_quantized_bins)
            thresholds_dict[tensor] = optimal_threshold

        return thresholds_dict

    def get_optimal_threshold(self, histogram, num_quantized_bins):
        from scipy.stats import entropy

        hist, hist_edges, _, _, _ = histogram
        num_bins = hist.size
        zero_bin_index = num_bins // 2
        num_half_quantized_bin = num_quantized_bins // 2
        
        kl_divergence = np.zeros(zero_bin_index - num_half_quantized_bin + 1)
        thresholds = [(0, 0) for i in range(kl_divergence.size)] 

        for i in range(num_half_quantized_bin, zero_bin_index + 1, 1):
            start_index = zero_bin_index - i 
            end_index = zero_bin_index + i + 1 if (zero_bin_index + i + 1) <= num_bins else num_bins

            thresholds[i - num_half_quantized_bin] = (float(hist_edges[start_index]), float(hist_edges[end_index]))

            sliced_distribution = copy.deepcopy(hist[start_index:end_index])

            # reference distribution p
            p = sliced_distribution.copy() # a copy of np array
            left_outliers_count = sum(hist[:start_index]) 
            right_outliers_count = sum(hist[end_index:])
            p[0] += left_outliers_count
            p[-1] += right_outliers_count

            # nonzeros[i] incidates whether p[i] is non-zero
            nonzeros = (p != 0).astype(np.int64)
            
            # quantize p.size bins into quantized bins (default 128 bins) 
            quantized_bins = np.zeros(num_quantized_bins, dtype=np.int64)
            num_merged_bins = sliced_distribution.size // num_quantized_bins

            # merge bins into quantized bins
            for index in range(num_quantized_bins):
                start = index * num_merged_bins 
                end = start + num_merged_bins
                quantized_bins[index] = sum(sliced_distribution[start:end]) 
                # print("start %s, end %s, sum %s" % (start, end, sum(sliced_distribution[start:end])))
            quantized_bins[-1] += sum(sliced_distribution[num_quantized_bins * num_merged_bins:])

            # in order to compare p and q, we need to make length of q equals to length of p
            # expand quantized bins into p.size bins
            q = np.zeros(p.size, dtype=np.int64)
            for index in range(num_quantized_bins):
                start = index * num_merged_bins
                end = start + num_merged_bins

                norm = sum(nonzeros[start:end])
                if norm != 0:
                    q[start:end] = float(quantized_bins[index]) / float(norm)
            
            p = smooth_distribution(p)
            q = smooth_distribution(q)

            if isinstance(q, np.ndarray):
                kl_divergence[i - num_half_quantized_bin] = entropy(p, q)
            else:
                kl_divergence[i - num_half_quantized_bin] = float('inf')

        min_kl_divergence_idx = np.argmin(kl_divergence)
        optimal_threshold = thresholds[min_kl_divergence_idx] 

        return optimal_threshold

class CalibrationDataReader(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'get_next') and callable(subclass.get_next) or NotImplemented)

    @abc.abstractmethod
    def get_next(self) -> dict:
        """generate the input data dict for ONNXinferenceSession run"""
        raise NotImplementedError


class ONNXCalibrater:
    def __init__(self, model, data_reader: CalibrationDataReader, calibrate_op_types, black_nodes, white_nodes,
                 augmented_model_path):
        '''
        :param model: ONNX model to calibrate. It can be a ModelProto or a model path
        :param data_reader: user implemented object to read in and preprocess calibration dataset
                            based on CalibrationDataReader Interface
        :param op_types: operator types to be calibrated and quantized, default = 'Conv,MatMul'
        :param black_nodes: operator names that should not be quantized, default = ''
        :param white_nodes: operator names that force to be quantized, default = ''
        :param augmented_model_path: save augmented_model to this path
        '''
        if isinstance(model, string_types):
            self.model = onnx.load(model)
        elif isinstance(model, ModelProto):
            self.model = model
        else:
            raise ValueError('model should be either model path or onnx.ModelProto.')

        self.data_reader = data_reader
        self.calibrate_op_types = calibrate_op_types
        self.black_nodes = black_nodes
        self.white_nodes = white_nodes
        self.augmented_model_path = augmented_model_path
        self.input_name_to_nodes = {}
        self.calibration_cache = {}  # save temporary calibration table
        self.collector = None

    def set_data_reader(self, data_reader):
        self.data_reader = data_reader

    def get_calibration_cache(self):
        if len(self.calibration_cache) > 0:
            return self.calibration_cache

        if not self.collector:
            print("No collector created and can't generate calibration data.")
            return None

        self.calibration_cache = self.collector.get_optimal_collection_result()

        return self.calibration_cache

    def augment_graph(self, calib_mode='naive'):
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
            should_be_calibrate = ((node.op_type in self.calibrate_op_types) and
                                       (node.name not in self.black_nodes)) or (node.name in self.white_nodes) or ((not self.calibrate_op_types) and (node.name not in self.black_nodes))
            if should_be_calibrate:
                for tensor_name in itertools.chain(node.input, node.output):
                    if tensor_name in value_infos.keys():
                        vi = value_infos[tensor_name]
                        if vi.type.HasField('tensor_type') and (vi.type.tensor_type.elem_type in tensor_type_to_calibrate) and (tensor_name not in initializer):
                            tensors_to_calibrate.add(tensor_name)


        for tensor in tensors_to_calibrate:
            if calib_mode == 'naive':

                # If augmenting all ops, it's possible that some nodes' input value are 0.
                # Can't reduce on dim with value of 0 if 'keepdims' is false, therefore set keepdims to 1.
                if self.calibrate_op_types:
                    keepdims_value = 0
                else:
                    keepdims_value = 1

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

            elif calib_mode == 'entropy':
                added_outputs.append(value_infos[tensor])

        model.graph.node.extend(added_nodes)
        model.graph.output.extend(added_outputs)

        return model

    #Using augmented outputs to generate inputs for quantization
    def get_intermediate_outputs(self, calib_mode='naive', providers=None, ort_graph_optimization_enable=True):
        ''' 
        Gather intermediate model outputs after running inference
        parameter calib_mode: type 'naive' gives (ReduceMin, ReduceMax) pairs
                              for each augmented node across test data sets, where
                              the first element is a minimum of all ReduceMin values
                              and the second element is a maximum of all ReduceMax
                              values;
        parameter providers: Onnxruntime execution providers
        parameter ort_graph_optimization_enable: Enable all OnnxRuntime graph optimizations, default = True
        :return: dictionary mapping: {added node names: (ReduceMin, ReduceMax) pairs }
        '''

        print(calib_mode)
        #conduct inference session and get intermediate outputs
        if ort_graph_optimization_enable:
            session = onnxruntime.InferenceSession(self.augmented_model_path, None) 
        else:            
            sess_options = onnxruntime.SessionOptions()
            sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL  #ORT_ENABLE_BASIC
            session = onnxruntime.InferenceSession(self.augmented_model_path,
                                                   sess_options=sess_options,
                                                   providers=providers)

        #number of outputs in original model
        num_model_outputs = len(self.model.graph.output)
        model_original_outputs = set(output.name for output in self.model.graph.output)

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

        # Characterizing distribution of a node's values across test data sets
        clean_merged_dict = dict((i, merged_dict[i]) for i in merged_dict if i not in model_original_outputs)

        if calib_mode == 'naive':

            node_names = [added_node_output_names[i].rpartition('_')[0]
                          for i in range(0, len(added_node_output_names), 2)]  #output names

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

            final_dict = dict(zip(node_names, pairs))

            # merge new calibration data with previous calibration data
            if len(self.calibration_cache) > 0:
                for key, value in self.calibration_cache.items():
                    min_value = min(value[0], final_dict[key][0])
                    max_value = max(value[1], final_dict[key][1])
                    final_dict[key] = (min_value, max_value)

        elif calib_mode == 'entropy':
            if not self.collector:
                self.collector = HistogramCollector()
            self.collector.collect(clean_merged_dict)

            return

        else:
            raise ValueError('Unknown value for calib_mode. Currently only naive mode is supported.')


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
                # attribute min and max:
                if (2 == len(next_node.attribute)):
                    for att_idx in [0, 1]:
                        if next_node.attribute[att_idx].name == 'min':
                            rmin = max(rmin, next_node.attribute[att_idx].f)
                        elif next_node.attribute[att_idx].name == 'max':
                            rmax = min(rmax, next_node.attribute[att_idx].f)

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

        self._get_input_name_to_nodes(self.model)

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


def get_calibrator(model,
                   data_reader: CalibrationDataReader,
                   op_types=[],
                   black_nodes=[],
                   white_nodes=[],
                   augmented_model_path='augmented_model.onnx'):

    calibrator = ONNXCalibrater(model, data_reader, op_types, black_nodes, white_nodes, augmented_model_path)

    return calibrator


def calculate_calibration_data(model,
                               calibrator=None,
                               calib_mode='entropy',
                               calibration_data_reader: CalibrationDataReader = None,
                               op_types_to_quantize=[],
                               activation_type=QuantType.QUInt8,
                               nodes_to_quantize=[],
                               nodes_to_exclude=[],
                               augmented_model_path='augmented_model.onnx'):

    if activation_type != QuantType.QUInt8:
        raise ValueError("Static quantization only support uint8 for activation now.")

    if not op_types_to_quantize or len(op_types_to_quantize) == 0:
        op_types_to_quantize = list(QLinearOpsRegistry.keys())

    print("augmented model path: %s" % augmented_model_path)

    if not calibrator:
        calibrator = get_calibrator(model,
                                    calibration_data_reader,
                                    op_types_to_quantize,
                                    nodes_to_exclude,                                    
                                    nodes_to_quantize,
                                    augmented_model_path=augmented_model_path)

    if not os.path.exists(augmented_model_path):
        augmented_model = calibrator.augment_graph(calib_mode=calib_mode)
        onnx.save(augmented_model, augmented_model_path)

    calibrator.get_intermediate_outputs(calib_mode=calib_mode, providers=["CUDAExecutionProvider"], ort_graph_optimization_enable=False)


def generate_calibration_table(calibrator,
                               model,
                               augmented_model_path,
                               remove_previous_flag,
                               data_reader,
                               calib_mode='entropy'):

    if remove_previous_flag and os.path.exists(augmented_model_path):
        os.remove(augmented_model_path)
        print("remove previously generated %s and start to generate a new one." % (augmented_model_path))

    if not calibrator:
        calibrator = get_calibrator(model, data_reader, augmented_model_path=augmented_model_path)
    calculate_calibration_data(model, calibrator, calib_mode, augmented_model_path=augmented_model_path)


def calibrate(model,
              data_reader: CalibrationDataReader,
              op_types=['Conv', 'MatMul'],
              black_nodes=[],
              white_nodes=[],
              augmented_model_path='augmented_model.onnx',
              providers=["CPUExecutionProvider"],
              ort_graph_optimization_enable=True,
              quantization_params_calculation_enable=True):
    '''
    Given an onnx model, augment and run the augmented model on calibration data set, aggregate and calculate the quantization parameters.
    :param model: ONNX model to calibrate. It can be a ModelProto or a model path
    :param data_reader: user implemented object to read in and preprocess calibration dataset based on CalibrationDataReader interface
    :param op_types: operator types to be calibrated and quantized, default = 'Conv,MatMul'. Empty means to quantize all FP32 tensors (except black_nodes)
    :param black_nodes: operator names that should not be quantized, default = ''
    :param white_nodes: operator names that force to be quantized, default = ''
    :param augmented_model_path: save augmented_model to this path
    :param providers: execution providers to run calibration
    :param ort_graph_optimization_enable: enable all OnnxRuntime graph optimizations, default = True
    :param quantization_params_calculation_enable: enable quantization parameter calculation, default = True 
    '''
    #1. initialize a calibrater
    calibrater = ONNXCalibrater(model, data_reader, op_types, black_nodes, white_nodes, augmented_model_path)
    #2. augment
    augmented_model = calibrater.augment_graph()
    onnx.save(augmented_model, augmented_model_path)
    #3. generate quantization thresholds
    dict_for_quantization = calibrater.get_intermediate_outputs(providers=providers, ort_graph_optimization_enable=ort_graph_optimization_enable)
    #4. generate quantization parameters dict
    quantization_params_dict = {}    
    if quantization_params_calculation_enable:
        quantization_params_dict = calibrater.calculate_quantization_params(dict_for_quantization)
    print("Calibrated,quantized parameters calculated and returned.")

    return quantization_params_dict if quantization_params_calculation_enable else dict_for_quantization
