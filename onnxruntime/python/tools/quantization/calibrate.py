#!/usr/bin/env python
# coding: utf-8
# -------------------------------------------------------------------------
# Copyright (c) Microsoft, Intel Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import abc
import itertools
import uuid
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import onnx
from onnx import ModelProto, TensorProto, helper, numpy_helper

import onnxruntime

from .quant_utils import apply_plot, clone_model_with_shape_infer, load_model, smooth_distribution


class CalibrationMethod(Enum):
    MinMax = 0
    Entropy = 1
    Percentile = 2


class CalibrationDataReader(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, "get_next") and callable(subclass.get_next) or NotImplemented

    @abc.abstractmethod
    def get_next(self) -> dict:
        """generate the input data dict for ONNXinferenceSession run"""
        raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self):
        result = self.get_next()
        if result is None:
            raise StopIteration
        return result


class CalibraterBase:
    def __init__(
        self,
        model,
        op_types_to_calibrate: Optional[Sequence[str]] = None,
        augmented_model_path="augmented_model.onnx",
        symmetric=False,
        use_external_data_format=False,
    ):
        """
        :param model: ONNX model to calibrate. It can be a ModelProto or a model path
        :param op_types_to_calibrate: operator types to calibrate. By default, calibrate all the float32/float16 tensors.
        :param augmented_model_path: save augmented model to this path.
        :param symmetric: make range of tensor symmetric (central point is 0).
        :param use_external_data_format: use external data format to store model which size is >= 2Gb
        """
        if isinstance(model, str):
            self.model = load_model(Path(model), False)
        elif isinstance(model, Path):
            self.model = load_model(model, False)
        elif isinstance(model, ModelProto):
            self.model = model
        else:
            raise ValueError("model should be either model path or onnx.ModelProto.")

        self.op_types_to_calibrate = op_types_to_calibrate
        self.augmented_model_path = augmented_model_path
        self.symmetric = symmetric
        self.use_external_data_format = use_external_data_format

        self.augment_model = None
        self.infer_session = None
        self.execution_providers = ["CPUExecutionProvider"]

    def set_execution_providers(self, execution_providers=["CPUExecutionProvider"]):
        """
        reset the execution providers to execute the collect_data. It triggers to re-creating inference session.
        """
        self.execution_providers = execution_providers
        self.create_inference_session()

    def create_inference_session(self):
        """
        create an OnnxRuntime InferenceSession.
        """
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        self.infer_session = onnxruntime.InferenceSession(
            self.augmented_model_path,
            sess_options=sess_options,
            providers=self.execution_providers,
        )

    def select_tensors_to_calibrate(self, model):
        """
        select all quantization_candidates op type nodes' input/output tensors.
        returns:
            tensors (set): set of tensor name.
            value_infos (dict): tensor name to value info.
        """
        value_infos = {vi.name: vi for vi in model.graph.value_info}
        value_infos.update({ot.name: ot for ot in model.graph.output})
        value_infos.update({it.name: it for it in model.graph.input})
        initializer = set(init.name for init in model.graph.initializer)

        tensors_to_calibrate = set()
        tensor_type_to_calibrate = set([TensorProto.FLOAT, TensorProto.FLOAT16])

        for node in model.graph.node:
            if not self.op_types_to_calibrate or node.op_type in self.op_types_to_calibrate:
                for tensor_name in itertools.chain(node.input, node.output):
                    if tensor_name in value_infos.keys():
                        vi = value_infos[tensor_name]
                        if (
                            vi.type.HasField("tensor_type")
                            and (vi.type.tensor_type.elem_type in tensor_type_to_calibrate)
                            and (tensor_name not in initializer)
                        ):
                            tensors_to_calibrate.add(tensor_name)

        return tensors_to_calibrate, value_infos

    def get_augment_model(self):
        """
        return: augmented onnx model
        """
        return self.augment_model

    def augment_graph(self):
        """
        abstract method: augment the input model to prepare for collecting data. It will:
            1. save augmented model to augmented_model_path.
            2. set the self.augment_model
        """
        raise NotImplementedError

    def collect_data(self, data_reader: CalibrationDataReader):
        """
        abstract method: collect the tensors that will be used for range computation. It can be called multiple times.
        """
        raise NotImplementedError

    def compute_range(self, data_reader: CalibrationDataReader):
        """
        abstract method: compute the [min, max] range for the tensors to calibrate based on the collected data.
        """
        raise NotImplementedError


class MinMaxCalibrater(CalibraterBase):
    def __init__(
        self,
        model,
        op_types_to_calibrate: Optional[Sequence[str]] = None,
        augmented_model_path="augmented_model.onnx",
        symmetric=False,
        use_external_data_format=False,
        moving_average=False,
        averaging_constant=0.01,
    ):
        """
        :param model: ONNX model to calibrate. It can be a ModelProto or a model path
        :param op_types_to_calibrate: operator types to calibrate. By default, calibrate all the float32/float16 tensors.
        :param augmented_model_path: save augmented model to this path.
        :param symmetric: make range of tensor symmetric (central point is 0).
        :param use_external_data_format: use external data format to store model which size is >= 2Gb
        :param moving_average: compute the moving average of the minimum and maximum values instead of the global minimum and maximum.
        :param averaging_constant: constant smoothing factor to use when computing the moving average.
        """
        super(MinMaxCalibrater, self).__init__(
            model,
            op_types_to_calibrate=op_types_to_calibrate,
            augmented_model_path=augmented_model_path,
            symmetric=symmetric,
            use_external_data_format=use_external_data_format,
        )
        self.intermediate_outputs = []
        self.calibrate_tensors_range = None
        self.num_model_outputs = len(self.model.graph.output)
        self.model_original_outputs = set(output.name for output in self.model.graph.output)
        self.moving_average = moving_average
        if moving_average and (averaging_constant < 0 or averaging_constant > 1):
            raise ValueError("Invalid averaging constant, which should not be < 0 or > 1.")
        self.averaging_constant = averaging_constant

    def augment_graph(self):
        """
        Adds ReduceMin and ReduceMax nodes to all quantization_candidates op type nodes in
        model and ensures their outputs are stored as part of the graph output
        :return: augmented ONNX model
        """
        model = clone_model_with_shape_infer(self.model)

        tensors, _ = self.select_tensors_to_calibrate(model)
        reshape_shape_name = str(uuid.uuid4())
        reshape_shape = numpy_helper.from_array(np.array([1], dtype=np.int64), reshape_shape_name)
        model.graph.initializer.append(reshape_shape)

        def add_reduce_min_max(tensor_name, reduce_op_name):
            # When doing ReduceMax/ReduceMin, ORT can't reduce on dim with value of 0 if 'keepdims' is false.
            # To make the code simple, we always let keepdims to be 1.
            keepdims = 1

            # Adding ReduceMin/ReduceMax nodes: ReduceMin/ReduceMax -> Reshape-> (output)
            reduce_output = tensor_name + "_" + reduce_op_name
            intermediate_output = reduce_output + "_Reshape"
            reduce_node = onnx.helper.make_node(
                reduce_op_name, [tensor_name], [intermediate_output], keepdims=keepdims, name=reduce_output
            )

            reshape_node = onnx.helper.make_node(
                "Reshape",
                inputs=[intermediate_output, reshape_shape_name],
                outputs=[reduce_output],
                name=intermediate_output,
            )

            model.graph.node.extend([reduce_node, reshape_node])
            model.graph.output.append(helper.make_tensor_value_info(reduce_output, TensorProto.FLOAT, [1]))

        for tensor in tensors:
            add_reduce_min_max(tensor, "ReduceMin")
            add_reduce_min_max(tensor, "ReduceMax")

        onnx.save(
            model,
            self.augmented_model_path,
            save_as_external_data=self.use_external_data_format,
        )
        self.augment_model = model

    def clear_collected_data(self):
        self.intermediate_outputs = []

    def collect_data(self, data_reader: CalibrationDataReader):
        while True:
            inputs = data_reader.get_next()
            if not inputs:
                break
            self.intermediate_outputs.append(self.infer_session.run(None, inputs))

        if len(self.intermediate_outputs) == 0:
            raise ValueError("No data is collected.")

        self.compute_range()
        self.clear_collected_data()

    def merge_range(self, old_range, new_range):
        if not old_range:
            return new_range

        for key, value in old_range.items():
            if self.moving_average:
                min_value = value[0] + self.averaging_constant * (new_range[key][0] - value[0])
                max_value = value[1] + self.averaging_constant * (new_range[key][1] - value[1])
            else:
                min_value = min(value[0], new_range[key][0])
                max_value = max(value[1], new_range[key][1])
            new_range[key] = (min_value, max_value)

        return new_range

    def compute_range(self):
        """
        Compute the min-max range of tensor
        :return: dictionary mapping: {added node names: (ReduceMin, ReduceMax) pairs }
        """

        if len(self.intermediate_outputs) == 0:
            return self.calibrate_tensors_range

        output_names = [self.infer_session.get_outputs()[i].name for i in range(len(self.intermediate_outputs[0]))]
        output_dicts_list = [
            dict(zip(output_names, intermediate_output)) for intermediate_output in self.intermediate_outputs
        ]

        merged_output_dict = {}
        for d in output_dicts_list:
            for k, v in d.items():
                merged_output_dict.setdefault(k, []).append(v)
        added_output_names = output_names[self.num_model_outputs :]
        calibrate_tensor_names = [
            added_output_names[i].rpartition("_")[0] for i in range(0, len(added_output_names), 2)
        ]  # output names

        merged_added_output_dict = dict(
            (i, merged_output_dict[i]) for i in merged_output_dict if i not in self.model_original_outputs
        )

        pairs = []
        for i in range(0, len(added_output_names), 2):
            min_value = 0
            max_value = 0
            if self.moving_average:
                min_value_array = np.mean(merged_added_output_dict[added_output_names[i]], axis=0)
                max_value_array = np.mean(merged_added_output_dict[added_output_names[i + 1]], axis=0)
            else:
                min_value_array = min(merged_added_output_dict[added_output_names[i]])
                max_value_array = max(merged_added_output_dict[added_output_names[i + 1]])
            if type(min_value_array) == int or min_value_array.size > 0:
                min_value = float(min_value_array)
            if type(max_value_array) == int or max_value_array.size > 0:
                max_value = float(max_value_array)

            if self.symmetric:
                max_absolute_value = max(abs(min_value), abs(max_value))
                pairs.append(tuple([-max_absolute_value, max_absolute_value]))
            else:
                pairs.append(tuple([min_value, max_value]))

        new_calibrate_tensors_range = dict(zip(calibrate_tensor_names, pairs))
        if self.calibrate_tensors_range:
            self.calibrate_tensors_range = self.merge_range(self.calibrate_tensors_range, new_calibrate_tensors_range)
        else:
            self.calibrate_tensors_range = new_calibrate_tensors_range

        return self.calibrate_tensors_range


class HistogramCalibrater(CalibraterBase):
    def __init__(
        self,
        model,
        op_types_to_calibrate: Optional[Sequence[str]] = None,
        augmented_model_path="augmented_model.onnx",
        use_external_data_format=False,
        method="percentile",
        symmetric=False,
        num_bins=128,
        num_quantized_bins=2048,
        percentile=99.999,
    ):
        """
        :param model: ONNX model to calibrate. It can be a ModelProto or a model path
        :param op_types_to_calibrate: operator types to calibrate. By default, calibrate all the float32/float16 tensors.
        :param augmented_model_path: save augmented model to this path.
        :param use_external_data_format: use external data format to store model which size is >= 2Gb
        :param method: A string. One of ['entropy', 'percentile'].
        :param symmetric: make range of tensor symmetric (central point is 0).
        :param num_bins: number of bins to create a new histogram for collecting tensor values.
        :param num_quantized_bins: number of quantized bins. Default 128.
        :param percentile: A float number between [0, 100]. Default 99.99.
        """
        super(HistogramCalibrater, self).__init__(
            model,
            op_types_to_calibrate=op_types_to_calibrate,
            augmented_model_path=augmented_model_path,
            symmetric=symmetric,
            use_external_data_format=use_external_data_format,
        )
        self.intermediate_outputs = []
        self.calibrate_tensors_range = None
        self.num_model_outputs = len(self.model.graph.output)
        self.model_original_outputs = set(output.name for output in self.model.graph.output)
        self.collector = None
        self.method = method
        self.num_bins = num_bins
        self.num_quantized_bins = num_quantized_bins
        self.percentile = percentile
        self.tensors_to_calibrate = None

    def augment_graph(self):
        """
        make all quantization_candidates op type nodes as part of the graph output.
        :return: augmented ONNX model
        """
        model = clone_model_with_shape_infer(self.model)

        self.tensors_to_calibrate, value_infos = self.select_tensors_to_calibrate(model)
        for tensor in self.tensors_to_calibrate:
            if tensor not in self.model_original_outputs:
                model.graph.output.append(value_infos[tensor])

        onnx.save(
            model,
            self.augmented_model_path,
            save_as_external_data=self.use_external_data_format,
        )
        self.augment_model = model

    def clear_collected_data(self):
        self.intermediate_outputs = []

    def collect_data(self, data_reader: CalibrationDataReader):
        """
        Entropy Calibrator collects operators' tensors as well as generates tensor histogram for each operator.
        """
        while True:
            inputs = data_reader.get_next()
            if not inputs:
                break
            self.intermediate_outputs.append(self.infer_session.run(None, inputs))

        if len(self.intermediate_outputs) == 0:
            raise ValueError("No data is collected.")

        output_names = [self.infer_session.get_outputs()[i].name for i in range(len(self.intermediate_outputs[0]))]
        output_dicts_list = [
            dict(zip(output_names, intermediate_output)) for intermediate_output in self.intermediate_outputs
        ]

        merged_dict = {}
        for d in output_dicts_list:
            for k, v in d.items():
                merged_dict.setdefault(k, []).append(v)

        clean_merged_dict = dict((i, merged_dict[i]) for i in merged_dict if i in self.tensors_to_calibrate)

        if not self.collector:
            self.collector = HistogramCollector(
                method=self.method,
                symmetric=self.symmetric,
                num_bins=self.num_bins,
                num_quantized_bins=self.num_quantized_bins,
                percentile=self.percentile,
            )
        self.collector.collect(clean_merged_dict)

        self.clear_collected_data()

    def compute_range(self):
        """
        Compute the min-max range of tensor
        :return: dictionary mapping: {tensor name: (min value, max value)}
        """
        if not self.collector:
            raise ValueError("No collector created and can't generate calibration data.")

        return self.collector.compute_collection_result()


class EntropyCalibrater(HistogramCalibrater):
    def __init__(
        self,
        model,
        op_types_to_calibrate: Optional[Sequence[str]] = None,
        augmented_model_path="augmented_model.onnx",
        use_external_data_format=False,
        method="entropy",
        symmetric=False,
        num_bins=128,
        num_quantized_bins=128,
    ):
        """
        :param model: ONNX model to calibrate. It can be a ModelProto or a model path
        :param op_types_to_calibrate: operator types to calibrate. By default, calibrate all the float32/float16 tensors.
        :param augmented_model_path: save augmented model to this path.
        :param use_external_data_format: use external data format to store model which size is >= 2Gb
        :param method: A string. One of ['entropy', 'percentile'].
        :param symmetric: make range of tensor symmetric (central point is 0).
        :param num_bins: number of bins to create a new histogram for collecting tensor values.
        :param num_quantized_bins: number of quantized bins. Default 128.
        """
        super(EntropyCalibrater, self).__init__(
            model,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format,
            method=method,
            symmetric=symmetric,
            num_bins=num_bins,
            num_quantized_bins=num_quantized_bins,
        )


class PercentileCalibrater(HistogramCalibrater):
    def __init__(
        self,
        model,
        op_types_to_calibrate: Optional[Sequence[str]] = None,
        augmented_model_path="augmented_model.onnx",
        use_external_data_format=False,
        method="percentile",
        symmetric=False,
        num_bins=2048,
        percentile=99.999,
    ):
        """
        :param model: ONNX model to calibrate. It can be a ModelProto or a model path
        :param op_types_to_calibrate: operator types to calibrate. By default, calibrate all the float32/float16 tensors.
        :param augmented_model_path: save augmented model to this path.
        :param use_external_data_format: use external data format to store model which size is >= 2Gb
        :param method: A string. One of ['entropy', 'percentile'].
        :param symmetric: make range of tensor symmetric (central point is 0).
        :param num_quantized_bins: number of quantized bins. Default 128.
        :param percentile: A float number between [0, 100]. Default 99.99.
        """
        super(PercentileCalibrater, self).__init__(
            model,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format,
            method=method,
            symmetric=symmetric,
            num_bins=num_bins,
            percentile=percentile,
        )


class CalibrationDataCollector(metaclass=abc.ABCMeta):
    """
    Base class for collecting data for calibration-based quantization.
    """

    @abc.abstractmethod
    def collect(self, name_to_arr):
        """
        Generate informative data based on given data.
            name_to_arr : dict
                tensor name to NDArray data
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compute_collection_result(self):
        """
        Get the optimal result among collection data.
        """
        raise NotImplementedError


class HistogramCollector(CalibrationDataCollector):
    """
    Collecting histogram for each tensor. Percentile and Entropy method are supported.

    ref: https://github.com//apache/incubator-mxnet/blob/master/python/mxnet/contrib/quantization.py
    ref: https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/_modules/
                 pytorch_quantization/calib/histogram.html
    """

    def __init__(self, method, symmetric, num_bins, num_quantized_bins, percentile):
        self.histogram_dict = {}
        self.method = method
        self.symmetric = symmetric
        self.num_bins = num_bins
        self.num_quantized_bins = num_quantized_bins
        self.percentile = percentile

    def get_histogram_dict(self):
        return self.histogram_dict

    def collect(self, name_to_arr):
        print("Collecting tensor data and making histogram ...")

        # TODO: Currently we have different collect() for entropy and percentile method respectively.
        #       Need unified collect in the future.
        if self.method == "entropy":
            return self.collect_value(name_to_arr)
        elif self.method == "percentile":
            if self.symmetric:
                return self.collect_absolute_value(name_to_arr)
            else:
                return self.collect_value(name_to_arr)
        else:
            raise ValueError("Only 'entropy' or 'percentile' method are supported")

    def collect_absolute_value(self, name_to_arr):
        """
        Collect histogram on absolute value
        """
        for tensor, data_arr in name_to_arr.items():
            data_arr = np.asarray(data_arr)
            data_arr = data_arr.flatten()
            if data_arr.size > 0:
                min_value = np.min(data_arr)
                max_value = np.max(data_arr)
            else:
                min_value = 0
                max_value = 0

            data_arr = np.absolute(data_arr)  # only consider absolute value

            if tensor not in self.histogram_dict:
                # first time it uses num_bins to compute histogram.
                hist, hist_edges = np.histogram(data_arr, bins=self.num_bins)
                self.histogram_dict[tensor] = (hist, hist_edges, min_value, max_value)
            else:
                old_histogram = self.histogram_dict[tensor]
                old_min = old_histogram[2]
                old_max = old_histogram[3]
                old_hist = old_histogram[0]
                old_hist_edges = old_histogram[1]
                temp_amax = np.max(data_arr)
                if temp_amax > old_hist_edges[-1]:
                    # increase the number of bins
                    width = old_hist_edges[1] - old_hist_edges[0]
                    # NOTE: np.arange may create an extra bin after the one containing temp_amax
                    new_bin_edges = np.arange(old_hist_edges[-1] + width, temp_amax + width, width)
                    old_hist_edges = np.hstack((old_hist_edges, new_bin_edges))
                hist, hist_edges = np.histogram(data_arr, bins=old_hist_edges)
                hist[: len(old_hist)] += old_hist
                self.histogram_dict[tensor] = (hist, hist_edges, min(old_min, min_value), max(old_max, max_value))

    def collect_value(self, name_to_arr):
        """
        Collect histogram on real value
        """
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
                self.histogram_dict[tensor] = self.merge_histogram(
                    old_histogram, data_arr, min_value, max_value, threshold
                )
            else:
                hist, hist_edges = np.histogram(data_arr, self.num_bins, range=(-threshold, threshold))
                self.histogram_dict[tensor] = (
                    hist,
                    hist_edges,
                    min_value,
                    max_value,
                    threshold,
                )

    def merge_histogram(self, old_histogram, data_arr, new_min, new_max, new_threshold):

        (old_hist, old_hist_edges, old_min, old_max, old_threshold) = old_histogram

        if new_threshold <= old_threshold:
            new_hist, _ = np.histogram(data_arr, len(old_hist), range=(-old_threshold, old_threshold))
            return (
                new_hist + old_hist,
                old_hist_edges,
                min(old_min, new_min),
                max(old_max, new_max),
                old_threshold,
            )
        else:
            if old_threshold == 0:
                hist, hist_edges = np.histogram(data_arr, len(old_hist), range=(-new_threshold, new_threshold))
                hist += old_hist
            else:
                old_num_bins = len(old_hist)
                old_stride = 2 * old_threshold / old_num_bins
                half_increased_bins = int((new_threshold - old_threshold) // old_stride + 1)
                new_num_bins = old_num_bins + 2 * half_increased_bins
                new_threshold = half_increased_bins * old_stride + old_threshold
                hist, hist_edges = np.histogram(data_arr, new_num_bins, range=(-new_threshold, new_threshold))
                hist[half_increased_bins : new_num_bins - half_increased_bins] += old_hist
            return (
                hist,
                hist_edges,
                min(old_min, new_min),
                max(old_max, new_max),
                new_threshold,
            )

    def compute_collection_result(self):
        if not self.histogram_dict or len(self.histogram_dict) == 0:
            raise ValueError("Histogram has not been collected. Please run collect() first.")
        print("Finding optimal threshold for each tensor using {} algorithm ...".format(self.method))

        if self.method == "entropy":
            return self.compute_entropy()
        elif self.method == "percentile":
            return self.compute_percentile()
        else:
            raise ValueError("Only 'entropy' or 'percentile' method are supported")

    def compute_percentile(self):
        if self.percentile < 0 or self.percentile > 100:
            raise ValueError("Invalid percentile. Must be in range 0 <= percentile <= 100.")

        histogram_dict = self.histogram_dict
        percentile = self.percentile

        thresholds_dict = {}  # per tensor thresholds

        print("Number of tensors : {}".format(len(histogram_dict)))
        print("Number of histogram bins : {}".format(self.num_bins))
        print("Percentile : ({},{})".format(100.0 - percentile, percentile))

        for tensor, histogram in histogram_dict.items():
            hist = histogram[0]
            hist_edges = histogram[1]
            total = hist.sum()
            cdf = np.cumsum(hist / total)
            if self.symmetric:
                idx_right = np.searchsorted(cdf, percentile / 100.0)

                thresholds_dict[tensor] = (
                    -float(hist_edges[idx_right]),
                    float(hist_edges[idx_right]),
                )
            else:
                percent_to_cut_one_side = (100.0 - percentile) / 200.0
                idx_right = np.searchsorted(cdf, 1.0 - percent_to_cut_one_side)
                idx_left = np.searchsorted(cdf, percent_to_cut_one_side)
                thresholds_dict[tensor] = (
                    float(hist_edges[idx_left]),
                    float(hist_edges[idx_right]),
                )
            min_value = histogram[2]
            max_value = histogram[3]
            if thresholds_dict[tensor][0] < min_value:
                thresholds_dict[tensor] = (min_value, thresholds_dict[tensor][1])
            if thresholds_dict[tensor][1] > max_value:
                thresholds_dict[tensor] = (thresholds_dict[tensor][0], max_value)
            # Plot histogram for debug only
            if False:
                apply_plot(hist, hist_edges)

        return thresholds_dict

    def compute_entropy(self):
        histogram_dict = self.histogram_dict
        num_quantized_bins = self.num_quantized_bins

        thresholds_dict = {}  # per tensor thresholds

        print("Number of tensors : {}".format(len(histogram_dict)))
        print(
            "Number of histogram bins : {} (The number may increase depends on the data it collects)".format(
                self.num_bins
            )
        )
        print("Number of quantized bins : {}".format(self.num_quantized_bins))

        for tensor, histogram in histogram_dict.items():
            optimal_threshold = self.get_entropy_threshold(histogram, num_quantized_bins)
            thresholds_dict[tensor] = optimal_threshold

            # Plot histogram for debug only
            if False:
                apply_plot(histogram[0], histogram[1])

        return thresholds_dict

    def get_entropy_threshold(self, histogram, num_quantized_bins):
        """Given a dataset, find the optimal threshold for quantizing it.
        The reference distribution is `q`, and the candidate distribution is `p`.
        `q` is a truncated version of the original distribution.
        Ref: http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
        """
        import copy

        from scipy.stats import entropy

        hist = histogram[0]
        hist_edges = histogram[1]
        num_bins = hist.size
        zero_bin_index = num_bins // 2
        num_half_quantized_bin = num_quantized_bins // 2

        kl_divergence = np.zeros(zero_bin_index - num_half_quantized_bin + 1)
        thresholds = [(0, 0) for i in range(kl_divergence.size)]

        # <------------ num bins ---------------->
        #        <--- quantized bins ---->
        # |======|===========|===========|=======|
        #              zero bin index
        #        ^                       ^
        #        |                       |
        #   start index               end index          (start of iteration)
        #     ^                             ^
        #     |                             |
        #  start index                  end index               ...
        # ^                                      ^
        # |                                      |
        # start index                    end index       (end of iteration)

        for i in range(num_half_quantized_bin, zero_bin_index + 1, 1):
            start_index = zero_bin_index - i
            end_index = zero_bin_index + i + 1 if (zero_bin_index + i + 1) <= num_bins else num_bins

            thresholds[i - num_half_quantized_bin] = (
                float(hist_edges[start_index]),
                float(hist_edges[end_index]),
            )

            sliced_distribution = copy.deepcopy(hist[start_index:end_index])

            # reference distribution p
            p = sliced_distribution.copy()  # a copy of np array
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
            quantized_bins[-1] += sum(sliced_distribution[num_quantized_bins * num_merged_bins :])

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
                kl_divergence[i - num_half_quantized_bin] = float("inf")

        min_kl_divergence_idx = np.argmin(kl_divergence)
        optimal_threshold = thresholds[min_kl_divergence_idx]
        min_value = histogram[2]
        max_value = histogram[3]
        if optimal_threshold[0] < min_value:
            optimal_threshold = (min_value, optimal_threshold[1])
        if optimal_threshold[1] > max_value:
            optimal_threshold = (optimal_threshold[0], max_value)
        return optimal_threshold


def create_calibrator(
    model,
    op_types_to_calibrate: Optional[Sequence[str]] = None,
    augmented_model_path="augmented_model.onnx",
    calibrate_method=CalibrationMethod.MinMax,
    use_external_data_format=False,
    extra_options={},
):

    calibrator = None
    if calibrate_method == CalibrationMethod.MinMax:
        # default settings for min-max algorithm
        symmetric = False if "symmetric" not in extra_options else extra_options["symmetric"]
        moving_average = False if "moving_average" not in extra_options else extra_options["moving_average"]
        averaging_constant = 0.01 if "averaging_constant" not in extra_options else extra_options["averaging_constant"]
        calibrator = MinMaxCalibrater(
            model,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format=use_external_data_format,
            symmetric=symmetric,
            moving_average=moving_average,
            averaging_constant=averaging_constant,
        )
    elif calibrate_method == CalibrationMethod.Entropy:
        # default settings for entropy algorithm
        num_bins = 128 if "num_bins" not in extra_options else extra_options["num_bins"]
        num_quantized_bins = 128 if "num_quantized_bins" not in extra_options else extra_options["num_quantized_bins"]
        symmetric = False if "symmetric" not in extra_options else extra_options["symmetric"]
        calibrator = EntropyCalibrater(
            model,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format=use_external_data_format,
            symmetric=symmetric,
            num_bins=num_bins,
            num_quantized_bins=num_quantized_bins,
        )
    elif calibrate_method == CalibrationMethod.Percentile:
        # default settings for percentile algorithm
        num_bins = 2048 if "num_bins" not in extra_options else extra_options["num_bins"]
        percentile = 99.999 if "percentile" not in extra_options else extra_options["percentile"]
        symmetric = True if "symmetric" not in extra_options else extra_options["symmetric"]
        calibrator = PercentileCalibrater(
            model,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format=use_external_data_format,
            symmetric=symmetric,
            num_bins=num_bins,
            percentile=percentile,
        )

    if calibrator:
        calibrator.augment_graph()
        calibrator.create_inference_session()
        return calibrator

    raise ValueError("Unsupported calibration method {}".format(calibrate_method))
