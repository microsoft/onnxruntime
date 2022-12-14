# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import logging

import numpy as np
import onnx
import onnx.numpy_helper
from onnx import onnx_pb as onnx_proto

from .onnx_model import ONNXModel
from .quant_utils import (
    TENSOR_NAME_QUANT_SUFFIX,
    QuantizationMode,
    QuantizedValue,
    QuantizedValueType,
    QuantType,
    __producer__,
    __version__,
    add_infer_metadata,
    attribute_to_kwarg,
    compute_scale_zp,
    find_by_name,
    get_qmin_qmax_for_qType,
    get_qrange_for_qType,
    model_has_infer_metadata,
    quantize_data,
    save_and_reload_model,
    tensor_proto_to_array,
)
from .registry import CreateOpQuantizer


class ONNXQuantizer:
    def __init__(
        self,
        model,
        per_channel,
        reduce_range,
        mode,
        static,
        weight_qType,
        activation_qType,
        tensors_range,
        nodes_to_quantize,
        nodes_to_exclude,
        op_types_to_quantize,
        extra_options=None,
    ):

        if not model_has_infer_metadata(model):
            model = save_and_reload_model(model)
        self.value_infos = {vi.name: vi for vi in model.graph.value_info}
        self.value_infos.update({ot.name: ot for ot in model.graph.output})
        self.value_infos.update({it.name: it for it in model.graph.input})

        self.model = ONNXModel(model)
        if not static:
            self.model.replace_gemm_with_matmul()

        self.per_channel = per_channel  # weight-pack per channel
        self.reduce_range = reduce_range
        self.mode = mode  # QuantizationMode.Value
        self.static = static  # use static quantization for inputs.
        self.fuse_dynamic_quant = False

        self.extra_options = extra_options if extra_options else {}
        self.enable_subgraph_quantization = (
            "EnableSubgraph" in self.extra_options and self.extra_options["EnableSubgraph"]
        )
        self.force_quantize_no_input_check = (
            "ForceQuantizeNoInputCheck" in self.extra_options and self.extra_options["ForceQuantizeNoInputCheck"]
        )
        self.q_matmul_const_b_only = "MatMulConstBOnly" in self.extra_options and self.extra_options["MatMulConstBOnly"]
        is_weight_int8 = weight_qType == QuantType.QInt8
        self.is_weight_symmetric = (
            is_weight_int8 if "WeightSymmetric" not in self.extra_options else self.extra_options["WeightSymmetric"]
        )
        self.is_activation_symmetric = (
            False if "ActivationSymmetric" not in self.extra_options else self.extra_options["ActivationSymmetric"]
        )

        self.activation_qType = (
            onnx_proto.TensorProto.INT8 if activation_qType == QuantType.QInt8 else onnx_proto.TensorProto.UINT8
        )
        self.weight_qType = (
            onnx_proto.TensorProto.INT8 if weight_qType == QuantType.QInt8 else onnx_proto.TensorProto.UINT8
        )
        """
            Dictionary specifying the min and max values for tensors. It has following format:
                {
                    "param_name": [min, max]
                }
            example:
                {
                    'Conv_3:0': [np.float32(0), np.float32(0.5)],
                    'Conv_4:0': [np.float32(1), np.float32(3.5)]
                }
        """
        self.tensors_range = tensors_range
        self.nodes_to_quantize = nodes_to_quantize  # specific nodes to quantize
        self.nodes_to_exclude = nodes_to_exclude  # specific nodes to exclude
        self.op_types_to_quantize = op_types_to_quantize
        self.new_nodes = []
        self.parent = None
        self.graph_scope = "/"  # for human readable debug information
        self.tensor_names = {}  # in case the shape inference not totally working
        self.tensor_names.update({ot.name: 1 for ot in model.graph.output})
        self.tensor_names.update({it.name: 1 for it in model.graph.input})
        for node in self.model.model.graph.node:
            self.tensor_names.update({output_name: 1 for output_name in node.output})

        self.opset_version = self.check_opset_version()

        if not self.mode in QuantizationMode:
            raise ValueError("unsupported quantization mode {}".format(self.mode))

        self.quantization_params = self.calculate_quantization_params()

        # QuantizeRange tensor name and zero tensor name for scale and zero point calculation.
        # Used when static is False
        self.fixed_qrange_uint8_name = "fixed_quantization_range_uint8"
        self.fixed_qrange_int8_name = "fixed_quantization_range_int8"
        # For uint8 data-type, to compute zero point, we subtract rmin from 0 (represented by fixed_zero_name tensor)
        self.fixed_zero_name = "fixed_zero"
        # For int8 data-type, zero point is always zero (respresented by fixed_zero_point_name tensor)
        self.fixed_zero_zp_name = "fixed_zero_zp"

        # Map of all original value names to quantized value names
        self.quantized_value_map = {}
        # some output from nodes will be quantized, yet itself should be treat as existing so
        # no dequantized will be applied when needed later
        self.generated_value_names = self.model.get_non_initializer_inputs()
        # to store specified scale and zeropoint instead of calculated value, tensor_name->(scale, zeropoint)
        self.used_scale_zp_map = {}

    # routines for subgraph support
    def quantize_subgraph(self, subgraph, graph_key):
        """
        generate submodel for the subgraph, so that we re-utilize current quantization implementation.
        quantize the submodel
        update subgraph and set it back to node
        """
        warped_model = onnx.helper.make_model(
            subgraph,
            producer_name="onnx-quantizer",
            opset_imports=self.model.model.opset_import,
        )
        add_infer_metadata(warped_model)
        sub_quanitzer = ONNXQuantizer(
            warped_model,
            self.per_channel,
            self.reduce_range,
            self.mode,
            self.static,
            self.weight_qType,
            self.activation_qType,
            self.tensors_range,
            self.nodes_to_quantize,
            self.nodes_to_exclude,
            self.op_types_to_quantize,
            self.extra_options,
        )
        sub_quanitzer.parent = self
        sub_quanitzer.graph_scope = "{}{}/".format(self.graph_scope, graph_key)
        sub_quanitzer.quantize_model()
        return sub_quanitzer.model.model.graph

    def quantize_node_with_sub_graph(self, node):
        """
        Check subgraph, if any, quantize it and replace it.
        return new_nodes added for quantizing subgraph
        """
        graph_attrs = [
            attr
            for attr in node.attribute
            if attr.type == onnx.AttributeProto.GRAPH or attr.type == onnx.AttributeProto.GRAPHS
        ]
        if len(graph_attrs) == 0:
            return node
        node_name = node.name if node.name != "" else "{}_node_count_{}".format(node.op_type, len(self.new_nodes))
        kwargs = {}
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                kv = {attr.name: self.quantize_subgraph(attr.g, "{}:{}".format(node_name, attr.name))}
            elif attr.type == onnx.AttributeProto.GRAPHS:
                value = []
                for subgraph in attr.graphs:
                    value.extend(
                        [
                            self.quantize_subgraph(
                                subgraph,
                                "{}:{}:{}".format(node_name, attr.name, len(value)),
                            )
                        ]
                    )
                kv = {attr.name: value}
            else:
                kv = attribute_to_kwarg(attr)
            kwargs.update(kv)
        return onnx.helper.make_node(node.op_type, node.input, node.output, name=node.name, **kwargs)

    def check_opset_version(self):
        ai_onnx_domain = [
            opset for opset in self.model.model.opset_import if not opset.domain or opset.domain == "ai.onnx"
        ]
        if 1 != len(ai_onnx_domain):
            raise ValueError("Failed to find proper ai.onnx domain")
        opset_version = ai_onnx_domain[0].version

        if opset_version == 10:
            logging.warning(
                "The original model opset version is {}, which does not support node fusions. Please update the model to opset >= 11 for better performance.".format(
                    opset_version
                )
            )
            return 10

        if opset_version < 10:
            logging.warning(
                "The original model opset version is {}, which does not support quantization. Please update the model to opset >= 11. Updating the model automatically to opset 11. Please verify the quantized model.".format(
                    opset_version
                )
            )
            self.model.model.opset_import.remove(ai_onnx_domain[0])
            self.model.model.opset_import.extend([onnx.helper.make_opsetid("", 11)])
            opset_version = 11

        self.fuse_dynamic_quant = True
        return opset_version

    def has_QDQ_nodes(self):
        """
        Detect if model already has QuantizeLinear or DequantizeLinear.
        """
        return any(
            node.op_type == "QuantizeLinear" or node.op_type == "DequantizeLinear" for node in self.model.nodes()
        )

    def find_initializer_in_path(self, initializer_name):
        if find_by_name(initializer_name, self.model.initializer()) is not None:
            return True
        if self.parent is not None:
            return self.parent.find_initializer_in_path(initializer_name)
        return False

    def add_new_nodes(self, nodes):
        self.new_nodes.extend(nodes)
        for node in nodes:
            for output_name in node.output:
                self.generated_value_names.add(output_name)

    def quantize_model(self):
        if self.has_QDQ_nodes():
            logging.warning(
                "Please check if the model is already quantized."
                "Note you don't need to quantize a QAT model. OnnxRuntime support to run QAT model directly."
            )

        for node in self.model.nodes():
            # quantize subgraphes if have
            if self.enable_subgraph_quantization:
                node = self.quantize_node_with_sub_graph(node)

            number_of_existing_new_nodes = len(self.new_nodes)
            op_quantizer = CreateOpQuantizer(self, node)
            op_quantizer.quantize()
            for i in range(number_of_existing_new_nodes, len(self.new_nodes)):
                for output_name in self.new_nodes[i].output:
                    self.generated_value_names.add(output_name)

        self._dequantize_outputs()

        # extend is used to append to the list for a protobuf fields
        # https://developers.google.com/protocol-buffers/docs/reference/python-generated?csw=1#fields
        self.model.graph().ClearField("node")
        self.model.graph().node.extend(self.new_nodes)

        # Remove ununsed initializers from graph, starting from the top level graph.
        if self.parent is None:
            _, initializers_not_found = self.model.clean_initializers()
            if len(initializers_not_found) > 0:
                raise RuntimeError("Invalid model with unknown initializers/tensors." + str(initializers_not_found))

        self.model.model.producer_name = __producer__
        self.model.model.producer_version = __version__

        return self.model.model

    def is_input_a_initializer(self, input_name):
        initializer = find_by_name(input_name, self.model.initializer())
        return initializer is not None

    def is_per_channel(self):
        return self.per_channel

    def is_valid_quantize_weight(self, weight_name):
        weight = find_by_name(weight_name, self.model.initializer())
        if weight is not None:
            return weight.data_type == onnx_proto.TensorProto.FLOAT
        if (not self.enable_subgraph_quantization) or (self.parent is None):
            return False
        return self.parent.is_valid_quantize_weight(weight_name)

    def is_float_tensor(self, tensor_name):
        if self.is_input_a_initializer(tensor_name):
            return self.is_valid_quantize_weight(tensor_name)

        if tensor_name in self.value_infos.keys():
            vi = self.value_infos[tensor_name]
            if vi.type.HasField("tensor_type") and vi.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                return True
        elif self.enable_subgraph_quantization and self.parent:
            return self.parent.is_float_tensor(tensor_name)
        else:
            logging.warning(
                "Failed to infer data type of tensor: {}. Please add data type info for this tensor "
                "if your model has customized operators.".format(tensor_name)
            )

        return False

    def should_quantize_node(self, node):
        if (
            self.nodes_to_quantize is not None
            and len(self.nodes_to_quantize) != 0
            and node.name not in self.nodes_to_quantize
        ):
            return False

        if node.op_type not in self.op_types_to_quantize:
            return False

        if self.nodes_to_exclude is not None and node.name in self.nodes_to_exclude:
            return False

        return True

    def _get_dynamic_input_quantization_params(self, input_name, nodes_list, qType):
        """
        Create nodes for dynamic quantization of input and add them to nodes_list.
            parameter input_name: Name of the input.
            parameter nodes_list: new nodes are appended to this list.
            parameter qType: type to quantize to.
            return: scale_name, zero_point_name, scale_shape, zero_point_shape.
        """
        if qType == onnx_proto.TensorProto.INT8:
            return self._get_dynamic_input_quantization_params_int8(input_name, nodes_list)

        return self._get_dynamic_input_quantization_params_uint8(input_name, nodes_list)

    def _get_dynamic_input_quantization_params_int8(self, input_name, nodes_list):
        """
        Create nodes for dynamic quantization of input to int8 and add them to nodes_list
            parameter input_name: Name of the input.
            parameter nodes_list: new nodes are appended to this list.
            return: scale_name, zero_point_name, scale_shape, zero_point_shape.
        """
        qType = onnx_proto.TensorProto.INT8

        # Reduce min and Reduce max
        input_scale_name = input_name + "_scale"

        reduce_min_name = input_name + "_ReduceMin"
        reduce_min_node = onnx.helper.make_node(
            "ReduceMin",
            [input_name],
            [reduce_min_name + ":0"],
            reduce_min_name,
            keepdims=0,
        )
        nodes_list.append(reduce_min_node)

        reduce_max_name = input_name + "_ReduceMax"
        reduce_max_node = onnx.helper.make_node(
            "ReduceMax",
            [input_name],
            [reduce_max_name + ":0"],
            reduce_max_name,
            keepdims=0,
        )
        nodes_list.append(reduce_max_node)

        # Compute scale
        #   Find abs(rmin)
        reduce_min_abs_name = reduce_min_name + "_Abs"
        reduce_min_abs_node = onnx.helper.make_node(
            "Abs",
            [reduce_min_node.output[0]],
            [reduce_min_abs_name + ":0"],
            reduce_min_abs_name,
        )
        nodes_list.append(reduce_min_abs_node)
        #   Find abs(rmax)
        reduce_max_abs_name = reduce_max_name + "_Abs"
        reduce_max_abs_node = onnx.helper.make_node(
            "Abs",
            [reduce_max_node.output[0]],
            [reduce_max_abs_name + ":0"],
            reduce_max_abs_name,
        )
        nodes_list.append(reduce_max_abs_node)
        #   Compute max of abs(rmin) and abs(rmax)
        abs_max_name = input_name + "_Abs_Max"
        abs_max_node = onnx.helper.make_node(
            "Max",
            [reduce_min_abs_node.output[0], reduce_max_abs_node.output[0]],
            [abs_max_name + ":0"],
            abs_max_name,
        )
        nodes_list.append(abs_max_node)
        #   and divide by (quantize_range/2.0) which will be equal to max(...)*2.0/quantize_range
        initializer_div = onnx.helper.make_tensor(
            self.fixed_qrange_int8_name,
            onnx_proto.TensorProto.FLOAT,
            [],
            [get_qrange_for_qType(qType) / 2.0],
        )
        self.model.add_initializer(initializer_div)
        scale_div_name = input_name + "scale_Div"
        scale_div_node = onnx.helper.make_node(
            "Div",
            [abs_max_node.output[0], self.fixed_qrange_int8_name],
            [input_scale_name],
            scale_div_name,
        )
        nodes_list.append(scale_div_node)

        # Zero point
        initializer_zp = onnx.helper.make_tensor(self.fixed_zero_zp_name, qType, [], [0])
        self.model.add_initializer(initializer_zp)

        return input_scale_name, self.fixed_zero_zp_name, [], []

    def _get_dynamic_input_quantization_params_uint8(self, input_name, nodes_list):
        """
        Create nodes for dynamic quantization of input to uint8 and add them to nodes_list
            parameter input_name: Name of the input.
            parameter nodes_list: new nodes are appended to this list.
            return: scale_name, zero_point_name, scale_shape, zero_point_shape.
        """
        qType = onnx_proto.TensorProto.UINT8
        # Reduce min and Reduce max
        input_scale_name = input_name + "_scale"
        input_zp_name = input_name + "_zero_point"

        reduce_min_name = input_name + "_ReduceMin"
        reduce_min_node = onnx.helper.make_node(
            "ReduceMin",
            [input_name],
            [reduce_min_name + ":0"],
            reduce_min_name,
            keepdims=0,
        )
        nodes_list.append(reduce_min_node)

        reduce_max_name = input_name + "_ReduceMax"
        reduce_max_node = onnx.helper.make_node(
            "ReduceMax",
            [input_name],
            [reduce_max_name + ":0"],
            reduce_max_name,
            keepdims=0,
        )
        nodes_list.append(reduce_max_node)

        # Add tensors for quantize range and zero value.
        initializer_qrange = onnx.helper.make_tensor(
            self.fixed_qrange_uint8_name,
            onnx_proto.TensorProto.FLOAT,
            [],
            [get_qrange_for_qType(qType)],
        )
        self.model.add_initializer(initializer_qrange)
        initializer_qvalue = onnx.helper.make_tensor(self.fixed_zero_name, onnx_proto.TensorProto.FLOAT, [], [0.0])
        self.model.add_initializer(initializer_qvalue)

        # Compute Scale
        #   Subtract rmax and rmin
        scale_sub_name = input_name + "_scale_Sub"
        scale_sub_node = onnx.helper.make_node(
            "Sub",
            [reduce_max_node.output[0], reduce_min_node.output[0]],
            [scale_sub_name + ":0"],
            scale_sub_name,
        )
        nodes_list.append(scale_sub_node)
        #   and divide by quantize range
        scale_div_name = input_name + "_scale_Div"
        scale_div_node = onnx.helper.make_node(
            "Div",
            [scale_sub_node.output[0], self.fixed_qrange_uint8_name],
            [input_scale_name],
            scale_div_name,
        )
        nodes_list.append(scale_div_node)

        # Compute zero point
        #   Subtract zero and rmin
        zp_sub_name = input_name + "_zero_point_Sub"
        zp_sub_node = onnx.helper.make_node(
            "Sub",
            [self.fixed_zero_name, reduce_min_node.output[0]],
            [zp_sub_name + ":0"],
            zp_sub_name,
        )
        nodes_list.append(zp_sub_node)
        #   Divide by scale
        zp_div_name = input_name + "_zero_point_Div"
        zp_div_node = onnx.helper.make_node(
            "Div",
            [zp_sub_node.output[0], input_scale_name],
            [zp_div_name + ":0"],
            zp_div_name,
        )
        nodes_list.append(zp_div_node)
        #   Compute floor
        zp_floor_name = input_name + "_zero_point_Floor"
        zp_floor_node = onnx.helper.make_node("Floor", zp_div_node.output, [zp_floor_name + ":0"], zp_floor_name)
        nodes_list.append(zp_floor_node)
        #   Cast to integer
        zp_cast_name = input_name + "_zero_point_Cast"
        zp_cast_node = onnx.helper.make_node("Cast", zp_floor_node.output, [input_zp_name], zp_cast_name, to=qType)
        nodes_list.append(zp_cast_node)

        return input_scale_name, input_zp_name, [], []

    def _get_quantization_params(self, param_name, use_scale=None, use_zeropoint=None):
        """
        Create initializers and inputs in the graph for zero point and scale of output.
        Zero point and scale values are obtained from self.quantization_params if specified.
            parameter param_name: Name of the quantization parameter.
            return: result, scale_name, zero_point_name, scale_shape, zero_point_shape.
        """
        if use_scale is None or use_zeropoint is None:
            if self.quantization_params is None or param_name not in self.quantization_params:
                logging.info('Quantization parameters for tensor:"{}" not specified'.format(param_name))
                return False, "", "", "", ""

            params = self.quantization_params[param_name]
            if params is None or len(params) != 2:
                raise ValueError(
                    "Quantization parameters should contain zero point and scale. "
                    "Specified values for output {}: {}".format(param_name, params)
                )

            zero_point_values = [params[0]]
            scale_values = [params[1]]
        else:
            zero_point_values = [use_zeropoint]
            scale_values = [use_scale]

        zero_point_shape = []
        zero_point_name = param_name + "_zero_point"
        zero_point_type = self.activation_qType
        scale_shape = []
        scale_name = param_name + "_scale"

        # Add initializers
        init_zp = onnx.helper.make_tensor(zero_point_name, zero_point_type, zero_point_shape, zero_point_values)
        self.model.add_initializer(init_zp)
        init_scale = onnx.helper.make_tensor(scale_name, onnx_proto.TensorProto.FLOAT, scale_shape, scale_values)
        self.model.add_initializer(init_scale)

        return True, scale_name, zero_point_name, scale_shape, zero_point_shape

    def _get_quantize_input_nodes(self, node, input_index, qType, given_scale_name=None, given_zp_name=None):
        """
        Given an input for a node (which is not a initializer), this function

        - add nodes to compute zero point and scale for this input if they don't exist.
        - add new QuantizeLinear node to quantize the input.

        :param node: node being quantized in NodeProto format.
        :param input_index: index of input in node.input.
        :param qType: type to quantize to.
        :param given_scale_name: if those inputs need to be quanitzed using this scale tensor.
        :param given_zp_name: if those inputs to be quantized using this zeropoint tensor.
        :return: List of newly created nodes in NodeProto format.
        """
        input_name = node.input[input_index]
        output_name = input_name + TENSOR_NAME_QUANT_SUFFIX
        ql_node_name = input_name + "_QuantizeLinear"

        if (given_scale_name is not None) and (given_zp_name is not None):
            data_found, scale_name, zp_name = (True, given_scale_name, given_zp_name)
        else:
            data_found, scale_name, zp_name, _, _ = self._get_quantization_params(input_name)

        nodes = []
        if data_found:
            qlinear_node = onnx.helper.make_node(
                "QuantizeLinear",
                [input_name, scale_name, zp_name],
                [output_name],
                ql_node_name,
            )
        else:
            if self.static:
                return None
            # dynamic mode
            # Scale and Zero Points not available for this input. Add nodes to dynamically compute it
            if self.fuse_dynamic_quant and qType == onnx_proto.TensorProto.UINT8:
                scale_name = input_name + "_scale"
                zp_name = input_name + "_zero_point"
                qlinear_node = onnx.helper.make_node(
                    "DynamicQuantizeLinear",
                    [input_name],
                    [output_name, scale_name, zp_name],
                    ql_node_name,
                )
            else:
                (
                    scale_name,
                    zp_name,
                    scale_shape,
                    zp_shape,
                ) = self._get_dynamic_input_quantization_params(input_name, nodes, qType)
                qlinear_node = onnx.helper.make_node(
                    "QuantizeLinear",
                    [input_name, scale_name, zp_name],
                    [output_name],
                    ql_node_name,
                )

        self.quantized_value_map[input_name] = QuantizedValue(input_name, output_name, scale_name, zp_name, qType)
        return nodes + [qlinear_node]

    def set_quant_scale_zp(self, tensor_name, value):
        assert isinstance(value, tuple) and len(value) == 2, "value must be scale(float) and zeropoint"
        assert tensor_name not in self.used_scale_zp_map, f"{tensor_name} has been setted before"
        self.used_scale_zp_map[tensor_name] = value

    def find_quant_scale_zp(self, input_name):
        if input_name in self.used_scale_zp_map:
            return self.used_scale_zp_map[input_name]
        if self.parent is not None:
            return self.parent.find_quantized_value(input_name)
        return (None, None)

    def find_quantized_value(self, input_name):
        if input_name in self.quantized_value_map:
            return self.quantized_value_map[input_name]
        if self.parent is not None:
            return self.parent.find_quantized_value(input_name)
        return None

    def quantize_bias_static(self, bias_name, input_name, weight_name, beta=1.0):
        """
        Quantized the bias. Zero Point == 0 and Scale == Input_Scale * Weight_Scale
        """

        # Handle case where bias already in quantization map
        if bias_name in self.quantized_value_map:
            return self.quantized_value_map[bias_name].q_name

        # get scale for weight
        weight_scale_name = self.quantized_value_map[weight_name].scale_name
        weight_initializer = find_by_name(weight_scale_name, self.model.initializer())
        weight_scale = tensor_proto_to_array(weight_initializer)

        # get bias
        bias_initializer = find_by_name(bias_name, self.model.initializer())
        bias_data = tensor_proto_to_array(bias_initializer)
        quantized_bias_name = bias_name + TENSOR_NAME_QUANT_SUFFIX

        # get scale for input
        if input_name in self.quantized_value_map:
            input_scale_name = self.quantized_value_map[input_name].scale_name
        elif input_name in self.quantization_params:
            _, input_scale_name, _, _, _ = self._get_quantization_params(input_name)
        else:
            raise ValueError("Expected {} to be in quantized value map for static quantization".format(input_name))

        inputscale_initializer = find_by_name(input_scale_name, self.model.initializer())
        input_scale = tensor_proto_to_array(inputscale_initializer)

        # calcuate scale for bias
        bias_scale = input_scale * weight_scale * beta

        # quantize bias
        quantized_data = (np.asarray(bias_data) / bias_scale).round().astype(np.int32)

        # update bias initializer
        bias_np_data = np.asarray(quantized_data, dtype=np.int32).reshape(bias_initializer.dims)
        packed_bias_initializer = onnx.numpy_helper.from_array(bias_np_data, quantized_bias_name)
        self.model.initializer().extend([packed_bias_initializer])

        # update scale initializer
        quantized_bias_scale_name = quantized_bias_name + "_scale"
        bias_scale_data = np.asarray(bias_scale, dtype=np.float32).reshape(-1)
        if self.is_per_channel():
            packed_bias_scale_initializer = onnx.numpy_helper.from_array(bias_scale_data, quantized_bias_scale_name)
        else:
            packed_bias_scale_initializer = onnx.helper.make_tensor(
                quantized_bias_scale_name, onnx_proto.TensorProto.FLOAT, [], bias_scale_data
            )
        self.model.initializer().extend([packed_bias_scale_initializer])

        # update zero initializer
        quantized_bias_zp_name = quantized_bias_name + "_zero_point"
        bias_zp_data = np.zeros(bias_scale.shape, dtype=np.int32).reshape(-1)
        if self.is_per_channel():
            packed_bias_zp_initializer = onnx.numpy_helper.from_array(bias_zp_data, quantized_bias_zp_name)
        else:
            packed_bias_zp_initializer = onnx.helper.make_tensor(
                quantized_bias_zp_name, onnx_proto.TensorProto.INT32, [], bias_zp_data
            )
        self.model.initializer().extend([packed_bias_zp_initializer])

        assert bias_name not in self.quantized_value_map
        quantized_value = QuantizedValue(
            bias_name,
            quantized_bias_name,
            quantized_bias_scale_name,
            quantized_bias_zp_name,
            QuantizedValueType.Initializer,
            0 if bias_scale_data.size > 1 else None,
        )
        self.quantized_value_map[bias_name] = quantized_value

        return quantized_bias_name

    def contains_tensor(self, tensor_name):
        """
        only check for value info and newly generated tensor names, initializers are checked separately
        """
        return (
            (tensor_name in self.value_infos)
            or (tensor_name in self.tensor_names)
            or (tensor_name in self.generated_value_names)
        )

    def quantize_activation(self, node, indices, from_subgraph=False):
        return self.__quantize_inputs(
            node=node,
            indices=indices,
            initializer_use_weight_qType=False,
            reduce_range=False,
            op_level_per_channel=False,
            axis=-1,
            from_subgraph=from_subgraph,
        )

    # In some circumstances a weight is not an initializer, for example of MatMul, if both A and B are not
    # initializer, B can still be considered as Weight
    def quantize_weight(
        self,
        node,
        indices,
        reduce_range=False,
        op_level_per_channel=False,
        axis=-1,
        from_subgraph=False,
    ):
        return self.__quantize_inputs(
            node=node,
            indices=indices,
            initializer_use_weight_qType=True,
            reduce_range=reduce_range,
            op_level_per_channel=op_level_per_channel,
            axis=axis,
            from_subgraph=from_subgraph,
        )

    def __quantize_inputs(
        self,
        node,
        indices,
        initializer_use_weight_qType=True,
        reduce_range=False,
        op_level_per_channel=False,
        axis=-1,
        from_subgraph=False,
    ):
        """
        Given a node, this function quantizes the inputs as follows:
            - If input is an initializer, quantize the initializer data, replace old initializer
              with new initializer
            - Else, add QuantizeLinear nodes to perform quantization
            parameter node: node being quantized in NodeProto format.
            parameter indices: input indices to quantize.
            return: (List of quantized input names,
                     List of zero point names used for input quantization,
                     List of scale names used for input quantization,
                     List of new QuantizeLinear nodes created)
        """

        scale_names = []
        zero_point_names = []
        quantized_input_names = []
        nodes = []

        for input_index in indices:
            node_input = node.input[input_index]

            # Find if this input is already quantized
            if node_input in self.quantized_value_map:
                quantized_value = self.quantized_value_map[node_input]
                scale_names.append(quantized_value.scale_name)
                zero_point_names.append(quantized_value.zp_name)
                quantized_input_names.append(quantized_value.q_name)
                continue

            # Quantize the input
            initializer = find_by_name(node_input, self.model.initializer())
            if initializer is not None:
                if self.per_channel and op_level_per_channel:
                    (q_weight_name, zp_name, scale_name,) = self.quantize_weight_per_channel(
                        initializer.name,
                        self.weight_qType if initializer_use_weight_qType else self.activation_qType,
                        axis,
                        reduce_range,
                    )
                else:
                    q_weight_name, zp_name, scale_name = self.quantize_initializer(
                        initializer,
                        self.weight_qType if initializer_use_weight_qType else self.activation_qType,
                        reduce_range,
                    )

                quantized_input_names.append(q_weight_name)
                zero_point_names.append(zp_name)
                scale_names.append(scale_name)
            elif self.contains_tensor(node_input):
                # Add QuantizeLinear node.
                qlinear_node = self.model.find_node_by_name(
                    node_input + "_QuantizeLinear", self.new_nodes, self.model.graph()
                )
                if qlinear_node is None:
                    quantize_input_nodes = self._get_quantize_input_nodes(node, input_index, self.activation_qType)
                    if quantize_input_nodes is None:
                        return (None, None, None, None)
                    if from_subgraph:
                        self.add_new_nodes(quantize_input_nodes)
                    else:
                        nodes.extend(quantize_input_nodes)
                    qlinear_node = quantize_input_nodes[-1]

                if qlinear_node.op_type == "QuantizeLinear":
                    quantized_input_names.extend(qlinear_node.output)
                    scale_names.append(qlinear_node.input[1])
                    zero_point_names.append(qlinear_node.input[2])
                else:
                    quantized_input_names.append(qlinear_node.output[0])
                    scale_names.append(qlinear_node.output[1])
                    zero_point_names.append(qlinear_node.output[2])
            elif self.parent is not None:
                (
                    parent_quantized_input_names,
                    parent_zero_point_names,
                    parent_scale_names,
                    _,
                ) = self.parent.__quantize_inputs(
                    node,
                    [input_index],
                    initializer_use_weight_qType=initializer_use_weight_qType,
                    reduce_range=reduce_range,
                    op_level_per_channel=op_level_per_channel,
                    axis=axis,
                    from_subgraph=True,
                )
                quantized_input_names.append(parent_quantized_input_names[0])
                scale_names.append(parent_scale_names[0])
                zero_point_names.append(parent_zero_point_names[0])
                # node should not be add this child level here
            else:
                raise ValueError(
                    "Invalid tensor name to quantize: {} @graph scope{}".format(node_input, self.graph_scope)
                )

        return quantized_input_names, zero_point_names, scale_names, nodes

    def quantize_initializer(self, weight, qType, reduce_range=False, keep_float_weight=False):
        """
        :param weight: TensorProto initializer
        :param qType: type to quantize to
        :param keep_float_weight: Whether to quantize the weight. In some cases, we only want to qunatize scale and zero point.
                                  If keep_float_weight is False, quantize the weight, or don't quantize the weight.
        :return: quantized weight name, zero point name, scale name
        """
        # Find if this input is already quantized
        if weight.name in self.quantized_value_map:
            quantized_value = self.quantized_value_map[weight.name]
            return (
                quantized_value.q_name,
                quantized_value.zp_name,
                quantized_value.scale_name,
            )

        q_weight_name = weight.name + TENSOR_NAME_QUANT_SUFFIX
        zp_name = weight.name + "_zero_point"
        scale_name = weight.name + "_scale"

        # Update packed weight, zero point, and scale initializers
        weight_data = tensor_proto_to_array(weight)
        _, _, zero_point, scale, q_weight_data = quantize_data(
            weight_data.flatten().tolist(),
            qType,
            self.is_weight_symmetric,
            self.reduce_range and reduce_range,
        )
        scale_initializer = onnx.helper.make_tensor(scale_name, onnx_proto.TensorProto.FLOAT, [], [scale])
        zero_initializer = onnx.helper.make_tensor(zp_name, qType, [], [zero_point])
        self.model.initializer().extend([scale_initializer, zero_initializer])

        if not keep_float_weight:
            q_weight_data = np.asarray(q_weight_data, dtype=onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[qType]).reshape(
                weight.dims
            )
            q_weight_initializer = onnx.numpy_helper.from_array(q_weight_data, q_weight_name)
            self.model.initializer().extend([q_weight_initializer])

        # Log entry for this quantized weight
        quantized_value = QuantizedValue(
            weight.name,
            q_weight_name,
            scale_name,
            zp_name,
            QuantizedValueType.Initializer,
            None,
        )
        self.quantized_value_map[weight.name] = quantized_value

        return q_weight_name, zp_name, scale_name

    def quantize_weight_per_channel(
        self,
        weight_name,
        weight_qType,
        channel_axis,
        reduce_range=True,
        keep_float_weight=False,
    ):
        # Find if this input is already quantized
        if weight_name in self.quantized_value_map:
            quantized_value = self.quantized_value_map[weight_name]
            return (
                quantized_value.q_name,
                quantized_value.zp_name,
                quantized_value.scale_name,
            )

        initializer = find_by_name(weight_name, self.model.initializer())
        if initializer is None:
            raise ValueError("{} is not an initializer", weight_name)

        weights = tensor_proto_to_array(initializer)
        channel_count = weights.shape[channel_axis]
        rmin_list = []
        rmax_list = []
        zero_point_list = []
        scale_list = []
        quantized_per_channel_data_list = []
        for i in range(channel_count):
            per_channel_data = weights.take(i, channel_axis)
            rmin, rmax, zero_point, scale, quantized_per_channel_data = quantize_data(
                per_channel_data.flatten().tolist(),
                weight_qType,
                self.is_weight_symmetric or weight_qType == onnx_proto.TensorProto.INT8,
                self.reduce_range and reduce_range,
            )
            rmin_list.append(rmin)
            rmax_list.append(rmax)
            zero_point_list.append(zero_point)
            scale_list.append(scale)
            quantized_per_channel_data_list.append(quantized_per_channel_data)

        # combine per_channel_data into one
        reshape_dims = list(weights.shape)  # deep copy
        reshape_dims[channel_axis] = 1  # only one per channel for reshape
        quantized_weights = np.asarray(quantized_per_channel_data_list[0]).reshape(reshape_dims)
        for i in range(1, len(quantized_per_channel_data_list)):
            channel_weights = np.asarray(quantized_per_channel_data_list[i]).reshape(reshape_dims)
            quantized_weights = np.concatenate((quantized_weights, channel_weights), channel_axis)

        q_weight_name = weight_name + TENSOR_NAME_QUANT_SUFFIX
        zp_name = weight_name + "_zero_point"
        scale_name = weight_name + "_scale"

        quantized_value = QuantizedValue(
            weight_name,
            q_weight_name,
            scale_name,
            zp_name,
            QuantizedValueType.Initializer,
            None,
        )
        self.quantized_value_map[weight_name] = quantized_value

        # Update packed weight, zero point, and scale initializers
        zero_scale_shape = [initializer.dims[channel_axis]]
        scale_initializer = onnx.helper.make_tensor(
            scale_name, onnx_proto.TensorProto.FLOAT, zero_scale_shape, scale_list
        )
        zero_initializer = onnx.helper.make_tensor(zp_name, weight_qType, zero_scale_shape, zero_point_list)

        self.model.initializer().extend([scale_initializer, zero_initializer])

        if not keep_float_weight:
            quantized_weights = np.asarray(
                quantized_weights,
                dtype=onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[weight_qType],
            ).reshape(initializer.dims)
            q_weight_initializer = onnx.numpy_helper.from_array(quantized_weights, q_weight_name)
            self.model.initializer().extend([q_weight_initializer])

        return q_weight_name, zp_name, scale_name

    def _dequantize_value(self, value_name):
        """
        Given a value (input/output) which is quantized, add a DequantizeLinear node to dequantize
        it back to float32
            parameter value_name: value to dequantize
            parameter new_nodes_list: List of new nodes created before processing current node
            return: None if there is already a DequantizeLinear node that dequantizes it
                    A DequantizeLinear node otherwise
        """
        if (value_name in self.quantized_value_map) and (value_name not in self.generated_value_names):
            quantized_value = self.quantized_value_map[value_name]
            # Add DequantizeLinear Node for this input
            dqlinear_name = value_name + "_DequantizeLinear"
            dqlinear_node = self.model.find_node_by_name(dqlinear_name, self.new_nodes, self.model.graph())
            if dqlinear_node is None:
                dqlinear_inputs = [
                    quantized_value.q_name,
                    quantized_value.scale_name,
                    quantized_value.zp_name,
                ]
                dequantize_node = onnx.helper.make_node(
                    "DequantizeLinear", dqlinear_inputs, [value_name], dqlinear_name
                )
                return dequantize_node
            else:
                # DQ op is already present, assert it's output matches the input of current node
                assert value_name == dqlinear_node.output[0]
        return None

    def _dequantize_outputs(self):
        """
        Dequantize output if it is quantized
            parameter new_nodes_list: List of new nodes created before processing current node
            return: List of new nodes created
        """

        for output in self.model.graph().output:
            dequantize_node = self._dequantize_value(output.name)
            if dequantize_node is not None:
                self.new_nodes.append(dequantize_node)

    def calculate_quantization_params(self):
        if self.tensors_range is None:
            return

        # adjust tensor_ranges for input of Clip and Relu node
        for node in self.model.nodes():
            if node.op_type not in ["Clip", "Relu"]:
                continue
            if self.is_activation_symmetric:
                continue
            if not self.should_quantize_node(node):
                continue
            if len(self.model.input_name_to_nodes()[node.input[0]]) != 1:
                continue
            if node.input[0] not in self.tensors_range.keys() or node.output[0] not in self.tensors_range.keys():
                continue
            self.tensors_range[node.input[0]] = self.tensors_range[node.output[0]]
        quantization_params = {}
        for tensor_name in self.tensors_range.keys():
            rmin, rmax = self.tensors_range[tensor_name]
            qmin, qmax = get_qmin_qmax_for_qType(self.activation_qType, symmetric=self.is_activation_symmetric)

            quantization_params[tensor_name] = compute_scale_zp(rmin, rmax, qmin, qmax, self.is_activation_symmetric)

        return quantization_params
