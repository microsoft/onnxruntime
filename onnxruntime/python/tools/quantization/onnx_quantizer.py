# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import logging

import onnx
import onnx.numpy_helper
from onnx import onnx_pb as onnx_proto

try:
    from onnx.reference.op_run import to_array_extended
except ImportError:
    # old version of onnx.
    to_array_extended = None

from .base_quantizer import BaseQuantizer
from .onnx_model import ONNXModel
from .quant_utils import (
    TENSOR_NAME_QUANT_SUFFIX,
    QuantizationMode,
    QuantizedValue,
    __producer__,
    __version__,
    add_infer_metadata,
    attribute_to_kwarg,
    find_by_name,
    get_qrange_for_qType,
    ms_domain,
    save_and_reload_model_with_shape_infer,
)
from .registry import CreateOpQuantizer


class ONNXQuantizer(BaseQuantizer):
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
        BaseQuantizer.__init__(
            self,
            model,
            per_channel,
            reduce_range,
            weight_qType,
            activation_qType,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            extra_options,
        )

        if not static:
            self.model.replace_gemm_with_matmul()
            # We need to update value_infos.
            model = save_and_reload_model_with_shape_infer(self.model.model)
            self.value_infos = {vi.name: vi for vi in model.graph.value_info}
            self.value_infos.update({ot.name: ot for ot in model.graph.output})
            self.value_infos.update({it.name: it for it in model.graph.input})
            self.model = ONNXModel(model)

        self.mode = mode  # QuantizationMode.Value
        self.static = static  # use static quantization for inputs.
        self.fuse_dynamic_quant = self.opset_version > 10

        self.q_matmul_const_b_only = "MatMulConstBOnly" in self.extra_options and self.extra_options["MatMulConstBOnly"]
        self.new_nodes = []
        self.graph_scope = "/"  # for human readable debug information
        self.tensor_names = {}  # in case the shape inference not totally working
        self.tensor_names.update({ot.name: 1 for ot in model.graph.output})
        self.tensor_names.update({it.name: 1 for it in model.graph.input})
        for node in self.model.model.graph.node:
            self.tensor_names.update({output_name: 1 for output_name in node.output})

        if self.mode not in QuantizationMode:
            raise ValueError(f"unsupported quantization mode {self.mode}")

        # QuantizeRange tensor name and zero tensor name for scale and zero point calculation.
        # Used when static is False
        self.fixed_qrange_uint8_name = "fixed_quantization_range_uint8"
        self.fixed_qrange_int8_name = "fixed_quantization_range_int8"
        # For uint8 data-type, to compute zero point, we subtract rmin from 0 (represented by fixed_zero_name tensor)
        self.fixed_zero_name = "fixed_zero"
        # For int8 data-type, zero point is always zero (respresented by fixed_zero_point_name tensor)
        self.fixed_zero_zp_name = "fixed_zero_zp"

        # some output from nodes will be quantized, yet itself should be treat as existing so
        # no dequantized will be applied when needed later
        self.generated_value_names = self.model.get_non_initializer_inputs()

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
        sub_quantizer = ONNXQuantizer(
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
        sub_quantizer.parent = self
        sub_quantizer.graph_scope = f"{self.graph_scope}{graph_key}/"
        sub_quantizer.quantize_model()
        return sub_quantizer.model.model.graph

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
        node_name = node.name if node.name else f"{node.op_type}_node_count_{len(self.new_nodes)}"
        kwargs = {}
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                kv = {attr.name: self.quantize_subgraph(attr.g, f"{node_name}:{attr.name}")}
            elif attr.type == onnx.AttributeProto.GRAPHS:
                value = []
                for subgraph in attr.graphs:
                    value.extend(
                        [
                            self.quantize_subgraph(
                                subgraph,
                                f"{node_name}:{attr.name}:{len(value)}",
                            )
                        ]
                    )
                kv = {attr.name: value}
            else:
                kv = attribute_to_kwarg(attr)
            kwargs.update(kv)
        return onnx.helper.make_node(node.op_type, node.input, node.output, name=node.name, **kwargs)

    def has_QDQ_nodes(self):  # noqa: N802
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
                "Please check if the model is already quantized. "
                "Note you don't need to quantize a QAT model. OnnxRuntime support to run QAT model directly."
            )

        for node in self.model.nodes():
            # quantize subgraphes if have
            if self.enable_subgraph_quantization:
                node = self.quantize_node_with_sub_graph(node)  # noqa: PLW2901

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
        # Add ms domain if needed
        ms_opset = [opset for opset in self.model.model.opset_import if opset.domain == ms_domain]
        if not ms_opset:
            ms_nodes = [node for node in self.new_nodes if node.domain == "com.microsoft"]
            if ms_nodes:
                opset = self.model.model.opset_import.add()
                opset.version = 1
                opset.domain = ms_domain

        return self.model.model

    def _get_default_tensor_type(self, tensor_name):
        if "DefaultTensorType" in self.extra_options:
            logging.info(
                "get_tensor_type returns DefaultTensorType for tensor name %r, use %d",
                tensor_name,
                self.extra_options["DefaultTensorType"],
            )
            return self.extra_options["DefaultTensorType"]
        raise RuntimeError(
            f"Unable to find data type for weight_name={tensor_name!r}. "
            f"shape_inference failed to return a type probably this node is "
            f"from a different domain or using an input produced by such an operator. "
            f"This may happen if you quantize a model already quantized. "
            f"You may use extra_options `DefaultTensorType` to indicate "
            f"the default weight type, usually `onnx.TensorProto.FLOAT`."
        )

    def get_tensor_type(self, tensor_name, mandatory=False):
        weight = find_by_name(tensor_name, self.model.initializer())
        if weight is not None:
            return weight.data_type
        if tensor_name in self.value_infos:
            vi = self.value_infos[tensor_name]
            if vi.type.HasField("tensor_type"):
                if mandatory and vi.type.tensor_type.elem_type == 0:
                    return self._get_default_tensor_type(tensor_name)
                return vi.type.tensor_type.elem_type
        if (not self.enable_subgraph_quantization) or (self.parent is None):
            if mandatory:
                return self._get_default_tensor_type(tensor_name)
            return None
        otype = self.parent.is_valid_quantize_weight(tensor_name)
        if otype is not None:
            return otype
        if self.enable_subgraph_quantization and self.parent:
            res = self.parent.get_tensor_type(tensor_name)
            if res is not None:
                return res
        if mandatory:
            return self._get_default_tensor_type(tensor_name)
        return None

    def is_float_tensor(self, tensor_name):
        if self.is_input_a_initializer(tensor_name):
            return self.is_valid_quantize_weight(tensor_name)

        if tensor_name in self.value_infos:
            vi = self.value_infos[tensor_name]
            if vi.type.HasField("tensor_type") and vi.type.tensor_type.elem_type in (
                onnx_proto.TensorProto.FLOAT,
                onnx_proto.TensorProto.FLOAT16,
            ):
                return True
            logging.warning(
                f"Inference failed or unsupported type to quantize for tensor {tensor_name!r}, type is {vi.type}."
            )
            return False

        if self.enable_subgraph_quantization and self.parent:
            return self.parent.is_float_tensor(tensor_name)

        logging.warning(
            f"Failed to infer data type of tensor: {tensor_name!r}. Please add data type info for this tensor "
            f"if your model has customized operators."
        )
        return False

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
        if qType == onnx_proto.TensorProto.UINT8:
            return self._get_dynamic_input_quantization_params_uint8(input_name, nodes_list)
        if qType == onnx_proto.TensorProto.FLOAT8E4M3FN:
            return self._get_dynamic_input_quantization_params_float8e4m3fn(input_name, nodes_list)
        raise ValueError(f"Unexpected value for qType={qType}.")

    def _get_dynamic_input_quantization_params_int8(self, input_name, nodes_list, initial_type):
        """
        Create nodes for dynamic quantization of input to int8 and add them to nodes_list
            parameter input_name: Name of the input.
            parameter nodes_list: new nodes are appended to this list.
            parameter initial_type: initial weight type (FLOAT or FLOAT16)
            return: scale_name, zero_point_name, scale_shape, zero_point_shape.
        """
        qType = onnx_proto.TensorProto.INT8  # noqa: N806

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
            initial_type,
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

    def _get_dynamic_input_quantization_params_uint8(self, input_name, nodes_list, initial_type):
        """
        Create nodes for dynamic quantization of input to uint8 and add them to nodes_list
            parameter input_name: Name of the input.
            parameter nodes_list: new nodes are appended to this list.
            parameter initial_type: initial weight type (FLAOT or FLOAT16)
            return: scale_name, zero_point_name, scale_shape, zero_point_shape.
        """
        qType = onnx_proto.TensorProto.UINT8  # noqa: N806
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
            initial_type,
            [],
            [get_qrange_for_qType(qType)],
        )
        self.model.add_initializer(initializer_qrange)
        initializer_qvalue = onnx.helper.make_tensor(self.fixed_zero_name, initial_type, [], [0.0])
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
        assert input_name != "", "Cannot access undefined variable in graph."
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
        return [*nodes, qlinear_node]

    def find_quantized_value(self, input_name):
        if input_name in self.quantized_value_map:
            return self.quantized_value_map[input_name]
        if self.parent is not None:
            return self.parent.find_quantized_value(input_name)
        return None

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
            # adding this for case embed_layernorm.py has optional segment_embedding
            if not node_input:
                quantized_input_names.append("")
                scale_names.append("")
                zero_point_names.append("")
                continue
            # Quantize the input
            initializer = find_by_name(node_input, self.model.initializer())
            if initializer is not None:
                if self.per_channel and op_level_per_channel:
                    (
                        q_weight_name,
                        zp_name,
                        scale_name,
                    ) = self.quantize_weight_per_channel(
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
                raise ValueError(f"Invalid tensor name to quantize: {node_input} @graph scope{self.graph_scope}")

        return quantized_input_names, zero_point_names, scale_names, nodes

    def _dequantize_value(self, value_name):
        """
        Given a value (input/output) which is quantized, add a DequantizeLinear node to dequantize
        it back to float32 or float16
            parameter value_name: value to dequantize
            parameter new_nodes_list: List of new nodes created before processing current node
            return: None if there is already a DequantizeLinear node that dequantizes it
                    A DequantizeLinear node otherwise
        """
        if (value_name in self.quantized_value_map) and (value_name not in self.generated_value_names):
            quantized_value = self.quantized_value_map[value_name]
            # Add DequantizeLinear Node for this input

            scale_init = find_by_name(quantized_value.scale_name, self.model.initializer())

            # In case we are working with subgraphs, the graph `producer_name` is set to `"onnx-quantizer"` in the `quantize_subgraph` method. In this case, the scale initializer may be on the top level graph, so the check below can not be done.
            if self.model.model.producer_name != "onnx-quantizer" or (
                self.model.model.producer_name == "onnx-quantizer" and scale_init is not None
            ):
                # axis is not specified so scale_init must be a scalar.
                assert onnx.numpy_helper.to_array(scale_init).size == 1

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
