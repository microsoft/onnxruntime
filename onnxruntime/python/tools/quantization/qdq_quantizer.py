# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import logging
from enum import Enum

import onnx
import onnx.helper
import onnx.numpy_helper
from onnx import TensorProto
from onnx import onnx_pb as onnx_proto
from onnx.helper import make_function

from .onnx_quantizer import ONNXQuantizer
from .quant_utils import (
    DEQUANT_OP_NAME,
    QUANT_OP_NAME,
    QuantizedValue,
    QuantizedValueType,
    __producer__,
    __version__,
    add_dequant_output_suffix,
    add_dequant_suffix,
    add_quant_input_suffix,
    add_quant_output_suffix,
    add_quant_suffix,
    find_by_name,
    get_qmin_qmax_for_qType,
    get_qrange_for_qType,
)
from .registry import CreateQDQQuantizer


class QDQQuantTensorType(Enum):
    ACTIVATION = 0
    WEIGHT = 1
    BIAS = 2


class QDQTensorQuantInfo:
    def __init__(self, tensor_type=QDQQuantTensorType.ACTIVATION, quant_para_provider=None, axis=None):
        self.tensor_type = tensor_type
        self.quant_para_provider = quant_para_provider
        self.axis = axis
        self.is_shared = quant_para_provider is not None


class QDQQuantizer(ONNXQuantizer):
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
        ONNXQuantizer.__init__(
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
            extra_options,
        )
        self.tensors_to_quantize = {}
        self.bias_to_quantize = []

        self.nodes_to_remove = []
        # Specific op types to exclude qdq quantization for their outputs.
        # In TRT, it's not recommended to quantize outputs for weighted ops such as Conv, Matmul, Gemm
        # because those ops may be followed by nodes that require high resolution inputs.
        # Adding QDQ for those ops' output may end up with worse accuracy.
        # So, we don't recommend to add QDQ to node's output under such condition.
        if extra_options != None and "OpTypesToExcludeOutputQuantization" in extra_options.keys():
            self.op_types_to_exclude_output_quantization = extra_options["OpTypesToExcludeOutputQuantization"]
        else:
            self.op_types_to_exclude_output_quantization = (
                []
                if static
                else [
                    "Conv",
                    "Matmul",
                    "MatMul",
                    "Gemm",
                    "Attention",
                    "LSTM",
                ]
            )

        # We do quantization on Dequantizelinear's input to remove Quantizelinear for weight as an optimization.
        # In some cases, for example QDQ BERT model for TensorRT, QDQ should always appear as a pair.
        # Therefore, we need to disable this optimization and add qdq pair to weight.
        self.add_qdq_pair_to_weight = (
            False if "AddQDQPairToWeight" not in extra_options else extra_options["AddQDQPairToWeight"]
        )

        # The default behavior is that multiple nodes can share a QDQ pair as their inputs.
        # In TRT, QDQ pair can’t be shared between nodes, so it will create dedicated QDQ pairs for each node.
        self.dedicated_qdq_pair = (
            False if "DedicatedQDQPair" not in extra_options else extra_options["DedicatedQDQPair"]
        )
        if self.dedicated_qdq_pair:
            self.tensor_to_its_receiving_nodes = {}

        # Let user set channel axis for specific op type and it's effective only when per channel quantization is supported and per_channel is True.
        self.qdq_op_type_per_channel_support_to_axis = (
            {}
            if "QDQOpTypePerChannelSupportToAxis" not in extra_options
            else extra_options["QDQOpTypePerChannelSupportToAxis"]
        )

        # Register the chosen dynamic subgraph compute quantization parameter function based on symmetric and qtype
        if not static:
            if self.is_activation_symmetric:
                self.compute_quantization_parameters_function = self.create_dynamic_symmetric_subgraph_function(
                    self.activation_qType
                )
            else:
                self.compute_quantization_parameters_function = self.create_dynamic_asymmetric_subgraph_function(
                    self.activation_qType
                )
            self.model.model.functions.append(self.compute_quantization_parameters_function)

    def _is_tensor_quantizable(self, tensor_name):
        """
        Check if tensor can be quantized
        """
        weight = find_by_name(tensor_name, self.model.initializer())
        if weight is not None:
            if weight.data_type == onnx_proto.TensorProto.FLOAT:
                return True
        elif tensor_name in self.value_infos.keys():
            vi = self.value_infos[tensor_name]
            if vi.type.HasField("tensor_type") and vi.type.tensor_type.elem_type == TensorProto.FLOAT:
                return True
        else:
            logging.warning(
                "failed to infer the type of tensor: {}. Skip to quantize it. Please check if it is expected.".format(
                    tensor_name
                )
            )
        return False

    def __quantize_tensor(self, tensor_name, quant_sharing_param=None, tensor_type=QDQQuantTensorType.ACTIVATION):
        """
        Quantize tensors. If quant_param_tensor is not None, tensor with name tensor_name will be quantized with same
        quantization parameters as tensor quant_param_tensor

        Args:
            tensor_name: name of the tensor to quantize
            quant_sharing_param: name of the tensor that provides quantization parameter
            tensor_type: QDQQuantTensorType default ACTIVATION
        """
        if self._is_tensor_quantizable(tensor_name):
            if quant_sharing_param:
                self.tensors_to_quantize[tensor_name] = QDQTensorQuantInfo(
                    tensor_type=tensor_type, quant_para_provider=quant_sharing_param
                )
            elif tensor_name not in self.tensors_to_quantize:
                self.tensors_to_quantize[tensor_name] = QDQTensorQuantInfo(tensor_type=tensor_type)

    def quantize_activation_tensor(self, tensor_name, quant_sharing_param=None):
        """
        Quantize Activation Tensor
        Args:
            tensor_name: name of the tensor to quantize
            quant_sharing_param: name of the tensor that provides quantization parameter

        """
        return self.__quantize_tensor(tensor_name, quant_sharing_param, QDQQuantTensorType.ACTIVATION)

    def quantize_weight_tensor(self, tensor_name, quant_sharing_param=None):
        """
        Quantize Weight Tensor
        Args:
            tensor_name: name of the tensor to quantize
            quant_sharing_param: name of the tensor that provides quantization parameter

        """
        return self.__quantize_tensor(tensor_name, quant_sharing_param, QDQQuantTensorType.WEIGHT)

    def quantize_weight_tensor_per_channel(self, tensor_name, axis):
        weight = find_by_name(tensor_name, self.model.initializer())
        if weight:
            if weight.data_type == onnx_proto.TensorProto.FLOAT:
                self.tensors_to_quantize[tensor_name] = QDQTensorQuantInfo(
                    tensor_type=QDQQuantTensorType.WEIGHT, axis=axis
                )
        else:
            logging.warning(
                "only support per-channel quantization on weight. Tensor: {} is not quantized.".format(tensor_name)
            )

    def quantize_bias_tensor(self, bias_name, input_name, weight_name, beta=1.0):
        weight = find_by_name(bias_name, self.model.initializer())
        if weight is not None:
            if weight.data_type == onnx_proto.TensorProto.FLOAT:
                self.bias_to_quantize.append((bias_name, input_name, weight_name, beta))
        else:
            logging.warning("Expected {} to be a weight".format(bias_name))

    def remove_node(self, node):
        self.nodes_to_remove.append(node)

    def remove_nodes(self):
        self.model.remove_nodes(self.nodes_to_remove)

    def quantize_model(self):
        for node in self.model.nodes():
            if self.should_quantize_node(node):
                op_quantizer = CreateQDQQuantizer(self, node)
                if op_quantizer is None:
                    continue  # Skip quantize if no quantizer returned
                op_quantizer.quantize()

                if self.dedicated_qdq_pair:
                    for tensor_name in node.input:
                        if tensor_name not in self.tensor_to_its_receiving_nodes:
                            self.tensor_to_its_receiving_nodes[tensor_name] = []
                        self.tensor_to_its_receiving_nodes[tensor_name].append(node)
        self._quantize_normal_tensors()
        self._quantize_sharing_param_tensors()
        self._quantize_bias_tensors()
        self.remove_nodes()
        if not self.add_qdq_pair_to_weight:
            self.model.clean_initializers()
        self.model.model.producer_name = __producer__
        self.model.model.producer_version = __version__

        return self.model.model

    def try_replacing_upstream_output(self, upstream_output_name, output_name):
        if (
            output_name in self.quantization_params.keys()
            and len(self.model.input_name_to_nodes()[upstream_output_name]) == 1
            and not self.model.is_graph_output(upstream_output_name)
            and not self.model.is_graph_input(upstream_output_name)
        ):
            self.model.replace_output_of_all_nodes(upstream_output_name, output_name)
            if upstream_output_name in self.tensors_to_quantize:
                del self.tensors_to_quantize[upstream_output_name]
            return True
        return False

    def create_dynamic_subgraph(self, input_name):
        """
        Create nodes for dynamic quantization of input and add them to nodes_list.
            parameter input_name: Name of the input.
            parameter nodes_list: new nodes are appended to this list.
            parameter qType: type to quantize to.
            return: scale_name, zero_point_name, scale_shape, zero_point_shape.
        """
        input_scale_name = input_name + "_scale"
        input_zp_name = input_name + "_zp"
        inputs = [input_name]
        compute_quant_param_node = onnx.helper.make_node(
            "ComputeQuantizationParameters",
            inputs,
            [input_scale_name, input_zp_name],
            input_name + "_ComputeQuantizationParameters",
            domain=self.compute_quantization_parameters_function.domain,
        )
        return input_scale_name, input_zp_name, [], [], compute_quant_param_node

    def create_dynamic_symmetric_subgraph_function(self, qType):
        """
        Create nodes for dynamic symmetric quantization of input and add them to nodes_list
            parameter qType: UInt8 or Int8.
            parameter symmetric: is scale and zp calculation symmetric?
            return: compute quantization parameters function.

        scale = max(abs(rmin), abs(rmax)) / (qrange / 2)
        zp = (qmax + qmin) / 2
        """
        input_name = "compute_quantization_parameters"
        qrange_name = (
            self.fixed_qrange_int8_name if qType == onnx_proto.TensorProto.INT8 else self.fixed_qrange_uint8_name
        )
        input_scale_name = input_name + "_scale"
        input_zp_name = input_name + "_zp"
        nodes_list = []

        # Create Constant tensors instead of initializers
        qrange_name = input_name + "_" + qrange_name
        qrange_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=[qrange_name],
            value=onnx.helper.make_tensor(
                name=input_name + "_init_" + qrange_name,
                data_type=onnx.TensorProto.FLOAT,
                dims=[],
                vals=[get_qrange_for_qType(qType, reduce_range=self.reduce_range, symmetric=True) / 2.0],
            ),
            name=qrange_name,
        )
        nodes_list.append(qrange_node)
        fixed_zero_zp_name = input_name + "_" + self.fixed_zero_zp_name
        qmin, qmax = get_qmin_qmax_for_qType(qType, reduce_range=self.reduce_range, symmetric=True)
        zp = int((qmin + qmax) / 2)
        fixed_zero_zp_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=[fixed_zero_zp_name],
            value=onnx.helper.make_tensor(
                name=input_name + "_init_" + self.fixed_zero_zp_name,
                data_type=qType,
                dims=[],
                vals=[zp],
            ),
            name=fixed_zero_zp_name,
        )
        nodes_list.append(fixed_zero_zp_node)

        # Reduce Min and Reduce Max
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
        scale_div_name = input_name + "scale_Div"
        scale_div_node = onnx.helper.make_node(
            "Div",
            [abs_max_node.output[0], qrange_name],
            [input_scale_name],
            scale_div_name,
        )
        nodes_list.append(scale_div_node)

        # # Zero point Cast to integer 8
        zp_cast_name = input_name + "_zero_point_Cast"
        zp_cast_node = onnx.helper.make_node(
            "Cast", [fixed_zero_zp_name], [input_zp_name], zp_cast_name, to=qType
        )  # TODO recast zp as int32 to avoid underflow...
        nodes_list.append(zp_cast_node)

        # Create function op
        func_domain = "com.microsoft"
        func_opset_imports = [onnx.helper.make_opsetid("", self.opset_version)]
        self.model.model.opset_import.extend(
            [onnx.helper.make_opsetid("", self.opset_version)]
        )  # , onnx.helper.make_opsetid(func_domain, 1)])
        return make_function(
            func_domain,  # TODO: What domain
            "ComputeQuantizationParameters",
            [input_name],
            [input_scale_name, input_zp_name],
            nodes_list,
            func_opset_imports,
        )

    def create_dynamic_asymmetric_subgraph_function(self, qType):
        """
        Create nodes for dynamic asymmetric quantization of input and add them to nodes_list
            parameter qType: UInt8 or Int8.
            parameter symmetric: is scale and zp calculation symmetric?
            return: compute quantization parameters function.

        scale = (rmax - rmin) / qrange
        zp = round(qmin - (rmin / scale))
        """
        input_name = "compute_quantization_parameters"
        # Reduce min and Reduce max
        input_scale_name = input_name + "_scale"
        input_zp_name = input_name + "_zero_point"
        qrange_name = (
            self.fixed_qrange_int8_name if qType == onnx_proto.TensorProto.INT8 else self.fixed_qrange_uint8_name
        )
        nodes_list = []

        # Create Constant tensors instead of initializers
        qrange_name = input_name + "_" + qrange_name
        qrange_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=[qrange_name],
            value=onnx.helper.make_tensor(
                name=input_name + "_init_" + qrange_name,
                data_type=onnx.TensorProto.FLOAT,
                dims=[],
                vals=[get_qrange_for_qType(qType, reduce_range=self.reduce_range, symmetric=False)],
            ),
            name=qrange_name,
        )
        nodes_list.append(qrange_node)
        fixed_zero_name = input_name + "_" + self.fixed_zero_name
        fixed_zero_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=[fixed_zero_name],
            value=onnx.helper.make_tensor(
                name=input_name + "_init_" + self.fixed_zero_name,
                data_type=onnx.TensorProto.FLOAT,
                dims=[],
                vals=[0.0],
            ),
            name=fixed_zero_name,
        )
        nodes_list.append(fixed_zero_node)
        qmin, qmax = get_qmin_qmax_for_qType(qType, reduce_range=self.reduce_range, symmetric=False)
        fixed_qmin_name = input_name + "_" + qrange_name
        fixed_qmin_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=[fixed_qmin_name],
            value=onnx.helper.make_tensor(
                name=input_name + "_init_" + fixed_qmin_name,
                data_type=onnx.TensorProto.FLOAT,
                dims=[],
                vals=[qmin],
            ),
            name=fixed_qmin_name,
        )
        nodes_list.append(fixed_qmin_node)

        # Reduce Min and Reduce Max
        reduce_min_name = input_name + "_ReduceMin"
        reduce_min_node = onnx.helper.make_node(
            "ReduceMin",
            [input_name],
            [reduce_min_name + ":0"],
            reduce_min_name,
            keepdims=0,
        )
        nodes_list.append(reduce_min_node)

        zero_min_name = input_name + "_Min"
        zero_min_node = onnx.helper.make_node(
            "Min", [reduce_min_name + ":0", fixed_zero_name], [zero_min_name + ":0"], zero_min_name
        )
        nodes_list.append(zero_min_node)

        reduce_max_name = input_name + "_ReduceMax"
        reduce_max_node = onnx.helper.make_node(
            "ReduceMax",
            [input_name],
            [reduce_max_name + ":0"],
            reduce_max_name,
            keepdims=0,
        )
        nodes_list.append(reduce_max_node)

        zero_max_name = input_name + "_Max"
        zero_max_node = onnx.helper.make_node(
            "Max", [reduce_max_name + ":0", fixed_zero_name], [zero_max_name + ":0"], zero_max_name
        )
        nodes_list.append(zero_max_node)

        # Compute Scale
        #   Subtract rmax and rmin
        scale_sub_name = input_name + "_scale_Sub"
        scale_sub_node = onnx.helper.make_node(
            "Sub",
            [zero_max_node.output[0], zero_min_node.output[0]],
            [scale_sub_name + ":0"],
            scale_sub_name,
        )
        nodes_list.append(scale_sub_node)
        #   and divide by quantize range
        scale_div_name = input_name + "_scale_Div"
        scale_div_node = onnx.helper.make_node(
            "Div",
            [scale_sub_node.output[0], qrange_name],
            [input_scale_name],
            scale_div_name,
        )
        nodes_list.append(scale_div_node)

        # Divide rmin by scale
        zp_div_name = input_name + "_zero_point_Div"
        zp_div_node = onnx.helper.make_node(
            "Div",
            [zero_min_node.output[0], input_scale_name],
            [zp_div_name + ":0"],
            zp_div_name,
        )
        nodes_list.append(zp_div_node)
        # Compute zero point
        #   Subtract zero and rmin/scale
        zp_sub_name = input_name + "_zero_point_Sub"
        zp_sub_node = onnx.helper.make_node(
            "Sub",
            [fixed_qmin_name, zp_div_node.output[0]],
            [zp_sub_name + ":0"],
            zp_sub_name,
        )
        nodes_list.append(zp_sub_node)

        # Compute round
        zp_round_name = input_name + "_zero_point_Round"
        zp_round_node = onnx.helper.make_node("Round", zp_sub_node.output, [zp_round_name + ":0"], zp_round_name)
        nodes_list.append(zp_round_node)
        # Cast to integer
        zp_cast_name = input_name + "_zero_point_Cast"
        zp_cast_node = onnx.helper.make_node(
            "Cast", zp_round_node.output, [input_zp_name], zp_cast_name, to=qType
        )  # TODO recast zp as int32 to avoid underflow...
        nodes_list.append(zp_cast_node)
        # Create function op
        func_domain = "com.microsoft"
        func_opset_imports = [onnx.helper.make_opsetid("", self.opset_version)]
        self.model.model.opset_import.extend(
            [onnx.helper.make_opsetid("", self.opset_version)]
        )  # , onnx.helper.make_opsetid(func_domain, 1)])
        return make_function(
            func_domain,  # TODO: What domain
            "ComputeQuantizationParameters",
            [input_name],
            [input_scale_name, input_zp_name],
            nodes_list,
            func_opset_imports,
        )

    def _create_qdq_nodes(
        self, q_input, q_output, quant_node_name, dq_input, dq_output, dequant_node_name, scale_name, zp_name, axis=None
    ):
        qlinear_node = onnx.helper.make_node(
            QUANT_OP_NAME,
            [q_input, scale_name, zp_name],
            [q_output],
            quant_node_name,
            axis=axis,
        )
        dequant_node = onnx.helper.make_node(
            DEQUANT_OP_NAME,
            [dq_input, scale_name, zp_name],
            [dq_output],
            dequant_node_name,
            axis=axis,
        )
        self.model.add_nodes([qlinear_node, dequant_node])

    def _add_qdq_pair_for_initializer(self, weight_proto, tensor_type, axis=None):
        weight_name = weight_proto.name
        if axis is not None:
            if self.opset_version < 13:
                raise ValueError("Per-Channel support with QDQ format requires onnx opset version 13 or above.")
            q_weight_name, zp_name, scale_name = self.quantize_weight_per_channel(
                weight_name, onnx_proto.TensorProto.INT8, axis, keep_float_weight=self.add_qdq_pair_to_weight
            )
        else:
            q_weight_name, zp_name, scale_name = self.quantize_initializer(
                weight_proto,
                self.weight_qType if tensor_type is QDQQuantTensorType.WEIGHT else self.activation_qType,
                keep_float_weight=self.add_qdq_pair_to_weight,
            )

        weight_dequant_output = add_dequant_output_suffix(weight_name)
        self.model.replace_input_of_all_nodes(weight_name, weight_dequant_output)
        if self.add_qdq_pair_to_weight:
            weight_quant_output = add_quant_output_suffix(weight_name)

            self._create_qdq_nodes(
                weight_name,
                weight_quant_output,
                add_quant_suffix(weight_name),
                weight_quant_output,
                weight_dequant_output,
                add_dequant_suffix(weight_name),
                scale_name,
                zp_name,
                axis,
            )
        else:
            dequant_node = onnx.helper.make_node(
                DEQUANT_OP_NAME,
                [q_weight_name, scale_name, zp_name],
                [weight_dequant_output],
                add_dequant_suffix(weight_name),
                axis=axis,
            )
            self.model.add_node(dequant_node)

    def _add_qdq_pair_for_activation(self, tensor_name, scale_name, zp_name):
        if (
            self.dedicated_qdq_pair
            and tensor_name in self.tensor_to_its_receiving_nodes
            and len(self.tensor_to_its_receiving_nodes[tensor_name]) > 1
        ):
            num_dedicated_qdq_pair = len(self.tensor_to_its_receiving_nodes[tensor_name])
            for i in range(num_dedicated_qdq_pair):
                postfix = f"_{i + 1}"
                tensor_name_quant_output_postfix = add_quant_output_suffix(tensor_name) + postfix
                tensor_name_dequant_output_postfix = add_dequant_output_suffix(tensor_name) + postfix
                self._create_qdq_nodes(
                    tensor_name,
                    tensor_name_quant_output_postfix,
                    add_quant_suffix(tensor_name),
                    tensor_name_quant_output_postfix,
                    tensor_name_dequant_output_postfix,
                    add_dequant_suffix(tensor_name),
                    scale_name,
                    zp_name,
                )

                node = self.tensor_to_its_receiving_nodes[tensor_name][i]
                self.model.replace_node_input(node, tensor_name, tensor_name_dequant_output_postfix)
                if i == 0:
                    quantized_value = QuantizedValue(
                        tensor_name,
                        tensor_name_dequant_output_postfix,
                        scale_name,
                        zp_name,
                        QuantizedValueType.Input,
                    )
                    self.quantized_value_map[tensor_name] = quantized_value
        else:
            q_input = tensor_name
            dq_output = add_dequant_output_suffix(tensor_name)
            if self.model.is_graph_output(tensor_name):
                q_input = add_quant_input_suffix(tensor_name)
                dq_output = tensor_name
                self.model.replace_output_of_all_nodes(tensor_name, q_input)
            else:
                self.model.replace_input_of_all_nodes(tensor_name, dq_output)

            self._create_qdq_nodes(
                q_input,
                add_quant_output_suffix(tensor_name),
                add_quant_suffix(tensor_name),
                add_quant_output_suffix(tensor_name),
                dq_output,
                add_dequant_suffix(tensor_name),
                scale_name,
                zp_name,
            )

            quantized_value = QuantizedValue(
                tensor_name,
                dq_output,
                scale_name,
                zp_name,
                QuantizedValueType.Input,
            )
            self.quantized_value_map[tensor_name] = quantized_value

    def _quantize_normal_tensors(self):
        for tensor_name, tensor_info in self.tensors_to_quantize.copy().items():
            if tensor_name in self.quantized_value_map.keys():
                continue

            if not tensor_info.is_shared:
                # Quantize the input
                initializer = find_by_name(tensor_name, self.model.initializer())
                if initializer:
                    self._add_qdq_pair_for_initializer(initializer, tensor_info.tensor_type, tensor_info.axis)
                else:
                    used_scale, used_zp = self.find_quant_scale_zp(tensor_name)
                    data_found, scale_name, zp_name, _, _ = self._get_quantization_params(
                        tensor_name, used_scale, used_zp
                    )
                    cqp_node = None
                    if data_found == False:
                        if self.static:
                            raise ValueError(
                                "Quantization parameters are not specified for param {}."
                                "In static mode quantization params for inputs and outputs of nodes to be quantized are required.".format(
                                    tensor_name
                                )
                            )
                        # Here we add dynamic subgraph, if we found no static params
                        # Scale and Zero Points not available for this input. Add nodes to dynamically compute it
                        if self.model.is_graph_output(tensor_name):  # Changes name to quantize output correctly
                            (
                                scale_name,
                                zp_name,
                                scale_shape,
                                zp_shape,
                                cqp_node,
                            ) = self.create_dynamic_subgraph(add_quant_input_suffix(tensor_name))
                        else:
                            (
                                scale_name,
                                zp_name,
                                scale_shape,
                                zp_shape,
                                cqp_node,
                            ) = self.create_dynamic_subgraph(tensor_name)

                    self._add_qdq_pair_for_activation(tensor_name, scale_name, zp_name)
                    if cqp_node != None:
                        self.model.add_node(cqp_node)

                del self.tensors_to_quantize[tensor_name]

    def _quantize_sharing_param_tensors(self):
        while self.tensors_to_quantize:
            for tensor_name, tensor_info in self.tensors_to_quantize.copy().items():
                tensor_provider_name = tensor_info.quant_para_provider
                if tensor_provider_name in self.quantized_value_map:
                    del self.tensors_to_quantize[tensor_name]
                    quantized_value = self.quantized_value_map[tensor_provider_name]
                    # Quantize the input
                    initializer = find_by_name(tensor_name, self.model.initializer())
                    if initializer is not None:
                        raise ValueError("Quantization parameter shared mode is not supported for weight yet")
                    self._add_qdq_pair_for_activation(tensor_name, quantized_value.scale_name, quantized_value.zp_name)

    def _quantize_bias_tensors(self):
        for bias_name, input_name, weight_name, beta in self.bias_to_quantize:
            if bias_name in self.quantized_value_map.keys():
                continue
            # Quantize the input
            # TODO: check if we have an input_scale initializer and decide whether to quantize bias static based off of that
            # get scale for input
            input_scale_name = ""
            if input_name in self.quantized_value_map:
                input_scale_name = self.quantized_value_map[input_name].scale_name
            elif input_name in self.quantization_params:
                _, input_scale_name, _, _, _ = self._get_quantization_params(input_name)
            inputscale_initializer = find_by_name(input_scale_name, self.model.initializer())
            if inputscale_initializer is None:
                continue
            self.quantize_bias_static(bias_name, input_name, weight_name, beta)
            self.model.remove_initializer(find_by_name(bias_name, self.model.initializer()))
            quant_value = self.quantized_value_map[bias_name]
            inputs = [quant_value.q_name, quant_value.scale_name, quant_value.zp_name]
            node_name = add_dequant_suffix(bias_name)
            if quant_value.axis is not None:
                dequant_node = onnx.helper.make_node(
                    "DequantizeLinear",
                    inputs,
                    [bias_name],
                    node_name,
                    axis=quant_value.axis,
                )
            else:
                dequant_node = onnx.helper.make_node(
                    "DequantizeLinear",
                    inputs,
                    [bias_name],
                    node_name,
                )
            self.model.add_node(dequant_node)

    def is_tensor_quantized(self, tensor_name):
        return tensor_name in self.tensors_to_quantize or tensor_name in self.bias_to_quantize
