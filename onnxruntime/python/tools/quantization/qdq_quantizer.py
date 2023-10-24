# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import logging
from enum import Enum

import onnx
import onnx.numpy_helper
from onnx import TensorProto
from onnx import onnx_pb as onnx_proto

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
    ms_domain,
)
from .registry import CreateQDQQuantizer


class QDQQuantTensorType(Enum):
    ACTIVATION = 0
    WEIGHT = 1
    BIAS = 2


class QDQTensorQuantInfo:
    def __init__(self, tensor_type=QDQQuantTensorType.ACTIVATION, quant_para_provider=None, axis=None, data_type=None):
        self.tensor_type = tensor_type
        self.quant_para_provider = quant_para_provider
        self.axis = axis
        self.is_shared = quant_para_provider is not None
        assert data_type is not None
        self.data_type = data_type


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
        self.op_types_to_exclude_output_quantization = (
            []
            if "OpTypesToExcludeOutputQuantization" not in extra_options
            else extra_options["OpTypesToExcludeOutputQuantization"]
        )

        # We do quantization on Dequantizelinear's input to remove Quantizelinear for weight as an optimization.
        # In some cases, for example QDQ BERT model for TensorRT, QDQ should always appear as a pair.
        # Therefore, we need to disable this optimization and add qdq pair to weight.
        self.add_qdq_pair_to_weight = (
            False if "AddQDQPairToWeight" not in extra_options else extra_options["AddQDQPairToWeight"]
        )

        # Some scenarios do not need the bias quantized. For example, in the case of Quantization Aware Training,
        # quantizing the bias is not needed. This is because in QAT, all model parameters are expected to be in
        # floating point format. To that end, we can use the FakeQuant operator for weights and activations that
        # can always have QDQ pairs (by using AddQDQPairToWeight). But for biases in a quantized model, we can't use
        # FakeQuant because it only ever appears before a DQ (since it is quantized as int32).
        self.quantize_bias = True if "QuantizeBias" not in extra_options else extra_options["QuantizeBias"]

        # The default behavior is that multiple nodes can share a QDQ pair as their inputs.
        # In TRT, QDQ pair can`t be shared between nodes, so it will create dedicated QDQ pairs for each node.
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

        self.qdq_op_domain = ms_domain if extra_options.get("UseQDQContribOps", False) else None

        # The ONNX spec does not yet support 16-bit Q/DQ ops. So, must override the Q/DQ op domain to 'com.microsoft'
        # if the activation or weight types are 16-bit integers.
        # TODO: Remove this override (and use only the 'UseQDQContribOps' option) if/when ONNX adds 16-bit support.
        int16_types = (TensorProto.UINT16, TensorProto.INT16)
        if not self.qdq_op_domain and (self.activation_qType in int16_types or self.weight_qType in int16_types):
            logging.warning(
                "ONNX QuantizeLinear and DequantizeLinear operators do not support 16-bit integer quantization types. "
                f"The domain of QuantizeLinear and DequantizeLinear operators will be set to '{ms_domain}' to "
                "enable support."
            )
            self.qdq_op_domain = ms_domain

    def _get_tensor_type(self, tensor_name):
        """
        Check if tensor can be quantized
        """
        weight = find_by_name(tensor_name, self.model.initializer())
        if weight is not None:
            return weight.data_type
        elif tensor_name in self.value_infos:
            vi = self.value_infos[tensor_name]
            if vi.type.HasField("tensor_type"):
                return vi.type.tensor_type.elem_type
        return None

    def _is_tensor_quantizable(self, tensor_name):
        """
        Check if tensor can be quantized
        """
        weight = find_by_name(tensor_name, self.model.initializer())
        if weight is not None:
            if weight.data_type in (onnx_proto.TensorProto.FLOAT, onnx_proto.TensorProto.FLOAT16):
                return True
        elif tensor_name in self.value_infos:
            vi = self.value_infos[tensor_name]
            if vi.type.HasField("tensor_type") and vi.type.tensor_type.elem_type in (
                TensorProto.FLOAT,
                TensorProto.FLOAT16,
            ):
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
                data_type = self._get_tensor_type(tensor_name)
                self.tensors_to_quantize[tensor_name] = QDQTensorQuantInfo(
                    tensor_type=tensor_type, quant_para_provider=quant_sharing_param, data_type=data_type
                )
            elif tensor_name not in self.tensors_to_quantize:
                data_type = self._get_tensor_type(tensor_name)
                self.tensors_to_quantize[tensor_name] = QDQTensorQuantInfo(tensor_type=tensor_type, data_type=data_type)

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
            if weight.data_type in (onnx_proto.TensorProto.FLOAT, onnx_proto.TensorProto.FLOAT16):
                self.tensors_to_quantize[tensor_name] = QDQTensorQuantInfo(
                    tensor_type=QDQQuantTensorType.WEIGHT, axis=axis, data_type=weight.data_type
                )
        else:
            logging.warning(f"only support per-channel quantization on weight. Tensor: {tensor_name} is not quantized.")

    def quantize_bias_tensor(self, bias_name, input_name, weight_name, beta=1.0):
        weight = find_by_name(bias_name, self.model.initializer())
        if weight is not None:
            if weight.data_type in (onnx_proto.TensorProto.FLOAT, onnx_proto.TensorProto.FLOAT16):
                self.bias_to_quantize.append((bias_name, input_name, weight_name, beta))
        else:
            logging.warning(f"Expected {bias_name} to be a weight")

    def remove_node(self, node):
        self.nodes_to_remove.append(node)

    def remove_nodes(self):
        self.model.remove_nodes(self.nodes_to_remove)

    def quantize_model(self):
        for node in self.model.nodes():
            if self.should_quantize_node(node):
                op_quantizer = CreateQDQQuantizer(self, node)
                op_quantizer.quantize()

                if self.dedicated_qdq_pair:
                    for tensor_name in node.input:
                        if tensor_name not in self.tensor_to_its_receiving_nodes:
                            self.tensor_to_its_receiving_nodes[tensor_name] = []
                        self.tensor_to_its_receiving_nodes[tensor_name].append(node)

        self._quantize_normal_tensors()
        self._quantize_sharing_param_tensors()
        if self.quantize_bias:
            self._quantize_bias_tensors()
        self.remove_nodes()
        if not self.add_qdq_pair_to_weight:
            self.model.clean_initializers()

        self.model.model.producer_name = __producer__
        self.model.model.producer_version = __version__

        return self.model.model

    def try_replacing_upstream_output(self, upstream_output_name, output_name):
        if (
            output_name in self.quantization_params
            and len(self.model.input_name_to_nodes()[upstream_output_name]) == 1
            and not self.model.is_graph_output(upstream_output_name)
            and not self.model.is_graph_input(upstream_output_name)
        ):
            self.model.replace_output_of_all_nodes(upstream_output_name, output_name)
            if upstream_output_name in self.tensors_to_quantize:
                del self.tensors_to_quantize[upstream_output_name]
            return True
        return False

    def _create_qdq_nodes(
        self, q_input, q_output, quant_node_name, dq_input, dq_output, dequant_node_name, scale_name, zp_name, axis=None
    ):
        qlinear_node = onnx.helper.make_node(
            QUANT_OP_NAME,
            [q_input, scale_name, zp_name],
            [q_output],
            quant_node_name,
            axis=axis,
            domain=self.qdq_op_domain,
        )
        dequant_node = onnx.helper.make_node(
            DEQUANT_OP_NAME,
            [dq_input, scale_name, zp_name],
            [dq_output],
            dequant_node_name,
            axis=axis,
            domain=self.qdq_op_domain,
        )
        self.model.add_nodes([qlinear_node, dequant_node])

    def _add_qdq_pair_for_initializer(self, weight_proto, tensor_type, axis=None):
        weight_name = weight_proto.name
        if axis is not None:
            if self.opset_version < 13:
                raise ValueError("Per-Channel support with QDQ format requires onnx opset version 13 or above.")
            q_weight_name, zp_name, scale_name = self.quantize_weight_per_channel(
                weight_name,
                # Quantization type is forced to be TensorProto.INT8.
                # when the expected value would be (see below)
                # self.weight_qType if tensor_type is QDQQuantTensorType.WEIGHT else self.activation_qType.
                # QLinearConv expects to have a unique value for all channels.
                # This code does not enforce that but it is necessarily the case when the
                # quantization is symmetric (as for INT8).
                onnx_proto.TensorProto.INT8,
                axis,
                keep_float_weight=self.add_qdq_pair_to_weight,
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
                domain=self.qdq_op_domain,
            )
            self.model.add_node(dequant_node)

    def _add_qdq_pair_for_activation(self, tensor_name, scale_name, zp_name, data_type=None):
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
                quant_node_name_postfix = add_quant_suffix(tensor_name) + postfix
                dequant_node_name_postfix = add_dequant_suffix(tensor_name) + postfix
                self._create_qdq_nodes(
                    tensor_name,
                    tensor_name_quant_output_postfix,
                    quant_node_name_postfix,
                    tensor_name_quant_output_postfix,
                    tensor_name_dequant_output_postfix,
                    dequant_node_name_postfix,
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
                        scale_type=data_type,
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
                scale_type=data_type,
            )
            self.quantized_value_map[tensor_name] = quantized_value

    def _quantize_normal_tensors(self):
        for tensor_name, tensor_info in self.tensors_to_quantize.copy().items():
            if tensor_name in self.quantized_value_map:
                continue

            if not tensor_info.is_shared:
                # Quantize the input
                initializer = find_by_name(tensor_name, self.model.initializer())
                if initializer:
                    self._add_qdq_pair_for_initializer(initializer, tensor_info.tensor_type, tensor_info.axis)
                else:
                    used_scale, used_zp = self.find_quant_scale_zp(tensor_name)
                    if used_scale is not None and not hasattr(used_scale, "dtype"):
                        raise TypeError(
                            f"Unexpected type {type(used_scale)} for used_scale and tensor_name={tensor_name!r}"
                        )
                    data_found, scale_name, zp_name, _, _ = self._get_quantization_params(
                        tensor_name, used_scale, used_zp
                    )

                    if not data_found:
                        raise ValueError(
                            f"Quantization parameters are not specified for param {tensor_name}. "
                            "In static mode quantization params for inputs and outputs of nodes to be quantized are required."
                        )

                    self._add_qdq_pair_for_activation(tensor_name, scale_name, zp_name, data_type=tensor_info.data_type)

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
            if bias_name in self.quantized_value_map:
                continue
            # Quantize the input
            self.quantize_bias_static(bias_name, input_name, weight_name, beta)
            init = find_by_name(bias_name, self.model.initializer())
            self.model.remove_initializer(init)
            quant_value = self.quantized_value_map[bias_name]
            if quant_value.node_type == "Cast":
                # simple cast to float 16 and not DequantizeLinear
                # cublasLtMatmul only supports (b)float16, float bias.
                if not isinstance(init.data_type, int):
                    raise TypeError(f"Unexpected type {type(init.data_type)} for input={input_name!r}")
                node_name = add_dequant_suffix(bias_name)
                dequant_node = onnx.helper.make_node(
                    "Cast",
                    [quant_value.q_name],
                    [bias_name],
                    name=node_name,
                    to=init.data_type,
                )
            elif quant_value.node_type in (None, "DequantizeLinear"):
                if quant_value.node_qtype in {
                    onnx.TensorProto.FLOAT16,
                    onnx.TensorProto.BFLOAT16,
                    onnx.TensorProto.FLOAT,
                }:
                    raise RuntimeError(f"Unexpected quantize type {quant_value.node_qtype} for DequantizeLinear.")
                inputs = [quant_value.q_name, quant_value.scale_name, quant_value.zp_name]
                node_name = add_dequant_suffix(bias_name)
                if quant_value.axis is not None:
                    dequant_node = onnx.helper.make_node(
                        "DequantizeLinear",
                        inputs,
                        [bias_name],
                        node_name,
                        axis=quant_value.axis,
                        domain=self.qdq_op_domain,
                    )
                else:
                    dequant_node = onnx.helper.make_node(
                        "DequantizeLinear",
                        inputs,
                        [bias_name],
                        node_name,
                        domain=self.qdq_op_domain,
                    )
            else:
                raise RuntimeError(f"Unexpected operator type {quant_value.node_type!r}.")
            self.model.add_node(dequant_node)

    def is_tensor_quantized(self, tensor_name):
        return tensor_name in self.tensors_to_quantize or tensor_name in self.bias_to_quantize
