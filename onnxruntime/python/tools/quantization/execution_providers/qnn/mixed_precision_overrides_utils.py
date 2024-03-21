# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from __future__ import annotations

import onnx
from dataclasses import dataclass
from typing import Any

from ...quant_utils import QuantType
from ...tensor_quant_overrides import QuantTypeInfo, TensorQuantOverridesHelper


@dataclass
class TensorTypeRequest:
    producer: QuantTypeInfo | None
    consumers: tuple[QuantTypeInfo, set[str]] | None


class MixedPrecisionTensorQuantOverridesFixer:
    def __init__(
        self,
        overrides: TensorQuantOverridesHelper,
        producers: dict[str, onnx.NodeProto],
        consumers: dict[str, onnx.NodeProto],
        value_infos: dict[str, onnx.ValueInfoProto],
        initializers: dict[str, onnx.TensorProto],
    ):
        self.overrides = overrides
        self.consumers = consumers
        self.producers = producers
        self.value_infos = value_infos
        self.initializers = initializers

    @staticmethod
    def create_from_model(overrides: TensorQuantOverridesHelper, model: onnx.ModelProto):
        consumers = {}
        producers = {}

        # Build dictionaries that map a tensor name to the consumer or producer nodes.
        for node in model.graph.node:
            for input_name in node.input:
                if input_name:
                    if input_name not in consumers:
                        consumers[input_name] = []

                    consumers[input_name].append(node)

            for output_name in node.output:
                producers[output_name] = node

        # Build dictionaries that enable convenient lookups of initializers and value_infos by name.
        initializers = {initializer.name: initializer for initializer in model.graph.initializer}
        value_infos = {vi.name: vi for vi in model.graph.value_info}
        value_infos.update({ot.name: ot for ot in model.graph.output})
        value_infos.update({it.name: it for it in model.graph.input})

        return MixedPrecisionTensorQuantOverridesFixer(overrides, producers, consumers, value_infos, initializers)

    def apply(
        self,
        default_activation_qtype: QuantType,
        default_activation_symmetric: bool,
    ):
        type_requests = self.get_desired_tensor_types(default_activation_qtype, default_activation_symmetric)

        # Use type requests to "fix" tensor quantization overrides by adding
        # quantization type conversions where necessary.
        for tensor_name, type_req in type_requests.items():
            all_consumers = set([node.name for node in self.consumers.get(tensor_name, [])])
            has_producer_req = type_req.producer is not None
            has_consumer_req = bool(type_req.consumers)

            # Only producer type: Add conversion back to default activation type
            if has_producer_req and not has_consumer_req:
                self._update_converted_tensor(
                    tensor_name, type_req.producer, QuantTypeInfo(default_activation_qtype), all_consumers
                )
            # Only consumers
            elif not has_producer_req and has_consumer_req:
                prod_type_info = self.overrides.get_node_output_qtype_info(tensor_name, default_activation_qtype)
                consumer_type_info = type_req.consumers[0]

                if prod_type_info != consumer_type_info:
                    self._update_converted_tensor(
                        tensor_name, prod_type_info, consumer_type_info, type_req.consumers[1]
                    )
                else:
                    if not self._check_nodes_are_not_convert_consumers(tensor_name, type_req.consumers[1]):
                        raise ValueError(
                            f"Tensor override for '{tensor_name}' converts the type for consumers that need the original type."
                        )
            # Both producer and consumers
            elif has_producer_req and has_consumer_req:
                prod_type_info = type_req.producer
                consumer_type_info = type_req.consumers[0]

                if prod_type_info != consumer_type_info:
                    self._update_converted_tensor(
                        tensor_name, prod_type_info, consumer_type_info, type_req.consumers[1]
                    )
                else:
                    consumers_for_original_type = all_consumers.difference(type_req.consumers[1])

                    if len(consumers_for_original_type) == 0:
                        # All consumers want the overridden type, so no need for convert nodes!
                        # Just add the override to the new new if not already present.
                        if tensor_name not in self.overrides:
                            self.overrides[tensor_name] = [{}]
                            prod_type_info.save_to_dict(self.overrides[tensor_name][0])

                        assert "convert" not in self.overrides[tensor_name][0]
                    else:
                        # Some consumers don't want the overridden type.
                        self._update_converted_tensor(
                            tensor_name,
                            prod_type_info,
                            QuantTypeInfo(default_activation_qtype),
                            consumers_for_original_type,
                        )
            else:
                raise ValueError(f"TypeRequest for tensor {tensor_name} has no producer or consumers.")

    def get_desired_tensor_types(
        self,
        default_activation_qtype: QuantType,
        default_activation_symmetric: bool,
    ) -> dict[str, TensorTypeRequest]:
        type_requests = {}
        default_activation_type_info = QuantTypeInfo(default_activation_qtype, default_activation_symmetric)

        # Scan tensor overrides for type conversion requests.
        for tensor_name, override_list in self.overrides.items():
            if not self.__is_tensor_quantizable(tensor_name):
                continue  # Skip non-quantizable tensors (e.g., not a float)

            if tensor_name in self.initializers:
                continue  # Skip initializers

            if not override_list or len(override_list) > 1:
                continue  # Skip per-channel stuff

            override_dict = override_list[0]
            quant_type_info = QuantTypeInfo.load_from_dict(override_dict, default_activation_type_info.quant_type)
            producer_node = self.producers.get(tensor_name)  # None if this is a model input

            if quant_type_info != default_activation_type_info and "convert" not in override_dict:
                if producer_node is not None:
                    self._add_type_requests_for_node(type_requests, quant_type_info, producer_node)

                # Find all consumer nodes of `tensor_name` and update their inputs/outputs to the new type.
                for consumer_node in self.consumers.get(tensor_name, []):
                    self._add_type_requests_for_node(type_requests, quant_type_info, consumer_node)

        return type_requests

    def _add_type_requests_for_node(
        self,
        type_requests: dict[str, TensorTypeRequest],
        quant_type_info: QuantTypeInfo,
        node: onnx.NodeProto,
    ):
        # Add output side
        for output_name in node.output:
            if not self.__is_tensor_quantizable(output_name):
                continue

            if output_name not in type_requests:
                type_requests[output_name] = TensorTypeRequest(quant_type_info, None)
            else:
                if (
                    type_requests[output_name].producer is not None
                    and type_requests[output_name].producer != quant_type_info
                ):
                    raise ValueError(f"Tensor {output_name} has multiple types.")

                type_requests[output_name].producer = quant_type_info

        # Add the consumer side
        for input_name in node.input:
            if input_name and input_name not in self.initializers and self.__is_tensor_quantizable(input_name):
                if input_name not in type_requests:
                    type_requests[input_name] = TensorTypeRequest(None, None)

                if type_requests[input_name].consumers is None:
                    type_requests[input_name].consumers = (quant_type_info, set())

                if type_requests[input_name].consumers[0] != quant_type_info:
                    raise ValueError(f"Tensor {input_name} has consumers requesting different types.")

                if not node.name:
                    raise ValueError(
                        f"Node of type {node.op_type} with output 0 {node.output[0]} does not have a name!"
                    )

                type_requests[input_name].consumers[1].add(node.name)

    def _update_converted_tensor(
        self,
        tensor_name: str,
        producer_type_info: QuantTypeInfo,
        consumer_type_info: QuantTypeInfo,
        consumer_names: set[str],
    ):
        if tensor_name not in self.overrides or not self.overrides[tensor_name]:
            self.overrides[tensor_name] = [{}]
            producer_type_info.save_to_dict(self.overrides[tensor_name][0])

        overrides = self.overrides[tensor_name][0]
        if producer_type_info != QuantTypeInfo.load_from_dict(overrides):
            raise ValueError(f"Desired producer quant_type for {tensor_name} doesn't match existing type.")

        if consumer_names:
            if "convert" not in overrides:
                overrides["convert"] = {}
                consumer_type_info.save_to_dict(overrides["convert"])

            convert_dict = overrides["convert"]
            if consumer_type_info != QuantTypeInfo.load_from_dict(convert_dict):
                raise ValueError(f"Desired consumer quant_type for {tensor_name} doesn't match existing type.")

            if "recv_nodes" not in convert_dict:
                convert_dict["recv_nodes"] = set()

            convert_dict["recv_nodes"].update(consumer_names)

    def _check_nodes_are_not_convert_consumers(self, tensor_name: str, node_names: set[str]):
        if tensor_name not in self.overrides or not self.overrides[tensor_name]:
            return True

        overrides = self.overrides[tensor_name][0]

        if "convert" not in overrides:
            return True

        convert_dict = overrides["convert"]

        if "recv_nodes" not in convert_dict:
            return False

        return not convert_dict["recv_nodes"].intersection(node_names)

    # TODO: This should either be a shared util or should be a closure that is passed in
    # to the constructor.
    def __is_tensor_quantizable(self, tensor_name):
        weight = self.initializers.get(tensor_name)
        if weight is not None:
            if weight.data_type in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16):
                return True
        elif tensor_name in self.value_infos:
            vi = self.value_infos[tensor_name]
            if vi.type.HasField("tensor_type") and vi.type.tensor_type.elem_type in (
                onnx.TensorProto.FLOAT,
                onnx.TensorProto.FLOAT16,
            ):
                return True

        return False

