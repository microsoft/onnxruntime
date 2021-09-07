#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

from fusion_base import Fusion
from logging import getLogger
from onnx import TensorProto, NodeProto
from onnx_model import OnnxModel
from fusion_utils import FusionUtils
from typing import Dict, List

logger = getLogger(__name__)


class FusionRemoveCast(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "", "Cast")
        self.utils = FusionUtils(model)
        self.shape_infer = None
        self.shape_infer_done = False

    def get_dtype_from_tensor_proto(self, tensor_proto: TensorProto) -> int:
        if tensor_proto.type.tensor_type.HasField('elem_type'):
            return tensor_proto.type.tensor_type.elem_type
        else:
            return None

    def get_dtype(self, input_or_output_name: str) -> int:
        dtype = self.model.get_dtype(input_or_output_name)
        if dtype is not None:
            return dtype

        if not self.shape_infer_done:
            self.shape_infer = self.model.infer_runtime_shape({}, update=True)
            self.shape_infer_done = True

        if self.shape_infer is not None:
            return self.get_dtype_from_tensor_proto(self.shape_infer.known_vi_[input_or_output_name])

        return None

    def fuse(self, cast_node: NodeProto, input_name_to_nodes: Dict[str, List[NodeProto]],
             output_name_to_node: Dict[str, NodeProto]):
        input_dtype = self.get_dtype(cast_node.input[0])
        output_dtype = self.get_dtype(cast_node.output[0])

        # Remove Cast nodes that are not needed: input data type of Cast is same as output data type.
        if input_dtype and input_dtype == output_dtype:
            if self.model.find_graph_output(cast_node.output[0]) is None:
                self.model.replace_input_of_all_nodes(cast_node.output[0], cast_node.input[0])
                self.fused_count += 1
                self.nodes_to_remove.append(cast_node)
            elif self.model.find_graph_input(cast_node.input[0]) is None:
                self.model.replace_output_of_all_nodes(cast_node.input[0], cast_node.output[0])
                self.fused_count += 1
                self.nodes_to_remove.append(cast_node)
