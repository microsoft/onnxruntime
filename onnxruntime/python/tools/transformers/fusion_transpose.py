# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from logging import getLogger
from typing import Dict, List

from fusion_base import Fusion
from fusion_utils import FusionUtils
from onnx import NodeProto, helper
from onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionTranspose(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "Transpose", "Transpose")

    def fuse(
        self,
        transpose_node: NodeProto,
        input_name_to_nodes: Dict[str, List[NodeProto]],
        output_name_to_node: Dict[str, NodeProto],
    ):
        """
        Case 1:
              (input)-->Transpose(perm=a)-->Transpose(perm=b)-->
        After:
              (input)-->Transpose(perm=a)-->  (this path can be removed if the output is not used anymore)
                |
                +----->Transpose(perm=a*b)-->

        Case 2 (Cast has only one child):
              (input)-->Transpose(perm=a)--> Cast -->Transpose(perm=b)-->
        After:
              (input)-->Transpose(perm=a)-->  (this path can be removed if the output is not used anymore)
                |
                +----->Cast --> Transpose(perm=a*b)-->


        """
        transpose_b = transpose_node
        if transpose_b.input[0] not in output_name_to_node:
            return

        transpose_a = output_name_to_node[transpose_b.input[0]]
        if transpose_a.op_type != "Cast":
            cast_node = None
        else:
            cast_node = transpose_a

            cast_children = self.model.get_children(cast_node, input_name_to_nodes)
            if cast_children and len(cast_children) > 1:
                return
            transpose_a = output_name_to_node[cast_node.input[0]]

        if transpose_a.op_type != "Transpose":
            return

        permutation = OnnxModel.get_node_attribute(transpose_b, "perm")
        assert isinstance(permutation, list)

        parent_permutation = OnnxModel.get_node_attribute(transpose_a, "perm")
        assert isinstance(parent_permutation, list)

        assert len(parent_permutation) == len(permutation)

        output_permutation = []
        for j, index in enumerate(permutation):
            output_permutation.append(parent_permutation[index])

        if cast_node is None:
            if FusionUtils.skip_parent(self.model, transpose_b, transpose_a, input_name_to_nodes):
                self.nodes_to_remove.append(transpose_a)
        else:
            if FusionUtils.skip_parent(self.model, cast_node, transpose_a, input_name_to_nodes):
                self.nodes_to_remove.append(transpose_a)
        transpose_b.ClearField("attribute")
        transpose_b.attribute.extend([helper.make_attribute("perm", output_permutation)])
