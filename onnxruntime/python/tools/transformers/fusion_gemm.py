#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------
from typing import Dict
from logging import getLogger
from onnx import helper, numpy_helper
from onnx_model import OnnxModel
from fusion_base import Fusion

logger = getLogger(__name__)


class FusionGemm(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "Gemm", "Add")

    def fuse(self, node, input_name_to_nodes: Dict, output_name_to_node: Dict):
        gemm_nodes = self.model.match_parent_path(node, ['MatMul'], [0])
        if gemm_nodes is None:
            return
        matmul_node = gemm_nodes[0]
        matmul_weight_input = matmul_node.input[1]
        add_weight_input = node.input[1]
        if matmul_weight_input is None or add_weight_input is None:
            return
        if self.model.get_initializer(add_weight_input) is None:
            return
        self.nodes_to_remove.extend(gemm_nodes)
        gemm_node = helper.make_node('Gemm',
                                      inputs=[matmul_node.input[0], matmul_weight_input, add_weight_input],
                                      outputs=[node.output[0]],
                                      name = 'Gemm_' + matmul_node.name)
        self.nodes_to_add.append(gemm_node)
