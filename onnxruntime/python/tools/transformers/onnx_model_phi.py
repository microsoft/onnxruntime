# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from logging import getLogger
from typing import List, Optional
from fusion_base import Fusion
from fusion_utils import FusionUtils
from onnx import GraphProto, ModelProto, TensorProto, ValueInfoProto, helper
from onnx_model import OnnxModel
from fusion_options import FusionOptions

logger = getLogger(__name__)


class FissionTransformerBlockPhi(Fusion):

    def __init__(
        self,
        model: OnnxModel,
    ):
        super().__init__(model, "DONOTUSE", ["model_modeling_mixformer_sequential_ParallelBlock_sub2_1_1",
                                             "model_modeling_mixformer_sequential_ParallelBlock_sub2_1_2"])

    def fuse(
            self,
            node,
            input_name_to_nodes,
            output_name_to_node,
    ):
        print(node.name)


class PhiOnnxModel(OnnxModel):
    def __init__(self, model: ModelProto, num_heads: int = 0, head_size: int = 0):
        super().__init__(model)

    def transformer_block_fission(self):
        print("herehere")

    def postprocess(self):
        self.prune_graph()

    def optimize(self, options: Optional[FusionOptions] = None, add_dynamic_axes: bool = False):
        self.transformer_block_fission()
