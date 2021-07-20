#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

from logging import getLogger
from onnx import helper
from onnx_model import OnnxModel
from fusion_base import Fusion
from fusion_utils import NumpyHelper

logger = getLogger(__name__)

#
#
# TODO(kreeger): Left off right here. Need to write these I think?
#
#

class FusionEmbedLayerNormBiasGelu(Fusion):
    """
    TODO(kreeger): Add some documentation here.
    """
    def __init__(self, model: OnnxModel):
        super().__init__(model, "EmbedLayerNormBiasGelu", "SkipLayerNormalization")
        # TODO(kreeger): what is this?
        # self.shape_infer_helper = self.model.infer_runtime_shape({"batch_size": 4, "seq_len": 7})

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        print("Fuse EmbedLayerNormFusionBiasGelu from Python")
