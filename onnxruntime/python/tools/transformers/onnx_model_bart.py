#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------
import logging
import onnx
import sys
import argparse
import numpy as np
from collections import deque
from onnx import ModelProto, TensorProto, numpy_helper
from onnx_model_bert import BertOnnxModel

logger = logging.getLogger(__name__)




class BartOnnxModel(BertOnnxModel):
    def __init__(self, model, num_heads, hidden_size):
        super().__init__(model, num_heads, hidden_size)

    def fuse_attention(self):
        #fusion = FusionGptAttention(self, self.num_heads)
        #fusion.apply()
        print("haha")