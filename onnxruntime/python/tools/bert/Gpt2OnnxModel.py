#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

from BertOnnxModel import BertOnnxModel


class Gpt2OnnxModel(BertOnnxModel):

    def __init(self, model, num_heads, hidden_size, sequence_length, input_int32, float16, gpu_only):
        super().__init__(model, num_heads, hidden_size, sequence_length, input_int32, float16, gpu_only)
