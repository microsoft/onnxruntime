# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from itertools import chain
from typing import TYPE_CHECKING, List, Tuple

import numpy as np
import torch
import onnx
from transformers import AutoConfig, AutoModelForCausalLM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# TODO: add cache dir
model_class = "microsoft/phi-2"
phi_config = AutoConfig.from_pretrained(model_class, trust_remote_code=True, cache_dir="./cache")
phi_model = AutoModelForCausalLM.from_pretrained(model_class, trust_remote_code=True, cache_dir="./cache")
phi_model.eval()
phi_model.to(device)

# dry run before dynamo export
batch_size, sequence_length, past_sequence_length = 2, 8, 0
max_sequence_length = 2048

input_ids = torch.randint(low=0, high=phi_config.vocab_size, size=(batch_size, sequence_length), dtype=torch.int64, device=device)
#attention_mask = torch.ones(input_ids.shape, dtype=torch.int64, device=device)
past_key_values = phi_model(input_ids, use_cache=True)["past_key_values"]
# dynamo
phi_model(input_ids, past_key_values=past_key_values)

from torch._dynamo import config
config.capture_scalar_outputs = True
temp_path = "phi-2_decoder_fp32.onnx"

input_ids = input_ids.expand(2, -1)
torch._dynamo.mark_dynamic(input_ids, 0)

# for i in range(len(past_key_values)):
#     print(i)
#     past_key, past_value = past_key_values[i]
#     past_key_values[i][0] = past_key.expand(2, -1, -1, -1)
#     past_key_values[i][1] = past_value.expand(2, -1, -1, -1)
#     torch._dynamo.mark_dynamic(past_key_values[i][0], 0)
#     torch._dynamo.mark_dynamic(past_key_values[i][1], 0)
torch.onnx.dynamo_export(
    phi_model, input_ids, past_key_values=past_key_values, export_options=torch.onnx.ExportOptions(dynamic_shapes=True)
).save(temp_path)
onnx.checker.check_model(temp_path)
onnx.shape_inference.infer_shapes_path(temp_path)