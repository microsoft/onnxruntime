# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import unittest

import numpy as np
from mpi4py import MPI
from onnx import TensorProto, helper

import onnxruntime

np.random.seed(3)

comm = MPI.COMM_WORLD


def get_rank():
    return comm.Get_rank()


def get_size():
    return comm.Get_size()


def print_out(*args):
    if get_rank() == 0:
        print(*args)


local_rank = get_rank()

ORT_DTYPE = TensorProto.FLOAT16
NP_TYPE = np.float16 if ORT_DTYPE == TensorProto.FLOAT16 else np.float32
THRESHOLD_TP = 3e-2
THRESHOLD_EP = 1e-6

onnx_model_full = "/wy/ORT_GENAI/wangye/mixtral/src/python/py/models/example-models/mixtral_2layer/model.onnx"
onnx_model_local = f"/wy/ORT_GENAI/wangye/mixtral/src/python/py/models/example-models/mixtral_2layer_rank_{local_rank}/model.onnx"

num_heads = 32
head_size = 80

batch_size = 1
sequence_length = 1
past_sequence_length = 31
ort_inputs = {
    "input_ids": np.random.randint(0, 32000, (batch_size, sequence_length)).astype(np.int64),
    "attention_mask": np.ones((batch_size, sequence_length + past_sequence_length)).astype(np.int64),
    "position_ids": np.arange(0, sequence_length).repeat(batch_size).reshape(batch_size, sequence_length).astype(np.int64),
    "past_key_values.0.key": np.random.normal(0, 0.1, batch_size*num_heads*past_sequence_length*head_size).reshape(batch_size, num_heads, past_sequence_length, head_size).astype(NP_TYPE),
    "past_key_values.0.value": np.random.normal(0, 0.1, batch_size*num_heads*past_sequence_length*head_size).reshape(batch_size, num_heads, past_sequence_length, head_size).astype(NP_TYPE),
    "past_key_values.1.key": np.random.normal(0, 0.1, batch_size*num_heads*past_sequence_length*head_size).reshape(batch_size, num_heads, past_sequence_length, head_size).astype(NP_TYPE),
    "past_key_values.1.value": np.random.normal(0, 0.1, batch_size*num_heads*past_sequence_length*head_size).reshape(batch_size, num_heads, past_sequence_length, head_size).astype(NP_TYPE),
}

nh_start = local_rank * num_heads // get_size()
nh_end = (local_rank + 1) * num_heads // get_size()
local_ort_inputs = {
    "input_ids": ort_inputs["input_ids"],
    "attention_mask": ort_inputs["attention_mask"],
    "position_ids": ort_inputs["position_ids"],
    "past_key_values.0.key": ort_inputs["past_key_values.0.key"][:, nh_start : nh_end, :, :],
    "past_key_values.0.value": ort_inputs["past_key_values.0.value"][:, nh_start : nh_end, :, :],
    "past_key_values.1.key": ort_inputs["past_key_values.1.key"][:, nh_start : nh_end, :, :],
    "past_key_values.1.value": ort_inputs["past_key_values.1.value"][:, nh_start : nh_end, :, :],
}

sess_options = onnxruntime.SessionOptions()
cuda_provider_options = {"device_id": local_rank}
execution_providers = [("CUDAExecutionProvider", cuda_provider_options)]

ort_session = onnxruntime.InferenceSession(onnx_model_full, sess_options, providers=execution_providers)
ort_session_local = onnxruntime.InferenceSession(onnx_model_local, sess_options, providers=execution_providers)

output = ort_session.run(None, ort_inputs)
sharded_output = ort_session_local.run(None, local_ort_inputs)

print_out("max diff:", np.max(np.abs(output[0] - sharded_output[0])))
print_out("output", output[0])
print_out("sharded_output", sharded_output[0])
assert np.allclose(output[0], sharded_output[0], atol=THRESHOLD_TP, rtol=THRESHOLD_TP)