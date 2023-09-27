# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import unittest

import numpy as np
import onnxscript
from mpi4py import MPI
from onnx import TensorProto, helper
from onnxscript import BOOL, FLOAT, INT64
from onnxscript import opset18 as opset
from onnxscript.onnx_types import TensorType
from parameterized import parameterized

comm = MPI.COMM_WORLD
#comm.Get_rank(), comm.Get_size()

import onnxruntime as ort

MICROSOFT_OPSET = onnxscript.values.Opset(domain="com.microsoft", version=1)

@onnxscript.script()
def MatMul2D(X: FLOAT[2, "s"], W: FLOAT["s", 2]) -> FLOAT[2, 2]:
    return MICROSOFT_OPSET.DistributedMatMul(
        X,
        W,
        device_mesh_shape=[2],
        device_mesh_elements=[0, 1],
        input_shard_specs=["RS[0]", "S[0]R"],
        output_shard_specs=["RR"])
onnx_model = MatMul2D.to_model_proto()

rank = comm.Get_rank()
print(rank)
print(onnx_model)
sess = ort.InferenceSession(onnx_model.SerializeToString(), providers=["CUDAExecutionProvider"],
            provider_options=[{"device_id": str(rank)}])

X = np.array([[1, 2, 3, 4],
             [3, 4, 5, 6]], dtype=np.float32)
W = np.array([[1, 1],
             [2, 2],
             [3, 3],
             [4, 4]], dtype=np.float32)

X_shard = np.split(X, 2, axis=1)[rank]
W_shard = np.split(W, 2, axis=0)[rank]
result = sess.run(None, {"X": X_shard, "W": W_shard})
if rank == 0:
    print(result[0])
    print(np.matmul(X, W))
