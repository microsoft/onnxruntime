# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest

import numpy as np
import onnxscript
from mpi4py import MPI
from onnxscript import FLOAT

import onnxruntime as ort

MICROSOFT_OPSET = onnxscript.values.Opset(domain="com.microsoft", version=1)
comm = MPI.COMM_WORLD


@onnxscript.script()
def MatMul2D(X: FLOAT[2, "s"], W: FLOAT["s", 2]) -> FLOAT[2, 2]:
    return MICROSOFT_OPSET.DistributedMatMul(
        X,
        W,
        device_mesh_shape=[2],
        device_mesh_elements=[0, 1],
        input_shard_specs=["RS[0]", "S[0]R"],
        output_shard_specs=["RR"]
    )


@onnxscript.script()
def MatMul2D_RS_RS_RR(X: FLOAT[2, "s"], W: FLOAT[4, "t"]) -> FLOAT[2, 2]:
    # Shape informaton should match the shapes seen by the operator.
    # If the tensor W with shape [4, 2] is sharded following "RS[0]", its shape
    # should be [4, 1] in ORT when calling ctx->Input<Tensor>(1)->Shape().
    return MICROSOFT_OPSET.DistributedMatMul(
        X,
        W,
        device_mesh_shape=[2],
        device_mesh_elements=[0, 1],
        input_shard_specs=["RS[0]", "RS[0]"],
        output_shard_specs=["RR"]
    )


@onnxscript.script()
def MatMul2D_RS_RS_RR(X: FLOAT[2, "s"], W: FLOAT[4, "t"]) -> FLOAT[2, 2]:
    return MICROSOFT_OPSET.DistributedMatMul(
        X,
        W,
        device_mesh_shape=[2],
        device_mesh_elements=[0, 1],
        input_shard_specs=["RS[0]", "RS[0]"],
        output_shard_specs=["RR"]
    )

@onnxscript.script()
def MatMul2D_RS_RS_RS(X: FLOAT[2, "s"], W: FLOAT[4, "t"]) -> FLOAT[2, "u"]:
    return MICROSOFT_OPSET.DistributedMatMul(
        X,
        W,
        device_mesh_shape=[2],
        device_mesh_elements=[0, 1],
        input_shard_specs=["RS[0]", "RS[0]"],
        output_shard_specs=["RS[0]"]
    )

def ShardTensor(X, rank, axis, num_shards):
    return np.split(X, num_shards, axis)[rank]

class TestDistributedMatMul(unittest.TestCase):

    def test_MatMul2D(self):
        rank = comm.Get_rank()
        X = np.array([[1, 2, 3, 4], [3, 4, 5, 6]], dtype=np.float32)
        W = np.array([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=np.float32)

        onnx_model = MatMul2D.to_model_proto()
        sess = ort.InferenceSession(onnx_model.SerializeToString(), providers=["CUDAExecutionProvider"],
                                    provider_options=[{"device_id": str(rank)}])

        X_shard = ShardTensor(X, rank=rank, axis=1, num_shards=2)
        W_shard = ShardTensor(W, rank=rank, axis=0, num_shards=2)

        result = sess.run(None, {"X": X_shard, "W": W_shard})

        if rank == 0:
            expected = np.matmul(X, W)
            np.testing.assert_allclose(result[0], expected, rtol=1e-5, atol=1e-8)

    def test_MatMul2D_RS_RS_RR(self):
        rank = comm.Get_rank()
        X = np.array([[1, 2, 3, 4], [3, 4, 5, 6]], dtype=np.float32)
        W = np.array([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=np.float32)

        onnx_model = MatMul2D_RS_RS_RR.to_model_proto()
        sess = ort.InferenceSession(onnx_model.SerializeToString(), providers=["CUDAExecutionProvider"],
                                    provider_options=[{"device_id": str(rank)}])

        X_shard = ShardTensor(X, rank=rank, axis=1, num_shards=2)
        W_shard = ShardTensor(W, rank=rank, axis=1, num_shards=2)

        result = sess.run(None, {"X": X_shard, "W": W_shard})

        if rank == 0:
            expected = np.matmul(X, W)
            np.testing.assert_allclose(result[0], expected, rtol=1e-5, atol=1e-8)

    def test_MatMul2D_RS_RS_RS(self):
        rank = comm.Get_rank()
        X = np.array([[1, 2, 3, 4], [3, 4, 5, 6]], dtype=np.float32)
        W = np.array([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=np.float32)

        onnx_model = MatMul2D_RS_RS_RS.to_model_proto()
        sess = ort.InferenceSession(onnx_model.SerializeToString(), providers=["CUDAExecutionProvider"],
                                    provider_options=[{"device_id": str(rank)}])

        X_shard = ShardTensor(X, rank=rank, axis=1, num_shards=2)
        W_shard = ShardTensor(W, rank=rank, axis=1, num_shards=2)

        result = sess.run(None, {"X": X_shard, "W": W_shard})

        if rank == 0:
            expected = ShardTensor(np.matmul(X, W), rank=rank, axis=1, num_shards=2)
            np.testing.assert_allclose(result[0], expected, rtol=1e-5, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
