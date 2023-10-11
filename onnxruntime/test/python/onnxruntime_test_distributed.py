# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest

import numpy as np
import onnxscript
from mpi4py import MPI
from onnxscript import FLOAT, INT64

import onnxruntime as ort

MICROSOFT_OPSET = onnxscript.values.Opset(domain="com.microsoft", version=1)
comm = MPI.COMM_WORLD


def shard_tensor(X, rank, axis, num_shards):
    return np.split(X, num_shards, axis)[rank]


class TestDistributed(unittest.TestCase):
    def test_matmul_rs_sr_rr(self):
        @onnxscript.script()
        def matmul_rs_sr_rr(tensor_x: FLOAT, tensor_w: FLOAT) -> FLOAT:
            return MICROSOFT_OPSET.DistributedMatMul(
                tensor_x,
                tensor_w,
                device_mesh_shape=[2],
                device_mesh_elements=[0, 1],
                input_shard_specs=["RS[0]", "S[0]R"],
                output_shard_specs=["RR"],
            )

        rank = comm.Get_rank()
        tensor_x = np.array([[1, 2, 3, 4], [3, 4, 5, 6]], dtype=np.float32)
        tensor_w = np.array([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=np.float32)

        onnx_model = matmul_rs_sr_rr.to_model_proto(
            input_types=[FLOAT[2, "s"], FLOAT["s", 2]],
            output_types=[FLOAT[2, 2]],
        )

        sess = ort.InferenceSession(
            onnx_model.SerializeToString(),
            providers=["CUDAExecutionProvider"],
            provider_options=[{"device_id": str(rank)}],
        )

        tensor_shard_x = shard_tensor(tensor_x, rank=rank, axis=1, num_shards=2)
        tensor_shard_w = shard_tensor(tensor_w, rank=rank, axis=0, num_shards=2)

        result = sess.run(None, {"tensor_x": tensor_shard_x, "tensor_w": tensor_shard_w})

        expected = np.matmul(tensor_x, tensor_w)
        np.testing.assert_allclose(result[0], expected, rtol=1e-5, atol=1e-8)

    def test_matmul2d_rs_rs_rr(self):
        @onnxscript.script()
        def matmul_rs_rs_rr(tensor_x: FLOAT, tensor_w: FLOAT) -> FLOAT:
            return MICROSOFT_OPSET.DistributedMatMul(
                tensor_x,
                tensor_w,
                device_mesh_shape=[2],
                device_mesh_elements=[0, 1],
                input_shard_specs=["RS[0]", "RS[0]"],
                output_shard_specs=["RR"],
            )

        rank = comm.Get_rank()
        tensor_x = np.array([[1, 2, 3, 4], [3, 4, 5, 6]], dtype=np.float32)
        tensor_w = np.array([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=np.float32)

        # Shape informaton should match the shapes seen by the operator.
        # If the tensor W with shape [4, 2] is sharded following "RS[0]", its shape
        # should be [4, 1] in ORT when calling ctx->Input<Tensor>(1)->Shape().
        onnx_model = matmul_rs_rs_rr.to_model_proto(
            input_types=[FLOAT[2, "s"], FLOAT[4, "t"]],
            output_types=[FLOAT[2, 2]],
        )

        sess = ort.InferenceSession(
            onnx_model.SerializeToString(),
            providers=["CUDAExecutionProvider"],
            provider_options=[{"device_id": str(rank)}],
        )

        tensor_shard_x = shard_tensor(tensor_x, rank=rank, axis=1, num_shards=2)
        tensor_shard_w = shard_tensor(tensor_w, rank=rank, axis=1, num_shards=2)

        result = sess.run(None, {"tensor_x": tensor_shard_x, "tensor_w": tensor_shard_w})

        expected = np.matmul(tensor_x, tensor_w)
        np.testing.assert_allclose(result[0], expected, rtol=1e-5, atol=1e-8)

    def test_matmul2d_rs_rs_rs(self):
        @onnxscript.script()
        def matmul2d_rs_rs_rs(tensor_x: FLOAT, tensor_w: FLOAT) -> FLOAT:
            return MICROSOFT_OPSET.DistributedMatMul(
                tensor_x,
                tensor_w,
                device_mesh_shape=[2],
                device_mesh_elements=[0, 1],
                input_shard_specs=["RS[0]", "RS[0]"],
                output_shard_specs=["RS[0]"],
            )

        rank = comm.Get_rank()
        tensor_x = np.array([[1, 2, 3, 4], [3, 4, 5, 6]], dtype=np.float32)
        tensor_w = np.array([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=np.float32)

        onnx_model = matmul2d_rs_rs_rs.to_model_proto(
            input_types=[FLOAT[2, "s"], FLOAT[4, "t"]],
            output_types=[FLOAT[2, "u"]],
        )

        sess = ort.InferenceSession(
            onnx_model.SerializeToString(),
            providers=["CUDAExecutionProvider"],
            provider_options=[{"device_id": str(rank)}],
        )

        tensor_shard_x = shard_tensor(tensor_x, rank=rank, axis=1, num_shards=2)
        tensor_shard_w = shard_tensor(tensor_w, rank=rank, axis=1, num_shards=2)

        result = sess.run(None, {"tensor_x": tensor_shard_x, "tensor_w": tensor_shard_w})

        expected = shard_tensor(np.matmul(tensor_x, tensor_w), rank=rank, axis=1, num_shards=2)
        np.testing.assert_allclose(result[0], expected, rtol=1e-5, atol=1e-8)

    def test_matmul_srr_rr_srr(self):
        @onnxscript.script()
        def matmul_srr_rr_srr(tensor_x: FLOAT, tensor_w: FLOAT) -> FLOAT:
            return MICROSOFT_OPSET.DistributedMatMul(
                tensor_x,
                tensor_w,
                device_mesh_shape=[2],
                device_mesh_elements=[0, 1],
                input_shard_specs=["S[0]RR", "RR"],
                output_shard_specs=["S[0]RR"],
            )

        rank = comm.Get_rank()
        # Shape [2, 2, 4]
        tensor_x = np.array([[[1, 2, 3, 4], [3, 4, 5, 6]], [[1, 2, 3, 4], [3, 4, 5, 6]]], dtype=np.float32)
        # Shape [4, 2]
        tensor_w = np.array([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=np.float32)

        onnx_model = matmul_srr_rr_srr.to_model_proto(
            input_types=[FLOAT["s", 2, 4], FLOAT[4, 2]],
            output_types=[FLOAT["s", 2, 2]],
        )

        sess = ort.InferenceSession(
            onnx_model.SerializeToString(),
            providers=["CUDAExecutionProvider"],
            provider_options=[{"device_id": str(rank)}],
        )

        tensor_shard_x = shard_tensor(tensor_x, rank=rank, axis=0, num_shards=2)
        tensor_shard_w = tensor_w

        result = sess.run(None, {"tensor_x": tensor_shard_x, "tensor_w": tensor_shard_w})

        expected = shard_tensor(np.matmul(tensor_x, tensor_w), rank=rank, axis=0, num_shards=2)
        np.testing.assert_allclose(result[0], expected, rtol=1e-5, atol=1e-8)

    def test_matmul_srr_rrrr_rsrr(self):
        @onnxscript.script()
        def matmul_srr_rrrr_rsrr(tensor_x: FLOAT, tensor_w: FLOAT) -> FLOAT:
            return MICROSOFT_OPSET.DistributedMatMul(
                tensor_x,
                tensor_w,
                device_mesh_shape=[2],
                device_mesh_elements=[0, 1],
                input_shard_specs=["S[0]RR", "RRRR"],
                output_shard_specs=["RS[0]RR"],
            )

        rank = comm.Get_rank()
        # Shape [2, 2, 4]
        tensor_x = np.array([[[1, 2, 3, 4], [3, 4, 5, 6]], [[1, 2, 3, 4], [3, 4, 5, 6]]], dtype=np.float32)
        # Shape [1, 2, 4, 2]
        tensor_w = np.array([[[[1, 1], [2, 2], [3, 3], [4, 4]], [[1, 1], [2, 2], [3, 3], [4, 4]]]], dtype=np.float32)

        onnx_model = matmul_srr_rrrr_rsrr.to_model_proto(
            input_types=[FLOAT["s", 2, 4], FLOAT[1, 2, 4, 2]],
            output_types=[FLOAT[1, "s", 2, 2]],
        )

        sess = ort.InferenceSession(
            onnx_model.SerializeToString(),
            providers=["CUDAExecutionProvider"],
            provider_options=[{"device_id": str(rank)}],
        )

        tensor_shard_x = shard_tensor(tensor_x, rank=rank, axis=0, num_shards=2)
        tensor_shard_w = tensor_w

        result = sess.run(None, {"tensor_x": tensor_shard_x, "tensor_w": tensor_shard_w})

        expected = shard_tensor(np.matmul(tensor_x, tensor_w), rank=rank, axis=1, num_shards=2)
        np.testing.assert_allclose(result[0], expected, rtol=1e-5, atol=1e-8)

    def test_matmul_sr_rs_rr(self):
        @onnxscript.script()
        def matmul_sr_rs_rr(tensor_x: FLOAT, tensor_w: FLOAT) -> FLOAT:
            return MICROSOFT_OPSET.DistributedMatMul(
                tensor_x,
                tensor_w,
                device_mesh_shape=[2],
                device_mesh_elements=[0, 1],
                input_shard_specs=["S[0]R", "RS[0]"],
                output_shard_specs=["RR"],
            )

        rank = comm.Get_rank()
        # Shape [4, 2]
        tensor_x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        # Shape [2, 2]
        tensor_w = np.array([[1, 1], [2, 2]], dtype=np.float32)

        onnx_model = matmul_sr_rs_rr.to_model_proto(
            input_types=[FLOAT["s", 2], FLOAT[2, "t"]],
            output_types=[FLOAT["s", "t"]],
        )

        sess = ort.InferenceSession(
            onnx_model.SerializeToString(),
            providers=["CUDAExecutionProvider"],
            provider_options=[{"device_id": str(rank)}],
        )

        tensor_shard_x = shard_tensor(tensor_x, rank=rank, axis=0, num_shards=2)
        tensor_shard_w = shard_tensor(tensor_w, rank=rank, axis=1, num_shards=2)

        result = sess.run(None, {"tensor_x": tensor_shard_x, "tensor_w": tensor_shard_w})

        expected = np.matmul(tensor_x, tensor_w)
        np.testing.assert_allclose(result[0], expected, rtol=1e-5, atol=1e-8)

    def test_matmul_rr_rs_rs(self):
        @onnxscript.script()
        def matmul_rr_rs_rs(tensor_x: FLOAT, tensor_w: FLOAT) -> FLOAT:
            return MICROSOFT_OPSET.DistributedMatMul(
                tensor_x,
                tensor_w,
                device_mesh_shape=[2],
                device_mesh_elements=[0, 1],
                input_shard_specs=["RR", "RS[0]"],
                output_shard_specs=["RS[0]"],
            )

        rank = comm.Get_rank()
        # Shape [4, 2]
        tensor_x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        # Shape [2, 4]
        tensor_w = np.array([[1, 1, 1, 1], [2, 2, 2, 2]], dtype=np.float32)

        onnx_model = matmul_rr_rs_rs.to_model_proto(
            input_types=[FLOAT[4, 2], FLOAT[2, "s"]],
            output_types=[FLOAT[4, "t"]],
        )

        sess = ort.InferenceSession(
            onnx_model.SerializeToString(),
            providers=["CUDAExecutionProvider"],
            provider_options=[{"device_id": str(rank)}],
        )

        tensor_shard_x = tensor_x
        tensor_shard_w = shard_tensor(tensor_w, rank=rank, axis=1, num_shards=2)

        result = sess.run(None, {"tensor_x": tensor_shard_x, "tensor_w": tensor_shard_w})

        expected = shard_tensor(np.matmul(tensor_x, tensor_w), rank=rank, axis=1, num_shards=2)
        np.testing.assert_allclose(result[0], expected, rtol=1e-5, atol=1e-8)

    def test_matmul_rr_sr_rr(self):
        @onnxscript.script()
        def matmul_rr_sr_rr(tensor_x: FLOAT, tensor_w: FLOAT) -> FLOAT:
            return MICROSOFT_OPSET.DistributedMatMul(
                tensor_x,
                tensor_w,
                device_mesh_shape=[2],
                device_mesh_elements=[0, 1],
                input_shard_specs=["RR", "S[0]R"],
                output_shard_specs=["RR"],
            )

        rank = comm.Get_rank()
        # Shape [4, 2]
        tensor_x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        # Shape [2, 6]
        tensor_w = np.array([[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2]], dtype=np.float32)

        onnx_model = matmul_rr_sr_rr.to_model_proto(
            input_types=[FLOAT[4, 2], FLOAT["s", 6]],
            output_types=[FLOAT[4, 6]],
        )

        sess = ort.InferenceSession(
            onnx_model.SerializeToString(),
            providers=["CUDAExecutionProvider"],
            provider_options=[{"device_id": str(rank)}],
        )

        tensor_shard_x = tensor_x
        tensor_shard_w = shard_tensor(tensor_w, rank=rank, axis=0, num_shards=2)

        result = sess.run(None, {"tensor_x": tensor_shard_x, "tensor_w": tensor_shard_w})

        expected = np.matmul(tensor_x, tensor_w)
        np.testing.assert_allclose(result[0], expected, rtol=1e-5, atol=1e-8)

    def test_slice_sr_axis1(self):
        @onnxscript.script()
        def slice_sr_axis1(tensor_x: FLOAT, tensor_starts: INT64, tensor_ends: INT64, tensor_axes: INT64) -> FLOAT:
            return MICROSOFT_OPSET.DistributedSlice(
                tensor_x,
                tensor_starts,
                tensor_ends,
                tensor_axes,
                device_mesh_shape=[2],
                device_mesh_elements=[0, 1],
                input_shard_specs=["S[0]R", "R", "R", "R", "R"],
                output_shard_specs=["S[0]R"],
            )

        rank = comm.Get_rank()
        # Shape [2, 4]
        tensor_x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
        tensor_starts = np.array([0], dtype=np.int64)
        tensor_ends = np.array([2], dtype=np.int64)
        tensor_axes = np.array([1], dtype=np.int64)

        onnx_model = slice_sr_axis1.to_model_proto(
            input_types=[FLOAT[1, 4], INT64[1], INT64[1], INT64[1]],
            output_types=[FLOAT[1, 2]],
        )

        sess = ort.InferenceSession(
            onnx_model.SerializeToString(),
            providers=["CUDAExecutionProvider"],
            provider_options=[{"device_id": str(rank)}],
        )

        tensor_shard_x = shard_tensor(tensor_x, rank=rank, axis=0, num_shards=2)

        result = sess.run(
            None,
            {
                "tensor_x": tensor_shard_x,
                "tensor_starts": tensor_starts,
                "tensor_ends": tensor_ends,
                "tensor_axes": tensor_axes,
            },
        )

        expected = tensor_shard_x[:, 0:2]
        np.testing.assert_allclose(result[0], expected, rtol=1e-5, atol=1e-8)

    def test_slice_rs_axis1(self):
        @onnxscript.script()
        def slice_sr_axis1(tensor_x: FLOAT, tensor_starts: INT64, tensor_ends: INT64, tensor_axes: INT64) -> FLOAT:
            return MICROSOFT_OPSET.DistributedSlice(
                tensor_x,
                tensor_starts,
                tensor_ends,
                tensor_axes,
                device_mesh_shape=[2],
                device_mesh_elements=[0, 1],
                input_shard_specs=["RS[0]", "R", "R", "R", "R"],
                output_shard_specs=["RS[0]"],
            )

        rank = comm.Get_rank()
        # Shape [2, 4]
        tensor_x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
        tensor_starts = np.array([0], dtype=np.int64)
        tensor_ends = np.array([2], dtype=np.int64)
        tensor_axes = np.array([1], dtype=np.int64)

        onnx_model = slice_sr_axis1.to_model_proto(
            input_types=[FLOAT[2, 2], INT64[1], INT64[1], INT64[1]],
            output_types=[FLOAT[2, 1]],
        )

        sess = ort.InferenceSession(
            onnx_model.SerializeToString(),
            providers=["CUDAExecutionProvider"],
            provider_options=[{"device_id": str(rank)}],
        )

        tensor_shard_x = shard_tensor(tensor_x, rank=rank, axis=1, num_shards=2)
        result = sess.run(
            None,
            {
                "tensor_x": tensor_shard_x,
                "tensor_starts": tensor_starts,
                "tensor_ends": tensor_ends,
                "tensor_axes": tensor_axes,
            },
        )

        expected = tensor_x[:, 0:2][:, rank : rank + 1]
        np.testing.assert_allclose(result[0], expected, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
