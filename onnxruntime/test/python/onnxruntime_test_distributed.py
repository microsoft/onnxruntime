# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
from typing import Tuple

import numpy as np
import onnxscript
from mpi4py import MPI
from onnxscript import FLOAT, FLOAT16, INT64

import onnxruntime as ort

MICROSOFT_OPSET = onnxscript.values.Opset(domain="com.microsoft", version=1)
comm = MPI.COMM_WORLD


def shard_tensor(X, rank, axis, num_shards):
    return np.split(X, num_shards, axis)[rank]


def shard_tensor_per_device_mesh(X, rank, axis, device_mesh):
    if axis is None:
        return X
    shards = np.split(X, len(device_mesh), axis)
    selected_shards = tuple(shard for device_id, shard in zip(device_mesh, shards) if device_id == rank)
    return np.concatenate(selected_shards, axis=axis)


def translate_single_device_mesh(device_mesh: np.ndarray):
    device_mesh_shape = "[" + ",".join(str(dim) for dim in device_mesh.shape) + "]"
    device_mesh_elements = "[" + ",".join(str(elem) for elem in device_mesh.flat) + "]"
    return device_mesh_shape, device_mesh_elements


def translate_all_device_meshes(device_meshes: np.ndarray):
    assert all(len(mesh.shape) == 1 for mesh in device_meshes)
    device_mesh_shapes = []
    device_mesh_elements = []
    for device_mesh in device_meshes:
        device_mesh_shape, device_mesh_element = translate_single_device_mesh(device_mesh)
        device_mesh_shapes.append(device_mesh_shape)
        device_mesh_elements.append(device_mesh_element)
    return device_mesh_shapes, device_mesh_elements


def parse_sharding_spec(spec: str):
    axis_conditions = []
    sharding_device_axes = []
    token_index = 0
    while True:
        token = spec[token_index]
        if token == "R":
            axis_conditions.append("R")
            sharding_device_axes.append(None)
            token_index += 1
        elif token == "S":
            axis_conditions.append("S")
            # Move token pointer to "[""
            token_index += 1
            assert spec[token_index] == "["
            number_tokens = ""
            while True:
                token_index += 1
                token = spec[token_index]
                if token == "]":
                    break
                number_tokens += token
            assert spec[token_index] == "]"
            # Skip "]" and point to next S/R token
            token_index += 1
            sharding_device_axes.append(int(number_tokens))
        else:
            raise ValueError(f"Invalid spec: {spec}")
        if token_index >= len(spec):
            break
    return axis_conditions, sharding_device_axes


def find_shard_axis(axis_conditions, shard_device_axes):
    sharded_axis = None
    sharded_axis_count = 0
    for i, cond in enumerate(axis_conditions):
        if cond == "S":
            sharded_axis = i
            sharded_axis_count += 1
    assert sharded_axis_count in (0, 1), "Can shard at most one axis per tensor."
    if sharded_axis is not None:
        assert shard_device_axes[sharded_axis] == 0, "Device mesh must be 1-D, so 0 is the only valid device mesh axis."
    return sharded_axis


def shard_tensor_per_spec(tensor: np.ndarray, rank: int, spec: str, device_mesh: np.ndarray):
    axis_conditions, shard_device_axes = parse_sharding_spec(spec)
    sharded_axis = find_shard_axis(axis_conditions, shard_device_axes)
    return shard_tensor_per_device_mesh(tensor, rank, sharded_axis, list(device_mesh.flat))


class TestDistributedReshape(unittest.TestCase):
    def _check_distributed_reshape(
        self,
        shape: Tuple[int, ...],
        target_shape: Tuple[int, ...],
        input_device_meshes: np.ndarray,
        input_shard_specs: Tuple[str, ...],
        output_device_meshes: np.ndarray,
        output_shard_specs: Tuple[str, ...],
    ):
        input_device_mesh_shapes, input_device_mesh_elements = translate_all_device_meshes(input_device_meshes)
        output_device_mesh_shapes, output_device_mesh_elements = translate_all_device_meshes(output_device_meshes)

        @onnxscript.script()
        def distributed_reshape_instance(data_tensor: FLOAT, shape_tensor: INT64):
            return MICROSOFT_OPSET.DistributedReshape(
                data_tensor,
                shape_tensor,
                input_device_mesh_shapes=input_device_mesh_shapes,
                input_device_mesh_elements=input_device_mesh_elements,
                input_shard_specs=input_shard_specs,
                output_device_mesh_shapes=output_device_mesh_shapes,
                output_device_mesh_elements=output_device_mesh_elements,
                output_shard_specs=output_shard_specs,
            )

        rank = comm.Get_rank()
        data_tensor = np.arange(np.prod(shape), dtype=np.float32).reshape(*shape)
        shape_tensor = np.array(
            target_shape,
            dtype=np.int64,
        )

        local_data_tensor = shard_tensor_per_spec(data_tensor, rank, input_shard_specs[0], input_device_meshes[0])
        assert "S" not in input_shard_specs[1], "Shape should not be sharded."

        expected = np.reshape(data_tensor, shape_tensor)
        local_expected = shard_tensor_per_spec(expected, rank, output_shard_specs[0], output_device_meshes[0])

        onnx_model = distributed_reshape_instance.to_model_proto(
            input_types=[FLOAT[tuple(local_data_tensor.shape)], INT64[tuple(shape_tensor.shape)]],
            output_types=[FLOAT[tuple(local_expected.shape)]],
        )

        # Each MPI process owns a sharded model.
        sess = ort.InferenceSession(
            onnx_model.SerializeToString(),
            providers=["CUDAExecutionProvider"],
            provider_options=[{"device_id": str(rank)}],
        )

        # Each MPI process executes its sharded model.
        # The result is `local` tensor stored on a specific MPI rank
        # instead of `logical` tensor.
        result = sess.run(
            None,
            {
                "data_tensor": local_data_tensor,
                "shape_tensor": shape_tensor,
            },
        )

        # Compare local tensor and the corresponding logical sub-tensor
        # obtained by sharding logical tensor following output's sharding spec.
        np.testing.assert_allclose(result[0], local_expected, rtol=1e-5, atol=1e-8)

    def test_reshape_two_axis_fusion_shape_2_3_sr_01_shape_6_s_01(self):
        # Two axis fusion.
        # S[0]R, shape=[2, 3], device_mesh=[0, 1] -> S[0], shape = [6], device_mesh=[0, 1]
        self._check_distributed_reshape(
            shape=(
                2,
                3,
            ),
            target_shape=(6,),
            input_device_meshes=[np.array([0, 1])] * 2,
            input_shard_specs=("S[0]R", "R"),
            output_device_meshes=[np.array([0, 1])],
            output_shard_specs=("S[0]",),
        )

    def test_reshape_two_axis_fusion_shape_2_4_rs_01_shape_8_s_0101(self):
        # Two axis fusion.
        # RS[0], shape=[2, 4], device_mesh=[0, 1] -> S[0], shape = [8], device_mesh=[0, 1, 0, 1]
        self._check_distributed_reshape(
            shape=(
                2,
                4,
            ),
            target_shape=(8,),
            input_device_meshes=[np.array([0, 1])] * 2,
            input_shard_specs=("RS[0]", "R"),
            output_device_meshes=[np.array([0, 1, 0, 1])],
            output_shard_specs=("S[0]",),
        )

    def test_reshape_two_axis_fusion_shape_2_3_5_srr_01_shape_2_15_sr_01(self):
        # Two axis fusion.
        # S[0]RR, shape=[2, 3, 5], device_mesh=[0, 1] -> S[0]R, shape = [2, 15], device_mesh=[0, 1]
        self._check_distributed_reshape(
            shape=(
                2,
                3,
                5,
            ),
            target_shape=(
                2,
                15,
            ),
            input_device_meshes=[np.array([0, 1])] * 2,
            input_shard_specs=("S[0]RR", "R"),
            output_device_meshes=[np.array([0, 1])],
            output_shard_specs=("S[0]R",),
        )

    def test_reshape_two_axis_fusion_shape_2_3_5_rsr_01_shape_2_15_sr_01(self):
        # Two axis fusion.
        # RS[0]R, shape=[2, 4, 5], device_mesh=[0, 1] -> RS[0], shape = [2, 20], device_mesh=[0, 1]
        self._check_distributed_reshape(
            shape=(
                2,
                4,
                5,
            ),
            target_shape=(
                2,
                20,
            ),
            input_device_meshes=[np.array([0, 1])] * 2,
            input_shard_specs=("RS[0]R", "R"),
            output_device_meshes=[np.array([0, 1])],
            output_shard_specs=("RS[0]",),
        )

    def test_reshape_two_axis_fusion_shape_2_3_6_rrs_01_shape_2_18_rs_010101(self):
        # Two axis fusion.
        # RRS[0], shape=[2, 3, 6], device_mesh=[0, 1] -> RS[0], shape = [2, 18], device_mesh=[0, 1, 0, 1, 0, 1]
        self._check_distributed_reshape(
            shape=(
                2,
                3,
                6,
            ),
            target_shape=(
                2,
                18,
            ),
            input_device_meshes=[np.array([0, 1])] * 2,
            input_shard_specs=("RRS[0]", "R"),
            output_device_meshes=[np.array([0, 1, 0, 1, 0, 1])],
            output_shard_specs=("RS[0]",),
        )
        # Two axis fusion.
        # RRS[0], shape=[2, 3, 8], device_mesh=[0, 1, 0, 1] -> RS[0], shape = [2, 24], device_mesh=[0, 1, 0, 1] * 3

        # Two axis fusion.
        # RS[0]R, shape=[2, 8, 3], device_mesh=[0, 1, 0, 1] -> RS[0], shape = [2, 24], device_mesh=[0, 1, 0, 1]

    def test_reshape_two_axis_decomposition_shape_6_s_01_shape_2_3_sr_01(self):
        # Two axis decomposition
        # S[0], shape=[6], device_mesh=[0, 1] -> S[0]R, shape=[2, 3], device_mesh=[0, 1]
        self._check_distributed_reshape(
            shape=(6,),
            target_shape=(
                2,
                3,
            ),
            input_device_meshes=[np.array([0, 1])] * 2,
            input_shard_specs=("S[0]", "R"),
            output_device_meshes=[np.array([0, 1])],
            output_shard_specs=("S[0]R",),
        )

    def test_reshape_two_axis_decomposition_shape_16_s_01_shape_1_16_sr_01(self):
        # Two axis decomposition
        # S[0], shape=[16], device_mesh=[0, 1] -> RS[0], shape=[1, 16], device_mesh=[0, 1]
        self._check_distributed_reshape(
            shape=(16,),
            target_shape=(
                1,
                16,
            ),
            input_device_meshes=[np.array([0, 1])] * 2,
            input_shard_specs=("S[0]", "R"),
            output_device_meshes=[np.array([0, 1])],
            output_shard_specs=("RS[0]",),
        )

    def test_reshape_two_axis_decomposition_shape_16_s_01_shape_2_8_sr_01(self):
        # Two axis decomposition
        # S[0], shape=[16], device_mesh=[0, 1] -> S[0]R, shape=[2, 8], device_mesh=[0, 1]
        self._check_distributed_reshape(
            shape=(16,),
            target_shape=(
                2,
                8,
            ),
            input_device_meshes=[np.array([0, 1])] * 2,
            input_shard_specs=("S[0]", "R"),
            output_device_meshes=[np.array([0, 1])],
            output_shard_specs=("S[0]R",),
        )

    def test_reshape_two_axis_decomposition_shape_16_s_01_shape_4_4_sr_01(self):
        # Two axis decomposition
        # S[0], shape=[16], device_mesh=[0, 1] -> S[0]R, shape=[4, 4], device_mesh=[0, 1]
        self._check_distributed_reshape(
            shape=(16,),
            target_shape=(
                4,
                4,
            ),
            input_device_meshes=[np.array([0, 1])] * 2,
            input_shard_specs=("S[0]", "R"),
            output_device_meshes=[np.array([0, 1])],
            output_shard_specs=("S[0]R",),
        )

    def test_reshape_two_axis_decomposition_shape_16_s_01_shape_8_2_sr_01(self):
        # Two axis decomposition
        # S[0], shape=[16], device_mesh=[0, 1] -> S[0]R, shape=[8, 2], device_mesh=[0, 1]
        self._check_distributed_reshape(
            shape=(16,),
            target_shape=(
                8,
                2,
            ),
            input_device_meshes=[np.array([0, 1])] * 2,
            input_shard_specs=("S[0]", "R"),
            output_device_meshes=[np.array([0, 1])],
            output_shard_specs=("S[0]R",),
        )

    def test_reshape_two_axis_decomposition_shape_16_s_01_shape_16_1_sr_01(self):
        # Two axis decomposition
        # S[0], shape=[16], device_mesh=[0, 1] -> S[0]R, shape=[16, 1], device_mesh=[0, 1]
        self._check_distributed_reshape(
            shape=(16,),
            target_shape=(
                16,
                1,
            ),
            input_device_meshes=[np.array([0, 1])] * 2,
            input_shard_specs=("S[0]", "R"),
            output_device_meshes=[np.array([0, 1])],
            output_shard_specs=("S[0]R",),
        )

    def test_reshape_two_axis_decomposition_shape_16_s_0101_shape_1_16_sr_0101(self):
        # Two axis decomposition
        # S[0], shape=[16], device_mesh=[0, 1, 0, 1] -> RS[0], shape=[1, 16], device_mesh=[0, 1, 0, 1]

        self._check_distributed_reshape(
            shape=(16,),
            target_shape=(
                1,
                16,
            ),
            input_device_meshes=[np.array([0, 1, 0, 1])] * 2,
            input_shard_specs=("S[0]", "R"),
            output_device_meshes=[np.array([0, 1, 0, 1])],
            output_shard_specs=("RS[0]",),
        )

    def test_reshape_two_axis_decomposition_shape_16_s_0101_shape_2_8_rs_01(self):
        # Two axis decomposition
        #                                 repeats=2                       8 = repeats * [unique IDs]
        # S[0], shape=[16], device_mesh=[0, 1, 0, 1] -> RS[0], shape=[2, 8], device_mesh=[0, 1]
        self._check_distributed_reshape(
            shape=(16,),
            target_shape=(
                2,
                8,
            ),
            input_device_meshes=[np.array([0, 1, 0, 1])] * 2,
            input_shard_specs=("S[0]", "R"),
            output_device_meshes=[np.array([0, 1])],
            output_shard_specs=("RS[0]",),
        )

    def test_reshape_two_axis_decomposition_shape_16_s_0101_shape_4_4_sr_0101(self):
        # Two axis decomposition
        # S[0], shape=[16], device_mesh=[0, 1, 0, 1] -> S[0]R, shape=[4, 4], device_mesh=[0, 1, 0, 1]
        self._check_distributed_reshape(
            shape=(16,),
            target_shape=(
                4,
                4,
            ),
            input_device_meshes=[np.array([0, 1, 0, 1])] * 2,
            input_shard_specs=("S[0]", "R"),
            output_device_meshes=[np.array([0, 1, 0, 1])],
            output_shard_specs=("S[0]R",),
        )

    def test_reshape_two_axis_decomposition_shape_16_s_0101_shape_8_2_sr_0101(self):
        # Two axis decomposition
        # S[0], shape=[16], device_mesh=[0, 1, 0, 1] -> S[0]R, shape=[8, 2], device_mesh=[0, 1, 0, 1]
        self._check_distributed_reshape(
            shape=(16,),
            target_shape=(
                8,
                2,
            ),
            input_device_meshes=[np.array([0, 1, 0, 1])] * 2,
            input_shard_specs=("S[0]", "R"),
            output_device_meshes=[np.array([0, 1, 0, 1])],
            output_shard_specs=("S[0]R",),
        )

    def test_reshape_two_axis_decomposition_shape_16_s_0101_shape_16_1_sr_0101(self):
        # Two axis decomposition
        # S[0], shape=[16], device_mesh=[0, 1, 0, 1] -> S[0]R, shape=[16, 1], device_mesh=[0, 1, 0, 1]
        self._check_distributed_reshape(
            shape=(16,),
            target_shape=(
                16,
                1,
            ),
            input_device_meshes=[np.array([0, 1, 0, 1])] * 2,
            input_shard_specs=("S[0]", "R"),
            output_device_meshes=[np.array([0, 1, 0, 1])],
            output_shard_specs=("S[0]R",),
        )

    def test_reshape_two_axis_decomposition_shape_21_4096_s_01_shape_3_7_4096_rrs_01(self):
        # Two axis decomposition
        # [21, 4096] -> [3, 7, 4096]
        # data: (21, 2048), (RS, [0, 1])
        # shape: None, (R, [0, 1])
        # reshaped: None, None
        # -----------------------------------
        # new reshaped: None, (RRS, [0, 1])
        self._check_distributed_reshape(
            shape=(
                21,
                4096,
            ),
            target_shape=(
                3,
                7,
                4096,
            ),
            input_device_meshes=[np.array([0, 1])] * 2,
            input_shard_specs=("RS[0]", "R"),
            output_device_meshes=[np.array([0, 1])],
            output_shard_specs=("RRS[0]",),
        )

    def test_reshape_two_axis_decomposition_shape_3_7_4096_rrs_01_shape_3_7_64_64_rrsr_01(self):
        # Two axis decomposition
        # [3, 7, 4096] -> [3, 7, 64, 64]
        # data: (3, 7, 2048), (RRS, [0, 1])
        # shape: None, (R, [0, 1])
        # reshaped: None, None
        # -----------------------------------
        # new reshaped: None, (RRSR, [0, 1])

        self._check_distributed_reshape(
            shape=(
                3,
                7,
                4096,
            ),
            target_shape=(
                3,
                7,
                64,
                64,
            ),
            input_device_meshes=[np.array([0, 1])] * 2,
            input_shard_specs=("RRS[0]", "R"),
            output_device_meshes=[np.array([0, 1])],
            output_shard_specs=("RRS[0]R",),
        )

    def test_reshape_two_axis_fusion_shape_3_7_4096_rrr_01_shape_21_4906_rr_01(self):
        # Two axis fusion
        # [3, 7, 4096] -> [21, 4096]
        # data: (3, 7, 4096), (RRR, [0, 1])
        # shape: None, (R, [0, 1])
        # reshaped: None, None
        # -----------------------------------
        # new reshaped: None, (RR, [0, 1])
        self._check_distributed_reshape(
            shape=(
                3,
                7,
                4096,
            ),
            target_shape=(
                21,
                4096,
            ),
            input_device_meshes=[np.array([0, 1])] * 2,
            input_shard_specs=("RRR", "R"),
            output_device_meshes=[np.array([0, 1])],
            output_shard_specs=("RR",),
        )

    def test_reshape_two_axis_fusion_shape_21_4096_rrr_01_shape_3_7_4906_rr_01(self):
        # Two axis fusion
        # [21, 4096] -> [3, 7, 4096]
        # data: (21, 4096), (RR, [0, 1])
        # shape: None, (R, [0, 1])
        # reshaped: None, None
        # -----------------------------------
        # new reshaped: None, (RRR, [0, 1])
        self._check_distributed_reshape(
            shape=(
                21,
                4096,
            ),
            target_shape=(
                3,
                7,
                4096,
            ),
            input_device_meshes=[np.array([0, 1])] * 2,
            input_shard_specs=("RR", "R"),
            output_device_meshes=[np.array([0, 1])],
            output_shard_specs=("RRR",),
        )

    def test_reshape_two_axis_fusion_shape_3_64_7_64_rsrr_01_shape_192_7_64_srr_010101(self):
        # Two axis fusion
        # [3, 64, 7, 64] -> [192, 7, 64]
        # data: (3, 32, 7, 64), (RSRR, [0, 1])
        # shape: None, (R, [0, 1])
        # reshaped: None, None
        # -----------------------------------
        # new reshaped: None, (SRR, [0, 1, 0, 1, 0, 1])

        self._check_distributed_reshape(
            shape=(
                3,
                64,
                7,
                64,
            ),
            target_shape=(
                192,
                7,
                64,
            ),
            input_device_meshes=[np.array([0, 1])] * 2,
            input_shard_specs=("RS[0]RR", "R"),
            output_device_meshes=[np.array([0, 1, 0, 1, 0, 1])],
            output_shard_specs=("S[0]RR",),
        )

    def test_reshape_two_axis_decomposition_shape_192_7_7_srr_010101_shape_3_64_7_7_rsrr_01(self):
        # Two axis decomposition
        # [192, 7, 7] -> [3, 64, 7, 7]
        # data: (96, 7, 7), (SRR, [0, 1, 0, 1, 0, 1])
        # shape: None, (R, [0, 1])
        # reshaped: None, None
        # -----------------------------------
        # new reshaped: None, (RSRR, [0.0, 1.0])

        self._check_distributed_reshape(
            shape=(
                192,
                7,
                7,
            ),
            target_shape=(
                3,
                64,
                7,
                7,
            ),
            input_device_meshes=[np.array([0, 1, 0, 1, 0, 1])] * 2,
            input_shard_specs=("S[0]RR", "R"),
            output_device_meshes=[np.array([0, 1])],
            output_shard_specs=("RS[0]RR",),
        )

    def test_reshape_two_axis_fusion_shape_3_64_7_7_rsrr_01_shape_192_7_7_srr_010101(self):
        # Two axis fusion
        # [3, 64, 7, 7] -> [192, 7, 7]
        # data: (3, 32, 7, 7), (RSRR, [0, 1])
        # shape: None, (R, [0, 1])
        # reshaped: None, None
        # -----------------------------------
        # new reshaped: None, (SRR, [0, 1, 0, 1, 0, 1])

        self._check_distributed_reshape(
            shape=(
                3,
                64,
                7,
                7,
            ),
            target_shape=(
                192,
                7,
                7,
            ),
            input_device_meshes=[np.array([0, 1])] * 2,
            input_shard_specs=("RS[0]RR", "R"),
            output_device_meshes=[np.array([0, 1, 0, 1, 0, 1])],
            output_shard_specs=("S[0]RR",),
        )

    def test_reshape_two_axis_decomposition_shape_192_7_64_srr_010101_shape_3_64_7_64_rsrr_01(self):
        # Two axis decomposition
        # [192, 7, 64] -> [3, 64, 7, 64]
        # data: (96, 7, 64), (SRR, [0, 1, 0, 1, 0, 1])
        # shape: None, (R, [0, 1])
        # reshaped: None, None
        # -----------------------------------
        # new reshaped: None, (RSRR, [0.0, 1.0])

        self._check_distributed_reshape(
            shape=(
                192,
                7,
                64,
            ),
            target_shape=(
                3,
                64,
                7,
                64,
            ),
            input_device_meshes=[np.array([0, 1, 0, 1, 0, 1])] * 2,
            input_shard_specs=("S[0]RR", "R"),
            output_device_meshes=[np.array([0, 1])],
            output_shard_specs=("RS[0]RR",),
        )

    def test_reshape_two_axis_fusion_shape_3_7_64_64_rrsr_01_shape_3_7_4096_rrs_01(self):
        # Two axis fusion
        # [3, 7, 64, 64] -> [3, 7, 4096]
        # data: (3, 7, 32, 64), (RRSR, [0, 1])
        # shape: None, (R, [0, 1])
        # reshaped: None, None
        # -----------------------------------
        # new reshaped: None, (RRS, [0, 1])

        self._check_distributed_reshape(
            shape=(
                3,
                7,
                64,
                64,
            ),
            target_shape=(
                3,
                7,
                4096,
            ),
            input_device_meshes=[np.array([0, 1])] * 2,
            input_shard_specs=("RRS[0]R", "R"),
            output_device_meshes=[np.array([0, 1])],
            output_shard_specs=("RRS[0]",),
        )

    def test_reshape_two_axis_fusion_shape_3_7_4096_rrs_01_shape_21_4906_rs_01(self):
        # Two axis fusion
        # [3, 7, 4096] -> [21, 4096]
        # data: (3, 7, 2048), (RRS, [0, 1])
        # shape: None, (R, [0, 1])
        # reshaped: None, None
        # -----------------------------------
        # new reshaped: None, (RS, [0, 1])
        self._check_distributed_reshape(
            shape=(
                3,
                7,
                4096,
            ),
            target_shape=(
                21,
                4096,
            ),
            input_device_meshes=[np.array([0, 1])] * 2,
            input_shard_specs=("RRS[0]", "R"),
            output_device_meshes=[np.array([0, 1])],
            output_shard_specs=("RS[0]",),
        )


class TestDistributedExpand(unittest.TestCase):
    def _check_distributed_expand(
        self,
        shape: Tuple[int, ...],
        target_shape: Tuple[int, ...],
        input_device_meshes: np.ndarray,
        input_shard_specs: Tuple[str, ...],
        output_device_meshes: np.ndarray,
        output_shard_specs: Tuple[str, ...],
    ):
        assert len(input_device_meshes) == len(input_shard_specs)
        assert len(output_device_meshes) == len(output_shard_specs)

        input_device_mesh_shapes, input_device_mesh_elements = translate_all_device_meshes(input_device_meshes)
        output_device_mesh_shapes, output_device_mesh_elements = translate_all_device_meshes(output_device_meshes)

        @onnxscript.script()
        def distributed_expand_instance(data_tensor: FLOAT, shape_tensor: INT64):
            return MICROSOFT_OPSET.DistributedExpand(
                data_tensor,
                shape_tensor,
                input_device_mesh_shapes=input_device_mesh_shapes,
                input_device_mesh_elements=input_device_mesh_elements,
                input_shard_specs=input_shard_specs,
                output_device_mesh_shapes=output_device_mesh_shapes,
                output_device_mesh_elements=output_device_mesh_elements,
                output_shard_specs=output_shard_specs,
            )

        rank = comm.Get_rank()
        data_tensor = np.arange(np.prod(shape), dtype=np.float32).reshape(*shape)
        shape_tensor = np.array(
            target_shape,
            dtype=np.int64,
        )

        local_data_tensor = shard_tensor_per_spec(data_tensor, rank, input_shard_specs[0], input_device_meshes[0])
        assert "S" not in input_shard_specs[1], "Shape should not be sharded."

        expected = data_tensor * np.ones(shape_tensor)
        local_expected = shard_tensor_per_spec(expected, rank, output_shard_specs[0], output_device_meshes[0])

        onnx_model = distributed_expand_instance.to_model_proto(
            input_types=[FLOAT[tuple(local_data_tensor.shape)], INT64[tuple(shape_tensor.shape)]],
            output_types=[FLOAT[tuple(local_expected.shape)]],
        )

        # Each MPI process owns a sharded model.
        sess = ort.InferenceSession(
            onnx_model.SerializeToString(),
            providers=["CUDAExecutionProvider"],
            provider_options=[{"device_id": str(rank)}],
        )

        # Each MPI process executes its sharded model.
        # The result is `local` tensor stored on a specific MPI rank
        # instead of `logical` tensor.
        result = sess.run(
            None,
            {
                "data_tensor": local_data_tensor,
                "shape_tensor": shape_tensor,
            },
        )

        # Compare local tensor and the corresponding logical sub-tensor
        # obtained by sharding logical tensor following output's sharding spec.
        np.testing.assert_allclose(result[0], local_expected, rtol=1e-5, atol=1e-8)

    def test_expand_sharded_on_expanded_axis(self):
        # data: shape=[8,1], spec=(RR, [0,1])
        # shape: shape=[2], spec=(R, [0,1]), value=[1,4]
        # output: shape=[8,4], spec=(RS, [0,1])
        self._check_distributed_expand(
            shape=(
                8,
                1,
            ),
            target_shape=(
                8,
                4,
            ),
            input_device_meshes=[np.array([0, 1])] * 2,
            input_shard_specs=("RR", "R"),
            output_device_meshes=[np.array([0, 1])],
            output_shard_specs=("RS[0]",),
        )

    def test_expand_sharded_on_expanded_axis_with_device_mesh_0101(self):
        # data: shape=[8,1], spec=(RR, [0,1])
        # shape: shape=[2], spec=(R, [0,1]), value=[1,4]
        # output: shape=[8,4], spec=(RS, [0,1])
        self._check_distributed_expand(
            shape=(
                8,
                1,
            ),
            target_shape=(
                8,
                8,
            ),
            input_device_meshes=[np.array([0, 1])] * 2,
            input_shard_specs=("RR", "R"),
            output_device_meshes=[np.array([0, 1, 0, 1])],
            output_shard_specs=("RS[0]",),
        )

    def test_expand_replicated_on_expanded_axis(self):
        # data: shape=[8,1], spec=(RR, [0,1])
        # shape: shape=[2], spec=(R, [0,1]), value=[1,4]
        # output: shape=[8,4], spec=(RR, [0,1])
        self._check_distributed_expand(
            shape=(
                8,
                1,
            ),
            target_shape=(
                1,
                4,
            ),
            input_device_meshes=[np.array([0, 1])] * 2,
            input_shard_specs=("RR", "R"),
            output_device_meshes=[np.array([0, 1])],
            output_shard_specs=("RR",),
        )

    def test_expand_with_pass_through_sharding_spec(self):
        # data: shape=[8,1], spec=(SR, [0,1])
        # shape: shape=[2], spec=(R, [0,1]), value=[1,4]
        # output: shape=[8,4], spec=(SR, [0,1])
        self._check_distributed_expand(
            shape=(
                8,
                1,
            ),
            target_shape=(
                1,
                4,
            ),
            input_device_meshes=[np.array([0, 1])] * 2,
            input_shard_specs=(
                "S[0]R",
                "R",
            ),
            output_device_meshes=[np.array([0, 1])],
            output_shard_specs=("S[0]R",),
        )

    def test_expand_in_tiny_llama(self):
        # data: shape=[2,4,256,4], spec=(RSRR, [0,1])
        # shape: shape=[4], spec=(R, [0,1,2,3]), value=[2,4,256,4]
        # output: shape=[2,4,256,4], spec=(RSRR, [0,1])
        self._check_distributed_expand(
            shape=(
                2,
                4,
                256,
                4,
            ),
            target_shape=(
                2,
                4,
                256,
                4,
            ),
            input_device_meshes=[np.array([0, 1])] * 2,
            input_shard_specs=("RS[0]RR", "R"),
            output_device_meshes=[np.array([0, 1])],
            output_shard_specs=("RS[0]RR",),
        )


class TestDistributedReduce(unittest.TestCase):
    def _check_distributed_reduce(
        self,
        keepdims: int,
        dtype: np.dtype,
        shape: Tuple[int, ...],
        axes: Tuple[int, ...],
        input_device_meshes: np.ndarray,
        input_shard_specs: Tuple[str, ...],
        output_device_meshes: np.ndarray,
        output_shard_specs: Tuple[str, ...],
    ):
        assert len(input_device_meshes) == len(input_shard_specs)
        assert len(output_device_meshes) == len(output_shard_specs)

        input_device_mesh_shapes, input_device_mesh_elements = translate_all_device_meshes(input_device_meshes)
        output_device_mesh_shapes, output_device_mesh_elements = translate_all_device_meshes(output_device_meshes)

        @onnxscript.script()
        def distributed_reduce_sum_instance(data_tensor: FLOAT, axes_tensor: INT64):
            return MICROSOFT_OPSET.DistributedReduceSum(
                data_tensor,
                axes_tensor,
                keepdims=keepdims,
                input_device_mesh_shapes=input_device_mesh_shapes,
                input_device_mesh_elements=input_device_mesh_elements,
                input_shard_specs=input_shard_specs,
                output_device_mesh_shapes=output_device_mesh_shapes,
                output_device_mesh_elements=output_device_mesh_elements,
                output_shard_specs=output_shard_specs,
            )

        @onnxscript.script()
        def distributed_reduce_max_instance(data_tensor: FLOAT, axes_tensor: INT64):
            return MICROSOFT_OPSET.DistributedReduceMax(
                data_tensor,
                axes_tensor,
                keepdims=keepdims,
                input_device_mesh_shapes=input_device_mesh_shapes,
                input_device_mesh_elements=input_device_mesh_elements,
                input_shard_specs=input_shard_specs,
                output_device_mesh_shapes=output_device_mesh_shapes,
                output_device_mesh_elements=output_device_mesh_elements,
                output_shard_specs=output_shard_specs,
            )

        @onnxscript.script()
        def distributed_reduce_mean_instance(data_tensor: FLOAT, axes_tensor: INT64):
            return MICROSOFT_OPSET.DistributedReduceMean(
                data_tensor,
                axes_tensor,
                keepdims=keepdims,
                input_device_mesh_shapes=input_device_mesh_shapes,
                input_device_mesh_elements=input_device_mesh_elements,
                input_shard_specs=input_shard_specs,
                output_device_mesh_shapes=output_device_mesh_shapes,
                output_device_mesh_elements=output_device_mesh_elements,
                output_shard_specs=output_shard_specs,
            )

        rank = comm.Get_rank()

        for onnx_func, np_func in zip(
            [distributed_reduce_sum_instance, distributed_reduce_max_instance, distributed_reduce_mean_instance],
            [np.sum, np.maximum.reduce, np.mean],
        ):
            data = np.random.randint(4, size=shape).astype(dtype)
            expected = np_func(data, axis=axes, keepdims=bool(keepdims))

            assert len(input_shard_specs) == 2 and len(input_device_meshes) == 2, "Reduce has two inputs."
            assert "S" not in input_shard_specs[1], "Tensor `axes` should not be sharded."
            assert len(output_shard_specs) == 1 and len(output_device_meshes) == 1, "Reduce has only one output."

            local_data = shard_tensor_per_spec(data, rank, input_shard_specs[0], input_device_meshes[0])
            local_expected = shard_tensor_per_spec(expected, rank, output_shard_specs[0], output_device_meshes[0])

            if dtype == np.float32:
                onnx_model = onnx_func.to_model_proto(
                    input_types=[FLOAT[tuple(local_data.shape)], INT64[len(axes)]],
                    output_types=[FLOAT[tuple(local_expected.shape)]],
                )
            elif dtype == np.int64:
                onnx_model = onnx_func.to_model_proto(
                    input_types=[INT64[tuple(local_data.shape)], INT64[len(axes)]],
                    output_types=[INT64[tuple(local_expected.shape)]],
                )
            elif dtype == np.float16:
                onnx_model = onnx_func.to_model_proto(
                    input_types=[FLOAT16[tuple(local_data.shape)], INT64[len(axes)]],
                    output_types=[FLOAT16[tuple(local_expected.shape)]],
                )
            else:
                raise RuntimeError(f"Unsupported dtype: {dtype}")

            # Each MPI process owns a sharded model.
            sess = ort.InferenceSession(
                onnx_model.SerializeToString(),
                providers=["CUDAExecutionProvider"],
                provider_options=[{"device_id": str(rank)}],
            )

            # Each MPI process executes its sharded model.
            # The result is `local` tensor stored on a specific MPI rank
            # instead of `logical` tensor.
            result = sess.run(
                None,
                {
                    "data_tensor": local_data,
                    "axes_tensor": np.array(axes, dtype=np.int64),
                },
            )

            # Compare local tensor and the corresponding logical sub-tensor
            # obtained by sharding logical tensor following output's sharding spec.
            np.testing.assert_allclose(result[0], local_expected, rtol=1e-5, atol=1e-8)

    def test_reduce(self):
        self._check_distributed_reduce(
            keepdims=1,
            dtype=np.float32,
            shape=(
                8,
                4,
            ),
            axes=(0,),
            input_device_meshes=[np.array([0, 1])] * 2,
            input_shard_specs=("RR", "R"),
            output_device_meshes=[np.array([0, 1])],
            output_shard_specs=("RR",),
        )

    def test_reduce_sharded(self):
        self._check_distributed_reduce(
            keepdims=1,
            dtype=np.float32,
            shape=(
                8,
                4,
            ),
            axes=(1,),
            input_device_meshes=[np.array([0, 1])] * 2,
            input_shard_specs=("S[0]R", "R"),
            output_device_meshes=[np.array([0, 1])],
            output_shard_specs=("S[0]R",),
        )


class TestDistributed(unittest.TestCase):
    def test_matmul_rs_sr_rr(self):
        # It means 1-D tensor with single element: [2].
        device_mesh_shape = "[2]"
        # It means 1-D tensor with two elements: [0, 1].
        device_mesh_elements = "[0,1]"

        @onnxscript.script()
        def matmul_rs_sr_rr(tensor_x: FLOAT, tensor_w: FLOAT) -> FLOAT:
            return MICROSOFT_OPSET.DistributedMatMul(
                tensor_x,
                tensor_w,
                input_shard_specs=["RS[0]", "S[0]R"],
                output_shard_specs=["RR"],
                input_device_mesh_shapes=[device_mesh_shape, device_mesh_shape],
                input_device_mesh_elements=[device_mesh_elements, device_mesh_elements],
                output_device_mesh_shapes=[device_mesh_shape],
                output_device_mesh_elements=[device_mesh_elements],
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
        device_mesh_shape = "[2]"
        device_mesh_elements = "[0, 1]"

        @onnxscript.script()
        def matmul_rs_rs_rr(tensor_x: FLOAT, tensor_w: FLOAT) -> FLOAT:
            return MICROSOFT_OPSET.DistributedMatMul(
                tensor_x,
                tensor_w,
                input_shard_specs=["RS[0]", "RS[0]"],
                output_shard_specs=["RR"],
                input_device_mesh_shapes=[device_mesh_shape, device_mesh_shape],
                input_device_mesh_elements=[device_mesh_elements, device_mesh_elements],
                output_device_mesh_shapes=[device_mesh_shape],
                output_device_mesh_elements=[device_mesh_elements],
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
        device_mesh_shape = "[2]"
        device_mesh_elements = "[0, 1]"

        @onnxscript.script()
        def matmul2d_rs_rs_rs(tensor_x: FLOAT, tensor_w: FLOAT) -> FLOAT:
            return MICROSOFT_OPSET.DistributedMatMul(
                tensor_x,
                tensor_w,
                input_shard_specs=["RS[0]", "RS[0]"],
                output_shard_specs=["RS[0]"],
                input_device_mesh_shapes=[device_mesh_shape, device_mesh_shape],
                input_device_mesh_elements=[device_mesh_elements, device_mesh_elements],
                output_device_mesh_shapes=[device_mesh_shape],
                output_device_mesh_elements=[device_mesh_elements],
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
        device_mesh_shape = "[2]"
        device_mesh_elements = "[0, 1]"

        @onnxscript.script()
        def matmul_srr_rr_srr(tensor_x: FLOAT, tensor_w: FLOAT) -> FLOAT:
            return MICROSOFT_OPSET.DistributedMatMul(
                tensor_x,
                tensor_w,
                input_shard_specs=["S[0]RR", "RR"],
                output_shard_specs=["S[0]RR"],
                input_device_mesh_shapes=[device_mesh_shape, device_mesh_shape],
                input_device_mesh_elements=[device_mesh_elements, device_mesh_elements],
                output_device_mesh_shapes=[device_mesh_shape],
                output_device_mesh_elements=[device_mesh_elements],
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
        device_mesh_shape = "[2]"
        device_mesh_elements = "[0, 1]"

        @onnxscript.script()
        def matmul_srr_rrrr_rsrr(tensor_x: FLOAT, tensor_w: FLOAT) -> FLOAT:
            return MICROSOFT_OPSET.DistributedMatMul(
                tensor_x,
                tensor_w,
                input_shard_specs=["S[0]RR", "RRRR"],
                output_shard_specs=["RS[0]RR"],
                input_device_mesh_shapes=[device_mesh_shape, device_mesh_shape],
                input_device_mesh_elements=[device_mesh_elements, device_mesh_elements],
                output_device_mesh_shapes=[device_mesh_shape],
                output_device_mesh_elements=[device_mesh_elements],
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
        device_mesh_shape = "[2]"
        device_mesh_elements = "[0, 1]"

        @onnxscript.script()
        def matmul_sr_rs_rr(tensor_x: FLOAT, tensor_w: FLOAT) -> FLOAT:
            return MICROSOFT_OPSET.DistributedMatMul(
                tensor_x,
                tensor_w,
                input_shard_specs=["S[0]R", "RS[0]"],
                output_shard_specs=["RR"],
                input_device_mesh_shapes=[device_mesh_shape, device_mesh_shape],
                input_device_mesh_elements=[device_mesh_elements, device_mesh_elements],
                output_device_mesh_shapes=[device_mesh_shape],
                output_device_mesh_elements=[device_mesh_elements],
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
        device_mesh_shape = "[2]"
        device_mesh_elements = "[0, 1]"

        @onnxscript.script()
        def matmul_rr_rs_rs(tensor_x: FLOAT, tensor_w: FLOAT) -> FLOAT:
            return MICROSOFT_OPSET.DistributedMatMul(
                tensor_x,
                tensor_w,
                input_shard_specs=["RR", "RS[0]"],
                output_shard_specs=["RS[0]"],
                input_device_mesh_shapes=[device_mesh_shape, device_mesh_shape],
                input_device_mesh_elements=[device_mesh_elements, device_mesh_elements],
                output_device_mesh_shapes=[device_mesh_shape],
                output_device_mesh_elements=[device_mesh_elements],
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
        device_mesh_shape = "[2]"
        device_mesh_elements = "[0, 1]"

        @onnxscript.script()
        def matmul_rr_sr_rr(tensor_x: FLOAT, tensor_w: FLOAT) -> FLOAT:
            return MICROSOFT_OPSET.DistributedMatMul(
                tensor_x,
                tensor_w,
                input_shard_specs=["RR", "S[0]R"],
                output_shard_specs=["RR"],
                input_device_mesh_shapes=[device_mesh_shape, device_mesh_shape],
                input_device_mesh_elements=[device_mesh_elements, device_mesh_elements],
                output_device_mesh_shapes=[device_mesh_shape],
                output_device_mesh_elements=[device_mesh_elements],
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
        device_mesh_shape = "[2]"
        device_mesh_elements = "[0, 1]"

        @onnxscript.script()
        def slice_sr_axis1(tensor_x: FLOAT, tensor_starts: INT64, tensor_ends: INT64, tensor_axes: INT64) -> FLOAT:
            return MICROSOFT_OPSET.DistributedSlice(
                tensor_x,
                tensor_starts,
                tensor_ends,
                tensor_axes,
                input_shard_specs=["S[0]R", "R", "R", "R", "R"],
                output_shard_specs=["S[0]R"],
                input_device_mesh_shapes=[device_mesh_shape] * 5,
                input_device_mesh_elements=[device_mesh_elements] * 5,
                output_device_mesh_shapes=[device_mesh_shape],
                output_device_mesh_elements=[device_mesh_elements],
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
        device_mesh_shape = "[2]"
        device_mesh_elements = "[0, 1]"

        @onnxscript.script()
        def slice_sr_axis1(tensor_x: FLOAT, tensor_starts: INT64, tensor_ends: INT64, tensor_axes: INT64) -> FLOAT:
            return MICROSOFT_OPSET.DistributedSlice(
                tensor_x,
                tensor_starts,
                tensor_ends,
                tensor_axes,
                input_shard_specs=["RS[0]", "R", "R", "R", "R"],
                output_shard_specs=["RS[0]"],
                input_device_mesh_shapes=[device_mesh_shape] * 5,
                input_device_mesh_elements=[device_mesh_elements] * 5,
                output_device_mesh_shapes=[device_mesh_shape],
                output_device_mesh_elements=[device_mesh_elements],
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
