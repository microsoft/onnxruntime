# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import unittest

import numpy as np
from mpi4py import MPI
from onnx import TensorProto, helper
from parameterized import parameterized

import onnxruntime as ort


class ORTBertPretrainTest(unittest.TestCase):
    @staticmethod
    def _create_model_with_opsets(
        graph_def,
    ):
        opset_imports = [
            helper.make_operatorsetid("", 18),
            helper.make_operatorsetid(".com.microsoft", 1),
        ]
        return helper.make_model(
            graph_def,
            producer_name="ORTBertPretrainTest in onnxruntime_test_collective.py",
            opset_imports=opset_imports,
        )

    def _create_allreduce_ut_model(self, shape, elem_type: TensorProto.DataType = TensorProto.FLOAT):
        X = helper.make_tensor_value_info("X", elem_type, shape)  # noqa: N806
        Y = helper.make_tensor_value_info("Y", elem_type, shape)  # noqa: N806
        node_def = helper.make_node("AllReduce", ["X"], ["Y"], domain="com.microsoft")
        graph_def = helper.make_graph(
            [node_def],
            "",
            [X],
            [Y],
        )
        return ORTBertPretrainTest._create_model_with_opsets(graph_def)

    def _get_rank_size(self):
        comm = MPI.COMM_WORLD
        return comm.Get_rank(), comm.Get_size()

    def _create_allgather_ut_model(
        self,
        shape,
        axis,
        # Element type for AllGather's input and output.
        elem_type: TensorProto.DataType = TensorProto.FLOAT,
        # Element type for Model's input and output.
        communication_elem_type: TensorProto.DataType = TensorProto.FLOAT,
    ):
        X = helper.make_tensor_value_info("X", elem_type, shape)  # noqa: N806
        _, group_size = self._get_rank_size()
        output_shape = [s * group_size if axis_index == axis else s for axis_index, s in enumerate(shape)]
        Y = helper.make_tensor_value_info("Y", elem_type, output_shape)  # noqa: N806
        if elem_type != communication_elem_type:
            # With elem_type and external_element_type, we use the pattern
            #   model input type -> Cast -> elem_type -> AllGather -> elem_type -> Cast -> model output type
            # so that we can test boolean tensors and other special types.
            node_defs = [
                helper.make_node("Cast", ["X"], ["X_casted"], to=communication_elem_type),
                helper.make_node(
                    "AllGather", ["X_casted"], ["Y_casted"], domain="com.microsoft", group_size=group_size, axis=axis
                ),
                helper.make_node("Cast", ["Y_casted"], ["Y"], to=elem_type),
            ]
        else:
            # When elem_type == external_element_type, the pattern
            #   model input type -> Cast -> elem_type -> AllGather -> elem_type -> Cast -> model output type
            # is reduced to
            #   model input type -> AllGather -> model output type
            node_defs = [
                helper.make_node("AllGather", ["X"], ["Y"], domain="com.microsoft", group_size=group_size, axis=axis),
            ]
        graph_def = helper.make_graph(
            node_defs,
            "",
            [X],
            [Y],
        )
        return ORTBertPretrainTest._create_model_with_opsets(graph_def)

    def _create_alltoall_ut_model(
        self,
        shape,
        elem_type: TensorProto.DataType = TensorProto.FLOAT,
        communication_elem_type: TensorProto.DataType = TensorProto.FLOAT,
    ):
        X = helper.make_tensor_value_info("X", elem_type, shape)  # noqa: N806
        Y = helper.make_tensor_value_info("Y", elem_type, shape)  # noqa: N806
        _, size = self._get_rank_size()
        # Explanation is in comments for model creation in _create_allgather_ut_model.
        # Basically, ORT Python API doesn't support bool tensor yet, so we need to feed int64
        # tensor and cast it to bool before running communication op.
        if elem_type != communication_elem_type:
            node_defs = [
                helper.make_node("Cast", ["X"], ["X_casted"], to=communication_elem_type),
                helper.make_node(
                    "AllToAll",
                    ["X_casted"],
                    ["Y_casted"],
                    domain="com.microsoft",
                    group_size=size,
                ),
                helper.make_node("Cast", ["Y_casted"], ["Y"], to=elem_type),
            ]
        else:
            node_defs = [
                helper.make_node("AllToAll", ["X"], ["Y"], domain="com.microsoft", group_size=size),
            ]
        graph_def = helper.make_graph(
            node_defs,
            "",
            [X],
            [Y],
        )
        return ORTBertPretrainTest._create_model_with_opsets(graph_def)

    def _create_alltoall_ut_model_for_boolean_tensor(
        self,
        shape,
        # Tuple or list of bool values; e.g., [True, False] if
        # shape is (2,) or [[True, False], [False, True]] if
        # shape is (2, 2).
        # It's input of AllToAll.
        value,
    ):
        Y = helper.make_tensor_value_info("Y", TensorProto.BOOL, shape)  # noqa: N806
        _, size = self._get_rank_size()
        # Explanation is in comments for model creation in _create_allgather_ut_model.
        # Basically, ORT Python API doesn't support bool tensor yet, so we need to feed int64
        # tensor and cast it to bool before running communication op.
        x_const_value = helper.make_tensor("condition", TensorProto.BOOL, shape, value)
        node_defs = [
            helper.make_node(
                "Constant",
                [],
                ["X"],
                value=x_const_value,
            ),
            helper.make_node(
                "AllToAll",
                ["X"],
                ["Y"],
                domain="com.microsoft",
                group_size=size,
            ),
        ]
        graph_def = helper.make_graph(
            node_defs,
            "",
            [],
            [Y],
        )
        return ORTBertPretrainTest._create_model_with_opsets(graph_def)

    @unittest.skipIf(not ort.has_collective_ops(), reason="onnx not compiled with mpi support")
    @parameterized.expand(
        [
            (np.float32, TensorProto.FLOAT),
        ]
    )
    def test_all_reduce(self, np_elem_type, elem_type):
        model = self._create_allreduce_ut_model((128, 128), elem_type)
        rank, size = self._get_rank_size()
        ort_sess = ort.InferenceSession(
            model.SerializeToString(),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            provider_options=[{"device_id": str(rank)}, {}],
        )

        # Input for size == 4
        # For rank == 0:
        #   Input:  [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        #   Output: [[4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4]]
        #
        # For rank == 1:
        #   Input:  [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        #   Output: [[4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4]]
        #
        # For rank == 2:
        #   Input:  [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        #   Output: [[4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4]]
        #
        # For rank == 3:
        #   Input:  [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        #   Output: [[4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4]]

        input = np.ones((128, 128), dtype=np_elem_type)
        outputs = ort_sess.run(None, {"X": input})

        np.testing.assert_allclose(
            outputs[0], size * input, err_msg=f"{rank}: AllGather ({np_elem_type}, {elem_type}): results mismatch"
        )

    @unittest.skipIf(not ort.has_collective_ops(), reason="onnx not compiled with mpi support")
    @parameterized.expand(
        [
            (np.float32, TensorProto.FLOAT, TensorProto.FLOAT),
        ]
    )
    def test_all_gather(self, np_elem_type, elem_type, communication_elem_type):
        model = self._create_allgather_ut_model((128, 128), 0, elem_type, communication_elem_type)
        rank, size = self._get_rank_size()
        ort_sess = ort.InferenceSession(
            model.SerializeToString(),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            provider_options=[{"device_id": str(rank)}, {}],
        )

        # Input for size == 2
        # For rank == 0:
        #   Input:  [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        #   Output: [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
        #            [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        #
        # For rank == 1:
        #   Input:  [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        #   Output: [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
        #            [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]

        input = np.ones((128, 128), dtype=np.float32) * rank
        outputs = ort_sess.run(None, {"X": input})

        expected_output = np.zeros((128, 128), dtype=np_elem_type)
        for _ in range(size - 1):
            expected_output = np.concatenate((expected_output, np.ones((128, 128), dtype=np_elem_type) * (_ + 1)))

        np.testing.assert_allclose(
            outputs[0],
            expected_output,
            err_msg=f"{rank}: AllGather (axis0) ({np_elem_type}, {elem_type}, {communication_elem_type}): results mismatch",
        )

    @unittest.skipIf(not ort.has_collective_ops(), reason="onnx not compiled with mpi support")
    def test_all_gather_bool(self):
        model = self._create_allgather_ut_model((4,), 0, TensorProto.INT64, TensorProto.INT64)
        rank, _ = self._get_rank_size()

        ort_sess = ort.InferenceSession(
            model.SerializeToString(),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            provider_options=[{"device_id": str(rank)}, {}],
        )

        x = np.array([True, True, False, False]).astype(np.int64)
        y = ort_sess.run(None, {"X": x})[0]

        y_expected = np.array(
            [True, True, False, False] * 4,
        ).astype(np.int64)

        np.testing.assert_allclose(y, y_expected, err_msg=f"{rank}: AllGather (bool): results mismatch")

    @unittest.skipIf(not ort.has_collective_ops(), reason="onnx not compiled with mpi support")
    def test_all_gather_axis1(self):
        model = self._create_allgather_ut_model((128, 128), 1)
        rank, size = self._get_rank_size()
        ort_sess = ort.InferenceSession(
            model.SerializeToString(),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            provider_options=[{"device_id": str(rank)}, {}],
        )

        input = np.ones((128, 128), dtype=np.float32) * rank
        outputs = ort_sess.run(None, {"X": input})

        expected_output = np.zeros((128, 128), dtype=np.float32)
        for _ in range(size - 1):
            expected_output = np.concatenate((expected_output, np.ones((128, 128), dtype=np.float32) * (_ + 1)), axis=1)

        np.testing.assert_allclose(outputs[0], expected_output, err_msg=f"{rank}: AllGather (axis1): results mismatch")

    @unittest.skipIf(not ort.has_collective_ops(), reason="onnx not compiled with mpi support")
    @parameterized.expand(
        [
            (np.float32, TensorProto.FLOAT, TensorProto.FLOAT),
            (np.int64, TensorProto.INT64, TensorProto.INT64),
            (np.int64, TensorProto.INT64, TensorProto.BOOL),
        ]
    )
    def test_all_to_all(self, np_elem_type, elem_type, communication_elem_type):
        model = self._create_alltoall_ut_model(
            (8, 8), elem_type=elem_type, communication_elem_type=communication_elem_type
        )
        rank, size = self._get_rank_size()
        ort_sess = ort.InferenceSession(
            model.SerializeToString(),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            provider_options=[{"device_id": str(rank)}, {}],
        )

        # Casting to Boolean needs to be handled as a special case because only absoluate zero equates
        # to False and everything else to True. Post casting, however, the original input can never be
        # receovered when going in the opposite direction. The end results are always going to be in the
        # set [0, 1].

        input = np.ones((8, 8), dtype=np_elem_type) * rank

        if communication_elem_type == TensorProto.BOOL:
            # Input for size == 4
            # For rank == 0:
            #   Input:  [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
            #   Output: [[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
            #
            # For rank == 1:
            #   Input:  [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
            #   Output: [[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
            #
            # For rank == 2:
            #   Input:  [[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]]
            #   Output: [[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
            #
            # For rank == 3:
            #   Input:  [[3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3]]
            #   Output: [[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]

            expected_output = np.concatenate(
                (
                    np.zeros((8 // size, 8), dtype=np_elem_type),
                    np.ones((8 - (8 // size), 8), dtype=np_elem_type),
                )
            )
        else:
            # Input for size == 4
            # For rank == 0:
            #   Input:  [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
            #   Output: [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]
            #
            # For rank == 1:
            #   Input:  [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
            #   Output: [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]
            #
            # For rank == 2:
            #   Input:  [[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]]
            #   Output: [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]
            #
            # For rank == 3:
            #   Input:  [[3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3]]
            #   Output: [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]

            expected_output = np.zeros((8 // size, 8), dtype=np_elem_type)
            for _ in range(size - 1):
                expected_output = np.concatenate(
                    (expected_output, np.ones((8 // size, 8), dtype=np_elem_type) * (_ + 1))
                )

        outputs = ort_sess.run(None, {"X": input})

        np.testing.assert_allclose(
            outputs[0],
            expected_output,
            err_msg=f"{rank}: AllToAll ({np_elem_type}, {elem_type}, {communication_elem_type}): results mismatch",
        )

    @unittest.skipIf(not ort.has_collective_ops(), reason="onnx not compiled with mpi support")
    def test_all_to_all_bool(self):
        rank, _ = self._get_rank_size()

        # Input for size == 2
        # For rank == 0:
        #   Input:  [True, True, True, True]
        #   Output: [True, True, False, False]
        #
        # For rank == 1:
        #   Input:  [False, False, False, False]
        #   Output: [True, True, False, False]

        if rank == 0:
            x = [True, True, True, True]
        else:
            x = [False, False, False, False]

        model = self._create_alltoall_ut_model_for_boolean_tensor((4,), x)

        ort_sess = ort.InferenceSession(
            model.SerializeToString(),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            provider_options=[{"device_id": str(rank)}, {}],
        )

        y = ort_sess.run(None, {})

        y_expected = np.array(
            [True, False, False, False],
        ).astype(np.int64)

        np.testing.assert_allclose(y[0], y_expected, err_msg=f"{rank}: AllToAll (bool): results mismatch")


if __name__ == "__main__":
    unittest.main(module=__name__, buffer=True)
