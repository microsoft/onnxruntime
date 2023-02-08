# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import unittest

import numpy as np
import onnx
from mpi4py import MPI
from onnx import AttributeProto, GraphProto, TensorProto, helper

import onnxruntime as ort


class ORTBertPretrainTest(unittest.TestCase):
    def _create_allreduce_ut_model(self, shape):
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, shape)
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, shape)
        node_def = helper.make_node("AllReduce", ["X"], ["Y"], domain="com.microsoft")
        graph_def = helper.make_graph(
            [node_def],
            "",
            [X],
            [Y],
        )
        return helper.make_model(graph_def, producer_name="ort-distributed-inference-unittest")

    def _get_rank_size(self):
        comm = MPI.COMM_WORLD
        return comm.Get_rank(), comm.Get_size()

    def _create_allgather_ut_model(self, shape):
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, shape)
        rank, size = self._get_rank_size()
        output_shape = [s * size if _ == 0 else s for _, s in enumerate(shape)]
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, output_shape)
        node_def = helper.make_node("AllGather", ["X"], ["Y"], domain="com.microsoft", group_size=size)
        graph_def = helper.make_graph(
            [node_def],
            "",
            [X],
            [Y],
        )
        return helper.make_model(graph_def, producer_name="ort-distributed-inference-unittest")

    def test_all_reduce(self):
        model = self._create_allreduce_ut_model((128, 128))
        rank, size = self._get_rank_size()
        ort_sess = ort.InferenceSession(
            model.SerializeToString(),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            provider_options=[{"device_id": str(rank)}, {}],
        )

        input = np.ones((128, 128), dtype=np.float32)
        outputs = ort_sess.run(None, {"X": input})
        assert np.allclose(outputs[0], size * input)

    def test_all_gather(self):
        model = self._create_allgather_ut_model((128, 128))
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
            expected_output = np.concatenate((expected_output, np.ones((128, 128), dtype=np.float32) * (_ + 1)))

        assert np.allclose(outputs[0], expected_output)


if __name__ == "__main__":
    unittest.main(module=__name__, buffer=True)
