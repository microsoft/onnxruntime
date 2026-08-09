# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import io
import unittest

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper
from ort_torch_ext import init_aten_op_executor
from torch.onnx import export

import onnxruntime as ort


class OrtOpTests(unittest.TestCase):
    def test_aten_embedding(self):
        class NeuralNetEmbedding(torch.nn.Module):
            def __init__(self, num_embeddings, embedding_dim, hidden_size):
                super().__init__()
                self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
                self.linear = torch.nn.Linear(embedding_dim, hidden_size)

            def forward(self, input):
                embedding_result = self.embedding(input)
                return embedding_result, self.linear(embedding_result)

        N, num_embeddings, embedding_dim, hidden_size = 64, 32, 128, 128  # noqa: N806
        model = NeuralNetEmbedding(num_embeddings, embedding_dim, hidden_size)

        with torch.no_grad():
            x = torch.randint(high=num_embeddings, size=(N,), dtype=torch.int64)
            dynamic_axes = {"x": {0: "x_dim0"}, "y": {0: "y_dim0", 1: "y_dim1"}}

            f = io.BytesIO()

            export(
                model,
                x,
                f=f,
                input_names=["x"],
                output_names=["y"],
                dynamic_axes=dynamic_axes,
                opset_version=14,
            )

            exported_model = onnx.load_model_from_string(f.getvalue())

            # PyTorch exporter emitting ATen op is still under development. Currently convert it manually for testing.
            for node in exported_model.graph.node:
                if node.op_type == "Gather":
                    node.domain = "org.pytorch.aten"
                    node.op_type = "ATen"
                    attr = node.attribute.add()
                    attr.name = "operator"
                    attr.type = 3
                    attr.s = b"embedding"
                    exported_model.graph.node.append(
                        helper.make_node(
                            "Constant",
                            [],
                            ["padding_idx"],
                            value=helper.make_tensor("padding_idx", TensorProto.INT64, (), [-1]),
                        )
                    )
                    exported_model.graph.node.append(
                        helper.make_node(
                            "Constant",
                            [],
                            ["scale_grad_by_freq"],
                            value=helper.make_tensor("scale_grad_by_freq", TensorProto.BOOL, (), [False]),
                        )
                    )
                    exported_model.graph.node.append(
                        helper.make_node(
                            "Constant",
                            [],
                            ["sparse"],
                            value=helper.make_tensor("sparse", TensorProto.BOOL, (), [False]),
                        )
                    )
                    node.input.append("padding_idx")
                    node.input.append("scale_grad_by_freq")
                    node.input.append("sparse")
                    exported_model.graph.value_info.append(
                        helper.make_value_info(
                            name=node.output[0],
                            type_proto=helper.make_tensor_type_proto(
                                elem_type=TensorProto.FLOAT, shape=[node.output[0] + "_dim0", node.output[0] + "_dim1"]
                            ),
                        )
                    )
                    break

        # The ONNX graph to run contains ATen Op.
        assert any(node.op_type == "ATen" for node in exported_model.graph.node)

        init_aten_op_executor()

        # Run w/o IO binding.
        for _ in range(8):
            x = torch.randint(high=num_embeddings, size=(N,), dtype=torch.int64)
            pt_y1, pt_y2 = model(x)
            session = ort.InferenceSession(exported_model.SerializeToString(), providers=["CPUExecutionProvider"])
            ort_y1, ort_y2 = session.run([], {"x": x.numpy()})
            np.testing.assert_almost_equal(ort_y1, pt_y1.detach().numpy(), decimal=6)
            np.testing.assert_almost_equal(ort_y2, pt_y2.detach().numpy(), decimal=6)

        # Run w/ IO binding.
        for _ in range(8):
            x = torch.randint(high=num_embeddings, size=(N,), dtype=torch.int64)
            ort_x = ort.OrtValue.ortvalue_from_numpy(x.detach().numpy(), "cpu")
            pt_y1, pt_y2 = model(x)
            np_y1 = np.zeros(tuple(pt_y1.size()), dtype=np.float32)
            np_y2 = np.zeros(tuple(pt_y2.size()), dtype=np.float32)
            ort_y1 = ort.OrtValue.ortvalue_from_numpy(np_y1, "cpu")
            ort_y2 = ort.OrtValue.ortvalue_from_numpy(np_y2, "cpu")
            session = ort.InferenceSession(exported_model.SerializeToString(), providers=["CPUExecutionProvider"])
            io_binding = session.io_binding()
            io_binding.bind_ortvalue_input(exported_model.graph.input[0].name, ort_x)
            io_binding.bind_ortvalue_output(exported_model.graph.output[0].name, ort_y1)
            io_binding.bind_ortvalue_output(exported_model.graph.output[1].name, ort_y2)
            session.run_with_iobinding(io_binding)
            np.testing.assert_almost_equal(np_y1, pt_y1.detach().numpy(), decimal=6)
            np.testing.assert_almost_equal(np_y2, pt_y2.detach().numpy(), decimal=6)


if __name__ == "__main__":
    unittest.main()
