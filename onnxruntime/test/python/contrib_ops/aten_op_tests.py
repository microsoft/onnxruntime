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
                super(NeuralNetEmbedding, self).__init__()
                self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=1)
                self.linear = torch.nn.Linear(embedding_dim, hidden_size)

            def forward(self, input):
                return self.linear(self.embedding(input))

        N, num_embeddings, embedding_dim, hidden_size = 64, 32, 128, 128
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
                    attr.s = "embedding".encode()
                    exported_model.graph.node.append(
                        helper.make_node(
                            "Constant",
                            [],
                            ["padding_idx"],
                            value=helper.make_tensor("padding_idx", TensorProto.INT64, (), [1]),
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
                                elem_type=1, shape=[node.output[0] + "_dim0", node.output[0] + "_dim1"]
                            ),
                        )
                    )
                    break

        # The ONNX graph to run contains ATen Op.
        assert all(node.op_type == "ATen" for node in exported_model.graph.node)

        init_aten_op_executor()

        for _ in range(8):
            x = torch.randint(high=num_embeddings, size=(N,), dtype=torch.int64)
            pt_y = model(x)
            session = ort.InferenceSession(exported_model.SerializeToString(), providers=["CPUExecutionProvider"])
            ort_y = session.run([], {"x": x.numpy()})[0]
            np.testing.assert_almost_equal(ort_y, pt_y.detach().numpy())


if __name__ == "__main__":
    unittest.main()
