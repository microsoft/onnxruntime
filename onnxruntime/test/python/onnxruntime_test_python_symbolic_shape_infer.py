# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# -*- coding: UTF-8 -*-
import onnx
from onnx import helper, AttributeProto, TensorProto, GraphProto
import os

if os.path.exists(os.path.join(os.path.dirname(__file__), '..', '..', 'python', 'tools', 'symbolic_shape_infer.py')):
    # Allow running this test script without installing onnxruntime package.
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'python', 'tools'))
    from symbolic_shape_infer import SymbolicShapeInference
else:
    from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

from pathlib import Path
import unittest


def unique_element(lst):
    assert len(lst) == 1
    return lst[0]


class TestSymbolicShapeInference(unittest.TestCase):
    def test_symbolic_shape_infer(self):
        cwd = os.getcwd()
        test_model_dir = os.path.join(cwd, '..', 'models')
        for filename in Path(test_model_dir).rglob('*.onnx'):
            if filename.name.startswith('.'):
                continue  # skip some bad model files
            print("Running symbolic shape inference on : " + str(filename))
            SymbolicShapeInference.infer_shapes(in_mp=onnx.load(str(filename)),
                                                auto_merge=True,
                                                int_max=100000,
                                                guess_output_rank=True)


class TestSymbolicShapeInferenceForOperators(unittest.TestCase):
    def _check_shapes(self, graph, inferred_graph, vis):  # type: (GraphProto, GraphProto, List[ValueInfoProto]) -> None
        names_in_vis = set(x.name for x in vis)
        vis = list(x for x in graph.value_info if x.name not in names_in_vis) + vis
        inferred_vis = list(inferred_graph.value_info)
        vis = list(sorted(vis, key=lambda x: x.name))
        inferred_vis = list(sorted(inferred_vis, key=lambda x: x.name))
        if vis == inferred_vis:
            return
        # otherwise some custom logic to give a nicer diff
        vis_names = set(x.name for x in vis)
        inferred_vis_names = set(x.name for x in inferred_vis)
        assert vis_names == inferred_vis_names, (vis_names, inferred_vis_names)
        for vi, inferred_vi in zip(vis, inferred_vis):
            assert vi == inferred_vi, '\n%s\n%s\n' % (vi, inferred_vi)
        assert False

    def test_unsqueeze_opset_11(self):
        graph = helper.make_graph([
            helper.make_node("Unsqueeze", ["input"], ["temp"], axes=[0]),
            helper.make_node("Identity", ["temp"], ["output"]),
        ], "Unsqueeze_Test", [
            helper.make_tensor_value_info('input', TensorProto.FLOAT, ['b', 's']),
        ], [
            helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 'b', 's']),
        ])
        model = helper.make_model(graph, producer_name='Unsqueeze_Test_Model')
        model.opset_import[0].version = 11

        inferred = SymbolicShapeInference.infer_shapes(model, auto_merge=True)
        expected_shapes = [
            helper.make_tensor_value_info('temp', TensorProto.FLOAT, [1, 'b', 's']),
            helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 'b', 's'])
        ]
        self._check_shapes(graph, inferred.graph, expected_shapes)

    def test_unsqueeze_opset_13(self):
        graph = helper.make_graph([
            helper.make_node("Unsqueeze", ["input", "axes"], ["temp"]),
            helper.make_node("Identity", ["temp"], ["output"]),
        ], "Unsqueeze_Test", [
            helper.make_tensor_value_info('input', TensorProto.FLOAT, ['b', 's']),
        ], [
            helper.make_tensor_value_info('output', TensorProto.FLOAT, ['b', 's', 1]),
        ], [
            helper.make_tensor('axes', TensorProto.INT64, [1], [-1]),
        ])
        model = helper.make_model(graph, producer_name='Unsqueeze_Test_Model')
        model.opset_import[0].version = 13

        inferred = SymbolicShapeInference.infer_shapes(model, auto_merge=True)
        expected_shapes = [
            helper.make_tensor_value_info('temp', TensorProto.FLOAT, ['b', 's', 1]),
            helper.make_tensor_value_info('output', TensorProto.FLOAT, ['b', 's', 1])
        ]
        self._check_shapes(graph, inferred.graph, expected_shapes)

    def test_embed_layer_norm(self):
        hidden_size = 32
        initializers = [
            helper.make_tensor('word_embedding', TensorProto.FLOAT, [100, hidden_size], [1.0] * (100 * hidden_size)),
            helper.make_tensor('position_embedding', TensorProto.FLOAT, [20, hidden_size], [1.0] * (20 * hidden_size)),
            helper.make_tensor('segment_embedding', TensorProto.FLOAT, [2, hidden_size], [1.0] * (2 * hidden_size)),
            helper.make_tensor('gamma', TensorProto.FLOAT, [hidden_size], [1.0] * hidden_size),
            helper.make_tensor('beta', TensorProto.FLOAT, [hidden_size], [1.0] * hidden_size)
        ]

        nodes = [
            helper.make_node("EmbedLayerNormalization",
                             inputs=[
                                 "input_ids", "segment_ids", "word_embedding", "position_embedding",
                                 "segment_embedding", "gamma", "beta"
                             ],
                             outputs=["output", "mask_index"],
                             domain="com.microsoft"),
        ]

        inputs = [
            helper.make_tensor_value_info('input_ids', TensorProto.FLOAT, ['b', 's']),
            helper.make_tensor_value_info('segment_ids', TensorProto.FLOAT, ['b', 's']),
        ]

        outputs = [
            helper.make_tensor_value_info('output', TensorProto.FLOAT, None),
            helper.make_tensor_value_info('mask_index', TensorProto.INT32, None),
        ]

        graph = helper.make_graph(nodes, "Unsqueeze_Test", inputs, outputs, initializers)
        model = helper.make_model(graph)

        inferred = SymbolicShapeInference.infer_shapes(model, auto_merge=True)
        expected_shapes = [
            helper.make_tensor_value_info('output', TensorProto.FLOAT, ['b', 's', hidden_size]),
            helper.make_tensor_value_info('mask_index', TensorProto.INT32, ['b'])
        ]
        self._check_shapes(graph, inferred.graph, expected_shapes)

    def test_softmax_cross_entropy_loss(self):
        hidden_size = 1024

        nodes = [
            helper.make_node("SoftmaxCrossEntropyLoss",
                             inputs=["logits", "labels"],
                             outputs=["loss"]),
        ]

        inputs = [
            helper.make_tensor_value_info('logits', TensorProto.FLOAT, ['b', 's', hidden_size]),
            helper.make_tensor_value_info('labels', TensorProto.INT32, ['b', 's']),
        ]

        outputs = [
            helper.make_tensor_value_info('loss', TensorProto.FLOAT, None),
        ]

        graph = helper.make_graph(nodes, "SoftmaxCrossEntropyLoss_Test", inputs, outputs, [])
        model = helper.make_model(graph)

        inferred = SymbolicShapeInference.infer_shapes(model, auto_merge=True)
        expected_shapes = [
            helper.make_tensor_value_info('loss', TensorProto.FLOAT, [])
        ]
        self._check_shapes(graph, inferred.graph, expected_shapes)

class TestSymbolicShapeInferenceForSlice(unittest.TestCase):
    def check_slice_of_concat(self, input_dims, start, end, step, expected_output_dim):
        _dimstrmap = {dim: f"dim{i}" for i, dim in enumerate(input_dims)}

        def dimstrmap(dim):
            return _dimstrmap.get(dim, dim)

        def get_initializer(name):
            valuemap = {"zero": 0, "one": 1, "two": 2, "ten": 10, "intmax": 2**32}
            value = -valuemap[name[4:]] if name.startswith("neg_") else valuemap[name]
            return onnx.helper.make_tensor(name, TensorProto.INT64, [1], [value])

        initializers = [
            get_initializer(name)
            for name in ["zero", "one", "two", "ten", "intmax", "neg_intmax", "neg_one", "neg_ten"]
        ]
        inputs = []
        nodes = []
        for i, dim in enumerate(input_dims):
            inputs.append(onnx.helper.make_tensor_value_info(f"t{i}", TensorProto.FLOAT, ["B", dim]))
            nodes.extend([
                onnx.helper.make_node("Shape", [f"t{i}"], [f"shape{i}"]),
                onnx.helper.make_node("Slice", [f"shape{i}", "one", "two", "zero", "one"], [f"dim{i}"]),
                onnx.helper.make_node("Neg", [f"dim{i}"], [f"neg_dim{i}"])
            ])

        def make_concat_dims(concat_name, dims):
            dims = [f"neg_{dimstrmap(dim[1:])}" if dim.startswith("-") else dimstrmap(dim) for dim in dims]
            return onnx.helper.make_node("Concat", dims, [concat_name], axis=0)

        nodes.extend([
            onnx.helper.make_node("Concat", [inp.name for inp in inputs], ["concat"], axis=1),
            make_concat_dims("starts", ["zero", start]),
            make_concat_dims("ends", ["intmax", end]),
            make_concat_dims("axes", ["zero", "one"]),
            make_concat_dims("steps", ["one", step]),
            onnx.helper.make_node("Slice", ["concat", "starts", "ends", "axes", "steps"], ["output"])
        ])
        output = onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ["d1", "d2"])
        graph_def = onnx.helper.make_graph(nodes, "graph", inputs, [output], initializer=initializers)
        model = SymbolicShapeInference.infer_shapes(onnx.helper.make_model(graph_def))
        output = unique_element(model.graph.output)
        shape = [d.dim_param if d.dim_param else d.dim_value for d in output.type.tensor_type.shape.dim]
        self.assertEqual(shape, ["B", expected_output_dim])

    def test_numeric_negative_indices_forward(self):
        self.check_slice_of_concat(["M"], "-ten", "-one", "one", 9)

    def test_numeric_negative_indices_backward(self):
        self.check_slice_of_concat(["M"], "-one", "-ten", "-one", 9)

    def test_symbolic_end_index(self):
        self.check_slice_of_concat(["M", "N"], "zero", "M", "one", "M")

    def test_symbolic_negative_start_index(self):
        self.check_slice_of_concat(["M", "N"], "-N", "intmax", "one", "N")

    def test_non_unit_step(self):
        self.check_slice_of_concat(["N", "N"], "zero", "intmax", "two", "N")

    def test_symbolic_step(self):
        self.check_slice_of_concat(["N", "N"], "zero", "intmax", "N", "floor(-1/N) + 3")

    def test_symbolic_negative_step(self):
        self.check_slice_of_concat(["N", "N"], "-one", "-intmax", "-N", "floor(-1/N) + 3")

    def test_flip(self):
        self.check_slice_of_concat(["N"], "-one", "-intmax", "-one", "N")

    def test_flip_of_concat(self):
        self.check_slice_of_concat(["N", "N", "N"], "-one", "-intmax", "-one", "3*N")


if __name__ == '__main__':
    unittest.main()
