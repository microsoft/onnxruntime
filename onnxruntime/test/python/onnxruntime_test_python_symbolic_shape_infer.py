# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# -*- coding: UTF-8 -*-
import onnx
import os
#from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
from symbolic_shape_infer import SymbolicShapeInference
from pathlib import Path
import unittest


class TestSymbolicShapeInference(unittest.TestCase):
    def test_symbolic_shape_infer(self):
        cwd = os.getcwd()
        test_model_dir = os.path.join(cwd, '..', 'models')
        for filename in Path(test_model_dir).rglob('*.onnx'):
            if filename.name.startswith('.'):
                continue  # skip some bad model files
            print("Running symbolic shape inference on : " + str(filename))
            SymbolicShapeInference.infer_shapes(
                in_mp=onnx.load(str(filename)),
                auto_merge=True,
                int_max=100000,
                guess_output_rank=True)

    def _check_shapes(self, graph, inferred_graph,
                         vis):  # type: (GraphProto, GraphProto, List[ValueInfoProto]) -> None
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
        from onnx import helper, TensorProto
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
        onnx.save_model(model, "unsqueeze.onnx")
        inferred = SymbolicShapeInference.infer_shapes(model, auto_merge=True)
        expected_shapes = [
            helper.make_tensor_value_info('temp', TensorProto.FLOAT, [1, 'b', 's']),
            helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 'b', 's'])
        ]
        self._check_shapes(graph, inferred.graph, expected_shapes)

    def test_unsqueeze_opset_13(self):
        from onnx import helper, TensorProto
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
        onnx.save_model(model, "unsqueeze.onnx")
        inferred = SymbolicShapeInference.infer_shapes(model, auto_merge=True)
        expected_shapes = [
            helper.make_tensor_value_info('temp', TensorProto.FLOAT, ['b', 's', 1]),
            helper.make_tensor_value_info('output', TensorProto.FLOAT, ['b', 's', 1])
        ]
        self._check_shapes(graph, inferred.graph, expected_shapes)


if __name__ == '__main__':
    unittest.main()
