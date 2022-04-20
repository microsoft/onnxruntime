# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx
import pathlib
import unittest

from ..qdq_model_utils import fix_dq_nodes_with_multiple_consumers

script_dir = pathlib.Path(__file__).parent
ort_root = script_dir.parents[4]

# example usage from <ort root>/tools/python
# python -m unittest util/qdq_helpers/test/test_qdq_model_utils.py
# NOTE: at least on Windows you must use that as the working directory for all the imports to be happy


class TestQDQUtils(unittest.TestCase):
    def test_fix_DQ_with_multiple_consumers(self):
        '''
        '''
        model_path = ort_root / 'onnxruntime' / 'test' / 'testdata' / 'qdq_with_multi_consumer_dq_nodes.onnx'
        model = onnx.load(str(model_path))

        orig_dq_nodes = [n for n in model.graph.node if n.op_type == 'DequantizeLinear']
        fix_dq_nodes_with_multiple_consumers(model)
        new_dq_nodes = [n for n in model.graph.node if n.op_type == 'DequantizeLinear']

        # there are 3 DQ nodes with 2 consumers (an earlier Conv and later Add)
        # additionally the last one also provides a graph output
        # based on that there should be 3 new DQ nodes for the internal consumers and 1 new one for the graph output
        self.assertEqual(len(orig_dq_nodes) + 4, len(new_dq_nodes))
