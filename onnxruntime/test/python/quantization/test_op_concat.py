# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
import numpy as np
from onnx import helper, TensorProto, numpy_helper, save
from onnxruntime.quantization import quantize_static, QuantFormat
from op_test_utils import InputFeedsNegOneZeroOne, check_model_correctness, check_op_type_count


class TestONNXModel(unittest.TestCase):
    def construct_model(self, model_path):
        #          (input)
        #         /    |  \
        #        /     |   \
        #       /      |    \
        #      /       |     \
        #  Conv(1)  Conv(2)  conv(3)
        #       \      |     /
        #         \    |    /
        #           \  |   /
        #            Concat
        #              |
        #           Identity
        #              |
        #           (output)
        initializers = []
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 15, 15])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 13, 13, 13])

        # Conv1 output [1, 2, 13, 13]
        conv1_weight_initializer = numpy_helper.from_array(
            np.random.randint(-1, 2, [2, 3, 3, 3]).astype(np.float32), name='conv1_weight')
        conv1_node = helper.make_node('Conv', ['input', 'conv1_weight'], ['conv1_output'], name='conv1_node')

        # Conv2 output [1, 5, 13, 13]
        conv2_weight_initializer = numpy_helper.from_array(
            np.random.randint(-1, 2, [5, 3, 3, 3]).astype(np.float32), name='conv2_weight')
        conv2_node = helper.make_node('Conv', ['input', 'conv2_weight'], ['conv2_output'], name='conv2_node')

        # Conv3 output [1, 6, 13, 13]
        conv3_weight_initializer = numpy_helper.from_array(
            np.random.randint(-1, 2, [6, 3, 3, 3]).astype(np.float32), name='conv3_weight')
        conv3_node = helper.make_node('Conv', ['input', 'conv3_weight'], ['conv3_output'], name='conv3_node')

        concat_node = helper.make_node('Concat', ['conv1_output', 'conv2_output', 'conv3_output'], [
                                            'concat_output'], name='concat_node', axis=1)

        identity_node = helper.make_node('Identity', ['concat_output'], ['output'], name='identity_node')

        initializers = [conv1_weight_initializer, conv2_weight_initializer, conv3_weight_initializer]
        graph = helper.make_graph([conv1_node, conv2_node, conv3_node, concat_node, identity_node],
                                  'qlinear_concat_op_test', [input], [output], initializer=initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        save(model, model_path)

    def test_quantize_concat(self):
        np.random.seed(1)

        model_fp32_path = 'concat_fp32.onnx'
        model_uint8_path = 'concat_uint8.onnx'
        model_uint8_qdq_path = 'concat_uint8_qdq.onnx'

        self.construct_model(model_fp32_path)

        # Verify QOperator mode
        data_reader = InputFeedsNegOneZeroOne(1, {'input': [1, 3, 15, 15]})
        quantize_static(model_fp32_path, model_uint8_path, data_reader)

        qnode_counts = {'QLinearConv': 3, 'QuantizeLinear': 1, 'DequantizeLinear': 1, 'QLinearConcat': 1}
        check_op_type_count(self, model_uint8_path, **qnode_counts)
        data_reader.rewind()
        check_model_correctness(self, model_fp32_path, model_uint8_path, data_reader.get_next())

        # Verify QDQ mode
        data_reader.rewind()
        quantize_static(model_fp32_path, model_uint8_qdq_path, data_reader, quant_format=QuantFormat.QDQ)
        qdqnode_counts = {'Conv': 3, 'QuantizeLinear': 5, 'DequantizeLinear': 8, 'Concat': 1}
        check_op_type_count(self, model_uint8_qdq_path, **qdqnode_counts)
        data_reader.rewind()
        check_model_correctness(self, model_fp32_path, model_uint8_qdq_path, data_reader.get_next())


if __name__ == '__main__':
    unittest.main()
