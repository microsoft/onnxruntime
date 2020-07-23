#!/usr/bin/env python
# coding: utf-8
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
import os
import onnx
import onnxruntime
from onnx import helper, TensorProto
from onnx.helper import make_node, make_tensor_value_info
import calibrate
import numpy as np
from onnx import numpy_helper


class TestCalibrate(unittest.TestCase):
    def test_augment_graph(self):
        # Creating graph
        A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 1, 5, 5])
        B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1, 1, 3, 3])
        C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [1, 1, 5, 5])
        D = helper.make_tensor_value_info('D', TensorProto.FLOAT, [1, 1, 5, 5])
        E = helper.make_tensor_value_info('E', TensorProto.FLOAT, [1, 1, 5, 1])
        F = helper.make_tensor_value_info('F', TensorProto.FLOAT, [1, 1, 5, 1])
        conv_node = onnx.helper.make_node('Conv', ['A', 'B'], ['C'],
                                          name='Conv',
                                          kernel_shape=[3, 3],
                                          pads=[1, 1, 1, 1])
        clip_node = onnx.helper.make_node('Clip', ['C'], ['D'], name='Clip')
        matmul_node = onnx.helper.make_node('MatMul', ['D', 'E'], ['F'], name='MatMul')
        graph = helper.make_graph([conv_node, clip_node, matmul_node], 'test_graph', [A, B, E], [F])
        model = helper.make_model(graph)
        onnx.save(model, 'test_model.onnx')

        # Augmenting graph
        augmented_model = calibrate.augment_graph(model)
        onnx.save(augmented_model, 'augmented_test_model.onnx')

        # Checking if each added ReduceMin and ReduceMax node and its output exists
        augmented_model_node_names = [node.name for node in augmented_model.graph.node]
        augmented_model_outputs = [output.name for output in augmented_model.graph.output]
        added_node_names = ['Conv_ReduceMin', 'Conv_ReduceMax', 'MatMul_ReduceMin', 'MatMul_ReduceMax']
        added_outputs = ['C_ReduceMin', 'C_ReduceMax', 'F_ReduceMin', 'F_ReduceMax']
        # original 3 nodes with 4 added ones
        self.assertEqual(len(augmented_model_node_names), 7)
        # original single graph output with 4 added ones
        self.assertEqual(len(augmented_model_outputs), 5)
        for name in added_node_names:
            self.assertTrue(name in augmented_model_node_names)
        for output in added_outputs:
            self.assertTrue(output in augmented_model_outputs)

        # Checking order of augmented graph outputs
        G = helper.make_tensor_value_info('G', TensorProto.FLOAT, [1, 1, 5, 5])
        H = helper.make_tensor_value_info('H', TensorProto.FLOAT, [1, 1, 5, 1])
        I = helper.make_tensor_value_info('I', TensorProto.FLOAT, [1, 1, 5, 1])
        J = helper.make_tensor_value_info('J', TensorProto.FLOAT, [1, 1, 5, 1])
        K = helper.make_tensor_value_info('K', TensorProto.FLOAT, [1, 1, 5, 1])
        matmul_node_1 = onnx.helper.make_node('MatMul', ['G', 'H'], ['I'], name='MatMul_1')
        matmul_node_2 = onnx.helper.make_node('MatMul', ['I', 'J'], ['K'], name='MatMul_2')
        graph = helper.make_graph([matmul_node_1, matmul_node_2], 'test_graph_order', [G, H, J], [K])
        model = helper.make_model(graph)
        onnx.save(model, 'test_model_matmul.onnx')

        # Augmenting graph
        augmented_model = calibrate.augment_graph(model)
        onnx.save(augmented_model, 'augmented_test_model_matmul.onnx')

        augmented_model_outputs = [output.name for output in augmented_model.graph.output]
        self.assertEqual(augmented_model_outputs[0], 'K')
        self.assertEqual(augmented_model_outputs[1], 'I_ReduceMin')
        self.assertEqual(augmented_model_outputs[2], 'I_ReduceMax')
        self.assertEqual(augmented_model_outputs[3], 'K_ReduceMin')
        self.assertEqual(augmented_model_outputs[4], 'K_ReduceMax')

    def test_load_batch(self):
        images_folder = 'test_images'
        session = onnxruntime.InferenceSession('augmented_test_model.onnx')
        (samples, channels, height, width) = session.get_inputs()[0].shape
        batch_data = calibrate.load_batch(images_folder, height, width, preprocess_func_name="preprocess_method1")
        # for 2D images like the ones in test_images
        self.assertEqual(len(batch_data.shape), 5)
        self.assertEqual(batch_data.shape[0], len(os.listdir(images_folder)))
        # checking for 3 channels for colored image
        self.assertEqual(batch_data.shape[2], 3)
        # checking if resized height is correct
        self.assertEqual(batch_data.shape[3], height)
        # checking if resized width is correct
        self.assertEqual(batch_data.shape[4], width)

    def test_load_pb(self):
        numpy_array = np.random.randn(3, 1, 3, 5, 5).astype(np.float32)
        tensor = numpy_helper.from_array(numpy_array)
        test_file_name = 'test_tensor.pb'
        with open(test_file_name, 'wb') as f:
            f.write(tensor.SerializeToString())

        # test size_limit < than number of samples in data set
        # expecting to load size_limit number of samples
        batch_data = calibrate.load_pb_file('test_tensor.pb', 2, 1, 3, 5, 5)
        self.assertEqual(len(batch_data.shape), 5)
        self.assertEqual(batch_data.shape[0], 2)
        self.assertEqual(batch_data.shape[2], 3)
        self.assertEqual(batch_data.shape[3], 5)
        self.assertEqual(batch_data.shape[4], 5)

        # test size_limit == 0
        # expecting to load all samples
        batch_data = calibrate.load_pb_file('test_tensor.pb', 0, 1, 3, 5, 5)
        self.assertEqual(len(batch_data.shape), 5)
        self.assertEqual(batch_data.shape[0], 3)
        self.assertEqual(batch_data.shape[2], 3)
        self.assertEqual(batch_data.shape[3], 5)
        self.assertEqual(batch_data.shape[4], 5)

        # test size_limit > than number of samples in data set
        # expecting to load all samples
        batch_data = calibrate.load_pb_file('test_tensor.pb', 6, 1, 3, 5, 5)
        self.assertEqual(len(batch_data.shape), 5)
        self.assertEqual(batch_data.shape[0], 3)
        self.assertEqual(batch_data.shape[2], 3)
        self.assertEqual(batch_data.shape[3], 5)
        self.assertEqual(batch_data.shape[4], 5)

        try:
            os.remove('test_tensor.pb')
        except:
            print("Warning: Trying to remove test file {} failed.".format(test_file_name))


if __name__ == '__main__':
    unittest.main()
