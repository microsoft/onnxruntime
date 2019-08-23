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

'''
from quantize import quantize, QuantizationMode
class ProcessLogFileTest(unittest.TestCase):
    # def test_baddir(self):
    #     self.assertRaises(ValueError, calibrate.process_logfiles('/non/existent/path'))

    def test_gooddata(self):
        expected = "gpu_0_conv1_1"
        sfacs, zpts = calibrate.process_logfiles('test_data')
        self.assertTrue(expected in sfacs)
        self.assertTrue(expected in zpts)

        self.assertEqual(8.999529411764707, sfacs[expected])
        self.assertEqual(23086, zpts[expected])
        
## Load the onnx model
# model = onnx.load('path/to/the/model.onnx')
'''

class TestCalibrate(unittest.TestCase):

    def test_augment_graph(self):
        # Creating graph
        A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 1, 5, 5])
        B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1, 1, 3, 3])
        C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [1, 1, 5, 5])
        D = helper.make_tensor_value_info('D', TensorProto.FLOAT, [1, 1, 5, 5])
        E = helper.make_tensor_value_info('E', TensorProto.FLOAT, [1, 1, 5, 1])
        F = helper.make_tensor_value_info('F', TensorProto.FLOAT, [1, 1, 5, 1])
        conv_node = onnx.helper.make_node('Conv', ['A', 'B'], ['C'], name='Conv', kernel_shape=[3, 3], pads=[1, 1, 1, 1])
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
        self.assertEqual(len(augmented_model_node_names), 7) # original 3 nodes with 4 added ones
        self.assertEqual(len(augmented_model_outputs), 5) # original single graph output with 4 added ones
        for name in added_node_names:
            self.assertTrue(name in augmented_model_node_names)
        for output in added_outputs:
            self.assertTrue(output in augmented_model_outputs)

    def test_load_and_resize_image(self):
        session = onnxruntime.InferenceSession('augmented_test_model.onnx')
        (samples, channels, height, width) = session.get_inputs()[0].shape
        nchw_data = calibrate.load_and_resize_image('test_images/' + os.listdir('test_images')[0], height, width)
        self.assertEqual(nchw_data.shape[1], 3) # checking for 3 channels for colored image
        self.assertEqual(nchw_data.shape[2], height) # checking if resized height is correct
        self.assertEqual(nchw_data.shape[3], width) # checking if resized width is correct

    def test_load_batch(self):
        images_folder = 'test_images'
        session = onnxruntime.InferenceSession('augmented_test_model.onnx')
        (samples, channels, height, width) = session.get_inputs()[0].shape
        batch_data = calibrate.load_batch(images_folder, height, width)
        self.assertEqual(len(batch_data.shape), 5) # for 2D images like the ones in test_images
        self.assertEqual(batch_data.shape[0], len(os.listdir(images_folder)))
        self.assertEqual(batch_data.shape[2], 3)
        self.assertEqual(batch_data.shape[3], height)
        self.assertEqual(batch_data.shape[4], width)

    def test_get_intermediate_outputs(self):
        # Creating graph
        A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 1, 5, 5])
        B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1, 1, 5, 5])
        C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [1, 1, 5, 5])
        clip_node = onnx.helper.make_node('Clip', ['A'], ['B'], name='Clip')
        matmul_node = onnx.helper.make_node('MatMul', ['B', 'B'], ['C'], name='MatMul')
        graph = helper.make_graph([clip_node, matmul_node], 'test_graph_small', [A], [C])
        model = helper.make_model(graph)
        model_path = 'test_model_small.onnx'
        onnx.save(model, model_path)

        # Augmenting graph
        augmented_model = calibrate.augment_graph(model)
        augmented_model_path = 'augmented_test_model_small.onnx'
        onnx.save(augmented_model, augmented_model_path)

        # Running inference
        images_folder = 'test_images'
        session = onnxruntime.InferenceSession(augmented_model_path)
        (samples, channels, height, width) = session.get_inputs()[0].shape
        inputs = calibrate.load_batch(images_folder, height, width)
        dict = calibrate.get_intermediate_outputs(model_path, session, inputs)

        min_results, max_results = [], []
        for file in os.listdir(images_folder):
            image_filepath = 'test_images/' + file
            min_results.append(session.run(["C_ReduceMin"], {'A': calibrate.load_and_resize_image(image_filepath, height, width)}))
            max_results.append(session.run(["C_ReduceMax"], {'A': calibrate.load_and_resize_image(image_filepath, height, width)}))
        self.assertTrue(len(dict.keys()) == 1 and 'C' in dict.keys())
        self.assertEqual(dict['C'][0], min(min_results)[0]) # check for correct minimum of ReduceMin values
        self.assertEqual(dict['C'][1], max(max_results)[0]) # check for correct maximum of ReduceMax values

if __name__ == '__main__':
    unittest.main()
