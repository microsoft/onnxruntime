#!/usr/bin/env python
# coding: utf-8
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
import onnx
import onnxruntime
import numpy as np
from onnx import helper, TensorProto, numpy_helper
from onnxruntime.quantization.calibrate import calibrate, CalibrationDataReader, ONNXCalibrater


def generate_input_initializer(tensor_shape, tensor_dtype, input_name):
  '''
  Helper function to generate initializers for test inputs
  '''
  tensor = np.random.ranf(tensor_shape).astype(tensor_dtype)
  init = numpy_helper.from_array(tensor, input_name)
  return init  

class TestDataReader(CalibrationDataReader):
    '''for test purpose'''
    def __init__(self):
        pass
    def get_next(self):
        return None

class TestDataReaderSecond(CalibrationDataReader):
    '''for test purpose'''
    def __init__(self):
        self.preprocess_flag = True
        self.enum_data_dicts = []

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            nhwc_data_list = []
            nhwc_data_list.append(np.array([[[[0.45,0.60,0.75]],
                                            [[0.25,0.50,0.75]],
                                            [[0.90,0.70,0.50]]]]).astype(np.float32))
            nhwc_data_list.append(np.array([[[[0.62,0.94,0.38]],
                                            [[0.70,0.13,0.07]],
                                            [[0.89,0.75,0.84]]]]).astype(np.float32))
            nhwc_data_list.append(np.array([[[[0.64,0.24,0.97]],
                                            [[0.82,0.58,0.27]],
                                            [[0.019,0.34,0.02]]]]).astype(np.float32))
            input_name = 'input0'
            self.enum_data_dicts = iter([{input_name: nhwc_data} for nhwc_data in nhwc_data_list])
        return next(self.enum_data_dicts, None)


class TestCalibrate(unittest.TestCase):
    
    def test_augment_graph(self):

        ''' TEST_CONFIG_1'''

        #     Conv 
        #      |   
        #     Clip
        #      |      
        #     MatMul
        
        A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 1, 5, 5])
        B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1, 1, 3, 3])
        E = helper.make_tensor_value_info('E', TensorProto.FLOAT, [1, 1, 5, 1])
        F = helper.make_tensor_value_info('F', TensorProto.FLOAT, [1, 1, 5, 1])
        conv_node = onnx.helper.make_node('Conv', ['A', 'B'], ['C'], name='Conv', kernel_shape=[3, 3], pads=[1, 1, 1, 1])
        clip_node = onnx.helper.make_node('Clip', ['C'], ['D'], name='Clip')
        matmul_node = onnx.helper.make_node('MatMul', ['D', 'E'], ['F'], name='MatMul')
        graph = helper.make_graph([conv_node, clip_node, matmul_node], 'test_graph_1', [A, B, E], [F])

        model = helper.make_model(graph)
        test_model_path = './test_model_1.onnx'
        onnx.save(model, test_model_path)

        # Augmenting graph
        data_reader = TestDataReader()
        augmented_model_path = './augmented_test_model_1.onnx'
        calibrater = ONNXCalibrater(test_model_path, data_reader, ['Conv', 'MatMul'], [], [], augmented_model_path)
        augmented_model = calibrater.augment_graph()
        onnx.save(augmented_model, augmented_model_path)

        # Checking if each added ReduceMin and ReduceMax node and its output exists
        augmented_model_node_names = [node.name for node in augmented_model.graph.node]
        augmented_model_outputs = [output.name for output in augmented_model.graph.output]
        added_node_names = ['C_ReduceMin', 'C_ReduceMax', 'D_ReduceMin', 'D_ReduceMax', 'F_ReduceMin', 'F_ReduceMax']
        added_outputs = ['C_ReduceMin', 'C_ReduceMax', 'D_ReduceMin', 'D_ReduceMax', 'F_ReduceMin', 'F_ReduceMax']
        # Original 3 nodes + added ReduceMin/Max nodes * 6 (exlude graph input/output)
        self.assertEqual(len(augmented_model_node_names), 9)
        # Original 1 graph output + added outputs * 6
        self.assertEqual(len(augmented_model_outputs), 7)
        for name in added_node_names:
            self.assertTrue(name in augmented_model_node_names)
        for output in added_outputs:
            self.assertTrue(output in augmented_model_outputs)

        print('Finished TEST_CONFIG_1')


        '''TEST_CONFIG_2'''

        #   Conv
        #    |   
        #   Conv

        G = helper.make_tensor_value_info('G', TensorProto.FLOAT, [1, 1, 5, 5])
        H = helper.make_tensor_value_info('H', TensorProto.FLOAT, [1, 1, 3, 3])
        J = helper.make_tensor_value_info('J', TensorProto.FLOAT, [1, 1, 3, 3])
        K = helper.make_tensor_value_info('K', TensorProto.FLOAT, [1, 1, 5, 5])
        conv_node_1 = onnx.helper.make_node('Conv', ['G', 'H'], ['I'], name='Conv', kernel_shape=[3, 3], pads=[1, 1, 1, 1])
        conv_node_2 = onnx.helper.make_node('Conv', ['I', 'J'], ['K'], name='Conv', kernel_shape=[3, 3], pads=[1, 1, 1, 1])
        graph = helper.make_graph([conv_node_1, conv_node_2], 'test_graph_2', [G, H, J], [K])
        model = helper.make_model(graph)
        test_model_path = './test_model_2.onnx'
        onnx.save(model, test_model_path)

        # Augmenting graph
        data_reader = TestDataReader()
        augmented_model_path = './augmented_test_model_2.onnx'
        calibrater = ONNXCalibrater(test_model_path, data_reader, ['Conv', 'MatMul'], [], [], augmented_model_path)
        augmented_model = calibrater.augment_graph()
        onnx.save(augmented_model, augmented_model_path)

        augmented_model_node_names = [node.name for node in augmented_model.graph.node]
        augmented_model_outputs = [output.name for output in augmented_model.graph.output]
        added_node_names = ['I_ReduceMin', 'I_ReduceMax', 'K_ReduceMin', 'K_ReduceMax']
        added_outputs = ['I_ReduceMin', 'I_ReduceMax', 'K_ReduceMin', 'K_ReduceMax']
        # Original 2 nodes + added ReduceMin/Max nodes * 4
        self.assertEqual(len(augmented_model_node_names), 6)
        # Original 1 graph output + added outputs * 4
        self.assertEqual(len(augmented_model_outputs), 5)
        for name in added_node_names:
            self.assertTrue(name in augmented_model_node_names)
        for output in added_outputs:
            self.assertTrue(output in augmented_model_outputs)

        print('Finished TEST_CONFIG_2')
        
        
        '''TEST_CONFIG_3'''
        
        #   Relu
        #    |  
        #   Conv  \ 
        #    |     |
        #   Clip   |
        #    |    /
        #   MatMul

        L = helper.make_tensor_value_info('L', TensorProto.FLOAT,  [1, 1, 5, 5])
        N = helper.make_tensor_value_info('N', TensorProto.FLOAT, [1, 1, 3, 3])
        Q = helper.make_tensor_value_info('Q', TensorProto.FLOAT, [1, 1, 5, 5])
        relu_node = onnx.helper.make_node('Relu', ['L'], ['M'], name='Relu')
        conv_node = onnx.helper.make_node('Conv', ['M', 'N'], ['O'], name='Conv', kernel_shape=[3, 3], pads=[1, 1, 1, 1])
        clip_node = onnx.helper.make_node('Clip', ['O'], ['P'], name='Clip')
        matmul_node = onnx.helper.make_node('MatMul', ['P','M'], ['Q'], name='MatMul')
        graph = helper.make_graph([relu_node, conv_node, clip_node, matmul_node], 'test_graph_3', [L, N], [Q])
        model = helper.make_model(graph)
        test_model_path = './test_model_3.onnx'
        onnx.save(model, test_model_path)

        # Augmenting graph
        data_reader = TestDataReader()
        augmented_model_path = './augmented_test_model_3.onnx'
        calibrater = ONNXCalibrater(test_model_path, data_reader, ['Conv', 'MatMul'], [], [], augmented_model_path)
        augmented_model = calibrater.augment_graph()
        onnx.save(augmented_model, augmented_model_path)

        augmented_model_node_names = [node.name for node in augmented_model.graph.node]
        augmented_model_outputs = [output.name for output in augmented_model.graph.output]
        added_node_names = ['M_ReduceMin', 'M_ReduceMax', 'O_ReduceMin', 'O_ReduceMax', 'P_ReduceMin', 'P_ReduceMax', 'Q_ReduceMin', 'Q_ReduceMax']
        added_outputs =  ['M_ReduceMin', 'M_ReduceMax', 'O_ReduceMin', 'O_ReduceMax', 'P_ReduceMin', 'P_ReduceMax',  'Q_ReduceMin', 'Q_ReduceMax']
        # Original 4 nodes + added ReduceMin/Max nodes * 8
        self.assertEqual(len(augmented_model_node_names), 12)
        # Original 1 graph output + added outputs * 8
        self.assertEqual(len(augmented_model_outputs), 9)
        for name in added_node_names:
            self.assertTrue(name in augmented_model_node_names)
        for output in added_outputs:
            self.assertTrue(output in augmented_model_outputs)
      
        print('Finished TEST_CONFIG_3')
    

    def test_quant_param_calculation(self):
       
        '''TEST_CONFIG_4'''
     
        #   Relu      
        #    |      \ 
        #   Conv     \
        #    |        \ 
        #   Relu       |  
        #    |       Conv  
        #   Conv      / 
        #      \     /  
        #         |
        #        Add
    
        input0 = helper.make_tensor_value_info('input0', TensorProto.FLOAT, [1, 3, 1, 3])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 1, 3])
        
        X1_weight = generate_input_initializer([3, 3, 1, 1], np.float32, 'X1_weight')
        X1_bias = generate_input_initializer([3], np.float32, 'X1_bias')
        X3_weight = generate_input_initializer([3, 3, 1, 1], np.float32, 'X3_weight')
        X3_bias = generate_input_initializer([3],np.float32, 'X3_bias')
        X5_weight = generate_input_initializer([3, 3, 1, 1], np.float32, 'X5_weight')
        X5_bias = generate_input_initializer([3],np.float32,'X5_bias')
       
        relu_node_1 = onnx.helper.make_node('Relu', ['input0'], ['X1'], name='Relu1')
        conv_node_1 = onnx.helper.make_node('Conv', ['X1', 'X1_weight', 'X1_bias'], ['X2'], name='Conv1')
        relu_node_2 = onnx.helper.make_node('Relu', ['X2'], ['X3'], name= 'Relu2')
        conv_node_2 = onnx.helper.make_node('Conv', ['X3', 'X3_weight', 'X3_bias'], ['X4'], name='Conv2')
        conv_node_3 = onnx.helper.make_node('Conv', ['X1', 'X5_weight', 'X5_bias'], ['X5'], name='Conv3')
        add_node = onnx.helper.make_node('Add', ['X4', 'X5'], ['output'], name='Add')
      
        graph = helper.make_graph([relu_node_1, conv_node_1, relu_node_2, conv_node_2, conv_node_3, add_node], 'test_graph_4', [input0], [output])
        graph.initializer.add().CopyFrom(X1_weight)
        graph.initializer.add().CopyFrom(X1_bias)
        graph.initializer.add().CopyFrom(X3_weight)
        graph.initializer.add().CopyFrom(X3_bias)
        graph.initializer.add().CopyFrom(X5_weight)
        graph.initializer.add().CopyFrom(X5_bias)
        
        model = helper.make_model(graph)
        test_model_path = './test_model_4.onnx'
        onnx.save(model, test_model_path)
        data_reader = TestDataReaderSecond()
        augmented_model_path = './augmented_test_model_4.onnx'
        calibrater = ONNXCalibrater(test_model_path, data_reader,['Conv', 'MatMul'], [], [], augmented_model_path)
        augmented_model = calibrater.augment_graph()
        onnx.save(augmented_model, augmented_model_path)

        #test calculation of quantization params
        #TO_DO: check rmin/rmax
        dict_for_quantization = calibrater.get_intermediate_outputs()
        quantization_params_dict = calibrater.calculate_quantization_params(dict_for_quantization)
        
        #check the size of the quantization dictionary
        self.assertEqual(len(quantization_params_dict), 5)
        
        #check the computation of zp and scale
        for key, value in quantization_params_dict.items():
          
            self.assertTrue(value is not None)
            self.assertTrue(len(value) == 2)
          
            thresholds = dict_for_quantization[key]
            rmin = min(thresholds[0], 0)
            rmax = max(thresholds[1], 0)
            if key == 'X2':  #next_node is Relu
               if rmin < 0: rmin = 0
           
            scale_expected = np.float32((rmax - rmin) / 255 if rmin != rmax else 1)
            zp_expected = np.uint8(round(max(0, min(255, (0 - rmin) / scale_expected))))
            zp_actual = value[0]
            scale_actual = value[1]

            self.assertEqual(zp_expected, zp_actual)
            self.assertEqual(scale_expected, scale_actual)
        
        print('Finished' + ' test calculation of quantization params.')
    

if __name__ == '__main__':
    unittest.main()