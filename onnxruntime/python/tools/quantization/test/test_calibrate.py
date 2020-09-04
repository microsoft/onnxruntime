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


def generate_input_initializer(tensor_shape,tensor_dtype,input_name):
  '''
  Helper function to generate initializers for test inputs
  '''
  tensor = np.random.ranf(tensor_shape).astype(tensor_dtype)
  init = numpy_helper.from_array(tensor,input_name)
  return init  

class TestDataReader(CalibrationDataReader):
    '''for test purpose'''
    def __init__(self, calibration_image_folder):
        self.image_folder = calibration_image_folder

    def get_next(self):
        return None

class TestDataReaderSecond(CalibrationDataReader):
    '''for test purpose'''
    def __init__(self, calibration_image_folder):
        self.image_folder = calibration_image_folder
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.datasize = 3

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            nhwc_data_list = []
            for i in range(3):
                nhwc_data_list.append(np.random.ranf([1,3,1,3]).astype(np.float32))
            input_name = 'input0'
            self.enum_data_dicts = iter([{input_name: nhwc_data_list[i]} for i in range(3)])
        return next(self.enum_data_dicts, None)


class TestCalibrate(unittest.TestCase):
    
    def test_augment_graph(self):

        ''' TEST_CONFIG_1'''

        #   Main graph:
        #    
        #     [A]
        #      |
        #     Conv 
        #      |  ---- [C]
        #     Clip
        #[D]-- |       [B]
        #       \      /
        #        MatMul
        #          |
        #         [E]
        
        A = helper.make_tensor_value_info('A', TensorProto.FLOAT, ())
        B = helper.make_tensor_value_info('B', TensorProto.FLOAT, ())
        C = helper.make_tensor_value_info('C', TensorProto.FLOAT, ())
        D = helper.make_tensor_value_info('D', TensorProto.FLOAT, ())
        E = helper.make_tensor_value_info('E', TensorProto.FLOAT, ())
        conv_node = onnx.helper.make_node('Conv', ['A'], ['C'],name='Conv')
        clip_node = onnx.helper.make_node('Clip', ['C'], ['D'], name='Clip')
        matmul_node = onnx.helper.make_node('MatMul', ['B','D'], ['E'], name='MatMul')
        graph = helper.make_graph([conv_node, clip_node, matmul_node], 'test_graph_1', [A,B], [E])
        model = helper.make_model(graph)
        test_model_path = './test_model_1.onnx'
        onnx.save(model, test_model_path)

        # Augmenting graph
        data_reader = TestDataReader('./test_images')
        augmented_model_path = './augmented_test_model_1.onnx'
        calibrater = ONNXCalibrater(test_model_path, data_reader, ['Conv','MatMul'], [], [], augmented_model_path)
        augmented_model = calibrater.augment_graph()
        onnx.save(augmented_model,augmented_model_path)

        # Checking if each added ReduceMin and ReduceMax node and its output exists
        augmented_model_node_names = [node.name for node in augmented_model.graph.node]
        augmented_model_outputs = [output.name for output in augmented_model.graph.output]
        added_node_names = ['C_ReduceMin', 'C_ReduceMax','D_ReduceMin', 'D_ReduceMax']
        added_outputs = ['C_ReduceMin', 'C_ReduceMax','D_ReduceMin', 'D_ReduceMax']
        # Original 3 nodes + added ReduceMin/Max nodes * 4 (exlude graph input/output)
        self.assertEqual(len(augmented_model_node_names), 7)
        # Original 1 graph output + added outputs * 4
        self.assertEqual(len(augmented_model_outputs), 5)
        for name in added_node_names:
            self.assertTrue(name in augmented_model_node_names)
        for output in added_outputs:
            self.assertTrue(output in augmented_model_outputs)

        print('Finished TEST_CONFIG_1')


        ''' TEST_CONFIG_2 '''

        #  main graph
        #
        #   [F]
        #    |
        #   Conv
        #    |   ---- [G]
        #   Conv
        #    |
        #   [H]
        
        F = helper.make_tensor_value_info('F', TensorProto.FLOAT, ())
        G = helper.make_tensor_value_info('G', TensorProto.FLOAT, ())
        H = helper.make_tensor_value_info('H', TensorProto.FLOAT, ())
        conv_node_1 = onnx.helper.make_node('Conv', ['F'], ['G'], name='Conv_1')
        conv_node_2 = onnx.helper.make_node('Conv', ['G'], ['H'], name='Conv_2')
        graph = helper.make_graph([conv_node_1, conv_node_2], 'test_graph_2', [F], [H])
        model = helper.make_model(graph)
        test_model_path = './test_model_2.onnx'
        onnx.save(model, test_model_path)

        # Augmenting graph
        data_reader = TestDataReader('./test_images')
        augmented_model_path = './augmented_test_model_2.onnx'
        calibrater = ONNXCalibrater(test_model_path, data_reader, ['Conv','MatMul'], [], [], augmented_model_path)
        augmented_model = calibrater.augment_graph()
        onnx.save(augmented_model, augmented_model_path)

        augmented_model_node_names = [node.name for node in augmented_model.graph.node]
        augmented_model_outputs = [output.name for output in augmented_model.graph.output]
        added_node_names = ['G_ReduceMin','G_ReduceMax']
        added_outputs = ['G_ReduceMin','G_ReduceMax']
        # Original 2 nodes + added ReduceMin/Max nodes * 2
        self.assertEqual(len(augmented_model_node_names), 4)
        # Original 1 graph output + added outputs * 2
        self.assertEqual(len(augmented_model_outputs), 3)
        for name in added_node_names:
            self.assertTrue(name in augmented_model_node_names)
        for output in added_outputs:
            self.assertTrue(output in augmented_model_outputs)

        print('Finished TEST_CONFIG_2')
        
        
        '''TEST_CONFIG_3'''
        
        # main graph
        #
        #   [I] 
        #    |
        #   Relu
        #    |  --- [J]
        #   Conv  \ 
        #[M] |     \
        #   Clip    |
        #[N] |     /
        #   Matmul
        #    |
        #   [O]
        
        I = helper.make_tensor_value_info('I', TensorProto.FLOAT, ())
        J = helper.make_tensor_value_info('J', TensorProto.FLOAT, ())
        M = helper.make_tensor_value_info('M', TensorProto.FLOAT, ())
        N = helper.make_tensor_value_info('N', TensorProto.FLOAT, ())
        O = helper.make_tensor_value_info('O', TensorProto.FLOAT, ())
        relu_node = onnx.helper.make_node('Relu', ['I'], ['J'], name='Relu')
        conv_node = onnx.helper.make_node('Conv', ['J'], ['M'], name='Conv')
        clip_node = onnx.helper.make_node('Clip', ['M'], ['N'], name='Clip')
        matmul_node = onnx.helper.make_node('MatMul', ['N','J'], ['O'], name='MatMul')
        graph = helper.make_graph([relu_node, conv_node, clip_node, matmul_node], 'test_graph_3', [I], [O])
        model = helper.make_model(graph)
        test_model_path = './test_model_3.onnx'
        onnx.save(model, test_model_path)

        # Augmenting graph
        data_reader = TestDataReader('./test_images')
        augmented_model_path = './augmented_test_model_3.onnx'
        calibrater = ONNXCalibrater(test_model_path, data_reader, ['Conv','MatMul'], [], [], augmented_model_path)
        augmented_model = calibrater.augment_graph()
        onnx.save(augmented_model, augmented_model_path)

        augmented_model_node_names = [node.name for node in augmented_model.graph.node]
        augmented_model_outputs = [output.name for output in augmented_model.graph.output]
        added_node_names = ['J_ReduceMin','J_ReduceMax','M_ReduceMin','M_ReduceMax','N_ReduceMin','N_ReduceMax']
        added_outputs =  ['J_ReduceMin','J_ReduceMax','M_ReduceMin','M_ReduceMax','N_ReduceMin','N_ReduceMax']
        # Original 4 nodes + added ReduceMin/Max nodes * 6
        self.assertEqual(len(augmented_model_node_names), 10)
        # Original 1 graph output + added outputs * 6
        self.assertEqual(len(augmented_model_outputs), 7)
        for name in added_node_names:
            self.assertTrue(name in augmented_model_node_names)
        for output in added_outputs:
            self.assertTrue(output in augmented_model_outputs)
      
        print('Finished TEST_CONFIG_3')
    
       
        ''' TEST_CONFIG_4'''

        # main graph
        #
        #       [P] 
        #        |     
        #       Relu    
        #     /        \
        #   Conv         \  - [Q] 
        #    | --[R]      \   
        #   Relu          |
        #    | --[S]     /  
        #   Conv        /
        # [T]   \      /   
        #         Add
        #          |
        #         [U]
        
        P = helper.make_tensor_value_info('P', TensorProto.FLOAT, ())
        Q = helper.make_tensor_value_info('Q', TensorProto.FLOAT, ())
        R = helper.make_tensor_value_info('R', TensorProto.FLOAT, ())
        S = helper.make_tensor_value_info('S', TensorProto.FLOAT, ())
        T = helper.make_tensor_value_info('T', TensorProto.FLOAT, ())
        U = helper.make_tensor_value_info('U', TensorProto.FLOAT, ())
        relu_node_1 = onnx.helper.make_node('Relu', ['P'], ['Q'], name='Relu1')
        conv_node_1 = onnx.helper.make_node('Conv', ['Q'], ['R'], name='Conv1')
        relu_node_2 = onnx.helper.make_node('Relu',['R'],['S'], name= 'Relu2')
        conv_node_2 = onnx.helper.make_node('Conv', ['S'], ['T'], name='Conv2')
        add_node = onnx.helper.make_node('Add', ['Q','T'], ['U'], name='Add')
      
        graph = helper.make_graph([relu_node_1, conv_node_1, relu_node_2, conv_node_2,add_node], 'test_graph_4', [P], [U])
        model = helper.make_model(graph)
        test_model_path = './test_model_4.onnx'
        onnx.save(model, test_model_path)

        #Augmenting graph
        data_reader = TestDataReader('./test_images')
        augmented_model_path = './augmented_test_model_4.onnx'
        calibrater = ONNXCalibrater(test_model_path, data_reader, ['Conv','MatMul'], [], [], augmented_model_path)
        augmented_model = calibrater.augment_graph()
        onnx.save(augmented_model, augmented_model_path)

        augmented_model_node_names = [node.name for node in augmented_model.graph.node]
        augmented_model_outputs = [output.name for output in augmented_model.graph.output]
        added_node_names = ['Q_ReduceMin','Q_ReduceMax','R_ReduceMin','R_ReduceMax','S_ReduceMin','S_ReduceMax','T_ReduceMin','T_ReduceMax']
        added_outputs = added_node_names = ['Q_ReduceMin','Q_ReduceMax','R_ReduceMin','R_ReduceMax','S_ReduceMin','S_ReduceMax','T_ReduceMin','T_ReduceMax']
        # Original 5 nodes + added ReduceMin/Max nodes * 8
        self.assertEqual(len(augmented_model_node_names), 13)
        # Original 1 graph output + added outputs * 8
        self.assertEqual(len(augmented_model_outputs), 9)
        for name in added_node_names:
            self.assertTrue(name in augmented_model_node_names)
        for output in added_outputs:
            self.assertTrue(output in augmented_model_outputs)

        print('Finished TEST_CONFIG_4')


    def test_quant_param_calculation(self):
        
        '''TEST_CONFIG'''

        # main graph
        #
        #  [input_0] 
        #    |     
        #   Relu      
        #    |  [X1]    \
        #   Conv         \   
        #    |  [X2]      \   
        #   Relu           |
        #    |  [X3]       /  [X1]
        #   Conv        Conv
        #      \ [X4]    /  [X5] 
        #          |
        #         Add
        #          |
        #       [output_0]

        input0 = helper.make_tensor_value_info('input0', TensorProto.FLOAT, [1,3,1,3])
        X1 = helper.make_tensor_value_info('X1', TensorProto.FLOAT,[])
        X2 = helper.make_tensor_value_info('X2', TensorProto.FLOAT,[])
        X3 = helper.make_tensor_value_info('X3', TensorProto.FLOAT,[])
        X4 = helper.make_tensor_value_info('X4', TensorProto.FLOAT,[])
        X5 = helper.make_tensor_value_info('X5', TensorProto.FLOAT,[])
        output0 = helper.make_tensor_value_info('output0',TensorProto.FLOAT,[1,3,1,3])
        
        X1_weight = generate_input_initializer([3,3,1,1],np.float32,'X1_weight')
        X1_bias = generate_input_initializer([3],np.float32,'X1_bias')
        X3_weight = generate_input_initializer([3,3,1,1],np.float32,'X3_weight')
        X3_bias = generate_input_initializer([3],np.float32,'X3_bias')
        X5_weight = generate_input_initializer([3,3,1,1],np.float32,'X5_weight')
        X5_bias = generate_input_initializer([3],np.float32,'X5_bias')
       
        relu_node_1 = onnx.helper.make_node('Relu', ['input0'], ['X1'], name='Relu1')
        conv_node_1 = onnx.helper.make_node('Conv', ['X1','X1_weight','X1_bias'], ['X2'], name='Conv1')
        relu_node_2 = onnx.helper.make_node('Relu',['X2'],['X3'], name= 'Relu2')
        conv_node_2 = onnx.helper.make_node('Conv', ['X3','X3_weight','X3_bias'], ['X4'], name='Conv2')
        conv_node_3 = onnx.helper.make_node('Conv', ['X1','X5_weight','X5_bias'], ['X5'], name='Conv3')
        add_node = onnx.helper.make_node('Add', ['X4','X5'], ['output0'], name='Add')
      
        graph = helper.make_graph([relu_node_1, conv_node_1, relu_node_2, conv_node_2, conv_node_3, add_node], 'test_graph_5', [input0], [output0])
        graph.initializer.add().CopyFrom(X1_weight)
        graph.initializer.add().CopyFrom(X1_bias)
        graph.initializer.add().CopyFrom(X3_weight)
        graph.initializer.add().CopyFrom(X3_bias)
        graph.initializer.add().CopyFrom(X5_weight)
        graph.initializer.add().CopyFrom(X5_bias)
        
        model = helper.make_model(graph)
        test_model_path = './test_model_5.onnx'
        onnx.save(model, test_model_path)
        data_reader = TestDataReaderSecond('./test_images')
        augmented_model_path = './augmented_test_model_5.onnx'
        calibrater = ONNXCalibrater(test_model_path, data_reader,['Conv','MatMul'], [], [], augmented_model_path)
        augmented_model = calibrater.augment_graph()
        onnx.save(augmented_model, augmented_model_path)

        #test calculation of quantization params
        dict_for_quantization = calibrater.get_intermediate_outputs()
        quantization_params_dict = calibrater.calculate_quantization_params(dict_for_quantization)
        
        #check the size of the quantization dictionary
        self.assertEqual(len(quantization_params_dict), 5)
        
        #check the computation of zp and scale
        for key,value in quantization_params_dict.items():
          
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
            self.assertEqual(scale_expected,scale_actual)
        
        print('Finished' + '  test calculation of quantization params.')
    

if __name__ == '__main__':
    unittest.main()
