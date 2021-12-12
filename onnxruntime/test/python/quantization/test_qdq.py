#!/usr/bin/env python
# coding: utf-8
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
import onnx
import numpy as np
from onnx import helper, TensorProto
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat, QuantizationMode, QDQQuantizer
from op_test_utils import TestDataFeeds, check_model_correctness, check_op_type_count, check_op_type_order

class TestQDQFormat(unittest.TestCase):
    def input_feeds(self, n, name2shape):
        input_data_list = []
        for i in range(n):
            inputs = {}
            for name, shape in name2shape.items():
                inputs.update({name: np.random.randint(-1, 2, shape).astype(np.float32)})
            input_data_list.extend([inputs])
        dr = TestDataFeeds(input_data_list)
        return dr

class TestQDQExtraOptions(unittest.TestCase):
    def test_qdq_extra_options(self):
        #   (input) 
        #      |    
        #     Add 
        #      |
        #     ReduceMean 
        #      |
        #     Add 
        #      |
        #   (output)

        initializers = []

        input_tensor = helper.make_tensor_value_info('L', TensorProto.FLOAT, [5, 5])
        output_tensor = helper.make_tensor_value_info('O', TensorProto.FLOAT, [5, 5])

        add_weight_data_1 = np.random.normal(0, 0.1, [5, 5]).astype(np.float32)
        initializers.append(onnx.numpy_helper.from_array(add_weight_data_1, name="M"))
        add_weight_data_2 = np.random.normal(0, 0.1, [5, 5]).astype(np.float32)
        initializers.append(onnx.numpy_helper.from_array(add_weight_data_2, name="N"))

        add_node_1 = onnx.helper.make_node('Add', ['L', 'M'], ['P'], name='Add1')
        reduce_mean_node = onnx.helper.make_node('ReduceMean', ['P'], ['Q'], keepdims=1, name='ReduceMean')
        add_node_2 = onnx.helper.make_node('Add', ['Q', 'N'], ['O'], name='Add2')

        graph = helper.make_graph([add_node_1, reduce_mean_node, add_node_2], 'QDQ_Test_Finetune', [input_tensor], [output_tensor], initializer=initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        test_model_path = './test_qdq_finetune.onnx'
        onnx.save(model, test_model_path)

        compute_range = {
            'P': [0.1, 0.1],
            'Q': [0.1, 0.1],
            'M': [0.1, 0.1],
            'N': [0.1, 0.1],
            'L': [0.1, 0.1],
            'O': [0.1, 0.1],
        }

        op_types_to_quantize = ['Add']

        mode = QuantizationMode.QLinearOps
        model = onnx.load_model(test_model_path, False)
        quantizer = QDQQuantizer(
            model,
            True, #per_channel
            False, #reduce_range
            mode,
            True,  #static
            QuantType.QInt8, #weight_type
            QuantType.QInt8, #activation_type
            compute_range,
            [], #nodes_to_quantize
            ['Add2'], #nodes_to_exclude
            op_types_to_quantize,
            {'ActivationSymmetric' : True, 'AddQDQPairToWeight' : True, 'OpTypesToExcludeOutputQuantizatioin': []}) #extra_options
        quantizer.quantize_model()
        qdq_model_path = './test_qdq_finetune_qdq.onnx'
        quantizer.model.save_model_to_file(qdq_model_path, False)

        # QDQ pair should be added to Add1 but not Add2
        # QDQ pair shoud be added to Add1 output as well.
        qdq_added_to_node_output_flag = False 
        for node in quantizer.model.nodes():
            if node.name == 'Add1':
                for input in node.input:
                    self.assertTrue("DequantizeLinear" in input)
                for output in node.output:
                    self.assertTrue("QuantizeLinear" not in output)

            if node.name == 'Add2':
                for input in node.input:
                    self.assertTrue("DequantizeLinear" not in input)
                for output in node.output:
                    self.assertTrue("QuantizeLinear" not in output)

            # This QuantizeLinear node should be followed by Add1
            if node.name == 'P_QuantizeLinear':
                qdq_added_to_node_output_flag = True
                self.assertTrue(node.input[0] == 'P')

        self.assertTrue(qdq_added_to_node_output_flag)


    def test_qdq_extra_options_2(self):
        #         (input) 
        #           |    
        #          Add 
        #       /   |   \
        #  MatMul MatMul MatMul 
        #     |     |      |
        # (output)(output)(output)

        initializers = []

        input_tensor = helper.make_tensor_value_info('L', TensorProto.FLOAT, [5, 5])
        output_tensor1 = helper.make_tensor_value_info('M', TensorProto.FLOAT, [5, 5])
        output_tensor2 = helper.make_tensor_value_info('N', TensorProto.FLOAT, [5, 5])
        output_tensor3 = helper.make_tensor_value_info('O', TensorProto.FLOAT, [5, 5])

        add_weight_data = np.random.normal(0, 0.1, [5, 5]).astype(np.float32)
        initializers.append(onnx.numpy_helper.from_array(add_weight_data, name="P"))
        matmul_weight_data_1 = np.random.normal(0, 0.1, [5, 5]).astype(np.float32)
        initializers.append(onnx.numpy_helper.from_array(matmul_weight_data_1, name="Q"))
        matmul_weight_data_2 = np.random.normal(0, 0.1, [5, 5]).astype(np.float32)
        initializers.append(onnx.numpy_helper.from_array(matmul_weight_data_2, name="R"))
        matmul_weight_data_3 = np.random.normal(0, 0.1, [5, 5]).astype(np.float32)
        initializers.append(onnx.numpy_helper.from_array(matmul_weight_data_2, name="S"))

        add_node = onnx.helper.make_node('Add', ['L', 'P'], ['T'], name='Add')
        matmul_node_1 = onnx.helper.make_node('MatMul', ['T', 'Q'], ['M'], name='MatMul1')
        matmul_node_2 = onnx.helper.make_node('MatMul', ['T', 'R'], ['N'], name='MatMul2')
        matmul_node_3 = onnx.helper.make_node('MatMul', ['T', 'S'], ['O'], name='MatMul3')

        graph = helper.make_graph([add_node, matmul_node_1, matmul_node_2, matmul_node_3], 'QDQ_Test_Finetune_2', [input_tensor], [output_tensor1, output_tensor2, output_tensor3], initializer=initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        test_model_path = './test_qdq_finetune_2.onnx'
        onnx.save(model, test_model_path)

        compute_range = {
            'L': [0.1, 0.1],
            'M': [0.1, 0.1],
            'N': [0.1, 0.1],
            'O': [0.1, 0.1],
            'P': [0.1, 0.1],
            'Q': [0.1, 0.1],
            'R': [0.1, 0.1],
            'S': [0.1, 0.1],
            'T': [0.1, 0.1],
        }

        op_types_to_quantize = ['Add', 'MatMul']

        mode = QuantizationMode.QLinearOps
        model = onnx.load_model(test_model_path, False)
        quantizer = QDQQuantizer(
            model,
            True, #per_channel
            False, #reduce_range
            mode,
            True,  #static
            QuantType.QInt8, #weight_type
            QuantType.QInt8, #activation_type
            compute_range,
            [], #nodes_to_quantize
            ['Add'], #nodes_to_exclude
            op_types_to_quantize,
            {'ActivationSymmetric' : True, 'AddQDQPairToWeight' : True, 'OpTypesToExcludeOutputQuantizatioin': op_types_to_quantize, 'DedicatedQDQPair': True}) #extra_options
        quantizer.quantize_model()
        qdq_model_path = './test_qdq_finetune_qdq_2.onnx'
        quantizer.model.save_model_to_file(qdq_model_path, False)

        # Three dedicated QDQ pair should be generated and feed into each MatMul node
        # Also QDQ pair should not be added to Add node 
        # QDQ pair shoud not be added to node's output
        for node in quantizer.model.nodes():
            if node.name == 'MatMul1':
                self.assertTrue("T_DequantizeLinear_1" in node.input)
            if node.name == 'MatMul2':
                self.assertTrue("T_DequantizeLinear_2" in node.input)
            if node.name == 'MatMul3':
                self.assertTrue("T_DequantizeLinear_3" in node.input)
            if node.name == 'Add':
                for input in node.input:
                    self.assertTrue("DequantizeLinear" not in input)

            # QDQ pair shoud not be added to MatMul's output
            if node.op_type == 'QuantizeLinear':
                self.assertTrue(node.input[0] not in ['M_QuantizeLinearInput', 'N_QuantizeLinearInput', 'O_QuantizeLinearInput']) 

class TestQDQFormatConv(TestQDQFormat):
    def construct_model_conv(self, output_model_path, input_shape, weight_shape, output_shape, has_bias):
        #    (input)
        #      |
        #     Conv
        #      |
        #    (output)
        input_name = 'input'
        output_name = 'output'
        initializers = []

        # make Conv node
        weight_name = 'conv_weight'
        bias_name = 'conv_bias'
        conv_inputs = [input_name, weight_name]
        conv_outputs = [output_name]
        conv_name = 'conv_node'
        conv_weight_data = np.random.normal(0, 0.1, weight_shape).astype(np.float32)
        initializers.append(onnx.numpy_helper.from_array(conv_weight_data, name=weight_name))
        if has_bias:
            conv_inputs.append(bias_name)
            bias_data = np.random.normal(0, 0.05, (weight_shape[0])).astype(np.float32)
            initializers.append(onnx.numpy_helper.from_array(bias_data, name=bias_name))
        conv_node = onnx.helper.make_node('Conv', conv_inputs, conv_outputs, name=conv_name)

        # make graph
        input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape)
        output_tensor = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, output_shape)
        graph_name = 'QDQ_Test_Conv'
        graph = helper.make_graph([conv_node], graph_name,
                                  [input_tensor], [output_tensor], initializer=initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7 # use stable onnx ir version

        onnx.save(model, output_model_path)

    def verify_quantize_conv(self, has_bias, per_channel):
        np.random.seed(1)
        model_fp32_path = 'conv_fp32.{}.{}.onnx'.format(has_bias, per_channel)
        model_int8_qdq_path = 'conv_quant_qdq.{}.{}.onnx'.format(has_bias, per_channel)
        model_int8_qop_path = 'conv_quant_qop.{}.{}.onnx'.format(has_bias, per_channel)
        data_reader = self.input_feeds(1, {'input': [1, 8, 33, 33]})
        self.construct_model_conv(model_fp32_path,
                                  [1, 8, 33, 33],
                                  [16, 8, 3, 3],
                                  [1, 16, 31, 31],
                                  has_bias)
        quantize_static(model_fp32_path,
                        model_int8_qdq_path,
                        data_reader,
                        quant_format=QuantFormat.QDQ,
                        per_channel = per_channel,
                        reduce_range = per_channel
                        )
        data_reader.rewind()
        qdq_nodes = {'Conv': 1, 'QuantizeLinear': 2, 'DequantizeLinear': 4 if has_bias else 3}
        check_op_type_count(self, model_int8_qdq_path, **qdq_nodes)
        check_model_correctness(self, model_fp32_path, model_int8_qdq_path, data_reader.get_next())

        data_reader.rewind()
        quantize_static(model_fp32_path,
                        model_int8_qop_path,
                        data_reader,
                        quant_format=QuantFormat.QOperator,
                        per_channel = per_channel,
                        reduce_range = per_channel
                        )
        data_reader.rewind()
        qop_nodes = {'QLinearConv': 1, 'QuantizeLinear': 1, 'DequantizeLinear': 1}
        check_op_type_count(self, model_int8_qop_path, **qop_nodes)
        check_model_correctness(self, model_fp32_path, model_int8_qop_path, data_reader.get_next())

    def test_quantize_conv_without_bias(self):
        self.verify_quantize_conv(False, False) # has_bias:False, per_channel:False
        self.verify_quantize_conv(False, True) # has_bias:False, per_channel:True
        self.verify_quantize_conv(True, False) # has_bias:True, per_channel:False
        self.verify_quantize_conv(True, True) # has_bias:True, per_channel:True

class TestQDQFormatConvClip(TestQDQFormat):
    def construct_model_conv_clip(self, output_model_path, input_shape, weight_shape, output_shape):
        #    (input)
        #      |
        #     Conv
        #      |
        #     Clip
        #      |
        #   Reshape
        #      |
        #    (output)
        input_name = 'input'
        output_name = 'output'
        initializers = []

        # make Conv node
        weight_name = 'conv_weight'
        conv_inputs = [input_name, weight_name]
        conv_outputs = ['conv_output']
        conv_name = 'conv_node'
        conv_weight_data = np.random.normal(0, 0.1, weight_shape).astype(np.float32)
        initializers.append(onnx.numpy_helper.from_array(conv_weight_data, name=weight_name))
        conv_node = onnx.helper.make_node('Conv', conv_inputs, conv_outputs, name=conv_name)

        # make Clip node
        clip_min_name = 'clip_min'
        clip_max_name = 'clip_max'
        clip_inputs = [conv_outputs[0], clip_min_name, clip_max_name]
        clip_outputs = ['clip_output']
        clip_name = 'clip_node'
        initializers.append(onnx.numpy_helper.from_array(np.array(-1.0, dtype=np.float32), name=clip_min_name))
        initializers.append(onnx.numpy_helper.from_array(np.array(1.0, dtype=np.float32), name=clip_max_name))
        clip_node = onnx.helper.make_node('Clip', clip_inputs, clip_outputs, name=clip_name)

        # make Identity node
        reshape_name = 'reshape_node'
        reshape_shape = 'reshape_shape'
        initializers.append(onnx.numpy_helper.from_array(np.array([-1], dtype=np.int64), name=reshape_shape))
        reshape_node = onnx.helper.make_node('Reshape', ['clip_output', reshape_shape], [output_name], name=reshape_name)

        # make graph
        input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape)
        output_tensor = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, output_shape)
        graph_name = 'QDQ_Test_Conv_clip'
        graph = helper.make_graph([conv_node, clip_node, reshape_node], graph_name,
                                  [input_tensor], [output_tensor], initializer=initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7 # use stable onnx ir version

        onnx.save(model, output_model_path)

    def verify(self, per_channel):
        np.random.seed(1)
        model_fp32_path = 'conv_clip_fp32.{}.onnx'.format(per_channel)
        model_int8_qdq_path = 'conv_clip_quant_qdq.{}.onnx'.format(per_channel)
        model_int8_qop_path = 'conv_clip_quant_qop.{}.onnx'.format(per_channel)
        data_reader = self.input_feeds(1, {'input': [1, 8, 33, 33]})
        self.construct_model_conv_clip(model_fp32_path,
                                       [1, 8, 33, 33],
                                       [16, 8, 3, 3],
                                       [15376])
        quantize_static(model_fp32_path,
                        model_int8_qdq_path,
                        data_reader,
                        quant_format=QuantFormat.QDQ,
                        per_channel = per_channel,
                        reduce_range = per_channel
                        )
        data_reader.rewind()
        #topo sort check
        check_op_type_order(self, model_int8_qdq_path, ['DequantizeLinear', 'QuantizeLinear', 'DequantizeLinear', 'Conv', 'QuantizeLinear', 'DequantizeLinear', 'Reshape', 'QuantizeLinear', 'DequantizeLinear'])
        check_model_correctness(self, model_fp32_path, model_int8_qdq_path, data_reader.get_next())

        data_reader.rewind()
        quantize_static(model_fp32_path,
                        model_int8_qop_path,
                        data_reader,
                        quant_format=QuantFormat.QOperator,
                        per_channel = per_channel,
                        reduce_range = per_channel
                        )
        data_reader.rewind()
        qop_nodes = {'QLinearConv': 1, 'QuantizeLinear': 1, 'DequantizeLinear': 1}
        check_op_type_count(self, model_int8_qop_path, **qop_nodes)
        check_model_correctness(self, model_fp32_path, model_int8_qop_path, data_reader.get_next())

    def test_quantize_conv_without_bias(self):
        self.verify(False) # per_channel:False
        #self.verify(True) # per_channel:True

if __name__ == '__main__':
    unittest.main()
