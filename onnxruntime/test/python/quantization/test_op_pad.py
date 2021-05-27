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
from onnxruntime.quantization import quantize_static, quantize_dynamic
from op_test_utils import TestDataFeeds, check_model_correctness, check_op_type_count


class TestOpQuatizerPad(unittest.TestCase):
    def input_feeds(self, n, name2shape):
        input_data_list = []
        for i in range(n):
            inputs = {}
            for name, shape in name2shape.items():
                inputs.update({name: np.random.randint(-1, 2, shape).astype(np.float32)})
            input_data_list.extend([inputs])
        dr = TestDataFeeds(input_data_list)
        return dr

    def construct_model_pad(self, output_model_path, pad_mode, pad_input_shape, pad_dims, constant_value=None):
        #    (input)
        #      |
        #     Pad
        #      |
        #    (output)
        rank = len(pad_input_shape)
        self.assertEqual(rank * 2, len(pad_dims))

        input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, pad_input_shape)
        pad_dims_initializer = helper.make_tensor('pad_dims', TensorProto.INT64, [2 * rank], pad_dims)
        output_shape = [sum(e) for e in list(zip(pad_input_shape, pad_dims[:rank], pad_dims[rank:]))]
        output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        inputs = ['input', 'pad_dims']
        initializers = [pad_dims_initializer]
        if (constant_value is not None) and (pad_mode is None or pad_mode == 'constant'):
            constant_value_tensor = helper.make_tensor('padding_value', TensorProto.FLOAT, [], [constant_value])
            inputs.extend(['padding_value'])
            initializers.extend([constant_value_tensor])
        kwargs = {'mode': pad_mode} if pad_mode is not None else {}
        pad_node = helper.make_node('Pad', inputs, ['output'], name='PadNode', **kwargs)

        graph = helper.make_graph([pad_node], 'TestOpQuantizerPad_test_model',
                                  [input_tensor], [output_tensor], initializer=initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7 # use stable onnx ir version

        onnx.save(model, output_model_path)

    def construct_model_conv_pad(self, output_model_path, conv_input_shape, conv_weight_shape,
                                 pad_input_shape, pad_mode, pad_dims, constant_value=None):
        #      (input)
        #          \
        #         Conv
        #        /    \
        #   Identity   Pad
        #    /            \
        # (identity_out)  (output)
        rank = len(pad_input_shape)
        self.assertEqual(rank * 2, len(pad_dims))

        input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, conv_input_shape)

        conv_weight_arr = np.random.randint(-1, 2, conv_weight_shape).astype(np.float32)
        conv_weight_initializer = onnx.numpy_helper.from_array(conv_weight_arr, name='conv1_weight')
        conv_node = onnx.helper.make_node('Conv', ['input', 'conv1_weight'], ['conv_output'], name='conv_node')

        identity_out = helper.make_tensor_value_info('identity_out', TensorProto.FLOAT, pad_input_shape)
        identity_node = helper.make_node('Identity', ['conv_output'], ['identity_out'], name='IdentityNode')

        pad_dims_initializer = helper.make_tensor('pad_dims', TensorProto.INT64, [2 * rank], pad_dims)
        output_shape = [sum(e) for e in list(zip(pad_input_shape, pad_dims[:rank], pad_dims[rank:]))]
        output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
        pad_inputs = ['conv_output', 'pad_dims']
        initializers = [conv_weight_initializer, pad_dims_initializer]
        if (constant_value is not None) and (pad_mode is None or pad_mode == 'constant'):
            constant_value_tensor = helper.make_tensor('padding_value', TensorProto.FLOAT, [], [constant_value])
            pad_inputs.extend(['padding_value'])
            initializers.extend([constant_value_tensor])
        kwargs = {'mode': pad_mode} if pad_mode is not None else {}
        pad_node = helper.make_node('Pad', pad_inputs, ['output'], name='pad_node', **kwargs)

        graph = helper.make_graph([conv_node, identity_node, pad_node], 'TestOpQuantizerPad_test_model',
                                  [input_tensor], [identity_out, output_tensor], initializer=initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7 # use stable onnx ir version
        onnx.save(model, output_model_path)

    def quantize_model(self, model_fp32_path, model_i8_path, data_reader=None):
        if data_reader is not None:
            quantize_static(model_fp32_path, model_i8_path, data_reader, reduce_range=True)
        else:
            quantize_dynamic(model_fp32_path, model_i8_path, reduce_range=True)

    def verify_should_not_trigger(self, quantize_mode='static'):
        np.random.seed(108)
        model_fp32_path = 'qop_pad_notrigger_fp32_{}.onnx'.format(quantize_mode)
        model_i8_path = 'qop_pad_notrigger_i8_{}.onnx'.format(quantize_mode)
        data_reader = self.input_feeds(1, {'input': [1, 16, 31, 31]})
        self.construct_model_pad(model_fp32_path, 'constant', [1, 16, 31, 31], [0, 0, 1, 2, 0, 0, 3, 4])
        self.quantize_model(model_fp32_path, model_i8_path, None if quantize_mode != 'static' else data_reader)
        data_reader.rewind()
        # DequantizeLinear=0 pad node is not been quantized as input is not quantized.
        check_op_type_count(self, model_i8_path, DynamicQuantizeLinear=0, QuantizeLinear=0, DequantizeLinear=0)
        check_model_correctness(self, model_fp32_path, model_i8_path, data_reader.get_next())

    def test_static_quantize_no_trigger(self):
        self.verify_should_not_trigger(quantize_mode='static')

    def test_dynamic_quantize_no_trigger(self):
        self.verify_should_not_trigger(quantize_mode='dynamic')

    def verify_quantize_with_pad_mode(self, pad_mode, constant_value=None, quantize_mode='static'):
        np.random.seed(108)
        tag_pad_mode = pad_mode if pad_mode is not None else 'none'
        tag_constant_value = '' if constant_value is None else '_value'
        model_fp32_path = 'qop_pad_{}_fp32_{}{}.onnx'.format(quantize_mode, tag_pad_mode, tag_constant_value)
        model_i8_path = 'qop_pad_{}_i8_{}{}.onnx'.format(quantize_mode, tag_pad_mode, tag_constant_value)
        data_reader = self.input_feeds(1, {'input': [1, 8, 33, 33]})
        self.construct_model_conv_pad(model_fp32_path, [1, 8, 33, 33], [16, 8, 3, 3], [1, 16, 31, 31],
                                      pad_mode, [0, 0, 1, 2, 0, 0, 3, 4], constant_value=constant_value)
        self.quantize_model(model_fp32_path, model_i8_path, None if quantize_mode != 'static' else data_reader)
        data_reader.rewind()
        # DequantizeLinear=2 means there are one DequantizeLinear Node aftr both conv and pad,
        # which means pad node is running in quantized semantic.
        # In dynamic quantize mode, pad operator in fact not quantized as input is fp32.
        kwargs = {'DynamicQuantizeLinear': 1} if quantize_mode != 'static' else {'DequantizeLinear': 2, 'QuantizeLinear': 1}
        check_op_type_count(self, model_i8_path, **kwargs)
        check_model_correctness(self, model_fp32_path, model_i8_path, data_reader.get_next())

    def test_static_mode_edge(self):
        self.verify_quantize_with_pad_mode('edge', constant_value=None)

    def test_static_mode_reflect(self):
        self.verify_quantize_with_pad_mode('reflect', constant_value=None)

    def test_static_mode_constant_default(self):
        self.verify_quantize_with_pad_mode('constant', constant_value=None)

    def test_static_mode_constant_value(self):
        self.verify_quantize_with_pad_mode('constant', constant_value=3.75)

    def test_dynamic_mode_edge(self):
        self.verify_quantize_with_pad_mode('edge', constant_value=None, quantize_mode='dynamic')

    def test_dynamic_mode_reflect(self):
        self.verify_quantize_with_pad_mode('reflect', constant_value=None, quantize_mode='dynamic')

    def test_dynamic_mode_constant_default(self):
        self.verify_quantize_with_pad_mode('constant', constant_value=None, quantize_mode='dynamic')

    def test_dynamic_mode_constant_value(self):
        self.verify_quantize_with_pad_mode('constant', constant_value=3.75, quantize_mode='dynamic')


if __name__ == '__main__':
    unittest.main()
