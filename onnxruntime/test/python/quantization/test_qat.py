# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# -*- coding: UTF-8 -*-

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto, ValueInfoProto
from onnx import shape_inference
import onnxruntime
from pathlib import Path
import unittest
import urllib.request

from onnxruntime.quantization.quantize import ONNXQuantizer

from onnxruntime.quantization.quant_utils import QuantizationMode
from onnx import onnx_pb as onnx_proto


def generate_input_initializer(tensor_shape, tensor_dtype, input_name):
    '''
  Helper function to generate initializers for inputs
  '''
    tensor = np.random.ranf(tensor_shape).astype(tensor_dtype)
    init = numpy_helper.from_array(tensor, input_name)
    return init


def generate_qat_model(model_names):

    test_models = []
    test_initializers = []
    '''
    TEST_MODEL_CONFIG_1
    '''
    # Main graph:
    #
    #   [A]  [input_bias]
    #     \     /
    #      Add      [scale_zp_const] [input_weight]
    #       |                   \      /
    #       |             QuantizeLinear_1
    #  QuantizeLinear_0           |
    #       |             DequantizeLinear_1
    #       |                  /
    #  DequantizeLinear_0   Transpose
    #        \              /
    #         \            /        <--- (actual graph: this branch is folded)
    #             Matmul
    #               |
    #               |
    #              [B]

    graph = helper.make_graph(
        [
            #nodes
            helper.make_node("Add", ["A", "input_bias"], ["add_out"], "add0"),
            helper.make_node("QuantizeLinear", ["add_out", "quant0_scale_const", "quant0_zp_const"], ["quant0_out"],
                             "qlinear0"),
            helper.make_node("DequantizeLinear", ["quant0_out", "dequant0_scale_const", "dequant0_zp_const"],
                             ["dequant0_out"], "dqlinear0"),
            helper.make_node("MatMul", ["dequant0_out", "trans_out"], ["B"], "matmul"),
        ],
        "QAT_model_1",  #name
        [  #input
            helper.make_tensor_value_info('A', TensorProto.FLOAT, ['unk_1'])
        ],
        [  #output
            helper.make_tensor_value_info('B', TensorProto.FLOAT, [1024])
        ],
        [  #initializers
            helper.make_tensor('quant0_scale_const', TensorProto.FLOAT, [], [0.01961481384932995]),
            helper.make_tensor('quant0_zp_const', TensorProto.INT8, [], [0]),
            helper.make_tensor('dequant0_scale_const', TensorProto.FLOAT, [], [0.01961481384932995]),
            helper.make_tensor('dequant0_zp_const', TensorProto.INT8, [], [0]),
        ])
    input_weight_1 = generate_input_initializer([1024, 1024], np.float32, 'trans_out')
    input_bias_1 = generate_input_initializer([1024], np.float32, 'input_bias')
    graph.initializer.add().CopyFrom(input_weight_1)
    graph.initializer.add().CopyFrom(input_bias_1)

    model_1 = onnx.helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model_1.ir_version = 7 # use stable onnx ir version
    onnx.save(model_1, model_names[0])

    test_models.extend([model_1])
    initiazliers_1 = [input_weight_1, input_bias_1]
    test_initializers.append(initiazliers_1)
    '''
      TEST_MODEL_CONFIG_2
    '''

    # Main graph:
    #
    #                  [A]
    #                   |
    #                MaxPool
    #               /        \
    #  QuantizeLinear_0     QuantizeLinear_1
    #       |                      |
    #  DequantizeLinear_0     DequantizeLinear_1
    #        |                      |
    #      Conv_0-[weight,bias]   Conv_1-[weight,bias]
    #        \                     /
    #         \                   /
    #                 Add
    #                  |
    #                 [B]

    graph = helper.make_graph(
        [
            #nodes
            helper.make_node("MaxPool", ["A"], ["maxpool_out"], "maxpool", kernel_shape = [1, 1]),
            helper.make_node("QuantizeLinear", ["maxpool_out", "quant0_scale_const", "quant0_zp_const"], ["quant0_out"],
                             "qlinear0"),
            helper.make_node("DequantizeLinear", ["quant0_out", "dequant0_scale_const", "dequant0_zp_const"],
                             ["dequant0_out"], "dqlinear0"),
            helper.make_node("Conv", ["dequant0_out", "conv_weight_0", "conv_bias_0"], ["conv0_out"], "conv0"),
            helper.make_node("QuantizeLinear", ["maxpool_out", "quant1_scale_const", "quant1_zp_const"], ["quant1_out"],
                             "qlinear1"),
            helper.make_node("DequantizeLinear", ["quant1_out", "dequant1_scale_const", "dequant1_zp_const"],
                             ["dequant1_out"], "dqlinear1"),
            helper.make_node("Conv", ["dequant1_out", "conv_weight_1", "conv_bias_1"], ["conv1_out"], "conv1"),
            helper.make_node("Add", ["conv0_out", "conv1_out"], ["B"], "add"),
        ],
        "QAT_model_2",  #name
        [  #input
            helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 64, 256, 256])
        ],
        [  #output
            helper.make_tensor_value_info('B', TensorProto.FLOAT, [1, 256, 256, 256])
        ],
        [  #initializers
            helper.make_tensor('quant0_scale_const', TensorProto.FLOAT, [], [0.2062656134366989]),
            helper.make_tensor('quant0_zp_const', TensorProto.UINT8, [], [165]),
            helper.make_tensor('dequant0_scale_const', TensorProto.FLOAT, [], [0.2062656134366989]),
            helper.make_tensor('dequant0_zp_const', TensorProto.UINT8, [], [165]),
            helper.make_tensor('quant1_scale_const', TensorProto.FLOAT, [], [0.10088317096233368]),
            helper.make_tensor('quant1_zp_const', TensorProto.UINT8, [], [132]),
            helper.make_tensor('dequant1_scale_const', TensorProto.FLOAT, [], [0.10088317096233368]),
            helper.make_tensor('dequant1_zp_const', TensorProto.UINT8, [], [132]),
        ])

    conv_weight_0 = generate_input_initializer([256, 64, 1, 1], np.float32, 'conv_weight_0')
    conv_bias_0 = generate_input_initializer([256], np.float32, 'conv_bias_0')
    graph.initializer.add().CopyFrom(conv_weight_0)
    graph.initializer.add().CopyFrom(conv_bias_0)

    conv_weight_1 = generate_input_initializer([256, 64, 1, 1], np.float32, 'conv_weight_1')
    conv_bias_1 = generate_input_initializer([256], np.float32, 'conv_bias_1')
    graph.initializer.add().CopyFrom(conv_weight_1)
    graph.initializer.add().CopyFrom(conv_bias_1)

    model_2 = onnx.helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model_2.ir_version = 7 # use stable onnx ir version
    onnx.save(model_2, model_names[1])

    test_models.extend([model_2])
    initializers_2 = [conv_weight_0, conv_bias_0, conv_weight_1, conv_weight_1]
    test_initializers.append(initializers_2)

    return test_models, test_initializers


def generate_qat_support_model(model_names, test_initializers):
    '''
      EXPECTED_TEST_RESULT_CONFIG_1
    '''

    test_qat_support_models = []

    # Main graph:

    #   [A]  [input_bias]
    #     \    /
    #       Add         [Transpose_output]
    #         \             |
    #          \           /
    #              Matmul -([input_weight])
    #               |
    #               |
    #              [B]
    graph = helper.make_graph(
        [  #nodes
            helper.make_node("Add", ["A", "input_bias"], ["add_out"], "add0"),
            helper.make_node("MatMul", ["add_out", "trans_out"], ["B"], "matmul"),
        ],
        "QAT_support_model_1",  #name
        [
            #input
            helper.make_tensor_value_info('A', TensorProto.FLOAT, ['unk_1'])
        ],
        [
            #output
            helper.make_tensor_value_info('B', TensorProto.FLOAT, [1024])
        ])

    #initializers
    init_1 = test_initializers[0]
    for init in init_1:
        graph.initializer.add().CopyFrom(init)

    model_1 = onnx.ModelProto()
    model_1.ir_version = 7 # use stable onnx ir version
    model_1 = onnx.helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model_1, model_names[0])

    test_qat_support_models.extend([model_1])
    '''
      EXPECTED_TEST_RESULT_CONFIG_2
    '''
    # Main graph:

    #                  [A]
    #                   |
    #                MaxPool
    #               /        \
    #  Conv_0-[weight,bias]   Conv_1-[weight,bias]
    #        \                     /
    #         \                   /
    #                 Add
    #                  |
    #                 [B]
    graph = helper.make_graph(
        [  #nodes
            helper.make_node("MaxPool", ["A"], ["maxpool_out"], "maxpool"),
            helper.make_node("Conv", ["maxpool_out", "conv_weight_0", "conv_bias_0"], ["conv0_out"], "conv0"),
            helper.make_node("Conv", ["maxpool_out", "conv_weight_1", "conv_bias_1"], ["conv1_out"], "conv1"),
            helper.make_node("Add", ["conv0_out", "conv1_out"], ["B"], "add"),
        ],
        "QAT_support_model_2",  #name
        [  #input
            helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 64, 256, 256])
        ],
        [  #output
            helper.make_tensor_value_info('B', TensorProto.FLOAT, [1, 256, 256, 256])
        ])

    #initializers
    init_2 = test_initializers[1]
    for init in init_2:
        graph.initializer.add().CopyFrom(init)

    model_2 = onnx.ModelProto()
    model_2.ir_version = 7 # use stable onnx ir version
    model_2 = onnx.helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model_1, model_names[1])

    test_qat_support_models.extend([model_2])

    return test_qat_support_models


def compare_two_models(model_1, model_2):
    '''
    Helper function to check if two models are the same
    :param: model_1 - expected model
    :param: model_2 - actual model
    Return true if two models are the same. Otherwise return false.
    '''
    check_1, check_2 = True, True

    #check nodes
    for node_1 in model_1.graph.node:
        node_found = False
        for node_2 in model_2.graph.node:
            if node_2.name == node_1.name:
                node_found = True
                if node_2.input != node_1.input or node_2.output != node_1.output:
                    check_1 = False
                    print("Error: Node {} in test model dismatch with the expected model.".format(node_2.name))
                break

        if not node_found:
            check_1 = False
            print("Error:Node {} in the expected model not found in test model.".format(node_1.name))
            break

    #check initializers:
    for init_1 in model_1.graph.initializer:
        init1_arr = numpy_helper.to_array(init_1)
        init_found = False
        for init_2 in model_2.graph.initializer:
            if init_2.name == init_1.name:
                init_found = True
                init2_arr = numpy_helper.to_array(init_2)
                if not np.array_equal(init1_arr, init2_arr):
                    check_2 = False
                    print("Error:  Initializer {} in test model dismatches with the expected model.".format(
                        init_2.name))
                break

        if not init_found:
            check_2 = False
            print("Error: Initializer {} in the expected model not found in test model.".format(init_1.name))
            break

    return check_1 and check_2


class TestQAT(unittest.TestCase):
    def test_remove_fakequant_nodes(self):

        model_names = ["qat_model_1.onnx", "qat_model_2.onnx"]
        qat_support_model_names = ["qat_support_model_1.onnx", "qat_support_model_2.onnx"]

        test_models, test_initializers = generate_qat_model(model_names)
        qat_support_models_expected = generate_qat_support_model(qat_support_model_names, test_initializers)

        for i in range(len(test_models)):
            quantizer = ONNXQuantizer(test_models[i], False, False,QuantizationMode.IntegerOps, False, TensorProto.INT8,
                                      TensorProto.INT8, None, None, None, ['Conv', 'MatMul', 'MaxPool'])
            #test remove editting to the graph
            qat_support_model_actual = quantizer.remove_fake_quantized_nodes()

            assert compare_two_models(qat_support_models_expected[i], qat_support_model_actual)
            print("TEST_MODEL {} finished:  ".format(i) + qat_support_model_names[i])


if __name__ == '__main__':
    unittest.main()
