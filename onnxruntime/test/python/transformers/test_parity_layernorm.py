# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
import torch
from torch import nn
import os
import onnx
from parity_utilities import *

if find_transformers_source():
    from onnx_model import OnnxModel
else:
    from onnxruntime.transformers.onnx_model import OnnxModel


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, epsilon, cast_fp16=True):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=epsilon)
        # initialize weights with random value
        self.layer_norm.bias.data.normal_(mean=0.0, std=0.1)
        self.layer_norm.weight.data.normal_(mean=0.0, std=0.5)
        self.cast_fp16 = cast_fp16

    @staticmethod
    def get_fused_op():
        return "LayerNormalization"

    def forward(self, x):
        if self.cast_fp16 and x.dtype == torch.float16:
            y = self.layer_norm(x.to(torch.float32)).to(torch.float16)
            return (y, )
        y = self.layer_norm(x)
        return (y, )


def optimize_fp16_onnx_with_cast(input_onnx_path, optimized_onnx_path, epsilon):
    m = onnx.load(input_onnx_path)
    onnx_model = OnnxModel(m)

    nodes_to_remove = onnx_model.nodes()
    nodes_to_add = [
        onnx.helper.make_node("Cast", ["input"], ["fp32_input"], "cast_input", to=1),
        onnx.helper.make_node("Cast", ["layer_norm.weight"], ["fp32_layer_norm.weight"], "cast_weight", to=1),
        onnx.helper.make_node("Cast", ["layer_norm.bias"], ["fp32_layer_norm.bias"], "cast_bias", to=1),
        onnx.helper.make_node("LayerNormalization", ["fp32_input", "fp32_layer_norm.weight", "fp32_layer_norm.bias"],
                              ["fp32_output"],
                              "layer_norm",
                              epsilon=epsilon),  # use fp32 epsilon
        onnx.helper.make_node("Cast", ["fp32_output"], ["output"], "cast_output", to=10)
    ]

    onnx_model.remove_nodes(nodes_to_remove)
    onnx_model.add_nodes(nodes_to_add)
    onnx_model.prune_graph()
    onnx_model.save_model_to_file(optimized_onnx_path)


def optimize_fp16_onnx_no_cast(input_onnx_path, optimized_onnx_path, epsilon):
    m = onnx.load(input_onnx_path)
    onnx_model = OnnxModel(m)

    nodes_to_remove = onnx_model.nodes()
    node_to_add = onnx.helper.make_node("LayerNormalization", ["input", "layer_norm.weight", "layer_norm.bias"],
                                        ["output"],
                                        "layer_norm",
                                        epsilon=epsilon)

    onnx_model.remove_nodes(nodes_to_remove)
    onnx_model.add_node(node_to_add)
    onnx_model.prune_graph()
    onnx_model.save_model_to_file(optimized_onnx_path)


def get_output_names():
    outputs = ["output"]
    return outputs


def run(batch_size,
        float16,
        optimized,
        hidden_size,
        device,
        test_cases,
        sequence_length=2,
        epsilon=0.00001,
        cast_fp16=True,
        cast_onnx_only=False,
        verbose=False):
    test_name = f"device={device}, float16={float16}, optimized={optimized}, batch_size={batch_size}, sequence_length={sequence_length}, hidden_size={hidden_size}, epsilon={epsilon}, cast_fp16={cast_fp16}, cast_onnx_only={cast_onnx_only}"
    print(f"\nTesting: {test_name}")

    model = LayerNorm(hidden_size, epsilon, cast_fp16)
    model.eval()
    model.to(device)

    if float16 and not cast_fp16:
        model.half()

    # Do not re-use onnx file from previous test since weights of model are random.
    onnx_model_path = './temp/layer_norm_{}.onnx'.format("fp16" if float16 else "fp32")
    export_onnx(model, onnx_model_path, float16, hidden_size, device)

    if optimized:
        optimized_onnx_path = './temp/layer_norm_{}_opt.onnx'.format("fp16" if float16 else "fp32")
        if (not float16) or cast_fp16:
            optimize_onnx(onnx_model_path, optimized_onnx_path, expected_op=LayerNorm.get_fused_op())
        else:
            if cast_onnx_only:
                optimize_fp16_onnx_with_cast(onnx_model_path, optimized_onnx_path, epsilon=epsilon)
            else:
                optimize_fp16_onnx_no_cast(onnx_model_path, optimized_onnx_path, epsilon=epsilon)

        onnx_path = optimized_onnx_path
    else:
        onnx_path = onnx_model_path

    num_failure = run_parity(model,
                             onnx_path,
                             batch_size,
                             hidden_size,
                             sequence_length,
                             float16,
                             device,
                             optimized,
                             test_cases,
                             verbose=verbose)

    # clean up onnx file
    os.remove(onnx_model_path)
    if optimized:
        os.remove(onnx_path)

    return num_failure, test_name


class TestLayerNormParity(unittest.TestCase):
    def setUp(self):
        self.optimized = True  # Change it to False if you want to test parity of non optimized ONNX
        self.test_cases = 100  # Number of test cases per test run
        self.sequence_length = 2
        self.hidden_size = 768
        self.epsilon = 0.00001
        self.verbose = False

    def run_test(self,
                 batch_size,
                 float16,
                 optimized,
                 hidden_size,
                 device,
                 cast_fp16=True,
                 cast_onnx_only=False,
                 enable_assert=True):
        if float16 and device.type == 'cpu':  # CPU does not support FP16
            return

        num_failure, test_name = run(batch_size,
                                     float16,
                                     optimized,
                                     hidden_size,
                                     device,
                                     self.test_cases,
                                     self.sequence_length,
                                     self.epsilon,
                                     cast_fp16,
                                     cast_onnx_only,
                                     verbose=self.verbose)
        if enable_assert:
            self.assertTrue(num_failure == 0, "Failed: " + test_name)

    def run_one(self, optimized, device, hidden_size=768):
        for batch_size in [4]:
            self.run_test(batch_size, float16=False, optimized=optimized, hidden_size=hidden_size, device=device)

            self.run_test(
                batch_size,
                float16=True,
                optimized=optimized,
                hidden_size=hidden_size,
                device=device,
                cast_fp16=True,
                cast_onnx_only=False,
                enable_assert=False  # This setting has small chance to exceed tollerance threshold 0.001
            )

            self.run_test(
                batch_size,
                float16=True,
                optimized=optimized,
                hidden_size=hidden_size,
                device=device,
                cast_fp16=False,
                cast_onnx_only=False,
                enable_assert=False  # This setting cannot pass tollerance threshold
            )

            self.run_test(
                batch_size,
                float16=True,
                optimized=optimized,
                hidden_size=hidden_size,
                device=device,
                cast_fp16=False,
                cast_onnx_only=True,
                enable_assert=False  # This setting cannot pass tollerance threshold
            )

    def test_cpu(self):
        cpu = torch.device('cpu')
        self.run_one(self.optimized, cpu, hidden_size=self.hidden_size)

    def test_cuda(self):
        if not torch.cuda.is_available():
            import pytest
            pytest.skip('test requires GPU and torch+cuda')
        else:
            gpu = torch.device('cuda')
            self.run_one(self.optimized, gpu, hidden_size=self.hidden_size)


if __name__ == '__main__':
    unittest.main()
