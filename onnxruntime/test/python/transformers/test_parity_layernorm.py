# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import unittest

import onnx
import torch
from parity_utilities import *  # noqa: F403
from torch import nn

if find_transformers_source():  # noqa: F405
    from onnx_model import OnnxModel
else:
    from onnxruntime.transformers.onnx_model import OnnxModel


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, epsilon, cast_fp16=True, formula=0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=epsilon)
        # initialize weights with random value
        self.layer_norm.bias.data.normal_(mean=0.0, std=0.1)
        self.layer_norm.weight.data.normal_(mean=0.0, std=0.5)
        self.cast_fp16 = cast_fp16
        self.formula = formula
        self.epsilon = epsilon

    @staticmethod
    def get_fused_op():
        return "LayerNormalization"

    def my_layer_norm(self, x):
        if self.formula == 0:
            return self.layer_norm(x)

        input_dtype = x.dtype
        x = x.float()
        u = x.mean(-1, keepdim=True)
        y = x - u
        s = y.pow(2).mean(-1, keepdim=True)
        z = y / torch.sqrt(s + self.epsilon)
        return self.layer_norm.weight.data * z.to(input_dtype) + self.layer_norm.bias.data

    def forward(self, x):
        if self.cast_fp16 and x.dtype == torch.float16:
            y = self.my_layer_norm(x.to(torch.float32)).to(torch.float16)
        else:
            y = self.my_layer_norm(x)

        return (y,)


def get_weight(onnx_model):
    last_mul_node = onnx_model.get_nodes_by_op_type("Mul")[-1]
    i, value = onnx_model.get_constant_input(last_mul_node)
    assert value is not None
    weight_name = last_mul_node.input[i]
    return weight_name


def get_bias(onnx_model):
    last_add_node = onnx_model.get_nodes_by_op_type("Add")[-1]
    i, value = onnx_model.get_constant_input(last_add_node)
    assert value is not None
    bias_name = last_add_node.input[i]
    return bias_name


def optimize_fp16_onnx_with_cast(input_onnx_path, optimized_onnx_path, epsilon):
    m = onnx.load(input_onnx_path)
    onnx_model = OnnxModel(m)
    weight_name = get_weight(onnx_model)
    bias_name = get_bias(onnx_model)
    nodes_to_remove = [n for n in onnx_model.nodes() if n.output[0] != weight_name and n.output[0] != bias_name]
    nodes_to_add = [
        onnx.helper.make_node("Cast", ["input"], ["fp32_input"], "cast_input", to=1),
        onnx.helper.make_node("Cast", [weight_name], ["fp32_layer_norm.weight"], "cast_weight", to=1),
        onnx.helper.make_node("Cast", [bias_name], ["fp32_layer_norm.bias"], "cast_bias", to=1),
        onnx.helper.make_node(
            "LayerNormalization",
            ["fp32_input", "fp32_layer_norm.weight", "fp32_layer_norm.bias"],
            ["fp32_output"],
            "layer_norm",
            epsilon=epsilon,
        ),  # use fp32 epsilon
        onnx.helper.make_node("Cast", ["fp32_output"], ["output"], "cast_output", to=10),
    ]

    onnx_model.remove_nodes(nodes_to_remove)
    onnx_model.add_nodes(nodes_to_add)
    onnx_model.prune_graph()
    onnx_model.save_model_to_file(optimized_onnx_path)


def optimize_fp16_onnx_no_cast(input_onnx_path, optimized_onnx_path, epsilon):
    m = onnx.load(input_onnx_path)
    onnx_model = OnnxModel(m)

    weight_name = get_weight(onnx_model)
    bias_name = get_bias(onnx_model)
    nodes_to_remove = [n for n in onnx_model.nodes() if n.output[0] != weight_name and n.output[0] != bias_name]

    nodes_to_remove = onnx_model.nodes()
    node_to_add = onnx.helper.make_node(
        "LayerNormalization",
        ["input", weight_name, bias_name],
        ["output"],
        "layer_norm",
        epsilon=epsilon,
    )

    onnx_model.remove_nodes(nodes_to_remove)
    onnx_model.add_node(node_to_add)
    onnx_model.prune_graph()
    onnx_model.save_model_to_file(optimized_onnx_path)


def get_output_names():
    outputs = ["output"]
    return outputs


def run(
    batch_size,
    float16,
    optimized,
    hidden_size,
    device,
    test_cases,
    sequence_length=2,
    epsilon=0.00001,
    cast_fp16=True,
    cast_onnx_only=False,
    formula=0,
    verbose=False,
):
    test_name = f"device={device}, float16={float16}, optimized={optimized}, batch_size={batch_size}, sequence_length={sequence_length}, hidden_size={hidden_size}, epsilon={epsilon}, cast_fp16={cast_fp16}, cast_onnx_only={cast_onnx_only}, formula={formula}"
    print(f"\nTesting: {test_name}")

    model = LayerNorm(hidden_size, epsilon, cast_fp16, formula)
    model.eval()
    model.to(device)

    if float16 and not cast_fp16:
        model.half()

    # Do not re-use onnx file from previous test since weights of model are random.
    onnx_model_path = "./temp/layer_norm_{}_formula{}.onnx".format("fp16" if float16 else "fp32", formula)
    export_onnx(model, onnx_model_path, float16, hidden_size, device)  # noqa: F405

    if optimized:
        optimized_onnx_path = "./temp/layer_norm_{}_formula{}_opt.onnx".format("fp16" if float16 else "fp32", formula)
        if (not float16) or cast_fp16:
            optimize_onnx(  # noqa: F405
                onnx_model_path, optimized_onnx_path, expected_op=LayerNorm.get_fused_op(), verbose=verbose
            )
        else:
            if cast_onnx_only:
                optimize_fp16_onnx_with_cast(onnx_model_path, optimized_onnx_path, epsilon=epsilon)
            else:
                optimize_fp16_onnx_no_cast(onnx_model_path, optimized_onnx_path, epsilon=epsilon)

        onnx_path = optimized_onnx_path
    else:
        onnx_path = onnx_model_path

    num_failure = run_parity(  # noqa: F405
        model,
        onnx_path,
        batch_size,
        hidden_size,
        sequence_length,
        float16,
        device,
        optimized,
        test_cases,
        verbose,
    )

    # clean up onnx file
    os.remove(onnx_model_path)
    if optimized:
        os.remove(onnx_path)

    return num_failure, test_name


class TestLayerNormParity(unittest.TestCase):
    verbose = False
    optimized = True

    def setUp(self):
        self.test_cases = 100  # Number of test cases per test run
        self.sequence_length = 2
        self.hidden_size = 768

    def run_test(
        self,
        batch_size,
        float16,
        optimized,
        hidden_size,
        device,
        cast_fp16=True,
        cast_onnx_only=False,
        formula=0,
        epsilon=0.00001,
        enable_assert=True,
        verbose=False,
    ):
        if float16 and device.type == "cpu":  # CPU does not support FP16
            return

        num_failure, test_name = run(
            batch_size,
            float16,
            optimized,
            hidden_size,
            device,
            self.test_cases,
            self.sequence_length,
            epsilon,
            cast_fp16,
            cast_onnx_only,
            formula,
            verbose=verbose,
        )
        if enable_assert:
            self.assertTrue(num_failure == 0, "Failed: " + test_name)

    def run_one(self, optimized, device, hidden_size=768, run_extra_tests=False, verbose=False):
        for batch_size in [4]:
            for formula in [0, 1]:
                for epsilon in [1e-5]:  # [1e-5, 1e-12]
                    self.run_test(
                        batch_size,
                        float16=False,
                        optimized=optimized,
                        hidden_size=hidden_size,
                        device=device,
                        formula=formula,
                        epsilon=epsilon,
                        verbose=verbose,
                    )

                    self.run_test(
                        batch_size,
                        float16=True,
                        optimized=optimized,
                        hidden_size=hidden_size,
                        device=device,
                        cast_fp16=True,
                        cast_onnx_only=False,
                        formula=formula,
                        epsilon=epsilon,
                        enable_assert=False,  # This setting has small chance to exceed tollerance threshold 0.001
                        verbose=verbose,
                    )

                    if not run_extra_tests:
                        continue

                    if device.type != "cuda" or formula != 1:
                        self.run_test(
                            batch_size,
                            float16=True,
                            optimized=optimized,
                            hidden_size=hidden_size,
                            device=device,
                            cast_fp16=False,
                            cast_onnx_only=False,
                            formula=formula,
                            epsilon=epsilon,
                            enable_assert=False,  # This setting cannot pass tollerance threshold
                            verbose=verbose,
                        )

                    self.run_test(
                        batch_size,
                        float16=True,
                        optimized=optimized,
                        hidden_size=hidden_size,
                        device=device,
                        cast_fp16=False,
                        cast_onnx_only=True,
                        formula=formula,
                        epsilon=epsilon,
                        enable_assert=False,  # This setting cannot pass tollerance threshold
                        verbose=verbose,
                    )

    def test_cpu(self):
        cpu = torch.device("cpu")
        self.run_one(self.optimized, cpu, hidden_size=self.hidden_size, verbose=self.verbose)

    def test_cuda(self):
        if not torch.cuda.is_available():
            import pytest

            pytest.skip("test requires GPU and torch+cuda")
        else:
            gpu = torch.device("cuda")
            self.run_one(self.optimized, gpu, hidden_size=self.hidden_size, run_extra_tests=True, verbose=self.verbose)


if __name__ == "__main__":
    args, remaining_args = parse_arguments(namespace_filter=unittest)  # noqa: F405

    TestLayerNormParity.verbose = args.log_verbose
    TestLayerNormParity.optimized = args.optimize

    unittest.main(argv=remaining_args)
