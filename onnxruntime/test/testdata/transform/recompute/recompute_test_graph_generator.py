# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""This file is used to generate test data for MemoryOptimizer tests in
onnxruntime/test/optimizer/memory_optimizer_test.cc.

Be noticed, after run this script, manually rename recompute_XXXX_execution_model_training.onnx to
recompute_XXXX.onnx
"""

import torch

from onnxruntime.training.ortmodule import DebugOptions, ORTModule


class LinearGeluLinearTest(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input1):
        out = self.fc1(input1)
        out = torch.nn.functional.gelu(out)
        out = self.fc2(out)
        return out


DEVICE = "cuda"


def generate_gelu_test_case():
    batch_size, dimension_in, hidden_size, dimension_out = 64, 784, 500, 10
    model = LinearGeluLinearTest(dimension_in, hidden_size, dimension_out).to(DEVICE)
    ort_model = ORTModule(model, DebugOptions(save_onnx=True, onnx_prefix="recompute_gelu"))

    input = torch.randn(batch_size, dimension_in, device=DEVICE)
    # Make sure model runs without any exception
    prediction = ort_model(input)
    assert prediction is not None
    prediction = prediction.sum()
    prediction.backward()


class TileTransposeLinearTest(torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        self._head = head

    # input1 -  float16[24,512,64]
    # repeat -  float16[4]
    # query_layer - float16[24*labels_dim0,512,64]
    def forward(self, input1, query_layer):
        # Tile to [24*labels_dim0,512,64]
        output = input1.repeat(query_layer.size(0) // self._head, 1, 1)

        # Transpose to [24*labels_dim0,64,512]
        output = output.permute(0, 2, 1).contiguous()
        return torch.matmul(query_layer, output)


def generate_tile_test_case():
    batch_size = 16
    head = 24
    seq_length = 512
    model = TileTransposeLinearTest(head).to(DEVICE)
    model = ORTModule(model, DebugOptions(save_onnx=True, onnx_prefix="recompute_tile"))

    input1 = torch.randn(head, seq_length, 64, device=DEVICE).requires_grad_(True)
    query_layer = torch.randn(batch_size * head, seq_length, 64, device=DEVICE).requires_grad_(True)

    # Make sure model runs without any exception
    prediction = model(input1, query_layer)
    assert prediction is not None
    prediction = prediction.sum()

    prediction.backward()


def main():
    """Main entry."""
    generate_gelu_test_case()
    generate_tile_test_case()


if __name__ == "__main__":
    main()
