# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""This file is used to generate test data for MemoryAlleviation tests in
   onnxruntime/test/optimizer/memory_alleviation_test.cc.

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


class MaskedSoftmaxDropoutLinearTest(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout()
        # self._seq_length = seq_length
        # self.value_layer_fc = torch.nn.Parameter(torch.randn(batch_size * head, seq_length, 64, dtype=torch.float32))

    # input1 -  float16[labels_dim0,24,512,512]
    # mask - boolean[labels_dim0,1,512,512]
    # value_layer - float16[24*labels_dim0,512,64]
    def forward(self, input1, mask, value_layer):
        output = input1.where(mask, torch.tensor(float("-inf"), dtype=input1.dtype, device=input1.device))
        output = torch.softmax(output, -1)
        attention_probs = output.where(mask, torch.tensor(float(0.0), dtype=output.dtype, device=input1.device))
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.bmm(
            attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        )
        return context_layer


def generate_softmax_dropout_test_case():
    batch_size = 16
    head = 24
    seq_length = 512
    model = MaskedSoftmaxDropoutLinearTest().to(DEVICE)
    model = ORTModule(model, DebugOptions(save_onnx=True, onnx_prefix="recompute_dropout"))

    input1 = torch.randn(batch_size, head, seq_length, seq_length, device=DEVICE).requires_grad_(True)
    value_layer = torch.randn(batch_size * head, seq_length, 64, device=DEVICE).requires_grad_(True)

    # ORTModule shape infer generate wrong shapes for Where's two inputs when input shape of Where are not same,
    # which are both graph inputs.
    # so we use same shape for where as a workaround.
    # input_mask = torch.randint(0, seq_length, (batch_size,1,seq_length,seq_length), dtype=torch.long, device=device).to(torch.bool)
    input_mask = torch.randint(
        0, seq_length, (batch_size, head, seq_length, seq_length), dtype=torch.long, device=DEVICE
    ).to(torch.bool)

    # Make sure model runs without any exception
    prediction = model(input1, input_mask, value_layer)
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
    generate_softmax_dropout_test_case()
    generate_tile_test_case()


if __name__ == "__main__":
    main()
