# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""This file is used to generate test data for ort format model tests in
   orttraining/orttraining/test/training_api/core/training_capi_tests.cc."""

import onnx
import torch
import torch.nn as nn

from onnxruntime.training import artifacts


class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def model_export(pt_model, model_path, input_size):
    # Generate random input data
    input_data = torch.randn(32, input_size)
    torch.onnx.export(
        pt_model,
        input_data,
        model_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )


def main():
    # Set the dimensions for input, hidden, and output layers
    input_size = 10
    hidden_size = 20
    output_size = 5

    # Create an instance of the neural network
    pt_model = SimpleNet(input_size, hidden_size, output_size)

    train_model_path = "simplenet_training.onnx"
    model_export(pt_model, train_model_path, input_size)

    onnx_model = onnx.load(train_model_path)

    requires_grad = ["fc2.weight", "fc2.bias"]
    frozen_params = [param.name for param in onnx_model.graph.initializer if param.name not in requires_grad]

    # Generate the training artifacts.
    artifacts.generate_artifacts(
        onnx_model,
        requires_grad=requires_grad,
        frozen_params=frozen_params,
        loss=artifacts.LossType.CrossEntropyLoss,
        optimizer=artifacts.OptimType.AdamW,
        ort_format=True,
    )


if __name__ == "__main__":
    main()
