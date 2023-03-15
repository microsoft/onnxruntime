# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import pytest
import tempfile
import torch

from onnxruntime.training.utils.hooks import SubscriberManager, StatisticsSubscriber


class NeuralNetMultiplePositionalArgumentsVarKeyword(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetMultiplePositionalArgumentsVarKeyword, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input1, input2, **kwargs):
        model_input = input1 + input2
        out = self.fc1(model_input)
        out = self.relu(out)
        out = self.fc2(out)
        return out


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", ["torch", "ortmodule"])
def test_statistic_subscriber(device, backend):
    input_size = 8
    hidden_size = 16
    num_classes = 32
    model = NeuralNetMultiplePositionalArgumentsVarKeyword(input_size, hidden_size, num_classes)
    model.to(device)
    model.train()

    with tempfile.TemporaryDirectory() as temporary_dir:
        output_dir_path = os.path.join(temporary_dir, f"{backend}_out")
        if backend == "ortmodule":
            from onnxruntime.training.ortmodule import ORTModule

            model = ORTModule(model)

        SubscriberManager.subscribe(model, [StatisticsSubscriber(output_dir_path, override_output_dir=True)])

        batch_size = 4
        input1_tensor = torch.randn(batch_size, input_size, device=device)
        input2_tensor = torch.randn(batch_size, input_size, device=device)
        for _ in range(5):
            y = model(input1_tensor, input2_tensor)
            y.sum().backward()

        assert os.path.exists(output_dir_path)

        expected_files = [
            "order.txt",
            "Linear_1_0th_output forward run",
            "Linear_1_0th_output backward run",
            "NeuralNetMultiplePositionalArgumentsVarKeyword_0_0th_output forward run",
            "NeuralNetMultiplePositionalArgumentsVarKeyword_0_0th_output backward run",
            "ReLU_2_0th_output forward run",
            "ReLU_2_0th_output backward run",
            "Linear_3_0th_output forward run",
            "Linear_3_0th_output backward run",
        ]

        for i in range(5):
            step_dir = os.path.join(output_dir_path, f"step_{i}")
            for file in expected_files:
                assert os.path.exists(os.path.join(step_dir, file))
