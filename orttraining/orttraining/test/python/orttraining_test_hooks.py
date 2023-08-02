# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import tempfile

import pytest
import torch

from onnxruntime.training.ortmodule import ORTModule
from onnxruntime.training.utils.hooks import GlobalSubscriberManager, StatisticsSubscriber, _InspectActivation


class NeuralNetSingleOutput(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input1, input2):
        model_input = input1 + input2
        out = self.fc1(model_input)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class NeuralNetMultipleOutputs(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input1, input2):
        model_input = input1 + input2
        out = self.fc1(model_input)
        out = self.relu(out)
        out = self.fc2(out)
        return out, model_input


@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("backend", ["torch"])
def test_statistic_subscriber_single_output(device, backend):
    input_size = 8
    hidden_size = 16
    num_classes = 32
    model = NeuralNetSingleOutput(input_size, hidden_size, num_classes)
    model.to(device)
    model.train()

    with tempfile.TemporaryDirectory() as temporary_dir:
        output_dir_path = os.path.join(temporary_dir, f"{backend}_out")
        GlobalSubscriberManager.subscribe(model, [StatisticsSubscriber(output_dir_path, override_output_dir=True)])
        from onnxruntime.training.ortmodule import DebugOptions, LogLevel, ORTModule

        if backend == "ortmodule":
            model = ORTModule(model, DebugOptions(save_onnx=True, log_level=LogLevel.VERBOSE, onnx_prefix="model_name"))

        batch_size = 4
        input1_tensor = torch.randn(batch_size, input_size, device=device)
        input2_tensor = torch.randn(batch_size, input_size, device=device)
        for _ in range(5):
            y = model(input1_tensor, input2_tensor)
            y.sum().backward()

        assert os.path.exists(output_dir_path)

        # for i in range(5):
        #     step_dir = os.path.join(output_dir_path, f"step_{i}")
        #     for file in expected_files:
        #         assert os.path.exists(os.path.join(step_dir, file))

        from os import walk

        for dirpath, dirnames, _filenames in walk(output_dir_path):
            print("@@@@@@@", dirnames)
            for dirname in dirnames:
                for dirpath2, dirnames2, filenames2 in walk(os.path.join(dirpath, dirname)):
                    print(">>>>>>>>>>", dirpath2, dirnames2, filenames2)

        # for i in range(1):
        #     step_dir = os.path.join(output_dir_path, f"step_{i}")
        #     for file in expected_files:
        #         assert os.path.exists(os.path.join(step_dir, file))


test_statistic_subscriber_single_output("cuda", "torch")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", ["torch", "ortmodule"])
def test_statistic_subscriber_multiple_outputs(device, backend):
    input_size = 8
    hidden_size = 16
    num_classes = 32
    model = NeuralNetMultipleOutputs(input_size, hidden_size, num_classes)
    model.to(device)
    model.train()

    with tempfile.TemporaryDirectory() as temporary_dir:
        output_dir_path = os.path.join(temporary_dir, f"{backend}_out")
        GlobalSubscriberManager.subscribe(model, [StatisticsSubscriber(output_dir_path, override_output_dir=True)])

        if backend == "ortmodule":
            model = ORTModule(model)

        batch_size = 4
        input1_tensor = torch.randn(batch_size, input_size, device=device).requires_grad_(True)
        input2_tensor = torch.randn(batch_size, input_size, device=device)
        for _ in range(5):
            y_output1, y_output2 = model(input1_tensor, input2_tensor)
            y = y_output1.sum() + y_output2.sum()
            y.backward()

        assert os.path.exists(output_dir_path)

        expected_files = [
            "order.txt",
            "NeuralNetMultipleOutputs_0_0th_output_forward",
            "NeuralNetMultipleOutputs_0_0th_output_backward",
            "NeuralNetMultipleOutputs_0_1th_output_forward",
            "NeuralNetMultipleOutputs_0_1th_output_backward",
            "Linear_1_0th_output_forward",
            "Linear_1_0th_output_backward",
            "ReLU_2_0th_output_forward",
            "ReLU_2_0th_output_backward",
            "Linear_3_0th_output_forward",
            "Linear_3_0th_output_backward",
        ]

        for i in range(5):
            step_dir = os.path.join(output_dir_path, f"step_{i}")

            assert len(os.listdir(step_dir)) == len(expected_files)
            for file in expected_files:
                assert os.path.exists(os.path.join(step_dir, file))


class NeuralNetUserAnnotateIntermediateTensor(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input1, input2):
        model_input = input1 + input2
        out = self.fc1(model_input)
        out = _InspectActivation.apply("fc1_out", None, GlobalSubscriberManager.get_run_context(), out)
        out = self.relu(out)
        out = _InspectActivation.apply("relu_out", None, GlobalSubscriberManager.get_run_context(), out)
        out = self.fc2(out)
        return out


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", ["torch", "ortmodule"])
def test_statistic_subscriber_user_annotate_intermediate_tensors(device, backend):
    input_size = 8
    hidden_size = 16
    num_classes = 32
    model = NeuralNetUserAnnotateIntermediateTensor(input_size, hidden_size, num_classes)
    model.to(device)
    model.train()

    with tempfile.TemporaryDirectory() as temporary_dir:
        output_dir_path = os.path.join(temporary_dir, f"{backend}_out")
        GlobalSubscriberManager.subscribe(model, [StatisticsSubscriber(output_dir_path, override_output_dir=True)])

        if backend == "ortmodule":
            model = ORTModule(model)

        batch_size = 4
        input1_tensor = torch.randn(batch_size, input_size, device=device)
        input2_tensor = torch.randn(batch_size, input_size, device=device)
        for _ in range(5):
            y = model(input1_tensor, input2_tensor)
            y.sum().backward()

        assert os.path.exists(output_dir_path)

        expected_files = [
            "order.txt",
            "Linear_1_0th_output_forward",
            "Linear_1_0th_output_backward",
            "NeuralNetUserAnnotateIntermediateTensor_0_0th_output_forward",
            "NeuralNetUserAnnotateIntermediateTensor_0_0th_output_backward",
            "ReLU_2_0th_output_forward",
            "ReLU_2_0th_output_backward",
            "Linear_3_0th_output_forward",
            "Linear_3_0th_output_backward",
            "fc1_out_forward",
            "fc1_out_backward",
            "relu_out_forward",
            "relu_out_backward",
        ]

        for i in range(5):
            step_dir = os.path.join(output_dir_path, f"step_{i}")
            for file in expected_files:
                assert os.path.exists(os.path.join(step_dir, file))
