import os
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from onnxruntime.capi import _pybind_state as torch_ort_eager
from onnxruntime.training import ORTModule


def my_loss(x, target):
    return F.nll_loss(F.log_softmax(x, dim=1), target)


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class NoOpNet(torch.nn.Module):
    def __init__(self):
        super(NoOpNet, self).__init__()
        self.dummy_weight = torch.nn.Parameter(torch.ones(128, 128, dtype=torch.float16))

    def forward(self, input):
        return input


class OrtModuleEagerTest(unittest.TestCase):
    def test_half_type(self):
        model = NoOpNet()
        device = torch.device("ort")
        model.to(device)
        model = ORTModule(model)
        input = torch.ones(2, 2).to(torch.float16)
        y = model(input.to(device))
        assert y.dtype == torch.float16

    def test_ortmodule_inference(self):
        input_size = 784
        hidden_size = 500
        num_classes = 10
        batch_size = 128
        model = NeuralNet(input_size, hidden_size, num_classes)
        device = torch.device("ort")
        model.to(device)
        model = ORTModule(model)

        with torch.no_grad():
            data = torch.rand(batch_size, input_size)
            y = model(data.to(device))
        print("Done")

    def test_ort_module_and_eager_mode(self):
        input_size = 784
        hidden_size = 500
        num_classes = 10
        batch_size = 128
        model = NeuralNet(input_size, hidden_size, num_classes)
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        data = torch.rand(batch_size, input_size)
        target = torch.randint(0, 10, (batch_size,))
        # save the initial state
        initial_state = model.state_dict()
        # run on cpu first
        x = model(data)
        loss = my_loss(x, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # record the updated parameters
        cpu_updated_state = model.state_dict()
        # reload initial state
        model.load_state_dict(initial_state)
        # run on ort with ORTModule and eager mode
        # use device_idx 1 to test non-zero device
        torch_ort_eager.set_device(1, "CPUExecutionProvider", {"dummy": "dummy"})
        device = torch.device("ort", index=0)
        model.to(device)
        model = ORTModule(model)
        ort_optimizer = optim.SGD(model.parameters(), lr=0.01)
        x = model(data.to(device))
        loss = my_loss(x.cpu(), target)
        loss.backward()
        ort_optimizer.step()
        ort_optimizer.zero_grad()

        ort_updated_state = model.state_dict()
        # compare the updated state
        for state_tensor in cpu_updated_state:
            assert state_tensor in ort_updated_state
            assert torch.allclose(cpu_updated_state[state_tensor], ort_updated_state[state_tensor].cpu(), atol=1e-3)


if __name__ == "__main__":
    unittest.main()
