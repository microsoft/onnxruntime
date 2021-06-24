import unittest

import torch
from onnxruntime.training import GradientGraphBuilder


class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input1):
        out = self.fc1(input1)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class GradientGraphBuilderTest(unittest.TestCase):
    def test_save(self):
        model = NeuralNet(input_size=10, hidden_size=5, num_classes=2)
        print(__file__)
        print(GradientGraphBuilder)
        # TODO Export the model, then try to pass that to `GradientGraphBuilder`.
        # Just make sure that this test runs.
        self.assertTrue(False)


if __name__ == '__main__':
    unittest.main()
