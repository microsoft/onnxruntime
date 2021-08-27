import os
import unittest
from pathlib import Path

import torch
from torch import nn

from onnxruntime.training import GradientGraphBuilder


class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, loss_fn):
        super(NeuralNet, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)
        self.loss_fn = loss_fn

    def forward(self, input1, label):
        out = self.fc1(input1)
        out = self.relu(out)
        out = self.fc2(out)
        loss = self.loss_fn(out, label)
        return out, loss


class GradientGraphBuilderTest(unittest.TestCase):
    def test_save(self):
        loss_fn = nn.CrossEntropyLoss()
        model = NeuralNet(input_size=10, hidden_size=5, num_classes=2, loss_fn=loss_fn)
        directory_path = Path(os.path.dirname(__file__)).resolve()
        path = directory_path / 'model.onnx'
        batch_size = 1
        x = torch.randn(batch_size, model.fc1.in_features, requires_grad=True)
        label = torch.tensor([1])
        torch.onnx.export(
            model, (x, label), str(path),
            export_params=True,
            opset_version=12, do_constant_folding=True,
            input_names=['input'],
            output_names=['output', 'loss'],
            dynamic_axes={
                'input': {0: 'batch_size', },
                'output': {0: 'batch_size', },
            })
        builder = GradientGraphBuilder(str(path), {'output'}, {'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'}, 'loss')
        builder.build()
        # TODO Maybe it should be .ort?
        gradient_graph_path = directory_path/'gradient_graph_model.onnx'
        builder.save(str(gradient_graph_path))


if __name__ == '__main__':
    unittest.main()
