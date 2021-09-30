import os
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch
from onnxruntime.training import export_gradient_graph
from torch import nn


class NeuralNet(torch.nn.Module):
    r"""
    Simple example model.
    """

    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super(NeuralNet, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class GradientGraphBuilderTest(unittest.TestCase):
    def test_save(self):
        loss_fn = nn.CrossEntropyLoss()
        input_size = 10
        model = NeuralNet(input_size=input_size, hidden_size=5,
                          num_classes=2)
        directory_path = Path(os.path.dirname(__file__)).resolve()
        for config in [{
            'input_weights': True,
        }, {
            'input_weights': False,
        }]:
            input_weights = config['input_weights']
            intermediate_graph_path = directory_path / \
                'gradient_graph_builder_test_model.onnx'
            gradient_graph_path = directory_path/'gradient_graph_model.onnx'

            batch_size = 1
            example_input = torch.randn(
                batch_size, input_size, requires_grad=True)
            example_labels = torch.tensor([1])

            export_gradient_graph(
                model, loss_fn, example_input, example_labels, gradient_graph_path, intermediate_graph_path,
                input_weights=input_weights)

            onnx_model = onnx.load(str(gradient_graph_path))
            onnx.checker.check_model(onnx_model)

            torch_out = model(example_input)
            ort_session = onnxruntime.InferenceSession(
                str(gradient_graph_path))
            inputs = ort_session.get_inputs()
            ort_inputs = {
                inputs[0].name: to_numpy(example_input),
                inputs[1].name: to_numpy(example_labels),
            }

            if input_weights:
                # TODO Update ort_inputs
                pass

            ort_outs = ort_session.run(None, ort_inputs)
            onnx_output_names = [node.name for node in onnx_model.graph.output]
            onnx_name_to_output = dict(zip(onnx_output_names, ort_outs))
            self.assertEqual(6, len(onnx_name_to_output))

            ort_output = onnx_name_to_output['output']
            np.testing.assert_allclose(
                to_numpy(torch_out), ort_output, rtol=1e-03, atol=1e-05)

            torch_loss = loss_fn(torch_out, example_labels)
            ort_loss = onnx_name_to_output['loss']
            np.testing.assert_allclose(
                to_numpy(torch_loss), ort_loss, rtol=1e-03, atol=1e-05)

            # Make sure the gradients have the right shape.
            model_param_names = tuple(
                name for name, _ in model.named_parameters())
            self.assertEqual(4, len(model_param_names))

            for name, param in model.named_parameters():
                grad = onnx_name_to_output['model.' + name + '_grad']
                self.assertEqual(param.size(), grad.shape)


if __name__ == '__main__':
    unittest.main()
