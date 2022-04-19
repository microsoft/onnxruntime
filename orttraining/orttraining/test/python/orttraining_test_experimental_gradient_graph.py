import os
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch
from onnxruntime.training.experimental import export_gradient_graph


class NeuralNet(torch.nn.Module):
    r"""
    Simple example model.
    """

    def __init__(self,
                 input_size: int,
                 embedding_size: int,
                 hidden_size: int,
                 num_classes: int):
        super(NeuralNet, self).__init__()

        self.frozen_layer = torch.nn.Linear(
            input_size, embedding_size, bias=False)
        # Freeze a layer (mainly to test that gradients don't get output for it).
        self.frozen_layer.requires_grad_(False)

        self.fc1 = torch.nn.Linear(embedding_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.frozen_layer(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def binary_cross_entropy_loss(inp, target):
    loss = -torch.sum(target * torch.log2(inp[:, 0]) +
                      (1-target) * torch.log2(inp[:, 1]))
    return loss


class GradientGraphBuilderTest(unittest.TestCase):
    def test_save(self):
        # We need a custom loss function to load the graph in an InferenceSession in ONNX Runtime Web.
        # You can still make the gradient graph with torch.nn.CrossEntropyLoss() and this test will pass.
        loss_fn = binary_cross_entropy_loss
        input_size = 10
        model = NeuralNet(input_size=input_size, embedding_size=20, hidden_size=5,
                          num_classes=2)
        directory_path = Path(os.path.dirname(__file__)).resolve()

        gradient_graph_path = directory_path/'gradient_graph_model.onnx'

        batch_size = 1
        example_input = torch.randn(
            batch_size, input_size, requires_grad=True)
        example_labels = torch.tensor([1])

        export_gradient_graph(
            model, loss_fn, example_input, example_labels, gradient_graph_path)

        onnx_model = onnx.load(str(gradient_graph_path))
        onnx.checker.check_model(onnx_model)

        # Expected inputs: input, labels, models parameters.
        self.assertEqual(
            1 + 1 + sum(1 for _ in model.parameters()), len(onnx_model.graph.input))
        
        # Expected outputs: prediction, loss, and parameters with gradients.
        self.assertEqual(
            1 + 1 + sum(1 if p.requires_grad else 0 for p in model.parameters()), len(onnx_model.graph.output))

        torch_out = model(example_input)

        try:
            ort_session = onnxruntime.InferenceSession(str(gradient_graph_path))
        except ValueError:
            # Sometimes it is required to pass the available providers.
            from onnxruntime.capi import _pybind_state as C
            available_providers = C.get_available_providers()
            ort_session = onnxruntime.InferenceSession(str(gradient_graph_path), providers=available_providers)

        ort_inputs = {
            onnx_model.graph.input[0].name: to_numpy(example_input),
            onnx_model.graph.input[1].name: to_numpy(example_labels),
        }

        for name, param in model.named_parameters():
            ort_inputs[name] = to_numpy(param.data)

        ort_outs = ort_session.run(None, ort_inputs)
        onnx_output_names = [node.name for node in onnx_model.graph.output]
        onnx_name_to_output = dict(zip(onnx_output_names, ort_outs))

        ort_output = onnx_name_to_output['output']
        np.testing.assert_allclose(
            to_numpy(torch_out), ort_output, rtol=1e-03, atol=1e-05)

        torch_loss = loss_fn(torch_out, example_labels)
        ort_loss = onnx_name_to_output['loss']
        np.testing.assert_allclose(
            to_numpy(torch_loss), ort_loss, rtol=1e-03, atol=1e-05)

        # Make sure the gradients have the right shape.
        model_param_names = tuple(
            name for name, param in model.named_parameters() if param.requires_grad)
        self.assertEqual(4, len(model_param_names))

        for name, param in model.named_parameters():
            if param.requires_grad:
                grad = onnx_name_to_output[name + '_grad']
                self.assertEqual(param.size(), grad.shape)


if __name__ == '__main__':
    unittest.main()
