import os
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch
from onnxruntime.training import export_gradient_graph
from torch import nn
from torch.onnx import TrainingMode


class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input1, label):
        out = self.fc1(input1)
        out = self.relu(out)
        out = self.fc2(out)
        loss = self.loss_fn(out, label)
        return out, loss


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class GradientGraphBuilderTest(unittest.TestCase):
    def test_save(self):
        loss_fn = nn.CrossEntropyLoss()
        model = NeuralNet(input_size=10, hidden_size=5,
                          num_classes=2)
        model.train()
        directory_path = Path(os.path.dirname(__file__)).resolve()
        intermediate_path = directory_path / 'gradient_graph_builder_test_model.onnx'
        gradient_graph_path = directory_path/'gradient_graph_model.onnx'
        batch_size = 1
        export_gradient_graph(model, loss_fn, batch_size, gradient_graph_path, intermediate_path)
        x = torch.randn(batch_size, model.fc1.in_features, requires_grad=True)
        label = torch.tensor([1])
        torch_out, torch_loss = model(x, label)

        torch.onnx.export(
            model, (x, label), str(intermediate_path),
            export_params=True,
            opset_version=12, do_constant_folding=False,
            training=TrainingMode.TRAINING,
            input_names=['input', 'label'],
            output_names=['output', 'loss'],
            dynamic_axes={
                'input': {0: 'batch_size', },
                'label': {0: 'batch_size', },
                'output': {0: 'batch_size', },
            })
        builder = GradientGraphBuilder(str(intermediate_path),
                                       {'loss'},
                                       {'fc1.weight', 'fc1.bias',
                                           'fc2.weight', 'fc2.bias'},
                                       'loss')
        builder.build()
        builder.save(str(gradient_graph_path))

        onnx_model = onnx.load(str(gradient_graph_path))
        onnx.checker.check_model(onnx_model)

        ort_session = onnxruntime.InferenceSession(
            str(gradient_graph_path))
        inputs = ort_session.get_inputs()
        ort_inputs = {
            inputs[0].name: to_numpy(x),
            inputs[1].name: to_numpy(label),
        }
        ort_outs = ort_session.run(None, ort_inputs)
        onnx_output_names = [node.name for node in onnx_model.graph.output]
        onnx_name_to_output = dict(zip(onnx_output_names, ort_outs))
        self.assertEqual(6, len(onnx_name_to_output))

        ort_output = onnx_name_to_output['output']
        np.testing.assert_allclose(
            to_numpy(torch_out), ort_output, rtol=1e-03, atol=1e-05)

        ort_loss = onnx_name_to_output['loss']
        np.testing.assert_allclose(
            to_numpy(torch_loss), ort_loss, rtol=1e-03, atol=1e-05)

        # Make sure the gradients have the right shape.
        for name, param in model.named_parameters():
            grad = onnx_name_to_output[name + '_grad']
            self.assertEqual(param.size(), grad.shape)


if __name__ == '__main__':
    unittest.main()
