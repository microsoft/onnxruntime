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

    def forward(self, input1):
        out = self.fc1(input1)
        out = self.relu(out)
        out = self.fc2(out)
        # loss = self.loss_fn(out, label)
        return out#, loss


class GradientGraphBuilderTest(unittest.TestCase):
    def test_save(self):
        loss_fn = nn.CrossEntropyLoss()
        model = NeuralNet(input_size=10, hidden_size=5, num_classes=2, loss_fn=loss_fn)
        directory_path = Path(os.path.dirname(__file__)).resolve()
        path = directory_path / 'model.onnx'
        batch_size = 1
        x = torch.randn(batch_size, model.fc1.in_features, requires_grad=True)
        torch.onnx.export(
            model, x, str(path),
            export_params=True,
            opset_version=12, do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', },
                'output': {0: 'batch_size', },
            })
        # FIXME Make sure we're setting up the parameters properly.
        # Currently gets an error at `level_to_transformer_map_.find(level)`
        # whebc calling
        # `graph_transformation_mgr_.ApplyTransformers(*graph_, TransformerLevel::Level2, logger_)` in
        # `GradientGraphBuilder::Build`.
        # Program received signal SIGFPE, Arithmetic exception.
        # 0x00007fff6a288cd7 in std::__detail::_Mod_range_hashing::operator() (
        #     this=0x7fffffffb3c8, __num=2, __den=0)
        #     at /usr/include/c++/7/bits/hashtable_policy.h:448
        # 448         { return __num % __den; }
        # Seems like `GraphTransformerManager::Register` is not called.

        # There's also problems because loss_node_arg_name isn't getting set properly (even when it's not an empty string).
        # `GradientGraphBuilder::loss_node_arg_name_` becomes a very long string.
        builder = GradientGraphBuilder(str(path), {'output'}, {'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'}, 'loss')
        builder.build()
        # TODO Maybe it should be .ort?
        gradient_graph_path = directory_path/'gradient_graph_model.onnx'
        builder.save(str(gradient_graph_path))


if __name__ == '__main__':
    unittest.main()
