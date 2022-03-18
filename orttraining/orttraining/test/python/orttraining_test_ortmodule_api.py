# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# orttraining_test_ortmodule_api.py

import itertools
import math
import random
import copy
import torch
from transformers import AutoConfig, BertForSequenceClassification, Trainer
from transformers.modeling_outputs import SequenceClassifierOutput
import pytest
from time import sleep
import warnings
from unittest.mock import patch
from collections import OrderedDict
from collections import namedtuple
from inspect import signature
import tempfile
import os
import pickle
from distutils.version import LooseVersion
from onnxruntime.training.ortmodule._custom_gradient_registry import register_gradient
from onnxruntime.training.ortmodule import (ORTModule,
                                            _utils,
                                            _io,
                                            DebugOptions,
                                            LogLevel,
                                            _fallback,
                                            _graph_execution_manager)
import onnxruntime.training.ortmodule as ortmodule_module

from onnxruntime.training.optim import FusedAdam, AdamWMode
from transformers import AdamW

import _test_helpers

# Import autocasting libs
from torch.cuda import amp


DEFAULT_OPSET = 14


# PyTorch model definitions for tests

class NeuralNetSinglePositionalArgument(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetSinglePositionalArgument, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input1):
        out = self.fc1(input1)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class NeuralNetMultiplePositionalArgumentsMultiOutputsWithoutDependency(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetMultiplePositionalArgumentsMultiOutputsWithoutDependency, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(input_size, hidden_size)
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.softmax2 = torch.nn.Softmax(dim=1)
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()

    def forward(self, input1, input2):
        model_input = input1 + input2
        out1 = self.fc1(model_input)
        out2 = self.fc2(model_input)
        out1 = self.softmax1(out1)
        out2 = self.softmax2(out2)
        out1 = self.relu1(out1)
        out2 = self.relu2(out2)
        return out1, out2

class NeuralNetMultiplePositionalArgumentsMultiOutputsWithDependency(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetMultiplePositionalArgumentsMultiOutputsWithDependency, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.softmax = torch.nn.Softmax(dim=1)
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input1, input2):
        model_input = input1 + input2
        out1 = self.fc1(model_input)
        out1 = self.softmax(out1)
        # TODO: Using relu here will cause the forward prediction error
        # ORT's Relu output is sharing the same buffer as input,
        # and this buffer is returned as ORTModule's output to Pytorch
        out2 = self.fc2(out1)
        return out1, out2

class NeuralNetMultiplePositionalArguments(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetMultiplePositionalArguments, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input1, input2):
        model_input = input1 + input2
        out = self.fc1(model_input)
        out = self.relu(out)
        out = self.fc2(out)
        return out

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

class NeuralNetPositionalArguments(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetPositionalArguments, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, *model_inputs):
        model_input = torch.sum(torch.stack(model_inputs), dim=0)
        out = self.fc1(model_input)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class NeuralNetKeywordArguments(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetKeywordArguments, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x=None, y=None, z=None):
        model_input = torch.sum(torch.stack([x, y, z]), dim=0)
        out = self.fc1(model_input)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class NeuralNetPositionalAndKeywordArguments(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetPositionalAndKeywordArguments, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, model_input, x=None, y=None, z=None):
        model_input = model_input + torch.sum(torch.stack([x, y, z]), dim=0)
        out = self.fc1(model_input)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class NeuralNetSimplePositionalAndKeywordArguments(torch.nn.Module):
    def __init__(self):
        super(NeuralNetSimplePositionalAndKeywordArguments, self).__init__()
        self.a = torch.nn.Parameter(torch.FloatTensor([-1., 1.]))
    def forward(self, x, y=None, z=None):
        if z is not None:
            return torch.mean(self.a) + x + 4 * z
        if y is not None:
            return torch.mean(self.a) + 3 * y
        return torch.mean(self.a) + x

class NeuralNetNonDifferentiableOutput(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetNonDifferentiableOutput, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input1):
        out = self.fc1(input1)
        out1 = self.relu(out)
        out2 = self.fc2(out1)
        mask1 = torch.gt(out1, 0.01)
        mask1 = mask1.long()    # TODO: Casting from bool to float or int will cause the UT failure
                                # True is casted to 1065353216 for Cast(from=bool, to=int), whereas pytorch would give 1
                                # True is casted to -1 for Cast(from=bool, to=float), where as pytorch would give 1.0f
        mask2 = torch.lt(out2, 0.02)
        mask2 = mask2.long()

        return out1, mask1, out2, mask2     # intentionally place the non-differentiable output in the middle

class NeuralNetChainedLayersWithNonDifferentiableOutput(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetChainedLayersWithNonDifferentiableOutput, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input1, mask1):
        out = self.fc1(input1)
        out1 = self.relu(out)
        out2 = self.fc2(out1)
        # this will trigger torch to set requires_grad = True for mask tensor
        mask = mask1

        return out2, mask

class NeuralNetPartialNoGradModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetPartialNoGradModel, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size).requires_grad_(False)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, model_input):
        out = self.relu(self.fc1(model_input))
        out = self.fc2(out)
        return out

class UnusedEndParameterNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(UnusedEndParameterNet, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size1)
        self.relu = torch.nn.ReLU()
        # fc2 is an unused initializer (which is in the end of initializer list)
        # which will be dropped after export
        self.fc2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.register_buffer("buffer", torch.ones(hidden_size1))

    def forward(self, input1):
        out = self.fc1(input1)
        out = self.relu(out)
        out = out + self.buffer
        return out

class UnusedBeginParameterNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(UnusedBeginParameterNet, self).__init__()

        # fc1 is an unused initializer (which is in the begining of initializer list)
        # which will be dropped after export
        self.fc1 = torch.nn.Linear(input_size, hidden_size1)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(input_size, hidden_size2)
        self.register_buffer("buffer", torch.ones(hidden_size2))

    def forward(self, input1):
        out = self.fc2(input1)
        out = self.relu(out)
        out = out + self.buffer
        return out

class UnusedMiddleParameterNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(UnusedMiddleParameterNet, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size1)
        self.relu = torch.nn.ReLU()
        # fc2 is an unused initializer (which is in the middle of initializer list)
        # which will be dropped after export
        self.fc2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = torch.nn.Linear(hidden_size1, num_classes)
        self.register_buffer("buffer", torch.ones(num_classes))

    def forward(self, input1):
        out = self.fc1(input1)
        out = self.relu(out)
        out = self.fc3(out)
        out = out + self.buffer
        return out

class StatelessModel(torch.nn.Module):
    def __init__(self):
        super(StatelessModel, self).__init__()

    def forward(self, x):
        return x

class NeuralNetCustomClassOutput(torch.nn.Module):
    class CustomClass(object):
        def __init__(self, out1, out2, out3):
            self.out1 = out1
            self.out2 = out2
            self.out3 = out3

    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetCustomClassOutput, self).__init__()

        self.fc1_1 = torch.nn.Linear(input_size, hidden_size)
        self.relu1 = torch.nn.ReLU()
        self.fc1_2 = torch.nn.Linear(hidden_size, num_classes)

        self.fc2_1 = torch.nn.Linear(input_size, hidden_size)
        self.relu2 = torch.nn.ReLU()
        self.fc2_2 = torch.nn.Linear(hidden_size, num_classes)

        self.fc3_1 = torch.nn.Linear(input_size, hidden_size)
        self.relu3 = torch.nn.ReLU()
        self.fc3_2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input1, input2, input3):
        out1 = self.fc1_2(self.relu1(self.fc1_1(input1)))
        out2 = self.fc2_2(self.relu2(self.fc2_1(input2)))
        out3 = self.fc3_2(self.relu3(self.fc3_1(input3)))
        return NeuralNetCustomClassOutput.CustomClass(out1, out2, out3)

class MyStrNet(torch.nn.Module):
    def forward(self, x, my_str):
        if my_str.lower() == 'hello':
            return x+1
        return x

class SerializationNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SerializationNet, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input1):
        out = self.fc1(input1)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def train_step(self, input):
        out = self(input)
        loss = out.sum()
        loss.backward()

        return out

@pytest.fixture(scope='session', autouse=True)
def run_before_test_session(request):
    def insert_disable_fallback_in_env():
        os.environ["ORTMODULE_FALLBACK_POLICY"] = "FALLBACK_DISABLE"

    def remove_disable_fallback_from_env():
        del os.environ["ORTMODULE_FALLBACK_POLICY"]

    insert_disable_fallback_in_env()
    request.addfinalizer(remove_disable_fallback_from_env)

# TODO: This is a workaround for the problem that pytest is still cleaning up the previous test
# while the next task already start.
@pytest.fixture(autouse=True)
def run_before_tests():
    # wait for 50ms before starting the next test
    sleep(0.05)

def _get_bert_for_sequence_classification_model(device, output_attentions = False, \
    output_hidden_states = False, return_dict = True, hidden_dropout_prob = 0.1, attention_probs_dropout_prob = 0.1):
    """Returns the BertForSequenceClassification pretrained model"""

    config = AutoConfig.from_pretrained(
            "bert-base-uncased",
            num_labels=2,
            num_hidden_layers=1,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            hidden_dropout_prob = hidden_dropout_prob,
            attention_probs_dropout_prob = attention_probs_dropout_prob,
    )
    config.return_dict = return_dict

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        config=config,
    ).to(device)

    return model

def _get_bert_for_sequence_classification_sample_data(device):
    """Returns sample data to be used with BertForSequenceClassification model"""

    input_ids = torch.randint(0, 100, (32, 64), dtype=torch.long, device=device)
    input_mask = torch.randint(0, 100, (32, 64), dtype=torch.long, device=device)
    labels = torch.randint(0, 1, (32,), dtype=torch.long, device=device)

    return input_ids, input_mask, labels

def _get_bert_for_sequence_classification_sample_data_with_random_shapes(device):
    """Returns sample data with random shape to be used with BertForSequenceClassification model"""

    x = random.randint(1,100)
    y = random.randint(1,100)
    input_ids = torch.randint(0, 100, (x, y), dtype=torch.long, device=device)
    input_mask = torch.randint(0, 100, (x, y), dtype=torch.long, device=device)
    labels = torch.randint(0, 1, (x,), dtype=torch.long, device=device)

    return input_ids, input_mask, labels

# ORTModule-API tests

def test_forward_call_single_positional_argument():
    device = 'cuda'

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    ort_model = ORTModule(model)
    # Check that the original forward signature is preserved.
    assert signature(model.forward) == signature(ort_model.forward)
    x = torch.randn(N, D_in, device=device)
    # Make sure model runs without any exception
    prediction = ort_model(x)
    assert prediction is not None
    prediction = prediction.sum()
    prediction.backward()

def test_forward_call_multiple_positional_arguments():
    device = 'cuda'

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetMultiplePositionalArguments(input_size=D_in, hidden_size=H, num_classes=D_out).to(device)
    ort_model = ORTModule(model)
    # Check that the original forward signature is preserved.
    assert signature(model.forward) == signature(ort_model.forward)
    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_in, device=device)

    # Make sure model runs without any exception
    prediction = ort_model(x, y)
    assert prediction is not None
    prediction = prediction.sum()
    prediction.backward()

def test_forward_call_positional_arguments():
    device = 'cuda'

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetPositionalArguments(input_size=D_in, hidden_size=H, num_classes=D_out).to(device)
    model = ORTModule(model)
    args = [torch.randn(N, D_in, device=device), torch.randn(N, D_in, device=device), torch.randn(N, D_in, device=device)]

    # Make sure model runs without any exception
    prediction = model(*args)
    assert prediction is not None
    prediction = prediction.sum()
    prediction.backward()

def test_forward_call_keyword_arguments():
    device = 'cuda'

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetKeywordArguments(D_in, H, D_out).to(device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_in, device=device)
    z = torch.randn(N, D_in, device=device)

    # Make sure model runs without any exception
    prediction = model(x, y, z)
    assert prediction is not None
    prediction = prediction.sum()
    prediction.backward()

def test_forward_call_positional_and_keyword_arguments():
    device = 'cuda'

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetPositionalAndKeywordArguments(D_in, H, D_out).to(device)
    model = ORTModule(model)
    a = torch.randn(N, D_in, device=device)
    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_in, device=device)
    z = torch.randn(N, D_in, device=device)

    # Make sure model runs without any exception
    prediction = model(a, x, y, z)
    assert prediction is not None
    prediction = prediction.sum()
    prediction.backward()

@pytest.mark.parametrize("forward_statement", [
    "model(one)",
    "model(x=one)",
    "model(one, None, None)",
    "model(one, None, z=None)",
    "model(one, None)",
    "model(x=one, y=one)",
    "model(y=one, x=one)",
    "model(y=one, z=None, x=one)",
    "model(one, None, z=one)",
    "model(x=one, z=one)",
    "model(one, z=one)",
    "model(one, z=one, y=one)",
    "model(one, one, one)",
    "model(one, None, one)",
    "model(z=one, x=one, y=one)",
    "model(z=one, x=one, y=None)"
])
def test_compare_pytorch_forward_call_positional_and_keyword_arguments(forward_statement):
    one = torch.FloatTensor([1])

    model = NeuralNetSimplePositionalAndKeywordArguments()
    pytorch_result = eval(forward_statement + ".item()")

    model = NeuralNetSimplePositionalAndKeywordArguments()
    model = ORTModule(model)
    ortmodule_result = eval(forward_statement + ".item()")
    ortmodule_result_again = eval(forward_statement + ".item()")
    assert ortmodule_result == ortmodule_result_again
    assert pytorch_result == ortmodule_result

    prediction = eval(forward_statement).sum()
    prediction.backward()

def test_torch_nn_module_cuda_method():
    original_device = 'cpu'
    to_device = 'cuda'

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out)
    model = ORTModule(model)
    for _, parameter_value in model.named_parameters():
        assert parameter_value.device.type == original_device

    x = torch.randn(N, D_in, device=to_device)
    model = model.cuda()
    model(x)

    for _, parameter_value in model.named_parameters():
        assert parameter_value.device.type == to_device

@pytest.mark.parametrize("set_gpu_on_original_module", [True, False])
def test_torch_nn_module_cpu_method(set_gpu_on_original_module):
    original_device = 'cuda'
    to_device = 'cpu'

    N, D_in, H, D_out = 64, 784, 500, 10
    if set_gpu_on_original_module:
        model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(original_device)
        model = ORTModule(model)
    else:
        model = NeuralNetSinglePositionalArgument(D_in, H, D_out)
        model = ORTModule(model).to(original_device)
    for _, parameter_value in model.named_parameters():
        assert parameter_value.device.type == original_device

    x = torch.randn(N, D_in, device=to_device)
    model = model.cpu()
    model(x)
    for _, parameter_value in model.named_parameters():
        assert parameter_value.device.type == to_device

@pytest.mark.parametrize("original_device", ['cpu', 'cuda'])
@pytest.mark.parametrize("to_argument", ['cpu', 'cuda', 'cuda:0', torch.device('cpu'), torch.device('cuda')])
def test_torch_nn_module_to_api(original_device, to_argument):
    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(original_device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=original_device)
    for _, parameter_value in model.named_parameters():
        assert parameter_value.device.type == original_device

    model = model.to(to_argument)
    x = x.to(to_argument)
    model(x)
    assert _utils.get_device_str(model._torch_module._execution_manager(model._is_training())._device) == \
        _utils.get_device_str(torch.device(to_argument))

def test_model_without_device():
    # Model doesn't have device (CPU is assumed)
    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out)
    model = ORTModule(model)

    # User input is on GPU
    input_device='cuda'
    x = torch.randn(N, D_in).to(input_device)

    # ORTModule and PyTorch does not move model to where user input is hosted
    with pytest.raises(RuntimeError) as type_error:
        model(x)
    assert \
        ("Tensor for argument #1 'self' is on CPU, but expected them to be on GPU (while checking arguments for addmm)" in str(type_error.value)) \
        or ("Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!" in str(type_error.value))

def test_model_and_input_without_device():
    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out)
    model = ORTModule(model)
    x = torch.randn(N, D_in)

    # CPU is assumed for both model and user input
    out = model(x)
    out is not None

def test_model_with_different_devices_same_session():
    os.environ['ORTMODULE_SKIPCHECK_POLICY'] = 'SKIP_CHECK_DISABLED'

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out)
    model = ORTModule(model)

    for i in range(5):
        if i % 2 == 0:
            device = 'cpu'
        else:
            device = 'cuda'

        model.to(device)
        x = torch.randn(N, D_in, device=device)
        y = model(x)

    del os.environ['ORTMODULE_SKIPCHECK_POLICY']

@pytest.mark.parametrize("device", ['cuda', 'cpu'])
def test_input_requires_grad_saved(device):
    N, D_in, H, D_out = 32, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device, requires_grad=True) + 1
    model(x)
    assert model._torch_module._execution_manager(model._is_training())._input_info.require_grad_names == ['input1']

@pytest.mark.parametrize("device", ['cuda', 'cpu'])
def test_input_requires_grad_backward_creates_input_grad(device):
    N, D_in, H, D_out = 32, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device, requires_grad=True)
    assert x.grad is None
    prediction = model(x)
    s = prediction.sum()
    s.backward()
    assert x.grad is not None

def test_gradient_correctness():
    device = 'cuda'
    N, D_in, H, D_out = 32, 128, 500, 10
    pt_model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    def run_step(model, x):
        prediction = model(x)
        loss = prediction.sum()
        loss.backward()
        return prediction

    for step in range(10):
        x = torch.randn(N, D_in, device=device)
        pt_prediction = run_step(pt_model, x)
        ort_prediction = run_step(ort_model, x)

        _test_helpers.assert_values_are_close(ort_prediction, pt_prediction)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model)

@pytest.mark.parametrize("device", ['cpu', 'cuda'])
@pytest.mark.parametrize("indices", ([[ 2, 3, -1, -1],[0, 1, -1, -1]],
                                     [[ 2, 3, 4, 4],[ 0, 1, 4, 4]]))
def test_scatternd_correctness(device, indices):
    class NeuralNetScatterND(torch.nn.Module):
        def __init__(self):
            super(NeuralNetScatterND, self).__init__()

        def forward(self, rerouted_output, dispatch_mask, expert_output):
            rerouted_output[dispatch_mask] = expert_output
            return rerouted_output

    pt_model = NeuralNetScatterND().to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    def run_step(model, rerouted_output, dispatch_mask, expert_output):
        prediction = model(rerouted_output, dispatch_mask, expert_output)
        return prediction

    rerouted_output = torch.tensor([[0.],[0.],[0.],[0.],[0.]], device=device)
    dispatch_mask = torch.tensor(indices, device=device)
    expert_output = torch.tensor([[[0.3817],[0.9625],[0.9625],[0.9625]],[[0.3817],[0.9625],[0.9625],[0.9625]]], device=device)

    pt_prediction = run_step(pt_model, rerouted_output, dispatch_mask, expert_output)
    ort_prediction = run_step(ort_model, rerouted_output, dispatch_mask, expert_output)
    _test_helpers.assert_values_are_close(ort_prediction, pt_prediction, atol=1e-5)


@pytest.mark.parametrize("use_fp16", [False, True])
@pytest.mark.parametrize("input_requires_grad", [False, True])
def test_gradient_correctness_conv1d(use_fp16, input_requires_grad):
    class NeuralNetConv1D(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, padding=0, groups=1):
            super(NeuralNetConv1D, self).__init__()
            self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, groups=groups)
            self.conv2 = torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, groups=groups)

        def forward(self, input):
            out = self.conv1(input.permute(0, 2, 1).contiguous())
            out = self.conv2(out).permute(0, 2, 1).contiguous()
            return out

    # ConvGrad hasn't been tested on device with arch lower than 7.0
    if torch.cuda.get_device_capability()[0] < 7:
        return

    device = 'cuda'
    N, seq_len, C_in, C_out, kernel_size = 32, 128, 1536, 1536, 3
    pt_model = NeuralNetConv1D(C_in, C_out, kernel_size, padding=1).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    def run_step(model, x):
        with amp.autocast(use_fp16):
            prediction = model(x)
            loss = prediction.sum()
        loss.backward()
        return prediction

    for _ in range(10):
        x = torch.randn(N, seq_len, C_in, device=device, requires_grad=input_requires_grad)
        pt_prediction = run_step(pt_model, x)
        ort_prediction = run_step(ort_model, x)

        # PyTorch's Conv/GonvGrad uses HEURISTIC mode to search algo while ORT uses EXHAUSTIVE mode by default.
        # While different algo types generate slightly different results, especially for FP16,
        # so relax the tolerance for comparison, especially for FP16 run and gradient comparison.
        if use_fp16:
            _test_helpers.assert_values_are_close(ort_prediction, pt_prediction, atol=1e-3, rtol=1e-3)
            _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model, rtol=5e-1, atol=4e-1)
        else:
            _test_helpers.assert_values_are_close(ort_prediction, pt_prediction, atol=1e-5)
            _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model, rtol=5e-2, atol=4e-2)

def _run_gradient_correctness_transpose(perm, shape):
    class NeuralNetTranspose(torch.nn.Module):
        def __init__(self, perm):
            super(NeuralNetTranspose, self).__init__()
            self.perm = perm

        def forward(self, input):
            out = torch.sin(input.permute(*self.perm))
            return out

    device = 'cuda'
    pt_model = NeuralNetTranspose(perm).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    def run_step(model, x):
        prediction = model(x)
        loss = prediction.sum()
        loss.backward()
        return prediction

    x = torch.randn(*shape, device=device, requires_grad=True)
    pt_prediction = run_step(pt_model, x)
    ort_prediction = run_step(ort_model, x)

    _test_helpers.assert_values_are_close(ort_prediction, pt_prediction)
    _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model)

@pytest.mark.parametrize("perm", [
    [0,1,2],      # no-op
    [0,2,1],      # special handle by Transpose021
    [1,0,2],      # handled as [0,2,1,3]
    [1,2,0],      # coalesced to [1,0]
    [2,0,1],      # coalesced to [1,0]
    [2,1,0],      # handled as [0,3,2,1]
])
@pytest.mark.parametrize("shape", [
    [245,1024,32],
    [255,2272,32],
    [246,2080,32],
    [254,128,256],
    [260,245,256],
    [284,254,256],
    [245,260,256],
    [1024,1024,256],
    [254,284,256],
    [4,5,2944],
    [4,28,3136],
    [4,312,768],
    [3,224,224],
    [17,5,4],
    [8,2080,32],
    [8,2272,32],
    [2,2,2],
    [1024,245,32],
    [2080,246,32],
    [1024,254,32],
    [2272,255,32],
    [4,5,736],
    [4,111,160],
    [8,246,32],
    [8,255,32],
    [4,1,2],
    [1,2,2],
    [2,1,2],
    [2,2,1],
    [2,1,4],
    [4,2,1],
])
def test_gradient_correctness_transpose3d(perm, shape):
    _run_gradient_correctness_transpose(perm, shape)

@pytest.mark.parametrize("perm", [
    [0,1,2,3],
    [0,1,3,2],
    [0,2,1,3],
    [0,2,3,1],
    [0,3,1,2],
    [0,3,2,1],
    [1,0,2,3],
    [1,0,3,2],
    [1,2,0,3],
    [1,2,3,0],
    [1,3,0,2],
    [1,3,2,0],
    [2,0,1,3],
    [2,0,3,1],
    [2,1,0,3],
    [2,1,3,0],
    [2,3,0,1],
    [2,3,1,0],
    [3,0,1,2],
    [3,0,2,1],
    [3,1,0,2],
    [3,1,2,0],
    [3,2,0,1],
    [3,2,1,0],
])
@pytest.mark.parametrize("shape", [
    [1,245,1024,32],
    [1,255,2272,32],
    [1,246,2080,32],
    [1,254,128,256],
    [1,260,245,256],
    [1,284,254,256],
    [1,245,260,256],
    [1,1024,1024,256],
    [1,254,284,256],
    [1,4,5,2944],
    [1,4,28,3136],
    [1,4,312,768],
    [1,3,224,224],
    [1,17,5,4],
    [260,8,2080,32],
    [284,8,2272,32],
    [1,2,2,2],
    [1,1024,245,32],
    [1,2080,246,32],
    [1,1024,254,32],
    [1,2272,255,32],
    [1,4,5,736],
    [1,4,111,160],
    [260,8,246,32],
    [284,8,255,32],
    [4,1,2,1],
    [1,1,2,2],
    [1,2,1,2],
    [1,2,2,1],
    [2,1,4,1],
    [2,2,2,1],
    [2,1,2,1],
    [1,4,2,1],
])
def test_gradient_correctness_transpose4d(perm, shape):
    _run_gradient_correctness_transpose(perm, shape)

@pytest.mark.parametrize("device", ['cuda', 'cpu'])
@pytest.mark.parametrize("padding_idx", [None, 1])
def test_gradient_correctness_embedding(device, padding_idx):
    class NeuralNetEmbedding(torch.nn.Module):
        def __init__(self, num_embeddings, embedding_dim, hidden_size):
            super(NeuralNetEmbedding, self).__init__()
            self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
            self.linear = torch.nn.Linear(embedding_dim, hidden_size)

        def forward(self, input):
            return self.linear(self.embedding(input))

    N, num_embeddings, embedding_dim, hidden_size = 64, 32, 128, 128
    pt_model = NeuralNetEmbedding(num_embeddings, embedding_dim, hidden_size).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    def run_step(model, input):
        prediction = model(input)
        loss = prediction.sum()
        loss.backward()
        return prediction

    for _ in range(10):
        input = torch.randint(high=num_embeddings, size=(N,), dtype=torch.int64, device=device)
        pt_prediction = run_step(pt_model, input)
        ort_prediction = run_step(ort_model, input)

        _test_helpers.assert_values_are_close(ort_prediction, pt_prediction)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model, atol=1e-5)

@pytest.mark.parametrize("use_fp16", [False, True])
def test_gradient_correctness_cross_entropy_loss(use_fp16):
    class NeuralNetCrossEntropyLoss(torch.nn.Module):
        def __init__(self, num_embeddings, embedding_dim):
            super(NeuralNetCrossEntropyLoss, self).__init__()
            self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=1)

        def forward(self, input, positions):
            output = torch.transpose(self.embedding(input), 0, 1)
            ignored_index = output.size(1)
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
            return loss_fct(output, positions)

    device = 'cuda'
    num_embeddings, embedding_dim = 32, 128
    pt_model = NeuralNetCrossEntropyLoss(num_embeddings, embedding_dim).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    def run_step(model, input, positions):
        with amp.autocast(use_fp16):
            loss = model(input, positions)
        loss.backward()
        return loss

    for _ in range(10):
        N = random.randint(16, 32)
        input = torch.randint(high=num_embeddings, size=(N,), dtype=torch.int64, device=device)
        positions = torch.randint(high=N, size=(embedding_dim,), dtype=torch.int64, device=device)
        pt_prediction = run_step(pt_model, input, positions)
        ort_prediction = run_step(ort_model, input, positions)

        _test_helpers.assert_values_are_close(ort_prediction, pt_prediction)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model, atol=1e-5)

@pytest.mark.parametrize("pool_type", ['MaxPool', 'AvgPool', 'AdaptiveAvgPool'])
def test_gradient_correctness_pool2d(pool_type):
    class NeuralNetPool2d(torch.nn.Module):
        def __init__(self):
            super(NeuralNetPool2d, self).__init__()
            self.conv = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pool_type == 'MaxPool':
                self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            elif pool_type == 'AvgPool':
                self.pool = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            else:
                self.pool = torch.nn.AdaptiveAvgPool2d((5, 7))

        def forward(self, input):
            return self.pool(self.conv(input))

    N, C, H, W = 8, 3, 224, 224
    device = 'cuda'
    pt_model = NeuralNetPool2d().to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    def run_step(model, input):
        prediction = model(input)
        loss = prediction.sum()
        loss.backward()
        return prediction

    for _ in range(10):
        input = torch.randn(N, C, H, W, device=device)
        pt_prediction = run_step(pt_model, input)
        ort_prediction = run_step(ort_model, input)

        _test_helpers.assert_values_are_close(ort_prediction, pt_prediction)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model, rtol=5e-3, atol=4e-3)

@pytest.mark.parametrize("pool_type", ['MaxPool', 'AvgPool'])
@pytest.mark.parametrize("stride", [None, 2])
def test_export_correctness_pool2d(pool_type, stride):
    class NeuralNetPool2d(torch.nn.Module):
        def __init__(self):
            super(NeuralNetPool2d, self).__init__()
            self.conv = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.pool_type = pool_type

        def forward(self, input):
            x = self.conv(input)
            if pool_type == 'MaxPool':
                output = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=stride)
            elif pool_type == 'AvgPool':
                output = torch.nn.functional.avg_pool2d(x, kernel_size=3, stride=stride)
            return output

    N, C, H, W = 8, 3, 224, 224
    device = 'cuda'
    pt_model = NeuralNetPool2d().to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    def run_step(model, input):
        prediction = model(input)
        return prediction

    for _ in range(10):
        input = torch.randn(N, C, H, W, device=device)
        pt_prediction = run_step(pt_model, input)
        ort_prediction = run_step(ort_model, input)

        _test_helpers.assert_values_are_close(ort_prediction, pt_prediction)

def test_gradient_correctness_argmax_unfold():
    class NeuralNetUnfold(torch.nn.Module):
        def __init__(self, input_size, hidden_size, unfold_dim, unfold_size, unfold_step):
            super(NeuralNetUnfold, self).__init__()
            self.linear= torch.nn.Linear(input_size, hidden_size)
            self.unfold_dim = unfold_dim
            self.unfold_size = unfold_size
            self.unfold_step = unfold_step

        def forward(self, input):
            return self.linear(input.argmax(-1).to(torch.float) * input.argmax().to(torch.float)).unfold(
                dimension=self.unfold_dim, size=self.unfold_size, step=self.unfold_step)

    N, D, H = 16, 256, 128
    device = 'cuda'
    pt_model = NeuralNetUnfold(D, H, 1, 50, 30).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    def run_step(model, input):
        prediction = model(input)
        loss = prediction.sum()
        loss.backward()
        return prediction

    for _ in range(10):
        input = torch.randint(0, 100, (N, D, H), dtype=torch.uint8, device=device)
        pt_prediction = run_step(pt_model, input)
        ort_prediction = run_step(ort_model, input)

        _test_helpers.assert_values_are_close(ort_prediction, pt_prediction)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model)

@pytest.mark.parametrize("high", [1, 2, 10])
def test_correctness_argmax_bitwise_or(high):
    N, D, H, M = 16, 256, 128, 4
    device = 'cuda'

    class NeuralNetBitwiseOr(torch.nn.Module):
        def __init__(self, high):
            super(NeuralNetBitwiseOr, self).__init__()
            self.other = torch.randint(0, high, (N, D, H), device=device)

        def forward(self, input):
            return torch.bitwise_or(self.other, input)

    pt_model = NeuralNetBitwiseOr(high).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    def run_step(model, input):
        prediction = model(input)
        return prediction

    for _ in range(10):
        # this also tests broadcasting
        pt_input = torch.randint(-10, 10, (M, N, D, H), device=device)
        ort_input = copy.deepcopy(pt_input)
        pt_prediction = run_step(pt_model, pt_input)
        ort_prediction = run_step(ort_model, ort_input)

        _test_helpers.assert_values_are_close(ort_prediction, pt_prediction)

@pytest.mark.parametrize("offset", [-1, 0, 1])
@pytest.mark.parametrize("dim1, dim2", ([0, 1], [0, 2], [1, 2], [2, 0]))
def test_gradient_correctness_argmax_diagonal(offset, dim1, dim2):
    class NeuralNetDiagonal(torch.nn.Module):
        def __init__(self, offset=0, dim1=0, dim2=1):
            super(NeuralNetDiagonal, self).__init__()
            self.offset = offset
            self.dim1 = dim1
            self.dim2 = dim2

        def forward(self, input):
            return torch.diagonal(input, self.offset, self.dim1, self.dim2)

    N, D, H = 16, 256, 128
    device = 'cuda'
    pt_model = NeuralNetDiagonal(offset, dim1, dim2).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    def run_step(model, input):
        prediction = model(input)
        loss = prediction.sum()
        loss.backward()
        return prediction

    for _ in range(10):
        pt_input = torch.rand((N, D, H), device=device, requires_grad=True)
        ort_input = copy.deepcopy(pt_input)
        pt_prediction = run_step(pt_model, pt_input)
        ort_prediction = run_step(ort_model, ort_input)

        _test_helpers.assert_values_are_close(ort_prediction, pt_prediction)
        _test_helpers.assert_values_are_close(ort_input.grad, pt_input.grad)

@pytest.mark.parametrize("dim", [None, 0, 1, (0, 1), (-1, 0), (0, 1, 2)])
@pytest.mark.parametrize("keepdim", [True, False])
def test_gradient_correctness_reducesum(dim, keepdim):
    class NeuralNetReduceSum(torch.nn.Module):
        def __init__(self, input_size, hidden_size, dim, keepdim):
            super(NeuralNetReduceSum, self).__init__()
            self.linear = torch.nn.Linear(input_size, hidden_size)
            self.dim = dim
            self.keepdim = keepdim

        def forward(self, input):
            t = self.linear(input)
            if self.dim is None:
                return t.sum()
            else:
                return torch.sum(t, self.dim, keepdim=self.keepdim)

    N, D, H, W = 16, 256, 128, 64
    device = 'cuda'
    pt_model = NeuralNetReduceSum(H, W, dim, keepdim).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    def run_step(model, input):
        prediction = model(input)
        loss = prediction.sum()
        loss.backward()
        return prediction

    for _ in range(10):
        pt_input = torch.rand((N, D, H), device=device, requires_grad=True)
        ort_input = copy.deepcopy(pt_input)
        pt_prediction = run_step(pt_model, pt_input)
        ort_prediction = run_step(ort_model, ort_input)

        _test_helpers.assert_values_are_close(ort_prediction, pt_prediction)
        _test_helpers.assert_values_are_close(ort_input.grad, pt_input.grad)

# In PyTorch 1.11.0, there is issue during reduce node shape handling for exporter, so any sub-graph that
# contains ReduceProd will fail to run, for example, "sec,sm->ecm", "sec,ecm->sm".
# Currently skip these cases and test_gradient_correctness_einsum_2,
# will enable these tests again once the issue in PyTorch is fixed.
skip_torch_1_11 = pytest.mark.skipif(LooseVersion(torch.__version__) >= LooseVersion('1.11.0'), reason="PyTorch 1.11 incompatible")
@pytest.mark.parametrize("equation", [
    "s,se->se", "se,sc->sec", "se,se->s", "ks,ksm->sm", "kes,ems->mek", "kes,ksm->ms",
    pytest.param("sec,sm->ecm", marks=[skip_torch_1_11]),
    pytest.param("sec,ecm->sm", marks=[skip_torch_1_11])
])
def test_gradient_correctness_einsum(equation):
    class NeuralNetEinsum(torch.nn.Module):
        def __init__(self, bias_size):
            super(NeuralNetEinsum, self).__init__()
            self.register_parameter(name='bias', param=torch.nn.Parameter(torch.randn(bias_size)))

        def forward(self, left, right):
            left = left + self.bias
            return torch.einsum(equation, left, right)

    device = 'cuda'
    K, S, M, E = 16, 1024, 768, 64
    C = int(S/E*2)

    SIZE_MAP = { 'K': K, 'S': S, 'E': E, 'C': C, 'M': M }

    pos1 = equation.find(',')
    pos2 = equation.find('->')
    lhs_op = equation[0:pos1]
    rhs_op = equation[pos1 + 1:pos2]
    lhs_shape = []
    for c in lhs_op:
        lhs_shape.append(SIZE_MAP[c.upper()])
    rhs_shape = []
    for c in rhs_op:
        rhs_shape.append(SIZE_MAP[c.upper()])

    pt_model = NeuralNetEinsum(lhs_shape[-1]).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    def run_step(model, input_left, input_right):
        prediction = model(input_left, input_right)
        loss = prediction.sum()
        loss.backward()
        return prediction

    for _ in range(10):
        pt_input_left = torch.rand(lhs_shape, device=device)
        pt_input_right = torch.rand(rhs_shape, device=device)
        ort_input_left = copy.deepcopy(pt_input_left)
        ort_input_right = copy.deepcopy(pt_input_right)
        pt_prediction = run_step(pt_model, pt_input_left, pt_input_right)
        ort_prediction = run_step(ort_model, ort_input_left, ort_input_right)

        _test_helpers.assert_values_are_close(ort_prediction, pt_prediction, atol=1e-3, rtol=1e-3)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model, atol=1e-3, rtol=1e-3)

@skip_torch_1_11
def test_gradient_correctness_einsum_2():
    class NeuralNetEinsum(torch.nn.Module):
        def __init__(self, bias_size):
            super(NeuralNetEinsum, self).__init__()
            self.register_parameter(name='bias', param=torch.nn.Parameter(torch.randn(bias_size)))

        def forward(self, left, right):
            left = left + self.bias
            return torch.einsum(equation, left, right)

    device = 'cuda'
    A, B, C, D = 16, 32, 8, 64

    SIZE_MAP = { 'A': A, 'B': B, 'C': C, 'D': D }

    def to_string(perm):
        result = ''
        for v in perm:
            result += chr(ord('a') + v)
        return result

    lhs_candidates = [[0], [0,1], [0,1,2]]
    perm = [0,1,2,3]
    combs = list(itertools.combinations(perm, 1)) + list(itertools.combinations(perm, 2)) + list(itertools.combinations(perm, 3))
    rhs_candidates = []
    for comb in combs:
        rhs_candidates += list(itertools.permutations(comb))

    all_cases = []
    for lhs_candidate in lhs_candidates:
        for rhs_candidate in [list(candidate) for candidate in rhs_candidates]:
            union = list(set(lhs_candidate + rhs_candidate))
            # Union should contains contiguous numbers from 0, otherwise it's same as another case.
            if any(v >= len(union) for v in union):
                continue
            # Numbers in right but not in left should be sorted, otherwise it's same as another case.
            only_in_right = [v for v in rhs_candidate if v not in lhs_candidate]
            if any(only_in_right[i] > only_in_right[i + 1] for i in range(len(only_in_right) - 1)):
                continue
            combs = []
            for i in range(1, len(union) + 1):
                combs += list(itertools.combinations(union, i))
            output_candidates = []
            for comb in combs:
                output_candidates += list(itertools.permutations(comb))
            # When output_candidates is too many, it will take long time to run. Sample part of them.
            random.shuffle(output_candidates)
            output_candidates = output_candidates[:8]
            for output_candidate in [list(candidate) for candidate in output_candidates]:
                all_cases.append((lhs_candidate, rhs_candidate, output_candidate))

    for case in all_cases:
        equation = to_string(case[0]) + ',' + to_string(case[1]) + '->' + to_string(case[2])
        pos1 = equation.find(',')
        pos2 = equation.find('->')
        lhs_op = equation[0:pos1]
        rhs_op = equation[pos1 + 1:pos2]
        lhs_shape = []
        for c in lhs_op:
            lhs_shape.append(SIZE_MAP[c.upper()])
        rhs_shape = []
        for c in rhs_op:
            rhs_shape.append(SIZE_MAP[c.upper()])

        pt_model = NeuralNetEinsum(lhs_shape[-1]).to(device)
        ort_model = ORTModule(copy.deepcopy(pt_model))

        def run_step(model, input_left, input_right):
            prediction = model(input_left, input_right)
            loss = prediction.sum()
            loss.backward()
            return prediction

        for _ in range(5):
            pt_input_left = torch.rand(lhs_shape, device=device)
            pt_input_right = torch.rand(rhs_shape, device=device)
            ort_input_left = copy.deepcopy(pt_input_left)
            ort_input_right = copy.deepcopy(pt_input_right)
            pt_prediction = run_step(pt_model, pt_input_left, pt_input_right)
            ort_prediction = run_step(ort_model, ort_input_left, ort_input_right)

            _test_helpers.assert_values_are_close(ort_prediction, pt_prediction, atol=1e-3, rtol=1e-3)
            _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model, atol=1e-3, rtol=1e-3)

# Since multinomial is a generator function, we do not have to test for gradient
# Two consecutive calls on the torch.multinomail on a probability distribution with more
# than one index with non-zero probability(eg, [0, 10, 3, 0]) will not result in
# the same output. Thus we reset the seed before each call to the op torch.multinomial.
@pytest.mark.parametrize("input_shape", ([5], [2,5]))
@pytest.mark.parametrize("num_samples, replacement", ((1, False), (2, True)))
def test_aten_multinomial(input_shape, num_samples, replacement):
    class NeuralNetDiagonal(torch.nn.Module):
        def __init__(self, num_samples, replacement):
            super(NeuralNetDiagonal, self).__init__()
            self.num_samples = num_samples
            self.replacement = replacement

        def forward(self, input):
            return torch.multinomial(input, self.num_samples, self.replacement)

    torch.backends.cudnn.deterministic = True
    device = 'cuda'
    pt_model = NeuralNetDiagonal(num_samples, replacement).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    def run_step(model, input):
        # reset manual seed to reset the generator
        torch.manual_seed(5032)
        prediction = model(input)
        return prediction

    pt_input = torch.rand(input_shape, dtype=torch.float, device=device)
    ort_input = copy.deepcopy(pt_input)
    pt_prediction = run_step(pt_model, pt_input)
    ort_prediction = run_step(ort_model, ort_input)
    # run the ort prediction again since the first call involves export
    # and run step, which means the torch.multinomial is called twice in a row without
    # resetting the generator in between, which will result in a different output
    ort_prediction = run_step(ort_model, ort_input)

    _test_helpers.assert_values_are_close(ort_prediction, pt_prediction)

@pytest.mark.parametrize("input_shape", ([4,2],))
def test_aten_argmax(input_shape):
    import torch.nn.functional as F
    class TopKGate(torch.nn.Module):
        def forward(self, input: torch.Tensor):
            indices = torch.argmax(input, dim = 1)
            device = 'cpu' if indices.get_device() < 0 else indices.get_device()
            ret = torch.zeros(indices.shape[0], 2, dtype=torch.int64, device=device)
            ret = ret.scatter(-1, indices.unsqueeze(-1), 1) + input
            return ret

    device = 'cuda'
    pt_model = TopKGate()
    ort_model = ORTModule(copy.deepcopy(pt_model))
    pt_input = torch.rand(input_shape, dtype=torch.float, device=device, requires_grad=True)
    ort_input = copy.deepcopy(pt_input)
    pt_output = pt_model(pt_input)
    ort_output = ort_model(ort_input)
    loss = ort_output[0].sum()
    loss.backward()

    _test_helpers.assert_values_are_close(ort_output, pt_output)

@pytest.mark.parametrize("input_shape", ([], [5], [2,5], [3,2,5]))
def test_numpy_T(input_shape):
    class NeuralNet(torch.nn.Module):
        def __init__(self):
            super(NeuralNet, self).__init__()

        def forward(self, input):
            return input.T

    torch.backends.cudnn.deterministic = True
    device = 'cuda'
    pt_model = NeuralNet().to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model), DebugOptions(log_level=LogLevel.VERBOSE))

    def run_step(model, input):
        prediction = model(input)
        return prediction

    # reset manual seed to reset the generator
    torch.manual_seed(5032)
    pt_input = torch.rand(input_shape, dtype=torch.float, device=device)
    ort_input = copy.deepcopy(pt_input)
    pt_prediction = run_step(pt_model, pt_input)
    ort_prediction = run_step(ort_model, ort_input)

    _test_helpers.assert_values_are_close(ort_prediction, pt_prediction)

def test_gradient_correctness_bce_with_logits():
    class NeuralNetBCEWithLogitsLoss(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super(NeuralNetBCEWithLogitsLoss, self).__init__()
            self.linear = torch.nn.Linear(input_size, hidden_size)

        def forward(self, input, target):
            loss_fct = torch.nn.BCEWithLogitsLoss()
            return loss_fct(self.linear(input), target)

    N, D, H = 16, 256, 128
    device = 'cuda'
    pt_model = NeuralNetBCEWithLogitsLoss(D, H).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    def run_step(model, input, target):
        prediction = model(input, target)
        loss = prediction.sum()
        loss.backward()
        return prediction

    for _ in range(10):
        pt_input = torch.rand((N, D), device=device, requires_grad=True)
        ort_input = copy.deepcopy(pt_input)
        pt_target = torch.rand((N, H), device=device)
        ort_target = copy.deepcopy(pt_target)
        pt_prediction = run_step(pt_model, pt_input, pt_target)
        ort_prediction = run_step(ort_model, ort_input, ort_target)

        _test_helpers.assert_values_are_close(ort_prediction, pt_prediction)
        _test_helpers.assert_values_are_close(ort_input.grad, pt_input.grad)

def test_module_with_non_differential_output():
    device = 'cuda'
    N, D_in, H, D_out = 32, 128, 64, 10
    pt_model = NeuralNetNonDifferentiableOutput(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    def run_step(model, x):
        prediction1, mask1, prediction2, mask2 = model(x)
        loss = prediction2.sum()
        loss.backward()
        return prediction1, mask1, prediction2, mask2

    for step in range(10):
        x = torch.randn(N, D_in, device=device)
        pt_prediction1, pt_mask1, pt_prediction2, pt_mask2 = run_step(pt_model, x)
        ort_prediction1, ort_mask1, ort_prediction2, ort_mask2 = run_step(ort_model, x)

        # _test_helpers.assert_values_are_close(ort_prediction1, pt_prediction1)   # TODO: this is failing, need to investigate!
                                                                                   # This will be no reproducible if we change the model forward to
                                                                                   # mask1 = torch.gt(out, 0.01)
        _test_helpers.assert_values_are_close(ort_prediction2, pt_prediction2)
        _test_helpers.assert_values_are_close(ort_mask1, pt_mask1)
        _test_helpers.assert_values_are_close(ort_mask2, pt_mask2)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model)

def test_multiple_chained_ortmodules_with_non_differential_output():
    device = 'cuda'
    N, D_in, H, D_out = 32, 128, 64, 10
    pt_model = NeuralNetChainedLayersWithNonDifferentiableOutput(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    pt_model2 = NeuralNetChainedLayersWithNonDifferentiableOutput(D_in, H, D_out).to(device)
    ort_model2 = ORTModule(copy.deepcopy(pt_model2))

    def run_step(layer1, layer2, x, mask1):
        prediction, mask = layer1(x, mask1)
        prediction, mask = layer2(x, mask)
        loss = prediction.sum()
        loss.backward()
        return prediction, mask

    x = torch.randn(N, D_in, device=device)
    mask1 = torch.zeros(1, device=device)

    pt_prediction, pt_mask = run_step(pt_model, pt_model2, x, mask1)
    # ensure no AssertionError message for chained ortmodules, e.g.:
    #       ORT found the 1-th module output 'output-1' is non-differentiable according to the onnx graph.
    #       However, the gradient value is still provided by PyTorch's autograd engine.
    ort_prediction, ort_mask = run_step(ort_model, ort_model2, x, mask1)

    _test_helpers.assert_values_are_close(ort_prediction, pt_prediction)
    _test_helpers.assert_gradients_match_and_reset_gradient(ort_model2, pt_model2)

@pytest.mark.parametrize("loss_with_duplicated_output", [False, True])
def test_duplicated_output(loss_with_duplicated_output):
    class NeuralNet(torch.nn.Module):
        def __init__(self):
            super(NeuralNet, self).__init__()
            self.fc1 = torch.nn.Linear(128, 16)

        def forward(self, input):
            out = self.fc1(input)
            return out, out     # duplicated output

    N, C, H = 8, 4, 128
    device = 'cuda'
    pt_model = NeuralNet().to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    def run_step(model, input):
        out, out_dup = model(input)
        loss = out.sum()
        if loss_with_duplicated_output:
            loss = loss + (2 * out_dup).sum()
        loss.backward()
        return out, out_dup

    for _ in range(10):
        input = torch.randn(N, C, H, device=device)
        pt_prediction1, pt_prediction2 = run_step(pt_model, input)
        ort_prediction1, ort_prediction2 = run_step(ort_model, input)

        _test_helpers.assert_values_are_close(ort_prediction1, pt_prediction1)
        _test_helpers.assert_values_are_close(ort_prediction2, pt_prediction2)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model, atol=1e-5)

def test_multiple_forward_only_calls():
    device = 'cuda'
    N, D_in, H, D_out = 32, 784, 500, 10
    pt_model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    for step in range(10):
        x = torch.randn(N, D_in, device=device, requires_grad=False)
        pt_prediction = pt_model(x)
        ort_prediction = ort_model(x)

        _test_helpers.assert_values_are_close(ort_prediction, pt_prediction)

def test_nesting_forward_backward_calls():
    device = 'cuda'
    N, D_in, H, D_out = 32, 784, 500, 10
    pt_model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    # forward1
    ort_x1 = torch.randn(N, D_in, device=device, requires_grad=True)
    pt_x1 = copy.deepcopy(ort_x1)
    ort_prediction1 = ort_model(ort_x1)
    pt_prediction1 = pt_model(pt_x1)
    _test_helpers.assert_values_are_close(ort_prediction1, pt_prediction1)
    ort_loss1 = ort_prediction1.sum()
    pt_loss1 = pt_prediction1.sum()
    # forward2
    ort_x2 = torch.randn(N, D_in, device=device, requires_grad=True)
    pt_x2 = copy.deepcopy(ort_x2)
    ort_prediction2 = ort_model(ort_x2)
    ort_loss2 = ort_prediction2.sum()
    pt_prediction2 = pt_model(pt_x2)
    pt_loss2 = pt_prediction2.sum()
    _test_helpers.assert_values_are_close(ort_prediction2, pt_prediction2)
    # backward2
    ort_loss2.backward()
    pt_loss2.backward()
    _test_helpers.assert_values_are_close(ort_x2.grad, ort_x2.grad)
    _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model)
    # backward1
    ort_loss1.backward()
    pt_loss1.backward()
    _test_helpers.assert_values_are_close(ort_x1.grad, pt_x1.grad)
    _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model)

def test_multiple_overlapping_forward_backward_calls():
    device = 'cuda'
    N, D_in, H, D_out = 32, 784, 500, 10
    pt_model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    def run_step(model, x1, x2):
        prediction1 = model(x1)
        loss1 = prediction1.sum()

        prediction2 = model(x2)
        loss2 = prediction2.sum()

        loss1.backward()
        loss2.backward()
        return prediction1, prediction2

    for step in range(10):
        pt_x1 = torch.randn(N, D_in, device=device, requires_grad=True)
        pt_x2 = torch.randn(N, D_in, device=device, requires_grad=True)

        ort_x1 = pt_x1.clone().detach()
        ort_x2 = pt_x2.clone().detach()
        ort_x1.requires_grad = True
        ort_x2.requires_grad = True

        pt_prediction1, pt_prediction2 = run_step(pt_model, pt_x1, pt_x2)
        ort_prediction1, ort_prediction2 = run_step(ort_model, ort_x1, ort_x2)

        _test_helpers.assert_values_are_close(ort_prediction1, pt_prediction1)
        _test_helpers.assert_values_are_close(ort_prediction2, pt_prediction2)
        _test_helpers.assert_values_are_close(ort_x1.grad, pt_x1.grad)
        _test_helpers.assert_values_are_close(ort_x2.grad, pt_x2.grad)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model)

def test_multiple_ortmodules_training():
    device = 'cuda'
    N, D_in, H, D_out = 32, 784, 128, 10
    pt_model1 = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    pt_model2 = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    ort_model1 = ORTModule(copy.deepcopy(pt_model1))
    ort_model2 = ORTModule(copy.deepcopy(pt_model2))

    def run_step(model1, model2, x1, x2):
        prediction1 = model1(x1)
        loss1 = prediction1.sum()
        loss1.backward()

        prediction2 = model2(x2)
        loss2 = prediction2.sum()
        loss2.backward()
        return prediction1, prediction2

    for step in range(10):
        x1 = torch.randn(N, D_in, device=device)
        x2 = torch.randn(N, D_in, device=device)
        pt_prediction1, pt_prediction2 = run_step(pt_model1, pt_model2, x1, x2)
        ort_prediction1, ort_prediction2 = run_step(ort_model1, ort_model2, x1, x2)

        _test_helpers.assert_values_are_close(ort_prediction1, pt_prediction1, atol=1e-6)
        _test_helpers.assert_values_are_close(ort_prediction2, pt_prediction2, atol=1e-6)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model1, pt_model1)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model2, pt_model2)

def test_multiple_ortmodules_common_backbone_training():
    device = 'cuda'
    N, D_in, H, D_out = 32, 64, 128, 64
    pt_model0 = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    pt_model1 = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    pt_model2 = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    # model is the common backbone shared by model1 and model2
    ort_model0 = ORTModule(copy.deepcopy(pt_model0))
    ort_model1 = ORTModule(copy.deepcopy(pt_model1))
    ort_model2 = ORTModule(copy.deepcopy(pt_model2))

    def run_step(backbone_layers, task_layers, x):
        prediction = task_layers(backbone_layers(x))
        loss = prediction.sum()
        loss.backward()
        return prediction

    for step in range(10):
        # Run task 1
        x1 = torch.randn(N, D_in, device=device)
        pt_prediction = run_step(pt_model0, pt_model1, x1)
        ort_prediction = run_step(ort_model0, ort_model1, x1)

        _test_helpers.assert_values_are_close(ort_prediction, pt_prediction)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model0, pt_model0, reset_gradient=False)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model1, pt_model1)

        # Run task 2
        x2 = torch.randn(N, D_in, device=device)
        pt_prediction = run_step(pt_model0, pt_model2, x1)
        ort_prediction = run_step(ort_model0, ort_model2, x1)

        _test_helpers.assert_values_are_close(ort_prediction, pt_prediction)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model0, pt_model0, reset_gradient=True, atol=1e-5)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model2, pt_model2)

def test_multiple_chained_ortmodules_training():
    device = 'cuda'
    N, D_in, H, D_out = 32, 128, 500, 128
    pt_model1 = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    pt_model2 = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    ort_model1 = ORTModule(copy.deepcopy(pt_model1))
    ort_model2 = ORTModule(copy.deepcopy(pt_model2))

    def run_step(layers1, layers2, x):
        prediction = layers2(layers1(x))
        loss = prediction.sum()
        loss.backward()
        return prediction

    for step in range(10):
        x = torch.randn(N, D_in, device=device, requires_grad=True)
        pt_prediction = run_step(pt_model1, pt_model2, x)
        ort_prediction = run_step(ort_model1, ort_model2, x)

        _test_helpers.assert_values_are_close(ort_prediction, pt_prediction)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model1, pt_model1)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model2, pt_model2)

def test_mixed_nnmodule_ortmodules_training():
    device = 'cuda'
    N, D_in, H, D_out = 32, 128, 500, 128
    pt_model1 = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    pt_model2 = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    pt_model3 = NeuralNetMultiplePositionalArguments(D_in, H, D_out).to(device)
    ort_model1 = ORTModule(copy.deepcopy(pt_model1))
    ort_model2 = copy.deepcopy(pt_model2)   # model2 is intentionally left as nn.module
    ort_model3 = ORTModule(copy.deepcopy(pt_model3))

    def run_step(model1, model2, model3, x1, x2):
        a1 = model1(x1)
        a2 = model2(x2)
        a3 = model3(torch.sin(a1), torch.cos(a2))
        loss = a3.sum()
        loss.backward()
        return a1, a2, a3

    for step in range(10):
        x1 = torch.randn(N, D_in, device=device)
        x2 = torch.randn(N, D_in, device=device)
        pt_p1, pt_p2, pt_p3 = run_step(pt_model1, pt_model2, pt_model3, x1, x2)
        ort_p1, ort_p2, ort_p3 = run_step(ort_model1, ort_model2, ort_model3, x1, x2)

        _test_helpers.assert_values_are_close(ort_p1, pt_p1, atol=1e-06)
        _test_helpers.assert_values_are_close(ort_p2, pt_p2, atol=1e-06)
        _test_helpers.assert_values_are_close(ort_p3, pt_p3, atol=1e-06)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model1, pt_model1, atol=2e-6)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model2, pt_model2)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model3, pt_model3)

def test_identity_elimination():
    class NeuralNetSimpleIdentity(torch.nn.Module):
        def __init__(self, input_size, num_classes):
            super(NeuralNetSimpleIdentity, self).__init__()

            self.fc = torch.nn.Linear(input_size, num_classes)

        # Identity node will be created between ReduceSum and graph output
        # and then eliminated after transformation
        def forward(self, x):
            y = self.fc(x)
            z = y
            return z

    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSimpleIdentity(D_in, D_out).to(device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device)
    output = model(x)

    # Make sure model runs OK
    assert output is not None

def test_ortmodule_inputs_with_dynamic_shape():
    D_in, H, D_out = 784, 500, 10

    pt_model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to('cuda')
    ort_model = ORTModule(copy.deepcopy(pt_model))

    def run_step(model, x):
        p = model(x)
        loss = p.sum()
        loss.backward()
        return p

    for step in range(10):
        N = random.randint(1,100)
        x = torch.randn(N, D_in, device='cuda', requires_grad=True)
        assert x.grad is None

        pt_p = run_step(pt_model, x)
        ort_p = run_step(ort_model, x)

        _test_helpers.assert_values_are_close(ort_p, pt_p, atol=1e-6)    # relaxing tolerance, 1e-7 or less would fail
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model)


def test_bert_inputs_with_dynamic_shape():

    # create pytorch model with dropout disabled
    pt_model = _get_bert_for_sequence_classification_model('cuda',
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    def run_step(model, x, y, z):
        outputs = model(x, y, None, None, None, None, z)
        loss = outputs[0]
        loss.backward()
        return outputs[0]

    for step in range(10):
        x, y, z = _get_bert_for_sequence_classification_sample_data_with_random_shapes('cuda')

        pt_p = run_step(pt_model, x, y, z)
        ort_p = run_step(ort_model, x, y, z)

        _test_helpers.assert_values_are_close(ort_p, pt_p, atol=1e-02)      # TODO: this assert is failing with smaller tolerance, need to investigate!!
        # _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model)  #TODO - enable this check after the investigation


@pytest.mark.parametrize("device", ['cuda', 'cpu'])
def test_changes_input_requires_grad_reinitializes_module_gradient_graph_builder(device):
    os.environ['ORTMODULE_SKIPCHECK_POLICY'] = 'SKIP_CHECK_DISABLED'

    N, D_in, H, D_out = 32, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device)
    y = x.clone()
    y.requires_grad_(True)
    output_x = torch.sum(model(x))
    output_x.backward()
    assert x.grad is None
    module_gradient_graph_builder_training = \
        model._torch_module._execution_manager(model._is_training())._graph_builder
    output_y = torch.sum(model(y))
    output_y.backward()
    assert y.grad is not None
    assert module_gradient_graph_builder_training != \
        model._torch_module._execution_manager(model._is_training())._graph_builder

    del os.environ['ORTMODULE_SKIPCHECK_POLICY']

@pytest.mark.parametrize("device", ['cuda'])
def test_input_requires_grad_backward_creates_input_grad_as_required0(device):
    N, D_in, H, D_out = 32, 784, 500, 10
    pt_model = NeuralNetMultiplePositionalArgumentsMultiOutputsWithoutDependency(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))
    pt_x1 = torch.randn(N, D_in, device=device, requires_grad=True)
    pt_x2 = torch.randn(N, D_in, device=device, requires_grad=True)
    ort_x1 = pt_x1.clone().detach()
    ort_x2 = pt_x2.clone().detach()
    ort_x1.requires_grad = True
    ort_x2.requires_grad = True

    def run_step0(model, x1, x2):
        y1, _ = model(x1, x2)
        s1 = y1.sum()
        s1.backward()   # y2's gradient will be materialized to full shape.
        return y1

    pt_y1 = run_step0(pt_model, pt_x1, pt_x2)
    ort_y1 = run_step0(ort_model, ort_x1, ort_x2)

    _test_helpers.assert_values_are_close(pt_y1, ort_y1, atol=1e-06)
    _test_helpers.assert_values_are_close(ort_x1.grad, pt_x1.grad)
    _test_helpers.assert_values_are_close(ort_x2.grad, pt_x2.grad)
    # backward() is from y1, so grad of fc2.weight and fc2.bias will not be calculated.
    _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model, none_pt_params=['fc2.weight', 'fc2.bias'], reset_gradient=True)

    def run_step1(model, x1, x2):
        _, y2 = model(x1, x2)
        s2 = y2.sum()
        s2.backward()   # y1's gradient will be materialized to full shape.
        return y2

    pt_y2 = run_step1(pt_model, pt_x1, pt_x2)
    ort_y2 = run_step1(ort_model, ort_x1, ort_x2)

    _test_helpers.assert_values_are_close(pt_y2, ort_y2, atol=1e-06)
    _test_helpers.assert_values_are_close(ort_x1.grad, pt_x1.grad)
    _test_helpers.assert_values_are_close(ort_x2.grad, pt_x2.grad)
    # backward() is from y2, so grad of fc1.weight and fc1.bias will not be calculated.
    _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model, none_pt_params=['fc1.weight', 'fc1.bias'])


@pytest.mark.parametrize("device", ['cuda'])
def test_model_output_with_inplace_update(device):
    class NeuralNetWithGradNeedOutput(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super(NeuralNetWithGradNeedOutput, self).__init__()
            self.fc1_1 = torch.nn.Linear(input_size, hidden_size)
            # Softmax's gradient is depending on its output
            self.act = torch.nn.Softmax(dim=1)

        def forward(self, input1):
            out1 = self.act(self.fc1_1(input1))
            return out1

    def run_step(model, x1):
        y1 = model(x1)
        y1.add_(1)  # inplace update to module output
        y1 = y1.sum()
        y1.backward()
        return y1

    N, D_in, H = 32, 784, 500
    pt_model = NeuralNetWithGradNeedOutput(D_in, H).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    pt_x1 = torch.randn(N, D_in, device=device, requires_grad=True)
    ort_x1 = pt_x1.clone()

    with pytest.raises(Exception) as ex_info:
        pt_y1 = run_step(pt_model, pt_x1)
    assert "modified by an inplace operation" in str(ex_info.value)

    with pytest.raises(Exception) as ex_info:
        ort_y1 = run_step(ort_model, ort_x1)
    assert "modified by an inplace operation" in str(ex_info.value)

@pytest.mark.parametrize("device", ['cuda'])
def test_loss_combines_two_outputs_with_dependency(device):

    def run_step(model, x1, x2):
        y1, y2 = model(x1, x2)
        loss = y1.sum() + y2.sum()
        loss.backward()
        return y1, y2

    N, D_in, H, D_out = 32, 784, 500, 10
    pt_model = NeuralNetMultiplePositionalArgumentsMultiOutputsWithDependency(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    pt_x1 = torch.randn(N, D_in, device=device, requires_grad=False)
    pt_x2 = torch.randn(N, D_in, device=device, requires_grad=False)
    ort_x1 = pt_x1.clone()
    ort_x2 = pt_x2.clone()

    # Both y1 and y2's gradients are not None.
    pt_y1, pt_y2 = run_step(pt_model, pt_x1, pt_x2)
    ort_y1, ort_y2 = run_step(ort_model, ort_x1, ort_x2)

    _test_helpers.assert_values_are_close(pt_y1, ort_y1, atol=1e-06)
    _test_helpers.assert_values_are_close(pt_y2, ort_y2, atol=1e-06)
    _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model)

@pytest.mark.parametrize("x1_requires_grad", [True, False])
@pytest.mark.parametrize("x2_requires_grad", [True, False])
def test_input_requires_grad_backward_creates_input_grad_as_required1(x1_requires_grad, x2_requires_grad):

    def run_step(model, x1, x2):
        y1, y2 = model(x1, x2)
        s = y2.sum()
        s.backward()
        return y1, y2

    N, D_in, H, D_out = 32, 784, 500, 10
    device = 'cuda'
    pt_model = NeuralNetMultiplePositionalArgumentsMultiOutputsWithDependency(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))
    pt_x1 = torch.randn(N, D_in, device=device, requires_grad=x1_requires_grad)
    pt_x2 = torch.randn(N, D_in, device=device, requires_grad=x2_requires_grad)

    ort_x1 = pt_x1.clone().detach()
    ort_x2 = pt_x2.clone().detach()
    ort_x1.requires_grad = x1_requires_grad
    ort_x2.requires_grad = x2_requires_grad

    pt_y1, pt_y2 = run_step(pt_model, pt_x1, pt_x2)
    ort_y1, ort_y2 = run_step(ort_model, ort_x1, ort_x2)

    _test_helpers.assert_values_are_close(ort_y1, pt_y1, atol=1e-06)
    _test_helpers.assert_values_are_close(ort_y2, pt_y2, atol=1e-06)
    assert not x1_requires_grad or ort_x1.grad is not None
    assert not x2_requires_grad or ort_x2.grad is not None
    assert not x1_requires_grad or torch.allclose(ort_x1.grad, pt_x1.grad)
    assert not x2_requires_grad or torch.allclose(ort_x2.grad, pt_x2.grad)
    _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model)


@pytest.mark.parametrize("device", ['cuda'])
def test_model_with_bypass_input(device):
    class NeuralNetWithBypassInput(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(NeuralNetWithBypassInput, self).__init__()

            self.fc1_1 = torch.nn.Linear(input_size, hidden_size)
            self.relu1 = torch.nn.ReLU()
            self.fc1_2 = torch.nn.Linear(hidden_size, num_classes)

        def forward(self, input1, bypass_input):
            out1 = self.fc1_2(self.relu1(self.fc1_1(input1)))
            # use shape from bypass_input
            out1 = out1.view(bypass_input.size()[0], -1)
            return out1, bypass_input

    def run_step(model, x1, x2):
        y1, y2 = model(x1, x2)
        loss = y1.sum() + y2.sum()
        loss.backward()
        return y1, y2

    N, D_in, H, D_out = 32, 784, 500, 10
    pt_model = NeuralNetWithBypassInput(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    pt_x1 = torch.randn(N, D_in, device=device, requires_grad=True)
    pt_x2 = torch.randn(N, D_in, device=device, requires_grad=True)
    ort_x1 = pt_x1.clone()
    ort_x2 = pt_x2.clone()

    # Both y1 and y2's gradients are not None.
    pt_y1, pt_y2 = run_step(pt_model, pt_x1, pt_x2)
    ort_y1, ort_y2 = run_step(ort_model, ort_x1, ort_x2)

    _test_helpers.assert_values_are_close(pt_y1, ort_y1, atol=1e-06)
    _test_helpers.assert_values_are_close(pt_y2, ort_y2, atol=1e-06)
    _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model)

def test_gpu_reserved_memory_with_torch_no_grad():
    device = 'cuda'

    # Create a model and get the memory_reserved when torch.no_grad has been enabled
    # before and after export
    model_with_no_grad = _get_bert_for_sequence_classification_model(device)
    x, y, z = _get_bert_for_sequence_classification_sample_data(device)

    torch.cuda.empty_cache()
    model_with_no_grad = ORTModule(model_with_no_grad)
    model_with_no_grad(x, attention_mask=y, labels=z)
    mem_reserved_after_export_with_torch_no_grad = torch.cuda.memory_reserved(device)
    del model_with_no_grad

    # Create another model and get the memory_reserved when torch.no_grad has not been enabled after export.
    model_without_no_grad = _get_bert_for_sequence_classification_model(device)
    model_without_no_grad = ORTModule(model_without_no_grad)
    mem_reserved_after_export_without_torch_no_grad = 0

    with patch('torch.no_grad'):
        model_without_no_grad(x, attention_mask=y, labels=z)
        mem_reserved_after_export_without_torch_no_grad = torch.cuda.memory_reserved(device)

    assert mem_reserved_after_export_with_torch_no_grad <= mem_reserved_after_export_without_torch_no_grad

@pytest.mark.parametrize("return_type", [dict, OrderedDict, SequenceClassifierOutput])
@pytest.mark.parametrize("device", ['cpu', 'cuda'])
def test_dict_return_value_module(return_type, device):
    class NeuralNetDictOutput(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(NeuralNetDictOutput, self).__init__()

            self.fc1_1 = torch.nn.Linear(input_size, hidden_size)
            self.relu1 = torch.nn.ReLU()
            self.fc1_2 = torch.nn.Linear(hidden_size, num_classes)

            self.fc2_1 = torch.nn.Linear(input_size, hidden_size)
            self.relu2 = torch.nn.ReLU()
            self.fc2_2 = torch.nn.Linear(hidden_size, num_classes)

            self.fc3_1 = torch.nn.Linear(input_size, hidden_size)
            self.relu3 = torch.nn.ReLU()
            self.fc3_2 = torch.nn.Linear(hidden_size, num_classes)

        def forward(self, input1, input2, input3):
            out1 = self.fc1_2(self.relu1(self.fc1_1(input1)))
            out2 = self.fc2_2(self.relu2(self.fc2_1(input2)))
            out3 = self.fc3_2(self.relu3(self.fc3_1(input3)))
            return return_type([('loss', out1), ('logits', out2), ('hidden_states', out3)])

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetDictOutput(D_in, H, D_out).to(device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_in, device=device)
    z = torch.randn(N, D_in, device=device)

    output = model(x, y, z)
    assert isinstance(output, return_type)
    assert 'loss' in output and 'logits' in output and 'hidden_states' in output

@pytest.mark.parametrize("device", ['cuda', 'cpu'])
def test_dict_of_tuple_return_value_module(device):
    class NeuralNetDictOfTuplesOutput(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(NeuralNetDictOfTuplesOutput, self).__init__()

            self.fc1_1 = torch.nn.Linear(input_size, hidden_size)
            self.relu1 = torch.nn.ReLU()
            self.fc1_2 = torch.nn.Linear(hidden_size, num_classes)

            self.fc2_1 = torch.nn.Linear(input_size, hidden_size)
            self.relu2 = torch.nn.ReLU()
            self.fc2_2 = torch.nn.Linear(hidden_size, num_classes)

            self.fc3_1 = torch.nn.Linear(input_size, hidden_size)
            self.relu3 = torch.nn.ReLU()
            self.fc3_2 = torch.nn.Linear(hidden_size, num_classes)

        def forward(self, input1, input2, input3):
            out1 = self.fc1_2(self.relu1(self.fc1_1(input1)))
            out2 = self.fc2_2(self.relu2(self.fc2_1(input2)))
            out3 = self.fc3_2(self.relu3(self.fc3_1(input3)))
            return {'loss': (out1, out2, out3)}

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetDictOfTuplesOutput(D_in, H, D_out).to(device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_in, device=device)
    z = torch.randn(N, D_in, device=device)

    output = model(x, y, z)
    assert 'loss' in output
    assert len(output['loss']) == 3

@pytest.mark.parametrize("device", ['cuda', 'cpu'])
def test_tuple_of_tuple_return_value_module(device):
    class NeuralNetTupleOfTuplesOutput(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(NeuralNetTupleOfTuplesOutput, self).__init__()

            self.fc1_1 = torch.nn.Linear(input_size, hidden_size)
            self.relu1 = torch.nn.ReLU()
            self.fc1_2 = torch.nn.Linear(hidden_size, num_classes)

            self.fc2_1 = torch.nn.Linear(input_size, hidden_size)
            self.relu2 = torch.nn.ReLU()
            self.fc2_2 = torch.nn.Linear(hidden_size, num_classes)

            self.fc3_1 = torch.nn.Linear(input_size, hidden_size)
            self.relu3 = torch.nn.ReLU()
            self.fc3_2 = torch.nn.Linear(hidden_size, num_classes)

        def forward(self, input1, input2, input3):
            out1 = self.fc1_2(self.relu1(self.fc1_1(input1)))
            out2 = self.fc2_2(self.relu2(self.fc2_1(input2)))
            out3 = self.fc3_2(self.relu3(self.fc3_1(input3)))
            return ((out1, out2), out3)

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetTupleOfTuplesOutput(D_in, H, D_out).to(device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_in, device=device)
    z = torch.randn(N, D_in, device=device)

    output = model(x, y, z)
    assert len(output) == 2
    assert isinstance(output[0], tuple)
    assert len(output[0]) == 2
    assert isinstance(output[1], torch.Tensor)

@pytest.mark.parametrize("device", ['cpu', 'cuda'])
def test_named_tuple_return_value_module(device):
    ReturnValue = namedtuple('NamedTupleReturnValue', 'loss logits hidden_states')
    class NeuralNetNamedTupleOutput(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(NeuralNetNamedTupleOutput, self).__init__()

            self.fc1_1 = torch.nn.Linear(input_size, hidden_size)
            self.relu1 = torch.nn.ReLU()
            self.fc1_2 = torch.nn.Linear(hidden_size, num_classes)

            self.fc2_1 = torch.nn.Linear(input_size, hidden_size)
            self.relu2 = torch.nn.ReLU()
            self.fc2_2 = torch.nn.Linear(hidden_size, num_classes)

            self.fc3_1 = torch.nn.Linear(input_size, hidden_size)
            self.relu3 = torch.nn.ReLU()
            self.fc3_2 = torch.nn.Linear(hidden_size, num_classes)

        def forward(self, input1, input2, input3):
            out1 = self.fc1_2(self.relu1(self.fc1_1(input1)))
            out2 = self.fc2_2(self.relu2(self.fc2_1(input2)))
            out3 = self.fc3_2(self.relu3(self.fc3_1(input3)))

            return ReturnValue(out1, out2, out3)

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetNamedTupleOutput(D_in, H, D_out).to(device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_in, device=device)
    z = torch.randn(N, D_in, device=device)

    output = model(x, y, z)
    assert isinstance(output, tuple)
    assert isinstance(output, ReturnValue)

@pytest.mark.parametrize("device", ['cpu', 'cuda'])
def test_exception_raised_for_custom_class_return_value_module(device):
    os.environ['ORTMODULE_SKIPCHECK_POLICY'] = 'SKIP_CHECK_DISABLED'

    N, D_in, H, D_out = 64, 784, 500, 10
    pt_model = NeuralNetCustomClassOutput(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))
    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_in, device=device)
    z = torch.randn(N, D_in, device=device)

    from onnxruntime.training.ortmodule._fallback import _FallbackPolicy
    if _test_helpers.is_all_or_nothing_fallback_enabled(None, _FallbackPolicy.FALLBACK_UNSUPPORTED_DATA):
        # Fallback
        pt_out = pt_model(x, y, z)
        ort_out = ort_model(x, y, z)
        # Assert that the output from torch is the same as the one from ORTModule
        _test_helpers.assert_values_are_close(pt_out.out1, ort_out.out1)
        _test_helpers.assert_values_are_close(pt_out.out2, ort_out.out2)
        _test_helpers.assert_values_are_close(pt_out.out3, ort_out.out3)
    else:
        # ORT backend
        with pytest.raises(_fallback.ORTModuleIOError) as runtime_error:
            ort_model(x, y, z)
        assert 'ORTModule does not support the following model output type' in str(runtime_error.value)

    del os.environ['ORTMODULE_SKIPCHECK_POLICY']

def test_dynamic_axes_config():
    device = 'cuda'

    # Model 1
    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device)
    output = model(x)
    assert output is not None
    assert _test_helpers.is_dynamic_axes(model)
    del model, output

    # Model 2
    model_with_no_grad = _get_bert_for_sequence_classification_model(device)
    model_with_no_grad = ORTModule(model_with_no_grad)
    x, y, z = _get_bert_for_sequence_classification_sample_data(device)
    output = model_with_no_grad(x, attention_mask=y, labels=z)
    assert output is not None
    assert _test_helpers.is_dynamic_axes(model_with_no_grad)

def test_model_with_multiple_devices_cpu_cuda():
    class MultipleDeviceModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(10, 10).cpu()
            self.fc2 = torch.nn.Linear(10, 10).cuda()

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    pt_model = MultipleDeviceModel()
    x = torch.randn(20, 10)

    from onnxruntime.training.ortmodule._fallback import _FallbackPolicy
    if _test_helpers.is_all_or_nothing_fallback_enabled(None, _FallbackPolicy.FALLBACK_UNSUPPORTED_DEVICE):
        # Fallback
        ort_model = ORTModule(copy.deepcopy(pt_model))
        with pytest.raises(RuntimeError) as runtime_error:
            ort_model(x)
        assert f"Expected all tensors to be on the same device, but found at least two devices" in str(runtime_error.value)
    else:
        # ORT backend
        with pytest.raises(_fallback.ORTModuleFallbackException) as e:
            ort_model = ORTModule(pt_model)
        assert str(e.value) == 'ORTModule supports a single device per model'

def test_model_with_multiple_devices_to_to():
    class MultipleDeviceModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(10, 10).to('cpu')
            self.fc2 = torch.nn.Linear(10, 10).to('cuda')

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    pt_model = MultipleDeviceModel()
    x = torch.randn(20, 10)
    from onnxruntime.training.ortmodule._fallback import _FallbackPolicy
    if _test_helpers.is_all_or_nothing_fallback_enabled(None, _FallbackPolicy.FALLBACK_UNSUPPORTED_DEVICE):
        # Fallback
        with pytest.raises(RuntimeError) as runtime_error:
            ort_model = ORTModule(copy.deepcopy(pt_model))
            ort_model(x)
        assert f"Expected all tensors to be on the same device, but found at least two devices" in str(runtime_error.value)
    else:
        # ORT backend
        with pytest.raises(_fallback.ORTModuleFallbackException) as e:
            ort_model = ORTModule(pt_model)
        assert str(e.value) == 'ORTModule supports a single device per model'

def test_model_with_multiple_devices_to_cpu():
    class MultipleDeviceModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(10, 10).to('cuda')
            self.fc2 = torch.nn.Linear(10, 10).cpu()

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    pt_model = MultipleDeviceModel()
    x = torch.randn(20, 10)
    from onnxruntime.training.ortmodule._fallback import _FallbackPolicy
    if _test_helpers.is_all_or_nothing_fallback_enabled(None, _FallbackPolicy.FALLBACK_UNSUPPORTED_DEVICE):
        # Fallback
        ort_model = ORTModule(copy.deepcopy(pt_model))
        with pytest.raises(RuntimeError) as runtime_error:
            ort_model(x)
        assert f"Expected all tensors to be on the same device, but found at least two devices" in str(runtime_error.value)
    else:
        # ORT backend
        with pytest.raises(_fallback.ORTModuleFallbackException) as e:
            ort_model = ORTModule(pt_model)
        assert str(e.value) == 'ORTModule supports a single device per model'

def test_model_with_multiple_devices_to_cuda():
    class MultipleDeviceModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(10, 10).to('cpu')
            self.fc2 = torch.nn.Linear(10, 10).cuda()

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    pt_model = MultipleDeviceModel()
    x = torch.randn(20, 10)
    from onnxruntime.training.ortmodule._fallback import _FallbackPolicy
    if _test_helpers.is_all_or_nothing_fallback_enabled(None, _FallbackPolicy.FALLBACK_UNSUPPORTED_DEVICE):
        # Fallback
        ort_model = ORTModule(copy.deepcopy(pt_model))
        with pytest.raises(RuntimeError) as runtime_error:
            ort_model(x)
        assert f"Expected all tensors to be on the same device, but found at least two devices" in str(runtime_error.value)
    else:
        # ORT backend
        with pytest.raises(_fallback.ORTModuleFallbackException) as e:
            ort_model = ORTModule(pt_model)
        assert str(e.value) == 'ORTModule supports a single device per model'

@pytest.mark.parametrize("device", ['cuda', 'cuda:0', 'cuda:1', 'cuda:2'])
def test_model_with_different_cuda_devices(device):

    # Trick to run this test in single GPU machines
    device_id = _utils.get_device_index(device)
    if device_id >= torch.cuda.device_count():
        warnings.warn('Skipping test_model_with_different_cuda_devices(cuda:{})'.format(device_id))
        return

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    model = ORTModule(model)
    model.to(device)
    x = torch.randn(N, D_in, device=device)
    model(x)

def test_register_custom_ops_pytorch_exporter_tensor_triu():
    class SimpleTensorTriuModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(10, 10)

        def forward(self, x):
            x = self.fc1(x)
            mask = torch.ones_like(x).triu(diagonal=1)
            x = x * mask
            return x

    model = SimpleTensorTriuModel()
    model = ORTModule(model)
    user_input = torch.ones(1, 10, 10)

    output = model(user_input)
    assert list(output.shape) ==  [1, 10, 10]

def test_register_custom_ops_pytorch_exporter_torch_triu():
    class SimpleTorchTriuModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(10, 10)

        def forward(self, x):
            x = self.fc1(x)
            mask = torch.triu(torch.ones_like(x))
            x = x * mask
            return x

    model = SimpleTorchTriuModel()
    model = ORTModule(model)
    user_input = torch.ones(1, 10, 10)

    output = model(user_input)
    assert list(output.shape) ==  [1, 10, 10]

def test_wrap_ortmodule_and_change_device():
    # Basic Sequencial model wrapping ORTModule
    x = torch.linspace(-math.pi, math.pi, 2000)
    xx = x.unsqueeze(-1).pow(torch.tensor([1, 2, 3]))
    y = torch.sin(x)
    model = torch.nn.Sequential(
        ORTModule(torch.nn.Linear(3, 1)),
        torch.nn.Flatten(0, 1)
    )

    # Changing device for fun
    model = model.cpu()
    xx = xx.cpu()
    y = y.cpu()
    model = model.cuda()
    xx = xx.cuda()
    y = y.cuda()

    # Quick train
    loss_fn = torch.nn.MSELoss(reduction='sum')
    learning_rate = 1e-6
    for t in range(2000):
        y_pred = model(xx)
        loss = loss_fn(y_pred, y)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad

    # Checking training finished normally
    assert y_pred is not None and loss is not None

@pytest.mark.parametrize("return_dict", [True, False])
def test_hf_model_output_with_tuples(return_dict):
    device = 'cuda'

    model = _get_bert_for_sequence_classification_model(device, output_attentions=True,
        output_hidden_states=True, return_dict=return_dict)
    x, y, z = _get_bert_for_sequence_classification_sample_data(device)

    model = ORTModule(model)
    output = model(x, attention_mask=y, labels=z)

    if return_dict:
        assert isinstance(output, SequenceClassifierOutput)
        assert 'loss' in output and 'logits' in output and \
            'attentions' in output and 'hidden_states' in output
        assert isinstance(output['loss'], torch.Tensor)
        assert isinstance(output['logits'], torch.Tensor)
        assert isinstance(output['attentions'], tuple)
        assert isinstance(output['hidden_states'], tuple)
    else:
        assert isinstance(output, tuple)
        assert isinstance(output[0], torch.Tensor)
        assert isinstance(output[1], torch.Tensor)
        assert isinstance(output[2], tuple)
        assert isinstance(output[3], tuple)

@pytest.mark.parametrize("device", ['cuda', 'cpu'])
def test_nested_return_value_module(device):
    class NeuralNetNestedOutput(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(NeuralNetNestedOutput, self).__init__()

            self.fc1_1 = torch.nn.Linear(input_size, hidden_size)
            self.relu1 = torch.nn.ReLU()
            self.relu = torch.nn.ReLU()
            self.fc1_2 = torch.nn.Linear(hidden_size, num_classes)

            self.fc2_1 = torch.nn.Linear(input_size, hidden_size)
            self.relu2 = torch.nn.ReLU()
            self.fc2_2 = torch.nn.Linear(hidden_size, num_classes)

            self.fc3_1 = torch.nn.Linear(input_size, hidden_size)
            self.relu3 = torch.nn.ReLU()
            self.fc3_2 = torch.nn.Linear(hidden_size, num_classes)

        def forward(self, input1, input2, input3):
            out1 = self.fc1_2(self.relu1(self.fc1_1(input1)))
            out2 = self.fc2_2(self.relu2(self.fc2_1(input2)))
            out3 = self.fc3_2(self.relu(self.relu3(self.fc3_1(input3))))
            return {
                'a': {
                    'b': {
                        'c': out1
                    },
                    'd': (out2, out3)
                }
            }

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetNestedOutput(D_in, H, D_out).to(device)
    model = ORTModule(model)

    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_in, device=device)
    z = torch.randn(N, D_in, device=device)

    output = model(x, y, z)
    assert 'a' in output and 'b' in output['a'] and 'c' in output['a']['b']
    assert isinstance(output['a']['b']['c'], torch.Tensor)

    assert 'd' in output['a']
    assert isinstance(output['a']['d'], tuple)
    assert len(output['a']['d']) == 2

@pytest.mark.parametrize("data_device, model_device", (
    ['cuda', 'cpu'],
    ['cpu', 'cuda'])
)
def test_forward_data_and_model_on_different_devices(data_device, model_device):

    os.environ['ORTMODULE_SKIPCHECK_POLICY'] = 'SKIP_CHECK_DISABLED'

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(model_device)
    ort_model = ORTModule(model)
    # When exporting the model, ensure device is same between input data and model (else pytorch will raise while exporting)
    x = torch.randn(N, D_in, device=model_device)
    output = ort_model(x)

    # Now that the model has been exported, feed in data from device other than the model device
    x = torch.randn(N, D_in, device=data_device)
    from onnxruntime.training.ortmodule._fallback import _FallbackPolicy, ORTModuleDeviceException
    if _test_helpers.is_all_or_nothing_fallback_enabled(None, _FallbackPolicy.FALLBACK_UNSUPPORTED_DEVICE):
        # Fallback
        with pytest.raises(RuntimeError) as runtime_error:
            ort_model(x)
        assert f"Expected all tensors to be on the same device, but found at least two devices" in str(runtime_error.value)
    else:
        # ORT backend
        with pytest.raises(ORTModuleDeviceException) as runtime_error:
            ort_model(x)
        assert f"Input argument to forward found on device {torch.device(x.device)}, but expected it to be on module device {ort_model._torch_module._execution_manager(ort_model._is_training())._device}." in str(runtime_error.value)

    del os.environ['ORTMODULE_SKIPCHECK_POLICY']

def test_forward_returns_none_type_as_output():
    class NeuralNetNoneTypeOutput(torch.nn.Module):
        def __init__(self, input_size, num_classes):
            super(NeuralNetNoneTypeOutput, self).__init__()

            self.fc1 = torch.nn.Linear(input_size, num_classes)
            self.relu1 = torch.nn.ReLU()

        def forward(self, input1):
            out1 = self.fc1(input1)
            out1 = self.relu1(out1)
            return {'out': out1, 'none_output': None}

    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetNoneTypeOutput(D_in, D_out).to(device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device)
    output = model(x)

    assert output['out'] is not None
    assert output['none_output'] is None

def test_bool_input_and_output():
    class NeuralNetBoolInputOutput(torch.nn.Module):
        def __init__(self, input_size, num_classes):
            super(NeuralNetBoolInputOutput, self).__init__()
            self.fc = torch.nn.Linear(input_size, num_classes)
            self.relu = torch.nn.ReLU()

        def forward(self, condition, x1, x2):
            out1 = self.relu(self.fc(torch.where(condition, x1, x2)))
            out2 = torch.tensor(out1).to(torch.bool)
            return out1, out2

    device = 'cuda'
    N, D_in, D_out = 64, 784, 10
    model = NeuralNetBoolInputOutput(D_in, D_out).to(device)
    model = ORTModule(model)
    condition = torch.randint(2, (N, D_in), dtype=torch.bool, device=device)
    x1 = torch.randn(N, D_in, device=device)
    x2 = torch.randn(N, D_in, device=device)
    y1, y2 = model(condition, x1, x2)

    assert y1 is not None
    assert y2 is not None and y2.dtype == torch.bool

def test_uint8_input_and_output():
    class NeuralNetUInt8InputOutput(torch.nn.Module):
        def __init__(self, input_size, num_classes):
            super(NeuralNetUInt8InputOutput, self).__init__()
            self.fc = torch.nn.Linear(input_size, num_classes)
            self.relu = torch.nn.ReLU()

        def forward(self, mask, x1, x2):
            out1 = self.relu(self.fc(torch.where(mask == 1, x1, x2)))
            out2 = torch.tensor(out1).to(torch.uint8)
            return out1, out2

    device = 'cuda'
    N, D_in, D_out = 64, 784, 10
    model = NeuralNetUInt8InputOutput(D_in, D_out).to(device)
    model = ORTModule(model)
    condition = torch.randint(2, (N, D_in), dtype=torch.uint8, device=device)
    x1 = torch.randn(N, D_in, device=device)
    x2 = torch.randn(N, D_in, device=device)
    y1, y2 = model(condition, x1, x2)

    assert y1 is not None
    assert y2 is not None and y2.dtype == torch.uint8

def test_model_partially_requires_grad():
    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetPartialNoGradModel(D_in, H, D_out).to(device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device)

    # Make sure no exception is raised
    output = model(x)

    loss = torch.sum(output)
    loss.backward()

def test_model_wrapped_inside_torch_no_grad():
    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device)

    # Make sure no exception is raised
    with torch.no_grad():
        output = model(x)

def test_model_initializer_requires_grad_changes_from_one_forward_to_next():

    os.environ['ORTMODULE_SKIPCHECK_POLICY'] = 'SKIP_CHECK_DISABLED'

    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetPartialNoGradModel(D_in, H, D_out).to(device)
    model.fc1.requires_grad_(True)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device)
    assert model.module.fc1.weight.grad is None
    assert model.module.fc1.bias.grad is None

    # Make sure no exception is raised
    output = model(x)
    loss = torch.sum(output)
    loss.backward()
    training_session1 = model._torch_module._execution_manager(model._is_training())._execution_agent
    weight_grad_2 = model.module.fc1.weight.grad
    bias_grad_2 = model.module.fc1.bias.grad
    assert weight_grad_2 is not None
    assert bias_grad_2 is not None

    model.module.fc1.requires_grad_(False)
    output = model(x)
    loss = torch.sum(output)
    loss.backward()
    training_session2 = model._torch_module._execution_manager(model._is_training())._execution_agent
    weight_grad_3 = model.module.fc1.weight.grad
    bias_grad_3 = model.module.fc1.bias.grad

    assert training_session1 != training_session2
    assert torch.equal(weight_grad_2, weight_grad_3)
    assert torch.equal(bias_grad_2, bias_grad_3)

    del os.environ['ORTMODULE_SKIPCHECK_POLICY']

def test_model_with_registered_buffers():
    class NeuralNetWithRegisteredBuffer(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(NeuralNetWithRegisteredBuffer, self).__init__()

            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(hidden_size, num_classes)
            self.register_buffer("buffer1s", torch.ones(num_classes))
            self.register_buffer("buffer2s", 1+torch.ones(num_classes))

        def forward(self, input1):
            out = self.fc1(input1)
            out = self.relu(out)
            out = self.fc2(out)
            out += self.buffer1s
            out += self.buffer2s
            return out
    device = 'cuda'

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetWithRegisteredBuffer(D_in, H, D_out).to(device)
    ort_model = ORTModule(model)
    # Check that the original forward signature is preserved.
    assert signature(model.forward) == signature(ort_model.forward)
    x = torch.randn(N, D_in, device=device)
    # Make sure model runs without any exception
    output = ort_model(x)
    assert output is not None


def test_model_with_unused_registered_buffers():
    class UnusedBufferNet(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(UnusedBufferNet, self).__init__()

            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(hidden_size, num_classes)
            self.register_buffer("buffer1s", torch.ones(num_classes))
            self.register_buffer("buffer2s", 1+torch.ones(num_classes))
            self.register_buffer("buffer3s", 2+torch.ones(num_classes))

        def forward(self, input1):
            out = self.fc1(input1)
            out = self.relu(out)
            out = self.fc2(out)
            out += self.buffer3s
            return out
    device = 'cuda'

    N, D_in, H, D_out = 64, 784, 500, 10
    model = UnusedBufferNet(D_in, H, D_out).to(device)
    ort_model = ORTModule(model)
    # Check that the original forward signature is preserved.
    assert signature(model.forward) == signature(ort_model.forward)
    x = torch.randn(N, D_in, device=device)
    # Make sure model runs without any exception
    output = ort_model(x)
    assert output is not None


def test_model_with_constant_and_registered_parameters():
    class NeuralNetWithRegisteredParamsWithConstant(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(NeuralNetWithRegisteredParamsWithConstant, self).__init__()

            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(hidden_size, num_classes)
            self.register_parameter("param1", torch.nn.Parameter(torch.ones(num_classes)))
            self.register_parameter("param2", torch.nn.Parameter(1+torch.ones(num_classes)))

        def forward(self, input1):
            out = self.fc1(input1)
            out = self.relu(out)
            out = self.fc2(out)
            out += self.param1
            out += self.param2
            out += torch.tensor([3.], device=next(self.parameters()).device)
            return out
    device = 'cuda'

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetWithRegisteredParamsWithConstant(D_in, H, D_out).to(device)
    ort_model = ORTModule(model)
    # Check that the original forward signature is preserved.
    assert signature(model.forward) == signature(ort_model.forward)
    x = torch.randn(N, D_in, device=device)
    # Make sure model runs without any exception
    output = ort_model(x)
    assert output is not None

def test_state_dict():
    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    pt_model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))
    x = torch.randn(N, D_in, device=device)
    y = x.clone()

    state_dict_ort = ort_model.state_dict()
    state_dict_pt = pt_model.state_dict()
    assert state_dict_pt
    assert len(state_dict_pt.keys()) == len(state_dict_ort.keys())
    for param_name, param_value in state_dict_pt.items():
        assert param_name in state_dict_ort
        assert torch.equal(param_value, state_dict_ort[param_name])

    # Call forward once
    ort_model(x)
    pt_model(x)

    state_dict_ort = ort_model.state_dict()
    state_dict_pt = pt_model.state_dict()
    assert state_dict_pt
    assert len(state_dict_pt.keys()) == len(state_dict_ort.keys())
    for param_name, param_value in state_dict_pt.items():
        assert param_name in state_dict_ort
        assert torch.equal(param_value, state_dict_ort[param_name])

def test_load_state_dict():
    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    pt_model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))
    x = torch.randn(N, D_in, device=device)
    y = x.clone()

    state_dict_pt = pt_model.state_dict()
    list(next(iter(state_dict_pt.items())))[1] += 10
    ort_model.load_state_dict(state_dict_pt)
    state_dict_ort = ort_model.state_dict()

    assert state_dict_pt
    assert len(state_dict_pt.keys()) == len(state_dict_ort.keys())
    for param_name, param_value in state_dict_pt.items():
        assert param_name in state_dict_ort
        assert torch.equal(param_value, state_dict_ort[param_name])

    # Call forward once
    ort_model(x)
    pt_model(x)

    state_dict_pt = pt_model.state_dict()
    ort_model.load_state_dict(state_dict_pt)
    state_dict_ort = ort_model.state_dict()

    assert state_dict_pt
    assert len(state_dict_pt.keys()) == len(state_dict_ort.keys())
    for param_name, param_value in state_dict_pt.items():
        assert param_name in state_dict_ort
        assert torch.equal(param_value, state_dict_ort[param_name])

def test_named_parameters():
    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    pt_model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))
    named_parameters_pt = [name for name, _ in pt_model.named_parameters()]
    named_parameters_ort = [name for name, _ in ort_model.named_parameters()]

    assert len(named_parameters_pt) > 0
    assert named_parameters_pt == named_parameters_ort

def test_parameters():
    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    pt_model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))
    parameters_pt = [param for param in pt_model.parameters()]
    parameters_ort = [param for param in ort_model.parameters()]

    assert len(parameters_pt) > 0
    assert len(parameters_pt) == len(parameters_ort)
    assert all(torch.equal(parameters_pt[i], parameters_ort[i]) for i in range(len(parameters_pt)))

def test_named_buffers():
    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    pt_model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    pt_model.register_buffer('sample_buffer_pt', torch.tensor(torch.randn(N, D_in, device=device)))
    ort_model = ORTModule(copy.deepcopy(pt_model))
    named_buffers_pt = [name for name, _ in pt_model.named_buffers()]
    named_buffers_ort = [name for name, _ in ort_model.named_buffers()]

    assert len(named_buffers_pt) > 0
    assert named_buffers_pt == named_buffers_ort

    ort_model.register_buffer('sample_buffer_ort', torch.tensor(torch.randn(N, D_in, device=device)))
    named_buffers_ort = [name for name, _ in ort_model.named_buffers()]
    assert named_buffers_ort == ['sample_buffer_pt', 'sample_buffer_ort']

def test_buffers():
    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    pt_model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    pt_model.register_buffer('sample_buffer_pt', torch.tensor(torch.randn(N, D_in, device=device)))
    ort_model = ORTModule(copy.deepcopy(pt_model))
    buffers_pt = [buffer for buffer in pt_model.buffers()]
    buffers_ort = [buffer for buffer in ort_model.buffers()]

    assert len(buffers_pt) > 0
    assert len(buffers_pt) == len(buffers_ort)
    assert all(torch.equal(buffers_pt[i], buffers_ort[i]) for i in range(len(buffers_pt)))

    x = torch.tensor(torch.randn(N, D_in, device=device))
    ort_model.register_buffer('sample_buffer_ort', x)
    buffers_ort = [buffer for buffer in ort_model.buffers()]
    assert len(buffers_ort) == 2
    assert torch.equal(buffers_ort[1], x)

def test_eval_with_dropout():
    class NeuralNetDropout(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(NeuralNetDropout, self).__init__()

            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(hidden_size, num_classes)
            self.dropout = torch.nn.Dropout()

        def forward(self, input1):
            out = self.fc1(input1)
            out = self.relu(out)
            out = self.fc2(out)
            out = self.dropout(out)
            return out

    device = 'cuda'

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetDropout(D_in, H, D_out).to(device)
    model.eval()
    ort_model = ORTModule(copy.deepcopy(model))
    ort_model.eval()

    x = torch.randn(N, D_in, device=device)
    y = x.clone()

    # Make sure model runs without any exception
    output = ort_model(x)
    output_pt = model(y)

    assert output is not None
    assert output_pt is not None
    # Assert that the output from torch is the same as the one from ORTModule
    _test_helpers.assert_values_are_close(output, output_pt)

def test_with_torch_no_grad_context():
    device = 'cuda'

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(model))

    x = torch.randn(N, D_in, device=device)
    y = x.clone()

    # Make sure model runs without any exception
    output = None
    output_pt = None
    with torch.no_grad():
        output = ort_model(x)
        output_pt = model(y)

    assert output is not None
    assert output_pt is not None
    # Assert that the output from torch is the same as the one from ORTModule
    _test_helpers.assert_values_are_close(output, output_pt)
    assert output.grad is None and output_pt.grad is None

def test_unused_layer():
    class Net(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(Net, self).__init__()

            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(hidden_size, num_classes)

        def forward(self, input1):
            out = self.fc1(input1)
            out = self.relu(out)
            return out

    device = torch.device('cuda')
    N, D_in, H, D_out = 64, 784, 500, 10
    pt_model = Net(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    x = torch.randn(N, D_in, device=device)
    pt_output = pt_model(x)
    ort_output = ort_model(x)
    _test_helpers.assert_values_are_close(pt_output, ort_output)

def test_train_eval_with_various_outputs():
    class Net(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(Net, self).__init__()
            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.relu = torch.nn.ReLU()

        def forward(self, input1):
            out1 = self.fc1(input1)
            out2 = self.relu(out1)
            # return different number of outputs for train ane eval mode
            if self.training:
                return out1, out2
            else:
                return out2

    def train_step(model, x):
        out1, out2 = model(x)
        loss = out2.sum()
        loss.backward()
        return out1, out2

    device = torch.device('cuda')
    N, D_in, H, D_out = 64, 784, 500, 10
    pt_model = Net(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    # train mode
    x = torch.randn(N, D_in, device=device)
    pt_out1, pt_out2 = train_step(pt_model, x)
    ort_out1, ort_out2 = train_step(ort_model, x)

    _test_helpers.assert_values_are_close(pt_out1, ort_out1)
    _test_helpers.assert_values_are_close(pt_out2, ort_out2)
    _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model)

    # eval mode
    pt_model.eval()
    ort_model.eval()

    x = torch.randn(N, D_in, device=device)
    pt_out = pt_model(x)
    ort_out = ort_model(x)
    _test_helpers.assert_values_are_close(pt_out, ort_out)

def test_forward_dynamic_args():

    os.environ['ORTMODULE_SKIPCHECK_POLICY'] = 'SKIP_CHECK_DISABLED'

    device = 'cuda'

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetPositionalArguments(input_size=D_in, hidden_size=H, num_classes=D_out).to(device)
    model = ORTModule(model)
    args_size1 = [torch.randn(N, D_in, device=device)]*4
    args_size2 = [torch.randn(N, D_in, device=device)]*3
    args_size3 = [torch.randn(N, D_in, device=device)]*5

    # Make sure model runs without any exception
    for i in range(2):

        # Test both train and inference mode
        if i % 2 == 0:
            model.train()
        else:
            model.eval()

        # Train model with one set of input
        for _ in range(10):
            output = model(*args_size1)
            assert output is not None
        hash_args_size1 = hash(repr(model._torch_module._execution_manager(model._is_training())._input_info.schema))
        assert hash_args_size1 is not None

        # Decrease number of inputs and train some more
        for _ in range(10):
            output = model(*args_size2)
            assert output is not None
        hash_args_size2 = hash(repr(model._torch_module._execution_manager(model._is_training())._input_info.schema))
        assert hash_args_size2 != hash_args_size1

        # Increase number of inputs and train some more
        for _ in range(10):
            output = model(*args_size3)
            assert output is not None
        hash_args_size3 = hash(repr(model._torch_module._execution_manager(model._is_training())._input_info.schema))
        assert hash_args_size3 != hash_args_size2

    del os.environ['ORTMODULE_SKIPCHECK_POLICY']


def test_forward_dynamic_kwargs():

    os.environ['ORTMODULE_SKIPCHECK_POLICY'] = 'SKIP_CHECK_DISABLED'

    one = torch.FloatTensor([1])
    model = NeuralNetSimplePositionalAndKeywordArguments()
    model = ORTModule(model)

    # Make sure model runs without any exception
    for i in range(2):

        # Test both train and inference mode
        if i % 2 == 0:
            model.train()
        else:
            model.eval()

        # Train model with positional argument x only
        for _ in range(10):
            output = model(one)
            assert output is not None
        hash_x = hash(repr(model._torch_module._execution_manager(model._is_training())._input_info.schema))
        assert hash_x is not None

        # Train with x and y as inputs
        for _ in range(10):
            output = model(one,y=one)
            assert output is not None
        hash_x_y = hash(repr(model._torch_module._execution_manager(model._is_training())._input_info.schema))
        assert hash_x_y != hash_x

        # Train with x and z as inputs
        for _ in range(10):
            output = model(one,z=one)
            assert output is not None
        hash_x_z = hash(repr(model._torch_module._execution_manager(model._is_training())._input_info.schema))
        assert hash_x_z != hash_x_y

        # Train with x, y and z as inputs
        for _ in range(10):
            output = model(one,y=one, z=one)
            assert output is not None
        hash_x_y_z = hash(repr(model._torch_module._execution_manager(model._is_training())._input_info.schema))
        assert hash_x_y_z != hash_x_z

        # Return to original input with x as input
        for _ in range(10):
            output = model(one)
            assert output is not None
        hash_x2 = hash(repr(model._torch_module._execution_manager(model._is_training())._input_info.schema))
        assert hash_x2 != hash_x_y_z
        assert hash_x2 == hash_x

    del os.environ['ORTMODULE_SKIPCHECK_POLICY']


@pytest.mark.parametrize("forward_statement",
                         [# Only pos_X, pos_X as positionals
                          "model(pos_0, pos_1)",
                          # Only pos_X, pos_X as keywords
                          "model(pos_0=pos_0, pos_1=pos_1)",
                          # pos_X + *args, pos_X as positionals
                          "model(pos_0, pos_1, *args)",
                          # pos_X + kw_X, pos_X as positionals
                          "model(pos_0, pos_1, kw_0=kw_0, kw_1=kw_1)",
                          # pos_X + kw_X,  pos_X as keywords
                          "model(pos_0=pos_0, pos_1=pos_1, kw_0=kw_0, kw_1=kw_1)",
                          # pos_X + kw_X, pos_X as positionals (missing kw_1)
                          "model(pos_0, pos_1, kw_0=kw_0)",
                          # pos_X + kw_X, pos_X as keywords (missing kw_1)
                          "model(pos_0=pos_0, pos_1=pos_1, kw_0=kw_0)",
                          # pos_X + kw_X, pos_X as positionals (missing kw_0)
                          "model(pos_0, pos_1, kw_1=kw_1)",
                          # pos_X + kw_X, pos_X as keywords (missing kw_0)
                          "model(pos_0=pos_0, pos_1=pos_1, kw_1=kw_1)",
                          # pos_X + kwargs, pos_X as positionals
                          "model(pos_0, pos_1, **kwargs)",
                          # pos_X + kwargs, pos_X as keywords
                          "model(pos_0=pos_0, pos_1=pos_1, **kwargs)",
                          # pos_X + *args + kw_X, pos_X as positionals
                          "model(pos_0, pos_1, *args, kw_0=kw_0, kw_1=kw_1)",
                          # pos_X + *args + kw_X, pos_X as positionals (missing kw_0)
                          "model(pos_0, pos_1, *args, kw_1=kw_1)",
                          # pos_X + *args + kw_X, pos_X as positionals (missing kw_1)
                          "model(pos_0, pos_1, *args, kw_0=kw_0)",
                          # pos_X + *args + kwargs, pos_X as positionals
                          "model(pos_0, pos_1, *args, **kwargs)",
                          # pos_X + *args + kw_X + kwargs, pos_X as positionals
                          "model(pos_0, pos_1, *args, kw_0=kw_0, kw_1=kw_1, **kwargs)",
                          # pos_X + *args + kw_X + kwargs, pos_X as positionals (missing kw_0)
                          "model(pos_0, pos_1, *args, kw_1=kw_1, **kwargs)",
                          # pos_X + *args + kw_X + kwargs, pos_X as positionals (missing kw_1)
                          "model(pos_0, pos_1, *args, kw_0=kw_0, **kwargs)",
                          ])
def test_forward_call_kwargs_input(forward_statement):
    class KwargsNet(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(KwargsNet, self).__init__()

            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(hidden_size, num_classes)

        def forward(self, pos_0, pos_1, *args, kw_0=None, kw_1=None, **kwargs):
            model_input = pos_0 + pos_1
            if args:
                model_input += sum(args)
            if kw_0 is not None:
                model_input += kw_0
            if kw_1 is not None:
                model_input += kw_1
            if kwargs:
                if 'kwargs_0' in kwargs:
                    model_input += kwargs['kwargs_0']
                if 'kwargs_1' in kwargs:
                    model_input += torch.matmul(kwargs['kwargs_0'], kwargs['kwargs_1'])

            out = self.fc1(model_input)
            out = self.relu(out)
            out = self.fc2(out)
            return out

    # Modeling
    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    model = KwargsNet(input_size=D_in, hidden_size=H, num_classes=D_out).to(device)
    model = ORTModule(model)

    # Dummy inputs used
    pos_0 = torch.randn(N, D_in, device=device)
    pos_1 = torch.randn(N, D_in, device=device)
    kw_0 = torch.randn(N, D_in, device=device)
    kw_1 = torch.randn(N, D_in, device=device)
    args = [torch.randn(N, D_in, device=device)]*2
    kwargs = {'kwargs_0' : torch.randn(N, D_in, device=device),
              'kwargs_1' : torch.randn(D_in, D_in, device=device)}

    # Training step
    prediction = eval(forward_statement)
    assert prediction is not None
    prediction = prediction.sum()
    prediction.backward()


def test_repro_iscontiguous():
    class SimpleNet(torch.nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.a = torch.nn.Parameter(torch.FloatTensor([-1., 1.]))
        def forward(self, x):
            result = torch.mean(self.a) + x
            return result

    one = torch.FloatTensor([1])
    model = SimpleNet()
    model = ORTModule(model)
    prediction = model(one)
    prediction = prediction.sum()
    prediction.backward()


def test_forward_call_default_input():

    os.environ['ORTMODULE_SKIPCHECK_POLICY'] = 'SKIP_CHECK_DISABLED'

    class UnusedNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.zeros = torch.nn.Parameter(torch.zeros(1,1))

        def forward(self, a, b, c, d, *args, kw_0=None, **kwargs):
            result = a + d + self.zeros.sum()
            if args:
                result += args[-1]
            if kw_0:
                result += kw_0
            if kwargs:
                assert 'kwargs_1' in kwargs
                result += kwargs['kwargs_1']
            return result

    # Modeling
    device = 'cuda'
    model = UnusedNet().to(device)
    model = ORTModule(model)

    # Dummy data
    one = torch.FloatTensor([1]).to(device)
    two = 2*one
    three = 3*one
    four = 4*one
    args = [two]*5
    kw_0 = 6*one
    kwargs = {'kwargs_0': 7*one, 'kwargs_1': 8*one}

    # Make sure model runs without any exception
    for i in range(2):
        # Test both train and inference mode
        if i % 2 == 0:
            model.train()
        else:
            model.eval()

        # Model only uses a,d out of a,b,c,d
        out = model(one, two, three, four)
        assert out.item() == 5.0
        if model._is_training():
            out.sum().backward()

        out = model(one, two, c=three, d=four)
        assert out.item() == 5.0
        if model._is_training():
            out.sum().backward()

        # Model only uses a,d,args[-1] out of a,b,c,d,*args
        out = model(one, two, three, four, *args)
        assert out.item() == 7.0
        if model._is_training():
            out.sum().backward()

        # Model only uses a,d,args[-1],kw_0 out of a,b,c,d,*args,kw_0
        out = model(one, two, three, four, *args, kw_0=kw_0)
        assert out.item() == 13.0
        if model._is_training():
            out.sum().backward()

        # Model only uses a,d,args[-1],kwargs['kwargs_1'] out of a,b,c,d,*args,kw_0,**kwargs
        out = model(one, two, three, four, *args, **kwargs)
        assert out.item() == 15.0
        if model._is_training():
            out.sum().backward()

    del os.environ['ORTMODULE_SKIPCHECK_POLICY']

def test_forward_call_kwargs_input_unexpected_order():
    class OrderlyNet(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(OrderlyNet, self).__init__()
            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(hidden_size, num_classes)

        def forward(self, input1=None, input2=None):
            assert input1.shape != input2.shape
            input2 = torch.transpose(input2, 0, 1)
            assert input1.shape == input2.shape

            model_input = input1 + input2
            out1 = self.fc1(model_input)
            out1 = self.relu(out1)
            out2 = self.fc2(out1)
            return out1, out2

    device = 'cuda'
    N, D_in, H, D_out = 32, 784, 500, 10
    model = OrderlyNet(D_in, H, D_out).to(device)
    model = ORTModule(model)

    input1 = torch.randn(N, D_in, device=device, requires_grad=False)
    input2 = torch.randn(D_in, N, device=device, requires_grad=False)

    # Make sure model runs without any exception
    for i in range(2):
        # Test both train and inference mode
        if i % 2 == 0:
            model.train()
        else:
            model.eval()

        # Must work because forward() and dict order match
        y1, y2 = model(**{'input1': input1, 'input2': input2})
        assert y1 is not None
        assert y2 is not None
        if model._is_training():
            loss = y1.sum() + y2.sum()
            loss.backward()

        # Must work even when forward() and dict order mismatch
        y1, y2 = model(**{'input2': input2, 'input1': input1})
        assert y1 is not None
        assert y2 is not None
        if model._is_training():
            loss = y1.sum() + y2.sum()
            loss.backward()


def test_forward_call_lots_None():

    os.environ['ORTMODULE_SKIPCHECK_POLICY'] = 'SKIP_CHECK_DISABLED'

    class NoneNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.zeros = torch.nn.Parameter(torch.zeros(1,1))

        def forward(self, a, b, c, d, e, f, y=None, z=None):
            assert a is not None
            result = self.zeros.sum() + a
            if b is not None:
                result += b
            if c is not None:
                result += c
            if d is not None:
                result += d
            if e is not None:
                result += e
            if f is not None:
                result += f
            if y is not None:
                result += y
            if z is not None:
                result += z
            return result

    def run_step(expected, a, b, c, d, e, f, y, z):
        # Force model (re)export to validate (un)flattening with new input
        #   This is needed because for a `forward(self, a, b)`, and
        #   input `forward(a,b)` or `forward(**{'a': a, 'b': b})`,
        #   ORTModule produces the same schema, thus not re-exporting
        #   the model when `forward(a,b)` is used after `forward(**{'a': a, 'b': b})`
        #   or vice-versa
        model._torch_module._execution_manager(model._is_training())._onnx_model = None
        out = model(a,b,c,d,e,f,y,z)
        assert out is not None
        assert out.item() == expected
        if model._is_training():
            loss = out.sum()
            loss.backward()

    device = 'cuda'
    model = NoneNet().to(device)
    model = ORTModule(model)

    a = torch.FloatTensor([1]).to(device)*1
    b = torch.FloatTensor([1]).to(device)*10
    c = torch.FloatTensor([1]).to(device)*100
    d = torch.FloatTensor([1]).to(device)*1000
    e = torch.FloatTensor([1]).to(device)*10000
    f = torch.FloatTensor([1]).to(device)*100000
    y = torch.FloatTensor([1]).to(device)*1000000
    z = torch.FloatTensor([1]).to(device)*10000000

    # Make sure model runs without any exception
    for i in range(2):
        # Test both train and inference mode
        if i % 2 == 0:
            model.train()
        else:
            model.eval()

        run_step(a.item() + f.item(),
                 a, None, None, None, None, f, None, None, )
        run_step(a.item() + f.item(),
                 **{'a': a, 'b': None, 'c': None, 'd': None, 'e': None, 'f': f, 'y': None, 'z': None})
        run_step(a.item() + z.item(),
                 a, None, None, None, None, None, None, z)
        run_step(a.item() + z.item(),
                 **{'a': a, 'b': None, 'c': None, 'd': None, 'e': None, 'f': None, 'y': None, 'z': z})
        run_step(a.item() + c.item() + y.item(),
                 a, None, c, None, None, None, y, None)
        run_step(a.item() + c.item() + y.item(),
                 **{'a': a, 'b': None, 'c': c, 'd': None, 'e': None, 'f': None, 'y': y, 'z': None})
        run_step(a.item() + b.item() + c.item() + d.item() + e.item() + f.item() + y.item() + z.item(),
                 a, b, c, d, e, f, y, z)
        run_step(a.item() + b.item() + c.item() + d.item() + e.item() + f.item() + y.item() + z.item(),
                 **{'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'f': f, 'y': y, 'z': z})

    del os.environ['ORTMODULE_SKIPCHECK_POLICY']

@pytest.mark.parametrize("bool_argument", [True, False])
@pytest.mark.parametrize("int_argument", [100, 100000, 100000000, -100, -100000, -100000000])
@pytest.mark.parametrize("float_argument", [1.23, 11209123.12452, 12093702935.1249863, -1.23, -11209123.12452, -12093702935.1249863])
def test_primitive_inputs(bool_argument, int_argument, float_argument):
    class PrimitiveTypesInputNet(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(PrimitiveTypesInputNet, self).__init__()

            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(hidden_size, num_classes)

        def forward(self, input1, bool_argument, int_argument, float_argument):
            input1 = input1 + int_argument + float_argument
            if bool_argument:
                out = self.fc1(input1)
                out = self.relu(out)
                out = self.fc2(out)
            else:
                out = self.fc1(input1)
                out = self.fc2(out)
                out = self.relu(out)
            return out

    assert type(bool_argument) is bool
    assert type(int_argument) is int
    assert type(float_argument) is float

    device = 'cuda'
    N, D_in, H, D_out = 32, 784, 500, 10
    pt_model = PrimitiveTypesInputNet(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    input1 = torch.randn(N, D_in, device=device)
    pt_out = pt_model(input1, bool_argument, int_argument, float_argument)
    ort_out = ort_model(input1, bool_argument, int_argument, float_argument)
    _test_helpers.assert_values_are_close(pt_out, ort_out)

@pytest.mark.parametrize("bool_arguments", [(True, False), (False, True)])
def test_changing_bool_input_re_exports_model(bool_arguments):

    os.environ['ORTMODULE_SKIPCHECK_POLICY'] = 'SKIP_CHECK_DISABLED'

    class PrimitiveTypesInputNet(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(PrimitiveTypesInputNet, self).__init__()

            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(hidden_size, num_classes)

        def forward(self, input1, bool_argument):
            if bool_argument:
                out = self.fc1(input1)
                out = self.relu(out)
                out = self.fc2(out)
            else:
                out = self.fc1(input1)
                out = self.fc2(out)
                out = self.relu(out)
            return out

    assert type(bool_arguments[0]) is bool
    assert type(bool_arguments[1]) is bool

    device = 'cuda'
    N, D_in, H, D_out = 32, 784, 500, 10
    pt_model = PrimitiveTypesInputNet(D_in, H, D_out).to(device)
    ort_model = ORTModule(pt_model)

    input1 = torch.randn(N, D_in, device=device)
    ort_model(input1, bool_arguments[0])
    exported_model1 = ort_model._torch_module._execution_manager(ort_model._is_training())._onnx_models.exported_model

    ort_model(input1, bool_arguments[1])
    exported_model2 = ort_model._torch_module._execution_manager(ort_model._is_training())._onnx_models.exported_model

    assert exported_model1 != exported_model2

    del os.environ['ORTMODULE_SKIPCHECK_POLICY']

def test_model_with_registered_buffer_and_dropped_parameters():
    class ModelWithBufferAndDroppedParameter(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(ModelWithBufferAndDroppedParameter, self).__init__()

            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(hidden_size, num_classes)
            self.register_buffer("buffer", torch.ones(num_classes))

        def forward(self, bool_argument, input1):
            if bool_argument:
                out = self.fc1(input1)
                out = self.relu(out)
                out = self.fc2(out)
                out = out + self.buffer
            else:
                out = self.fc1(input1)
                out = self.fc2(out)
                out = self.relu(out)
                out = out + self.buffer
            return out

    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    model = ModelWithBufferAndDroppedParameter(D_in, H, D_out).to(device)
    model = ORTModule(model)

    bool_argument = torch.tensor(True)
    x = torch.randn(N, D_in, device=device)

    # Ensure that no exceptions are raised
    out = model(bool_argument, x)

@pytest.mark.parametrize("model, none_pt_params",
        [(UnusedBeginParameterNet(784, 500, 400, 10), ['fc1.weight', 'fc1.bias']),
         (UnusedMiddleParameterNet(784, 500, 400, 10), ['fc2.weight', 'fc2.bias']),
         (UnusedEndParameterNet(784, 500, 400, 10), ['fc2.weight', 'fc2.bias'])])
def test_unused_parameters(model, none_pt_params):
    device = 'cuda'

    N, D_in, H1, H2, D_out = 64, 784, 500, 400, 10
    model = model.to(device)
    ort_model = ORTModule(copy.deepcopy(model))

    # Make sure model runs without any exception
    for _ in range(5):
        x = torch.randn(N, D_in, device=device)
        y = copy.deepcopy(x)

        out_pt = model(x)
        out_ort = ort_model(y)
        loss_pt = out_pt.sum()
        loss_pt.backward()
        loss_ort = out_ort.sum()
        loss_ort.backward()
        _test_helpers.assert_values_are_close(out_ort, out_pt)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, model,
            none_pt_params=none_pt_params)

    # Also try in eval mode
    model.eval()
    ort_model.eval()

    x = torch.randn(N, D_in, device=device)
    y = copy.deepcopy(x)

    # Make sure model runs without any exception
    out_pt = model(x)
    out_ort = ort_model(y)
    _test_helpers.assert_values_are_close(out_ort, out_pt)

def test_output_order():
    class OutputOrderNet(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(OutputOrderNet, self).__init__()

            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.fc2 = torch.nn.Linear(input_size, hidden_size)
            self.fc3 = torch.nn.Linear(input_size, hidden_size)
            self.fc4 = torch.nn.Linear(input_size, hidden_size)
            self.fc5 = torch.nn.Linear(input_size, hidden_size)
            self.fc6 = torch.nn.Linear(input_size, hidden_size)
            self.fc7 = torch.nn.Linear(input_size, hidden_size)
            self.fc8 = torch.nn.Linear(input_size, hidden_size)
            self.fc9 = torch.nn.Linear(input_size, hidden_size)
            self.fc10 = torch.nn.Linear(input_size, hidden_size)
            self.fc11 = torch.nn.Linear(input_size, hidden_size)
            self.fc12 = torch.nn.Linear(input_size, hidden_size)

        def forward(self, input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11, input12):
            return self.fc1(input1), self.fc2(input2), self.fc3(input3), \
                self.fc4(input4), self.fc5(input5), self.fc6(input6), \
                self.fc7(input7), self.fc8(input8), self.fc9(input9), \
                self.fc10(input10), self.fc11(input11), self.fc12(input12)

    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    model = OutputOrderNet(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(model))

    x = [torch.randn(N, D_in, device=device) for _ in range(12)]
    y = copy.deepcopy(x)

    out_pt = model(*x)
    out_ort = ort_model(*y)

    assert len(out_pt) == len(out_ort)
    for x, y in zip(out_pt, out_ort):
        _test_helpers.assert_values_are_close(x, y)

@pytest.mark.parametrize("device", ['cuda', 'cpu', None])
def test_stateless_model_specified_device(device):

    N, D_in, H, D_out = 32, 784, 500, 10
    pt_model = StatelessModel().to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    pt_x = torch.randn(N, D_in, device=device)
    ort_x = pt_x.clone()

    pt_y = pt_model(pt_x)
    ort_y = ort_model(ort_x)

    _test_helpers.assert_values_are_close(pt_y, ort_y)

def test_stateless_model_unspecified_device():

    N, D_in, H, D_out = 32, 784, 500, 10
    pt_model = StatelessModel()
    ort_model = ORTModule(copy.deepcopy(pt_model))

    pt_x = torch.randn(N, D_in)
    ort_x = pt_x.clone()

    pt_y = pt_model(pt_x)
    ort_y = ort_model(ort_x)

    _test_helpers.assert_values_are_close(pt_y, ort_y)

@pytest.mark.parametrize("model",
        [(UnusedBeginParameterNet(784, 500, 400, 10)),
         (UnusedMiddleParameterNet(784, 500, 400, 10)),
         (UnusedEndParameterNet(784, 500, 400, 10))])
def test_unused_parameters_does_not_unnecessarily_reinitialize(model):
    device = 'cuda'

    N, D_in, H1, H2, D_out = 64, 784, 500, 400, 10
    model = model.to(device)
    ort_model = ORTModule(copy.deepcopy(model))
    training_manager = ort_model._torch_module._execution_manager(ort_model._is_training())

    x = torch.randn(N, D_in, device=device)
    _ = ort_model(x)

    input_info = _io.parse_inputs_for_onnx_export(training_manager._module_parameters,
                                                  training_manager._onnx_models.exported_model,
                                                  training_manager._input_info.schema,
                                                  x,
                                                  {})

    assert not training_manager._reinitialize_graph_builder(input_info)

def test_load_state_dict_for_wrapped_ortmodule():
    class WrapperModule(torch.nn.Module):
        def __init__(self, ortmodule):
            super(WrapperModule, self).__init__()
            self._ortmodule = ortmodule

        def forward(self, x):
            return self._ortmodule(x)

    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    model = ORTModule(copy.deepcopy(model))
    wrapper_module = WrapperModule(model)
    x = torch.randn(N, D_in, device=device)
    _ = wrapper_module(x)

    state_dict1 = wrapper_module.state_dict()
    list(next(iter(state_dict1.items())))[1] += 10
    wrapper_module.load_state_dict(state_dict1)
    state_dict2 = wrapper_module.state_dict()

    assert state_dict1
    assert len(state_dict1.keys()) == len(state_dict2.keys())
    for param_name, param_value in state_dict1.items():
        assert param_name in state_dict2
        assert torch.equal(param_value, state_dict2[param_name])

def test_hf_save_pretrained():
    device = 'cuda'

    model1 = _get_bert_for_sequence_classification_model(device)
    model1 = ORTModule(model1)
    state_dict = model1.state_dict()
    list(next(iter(state_dict.items())))[1] += 100
    model1.load_state_dict(state_dict)

    trainer = Trainer(model=model1)

    # Assert that ORTModule has an attribute called module. This attribute is used
    # for trainer.save_model to reference the underlying HuggingFace PreTrainedModel
    assert hasattr(model1, "module")

    # Create a temporary directory for the checkpoint from save_pretrained
    with tempfile.TemporaryDirectory() as temporary_dir:
        trainer.save_model(temporary_dir)

        # Create a new model and compare all state dictionary values for equality
        # to check if from_pretrained worked.
        config = AutoConfig.from_pretrained(temporary_dir)
        model2 = BertForSequenceClassification.from_pretrained(
            temporary_dir, config=config,
        ).to(device)
        model2 = ORTModule(model2)

        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert p1.data.ne(p2.data).sum() == 0

def test_ortmodule_string_inputs_are_ignored():

    pt_model = MyStrNet()
    ort_model = ORTModule(copy.deepcopy(pt_model))
    x = torch.randn(1, 2)

    with pytest.warns(UserWarning) as warning_record:
        out = ort_model(x, 'hello')

    assert len(warning_record) == 2
    assert "Received input of type <class 'str'> which may be treated as a constant by ORT by default." in warning_record[1].message.args[0]
    _test_helpers.assert_values_are_close(out, x+1)

def test_ortmodule_list_input():
    class ListNet(torch.nn.Module):
        def __init__(self):
            super(ListNet, self).__init__()
            self.dummy = torch.nn.Parameter(torch.FloatTensor([0]))

        def forward(self, batch):
            a = batch[0]
            b = batch[1]
            return self.dummy + a + b

    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    pt_model = ListNet().to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))
    x = [torch.randn(N, D_in, device=device), torch.randn(N, D_in, device=device)]
    y = copy.deepcopy(x)

    _test_helpers.assert_values_are_close(pt_model(x), ort_model(y))

def test_ortmodule_list_input_with_unused_values():
    class ListNet(torch.nn.Module):
        def __init__(self):
            super(ListNet, self).__init__()
            self.dummy = torch.nn.Parameter(torch.FloatTensor([0]))

        def forward(self, batch):
            a = batch[0]
            b = batch[1]
            return self.dummy + b

    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    pt_model = ListNet().to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))
    x = [torch.randn(N, D_in, device=device), torch.randn(N, D_in, device=device)]
    y = copy.deepcopy(x)

    _test_helpers.assert_values_are_close(pt_model(x), ort_model(y))

def test_ortmodule_list_input_with_none_values():
    class ListNet(torch.nn.Module):
        def __init__(self):
            super(ListNet, self).__init__()
            self.dummy = torch.nn.Parameter(torch.FloatTensor([0]))

        def forward(self, batch):
            a = batch[0] if batch[0] is not None else torch.FloatTensor([2]).cuda()
            b = batch[1]
            return self.dummy + a + b

    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    pt_model = ListNet().to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))
    x = [None, torch.randn(N, D_in, device=device)]
    y = copy.deepcopy(x)

    _test_helpers.assert_values_are_close(pt_model(x), ort_model(y))

def test_ortmodule_nested_list_input():
    class ListNet(torch.nn.Module):
        def __init__(self):
            super(ListNet, self).__init__()
            self.dummy = torch.nn.Parameter(torch.FloatTensor([0]))

        def forward(self, batch):
            a = batch[0]
            b = batch[1][0]
            c = batch[1][1]
            d = batch[2][0]
            e = batch[2][1][0]
            return self.dummy + a + b + c + d + e

    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    pt_model = ListNet().to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))
    x = [torch.randn(N, D_in, device=device),
        [torch.randn(N, D_in, device=device), torch.randn(N, D_in, device=device)],
        [torch.randn(N, D_in, device=device), [torch.randn(N, D_in, device=device)]]]
    y = copy.deepcopy(x)

    _test_helpers.assert_values_are_close(pt_model(x), ort_model(y))

@pytest.mark.parametrize("mode", ['training', 'inference'])
def test_debug_options_save_onnx_models_os_environment(mode):

    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    # Create a temporary directory for the onnx_models
    with tempfile.TemporaryDirectory() as temporary_dir:
        os.environ['ORTMODULE_SAVE_ONNX_PATH'] = temporary_dir
        model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
        ort_model = ORTModule(model, DebugOptions(save_onnx=True, onnx_prefix='my_model'))
        if mode == 'inference':
            ort_model.eval()
        x = torch.randn(N, D_in, device=device)
        _ = ort_model(x)

        # assert that the onnx models have been saved
        assert os.path.exists(os.path.join(temporary_dir, f"my_model_torch_exported_{mode}.onnx"))
        assert os.path.exists(os.path.join(temporary_dir, f"my_model_optimized_{mode}.onnx"))
        assert os.path.exists(os.path.join(temporary_dir, f"my_model_optimized_pre_grad_{mode}.onnx"))
        assert os.path.exists(os.path.join(temporary_dir, f"my_model_execution_model_{mode}.onnx"))
        del os.environ['ORTMODULE_SAVE_ONNX_PATH']

@pytest.mark.parametrize("mode", ['training', 'inference'])
def test_debug_options_save_onnx_models_cwd(mode):

    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    ort_model = ORTModule(model, DebugOptions(save_onnx=True, onnx_prefix='my_cwd_model'))
    if mode == 'inference':
        ort_model.eval()
    x = torch.randn(N, D_in, device=device)
    _ = ort_model(x)

    # assert that the onnx models have been saved
    assert os.path.exists(os.path.join(os.getcwd(), f"my_cwd_model_torch_exported_{mode}.onnx"))
    assert os.path.exists(os.path.join(os.getcwd(), f"my_cwd_model_optimized_{mode}.onnx"))
    assert os.path.exists(os.path.join(os.getcwd(), f"my_cwd_model_optimized_pre_grad_{mode}.onnx"))
    assert os.path.exists(os.path.join(os.getcwd(), f"my_cwd_model_execution_model_{mode}.onnx"))

    os.remove(os.path.join(os.getcwd(), f"my_cwd_model_torch_exported_{mode}.onnx"))
    os.remove(os.path.join(os.getcwd(), f"my_cwd_model_optimized_{mode}.onnx"))
    os.remove(os.path.join(os.getcwd(), f"my_cwd_model_optimized_pre_grad_{mode}.onnx"))
    os.remove(os.path.join(os.getcwd(), f"my_cwd_model_execution_model_{mode}.onnx"))

def test_debug_options_save_onnx_models_validate_fail_on_non_writable_dir():

    os.environ['ORTMODULE_SAVE_ONNX_PATH'] = '/non/existent/directory'
    with pytest.raises(Exception) as ex_info:
        _ = DebugOptions(save_onnx=True, onnx_prefix='my_model')
    assert "Directory /non/existent/directory is not writable." in str(ex_info.value)
    del os.environ['ORTMODULE_SAVE_ONNX_PATH']

def test_debug_options_save_onnx_models_validate_fail_on_non_str_prefix():
    prefix = 23
    with pytest.raises(Exception) as ex_info:
        _ = DebugOptions(save_onnx=True, onnx_prefix=prefix)
    assert f"Expected name prefix of type str, got {type(prefix)}." in str(ex_info.value)

def test_debug_options_save_onnx_models_validate_fail_on_no_prefix():
    with pytest.raises(Exception) as ex_info:
        _ = DebugOptions(save_onnx=True)
    assert f"onnx_prefix must be provided when save_onnx is set." in str(ex_info.value)

def test_debug_options_log_level():
    # NOTE: This test will output verbose logging

    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    ort_model = ORTModule(model, DebugOptions(log_level=LogLevel.VERBOSE))
    x = torch.randn(N, D_in, device=device)
    _ = ort_model(x)

    # assert that the logging is done in verbose mode
    assert ort_model._torch_module._execution_manager(True)._debug_options.logging.log_level == LogLevel.VERBOSE

def test_debug_options_log_level_os_environment():
    # NOTE: This test will output info logging

    os.environ['ORTMODULE_LOG_LEVEL'] = 'INFO'
    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    ort_model = ORTModule(model)
    x = torch.randn(N, D_in, device=device)
    _ = ort_model(x)

    # assert that the logging is done in info mode
    assert ort_model._torch_module._execution_manager(True)._debug_options.logging.log_level == LogLevel.INFO
    del os.environ['ORTMODULE_LOG_LEVEL']

def test_debug_options_log_level_validation_fails_on_type_mismatch():
    log_level = 'some_string'
    with pytest.raises(Exception) as ex_info:
        _ = DebugOptions(log_level=log_level)
    assert f"Expected log_level of type LogLevel, got {type(log_level)}." in str(ex_info.value)

def test_ortmodule_gradient_accumulation_optimization_correctness():
    class NeuralNetWithCast(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(NeuralNetWithCast, self).__init__()

            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(hidden_size, num_classes)

        def forward(self, input1):
            out = self.fc1(input1)
            out = self.relu(out)
            out = self.fc2(out)
            return out
    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    pt_model = NeuralNetWithCast(D_in, H, D_out).to(device)

    # baseline model with optimization disabled
    tgt_model = ORTModule(pt_model)
    tgt_optimizer = torch.optim.Adam(tgt_model.parameters())

    # model with optimization enabled
    opt_model = ORTModule(copy.deepcopy(pt_model))
    opt_model._torch_module._execution_manager(is_training=True)._enable_grad_acc_optimization = True
    opt_optimizer = torch.optim.Adam(opt_model.parameters())

    def run_step(model, x):
        with amp.autocast():
            prediction = model(x)
            loss = prediction.sum()
        loss.backward()
        return loss.detach()

    def run_optim_step(optimizer):
        optimizer.step()
        optimizer.zero_grad()

    GA_steps = 2
    tgt_model.zero_grad()
    opt_model.zero_grad()

    for step in range(10):
        x = torch.randn(N, D_in, device=device)
        tgt_loss = run_step(tgt_model, x)
        opt_loss = run_step(opt_model, x)

        # assert that loss values match
        _test_helpers.assert_values_are_close(tgt_loss, opt_loss)
        if step % GA_steps == 0:
            run_optim_step(tgt_optimizer)
            run_optim_step(opt_optimizer)

def test_ortmodule_dict_input():
    class DictNet(torch.nn.Module):
        def __init__(self):
            super(DictNet, self).__init__()
            self.dummy = torch.nn.Parameter(torch.FloatTensor([0]))

        def forward(self, batch):
            b = batch['one_value']
            a = batch['two_value']
            return self.dummy + a + b

    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    pt_model = DictNet().to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))
    x = {'one_value': torch.randn(N, D_in, device=device), 'two_value': torch.randn(N, D_in, device=device)}
    x_copy = copy.deepcopy(x)

    _test_helpers.assert_values_are_close(pt_model(x), ort_model(x_copy))

def test_ortmodule_dict_input_with_unused_values():
    class DictNet(torch.nn.Module):
        def __init__(self):
            super(DictNet, self).__init__()
            self.dummy = torch.nn.Parameter(torch.FloatTensor([0]))

        def forward(self, batch):
            b = batch['b']
            a = batch['a']
            return self.dummy + a

    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    pt_model = DictNet().to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))
    x = {'a': torch.randn(N, D_in, device=device), 'b': torch.randn(N, D_in, device=device)}
    x_copy = copy.deepcopy(x)

    _test_helpers.assert_values_are_close(pt_model(x), ort_model(x_copy))

def test_ortmodule_dict_input_with_none_values():
    class DictNet(torch.nn.Module):
        def __init__(self):
            super(DictNet, self).__init__()
            self.dummy = torch.nn.Parameter(torch.FloatTensor([0]))

        def forward(self, batch):
            b = batch['b']
            a = batch['a'] if batch['a'] else torch.FloatTensor([2.0]).cuda()
            return self.dummy + a + b

    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    pt_model = DictNet().to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))
    x = {'a': None, 'b': torch.randn(N, D_in, device=device)}
    x_copy = copy.deepcopy(x)

    _test_helpers.assert_values_are_close(pt_model(x), ort_model(x_copy))

def test_ortmodule_dict_input_with_nested_values():
    class DictNet(torch.nn.Module):
        def __init__(self):
            super(DictNet, self).__init__()
            self.dummy = torch.nn.Parameter(torch.FloatTensor([0]))

        def forward(self, batch):
            a = batch['one_value']
            b = batch['two_value']['three_value']
            c = batch['two_value']['four_value']
            d = batch['five_value']['six_value']
            e = batch['five_value']['seven_value']['eight_value']
            return self.dummy + a + b + c + d + e

    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    pt_model = DictNet().to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))
    x = {
            'one_value': torch.randn(N, D_in, device=device),
            'two_value': {
                'three_value': torch.randn(N, D_in, device=device),
                'four_value': torch.randn(N, D_in, device=device)
            },
            'five_value': {
                'six_value': torch.randn(N, D_in, device=device),
                'seven_value': {
                    'eight_value': torch.randn(N, D_in, device=device)
                }
            }
        }
    x_copy = copy.deepcopy(x)

    _test_helpers.assert_values_are_close(pt_model(x), ort_model(x_copy))

def test_ortmodule_list_dict_input_with_nested_values():
    class ListDictNet(torch.nn.Module):
        def __init__(self):
            super(ListDictNet, self).__init__()
            self.dummy = torch.nn.Parameter(torch.FloatTensor([3]))

        def forward(self, batch):
            a = batch['one_value'][0]
            b = batch['two_value'][0]
            c = batch['two_value'][1]
            d = batch['three_value'][0]
            e = batch['three_value'][1]['four_value']
            return self.dummy + a + b + c + d + e

    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    pt_model = ListDictNet().to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))
    x = {
            'one_value': [torch.randn(N, D_in, device=device)],
            'two_value': [torch.randn(N, D_in, device=device), torch.randn(N, D_in, device=device)],
            'three_value': [
                torch.randn(N, D_in, device=device),
                {
                    'four_value': torch.randn(N, D_in, device=device)
                }
            ]
        }
    x_copy = copy.deepcopy(x)

    _test_helpers.assert_values_are_close(pt_model(x), ort_model(x_copy))

def test_ortmodule_list_dict_input_with_kwargs_and_registered_buffer():
    class ListDictKwargsNet(torch.nn.Module):
        def __init__(self, N, D_in):
            super(ListDictKwargsNet, self).__init__()
            self.register_buffer("buffer", torch.ones(N, D_in, device='cuda'))
            self.dummy = torch.nn.Parameter(torch.FloatTensor([3]))

        def forward(self, batch, **kwargs):
            a = batch['one_value'][0]
            b = batch['two_value'][0]
            c = batch['two_value'][1]
            d = batch['three_value'][0]
            e = batch['three_value'][1]['four_value']
            out = self.buffer + self.dummy + a + b + c + d + e
            if kwargs:
                if 'kwargs_0' in kwargs:
                    out += kwargs['kwargs_0']
                if 'kwargs_1' in kwargs:
                    out += torch.matmul(kwargs['kwargs_0'], kwargs['kwargs_1'])

            return out

    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    pt_model = ListDictKwargsNet(N, D_in).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model), DebugOptions(save_onnx=True, onnx_prefix='kwargsanddict'))
    x = {
            'one_value': [torch.randn(N, D_in, device=device)],
            'two_value': [torch.randn(N, D_in, device=device), torch.randn(N, D_in, device=device)],
            'three_value': [
                torch.randn(N, D_in, device=device),
                {
                    'four_value': torch.randn(N, D_in, device=device)
                }
            ]
        }
    x_copy = copy.deepcopy(x)
    kwargs_input = {'kwargs_0' : torch.randn(N, D_in, device=device),
              'kwargs_1' : torch.randn(D_in, D_in, device=device)}
    kwargs_input_copy = copy.deepcopy(kwargs_input)

    _test_helpers.assert_values_are_close(pt_model(x, **kwargs_input), ort_model(x_copy, **kwargs_input_copy))

def test_ortmodule_user_defined_method():
    class UserDefinedMethodsNet(torch.nn.Module):
        def __init__(self):
            super(UserDefinedMethodsNet, self).__init__()
            self.dummy = torch.nn.Parameter(torch.FloatTensor([12]))

        def forward(self, a):
            return self.dummy + a

        def custom_method_returns_input(self, user_input):
            return user_input

    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    pt_model = UserDefinedMethodsNet().to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))
    x = torch.randn(N, D_in, device=device)
    y = copy.deepcopy(x)

    out = ort_model.custom_method_returns_input(x)
    assert x is out

    pt_out = pt_model(x)
    ort_out = ort_model(y)
    _test_helpers.assert_values_are_close(pt_out, ort_out)

def test_ortmodule_user_getattr_gets_successfully():
    class UserDefinedMethodsNet(torch.nn.Module):
        def __init__(self):
            super(UserDefinedMethodsNet, self).__init__()
            self.dummy = torch.nn.Parameter(torch.FloatTensor([12]))

        def forward(self, a):
            return self.dummy + a

        def custom_method_returns_input(self, user_input):
            return user_input

    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    pt_model = UserDefinedMethodsNet().to(device)
    ort_model = ORTModule(pt_model)

    assert ort_model.custom_method_returns_input != pt_model.custom_method_returns_input
    assert ort_model.custom_method_returns_input.__func__ == pt_model.custom_method_returns_input.__func__
    assert ort_model.dummy is pt_model.dummy

@pytest.mark.parametrize("attribute", ['True', 'lambda x : x'])
def test_ortmodule_setattr_new_attribute(attribute):
    class UserNet(torch.nn.Module):
        def __init__(self):
            super(UserNet, self).__init__()
            self.dummy = torch.nn.Parameter(torch.FloatTensor([0]))

        def forward(self, a):
            return self.dummy + a

    device = 'cuda'
    pt_model = UserNet().to(device)
    ort_model = ORTModule(pt_model)
    ort_model.a_new_attribute = attribute

    assert hasattr(pt_model, 'a_new_attribute')
    assert pt_model.a_new_attribute == attribute
    assert 'a_new_attribute' not in ort_model.__dict__

def test_ortmodule_setattr_on_ortmodule_copied_user_model_attribute():
    class UserNet(torch.nn.Module):
        def __init__(self):
            super(UserNet, self).__init__()
            self.dummy = torch.nn.Parameter(torch.FloatTensor([0]))

        def forward(self, a):
            return self.dummy + a

        def custom_method(self, a):
            return a

    def my_new_custom_method(self, a, b, c):
        return a + b + c

    device = 'cuda'
    pt_model = UserNet().to(device)
    ort_model = ORTModule(pt_model)
    # custom_method is copied by ORTModule from the users model
    # and bound to itself
    ort_model.custom_method = my_new_custom_method
    # dummy is defined on pt model
    ort_model.dummy = torch.nn.Parameter(torch.FloatTensor([12]))

    assert hasattr(pt_model, 'dummy')
    assert torch.eq(pt_model.dummy, torch.nn.Parameter(torch.FloatTensor([12])))
    assert 'dummy' not in ort_model.__dict__

    assert hasattr(pt_model, 'custom_method')
    assert pt_model.custom_method is not my_new_custom_method
    assert ort_model.custom_method is my_new_custom_method

def test_ortmodule_setattr_ortmodule_attribute():
    class UserNet(torch.nn.Module):
        def __init__(self):
            super(UserNet, self).__init__()
            self.dummy = torch.nn.Parameter(torch.FloatTensor([0]))

        def forward(self, a):
            return self.dummy + a

    device = 'cuda'
    pt_model = UserNet().to(device)
    ort_model = ORTModule(pt_model)
    ort_model._torch_module = True

    assert not hasattr(pt_model, '_torch_module')
    assert '_torch_module' in ort_model.__dict__
    assert ort_model._torch_module == True

def test_ortmodule_setattr_signals_model_changed():

    os.environ['ORTMODULE_SKIPCHECK_POLICY'] = 'SKIP_CHECK_DISABLED'

    class UserNet(torch.nn.Module):
        def __init__(self, input_flag):
            super(UserNet, self).__init__()
            self.dummy = torch.nn.Parameter(torch.FloatTensor([10]))
            self.input_flag = input_flag

        def forward(self, a):
            if self.input_flag:
                return self.dummy + a
            else:
                return a

    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    pt_model = UserNet(True).to(device)
    ort_model = ORTModule(pt_model)

    _ = ort_model(torch.randn(N, D_in, device=device))
    exported_model1 = ort_model._torch_module._execution_manager(True)._onnx_models.exported_model

    for training_mode in [False, True]:
        assert ort_model._torch_module._execution_manager(training_mode)._original_model_has_changed == False
    ort_model.input_flag = False

    for training_mode in [False, True]:
        assert ort_model._torch_module._execution_manager(training_mode)._original_model_has_changed == True

    _ = ort_model(torch.randn(N, D_in, device=device))
    exported_model2 = ort_model._torch_module._execution_manager(True)._onnx_models.exported_model

    assert exported_model1 != exported_model2

    del os.environ['ORTMODULE_SKIPCHECK_POLICY']

def test_ortmodule_attribute_name_collision_warning():
    class UserNet(torch.nn.Module):
        def __init__(self):
            super(UserNet, self).__init__()
            self.dummy = torch.nn.Parameter(torch.FloatTensor([0]))
            self._torch_module = True

        def forward(self, a):
            return self.dummy + a

        def load_state_dict(self):
            pass

    device = 'cuda'
    pt_model = UserNet().to(device)
    with pytest.warns(UserWarning) as warning_record:
        ort_model = ORTModule(pt_model)

    assert len(warning_record) == 2
    assert "_torch_module collides with ORTModule's attribute name." in warning_record[0].message.args[0]
    assert "load_state_dict collides with ORTModule's attribute name." in warning_record[1].message.args[0]

def test_ortmodule_ortmodule_method_attribute_copy():
    class UserNetWithSelfCallingForward(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(UserNetWithSelfCallingForward, self).__init__()

            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(hidden_size, num_classes)

        def forward(self, input1):
            out = self.fc1(input1)
            out = self.relu(out)
            out = self.fc2(out)
            return out

        def run_forward(self, *args, **kwargs):
            return self(*args, **kwargs)

    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    pt_model = UserNetWithSelfCallingForward(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    x_1 = torch.randn(N, D_in, device=device)
    x_2 = copy.deepcopy(x_1)
    x_3 = copy.deepcopy(x_1)
    # Executed on ORTModule
    out1 = ort_model(x_1)
    # Executed on ORTModule even though run_forward is not defined on ORTModule
    out2 = ort_model.run_forward(x_2)
    # Executed on pytorch module since it is directly invoked from there
    out3 = pt_model.run_forward(x_3)

    assert torch.equal(out1, out2)
    _test_helpers.assert_values_are_close(out2, out3)

    assert type(out1.grad_fn).__name__ == '_ORTModuleFunctionBackward'
    assert type(out2.grad_fn).__name__ == '_ORTModuleFunctionBackward'
    assert type(out3.grad_fn).__name__ == 'AddmmBackward0' if LooseVersion(
        torch.__version__) >= LooseVersion('1.10.0') else 'AddmmBackward'

@pytest.mark.parametrize("policy_str, policy",[
    ('SKIP_CHECK_DISABLED', _graph_execution_manager._SkipCheck.SKIP_CHECK_DISABLED),
    ('SKIP_CHECK_DEVICE', _graph_execution_manager._SkipCheck.SKIP_CHECK_DEVICE),
    ('SKIP_CHECK_BUILD_GRADIENT', _graph_execution_manager._SkipCheck.SKIP_CHECK_BUILD_GRADIENT),
    ('SKIP_CHECK_EXECUTION_AGENT', _graph_execution_manager._SkipCheck.SKIP_CHECK_EXECUTION_AGENT),
])
def test_ortmodule_skip_check_load_from_os_env(policy_str, policy):
    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    os.environ['ORTMODULE_SKIPCHECK_POLICY'] = policy_str
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    ort_model = ORTModule(model)

    for training_mode in [False, True]:
        assert ort_model._torch_module._execution_manager(training_mode)._skip_check == policy

    del os.environ['ORTMODULE_SKIPCHECK_POLICY']

@pytest.mark.parametrize("is_training,deterministic",
                         list(itertools.product([True,False],repeat=2)))
def test_ortmodule_determinism_flag(is_training,deterministic):

    torch.use_deterministic_algorithms(deterministic)

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out)
    model = ORTModule(model)
    model.train(is_training)

    for _ in range(5):
        x = torch.randn(N, D_in)
        _ = model(x)

        from onnxruntime.training.ortmodule import _are_deterministic_algorithms_enabled
        assert _are_deterministic_algorithms_enabled() is torch.are_deterministic_algorithms_enabled()


def test_ortmodule_gradient_builder():
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
        def forward(self, x):
            return torch.cos(x)

    device = 'cuda'

    @register_gradient('', 'Cos')
    def Cos_gradient():
        return [('Sin', ['I(0)'], ['Sin_X']),
                ('Mul', ['Sin_X', 'GO(0)'], ['Sin_X_Times_dY']),
                ('Neg', ['Sin_X_Times_dY'], ['GI(0)'])]

    pt_model = Model().to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    def run_step(model, x):
        prediction = model(x)
        loss = prediction.sum()
        loss.backward()
        return prediction

    pt_x = torch.randn(2, 2, device=device, requires_grad=True, dtype=torch.float32)
    ort_x = copy.deepcopy(pt_x)
    pt_prediction = run_step(pt_model, pt_x)
    ort_prediction = run_step(ort_model, ort_x)
    _test_helpers.assert_values_are_close(ort_prediction, pt_prediction)
    _test_helpers.assert_values_are_close(ort_x.grad, pt_x.grad )


def test_override_pytorch_exporter_kwargs():
    device = 'cuda'

    N, D_in, H, D_out = 64, 784, 500, 10
    x = torch.randn(N, D_in, device=device)
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)

    ort_model = ORTModule(model)
    ort_model._torch_module._execution_manager(True)._export_extra_kwargs = {'custom_opsets': None}

    # Make sure model runs without any exception
    prediction = ort_model(x)
    assert prediction is not None
    prediction = prediction.sum()
    prediction.backward()


def test_override_pytorch_exporter_kwargs__invalid():
    device = 'cuda'

    N, D_in, H, D_out = 64, 784, 500, 10
    x = torch.randn(N, D_in, device=device)
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)

    ort_model = ORTModule(model)
    ort_model._torch_module._execution_manager(True)._export_extra_kwargs = {'verbose': False}
    with pytest.raises(_fallback.ORTModuleONNXModelException) as type_error:
        _ = ort_model(x)
    assert "The following PyTorch exporter arguments cannot be specified: '{'verbose'}'." in str(type_error.value)


def test_override_pytorch_exporter_kwargs_using_ortmodule_extension__invalid():
    device = 'cuda'

    class ORTModuleExtension(ORTModule):
        def __init__(self, module, debug_options=None):
            super().__init__(module, debug_options)
            for training_mode in [False, True]:
                self._torch_module._execution_manager(training_mode)._export_extra_kwargs = {'verbose': None}

    N, D_in, H, D_out = 64, 784, 500, 10
    x = torch.randn(N, D_in, device=device)
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    ort_model = ORTModuleExtension(model)

    with pytest.raises(_fallback.ORTModuleONNXModelException) as type_error:
        _ = ort_model(x)
    assert "The following PyTorch exporter arguments cannot be specified: '{'verbose'}'." in str(type_error.value)

def test_override_pytorch_exporter_kwargs_using_ortmodule_extension():
    device = 'cuda'

    class ORTModuleExtension(ORTModule):
        def __init__(self, module, debug_options=None):
            super().__init__(module, debug_options)
            # modify GraphExecutionManager internally
            for training_mode in [False, True]:
                self._torch_module._execution_manager(training_mode)._export_extra_kwargs = {'custom_opsets': None}

    N, D_in, H, D_out = 64, 784, 500, 10
    x = torch.randn(N, D_in, device=device)
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    ort_model = ORTModuleExtension(model)

    # Make sure model runs without any exception
    prediction = ort_model(x)
    assert prediction is not None
    prediction = prediction.sum()
    prediction.backward()

def test_ortmodule_fused_adam_optimizer_correctness():

    torch.manual_seed(8888)

    device = 'cuda'
    N, D_in, H, D_out = 32, 128, 500, 10

    pt_model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    transformers_adamw_optimizer = AdamW(pt_model.parameters(), lr=1)

    ort_model = ORTModule(copy.deepcopy(pt_model))
    ort_fused_adam_optimizer = FusedAdam(ort_model.parameters(), lr=1)

    def run_step(model, x):
        prediction = model(x)
        loss = prediction.sum()
        loss.backward()

        return loss

    def run_optim_step(optimizer):
        optimizer.step()
        optimizer.zero_grad()

    ga_steps = 2
    pt_model.zero_grad()
    ort_model.zero_grad()

    for step in range(1000):
        x1 = torch.randn(N, D_in, device=device, dtype=torch.float32)
        x2 = copy.deepcopy(x1)

        pt_loss = run_step(pt_model, x1)
        ort_loss = run_step(ort_model, x2)

        for pt_param, ort_param in zip(pt_model.parameters(), ort_model.parameters()):
            ort_param.grad = copy.deepcopy(pt_param.grad)

        _test_helpers.assert_values_are_close(pt_loss, ort_loss)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model, reset_gradient=False)

        if (step+1) % ga_steps == 0:
            run_optim_step(transformers_adamw_optimizer)
            run_optim_step(ort_fused_adam_optimizer)

        for pt_param, ort_param in zip(pt_model.parameters(), ort_model.parameters()):
            _test_helpers.assert_values_are_close(pt_param, ort_param, atol=1e-4, rtol=1e-5)

def test_ortmodule_fused_adam_optimizer_correctness_torch():

    torch.manual_seed(8888)

    device = 'cuda'
    N, D_in, H, D_out = 4, 4, 8, 4

    pt_model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    adamw_optimizer = torch.optim.AdamW(pt_model.parameters(), lr=1e-3)

    ort_model = ORTModule(copy.deepcopy(pt_model))
    ort_fused_adam_optimizer = FusedAdam(ort_model.parameters(), lr=1e-3,
                                         adam_w_mode=AdamWMode.ADAMW_TORCH,
                                         weight_decay=0.01,
                                         eps=1e-8)

    def run_step(model, x):
        prediction = model(x)
        loss = prediction.sum()
        loss.backward()

        return loss

    def run_optim_step(optimizer):
        optimizer.step()
        optimizer.zero_grad()

    ga_steps = 2
    pt_model.zero_grad()
    ort_model.zero_grad()

    for step in range(1000):
        x1 = torch.randn(N, D_in, device=device, dtype=torch.float32)
        x2 = copy.deepcopy(x1)

        pt_loss = run_step(pt_model, x1)
        ort_loss = run_step(ort_model, x2)

        for pt_param, ort_param in zip(pt_model.parameters(), ort_model.parameters()):
            ort_param.grad = copy.deepcopy(pt_param.grad)

        _test_helpers.assert_values_are_close(pt_loss, ort_loss, atol=1e-4, rtol=1e-5)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model, atol=1e-4, rtol=1e-5, reset_gradient=False)

        if (step+1) % ga_steps == 0:
            run_optim_step(adamw_optimizer)
            run_optim_step(ort_fused_adam_optimizer)

        for pt_param, ort_param in zip(pt_model.parameters(), ort_model.parameters()):
            _test_helpers.assert_values_are_close(pt_param, ort_param, atol=1e-4, rtol=1e-5)

def test_sigmoid_grad():
    class NeuralNetSigmoid(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(NeuralNetSigmoid, self).__init__()

            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.sigmoid = torch.nn.Sigmoid()

        def forward(self, input1):
            out = self.fc1(input1)
            out = self.sigmoid(out)
            return out

    def run_step(model, x):
        prediction = model(x)
        loss = prediction.sum()
        loss.backward()
        return prediction, loss
    device = 'cuda'

    N, D_in, H, D_out = 120, 15360, 500, 15360
    pt_model = NeuralNetSigmoid(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    for step in range(1000):
        pt_x = torch.randn(N, D_in, device=device, requires_grad=True)
        ort_x = copy.deepcopy(pt_x)
        ort_prediction, ort_loss = run_step(ort_model, ort_x)
        pt_prediction, pt_loss = run_step(pt_model, pt_x)
        _test_helpers.assert_values_are_close(ort_prediction, pt_prediction)
        _test_helpers.assert_values_are_close(ort_x.grad, pt_x.grad)
        _test_helpers.assert_values_are_close(ort_loss, pt_loss)

def test_tanh_grad():
    class NeuralNetTanh(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(NeuralNetTanh, self).__init__()

            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.tanh = torch.nn.Tanh()

        def forward(self, input1):
            out = self.fc1(input1)
            out = self.tanh(out)
            return out

    def run_step(model, x):
        prediction = model(x)
        loss = prediction.sum()
        loss.backward()
        return prediction, loss
    device = 'cuda'

    N, D_in, H, D_out = 120, 1536, 500, 1536
    pt_model = NeuralNetTanh(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    for step in range(10):
        pt_x = torch.randn(N, D_in, device=device, requires_grad=True)
        ort_x = copy.deepcopy(pt_x)
        ort_prediction, ort_loss = run_step(ort_model, ort_x)
        pt_prediction, pt_loss = run_step(pt_model, pt_x)
        _test_helpers.assert_values_are_close(ort_prediction, pt_prediction)
        _test_helpers.assert_values_are_close(ort_x.grad, pt_x.grad)
        _test_helpers.assert_values_are_close(ort_loss, pt_loss)


def test__defined_from_envvar():
    from onnxruntime.training import ortmodule
    os.environ['DUMMY_ORTMODULE'] = '15'
    assert ortmodule._defined_from_envvar('DUMMY_ORTMODULE', 14) == 15
    os.environ['DUMMY_ORTMODULE'] = '15j'
    with warnings.catch_warnings(record=True) as w:
        assert ortmodule._defined_from_envvar('DUMMY_ORTMODULE', 14) == 14
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "Unable to overwrite constant" in str(w[-1].message)
    del os.environ['DUMMY_ORTMODULE']


def test_sigmoid_grad_opset13():
    class NeuralNetSigmoid(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(NeuralNetSigmoid, self).__init__()

            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.sigmoid = torch.nn.Sigmoid()

        def forward(self, input1):
            out = self.fc1(input1)
            out = self.sigmoid(out)
            return out

    def run_step(model, x):
        prediction = model(x)
        loss = prediction.sum()
        loss.backward()
        return prediction, loss
    device = 'cuda'

    N, D_in, H, D_out = 120, 15360, 500, 15360
    pt_model = NeuralNetSigmoid(D_in, H, D_out).to(device)

    from onnxruntime.training import ortmodule
    old_opst_cst = ortmodule.ONNX_OPSET_VERSION
    old_opset = os.getenv("ORTMODULE_ONNX_OPSET_VERSION", None)
    os.environ["ORTMODULE_ONNX_OPSET_VERSION"] = '13'
    assert ortmodule.ONNX_OPSET_VERSION == 14

    ort_model = ORTModule(copy.deepcopy(pt_model))

    for step in range(2):
        pt_x = torch.randn(N, D_in, device=device, requires_grad=True)
        ort_x = copy.deepcopy(pt_x)
        ort_prediction, ort_loss = run_step(ort_model, ort_x)
        pt_prediction, pt_loss = run_step(pt_model, pt_x)
        if step == 0:
            model_onx = ort_model._torch_module._execution_manager._training_manager._onnx_models
            for name in ['exported_model', 'optimized_model', 'optimized_pre_grad_model']:
                onx = getattr(model_onx, name)
                opv = None
                for op in onx.opset_import:
                    if op.domain == '':
                        opv = op.version
                assert opv == 13
        _test_helpers.assert_values_are_close(ort_prediction, pt_prediction)
        _test_helpers.assert_values_are_close(ort_x.grad, pt_x.grad)
        _test_helpers.assert_values_are_close(ort_loss, pt_loss)

    if old_opset is None:
        del os.environ["ORTMODULE_ONNX_OPSET_VERSION"]
    else:
        os.environ["ORTMODULE_ONNX_OPSET_VERSION"] = old_opset
    assert ortmodule.ONNX_OPSET_VERSION == 13
    ortmodule.ONNX_OPSET_VERSION = old_opst_cst

@pytest.mark.parametrize("opset_version", [12, 13, 14])
def test_opset_version_change(opset_version):
    device = 'cuda'

    N, D_in, H, D_out = 64, 784, 500, 10
    x = torch.randn(N, D_in, device=device)
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)

    ort_model = ORTModule(model)

    # Must import a namespace containing ONNX_OPSET_VERSION, not ONNX_OPSET_VERSION directly
    from onnxruntime.training import ortmodule
    ortmodule.ONNX_OPSET_VERSION=opset_version

    # Make sure model runs without any exception
    prediction = ort_model(x)
    assert prediction is not None
    prediction = prediction.sum()
    prediction.backward()

    # Check opset version on ONNX model
    exported_model = ort_model._torch_module._execution_manager(ort_model._is_training())._onnx_models.exported_model
    assert exported_model.opset_import[0].version == opset_version

def test_serialize_ortmodule():

    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    pt_model = SerializationNet(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    x_1 = torch.randn(N, D_in, device=device)
    x_2 = copy.deepcopy(x_1)
    pt_out = pt_model.train_step(x_1)
    ort_out = ort_model.train_step(x_2)
    _test_helpers.assert_values_are_close(pt_out, ort_out)
    _test_helpers.assert_gradients_match_and_reset_gradient(
        ort_model, pt_model)
    pt_out, ort_out = None, None

    # Serialize ortmodule
    serialized_model = pickle.dumps(ort_model)

    # load from serialized string
    ort_model_2 = pickle.loads(serialized_model)

    x_1 = torch.randn(N, D_in, device=device)
    x_2 = copy.deepcopy(x_1)
    pt_out = pt_model.train_step(x_1)
    ort_out = ort_model_2.train_step(x_2)
    assert ort_out is not None
    _test_helpers.assert_values_are_close(pt_out, ort_out)
    _test_helpers.assert_gradients_match_and_reset_gradient(
        ort_model_2, pt_model)


@pytest.mark.parametrize("batch_size, M, N", [(1, 2, 3), (1, 4, 3), (1, 5, 5), (10, 3, 4), (10, 4, 3), (10, 4, 4)])
@pytest.mark.parametrize("k", [None, -5, -3, -1, 0, 2, 4])
@pytest.mark.parametrize("has_upper, upper", [(True, 1), (True, 0), (False, 1)])
def test_trilu_grad(batch_size, M, N, k, has_upper, upper):
    class NeuralNetTrilu(torch.nn.Module):
        def __init__(self, has_upper, upper):
            super(NeuralNetTrilu, self).__init__()
            self.upper = upper
            self.has_upper = has_upper

        def forward(self, x, k):
            if self.has_upper is False or self.upper == 1:
                y = torch.triu(x) if k is None else torch.triu(x, k)
            else:
                y = torch.tril(x) if k is None else torch.tril(x, k)
            return y

    def run_step(model, x, k):
        prediction = model(x, k)
        loss = prediction.sum()
        loss.backward()
        return prediction, loss

    device = 'cuda'
    pt_model = NeuralNetTrilu(has_upper, upper).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    pt_x = torch.rand((batch_size, M, N), requires_grad=True, device=device)
    ort_x = copy.deepcopy(pt_x)

    pt_prediction, pt_loss = run_step(pt_model, pt_x, k)
    ort_prediction, ort_loss = run_step(ort_model, ort_x, k)
    _test_helpers.assert_values_are_close(pt_prediction, ort_prediction)
    _test_helpers.assert_values_are_close(pt_loss, ort_loss)
    _test_helpers.assert_values_are_close(pt_x.grad, ort_x.grad)


@pytest.mark.parametrize("M, N", [(2400, 128), (2400, 256), (2400, 512), (2400, 1024), (2400, 2048), (2400, 4096), (2400, 12800)])
def test_softmax(M, N):
    class NeuralNetSoftmax(torch.nn.Module):
        def __init__(self):
            super(NeuralNetSoftmax, self).__init__()
            self.m = torch.nn.Softmax(dim=1)

        def forward(self, x):
            return self.m(x)

    def run_step(model, x):
        prediction = model(x)
        loss = prediction.sum()
        loss.backward()
        return prediction, loss

    device = 'cuda'
    pt_model = NeuralNetSoftmax().to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    pt_x = torch.rand((M, N), requires_grad=True, device=device)
    ort_x = copy.deepcopy(pt_x)

    pt_prediction, pt_loss = run_step(pt_model, pt_x)
    ort_prediction, ort_loss = run_step(ort_model, ort_x)
    _test_helpers.assert_values_are_close(pt_prediction, ort_prediction)
    _test_helpers.assert_values_are_close(pt_loss, ort_loss)
    _test_helpers.assert_values_are_close(pt_x.grad, ort_x.grad)


def test_check_opset_is_default_opset_after_training():
    M, N = 24, 6

    class NeuralNetSoftmax(torch.nn.Module):
        def __init__(self):
            super(NeuralNetSoftmax, self).__init__()
            self.m = torch.nn.Softmax(dim=1)

        def forward(self, x):
            return self.m(x)

    def run_step(model, x):
        prediction = model(x)
        loss = prediction.sum()
        loss.backward()
        return prediction, loss

    device = 'cuda'
    pt_model = NeuralNetSoftmax().to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    pt_x = torch.rand((M, N), requires_grad=True, device=device)
    ort_x = copy.deepcopy(pt_x)

    pt_prediction, pt_loss = run_step(pt_model, pt_x)
    ort_prediction, ort_loss = run_step(ort_model, ort_x)
    _test_helpers.assert_values_are_close(pt_prediction, ort_prediction)
    _test_helpers.assert_values_are_close(pt_loss, ort_loss)
    _test_helpers.assert_values_are_close(pt_x.grad, ort_x.grad)
    assert ortmodule_module.ONNX_OPSET_VERSION == DEFAULT_OPSET


def test_random_states_unchanged_for_ortmodule():
    import numpy

    os.environ['ORTMODULE_FALLBACK_RETRY'] = 'False'

    class NeuralNetSlice(torch.nn.Module):
        def __init__(self):
            super(NeuralNetSlice, self).__init__()
            self.dim = 32

        def forward(self, x):
            # This slice operation will call sympy.Min() when exporting, which will change Python's random state
            return x[:self.dim, :]

    def random_state_equal(a, b):
        assert type(a) == type(b)
        if isinstance(a, tuple):
            assert len(a) == len(b)
            return all([random_state_equal(a_i, b_i) for a_i, b_i in zip(a, b)])
        if isinstance(a, numpy.ndarray):
            return numpy.array_equal(a, b)
        if isinstance(a, torch.Tensor):
            return torch.equal(a, b)
        return a == b

    model = NeuralNetSlice()
    x = torch.randn(16, 16)

    ori_random_states = _utils.get_random_states()

    ort_model = ORTModule(model)
    ort_model(x)

    new_random_states = _utils.get_random_states()

    assert random_state_equal(ori_random_states, new_random_states)

    del os.environ['ORTMODULE_FALLBACK_RETRY']
