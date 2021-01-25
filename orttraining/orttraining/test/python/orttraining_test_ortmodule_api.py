# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# orttraining_test_ortmodule_api.py

import torch
import pytest
import warnings

import onnxruntime
from onnxruntime.training import _utils, ORTModule

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

# ORTModule-API tests

def test_forward_call_single_positional_argument():
    device = 'cuda'

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device)
    try:
        model(x)
    except Exception as exception:
        raise exception

def test_forward_call_multiple_positional_arguments():
    device = 'cuda'

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetMultiplePositionalArguments(input_size=D_in, hidden_size=H, num_classes=D_out).to(device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_in, device=device)
    try:
        model(x, y)
    except Exception as exception:
        raise exception

def test_forward_call_positional_arguments():
    device = 'cuda'

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetPositionalArguments(input_size=D_in, hidden_size=H, num_classes=D_out).to(device)
    model = ORTModule(model)
    args = [torch.randn(N, D_in, device=device), torch.randn(N, D_in, device=device), torch.randn(N, D_in, device=device)]
    try:
        model(*args)
    except Exception as exception:
        raise exception

def test_forward_call_keyword_arguments():
    device = 'cuda'

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetKeywordArguments(D_in, H, D_out).to(device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_in, device=device)
    z = torch.randn(N, D_in, device=device)
    try:
        model(x, y, z)
    except Exception as exception:
        raise exception

def test_forward_call_positional_and_keyword_arguments():
    device = 'cuda'

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetPositionalAndKeywordArguments(D_in, H, D_out).to(device)
    model = ORTModule(model)
    a = torch.randn(N, D_in, device=device)
    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_in, device=device)
    z = torch.randn(N, D_in, device=device)
    try:
        model(a, x, y, z)
    except Exception as exception:
        raise exception

def test_model_cuda():
    original_device = 'cpu'
    to_device = 'cuda'

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=to_device)
    for _, parameter_value in model.named_parameters():
        assert parameter_value.device.type == original_device

    model = model.cuda()
    model(x)

    for _, parameter_value in model.named_parameters():
        assert parameter_value.device.type == to_device

def test_model_cpu():
    original_device = 'cuda'
    to_device = 'cpu'

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(original_device)
    model = ORTModule(model)
    x = torch.randn(N, D_in)
    for _, parameter_value in model.named_parameters():
        assert parameter_value.device.type == original_device

    model = model.cpu()
    model(x)

    for _, parameter_value in model.named_parameters():
        assert parameter_value.device.type == to_device

@pytest.mark.parametrize("original_device, to_argument, requires_export, device_type, device_index", [
    ('cpu', torch.device('cuda'), True, 'cuda', 0),
    ('cpu', 'cuda', True, 'cuda', 0),
    ('cpu', 'cuda:0', True, 'cuda', 0),
    ('cpu', 'cuda', True, 'cuda', 0),
    ('cuda', 'cuda', False, 'cuda', 0),
    ('cuda', 'cuda:0', False, 'cuda', 0),
    ('cuda', torch.device('cuda'), False, 'cuda', 0),
    ('cuda', 'cpu', True, 'cpu', 0),
    ('cuda', torch.device('cpu'), True, 'cpu', 0),
    ('cpu', 'cpu', False, 'cpu', None),
    ('cpu', torch.device('cpu'), False, 'cpu', None),
    ('cpu', torch.zeros(2, device=torch.device('cuda')), True, 'cuda', 0),
    ])
def test_model_to_device(original_device, to_argument, requires_export, device_type, device_index):
    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(original_device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device_type)
    for _, parameter_value in model.named_parameters():
        assert parameter_value.device.type == original_device

    model = model.to(to_argument)
    assert model._require_export == requires_export
    assert model._device == torch.device(device_type+':'+str(device_index) if device_index is not None else device_type)
    model(x)

    for _, parameter_value in model.named_parameters():
        assert parameter_value.device.type == device_type

@pytest.mark.parametrize("original_device, to_device", [
    ('cuda', 'cpu'),
    ('cpu', 'cuda')
    ])
def test_model_to_device_and_back_to_original(original_device, to_device):
    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(original_device)
    model = ORTModule(model)
    for _, parameter_value in model.named_parameters():
        assert parameter_value.device.type == original_device

    model = model.to(to_device)
    assert model._require_export == True
    assert model._device == torch.device(to_device+':0')

    for _, parameter_value in model.named_parameters():
        assert parameter_value.device.type == to_device

    model = model.to(original_device)
    assert model._require_export == True
    assert model._device == torch.device(original_device+':0')
    for _, parameter_value in model.named_parameters():
        assert parameter_value.device.type == original_device

@pytest.mark.parametrize("device", ['cuda', 'cpu'])
def test_model_without_parameters(device):
    class Net(torch.nn.Module):
        def forward(self, x):
            return x

    model = Net().to(device)
    model = ORTModule(model).to(device)
    with pytest.raises(RuntimeError) as e:
        model(torch.tensor(1.))
        assert e.value == 'ORTModule only supports model with at least one trainable parameter'

@pytest.mark.parametrize("device", ['cuda', 'cpu'])
def test_model_without_trainable_parameters(device):
    class Net(torch.nn.Module):
        def forward(self, x):
            return torch.nn.ReLU(x)

    model = Net()
    model = ORTModule(model).to(device)
    with pytest.raises(RuntimeError) as e:
        model(torch.tensor(1.).to(device))
        assert e.value == 'ORTModule only supports model with at least one trainable parameter'

@pytest.mark.parametrize("device", ['cuda', 'cpu'])
def test_model_with_unused_trainable_parameters(device):
    class Net(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(Net, self).__init__()
            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(hidden_size, num_classes)

        def forward(self, input1):
            return input1

    model = Net(784, 500, 10)
    model = ORTModule(model).to(device)
    with pytest.raises(RuntimeError) as e:
        model(torch.tensor(1.).to(device))
        assert e.value == 'ORTModule only supports model with at least one trainable parameter'

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

    model = MultipleDeviceModel()
    with pytest.raises(RuntimeError) as e:
        model = ORTModule(model)
        assert e.value == 'ORTModule supports a single device per model for now'

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

    model = MultipleDeviceModel()
    with pytest.raises(RuntimeError) as e:
        model = ORTModule(model)
        assert e.value == 'ORTModule supports a single device per model for now'

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

    model = MultipleDeviceModel()
    with pytest.raises(RuntimeError) as e:
        model = ORTModule(model)
        assert e.value == 'ORTModule supports a single device per model for now'

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

    model = MultipleDeviceModel()
    with pytest.raises(RuntimeError) as e:
        model = ORTModule(model)
        assert e.value == 'ORTModule supports a single device per model for now'

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

def test_input_requires_grad_saved():
    device = 'cuda'

    N, D_in, H, D_out = 32, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device, requires_grad=True) + 1
    model(x)
    assert(model._input_names_require_grad == ['input1'])

def test_input_requires_grad_backward_creates_input_grad():
    device = 'cuda'

    N, D_in, H, D_out = 32, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device, requires_grad=True)
    prediction = model(x)
    s = prediction.sum()
    s.backward()
    assert x.grad is not None

def test_changes_input_requires_grad_reinitializes_module_gradient_graph_builder():
    device = 'cuda'

    N, D_in, H, D_out = 32, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device, requires_grad=True)
    model(x.data)
    module_gradient_graph_builder = model._module_gradient_graph_builder
    model(x)
    assert module_gradient_graph_builder != model._module_gradient_graph_builder
