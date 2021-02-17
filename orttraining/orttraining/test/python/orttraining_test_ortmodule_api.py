# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# orttraining_test_ortmodule_api.py

import torch
from transformers import AutoConfig, BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
import pytest
import warnings
from unittest.mock import patch
from collections import OrderedDict
from collections import namedtuple

from onnxruntime.training import _utils, ORTModule
import _test_helpers


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

def _get_bert_for_sequence_classification_model(device):
    """Returns the BertForSequenceClassification pretrained model"""

    config = AutoConfig.from_pretrained(
            "bert-base-uncased",
            num_labels=2,
            num_hidden_layers=1,
            output_attentions = False,
            output_hidden_states = False,
    )

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

# ORTModule-API tests

def test_forward_call_single_positional_argument():
    device = 'cuda'

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device)
    # Make sure model runs without any exception
    output = model(x)
    assert output is not None

def test_forward_call_multiple_positional_arguments():
    device = 'cuda'

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetMultiplePositionalArguments(input_size=D_in, hidden_size=H, num_classes=D_out).to(device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_in, device=device)

    # Make sure model runs without any exception
    output = model(x, y)
    assert output is not None

def test_forward_call_positional_arguments():
    device = 'cuda'

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetPositionalArguments(input_size=D_in, hidden_size=H, num_classes=D_out).to(device)
    model = ORTModule(model)
    args = [torch.randn(N, D_in, device=device), torch.randn(N, D_in, device=device), torch.randn(N, D_in, device=device)]

    # Make sure model runs without any exception
    output = model(*args)
    assert output is not None

def test_forward_call_keyword_arguments():
    device = 'cuda'

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetKeywordArguments(D_in, H, D_out).to(device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_in, device=device)
    z = torch.randn(N, D_in, device=device)

    # Make sure model runs without any exception
    output = model(x, y, z)
    assert output is not None

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
    output = model(a, x, y, z)
    assert output is not None

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
    assert model._device_changed == requires_export
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
    assert model._device_changed == True
    assert model._device == torch.device(to_device+':0')

    for _, parameter_value in model.named_parameters():
        assert parameter_value.device.type == to_device

    model = model.to(original_device)
    assert model._device_changed == True
    assert model._device == torch.device(original_device+':0')
    for _, parameter_value in model.named_parameters():
        assert parameter_value.device.type == original_device

# TODO: Re-enable this Test when .to(), .cpu() and .cuda() are fixed
# def test_model_with_different_devices_same_session():
#     N, D_in, H, D_out = 64, 784, 500, 10
#     model = NeuralNetSinglePositionalArgument(D_in, H, D_out)
#     model = ORTModule(model)

#     for i in range(5):
#         if i % 2 == 0:
#             device = 'cpu'
#         else:
#             device = 'cuda'

#         model.to(device)
#         x = torch.randn(N, D_in, device=device)
#         y = model(x)

# TODO: Re-enable this Test when .to(), .cpu() and .cuda() are fixed
@pytest.mark.parametrize("device", ['cuda', 'cpu'])
def test_input_requires_grad_saved(device):
    N, D_in, H, D_out = 32, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device, requires_grad=True) + 1
    model(x)
    assert model._input_names_require_grad == ['input1']

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

# TODO: Re-enable this Test when .to(), .cpu() and .cuda() are fixed
# @pytest.mark.parametrize("device", ['cuda', 'cpu'])
# def test_changes_input_requires_grad_reinitializes_module_gradient_graph_builder(device):
#     N, D_in, H, D_out = 32, 784, 500, 10
#     model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
#     model = ORTModule(model)
#     x = torch.randn(N, D_in, device=device, requires_grad=True)
#     model(x.data)
#     module_gradient_graph_builder = model._module_gradient_graph_builder
#     model(x)
#     assert module_gradient_graph_builder != model._module_gradient_graph_builder

def test_gpu_reserved_memory_with_torch_no_grad():
    device = 'cuda'

    # Create a model and get the memory_reserved when torch.no_grad has been enabled
    # before and after export
    model_with_no_grad = _get_bert_for_sequence_classification_model(device)
    x, y, z = _get_bert_for_sequence_classification_sample_data(device)

    torch.cuda.empty_cache()
    model_with_no_grad = ORTModule(model_with_no_grad)
    mem_reserved_before_export = torch.cuda.memory_reserved(device)
    model_with_no_grad(x, y, None, None, None, None, z)
    mem_reserved_after_export_with_torch_no_grad = torch.cuda.memory_reserved(device)
    del model_with_no_grad
    torch.cuda.empty_cache()
    mem_reserved_after_cache_empty = torch.cuda.memory_reserved(device)
    assert mem_reserved_before_export == mem_reserved_after_cache_empty

    # Create another model and get the memory_reserved when torch.no_grad and torch.cuda.empty_cache
    # has not been enabled after export
    model_without_no_grad = _get_bert_for_sequence_classification_model(device)
    model_without_no_grad = ORTModule(model_without_no_grad)
    mem_reserved_after_export_without_torch_no_grad = 0

    with patch('torch.no_grad'), patch('torch.cuda.empty_cache'):
        model_without_no_grad(x, y, None, None, None, None, z)
        mem_reserved_after_export_without_torch_no_grad = torch.cuda.memory_reserved(device)

    assert mem_reserved_after_export_with_torch_no_grad < mem_reserved_after_export_without_torch_no_grad
    assert mem_reserved_before_export == mem_reserved_after_export_with_torch_no_grad

@pytest.mark.parametrize("return_type, device", [
    (dict, 'cpu'),
    (dict, 'cuda'),
    (OrderedDict, 'cpu'),
    (OrderedDict, 'cuda'),
    (SequenceClassifierOutput, 'cpu'),
    (SequenceClassifierOutput, 'cuda')
    ])
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

    with pytest.raises(TypeError) as type_error:
        model(x, y, z)
    assert 'ORTModule does not support the following model output type' in str(type_error.value)

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
    class CustomClass(object):
        def __init__(self, out1, out2, out3):
            self.out1 = out1
            self.out2 = out2
            self.out3 = out3

    class NeuralNetCustomClassOutput(torch.nn.Module):
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
            return CustomClass(out1, out2, out3)

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetCustomClassOutput(D_in, H, D_out).to(device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_in, device=device)
    z = torch.randn(N, D_in, device=device)

    with pytest.raises(TypeError) as runtime_error:
        model(x, y, z)
    assert 'ORTModule does not support the following model output type' in str(runtime_error.value)

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
    output = model_with_no_grad(x, y, None, None, None, None, z)
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

    model = MultipleDeviceModel()
    with pytest.raises(RuntimeError) as e:
        model = ORTModule(model)
    assert str(e.value) == 'ORTModule supports a single device per model for now'

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
    assert str(e.value) == 'ORTModule supports a single device per model for now'

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
    assert str(e.value) == 'ORTModule supports a single device per model for now'

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
    assert str(e.value) == 'ORTModule supports a single device per model for now'

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
