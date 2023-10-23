import copy
import io
import os
import random
import tempfile

import numpy as np
import onnx
import pytest
import torch

import onnxruntime
import onnxruntime.training.onnxblock as onnxblock
from onnxruntime.capi import _pybind_state as C
from onnxruntime.training import artifacts

# PyTorch Module definitions


class SimpleNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, model_input):
        out = self.fc1(model_input)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# onnxblock Model definitions


class SimpleBlockWithMSELoss(onnxblock.ForwardBlock):
    def __init__(self):
        super().__init__()
        self.loss = onnxblock.loss.MSELoss()

    def build(self, output_name):
        return self.loss(output_name)


class SimpleBlockWithCrossEntropyLoss(onnxblock.ForwardBlock):
    def __init__(self):
        super().__init__()
        self.loss = onnxblock.loss.CrossEntropyLoss()

    def build(self, output_name):
        return self.loss(output_name)


class SimpleTrainingBlockWithMSELoss(onnxblock.TrainingBlock):
    def __init__(self):
        super().__init__()
        self.loss = onnxblock.loss.MSELoss()

    def build(self, output_name):
        return self.loss(output_name)


class SimpleTrainingBlockWithCrossEntropyLoss(onnxblock.TrainingBlock):
    def __init__(self):
        super().__init__()
        self.loss = onnxblock.loss.CrossEntropyLoss()

    def build(self, output_name):
        return self.loss(output_name)


class SimpleBlockWithBCEWithLogitsLoss(onnxblock.ForwardBlock):
    def __init__(self):
        super().__init__()
        self.loss = onnxblock.loss.BCEWithLogitsLoss()

    def build(self, output_name):
        return self.loss(output_name)


class SimpleTrainingBlockWithBCEWithLogitsLoss(onnxblock.TrainingBlock):
    def __init__(self):
        super().__init__()
        self.loss = onnxblock.loss.BCEWithLogitsLoss()

    def build(self, output_name):
        return self.loss(output_name)


class SimpleBlockWithL1Loss(onnxblock.ForwardBlock):
    def __init__(self):
        super().__init__()
        self.loss = onnxblock.loss.L1Loss()

    def build(self, output_name):
        return self.loss(output_name)


class SimpleTrainingBlockWithL1Loss(onnxblock.TrainingBlock):
    def __init__(self):
        super().__init__()
        self.loss = onnxblock.loss.L1Loss()

    def build(self, output_name):
        return self.loss(output_name)


# Test utility methods


def _get_onnx_model(torch_model, model_inputs):
    model_outputs = torch_model(*model_inputs)
    if isinstance(model_outputs, torch.Tensor):
        model_outputs = [model_outputs]
    dynamic_axes = {}
    input_names = []
    output_names = []
    for i, model_input in enumerate(model_inputs):
        input_name = f"input-{i}"
        input_names.append(input_name)
        dynamic_axes[input_name] = {}
        for dim_idx in range(len(model_input.shape)):
            dynamic_axes[input_name].update({dim_idx: f"{input_name}_dim{dim_idx}"})

    for i, model_output in enumerate(model_outputs):
        output_name = f"output-{i}"
        output_names.append(output_name)
        dynamic_axes[output_name] = {}
        for dim_idx in range(len(model_output.shape)):
            dynamic_axes[output_name].update({dim_idx: f"{output_name}_dim{dim_idx}"})

    f = io.BytesIO()
    torch.onnx.export(
        torch_model,
        model_inputs,
        f,
        input_names=input_names,
        output_names=output_names,
        opset_version=14,
        do_constant_folding=False,
        training=torch.onnx.TrainingMode.TRAINING,
        dynamic_axes=dynamic_axes,
        export_params=True,
        keep_initializers_as_inputs=False,
    )
    return onnx.load_model_from_string(f.getvalue())


def _to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def _get_models(device, batch_size, input_size, hidden_size, output_size, zero_flag=False):
    """Returns the pt and onnx models for SimpleNet"""
    pt_model = SimpleNet(input_size, hidden_size, output_size).to(device)

    # setting all initial weights to zero
    if zero_flag:
        with torch.no_grad():
            for param in pt_model.parameters():
                param.zero_()

    x = torch.randn(batch_size, input_size, device=device)
    onnx_model = _get_onnx_model(pt_model, (x,))

    return pt_model, onnx_model


def _get_training_ort_output_names(pt_model, onnx_model):
    """Returns the ort output names"""
    ort_output_names = [onnx_model.graph.output[0].name]
    for name, _ in pt_model.named_parameters():
        ort_output_names.append(f"{name}_grad.accumulation.out")

    return ort_output_names


def _get_training_ort_inputs(x, target, pt_model, onnx_model, target_type=None):
    """Returns the ort inputs"""

    ort_inputs = {
        onnx_model.graph.input[0].name: _to_numpy(copy.deepcopy(x)),
        onnx_model.graph.input[1].name: _to_numpy(copy.deepcopy(target))
        if target_type is None
        else _to_numpy(copy.deepcopy(target).type(target_type)),
    }
    if target_type is not None:
        ort_inputs[onnx_model.graph.input[1].name]
    for name, param in pt_model.named_parameters():
        ort_inputs[name] = _to_numpy(copy.deepcopy(param))
        ort_inputs[f"{name}_grad.accumulation.buffer"] = _to_numpy(torch.zeros_like(param))
    ort_inputs["lazy_reset_grad"] = np.full(1, True)

    return ort_inputs


# All unit tests


@pytest.mark.parametrize(
    "block",
    [
        SimpleBlockWithMSELoss,
        SimpleBlockWithCrossEntropyLoss,
        SimpleTrainingBlockWithMSELoss,
        SimpleTrainingBlockWithCrossEntropyLoss,
        SimpleBlockWithBCEWithLogitsLoss,
        SimpleTrainingBlockWithBCEWithLogitsLoss,
        SimpleBlockWithL1Loss,
        SimpleTrainingBlockWithL1Loss,
    ],
)
def test_loss_composition(block):
    # Given
    device = "cuda"
    batch_size, input_size, hidden_size, output_size = 64, 784, 500, 10
    _, base_model = _get_models(device, batch_size, input_size, hidden_size, output_size)

    # When / Then no error occurs
    simple_block = block()
    with onnxblock.base(base_model):
        _ = simple_block(base_model.graph.output[0].name)


@pytest.mark.parametrize(
    "block",
    [
        SimpleBlockWithMSELoss,
        SimpleBlockWithL1Loss,
    ],
)
def test_mse_loss_execution(block):
    # Given
    device = "cuda"
    batch_size, input_size, hidden_size, output_size = 64, 784, 500, 10
    pt_model, base_model = _get_models(device, batch_size, input_size, hidden_size, output_size)

    x = torch.randn(batch_size, input_size, device=device)
    target = torch.randn(batch_size, output_size, device=device)

    # Build the onnx model with loss
    simple_block = block()
    with onnxblock.base(base_model):
        _ = simple_block(base_model.graph.output[0].name)

    onnx_model = simple_block.to_model_proto()

    ort_output_names = [onnx_model.graph.output[0].name]
    ort_inputs = {
        onnx_model.graph.input[0].name: _to_numpy(copy.deepcopy(x)),
        onnx_model.graph.input[1].name: _to_numpy(copy.deepcopy(target)),
    }

    def mse_loss(prediction, target):
        loss = torch.nn.MSELoss()
        return loss(prediction, target)

    def l1_loss(prediction, target):
        loss = torch.nn.L1Loss()
        return loss(prediction, target)

    loss = mse_loss if block == SimpleBlockWithMSELoss else l1_loss

    # When
    with tempfile.NamedTemporaryFile(suffix=".onnx") as onnx_fo:
        onnx.save(onnx_model, onnx_fo.name)
        ort_session = onnxruntime.InferenceSession(onnx_fo.name, providers=C.get_available_providers())

        ort_outs = ort_session.run(ort_output_names, ort_inputs)
        torch_outs = loss(pt_model(x), target)

        # Then
        assert np.allclose(ort_outs[0], _to_numpy(torch_outs))


def test_crossentropy_loss_execution():
    # Given
    device = "cuda"
    batch_size, input_size, hidden_size, output_size = 64, 784, 500, 10
    pt_model, base_model = _get_models(device, batch_size, input_size, hidden_size, output_size)

    x = torch.randn(batch_size, input_size, device=device)
    target = torch.randint(high=output_size, size=(batch_size,), dtype=torch.int64, device=device)

    # Build the onnx model with loss
    simple_block = SimpleBlockWithCrossEntropyLoss()
    with onnxblock.base(base_model):
        _ = simple_block(base_model.graph.output[0].name)

    onnx_model = simple_block.to_model_proto()

    ort_output_names = [onnx_model.graph.output[0].name]
    ort_inputs = {
        onnx_model.graph.input[0].name: _to_numpy(copy.deepcopy(x)),
        onnx_model.graph.input[1].name: _to_numpy(copy.deepcopy(target).type(torch.int64)),
    }

    def crossentropy_loss(prediction, target):
        loss = torch.nn.CrossEntropyLoss()
        return loss(prediction, target)

    # When
    with tempfile.NamedTemporaryFile(suffix=".onnx") as onnx_fo:
        onnx.save(onnx_model, onnx_fo.name)
        ort_session = onnxruntime.InferenceSession(onnx_fo.name, providers=C.get_available_providers())

        ort_outs = ort_session.run(ort_output_names, ort_inputs)
        torch_outs = crossentropy_loss(pt_model(x), target)

        # Then
        assert np.allclose(ort_outs[0], _to_numpy(torch_outs))


def test_bcewithlogits_loss_execution():
    # Given
    device = "cuda"
    batch_size, input_size, hidden_size, output_size = 64, 784, 500, 10
    pt_model, base_model = _get_models(device, batch_size, input_size, hidden_size, output_size)

    x = torch.randn(batch_size, input_size, device=device)
    target = torch.randn(batch_size, output_size, device=device)

    # Build the onnx model with loss
    simple_block = SimpleBlockWithBCEWithLogitsLoss()
    with onnxblock.base(base_model):
        _ = simple_block(base_model.graph.output[0].name)

    onnx_model = simple_block.to_model_proto()

    ort_output_names = [onnx_model.graph.output[0].name]
    ort_inputs = {
        onnx_model.graph.input[0].name: _to_numpy(copy.deepcopy(x)),
        onnx_model.graph.input[1].name: _to_numpy(copy.deepcopy(target)),
    }

    def bcewithlogits_loss(prediction, target):
        loss = torch.nn.BCEWithLogitsLoss()
        return loss(prediction, target)

    # When
    with tempfile.NamedTemporaryFile(suffix=".onnx") as onnx_fo:
        onnx.save(onnx_model, onnx_fo.name)
        ort_session = onnxruntime.InferenceSession(onnx_fo.name, providers=C.get_available_providers())

        ort_outs = ort_session.run(ort_output_names, ort_inputs)
        torch_outs = bcewithlogits_loss(pt_model(x), target)

        # Then
        assert np.allclose(ort_outs[0], _to_numpy(torch_outs))


def test_mse_loss_training_graph_execution():
    # Given
    device = "cuda"
    batch_size, input_size, hidden_size, output_size = 64, 784, 500, 10
    pt_model, base_model = _get_models(device, batch_size, input_size, hidden_size, output_size)

    x = torch.randn(batch_size, input_size, device=device)
    target = torch.randn(batch_size, output_size, device=device)

    # Build the onnx trainingmodel with loss
    simple_block = SimpleTrainingBlockWithMSELoss()
    for name, _ in pt_model.named_parameters():
        simple_block.requires_grad(name)

    with onnxblock.base(base_model):
        _ = simple_block(base_model.graph.output[0].name)

    onnx_model, _ = simple_block.to_model_proto()

    ort_output_names = _get_training_ort_output_names(pt_model, onnx_model)
    ort_inputs = _get_training_ort_inputs(x, target, pt_model, onnx_model)

    def mse_loss(prediction, target):
        loss = torch.nn.MSELoss()
        return loss(prediction, target)

    # When
    with tempfile.NamedTemporaryFile(suffix=".onnx") as onnx_fo:
        onnx.save(onnx_model, onnx_fo.name)
        ort_session = onnxruntime.InferenceSession(onnx_fo.name, providers=C.get_available_providers())

        ort_outs = ort_session.run(ort_output_names, ort_inputs)
        torch_outs = mse_loss(pt_model(x), target)
        torch_outs.backward()

        # Then
        # assert loss is close
        assert np.allclose(ort_outs[0], _to_numpy(torch_outs))


def test_crossentropy_loss_training_graph_execution():
    # Given
    device = "cuda"
    batch_size, input_size, hidden_size, output_size = 64, 784, 500, 10
    pt_model, base_model = _get_models(device, batch_size, input_size, hidden_size, output_size)

    x = torch.randn(batch_size, input_size, device=device)
    target = torch.randint(high=output_size, size=(batch_size,), dtype=torch.int64, device=device)

    # Build the onnx trainingmodel with loss
    simple_block = SimpleTrainingBlockWithCrossEntropyLoss()
    for name, _ in pt_model.named_parameters():
        simple_block.requires_grad(name)

    with onnxblock.base(base_model):
        _ = simple_block(base_model.graph.output[0].name)

    onnx_model, _ = simple_block.to_model_proto()

    ort_output_names = _get_training_ort_output_names(pt_model, onnx_model)
    ort_inputs = _get_training_ort_inputs(x, target, pt_model, onnx_model, target_type=torch.int64)

    def crossentropy_loss(prediction, target):
        loss = torch.nn.CrossEntropyLoss()
        return loss(prediction, target)

    # When
    with tempfile.NamedTemporaryFile(suffix=".onnx") as onnx_fo:
        onnx.save(onnx_model, onnx_fo.name)
        ort_session = onnxruntime.InferenceSession(onnx_fo.name, providers=C.get_available_providers())

        ort_outs = ort_session.run(ort_output_names, ort_inputs)
        torch_outs = crossentropy_loss(pt_model(x), target)
        torch_outs.backward()

        # Then
        # assert loss is close
        assert np.allclose(ort_outs[0], _to_numpy(torch_outs))


def test_bcewithlogits_loss_training_graph_execution():
    # Given
    device = "cuda"
    batch_size, input_size, hidden_size, output_size = 64, 784, 500, 10
    pt_model, base_model = _get_models(device, batch_size, input_size, hidden_size, output_size)

    x = torch.randn(batch_size, input_size, device=device)
    target = torch.randn(batch_size, output_size, device=device)

    # Build the onnx model with loss
    simple_block = SimpleTrainingBlockWithBCEWithLogitsLoss()
    for name, _ in pt_model.named_parameters():
        simple_block.requires_grad(name)

    with onnxblock.base(base_model):
        _ = simple_block(base_model.graph.output[0].name)

    onnx_model, _ = simple_block.to_model_proto()

    ort_output_names = _get_training_ort_output_names(pt_model, onnx_model)
    ort_inputs = _get_training_ort_inputs(x, target, pt_model, onnx_model)

    def bcewithlogits_loss(prediction, target):
        loss = torch.nn.BCEWithLogitsLoss()
        return loss(prediction, target)

    # When
    with tempfile.NamedTemporaryFile(suffix=".onnx") as onnx_fo:
        onnx.save(onnx_model, onnx_fo.name)
        ort_session = onnxruntime.InferenceSession(onnx_fo.name, providers=C.get_available_providers())

        ort_outs = ort_session.run(ort_output_names, ort_inputs)
        torch_outs = bcewithlogits_loss(pt_model(x), target)
        torch_outs.backward()

        # Then
        # assert loss is close
        assert np.allclose(ort_outs[0], _to_numpy(torch_outs))


@pytest.mark.parametrize(
    "block",
    [SimpleTrainingBlockWithMSELoss, SimpleTrainingBlockWithCrossEntropyLoss, SimpleTrainingBlockWithBCEWithLogitsLoss],
)
@pytest.mark.parametrize("grad_clipping", [None, onnxblock.optim.ClipGradNorm(2.5)])
def test_adamw_optimizer_composition(block, grad_clipping):
    # Given
    device = "cuda"
    batch_size, input_size, hidden_size, output_size = 64, 784, 500, 10
    pt_model, base_model = _get_models(device, batch_size, input_size, hidden_size, output_size)

    # When / Then no error occurs
    simple_block = block()
    for name, _ in pt_model.named_parameters():
        simple_block.requires_grad(name)

    with onnxblock.base(base_model):
        _ = simple_block(base_model.graph.output[0].name)

    optimizer = onnxblock.optim.AdamW(clip_grad=grad_clipping)
    with onnxblock.empty_base() as accessor:
        _ = optimizer(simple_block.parameters())
        optimizer_model = accessor.model
        assert optimizer_model


# TODO: Add a test for correctness when creation of ortvalues of
# tensor seq is possible on cuda
def test_adamw_optimizer_execution():
    # Given
    device = "cuda"
    batch_size, input_size, hidden_size, output_size = 64, 784, 500, 10
    pt_model, base_model = _get_models(device, batch_size, input_size, hidden_size, output_size)

    x = torch.randn(batch_size, input_size, device=device)
    target = torch.randn(batch_size, output_size, device=device)

    simple_block = SimpleTrainingBlockWithMSELoss()
    for name, _ in pt_model.named_parameters():
        simple_block.requires_grad(name)

    with onnxblock.base(base_model):
        _ = simple_block(base_model.graph.output[0].name)

    optimizer = onnxblock.optim.AdamW()
    with onnxblock.empty_base() as accessor:
        output_name = optimizer(simple_block.parameters())
        optimizer_model = accessor.model

    learning_rate = 0.001
    step = 1
    ort_output_names = [output_name]

    def mse_loss(prediction, target):
        loss = torch.nn.MSELoss()
        return loss(prediction, target)

    # When
    with tempfile.NamedTemporaryFile(suffix=".onnx") as onnx_fo:
        onnx.save(optimizer_model, onnx_fo.name)

        loss = mse_loss(pt_model(x), target)
        loss.backward()

        ort_inputs = {
            "learning_rate": np.full(1, learning_rate, dtype=np.float32),
            "step": np.full(1, step, dtype=np.int64),
            "params": [],
            "gradients": [],
            "first_order_moments": [],
            "second_order_moments": [],
        }
        for _, param in pt_model.named_parameters():
            ort_inputs["params"].append(_to_numpy(copy.deepcopy(param)))
            ort_inputs["gradients"].append(_to_numpy(copy.deepcopy(param.grad)))
            ort_inputs["first_order_moments"].append(_to_numpy(torch.zeros_like(param)))
            ort_inputs["second_order_moments"].append(_to_numpy(torch.zeros_like(param)))

        # Then no error occurs when executing the model
        ort_session = onnxruntime.InferenceSession(onnx_fo.name, providers=C.get_available_providers())
        _ = ort_session.run(ort_output_names, ort_inputs)


@pytest.mark.parametrize(
    "block",
    [SimpleTrainingBlockWithMSELoss, SimpleTrainingBlockWithCrossEntropyLoss, SimpleTrainingBlockWithBCEWithLogitsLoss],
)
@pytest.mark.parametrize("grad_clipping", [None, onnxblock.optim.ClipGradNorm(2.5)])
def test_sgd_optimizer_composition(block, grad_clipping):
    # Given
    device = "cpu"
    batch_size, input_size, hidden_size, output_size = 64, 784, 500, 10
    pt_model, base_model = _get_models(device, batch_size, input_size, hidden_size, output_size)

    # When / Then no error occurs
    simple_block = block()
    for name, _ in pt_model.named_parameters():
        simple_block.requires_grad(name)

    with onnxblock.base(base_model):
        _ = simple_block(base_model.graph.output[0].name)

    optimizer = onnxblock.optim.SGD(clip_grad=grad_clipping)
    with onnxblock.empty_base() as accessor:
        _ = optimizer(simple_block.parameters())
        optimizer_model = accessor.model
        assert optimizer_model


def test_retrieve_parameters():
    # Given
    device = "cuda"
    batch_size, input_size, hidden_size, output_size = 64, 784, 500, 10
    pt_model, base_model = _get_models(device, batch_size, input_size, hidden_size, output_size)

    simple_block = SimpleTrainingBlockWithMSELoss()
    with onnxblock.base(base_model):
        _ = simple_block(base_model.graph.output[0].name)

    # When
    trainable_params, non_trainable_params = simple_block.parameters()

    # Then
    assert not non_trainable_params
    for ort_param, (pt_param_name, pt_param) in zip(trainable_params, pt_model.named_parameters()):
        assert ort_param.name == pt_param_name
        assert np.allclose(
            np.frombuffer(ort_param.raw_data, dtype=np.float32).reshape(pt_param.shape),
            _to_numpy(pt_param),
        )


def test_retrieve_parameters_before_building_gradient_graph():
    # Given
    device = "cuda"
    batch_size, input_size, hidden_size, output_size = 64, 784, 500, 10
    _, base_model = _get_models(device, batch_size, input_size, hidden_size, output_size)

    simple_block = SimpleTrainingBlockWithMSELoss()

    # When / Then
    with pytest.raises(Exception) as ex_info:
        _, _ = simple_block.parameters()
    assert "Please build the training model first before trying to retrieve the parameters." in str(ex_info.value)


def test_save_checkpoint():
    # Given
    device = "cuda"
    batch_size, input_size, hidden_size, output_size = 64, 784, 500, 10
    _, base_model = _get_models(device, batch_size, input_size, hidden_size, output_size)

    simple_block = SimpleTrainingBlockWithMSELoss()
    with onnxblock.base(base_model):
        _ = simple_block(base_model.graph.output[0].name)
    trainable_params, non_trainable_params = simple_block.parameters()

    # When
    with tempfile.TemporaryDirectory() as checkpoint_dir_name:
        checkpoint_file_path = os.path.join(checkpoint_dir_name, "checkpoint")
        onnxblock.save_checkpoint((trainable_params, non_trainable_params), checkpoint_file_path)
        # Then
        assert os.path.exists(checkpoint_file_path)


def test_load_checkpoint():
    # Given
    device = "cuda"
    batch_size, input_size, hidden_size, output_size = 64, 784, 500, 10
    _, zero_onnx_model = _get_models(device, batch_size, input_size, hidden_size, output_size, zero_flag=True)
    for initializer in zero_onnx_model.graph.initializer:
        zero_np = onnx.numpy_helper.to_array(initializer)
        assert np.allclose(zero_np, np.zeros(zero_np.shape))

    _, onnx_model = _get_models(device, batch_size, input_size, hidden_size, output_size)

    # Copy of onnx_model for comparison
    onnx_model_copy = copy.deepcopy(onnx_model)

    simple_block = SimpleTrainingBlockWithMSELoss()

    # When
    simple_block.requires_grad("fc2.weight", True)
    simple_block.requires_grad("fc1.bias", True)
    simple_block.requires_grad("fc1.weight", True)
    simple_block.requires_grad("fc2.bias", True)

    with onnxblock.base(onnx_model):
        _ = simple_block(onnx_model.graph.output[0].name)
    trainable_params, non_trainable_params = simple_block.parameters()

    with tempfile.TemporaryDirectory() as checkpoint_dir_name:
        checkpoint_file_path = os.path.join(checkpoint_dir_name, "checkpoint")
        onnxblock.save_checkpoint((trainable_params, non_trainable_params), checkpoint_file_path)

        # Load checkpoint parameters to the new simple model
        onnxblock.load_checkpoint_to_model(checkpoint_file_path, zero_onnx_model)

        # Then
        onnx_model_copy.graph.initializer.sort(key=lambda x: x.name)
        zero_onnx_model.graph.initializer.sort(key=lambda x: x.name)

        for i, initializer in enumerate(onnx_model_copy.graph.initializer):
            if initializer.name in ["fc2.weight", "fc1.bias"]:
                onnx_np = onnx.numpy_helper.to_array(onnx_model_copy.graph.initializer[i])
                zero_np = onnx.numpy_helper.to_array(zero_onnx_model.graph.initializer[i])
                assert np.allclose(onnx_np, zero_np)


def test_set_requires_grad_on_parameters():
    # Given
    device = "cuda"
    batch_size, input_size, hidden_size, output_size = 64, 784, 500, 10
    _, base_model = _get_models(device, batch_size, input_size, hidden_size, output_size)

    simple_block = SimpleTrainingBlockWithMSELoss()

    # When
    simple_block.requires_grad("fc2.weight", False)
    simple_block.requires_grad("fc1.bias", False)

    with onnxblock.base(base_model):
        _ = simple_block(base_model.graph.output[0].name)
    trainable_params, non_trainable_params = simple_block.parameters()

    # Then
    expected_trainable_parameters = {"fc1.weight", "fc2.bias"}
    expected_non_trainable_parameters = {"fc2.weight", "fc1.bias"}
    for param in trainable_params:
        assert param.name in expected_trainable_parameters
    for param in non_trainable_params:
        assert param.name in expected_non_trainable_parameters


def test_set_requires_grad_on_inputs():
    # Given
    device = "cuda"
    batch_size, input_size, hidden_size, output_size = 64, 784, 500, 10
    _, base_model = _get_models(device, batch_size, input_size, hidden_size, output_size)

    # When
    simple_block = SimpleTrainingBlockWithMSELoss()
    simple_block.requires_grad("input-0")
    with onnxblock.base(base_model):
        _ = simple_block(base_model.graph.output[0].name)

    onnx_model, _ = simple_block.to_model_proto()

    # Then
    expected_gradient_buffer_name = "input-0_grad.accumulation.buffer"
    expectedt_gradient_output_name = "input-0_grad.accumulation.out"
    graph_input_names = {graph_input.name for graph_input in onnx_model.graph.input}
    graph_output_names = {graph_output.name for graph_output in onnx_model.graph.output}

    assert expected_gradient_buffer_name in graph_input_names
    assert expectedt_gradient_output_name in graph_output_names


@pytest.mark.parametrize("model_type", [onnxblock.ForwardBlock, onnxblock.TrainingBlock])
def test_weighted_average_model_composition(model_type):
    # Given
    class TwoOutputNet(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super().__init__()

            self.fc1_1 = torch.nn.Linear(input_size, hidden_size)
            self.relu1 = torch.nn.ReLU()
            self.fc1_2 = torch.nn.Linear(hidden_size, num_classes)

            self.fc2_1 = torch.nn.Linear(input_size, hidden_size)
            self.relu2 = torch.nn.ReLU()
            self.fc2_2 = torch.nn.Linear(hidden_size, num_classes)

        def forward(self, model_input1, model_input2):
            out1 = self.fc1_2(self.relu1(self.fc1_1(model_input1)))
            out2 = self.fc2_2(self.relu2(self.fc2_1(model_input2)))
            return out1, out2

    class WeightedAvg(model_type):
        def __init__(self, w1, w2):
            super().__init__()

            self.loss1 = onnxblock.loss.CrossEntropyLoss()
            self.loss2 = onnxblock.loss.CrossEntropyLoss()
            self.w1 = onnxblock.blocks.Constant(w1)
            self.w2 = onnxblock.blocks.Constant(w2)
            self.mul = onnxblock.blocks.Mul()
            self.add = onnxblock.blocks.Add()

        def build(self, loss_input_name1, loss_input_name2):
            return self.add(
                self.mul(self.w1(), self.loss1(loss_input_name1, labels_name="labels1")),
                self.mul(self.w2(), self.loss2(loss_input_name2, labels_name="labels2")),
            )

    device = "cuda"
    batch_size, input_size, hidden_size, output_size = 64, 784, 500, 10
    pt_model = TwoOutputNet(input_size, hidden_size, output_size).to(device)
    x1 = torch.randn(batch_size, input_size, device=device)
    x2 = torch.randn(batch_size, input_size, device=device)
    base_model = _get_onnx_model(pt_model, (x1, x2))

    # When / Then no error occurs
    weighted_model = WeightedAvg(random.random(), random.random())
    with onnxblock.base(base_model):
        _ = weighted_model(base_model.graph.output[0].name, base_model.graph.output[1].name)


def test_grad_clipping_execution():
    # Given
    device = "cuda"
    batch_size, input_size, hidden_size, output_size = 64, 784, 500, 10
    pt_model, _ = _get_models(device, batch_size, input_size, hidden_size, output_size)
    x = torch.randn(batch_size, input_size, device=device)
    target = torch.randn(batch_size, output_size, device=device)

    # Prepare the onnx model with only grad clipping
    base_model = onnx.ModelProto()
    base_model.graph.name = "ClipGradNorm Model"
    base_model.producer_name = "grad clipping test"
    base_model.opset_import.extend(
        [
            onnx.helper.make_opsetid("com.microsoft", 1),
            onnx.helper.make_opsetid("", onnx.defs.onnx_opset_version()),
        ]
    )
    base_model.ir_version = onnx.IR_VERSION

    class GradClippingModel(onnxblock.ForwardBlock):
        def __init__(self, max_norm):
            super().__init__()
            self._grad_clip = onnxblock.optim.ClipGradNorm(max_norm)

        def build(self, grads_name):
            return self._grad_clip(grads_name)

    base_model.graph.input.append(
        onnx.helper.make_tensor_sequence_value_info("gradients", onnx.TensorProto.FLOAT, None)
    )

    grad_clip = GradClippingModel(2.5)
    with onnxblock.base(base_model):
        ort_output_names = grad_clip("gradients")

    onnx_model = grad_clip.to_model_proto()

    onnx_model.graph.output.append(
        onnx.helper.make_tensor_sequence_value_info(ort_output_names, onnx.TensorProto.FLOAT, None)
    )

    def mse_loss(prediction, target):
        loss = torch.nn.MSELoss()
        return loss(prediction, target)

    # When
    with tempfile.NamedTemporaryFile(suffix=".onnx") as onnx_fo:
        onnx.save(onnx_model, onnx_fo.name)

        loss = mse_loss(pt_model(x), target)
        loss.backward()

        ort_inputs = {"gradients": []}
        for _, param in pt_model.named_parameters():
            ort_inputs["gradients"].append(_to_numpy(copy.deepcopy(param.grad)))

        torch.nn.utils.clip_grad_norm_(pt_model.parameters(), 2.5)

        # Then no error occurs when executing the model
        ort_session = onnxruntime.InferenceSession(onnx_fo.name, providers=C.get_available_providers())
        ort_outs = ort_session.run([ort_output_names], ort_inputs)

        # assert all the gradients are close
        for ort_grad, pt_param in zip(ort_outs[0], pt_model.parameters()):
            assert np.allclose(ort_grad, _to_numpy(pt_param.grad))


def test_additional_output_names():
    class DropoutModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = torch.nn.Dropout(p=0.5)

        def forward(self, x):
            return self.dropout(x)

    model = DropoutModel()
    onnx_model = _get_onnx_model(model, (torch.randn(1, 3, 224, 224),))

    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts.generate_artifacts(onnx_model, loss=artifacts.LossType.CrossEntropyLoss, artifact_directory=temp_dir)

        eval_model = onnx.load(os.path.join(temp_dir, "eval_model.onnx"))

        # Make sure only loss is the output
        assert len(eval_model.graph.output) == 1

        # Re-generate artifacts with additional output names
        artifacts.generate_artifacts(
            onnx_model,
            loss=artifacts.LossType.CrossEntropyLoss,
            artifact_directory=temp_dir,
            additional_output_names=["output-0"],
        )

        # Make sure the eval model has two outputs
        eval_model = onnx.load(os.path.join(temp_dir, "eval_model.onnx"))
        assert len(eval_model.graph.output) == 2


def test_eval_model_has_no_training_mode_dropout():
    class DropoutModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = torch.nn.Dropout(p=0.5)

        def forward(self, x):
            return self.dropout(x)

    model = DropoutModel()
    onnx_model = _get_onnx_model(model, (torch.randn(1, 3, 224, 224),))

    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts.generate_artifacts(onnx_model, loss=artifacts.LossType.CrossEntropyLoss, artifact_directory=temp_dir)

        eval_model = onnx.load(os.path.join(temp_dir, "eval_model.onnx"))

        flag = False
        for node in eval_model.graph.node:
            if node.op_type == "Dropout":
                assert not node.input[2]
                flag = True

        assert flag


def test_eval_model_has_no_training_mode_batchnorm():
    class BatchNormModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.batchnorm = torch.nn.BatchNorm2d(100)

        def forward(self, x):
            return self.batchnorm(x)

    model = BatchNormModel()
    onnx_model = _get_onnx_model(model, (torch.randn(20, 100, 35, 45),))

    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts.generate_artifacts(onnx_model, loss=artifacts.LossType.CrossEntropyLoss, artifact_directory=temp_dir)

        eval_model = onnx.load(os.path.join(temp_dir, "eval_model.onnx"))

        flag = False
        for node in eval_model.graph.node:
            if node.op_type == "BatchNormalization":
                for attr in node.attribute:
                    if attr.name == "training_mode":
                        assert attr.i == 0
                        flag = True

        assert flag


def test_label_encoder_composition():
    device = "cuda"
    batch_size, input_size, hidden_size, output_size = 64, 784, 500, 10
    _, base_model = _get_models(device, batch_size, input_size, hidden_size, output_size)
    base_model.opset_import.append(
        onnx.helper.make_opsetid("ai.onnx.ml", onnx.defs.onnx_opset_version()),
    )

    all_nodes = [node.op_type for node in base_model.graph.node]
    assert "LabelEncoder" not in all_nodes

    class SCELossWithLabelEncoder(onnxblock.ForwardBlock):
        def __init__(self):
            super().__init__()
            self._loss = onnxblock.loss.CrossEntropyLoss()

        def build(self, output_name):
            keys_int64s = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            values_int64s = [521, 522, 523, 524, 525, 526, 527, 528, 529, 530]

            # Create a new graph input for the labels
            labels_name = "labels"
            labels_input = copy.deepcopy(self.base.graph.output[0])
            labels_input.type.tensor_type.elem_type = onnx.TensorProto.INT64
            labels_input.name = labels_name
            del labels_input.type.tensor_type.shape.dim[1]
            self.base.graph.input.append(labels_input)

            label_encoder = onnxblock.blocks.LabelEncoder(
                default_int64=521, keys_int64s=keys_int64s, values_int64s=values_int64s
            )

            return self._loss(output_name, label_encoder(labels_name))

    block = SCELossWithLabelEncoder()
    model = None
    with onnxblock.base(base_model):
        _ = block(base_model.graph.output[0].name)
        model = block.to_model_proto()

    all_nodes = [node.op_type for node in model.graph.node]
    assert "LabelEncoder" in all_nodes


def test_save_ort_format():
    device = "cpu"
    batch_size, input_size, hidden_size, output_size = 64, 784, 500, 10
    _, base_model = _get_models(device, batch_size, input_size, hidden_size, output_size)

    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts.generate_artifacts(
            base_model,
            requires_grad=["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"],
            loss=artifacts.LossType.CrossEntropyLoss,
            optimizer=artifacts.OptimType.AdamW,
            artifact_directory=temp_dir,
            ort_format=True,
        )

        assert os.path.exists(os.path.join(temp_dir, "training_model.onnx"))
        assert os.path.exists(os.path.join(temp_dir, "training_model.ort"))
        assert os.path.exists(os.path.join(temp_dir, "eval_model.onnx"))
        assert os.path.exists(os.path.join(temp_dir, "eval_model.ort"))
        assert os.path.exists(os.path.join(temp_dir, "optimizer_model.onnx"))
        assert os.path.exists(os.path.join(temp_dir, "optimizer_model.ort"))
