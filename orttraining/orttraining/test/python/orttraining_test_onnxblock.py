import torch
import io
import onnx
import onnxruntime.training.onnxblock as onnxblock
import onnxruntime
from onnxruntime.capi import _pybind_state as C
import pytest
import tempfile
import os
import copy
import numpy as np


# PyTorch Module definitions


class SimpleNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNet, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, model_input):
        out = self.fc1(model_input)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# onnxblock Graph definitions


class SimpleGraphWithMSELoss(onnxblock.Graph):
    def __init__(self, base_model):
        super(SimpleGraphWithMSELoss, self).__init__()
        self.loss = onnxblock.loss.MSELoss()
        self.base_model = base_model

    def build(self):
        outputs = self.base_model.graph.output
        lossful_graph = self.loss(self.base_model, outputs[0].name)
        return lossful_graph


class SimpleGraphWithCrossEntropyLoss(onnxblock.Graph):
    def __init__(self, base_model):
        super(SimpleGraphWithCrossEntropyLoss, self).__init__()
        self.loss = onnxblock.loss.CrossEntropyLoss()
        self.base_model = base_model

    def build(self):
        outputs = self.base_model.graph.output
        lossful_graph = self.loss(self.base_model, outputs[0].name)
        return lossful_graph


class SimpleTrainingGraphWithMSELoss(onnxblock.TrainingGraph):
    def __init__(self, base_model):
        super(SimpleTrainingGraphWithMSELoss, self).__init__()
        self.loss = onnxblock.loss.MSELoss()
        self.base_model = base_model

    def build(self):
        outputs = self.base_model.graph.output
        lossful_graph = self.loss(self.base_model, outputs[0].name)
        return lossful_graph


class SimpleTrainingGraphWithCrossEntropyLoss(onnxblock.TrainingGraph):
    def __init__(self, base_model):
        super(SimpleTrainingGraphWithCrossEntropyLoss, self).__init__()
        self.loss = onnxblock.loss.CrossEntropyLoss()
        self.base_model = base_model

    def build(self):
        outputs = self.base_model.graph.output
        lossful_graph = self.loss(self.base_model, outputs[0].name)
        return lossful_graph


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
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


# All unit tests


@pytest.mark.parametrize(
    "graph",
    [
        SimpleGraphWithMSELoss,
        SimpleGraphWithCrossEntropyLoss,
        SimpleTrainingGraphWithMSELoss,
        SimpleTrainingGraphWithCrossEntropyLoss,
    ],
)
def test_loss_composition(graph):
    # Given
    device = "cuda"
    N, D_in, H, D_out = 64, 784, 500, 10
    model = SimpleNet(D_in, H, D_out).to(device)
    x = torch.randn(N, D_in, device=device)
    onnx_model = _get_onnx_model(model, (x,))

    # When / Then no error occurs
    simple_graph = graph(onnx_model)
    lossful_graph = simple_graph()


def test_mse_loss_execution():
    # Given
    device = "cuda"
    N, D_in, H, D_out = 64, 784, 500, 10
    model = SimpleNet(D_in, H, D_out).to(device)
    x = torch.randn(N, D_in, device=device)
    target = torch.randn(N, D_out, device=device)
    onnx_model = _get_onnx_model(model, (copy.deepcopy(x),))
    simple_graph = SimpleGraphWithMSELoss(onnx_model)
    lossful_model = simple_graph()
    model_name = "model_with_mse_loss.onnx"
    ort_output_names = [lossful_model.graph.output[0].name]
    ort_inputs = {
        lossful_model.graph.input[0].name: _to_numpy(copy.deepcopy(x)),
        lossful_model.graph.input[1].name: _to_numpy(copy.deepcopy(target)),
    }

    def mse_loss(prediction, target):
        loss = torch.nn.MSELoss()
        return loss(prediction, target)

    # When
    with tempfile.TemporaryDirectory() as onnx_dir_name:
        onnx_file_path = os.path.join(onnx_dir_name, model_name)
        onnx.save(lossful_model, onnx_file_path)
        ort_session = onnxruntime.InferenceSession(
            onnx_file_path, providers=C.get_available_providers()
        )

        ort_outs = ort_session.run(ort_output_names, ort_inputs)
        torch_outs = mse_loss(model(copy.deepcopy(x)), copy.deepcopy(target))

        # Then
        assert np.allclose(ort_outs[0], _to_numpy(torch_outs))


def test_crossentropy_loss_execution():
    # Given
    device = "cuda"
    N, D_in, H, D_out = 64, 784, 500, 10
    model = SimpleNet(D_in, H, D_out).to(device)
    x = torch.randn(N, D_in, device=device)
    target = torch.randint(high=D_out, size=(N,), dtype=torch.int64, device=device)
    onnx_model = _get_onnx_model(model, (copy.deepcopy(x),))
    simple_graph = SimpleGraphWithCrossEntropyLoss(onnx_model)
    lossful_model = simple_graph()
    model_name = "model_with_crossentropy_loss.onnx"
    ort_output_names = [lossful_model.graph.output[0].name]
    ort_inputs = {
        lossful_model.graph.input[0].name: _to_numpy(copy.deepcopy(x)),
        lossful_model.graph.input[1].name: _to_numpy(
            copy.deepcopy(target).type(torch.int32)
        ),
    }

    def crossentropy_loss(prediction, target):
        loss = torch.nn.CrossEntropyLoss()
        return loss(prediction, target)

    # When
    with tempfile.TemporaryDirectory() as onnx_dir_name:
        onnx_file_path = os.path.join(onnx_dir_name, model_name)
        onnx.save(lossful_model, onnx_file_path)
        ort_session = onnxruntime.InferenceSession(
            onnx_file_path, providers=C.get_available_providers()
        )

        ort_outs = ort_session.run(ort_output_names, ort_inputs)
        torch_outs = crossentropy_loss(model(copy.deepcopy(x)), copy.deepcopy(target))

        # Then
        assert np.allclose(ort_outs[0], _to_numpy(torch_outs))


def test_mse_loss_training_graph_execution():
    # Given
    device = "cuda"
    N, D_in, H, D_out = 64, 784, 500, 10
    model = SimpleNet(D_in, H, D_out).to(device)
    x = torch.randn(N, D_in, device=device)
    target = torch.randn(N, D_out, device=device)
    onnx_model = _get_onnx_model(model, (copy.deepcopy(x),))
    simple_graph = SimpleTrainingGraphWithMSELoss(onnx_model)
    lossful_model = simple_graph()
    model_name = "model_with_mse_loss.onnx"
    ort_output_names = [lossful_model.graph.output[0].name]
    for name, _ in model.named_parameters():
        ort_output_names.append(f"{name}_grad.accumulation.out")
    ort_inputs = {
        lossful_model.graph.input[0].name: _to_numpy(copy.deepcopy(x)),
        lossful_model.graph.input[1].name: _to_numpy(copy.deepcopy(target)),
    }
    for name, param in model.named_parameters():
        ort_inputs[name] = _to_numpy(copy.deepcopy(param))
        ort_inputs[f"{name}_grad.accumulation.buffer"] = _to_numpy(
            torch.zeros_like(param)
        )
    ort_inputs["lazy_reset_grad"] = np.full(1, True)

    def mse_loss(prediction, target):
        loss = torch.nn.MSELoss()
        return loss(prediction, target)

    # When
    with tempfile.TemporaryDirectory() as onnx_dir_name:
        onnx_file_path = os.path.join(onnx_dir_name, model_name)
        onnx.save(lossful_model, onnx_file_path)
        ort_session = onnxruntime.InferenceSession(
            onnx_file_path, providers=C.get_available_providers()
        )

        ort_outs = ort_session.run(ort_output_names, ort_inputs)
        torch_outs = mse_loss(model(copy.deepcopy(x)), copy.deepcopy(target))
        torch_outs.backward()

        # Then
        # assert loss is close
        assert np.allclose(ort_outs[0], _to_numpy(torch_outs))

        # assert all the gradients are close
        for ort_grad, pt_param in zip(ort_outs[1:], model.parameters()):
            assert np.allclose(ort_grad, _to_numpy(pt_param.grad))


def test_crossentropy_loss_training_graph_execution():
    # Given
    device = "cuda"
    N, D_in, H, D_out = 64, 784, 500, 10
    model = SimpleNet(D_in, H, D_out).to(device)
    x = torch.randn(N, D_in, device=device)
    target = torch.randint(high=D_out, size=(N,), dtype=torch.int64, device=device)
    onnx_model = _get_onnx_model(model, (copy.deepcopy(x),))
    simple_graph = SimpleTrainingGraphWithCrossEntropyLoss(onnx_model)
    lossful_model = simple_graph()
    model_name = "model_with_crossentropy_loss.onnx"
    ort_output_names = [lossful_model.graph.output[0].name]
    for name, _ in model.named_parameters():
        ort_output_names.append(f"{name}_grad.accumulation.out")
    ort_inputs = {
        lossful_model.graph.input[0].name: _to_numpy(copy.deepcopy(x)),
        lossful_model.graph.input[1].name: _to_numpy(
            copy.deepcopy(target).type(torch.int32)
        ),
    }
    for name, param in model.named_parameters():
        ort_inputs[name] = _to_numpy(copy.deepcopy(param))
        ort_inputs[f"{name}_grad.accumulation.buffer"] = _to_numpy(
            torch.zeros_like(param)
        )
    ort_inputs["lazy_reset_grad"] = np.full(1, True)

    def crossentropy_loss(prediction, target):
        loss = torch.nn.CrossEntropyLoss()
        return loss(prediction, target)

    # When
    with tempfile.TemporaryDirectory() as onnx_dir_name:
        onnx_file_path = os.path.join(onnx_dir_name, model_name)
        onnx.save(lossful_model, onnx_file_path)
        ort_session = onnxruntime.InferenceSession(
            onnx_file_path, providers=C.get_available_providers()
        )

        ort_outs = ort_session.run(ort_output_names, ort_inputs)
        torch_outs = crossentropy_loss(model(copy.deepcopy(x)), copy.deepcopy(target))
        torch_outs.backward()

        # Then
        # assert loss is close
        assert np.allclose(ort_outs[0], _to_numpy(torch_outs))

        # assert all the gradients are close
        for ort_grad, pt_param in zip(ort_outs[1:], model.parameters()):
            assert np.allclose(ort_grad, _to_numpy(pt_param.grad))


@pytest.mark.parametrize(
    "graph", [SimpleTrainingGraphWithMSELoss, SimpleTrainingGraphWithCrossEntropyLoss]
)
def test_adamw_optimizer_composition(graph):
    # Given
    device = "cuda"
    N, D_in, H, D_out = 64, 784, 500, 10
    model = SimpleNet(D_in, H, D_out).to(device)
    x = torch.randn(N, D_in, device=device)
    onnx_model = _get_onnx_model(model, (x,))

    # When / Then no error occurs
    simple_graph = graph(onnx_model)
    lossful_graph = simple_graph()
    optimizer = onnxblock.optim.AdamW()
    optimizer_model = optimizer(lossful_graph)


def test_adamw_optimizer_execution():
    # Given
    device = "cuda"
    N, D_in, H, D_out = 64, 784, 500, 10
    model = SimpleNet(D_in, H, D_out).to(device)
    x = torch.randn(N, D_in, device=device)
    target = torch.randn(N, D_out, device=device)
    onnx_model = _get_onnx_model(model, (x,))

    simple_graph = SimpleTrainingGraphWithMSELoss(onnx_model)
    lossful_graph = simple_graph()
    optimizer = onnxblock.optim.AdamW()
    optimizer_model = optimizer(lossful_graph)
    model_name = "optimizer_model.onnx"
    learning_rate = 0.001
    step = 1
    ort_output_names = []
    for name, _ in model.named_parameters():
        ort_output_names.append(f"{name}.out")

    def mse_loss(prediction, target):
        loss = torch.nn.MSELoss()
        return loss(prediction, target)

    # When
    with tempfile.TemporaryDirectory() as onnx_dir_name:
        onnx_file_path = os.path.join(onnx_dir_name, model_name)
        onnx.save(optimizer_model, onnx_file_path)

        loss = mse_loss(model(copy.deepcopy(x)), copy.deepcopy(target))
        loss.backward()

        ort_inputs = {
            "learning_rate": np.full(1, learning_rate, dtype=np.float32),
            "step": np.full(1, step, dtype=np.int64),
        }
        for name, param in model.named_parameters():
            ort_inputs[name] = _to_numpy(copy.deepcopy(param))
            ort_inputs[f"{name}_grad.accumulation.out"] = _to_numpy(
                copy.deepcopy(param.grad)
            )
            ort_inputs[f"{name}.exp_avg"] = _to_numpy(torch.zeros_like(param))
            ort_inputs[f"{name}.exp_avg_sq"] = _to_numpy(torch.zeros_like(param))

        # Then no error occurs when executing the model
        ort_session = onnxruntime.InferenceSession(
            onnx_file_path, providers=C.get_available_providers()
        )
        ort_outs = ort_session.run(ort_output_names, ort_inputs)


def test_retrieve_parameters():
    # Given
    device = "cuda"
    N, D_in, H, D_out = 64, 784, 500, 10
    model = SimpleNet(D_in, H, D_out).to(device)
    x = torch.randn(N, D_in, device=device)
    onnx_model = _get_onnx_model(model, (x,))

    simple_graph = SimpleTrainingGraphWithMSELoss(onnx_model)
    training_model = simple_graph()

    # When
    trainable_params, non_trainable_params = simple_graph.parameters()

    # Then
    assert not non_trainable_params
    for ort_param, (pt_param_name, pt_param) in zip(
        trainable_params, model.named_parameters()
    ):
        assert ort_param.name == pt_param_name
        assert np.allclose(
            np.frombuffer(ort_param.raw_data, dtype=np.float32).reshape(pt_param.shape),
            _to_numpy(pt_param),
        )


def test_retrieve_parameters_before_building_gradient_graph():
    # Given
    device = "cuda"
    N, D_in, H, D_out = 64, 784, 500, 10
    model = SimpleNet(D_in, H, D_out).to(device)
    x = torch.randn(N, D_in, device=device)
    onnx_model = _get_onnx_model(model, (x,))

    simple_graph = SimpleTrainingGraphWithMSELoss(onnx_model)

    # When / Then
    with pytest.raises(Exception) as ex_info:
        trainable_params, non_trainable_params = simple_graph.parameters()
    assert (
        "Please build the training graph first before trying to retrieve the parameters."
        in str(ex_info.value)
    )


def test_save_checkpoint():
    # Given
    device = "cuda"
    N, D_in, H, D_out = 64, 784, 500, 10
    model = SimpleNet(D_in, H, D_out).to(device)
    x = torch.randn(N, D_in, device=device)
    onnx_model = _get_onnx_model(model, (x,))

    simple_graph = SimpleTrainingGraphWithMSELoss(onnx_model)
    training_model = simple_graph()
    trainable_params, non_trainable_params = simple_graph.parameters()

    # When
    with tempfile.TemporaryDirectory() as checkpoint_dir_name:
        checkpoint_file_path = os.path.join(checkpoint_dir_name, "checkpoint")
        onnxblock.save_checkpoint(
            (trainable_params, non_trainable_params), checkpoint_file_path
        )

        # Then
        assert os.path.exists(checkpoint_file_path)


def test_set_requires_grad_on_parameters():
    # Given
    device = "cuda"
    N, D_in, H, D_out = 64, 784, 500, 10
    model = SimpleNet(D_in, H, D_out).to(device)
    x = torch.randn(N, D_in, device=device)
    onnx_model = _get_onnx_model(model, (x,))

    simple_graph = SimpleTrainingGraphWithMSELoss(onnx_model)

    # When
    simple_graph.requires_grad("fc2.weight", False)
    simple_graph.requires_grad("fc1.bias", False)
    training_model = simple_graph()
    trainable_params, non_trainable_params = simple_graph.parameters()

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
    N, D_in, H, D_out = 64, 784, 500, 10
    model = SimpleNet(D_in, H, D_out).to(device)
    x = torch.randn(N, D_in, device=device)
    onnx_model = _get_onnx_model(model, (x,))

    simple_graph = SimpleTrainingGraphWithMSELoss(onnx_model)

    # When
    simple_graph.requires_grad("input-0")
    training_model = simple_graph()
    trainable_params, non_trainable_params = simple_graph.parameters()

    # Then
    expected_input_gradient_buffer_name = "input-0_grad.accumulation.buffer"
    expected_input_gradient_output_name = "input-0_grad.accumulation.out"
    graph_input_names = {graph_input.name for graph_input in training_model.graph.input}
    graph_output_names = {
        graph_output.name for graph_output in training_model.graph.output
    }

    assert expected_input_gradient_buffer_name in graph_input_names
    assert expected_input_gradient_output_name in graph_output_names
