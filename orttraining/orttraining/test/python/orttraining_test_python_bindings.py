import io
import os
import tempfile

import numpy as np
import onnx
import torch

import onnxruntime.training.onnxblock as onnxblock
from onnxruntime.capi.onnxruntime_inference_collection import OrtValue
from onnxruntime.capi.onnxruntime_pybind11_state import OrtValueVector
from onnxruntime.training.onnxblock import Module, Optimizer


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


class SimpleModelWithCrossEntropyLoss(onnxblock.TrainingModel):
    def __init__(self):
        super(SimpleModelWithCrossEntropyLoss, self).__init__()
        self.loss = onnxblock.loss.CrossEntropyLoss()

    def build(self, output_name):
        return self.loss(output_name)


def _get_models(device, batch_size, input_size, hidden_size, output_size, zero_flag=False):
    """Returns the pt and onnx models for SimpleNet"""
    pt_model = SimpleNet(input_size, hidden_size, output_size).to(device)
    pt_model.train()
    # setting all initial weights to zero
    if zero_flag:
        with torch.no_grad():
            for param in pt_model.parameters():
                param.zero_()

    x = torch.randn(batch_size, input_size, device=device)
    onnx_model = _get_onnx_model(pt_model, (x,))

    return pt_model, onnx_model


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


def _create_training_models():
    # Given
    device = "cuda"
    batch_size, input_size, hidden_size, output_size = 64, 784, 500, 10
    _, onnx_model = _get_models(device, batch_size, input_size, hidden_size, output_size)

    # Build the onnx model with loss
    simple_model = SimpleModelWithCrossEntropyLoss()
    with onnxblock.onnx_model(onnx_model) as accessor:
        _ = simple_model(onnx_model.graph.output[0].name)
        eval_model = accessor.eval_model

    optimizer = onnxblock.optim.AdamW()
    with onnxblock.onnx_model() as accessor:
        _ = optimizer(simple_model.parameters())
        optimizer_model = accessor.model

    return simple_model, onnx_model, optimizer_model, eval_model


def test_train_step():
    # Initialize Models
    simple_model, onnx_model, _, _ = _create_training_models()
    trainable_params, non_trainable_params = simple_model.parameters()

    # Generating random data for testing.
    inputs = torch.randn(64, 784).numpy()
    labels = torch.randint(high=10, size=(64,), dtype=torch.int32).numpy()
    forward_inputs = OrtValueVector()
    forward_inputs.reserve(2)
    forward_inputs.push_back(OrtValue.ortvalue_from_numpy(inputs)._ortvalue)
    forward_inputs.push_back(OrtValue.ortvalue_from_numpy(labels)._ortvalue)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save models & checkpoint files to load them later.
        checkpoint_file_path = os.path.join(temp_dir, "checkpoint")
        onnxblock.save_checkpoint((trainable_params, non_trainable_params), checkpoint_file_path)

        model_file_path = os.path.join(temp_dir, "training_model.onnx")
        onnx.save(onnx_model, model_file_path)

        # Create a Module.
        model = Module(model_file_path, checkpoint_file_path)
        model.train()
        fetches = model(forward_inputs)
        assert fetches


def test_eval_step():
    # Initialize Models
    simple_model, onnx_model, _, eval_model = _create_training_models()
    trainable_params, non_trainable_params = simple_model.parameters()

    # Generating random data for testing.
    # TODO : add utility function to convert python to OrtValueVector.
    inputs = torch.randn(64, 784).numpy()
    labels = torch.randint(high=10, size=(64,), dtype=torch.int32).numpy()
    forward_inputs = OrtValueVector()
    forward_inputs.reserve(2)
    forward_inputs.push_back(OrtValue.ortvalue_from_numpy(inputs)._ortvalue)
    forward_inputs.push_back(OrtValue.ortvalue_from_numpy(labels)._ortvalue)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save models & checkpoint files to load them later.
        checkpoint_file_path = os.path.join(temp_dir, "checkpoint")
        onnxblock.save_checkpoint((trainable_params, non_trainable_params), checkpoint_file_path)

        model_file_path = os.path.join(temp_dir, "training_model.onnx")
        onnx.save(onnx_model, model_file_path)

        eval_model_file_path = os.path.join(temp_dir, "eval_model.onnx")
        onnx.save(eval_model, eval_model_file_path)

        # Create a Module.
        model = Module(model_file_path, checkpoint_file_path, eval_model_file_path)
        model.train()
        model(forward_inputs)

        model.eval()
        fetches = model(forward_inputs)
        assert fetches


def test_optimizer_step():
    # Initialize Models
    simple_model, onnx_model, optimizer_model, _ = _create_training_models()
    trainable_params, non_trainable_params = simple_model.parameters()

    # Generating random data for testing.
    inputs = torch.randn(64, 784).numpy()
    labels = torch.randint(high=10, size=(64,), dtype=torch.int32).numpy()
    forward_inputs = OrtValueVector()
    forward_inputs.reserve(2)
    forward_inputs.push_back(OrtValue.ortvalue_from_numpy(inputs)._ortvalue)
    forward_inputs.push_back(OrtValue.ortvalue_from_numpy(labels)._ortvalue)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save models & checkpoint files to load them later.
        checkpoint_file_path = os.path.join(temp_dir, "checkpoint")
        onnxblock.save_checkpoint((trainable_params, non_trainable_params), checkpoint_file_path)

        model_file_path = os.path.join(temp_dir, "training_model.onnx")
        onnx.save(onnx_model, model_file_path)

        optimizer_file_path = os.path.join(temp_dir, "optimizer.onnx")
        onnx.save(optimizer_model, optimizer_file_path)

        # Create a Module and Optimizer.
        model = Module(model_file_path, checkpoint_file_path)
        optimizer = Optimizer(optimizer_file_path, model)

        model.train()
        model(forward_inputs)
        optimizer.step()
        # TODO : Check if parameters changed from before and after optimizer step.


def test_training_module_checkpoint():
    # Initialize Models
    simple_model, onnx_model, optimizer_model, _ = _create_training_models()
    trainable_params, non_trainable_params = simple_model.parameters()

    # Generating random data for testing.
    inputs = torch.randn(64, 784).numpy()
    labels = torch.randint(high=10, size=(64,), dtype=torch.int32).numpy()
    forward_inputs = OrtValueVector()
    forward_inputs.reserve(2)
    forward_inputs.push_back(OrtValue.ortvalue_from_numpy(inputs)._ortvalue)
    forward_inputs.push_back(OrtValue.ortvalue_from_numpy(labels)._ortvalue)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save models & checkpoint files to load them later.
        checkpoint_file_path = os.path.join(temp_dir, "checkpoint")
        onnxblock.save_checkpoint((trainable_params, non_trainable_params), checkpoint_file_path)

        model_file_path = os.path.join(temp_dir, "training_model.onnx")
        onnx.save(onnx_model, model_file_path)

        optimizer_file_path = os.path.join(temp_dir, "optimizer.onnx")
        onnx.save(optimizer_model, optimizer_file_path)

        # Create a Training Module and Training Optimizer.
        model = Module(model_file_path, checkpoint_file_path)

        model.train()
        model(forward_inputs)

        checkpoint_save_path = os.path.join(temp_dir, "checkpoint_export.ckpt")

        model.save_checkpoint(checkpoint_save_path)

        # TODO : Load checkpoint to a zeroed model and assert parameters are different.
        assert os.path.exists(checkpoint_save_path)

def test_get_delta_as_list():
    # Initialize Models
    simple_model, onnx_model, optimizer_model, _ = _create_training_models()
    trainable_params, non_trainable_params = simple_model.parameters()

    # Generating random data for testing.
    inputs = torch.randn(64, 784).numpy()
    labels = torch.randint(high=10, size=(64,), dtype=torch.int32).numpy()
    forward_inputs = OrtValueVector()
    forward_inputs.reserve(2)
    forward_inputs.push_back(OrtValue.ortvalue_from_numpy(inputs)._ortvalue)
    forward_inputs.push_back(OrtValue.ortvalue_from_numpy(labels)._ortvalue)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save models & checkpoint files to load them later.
        checkpoint_file_path = os.path.join(temp_dir, "checkpoint")
        onnxblock.save_checkpoint((trainable_params, non_trainable_params), checkpoint_file_path)

        model_file_path = os.path.join(temp_dir, "training_model.onnx")
        onnx.save(onnx_model, model_file_path)

        optimizer_file_path = os.path.join(temp_dir, "optimizer.onnx")
        onnx.save(optimizer_model, optimizer_file_path)

        # Create a Module and Optimizer.
        model = Module(model_file_path, checkpoint_file_path)
        optimizer = Optimizer(optimizer_file_path, model)

        old_output_params = model.get_contagious_parameters()
        print(old_output_params.numpy())

        model.train()
        model(forward_inputs)
        optimizer.step()

        output_params = model.get_contagious_parameters()
        print(output_params.numpy())

        delta = onnxblock.get_delta_as_list(output_params, old_output_params)
        print(delta.numpy())
        print(old_output_params.numpy()[0] , output_params.numpy()[0])
        print(np.float32(old_output_params.numpy()[0]) - np.float32(output_params.numpy()[0]), delta.numpy()[0])
        # TODO : Check if parameters changed from before and after optimizer step.
