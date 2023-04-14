# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import tempfile

import numpy as np
import pytest
import torch
from orttraining_test_onnxblock import _get_models

import onnxruntime.training.onnxblock as onnxblock
from onnxruntime.training import artifacts
from onnxruntime.training.api import CheckpointState, LinearLRScheduler, Module, Optimizer


class SimpleModelWithCrossEntropyLoss(onnxblock.TrainingBlock):
    def __init__(self):
        super().__init__()
        self.loss = onnxblock.loss.CrossEntropyLoss()

    def build(self, output_name):
        return self.loss(output_name)


def _create_training_artifacts(artifact_directory: str | os.PathLike):
    device = "cpu"
    batch_size, input_size, hidden_size, output_size = 64, 784, 500, 10
    pt_model, onnx_model = _get_models(device, batch_size, input_size, hidden_size, output_size)

    requires_grad = [name for name, param in pt_model.named_parameters() if param.requires_grad]
    frozen_params = [name for name, param in pt_model.named_parameters() if not param.requires_grad]

    artifacts.generate_artifacts(
        onnx_model,
        optimizer=artifacts.OptimType.AdamW,
        loss=artifacts.LossType.CrossEntropyLoss,
        requires_grad=requires_grad,
        frozen_params=frozen_params,
        artifact_directory=artifact_directory,
    )

    training_model_file = os.path.join(artifact_directory, "training_model.onnx")
    eval_model_file = os.path.join(artifact_directory, "eval_model.onnx")
    optimizer_model_file = os.path.join(artifact_directory, "optimizer_model.onnx")
    checkpoint_file = os.path.join(artifact_directory, "checkpoint")

    return checkpoint_file, training_model_file, eval_model_file, optimizer_model_file, pt_model


def test_train_step():
    # Generating random data for testing.
    inputs = torch.randn(64, 784).numpy()
    labels = torch.randint(high=10, size=(64,), dtype=torch.int64).numpy()

    with tempfile.TemporaryDirectory() as temp_dir:
        (
            checkpoint_file_path,
            training_model_file_path,
            _,
            _,
            pt_model,
        ) = _create_training_artifacts(temp_dir)
        # Create Checkpoint State.
        state = CheckpointState.load_checkpoint(checkpoint_file_path)
        # Create a Module.
        print(training_model_file_path)
        model = Module(training_model_file_path, state)
        model.train()
        ort_loss = model(inputs, labels)

        # Calculate loss using pytorch model to compare it with Module's output.
        pt_outputs = pt_model(torch.from_numpy(inputs))
        loss_fn = torch.nn.CrossEntropyLoss()
        pt_loss = loss_fn(pt_outputs, torch.from_numpy(labels).long())

        assert np.allclose(ort_loss, pt_loss.detach().numpy())


def test_eval_step():
    # Generating random data for testing.
    inputs = torch.randn(64, 784).numpy()
    labels = torch.randint(high=10, size=(64,), dtype=torch.int64).numpy()

    with tempfile.TemporaryDirectory() as temp_dir:
        (
            checkpoint_file_path,
            training_model_file_path,
            eval_model_file_path,
            _,
            _,
        ) = _create_training_artifacts(temp_dir)
        # Create Checkpoint State.
        state = CheckpointState.load_checkpoint(checkpoint_file_path)
        # Create a Module.
        model = Module(training_model_file_path, state, eval_model_file_path)
        model.train()
        model(inputs, labels)

        model.eval()
        fetches = model(inputs, labels)
        assert fetches


def test_optimizer_step():
    # Generating random data for testing.
    inputs = torch.randn(64, 784).numpy()
    labels = torch.randint(high=10, size=(64,), dtype=torch.int64).numpy()

    with tempfile.TemporaryDirectory() as temp_dir:
        (
            checkpoint_file_path,
            training_model_file_path,
            _,
            optimizer_model_file_path,
            _,
        ) = _create_training_artifacts(temp_dir)
        # Create Checkpoint State.
        state = CheckpointState.load_checkpoint(checkpoint_file_path)
        # Create a Module and Optimizer.
        model = Module(training_model_file_path, state)
        optimizer = Optimizer(optimizer_model_file_path, model)

        model.train()
        old_flatten_params = model.get_contiguous_parameters()
        model(inputs, labels)

        optimizer.step()
        new_params = model.get_contiguous_parameters()
        # Assert that the parameters are updated.
        assert not np.array_equal(old_flatten_params.numpy(), new_params.numpy())


def test_get_and_set_lr():
    with tempfile.TemporaryDirectory() as temp_dir:
        (
            checkpoint_file_path,
            training_model_file_path,
            _,
            optimizer_model_file_path,
            _,
        ) = _create_training_artifacts(temp_dir)
        # Create Checkpoint State.
        state = CheckpointState.load_checkpoint(checkpoint_file_path)
        # Create a Module and Optimizer.
        model = Module(training_model_file_path, state)
        optimizer = Optimizer(optimizer_model_file_path, model)

        # Test get and set learning rate.
        lr = optimizer.get_learning_rate()
        assert round(lr, 3) == 0.001

        optimizer.set_learning_rate(0.5)
        new_lr = optimizer.get_learning_rate()

        assert np.isclose(new_lr, 0.5)
        assert lr != new_lr


def test_scheduler_step():
    # Generating random data for testing.
    inputs = torch.randn(64, 784).numpy()
    labels = torch.randint(high=10, size=(64,), dtype=torch.int64).numpy()

    with tempfile.TemporaryDirectory() as temp_dir:
        (
            checkpoint_file_path,
            training_model_file_path,
            _,
            optimizer_model_file_path,
            _,
        ) = _create_training_artifacts(temp_dir)
        # Create Checkpoint State.
        state = CheckpointState.load_checkpoint(checkpoint_file_path)
        # Create a Module and Optimizer.
        model = Module(training_model_file_path, state)
        optimizer = Optimizer(optimizer_model_file_path, model)
        scheduler = LinearLRScheduler(optimizer, 1, 2, 0.2)

        # Test get and set learning rate.
        lr = optimizer.get_learning_rate()
        assert np.allclose(lr, 0.0)

        model.train()
        model(inputs, labels)
        optimizer.step()
        scheduler.step()

        # Get new learning rate.
        new_lr = optimizer.get_learning_rate()
        assert new_lr != lr


def test_training_module_checkpoint():
    # Generating random data for testing.
    inputs = torch.randn(64, 784).numpy()
    labels = torch.randint(high=10, size=(64,), dtype=torch.int64).numpy()

    with tempfile.TemporaryDirectory() as temp_dir:
        (
            checkpoint_file_path,
            training_model_file_path,
            _,
            _,
            _,
        ) = _create_training_artifacts(temp_dir)
        # Create Checkpoint State.
        state = CheckpointState.load_checkpoint(checkpoint_file_path)
        # Create a Training Module and Training Optimizer.
        model = Module(training_model_file_path, state)

        model.train()
        model(inputs, labels)

        checkpoint_save_path = os.path.join(temp_dir, "checkpoint_export.ckpt")

        session_state = model.get_state()
        CheckpointState.save_checkpoint(session_state, checkpoint_save_path)
        old_flatten_params = model.get_contiguous_parameters()

        # Assert the checkpoint was saved.
        assert os.path.exists(checkpoint_save_path)

        # Assert the checkpoint parameters remain after saving.
        state = CheckpointState.load_checkpoint(checkpoint_save_path)
        new_model = Module(training_model_file_path, state)

        new_params = new_model.get_contiguous_parameters()

        assert np.array_equal(old_flatten_params.numpy(), new_params.numpy())


def test_copy_buffer_to_parameters():
    # Generating random data for testing.
    inputs = torch.randn(64, 784).numpy()
    labels = torch.randint(high=10, size=(64,), dtype=torch.int64).numpy()

    with tempfile.TemporaryDirectory() as temp_dir:
        (
            checkpoint_file_path,
            training_model_file_path,
            _,
            optimizer_model_file_path,
            _,
        ) = _create_training_artifacts(temp_dir)
        state = CheckpointState.load_checkpoint(checkpoint_file_path)

        # Create a Module and Optimizer.
        model = Module(training_model_file_path, state)
        optimizer = Optimizer(optimizer_model_file_path, model)

        # Keep a copy of the parameters.
        old_output_params = model.get_contiguous_parameters()

        # Run a Training Step.
        model.train()
        model(inputs, labels)
        optimizer.step()

        # Get the new parameters.
        output_params = model.get_contiguous_parameters()
        # Make sure old params are different from new params.
        assert not np.array_equal(old_output_params.numpy(), output_params.numpy())

        # Copy the old parameters to the model.
        model.copy_buffer_to_parameters(old_output_params)

        # Get the saved parameters.
        saved_params = model.get_contiguous_parameters()

        # Make sure the saved parameters are the same as the old parameters.
        assert np.array_equal(old_output_params.numpy(), saved_params.numpy())


def test_export_model_for_inferencing():
    with tempfile.TemporaryDirectory() as temp_dir:
        (
            checkpoint_file_path,
            training_model_file_path,
            eval_model_file_path,
            _,
            _,
        ) = _create_training_artifacts(temp_dir)

        # Create Checkpoint State.
        state = CheckpointState.load_checkpoint(checkpoint_file_path)

        # Create a Module.
        model = Module(training_model_file_path, state, eval_model_file_path)

        # Export inference model
        inference_model_file_path = os.path.join(temp_dir, "inference_model.onnx")
        model.export_model_for_inferencing(inference_model_file_path, ["output-0"])
        assert os.path.exists(inference_model_file_path)


def test_cuda_execution_provider():
    with tempfile.TemporaryDirectory() as temp_dir:
        (
            checkpoint_file_path,
            training_model_file_path,
            _,
            _,
            _,
        ) = _create_training_artifacts(temp_dir)

        # Create Checkpoint State.
        state = CheckpointState.load_checkpoint(checkpoint_file_path)
        # Create a Module.
        model = Module(training_model_file_path, state, device="cuda")
        params = model.get_contiguous_parameters()

        # Check if parameters are moved to cuda.
        assert params.device_name() == "Cuda"


@pytest.mark.parametrize(
    "property_value",
    [-1, 0, 1, 1234567890, -1.0, -0.1, 0.1, 1.0, 1234.0, "hello", "world", "onnxruntime"],
)
def test_add_property(property_value):
    with tempfile.TemporaryDirectory() as temp_dir:
        (
            checkpoint_file_path,
            _,
            _,
            _,
            _,
        ) = _create_training_artifacts(temp_dir)

        # Create Checkpoint State.
        state = CheckpointState.load_checkpoint(checkpoint_file_path)

        # Float values in python are double precision.
        # Convert to float32 to match the type of the property.
        if isinstance(property_value, float):
            property_value = float(np.float32(property_value))

        state["property"] = property_value
        assert "property" in state
        assert state["property"] == property_value


def test_get_input_output_names():
    with tempfile.TemporaryDirectory() as temp_dir:
        (
            checkpoint_file_path,
            training_model_file_path,
            eval_model_file_path,
            _,
            _,
        ) = _create_training_artifacts(temp_dir)

        # Create Checkpoint State.
        state = CheckpointState.load_checkpoint(checkpoint_file_path)

        # Create a Module.
        model = Module(training_model_file_path, state, eval_model_file_path)

        assert model.input_names() == ["input-0", "labels"]
        assert model.output_names() == ["onnx::loss::128"]
