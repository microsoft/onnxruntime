import os
import tempfile

import numpy as np
import onnx
import torch
from orttraining_test_onnxblock import _get_models

import onnxruntime.training.onnxblock as onnxblock
from onnxruntime.training.api import CheckpointState, LinearLRScheduler, Module, Optimizer


class SimpleModelWithCrossEntropyLoss(onnxblock.TrainingModel):
    def __init__(self):
        super(SimpleModelWithCrossEntropyLoss, self).__init__()
        self.loss = onnxblock.loss.CrossEntropyLoss()

    def build(self, output_name):
        return self.loss(output_name)


def _create_training_models():
    # Given
    device = "cpu"
    batch_size, input_size, hidden_size, output_size = 64, 784, 500, 10
    pt_model, onnx_model = _get_models(device, batch_size, input_size, hidden_size, output_size)

    # Build the onnx model with loss
    simple_model = SimpleModelWithCrossEntropyLoss()
    with onnxblock.onnx_model(onnx_model) as accessor:
        _ = simple_model(onnx_model.graph.output[0].name)
        eval_model = accessor.eval_model

    optimizer = onnxblock.optim.AdamW()
    with onnxblock.onnx_model() as accessor:
        _ = optimizer(simple_model.parameters())
        optimizer_model = accessor.model

    return simple_model, onnx_model, optimizer_model, eval_model, pt_model


def _get_test_models_path(directory, simple_model, onnx_model, optimizer_model=None, eval_model=None):
    trainable_params, non_trainable_params = simple_model.parameters()
    paths = []
    checkpoint_file_path = os.path.join(directory, "checkpoint")
    onnxblock.save_checkpoint((trainable_params, non_trainable_params), checkpoint_file_path)
    paths.append(checkpoint_file_path)

    model_file_path = os.path.join(directory, "training_model.onnx")
    onnx.save(onnx_model, model_file_path)
    paths.append(model_file_path)

    if optimizer_model:
        optimizer_file_path = os.path.join(directory, "optimizer.onnx")
        onnx.save(optimizer_model, optimizer_file_path)
        paths.append(optimizer_file_path)

    if eval_model:
        eval_model_file_path = os.path.join(directory, "eval_model.onnx")
        onnx.save(eval_model, eval_model_file_path)
        paths.append(eval_model_file_path)

    return tuple(paths)


def test_train_step():
    # Initialize Models
    simple_model, onnx_model, _, _, pt_model = _create_training_models()
    # Generating random data for testing.
    inputs = torch.randn(64, 784).numpy()
    labels = torch.randint(high=10, size=(64,), dtype=torch.int32).numpy()
    forward_inputs = [inputs, labels]

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save models & checkpoint files to load them later.
        checkpoint_file_path, model_file_path = _get_test_models_path(temp_dir, simple_model, onnx_model)
        # Create Checkpoint State.
        state = CheckpointState(checkpoint_file_path)
        # Create a Module.
        model = Module(model_file_path, state)
        model.train()
        fetches = model(forward_inputs)

        # Calculate loss using pytorch model to compare it with Module's output.
        pt_outputs = pt_model(torch.from_numpy(inputs))
        loss_fn = torch.nn.CrossEntropyLoss()
        pt_loss = loss_fn(pt_outputs, torch.from_numpy(labels).long())

        assert np.allclose(fetches[0], pt_loss.detach().numpy())


def test_eval_step():
    # Initialize Models
    simple_model, onnx_model, _, eval_model, _ = _create_training_models()

    # Generating random data for testing.
    # TODO : add utility function to convert numpy arrays to OrtValueVector.
    inputs = torch.randn(64, 784).numpy()
    labels = torch.randint(high=10, size=(64,), dtype=torch.int32).numpy()
    forward_inputs = [inputs, labels]

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save models & checkpoint files to load them later.
        checkpoint_file_path, model_file_path, eval_model_file_path = _get_test_models_path(
            temp_dir, simple_model, onnx_model, eval_model=eval_model
        )
        # Create Checkpoint State.
        state = CheckpointState(checkpoint_file_path)
        # Create a Module.
        model = Module(model_file_path, state, eval_model_file_path)
        model.train()
        model(forward_inputs)

        model.eval()
        fetches = model(forward_inputs)
        assert fetches


def test_optimizer_step():
    # Initialize Models
    simple_model, onnx_model, optimizer_model, _, _ = _create_training_models()

    # Generating random data for testing.
    inputs = torch.randn(64, 784).numpy()
    labels = torch.randint(high=10, size=(64,), dtype=torch.int32).numpy()
    forward_inputs = [inputs, labels]

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save models & checkpoint files to load them later.
        checkpoint_file_path, model_file_path, optimizer_file_path = _get_test_models_path(
            temp_dir, simple_model, onnx_model, optimizer_model=optimizer_model
        )
        # Create Checkpoint State.
        state = CheckpointState(checkpoint_file_path)
        # Create a Module and Optimizer.
        model = Module(model_file_path, state)
        optimizer = Optimizer(optimizer_file_path, model)

        model.train()
        old_flatten_params = model.get_contiguous_parameters()
        model(forward_inputs)

        optimizer.step()
        new_params = model.get_contiguous_parameters()
        # Assert that the parameters are updated.
        assert not np.array_equal(old_flatten_params.numpy(), new_params.numpy())


def test_get_and_set_lr():
    # Initialize Models
    simple_model, onnx_model, optimizer_model, _, _ = _create_training_models()

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save models & checkpoint files to load them later.
        checkpoint_file_path, model_file_path, optimizer_file_path = _get_test_models_path(
            temp_dir, simple_model, onnx_model, optimizer_model=optimizer_model
        )
        # Create Checkpoint State.
        state = CheckpointState(checkpoint_file_path)
        # Create a Module and Optimizer.
        model = Module(model_file_path, state)
        optimizer = Optimizer(optimizer_file_path, model)

        # Test get and set learning rate.
        lr = optimizer.get_learning_rate()
        assert round(lr, 3) == 0.001

        optimizer.set_learning_rate(0.5)
        new_lr = optimizer.get_learning_rate()

        assert np.isclose(new_lr, 0.5)
        assert lr != new_lr


def test_scheduler_step():
    # Initialize Models
    simple_model, onnx_model, optimizer_model, _, _ = _create_training_models()

    # Generating random data for testing.
    inputs = torch.randn(64, 784).numpy()
    labels = torch.randint(high=10, size=(64,), dtype=torch.int32).numpy()
    forward_inputs = [inputs, labels]

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save models & checkpoint files to load them later.
        checkpoint_file_path, model_file_path, optimizer_file_path = _get_test_models_path(
            temp_dir, simple_model, onnx_model, optimizer_model=optimizer_model
        )
        # Create Checkpoint State.
        state = CheckpointState(checkpoint_file_path)
        # Create a Module and Optimizer.
        model = Module(model_file_path, state)
        optimizer = Optimizer(optimizer_file_path, model)
        scheduler = LinearLRScheduler(optimizer, 1, 2, 0.2)

        # Test get and set learning rate.
        lr = optimizer.get_learning_rate()
        assert np.allclose(lr, 0.0)

        model.train()
        model(forward_inputs)
        optimizer.step()
        scheduler.step()

        # Get new learning rate.
        new_lr = optimizer.get_learning_rate()
        assert new_lr != lr


def test_training_module_checkpoint():
    # Initialize Models
    simple_model, onnx_model, _, _, _ = _create_training_models()

    # Generating random data for testing.
    inputs = torch.randn(64, 784).numpy()
    labels = torch.randint(high=10, size=(64,), dtype=torch.int32).numpy()
    forward_inputs = [inputs, labels]

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save models & checkpoint files to load them later.
        checkpoint_file_path, model_file_path = _get_test_models_path(temp_dir, simple_model, onnx_model)
        # Create Checkpoint State.
        state = CheckpointState(checkpoint_file_path)
        # Create a Training Module and Training Optimizer.
        model = Module(model_file_path, state)

        model.train()
        model(forward_inputs)

        checkpoint_save_path = os.path.join(temp_dir, "checkpoint_export.ckpt")

        model.save_checkpoint(checkpoint_save_path)
        old_flatten_params = model.get_contiguous_parameters()

        # Assert the checkpoint was saved.
        assert os.path.exists(checkpoint_save_path)

        # Assert the checkpoint parameters remain after saving.
        state = CheckpointState(checkpoint_save_path)
        new_model = Module(model_file_path, state)

        new_params = new_model.get_contiguous_parameters()

        assert np.array_equal(old_flatten_params.numpy(), new_params.numpy())


def test_copy_buffer_to_parameters():
    # Initialize Models
    simple_model, onnx_model, optimizer_model, _, _ = _create_training_models()

    # Generating random data for testing.
    inputs = torch.randn(64, 784).numpy()
    labels = torch.randint(high=10, size=(64,), dtype=torch.int32).numpy()
    forward_inputs = [inputs, labels]

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save models & checkpoint files to load them later.
        checkpoint_file_path, model_file_path, optimizer_file_path = _get_test_models_path(
            temp_dir, simple_model, onnx_model, optimizer_model=optimizer_model
        )
        state = CheckpointState(checkpoint_file_path)

        # Create a Module and Optimizer.
        model = Module(model_file_path, state)
        optimizer = Optimizer(optimizer_file_path, model)

        # Keep a copy of the parameters.
        old_output_params = model.get_contiguous_parameters()

        # Run a Training Step.
        model.train()
        model(forward_inputs)
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
    # Initialize Models
    simple_model, onnx_model, _, eval_model, _ = _create_training_models()

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save models & checkpoint files to load them later.
        checkpoint_file_path, model_file_path, eval_model_file_path = _get_test_models_path(
            temp_dir, simple_model, onnx_model, eval_model=eval_model
        )

        # Create Checkpoint State.
        state = CheckpointState(checkpoint_file_path)

        # Create a Module.
        model = Module(model_file_path, state, eval_model_file_path)

        # Export inference model
        inference_model_file_path = os.path.join(temp_dir, "inference_model.onnx")
        model.export_model_for_inferencing(inference_model_file_path, ["output-0"])
        assert os.path.exists(inference_model_file_path)


def test_cuda_execution_provider():
    # Initialize Models
    simple_model, onnx_model, _, _, pt_model = _create_training_models()

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save models & checkpoint files to load them later.
        checkpoint_file_path, model_file_path = _get_test_models_path(temp_dir, simple_model, onnx_model)
        # Create Checkpoint State.
        state = CheckpointState(checkpoint_file_path)
        # Create a Module.
        model = Module(model_file_path, state, device="cuda")
        params = model.get_contiguous_parameters()

        # Check if parameters are moved to cuda.
        assert params.device_name() == "Cuda"
