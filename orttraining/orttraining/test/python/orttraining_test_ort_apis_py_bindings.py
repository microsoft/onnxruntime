# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations

import os
import pathlib
import tempfile
from dataclasses import dataclass

import numpy as np
import onnx
import pytest
import torch
from orttraining_test_ort_apis_onnxblock import _get_models

import onnxruntime.training.onnxblock as onnxblock
from onnxruntime import OrtValue, SessionOptions
from onnxruntime.training import artifacts
from onnxruntime.training.api import CheckpointState, LinearLRScheduler, Module, Optimizer


class SimpleModelWithCrossEntropyLoss(onnxblock.TrainingBlock):
    def __init__(self):
        super().__init__()
        self.loss = onnxblock.loss.CrossEntropyLoss()

    def build(self, output_name):
        return self.loss(output_name)


@dataclass
class Artifacts:
    checkpoint_file_path: str
    training_model_file_path: str
    eval_model_file_path: str
    optimizer_model_file_path: str
    pt_model: torch.nn.Module
    nominal_checkpoint_file_path: str | None = None


def _create_training_artifacts(
    artifact_directory: str | os.PathLike,
    requires_grad: list[str] | None = None,
    frozen_params: list[str] | None = None,
    optimizer_type=artifacts.OptimType.AdamW,
    nominal_checkpoint: bool = False,
):
    device = "cpu"
    batch_size, input_size, hidden_size, output_size = 64, 784, 500, 10
    pt_model, onnx_model = _get_models(device, batch_size, input_size, hidden_size, output_size)

    if requires_grad is None:
        requires_grad = [name for name, param in pt_model.named_parameters() if param.requires_grad]

    if frozen_params is None:
        frozen_params = [name for name, param in pt_model.named_parameters() if not param.requires_grad]

    artifacts.generate_artifacts(
        onnx_model,
        optimizer=optimizer_type,
        loss=artifacts.LossType.CrossEntropyLoss,
        requires_grad=requires_grad,
        frozen_params=frozen_params,
        artifact_directory=artifact_directory,
        nominal_checkpoint=nominal_checkpoint,
    )

    training_model_file = os.path.join(artifact_directory, "training_model.onnx")
    eval_model_file = os.path.join(artifact_directory, "eval_model.onnx")
    optimizer_model_file = os.path.join(artifact_directory, "optimizer_model.onnx")
    checkpoint_file = os.path.join(artifact_directory, "checkpoint")
    nominal_checkpoint_file = None
    if nominal_checkpoint:
        nominal_checkpoint_file = os.path.join(artifact_directory, "nominal_checkpoint")

    return Artifacts(
        checkpoint_file, training_model_file, eval_model_file, optimizer_model_file, pt_model, nominal_checkpoint_file
    )


def test_train_step():
    # Generating random data for testing.
    inputs = torch.randn(64, 784).numpy()
    labels = torch.randint(high=10, size=(64,), dtype=torch.int64).numpy()

    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts = _create_training_artifacts(temp_dir)
        # Create Checkpoint State.
        state = CheckpointState.load_checkpoint(artifacts.checkpoint_file_path)
        # Create a Module.
        model = Module(artifacts.training_model_file_path, state)
        model.train()
        ort_loss = model(inputs, labels)

        # Calculate loss using pytorch model to compare it with Module's output.
        pt_outputs = artifacts.pt_model(torch.from_numpy(inputs))
        loss_fn = torch.nn.CrossEntropyLoss()
        pt_loss = loss_fn(pt_outputs, torch.from_numpy(labels).long())

        assert np.allclose(ort_loss, pt_loss.detach().numpy())


def test_eval_step():
    # Generating random data for testing.
    inputs = torch.randn(64, 784).numpy()
    labels = torch.randint(high=10, size=(64,), dtype=torch.int64).numpy()

    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts = _create_training_artifacts(temp_dir)
        # Create Checkpoint State.
        state = CheckpointState.load_checkpoint(artifacts.checkpoint_file_path)
        # Create a Module.
        model = Module(artifacts.training_model_file_path, state, artifacts.eval_model_file_path)
        model.train()
        model(inputs, labels)

        model.eval()
        fetches = model(inputs, labels)
        assert fetches


@pytest.mark.parametrize("optimizer_type", [artifacts.OptimType.SGD, artifacts.OptimType.AdamW])
def test_optimizer_step(optimizer_type):
    # Generating random data for testing.
    inputs = torch.randn(64, 784).numpy()
    labels = torch.randint(high=10, size=(64,), dtype=torch.int64).numpy()

    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts = _create_training_artifacts(temp_dir, optimizer_type=optimizer_type)
        # Create Checkpoint State.
        state = CheckpointState.load_checkpoint(artifacts.checkpoint_file_path)
        # Create a Module and Optimizer.
        model = Module(artifacts.training_model_file_path, state)
        optimizer = Optimizer(artifacts.optimizer_model_file_path, model)

        model.train()
        old_flatten_params = model.get_contiguous_parameters()
        model(inputs, labels)

        optimizer.step()
        new_params = model.get_contiguous_parameters()
        # Assert that the parameters are updated.
        assert not np.array_equal(old_flatten_params.numpy(), new_params.numpy())


@pytest.mark.parametrize("optimizer_type", [artifacts.OptimType.SGD, artifacts.OptimType.AdamW])
def test_get_and_set_lr(optimizer_type):
    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts = _create_training_artifacts(temp_dir, optimizer_type=optimizer_type)
        # Create Checkpoint State.
        state = CheckpointState.load_checkpoint(artifacts.checkpoint_file_path)
        # Create a Module and Optimizer.
        model = Module(artifacts.training_model_file_path, state)
        optimizer = Optimizer(artifacts.optimizer_model_file_path, model)

        # Test get and set learning rate.
        lr = optimizer.get_learning_rate()
        assert round(lr, 3) == 0.001

        optimizer.set_learning_rate(0.5)
        new_lr = optimizer.get_learning_rate()

        assert np.isclose(new_lr, 0.5)
        assert lr != new_lr


@pytest.mark.parametrize("optimizer_type", [artifacts.OptimType.SGD, artifacts.OptimType.AdamW])
def test_scheduler_step(optimizer_type):
    # Generating random data for testing.
    inputs = torch.randn(64, 784).numpy()
    labels = torch.randint(high=10, size=(64,), dtype=torch.int64).numpy()

    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts = _create_training_artifacts(temp_dir, optimizer_type=optimizer_type)
        state = CheckpointState.load_checkpoint(artifacts.checkpoint_file_path)
        # Create a Module and Optimizer.
        model = Module(artifacts.training_model_file_path, state)
        optimizer = Optimizer(artifacts.optimizer_model_file_path, model)
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
        artifacts = _create_training_artifacts(temp_dir)
        # Create Checkpoint State.
        state = CheckpointState.load_checkpoint(artifacts.checkpoint_file_path)
        # Create a Training Module and Training Optimizer.
        model = Module(artifacts.training_model_file_path, state)

        model.train()
        model(inputs, labels)

        checkpoint_save_path = os.path.join(temp_dir, "checkpoint_export.ckpt")

        CheckpointState.save_checkpoint(state, checkpoint_save_path)
        old_flatten_params = model.get_contiguous_parameters()

        # Assert the checkpoint was saved.
        assert os.path.exists(checkpoint_save_path)

        # Assert the checkpoint parameters remain after saving.
        new_state = CheckpointState.load_checkpoint(checkpoint_save_path)
        new_model = Module(artifacts.training_model_file_path, new_state)

        new_params = new_model.get_contiguous_parameters()

        assert np.array_equal(old_flatten_params.numpy(), new_params.numpy())


@pytest.mark.parametrize("optimizer_type", [artifacts.OptimType.SGD, artifacts.OptimType.AdamW])
@pytest.mark.parametrize("trainable_only", [True, False])
def test_copy_buffer_to_parameters(trainable_only, optimizer_type):
    # Generating random data for testing.
    inputs = torch.randn(64, 784).numpy()
    labels = torch.randint(high=10, size=(64,), dtype=torch.int64).numpy()

    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts = _create_training_artifacts(
            temp_dir,
            requires_grad=["fc2.weight", "fc2.bias"],
            frozen_params=["fc1.weight", "fc1.bias"],
            optimizer_type=optimizer_type,
        )
        state = CheckpointState.load_checkpoint(artifacts.checkpoint_file_path)

        # Create a Module and Optimizer.
        model = Module(artifacts.training_model_file_path, state)
        optimizer = Optimizer(artifacts.optimizer_model_file_path, model)

        # Keep a copy of the parameters.
        old_output_params = model.get_contiguous_parameters(trainable_only=trainable_only)

        # Run a Training Step.
        model.train()
        model(inputs, labels)
        optimizer.step()

        # Get the new parameters.
        output_params = model.get_contiguous_parameters(trainable_only=trainable_only)
        # Make sure old params are different from new params.
        assert not np.array_equal(old_output_params.numpy(), output_params.numpy())

        # Copy the old parameters to the model.
        model.copy_buffer_to_parameters(old_output_params, trainable_only=trainable_only)

        # Get the saved parameters.
        saved_params = model.get_contiguous_parameters(trainable_only=trainable_only)

        # Make sure the saved parameters are the same as the old parameters.
        assert np.array_equal(old_output_params.numpy(), saved_params.numpy())


def test_export_model_for_inferencing():
    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts = _create_training_artifacts(temp_dir)

        # Create Checkpoint State.
        state = CheckpointState.load_checkpoint(artifacts.checkpoint_file_path)

        # Create a Module.
        model = Module(artifacts.training_model_file_path, state, artifacts.eval_model_file_path)

        # Export inference model
        inference_model_file_path = os.path.join(temp_dir, "inference_model.onnx")
        model.export_model_for_inferencing(inference_model_file_path, ["output-0"])
        assert os.path.exists(inference_model_file_path)


def test_cuda_execution_provider():
    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts = _create_training_artifacts(temp_dir)

        # Create Checkpoint State.
        state = CheckpointState.load_checkpoint(artifacts.checkpoint_file_path)
        # Create a Module.
        model = Module(artifacts.training_model_file_path, state, device="cuda")
        params = model.get_contiguous_parameters()

        # Check if parameters are moved to cuda.
        assert params.device_name() == "Cuda"


@pytest.mark.parametrize(
    "property_value",
    [-1, 0, 1, 1234567890, -1.0, -0.1, 0.1, 1.0, 1234.0, "hello", "world", "onnxruntime"],
)
def test_add_get_property(property_value):
    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts = _create_training_artifacts(temp_dir)

        # Create Checkpoint State.
        state = CheckpointState.load_checkpoint(artifacts.checkpoint_file_path)

        # Create a Module.
        _ = Module(artifacts.training_model_file_path, state)

        # Float values in python are double precision.
        # Convert to float32 to match the type of the property.
        if isinstance(property_value, float):
            property_value = float(np.float32(property_value))

        assert len(state.properties) == 0

        state.properties["property"] = property_value
        assert "property" in state.properties
        assert state.properties["property"] == property_value
        assert len(state.properties) == 1

        CheckpointState.save_checkpoint(state, artifacts.checkpoint_file_path)
        new_state = CheckpointState.load_checkpoint(artifacts.checkpoint_file_path)
        assert "property" in new_state.properties
        assert new_state.properties["property"] == property_value
        assert len(new_state.properties) == 1


def test_get_input_output_names():
    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts = _create_training_artifacts(temp_dir)

        # Create Checkpoint State.
        state = CheckpointState.load_checkpoint(artifacts.checkpoint_file_path)

        # Create a Module.
        model = Module(artifacts.training_model_file_path, state, artifacts.eval_model_file_path)

        training_model = onnx.load(artifacts.training_model_file_path)
        assert model.input_names() == [input.name for input in training_model.graph.input][:2]
        assert model.output_names() == [output.name for output in training_model.graph.output][:1]


def test_ort_custom_ops():
    def _create_custom_op_trainable_onnx_model():
        """This function takes in a pre generated custom op model and adds a trainable linear layer to it"""
        onnx_model = onnx.load(os.path.join("testdata", "custom_op_library", "custom_op_test.onnx"))
        onnx_model.graph.value_info.append(
            onnx.helper.make_tensor_value_info("output_1", onnx.TensorProto.FLOAT, [3, 5])
        )

        class CustomOpBlockWithLinear(onnxblock.ForwardBlock):
            def __init__(self):
                super().__init__()
                self.linear = onnxblock.blocks.Linear(5, 10)

            def build(self, linear_input):
                return self.linear(linear_input)

        custom_op_block = CustomOpBlockWithLinear()
        with onnxblock.base(onnx_model) as model_accessor:
            model_accessor.model.opset_import.append(onnx.helper.make_opsetid("test.customop", 1))
            model_accessor.model.opset_import.append(onnx.helper.make_opsetid("", 14))
            model_accessor.model.ir_version = 7
            _ = custom_op_block("output_1")

        return custom_op_block.to_model_proto()

    onnx_model = _create_custom_op_trainable_onnx_model()
    custom_op_library = os.path.join(os.getcwd(), "libcustom_op_library.so")
    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts.generate_artifacts(
            onnx_model,
            optimizer=artifacts.OptimType.AdamW,
            loss=artifacts.LossType.CrossEntropyLoss,
            requires_grad=[param.name for param in onnx_model.graph.initializer],
            artifact_directory=temp_dir,
            custom_op_library=custom_op_library,
        )

        session_options = SessionOptions()
        session_options.register_custom_ops_library(custom_op_library)

        training_model_file_path = pathlib.Path(temp_dir) / "training_model.onnx"
        eval_model_file_path = pathlib.Path(temp_dir) / "eval_model.onnx"
        checkpoint_file_path = pathlib.Path(temp_dir) / "checkpoint"

        # Create Checkpoint State.
        state = CheckpointState.load_checkpoint(checkpoint_file_path)

        # Create a Module.
        # The custom op library is built either for cuda or cpu (but not for both).
        # Since the training api pipeline build uses cuda, we need to specify the device as cuda.
        # Otherwise the custom op library will not have the required kernels.
        model = Module(
            training_model_file_path, state, eval_model_file_path, device="cuda", session_options=session_options
        )

        x = np.random.randn(3, 5).astype(np.float32)
        y = np.random.randn(3, 5).astype(np.float32)
        labels = np.random.randint(0, 10, size=(3,), dtype=np.int64)
        _ = model(x, y, labels)


def test_string_inputs():
    def _create_string_input_trainable_model():
        """This function creates an onnx model with string inputs"""

        class BlockWithStringInputs(onnxblock.ForwardBlock):
            def __init__(self):
                super().__init__()
                self.cast = onnxblock.blocks.Cast(to=onnx.TensorProto.FLOAT)
                self.linear = onnxblock.blocks.Linear(4, 2)

            def build(self, string_input):
                return self.linear(self.cast(string_input))

        string_block = BlockWithStringInputs()
        with onnxblock.empty_base() as model_accessor:
            model_accessor.model.graph.input.extend(
                [
                    onnx.helper.make_tensor_value_info("input", onnx.TensorProto.STRING, [1, 4]),
                ]
            )
            _ = string_block("input")

        return string_block.to_model_proto()

    onnx_model = _create_string_input_trainable_model()
    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts.generate_artifacts(
            onnx_model,
            optimizer=artifacts.OptimType.AdamW,
            loss=artifacts.LossType.CrossEntropyLoss,
            requires_grad=[param.name for param in onnx_model.graph.initializer],
            artifact_directory=temp_dir,
        )

        training_model_file_path = pathlib.Path(temp_dir) / "training_model.onnx"
        eval_model_file_path = pathlib.Path(temp_dir) / "eval_model.onnx"
        checkpoint_file_path = pathlib.Path(temp_dir) / "checkpoint"

        # Create Checkpoint State.
        state = CheckpointState.load_checkpoint(checkpoint_file_path)

        # Create a Module.
        model = Module(training_model_file_path, state, eval_model_file_path)

        strs = np.array([["1.0", "2.0", "3.0", "4.0"]], dtype=str)
        labels = np.random.randint(0, 2, size=(1,), dtype=np.int64)

        model.train()
        _ = model(strs, labels)

        model.eval()
        _ = model(strs, labels)


def test_train_step_with_ort_values():
    # Generating random data for testing.
    inputs_np = torch.randn(64, 784).numpy()
    inputs = OrtValue.ortvalue_from_numpy(inputs_np)
    labels_np = torch.randint(high=10, size=(64,), dtype=torch.int64).numpy()
    labels = OrtValue.ortvalue_from_numpy(labels_np)

    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts = _create_training_artifacts(temp_dir)

        # Create Checkpoint State.
        state = CheckpointState.load_checkpoint(artifacts.checkpoint_file_path)
        # Create a Module.
        model = Module(artifacts.training_model_file_path, state)
        model.train()
        ort_loss = model(inputs, labels)
        assert isinstance(ort_loss, OrtValue)

        # Calculate loss using pytorch model to compare it with Module's output.
        pt_outputs = artifacts.pt_model(torch.from_numpy(inputs_np))
        loss_fn = torch.nn.CrossEntropyLoss()
        pt_loss = loss_fn(pt_outputs, torch.from_numpy(labels_np).long())

        assert np.allclose(ort_loss.numpy(), pt_loss.detach().numpy())


def test_eval_step_with_ort_values():
    # Generating random data for testing.
    inputs_np = torch.randn(64, 784).numpy()
    inputs = OrtValue.ortvalue_from_numpy(inputs_np)
    labels_np = torch.randint(high=10, size=(64,), dtype=torch.int64).numpy()
    labels = OrtValue.ortvalue_from_numpy(labels_np)

    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts = _create_training_artifacts(temp_dir)
        # Create Checkpoint State.
        state = CheckpointState.load_checkpoint(artifacts.checkpoint_file_path)
        # Create a Module.
        model = Module(artifacts.training_model_file_path, state, artifacts.eval_model_file_path)
        model.train()
        model(inputs, labels)

        model.eval()
        fetches = model(inputs, labels)
        assert isinstance(fetches, OrtValue)
        assert fetches


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_get_and_set_parameter_values(device):
    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts = _create_training_artifacts(
            temp_dir, requires_grad=["fc2.weight", "fc2.bias"], frozen_params=["fc1.weight", "fc1.bias"]
        )

        state = CheckpointState.load_checkpoint(artifacts.checkpoint_file_path)

        model = Module(artifacts.training_model_file_path, state, artifacts.eval_model_file_path, device=device)

        state_dict = artifacts.pt_model.state_dict()
        assert len(state_dict) == len(state.parameters)
        for parameter_name, _ in state.parameters:
            assert parameter_name in state_dict

        for name, pt_param in artifacts.pt_model.named_parameters():
            ort_param = state.parameters[name]
            assert ort_param.name == name
            assert np.allclose(pt_param.detach().cpu().numpy(), ort_param.data)
            if name in ["fc1.weight", "fc1.bias"]:
                assert ort_param.requires_grad is False
                assert ort_param.grad is None
            else:
                assert ort_param.requires_grad is True
                assert np.allclose(ort_param.grad, np.zeros_like(ort_param.data, dtype=np.float32))

        original_param = state.parameters["fc1.weight"].data
        state.parameters["fc1.weight"].data = np.ones_like(state.parameters["fc1.weight"].data, dtype=np.float32)
        updated_param = state.parameters["fc1.weight"].data
        assert np.allclose(updated_param, np.ones_like(updated_param, dtype=np.float32))

        model.train()
        inputs = torch.randn(64, 784).numpy()
        labels = torch.randint(high=10, size=(64,), dtype=torch.int64).numpy()
        loss = model(inputs, labels)
        assert loss is not None
        for name, _ in artifacts.pt_model.named_parameters():
            ort_param = state.parameters[name]
            assert ort_param.name == name
            if name in ["fc1.weight", "fc1.bias"]:
                assert ort_param.requires_grad is False
                assert ort_param.grad is None
            else:
                assert ort_param.requires_grad is True
                assert ort_param.grad.any()

        state.parameters["fc1.weight"] = original_param
        assert np.allclose(state.parameters["fc1.weight"].data, original_param)


def test_model_construction_with_nominal_checkpoint():
    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts = _create_training_artifacts(temp_dir, nominal_checkpoint=True)

        nominal_state = CheckpointState.load_checkpoint(artifacts.nominal_checkpoint_file_path)
        model_with_nominal_state = Module(
            artifacts.training_model_file_path, nominal_state, artifacts.eval_model_file_path
        )
        optimizer_with_nominal_state = Optimizer(artifacts.optimizer_model_file_path, model_with_nominal_state)

        inputs = torch.randn(64, 784).numpy()
        labels = torch.randint(high=10, size=(64,), dtype=torch.int64).numpy()

        err_msg = "Please load the parameter states first"

        # Accessing the checkpoint parameter raises
        state_dict = artifacts.pt_model.state_dict()
        for param_name in state_dict:
            assert param_name in nominal_state.parameters
        with pytest.raises(Exception) as exc_info:
            _ = nominal_state.parameters["fc1.weight"]

        assert err_msg in str(exc_info.value)

        err_msg = "Please load all the parameter states first"
        with pytest.raises(Exception) as exc_info:
            nominal_state.parameters["fc1.weight"] = np.ones((10, 10), dtype=np.float32)

        assert err_msg in str(exc_info.value)

        err_msg = "Please load the model parameters first."

        # Getting contiguous parameters raises
        with pytest.raises(Exception) as exc_info:
            _ = model_with_nominal_state.get_contiguous_parameters()

        assert err_msg in str(exc_info.value)

        # Train step raises
        with pytest.raises(Exception) as exc_info:
            model_with_nominal_state.train()
            model_with_nominal_state(inputs, labels)

        assert err_msg in str(exc_info.value)

        # Optimizer step raises
        with pytest.raises(Exception) as exc_info:
            optimizer_with_nominal_state.step()

        assert err_msg in str(exc_info.value)

        # Eval step raises
        with pytest.raises(Exception) as exc_info:
            model_with_nominal_state.eval()
            model_with_nominal_state(inputs, labels)

        assert err_msg in str(exc_info.value)

        # Get parameters size does not raise
        params_size = model_with_nominal_state.get_parameters_size()
        assert params_size > 0


def test_train_with_nominal_checkpoint():
    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts = _create_training_artifacts(temp_dir, nominal_checkpoint=True)

        # Create Checkpoint State with nominal checkpoint as well as the complete checkpoint.
        complete_state = CheckpointState.load_checkpoint(artifacts.checkpoint_file_path)
        nominal_state = CheckpointState.load_checkpoint(artifacts.nominal_checkpoint_file_path)

        # Create a Module with both complete and nominal checkpoint states.
        model_with_complete_state = Module(artifacts.training_model_file_path, complete_state)
        model_with_nominal_state = Module(artifacts.training_model_file_path, nominal_state)

        optimizer_with_complete_state = Optimizer(artifacts.optimizer_model_file_path, model_with_complete_state)
        optimizer_with_nominal_state = Optimizer(artifacts.optimizer_model_file_path, model_with_nominal_state)

        parameter_buffer = model_with_complete_state.get_contiguous_parameters()
        model_with_nominal_state.copy_buffer_to_parameters(parameter_buffer, trainable_only=False)

        model_with_complete_state.train()
        model_with_nominal_state.train()

        # Generate random data for testing.
        inputs = torch.randn(64, 784).numpy()
        labels = torch.randint(high=10, size=(64,), dtype=torch.int64).numpy()

        ort_loss_1 = model_with_complete_state(inputs, labels)
        ort_loss_2 = model_with_nominal_state(inputs, labels)

        # Calculate loss using pytorch model to compare it with both the Modules' output.
        pt_outputs = artifacts.pt_model(torch.from_numpy(inputs))
        loss_fn = torch.nn.CrossEntropyLoss()
        pt_loss = loss_fn(pt_outputs, torch.from_numpy(labels).long())

        assert np.allclose(ort_loss_1, ort_loss_2)
        assert np.allclose(ort_loss_1, pt_loss.detach().numpy())

        optimizer_with_complete_state.step()
        optimizer_with_nominal_state.step()

        new_params_1 = model_with_complete_state.get_contiguous_parameters()
        new_params_2 = model_with_nominal_state.get_contiguous_parameters()

        assert np.allclose(new_params_1.numpy(), new_params_2.numpy())
