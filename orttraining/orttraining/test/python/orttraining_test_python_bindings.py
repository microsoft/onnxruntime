import io
import os
import tempfile

import onnx
import pytest
import torch

import onnxruntime.training.onnxblock as onnxblock
from onnxruntime.capi import _pybind_state as C
from onnxruntime.capi.onnxruntime_inference_collection import OrtValue
from onnxruntime.capi.onnxruntime_pybind11_state import OrtValueVector
from onnxruntime.training.onnxblock import TrainingModule, TrainingOptimizer


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

def create_training_model(model,input_size, batch_size, device = "cuda",zero_flag = False):
    model.train()
    # setting all initial weights to zero
    if zero_flag:
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()
    x = torch.randn(batch_size, input_size, device = device)

    onnx_model = _get_onnx_model(model, (x,))
    with onnxblock.onnx_model(onnx_model):
        _ = model(onnx_model.graph.output[0].name)

    trainable_params, non_trainable_params = model.parameters()

    with tempfile.TemporaryDirectory() as checkpoint_dir_name:
        checkpoint_file_path = os.path.join(checkpoint_dir_name, "checkpoint")
        onnxblock.save_checkpoint((trainable_params, non_trainable_params), checkpoint_file_path)

        model_file_path = os.path.join(checkpoint_dir_name, "training_model.onnx")

        onnx.save(onnx_model, model_file_path)


def create_ort_vectors_data():
    # Generate random data to test with.
    inputs = torch.randn(64, 784).numpy()
    labels = torch.randint(high=10, size=(64,), dtype=torch.int32).numpy()
    forward_inputs = OrtValueVector()
    forward_inputs.reserve(2)
    forward_inputs.push_back(OrtValue.ortvalue_from_numpy(inputs)._ortvalue)
    forward_inputs.push_back(OrtValue.ortvalue_from_numpy(labels)._ortvalue)

    return forward_inputs, OrtValueVector()


def test_train_step_training_module():

    # Given
    device = "cuda"
    batch_size, input_size, hidden_size, output_size = 64, 784, 500, 10
    _, onnx_model = _get_models(device, batch_size, input_size, hidden_size, output_size)

    # Build the onnx model with loss
    simple_model = SimpleModelWithCrossEntropyLoss()
    with onnxblock.onnx_model(onnx_model) as accessor:
        _ = simple_model(onnx_model.graph.output[0].name)
        eval_model = accessor.eval_model


    trainable_params, non_trainable_params = simple_model.parameters()
    optimizer = onnxblock.optim.AdamW()
    with onnxblock.onnx_model() as accessor:
        _ = optimizer(simple_model.parameters())
        optimizer_model = accessor.model

    with tempfile.TemporaryDirectory() as checkpoint_dir_name:
        checkpoint_file_path = os.path.join(checkpoint_dir_name, "checkpoint")
        onnxblock.save_checkpoint((trainable_params, non_trainable_params), checkpoint_file_path)

        model_file_path = os.path.join(checkpoint_dir_name, "training_model.onnx")
        onnx.save(onnx_model, model_file_path)

        eval_model_file_path = os.path.join(checkpoint_dir_name, "eval_model.onnx")
        onnx.save(eval_model, eval_model_file_path)


        optimizer_file_path = os.path.join(checkpoint_dir_name, "optimizer.onnx")
        onnx.save(optimizer_model, optimizer_file_path)


        inputs = torch.randn(64, 784).numpy()
        labels = torch.randint(high=10, size=(64,), dtype=torch.int32).numpy()
        forward_inputs = OrtValueVector()
        forward_inputs.reserve(2)
        forward_inputs.push_back(OrtValue.ortvalue_from_numpy(inputs)._ortvalue)
        forward_inputs.push_back(OrtValue.ortvalue_from_numpy(labels)._ortvalue)



        test_inputs = torch.randn(64, 784).numpy()
        test_labels = torch.randint(high=10, size=(64,), dtype=torch.int32).numpy()
        test_forward_inputs = OrtValueVector()
        test_forward_inputs.reserve(2)
        test_forward_inputs.push_back(OrtValue.ortvalue_from_numpy(test_inputs)._ortvalue)
        test_forward_inputs.push_back(OrtValue.ortvalue_from_numpy(test_labels)._ortvalue)


        # Create a Training Module and Training Optimizer.
        model = TrainingModule(model_file_path, checkpoint_file_path, eval_model_file_path)
        optimizer = TrainingOptimizer(optimizer_file_path, model.get_model())

        # Training Loop
        for epoch in range(10):
            model.train()
            loss = model(forward_inputs)
            optimizer.step()

            model.eval()
            test_loss = model(test_forward_inputs)
            print('Epoch: {} , Training Loss: {}, Test Loss: {}'.format(epoch, loss, test_loss))
