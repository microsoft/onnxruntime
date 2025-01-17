import io
import logging
import os

import onnx
import torch

from onnxruntime.training import artifacts


class MNIST(torch.nn.Module):
    """MNIST PyTorch model"""

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


def get_model(device, batch_size, input_size, hidden_size, output_size):
    return MNIST(input_size, hidden_size, output_size).to(device)


def export(torch_model, model_inputs):
    """Exports the given torch model to ONNX."""

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


def get_onnx_model(pt_model, device, batch_size, input_size, hidden_size, output_size):
    """Returns the ONNX model for the given pytorch model."""
    return export(pt_model, (torch.randn(batch_size, input_size, device=device),))


def get_models():
    """Returns a tuple of (PyTorch model, ONNX model) for MNIST."""

    logging.info("Creating PyTorch MNIST model and exporting to its equivalent ONNX model.")
    device = "cuda"
    batch_size, input_size, hidden_size, output_size = 64, 784, 500, 10
    pt_model = get_model(device, batch_size, input_size, hidden_size, output_size)
    onnx_model = get_onnx_model(pt_model, device, batch_size, input_size, hidden_size, output_size)
    return pt_model, onnx_model


def create_training_artifacts(model_path, artifacts_dir, model_prefix):
    """Using onnxblock, this function creates the training artifacts for the model at the path provided.

    The artifacts created can be used to train the model using onnxruntime.training.api. The artifacts are:
    1. The training graph
    2. The eval graph
    3. The optimizer graph
    4. The checkpoint file
    """

    onnx_model = onnx.load(model_path)

    requires_grad = [
        param.name
        for param in onnx_model.graph.initializer
        if (not param.name.endswith("_scale") and not param.name.endswith("_zero_point"))
    ]
    artifacts.generate_artifacts(
        onnx_model,
        requires_grad=requires_grad,
        loss=artifacts.LossType.CrossEntropyLoss,
        optimizer=artifacts.OptimType.AdamW,
        artifact_directory=artifacts_dir,
        prefix=model_prefix,
    )

    # Create the training artifacts
    train_model_path = os.path.join(artifacts_dir, f"{model_prefix}training_model.onnx")
    eval_model_path = os.path.join(artifacts_dir, f"{model_prefix}eval_model.onnx")
    optimizer_model_path = os.path.join(artifacts_dir, f"{model_prefix}optimizer_model.onnx")
    checkpoint_path = os.path.join(artifacts_dir, f"{model_prefix}checkpoint")

    return train_model_path, eval_model_path, optimizer_model_path, checkpoint_path
