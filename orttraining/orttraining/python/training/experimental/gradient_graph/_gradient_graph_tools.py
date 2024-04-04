import io
from pathlib import Path
from typing import Any, Callable, Optional, Union  # noqa: F401

import torch
from torch.onnx import TrainingMode

from onnxruntime.capi._pybind_state import GradientGraphBuilder

from ...ortmodule._custom_op_symbolic_registry import CustomOpSymbolicRegistry


def export_gradient_graph(
    model: torch.nn.Module,
    loss_fn: Callable[[Any, Any], Any],
    example_input: torch.Tensor,
    example_labels: torch.Tensor,
    gradient_graph_path: Union[Path, str],
    opset_version=12,
) -> None:
    r"""
    Build a gradient graph for `model` so that you can output gradients in an inference session when given specific input and corresponding labels.

    Args:
        model (torch.nn.Module): A gradient graph will be built for this model.

        loss_fn (Callable[[Any, Any], Any]): A function to compute the loss given the model's output and the `example_labels`.
        Predefined loss functions such as `torch.nn.CrossEntropyLoss()` will work but you might not be able to load the graph in other environments such as an InferenceSession in ONNX Runtime Web, instead, use a custom Python method.

        example_input (torch.Tensor): Example input that you would give your model for inference/prediction.

        example_labels (torch.Tensor): The expected labels for `example_input`.
        This could be the output of your model when given `example_input` but it might be different if your loss function expects labels to be different (e.g. when using cross entropy loss).

        gradient_graph_path (Union[Path, str]): The path to where you would like to save the gradient graph.

        opset_version (int): See `torch.onnx.export`.
    """

    # Make sure that loss nodes that expect multiple outputs are set up.
    CustomOpSymbolicRegistry.register_all(opset_version)

    if not isinstance(gradient_graph_path, str):
        gradient_graph_path = str(gradient_graph_path)

    class WrapperModule(torch.nn.Module):
        def forward(self, model_input, expected_labels, *model_params):
            for param, set_param in zip(model.parameters(), model_params):
                param.data = set_param.data
            output = model(model_input)
            loss = loss_fn(output, expected_labels)
            return output, loss

    wrapped_model = WrapperModule()

    dynamic_axes = {
        "input": {
            0: "batch_size",
        },
        "labels": {
            0: "batch_size",
        },
        "output": {
            0: "batch_size",
        },
    }

    args = (example_input, example_labels, *tuple(model.parameters()))
    model_param_names = tuple(name for name, _ in model.named_parameters())
    input_names = ["input", "labels", *model_param_names]
    nodes_needing_gradients = {name for name, param in model.named_parameters() if param.requires_grad}

    f = io.BytesIO()
    torch.onnx.export(
        wrapped_model,
        args,
        f,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=False,
        training=TrainingMode.TRAINING,
        input_names=input_names,
        output_names=["output", "loss"],
        dynamic_axes=dynamic_axes,
    )

    exported_model = f.getvalue()
    builder = GradientGraphBuilder(exported_model, {"loss"}, nodes_needing_gradients, "loss")
    builder.build()
    builder.save(gradient_graph_path)
