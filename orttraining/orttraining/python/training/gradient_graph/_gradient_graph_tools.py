from pathlib import Path
from typing import Any, Callable, Union

import torch
from onnxruntime.capi._pybind_state import GradientGraphBuilder
from torch.onnx import TrainingMode

from ..ortmodule._custom_op_symbolic_registry import CustomOpSymbolicRegistry


def export_gradient_graph(
        model: torch.nn.Module,
        loss_fn: Callable[[Any, Any], Any],
        example_input: torch.Tensor,
        intermediate_graph_path: Union[Path, str],
        gradient_graph_path: Union[Path, str]):
    r"""
    Build a gradient graph for `model`.

    Args:
        model (torch.nn.Module): A gradient will be built for this model.
        TODO Documents arguments.
    """

    model.train()

    # Make sure that loss nodes that expect multiple outputs are set up.
    CustomOpSymbolicRegistry.register_all()

    example_labels = model(example_input)

    if not isinstance(intermediate_graph_path, str):
        intermediate_graph_path = str(intermediate_graph_path)

    if not isinstance(gradient_graph_path, str):
        gradient_graph_path = str(gradient_graph_path)

    class WrapperModule(torch.nn.Module):
        def forward(self, model_input, expected_labels):
            output = model(model_input)
            loss = loss_fn(output, expected_labels)
            return output, loss

    wrapped_model = WrapperModule()

    torch.onnx.export(
        wrapped_model, (example_input, example_labels), intermediate_graph_path,
        export_params=True,
        opset_version=12, do_constant_folding=False,
        training=TrainingMode.TRAINING,
        # TODO Allow customizing.
        input_names=['input', 'labels'],
        output_names=['output', 'loss'],
        dynamic_axes={
            'input': {0: 'batch_size', },
            'labels': {0: 'batch_size', },
            'output': {0: 'batch_size', },
        })

    # TODO Allow customizing.
    nodes_needing_gradients = set()
    for name, _ in model.named_parameters():
        # Should we check `if _.requires_grad:` first?
        nodes_needing_gradients.add(name)

    builder = GradientGraphBuilder(intermediate_graph_path,
                                   {'loss'},
                                   nodes_needing_gradients,
                                   'loss')
    builder.build()
    builder.save(gradient_graph_path)
