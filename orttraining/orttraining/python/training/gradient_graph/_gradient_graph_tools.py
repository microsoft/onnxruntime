from pathlib import Path
from typing import Any, Callable, Optional, Union

import torch
from onnxruntime.capi._pybind_state import GradientGraphBuilder
from torch.onnx import TrainingMode

from ..ortmodule._custom_op_symbolic_registry import CustomOpSymbolicRegistry


def export_gradient_graph(
        model: torch.nn.Module,
        loss_fn: Callable[[Any, Any], Any],
        example_input: torch.Tensor,
        example_labels: torch.Tensor,
        gradient_graph_path: Union[Path, str], intermediate_graph_path: Optional[Union[Path, str]] = None,
        # FIXME Use a clearer name that involes "parameters". `use_parameters_as_input`?
        input_weights=True):
    r"""
    Build a gradient graph for `model` so that you can output gradients when given a specific input and labels.

    Args:
        model (torch.nn.Module): A gradient will be built for this model.
        loss_fn (Callable[[Any, Any], Any]): A function to compute the loss given the model's output and the `example_labels`.
        example_input (torch.Tensor): Example input that you would give your model for inference/prediction.
        example_labels (torch.Tensor): The expected labels for `example_input`.
            This could be the output of your model when given `example_input` but it might be different if your loss function expects labels to be different (e.g. when using CrossEntropyLoss).
        gradient_graph_path (Union[Path, str]): The path to where you would like to save the gradient graph.
        intermediate_graph_path (Optional[Union[Path, str]): The path to where you would like to save any intermediate graphs that are needed to make the gradient graph. Defaults to `gradient_graph_path` and it will be overwritten if this function executes successfully.
        input_weights (bool): `True` if the gradient graph should have inputs for the model weights. Useful if TODO...
    """

    model.train()

    # Make sure that loss nodes that expect multiple outputs are set up.
    CustomOpSymbolicRegistry.register_all()

    if intermediate_graph_path is None:
        intermediate_graph_path = gradient_graph_path

    if not isinstance(intermediate_graph_path, str):
        intermediate_graph_path = str(intermediate_graph_path)

    if not isinstance(gradient_graph_path, str):
        gradient_graph_path = str(gradient_graph_path)

    class WrapperModule(torch.nn.Module):
        def __init__(self, model: torch.nn.Module):
            super(WrapperModule, self).__init__()
            self.model = model

        def forward(self, model_input, expected_labels):
            output = self.model(model_input)
            loss = loss_fn(output, expected_labels)
            return output, loss

    dynamic_axes = {
        'input': {0: 'batch_size', },
        'labels': {0: 'batch_size', },
        'output': {0: 'batch_size', },
    }

    if input_weights:
        # FIXME Probably need to expand the list of model params.
        class WeightsWrapperModule(WrapperModule):
            def forward(self, model_input, model_params, expected_labels):
                for param, set_param in zip(model.parameters(), model_params):
                    param.data = set_param.data
                output = self.model(model_input)
                loss = loss_fn(output, expected_labels)
                return output, loss
        wrapped_model = WeightsWrapperModule(model)
        args = (example_input, tuple(model.parameters()), example_labels)
        input_names = ['input', 'model_params', 'labels']
        dynamic_axes['model_params'] = {0: 'batch_size', }
    else:
        wrapped_model = WrapperModule(model)
        args = (example_input, example_labels)
        input_names = ['input', 'labels']

    torch.onnx.export(
        wrapped_model, args,
        intermediate_graph_path,
        export_params=True,
        opset_version=12, do_constant_folding=False,
        training=TrainingMode.TRAINING,
        input_names=input_names,
        output_names=['output', 'loss'],
        dynamic_axes=dynamic_axes)

    # TODO Allow customizing.
    nodes_needing_gradients = set()
    for name, _ in model.named_parameters():
        # Should we check `if _.requires_grad:` first?
        nodes_needing_gradients.add(name)

    # 'model.' will be prepended to nodes because we wrapped `model` using a member called `model`.
    nodes_needing_gradients = set(
        'model.' + name for name in nodes_needing_gradients)

    builder = GradientGraphBuilder(intermediate_graph_path,
                                   {'loss'},
                                   nodes_needing_gradients,
                                   'loss')
    builder.build()
    builder.save(gradient_graph_path)
