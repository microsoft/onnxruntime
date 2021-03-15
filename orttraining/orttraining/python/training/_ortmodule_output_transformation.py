from collections import abc
import copy
import functools
import inspect
import torch
import warnings

def deepcopy_model_input(*inputs, **kwargs):
    sample_inputs_copy = []
    for model_input in inputs:
        sample_inputs_copy.append(model_input.data if isinstance(model_input, torch.Tensor) else model_input)
    sample_inputs_copy = copy.deepcopy(tuple(sample_inputs_copy))

    sample_kwargs_copy = {}
    for name, model_input in kwargs.items():
        sample_kwargs_copy[name] = model_input.data if isinstance(model_input, torch.Tensor) else model_input
    sample_kwargs_copy = copy.deepcopy(sample_kwargs_copy)

    return sample_inputs_copy, sample_kwargs_copy

class _TensorStub:
    # Stub for a torch.Tensor value to be used to formulate the output schema
    pass

def populate_user_output_from_schema_and_outputs(output_schema, output_names, outputs):
    """Follows the schema to generate an output that is expected by the user"""

    def _replace_stub_with_tensor_value(user_output, outputs, output_idx):
        # Recursively traverse across user_output and replace all _TensorStub
        # with torch.Tensor values from outputs following output_idx

        if user_output is None:
            return None
        elif isinstance(user_output, _TensorStub):
            output_idx[0] += 1
            return outputs[output_idx[0]-1]

        if isinstance(user_output, abc.Sequence):
            sequence_type = type(user_output)
            user_output = list(user_output)
            for idx in range(len(user_output)):
                user_output[idx] = _replace_stub_with_tensor_value(user_output[idx], outputs, output_idx)
            try:
                # namedtuple can be created by passing the list sequence to method _make
                user_output = sequence_type._make(user_output)
            except AttributeError:
                # If attribute error encountered, create the sequence directly
                user_output = sequence_type(user_output)
        elif isinstance(user_output, abc.Mapping):
            for key in sorted(user_output):
                user_output[key] = _replace_stub_with_tensor_value(user_output[key], outputs, output_idx)
        else:
            raise TypeError(f'ORTModule does not support the following model output type {type(user_output)}.')

        return user_output

    # Order the outputs according to the names so that the traversal order is consistent
    outputs = [x for _, x in sorted(zip(output_names, outputs))]

    # Replace every _TensorStub value in the schema with the torch.Tensor outputs calculated
    output_schema_copy = copy.deepcopy(output_schema)
    output_idx = [0]
    user_output = _replace_stub_with_tensor_value(output_schema_copy, outputs, output_idx)

    return user_output

def _extract_output_schema(output):
    """Extract the output schema by replacing every torch.Tensor value with _TensorStub"""

    if output is None:
        return None
    # Depth first traversal to iterate over the output to replace every tensor with a stub
    elif isinstance(output, torch.Tensor):
        return _TensorStub()

    if isinstance(output, abc.Sequence):
        sequence_type = type(output)
        output = list(output)
        for idx in range(len(output)):
            output[idx] = _extract_output_schema(output[idx])
        try:
            # namedtuple can be created by passing the list sequence to method _make
            output = sequence_type._make(output)
        except AttributeError:
            # If attribute error encountered, create the sequence directly
            output = sequence_type(output)
    elif isinstance(output, abc.Mapping):
        for key in sorted(output):
            output[key] = _extract_output_schema(output[key])
    else:
        raise TypeError(f'ORTModule does not support the following model output type {type(output)}')

    return output

def _parse_outputs_and_extract_names_and_dynamic_axes(module_output):
    """Parses through the module output and returns output names and dynamic axes"""

    def _populate_output_names_and_dynamic_axes(output, output_names, output_dynamic_axes, output_idx):
        # Depth first traversal to traverse through the entire output collecting output names and dynamic axes

        if output is None:
            return
        elif isinstance(output, torch.Tensor):
            output_name = f'output{output_idx[0]}'
            output_idx[0] += 1
            output_names.append(output_name)
            output_dynamic_axes[output_name] = {}
            for dim_idx in range(len(output.shape)):
                output_dynamic_axes[output_name].update({dim_idx: f'{output_name}_dim{dim_idx}'})
            return

        if isinstance(output, abc.Sequence):
            for value in output:
                _populate_output_names_and_dynamic_axes(value, output_names, output_dynamic_axes, output_idx)
        elif isinstance(output, abc.Mapping):
            for _, value in sorted(output.items()):
                _populate_output_names_and_dynamic_axes(value, output_names, output_dynamic_axes, output_idx)
        else:
            raise TypeError(f'ORTModule does not support the following model output type {type(output)}')

    output_names = []
    output_dynamic_axes = {}
    output_idx = [0]
    _populate_output_names_and_dynamic_axes(module_output, output_names, output_dynamic_axes, output_idx)

    return output_names, output_dynamic_axes

def get_flattened_output_module(original_module):
    """Returns a torch.nn.Module that flattens the output of the original module in its forward method"""

    def _transform_output_to_flat_tuple(output):
        """Converts the output to a flat tuple by iterating over the entire output structure"""

        def _flatten_output(output, flat_output):
            # Recursively traverse over the output and populate the flat_output with torch.Tensors

            if output is None:
                return
            elif isinstance(output, torch.Tensor):
                flat_output.append(output)
            elif isinstance(output, abc.Sequence):
                for value in output:
                    _flatten_output(value, flat_output)
            elif isinstance(output, abc.Mapping):
                for _, value in sorted(output.items()):
                    _flatten_output(value, flat_output)
            else:
                raise TypeError(f'ORTModule does not support the following output type {type(output)}.')

        flat_output = []
        _flatten_output(output, flat_output)
        return tuple(flat_output)

    class FlattenedOutputModule(torch.nn.Module):
        def __init__(self, module):
            super(FlattenedOutputModule, self).__init__()
            self._base_module = module

            def _forward(self, *args, **kwargs):
                return _transform_output_to_flat_tuple(self._base_module(*args, **kwargs))

            # Exporter does not support use of **kwargs in the forward method.
            # Work around it by making the signature of the forward method to resemble that of the
            # original model
            # Copy the forward signature from the original PyTorch module.
            self.forward = _forward.__get__(self)
            functools.update_wrapper(self.forward.__func__, module.forward.__func__)

    return FlattenedOutputModule(original_module)

def parse_inputs_for_onnx_export(all_input_parameters, onnx_graph, *inputs, **kwargs):
    # Ignore optional inputs explicitly specified as None
    # ONNX exporter may remove unused inputs
    onnx_graph_input_names = []
    if onnx_graph is not None:
        onnx_graph_input_names = set([inp.name for inp in onnx_graph.graph.input])

    input_names = []
    dynamic_axes = {}
    input_names_require_grad = []
    input_shape = []

    for input_idx, input_parameter in enumerate(all_input_parameters):
        if input_parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            # Looking at VAR_POSITIONAL parameter (*args) in the original forward method.
            # All the rest positional inputs go into this parameter.
            var_positional_idx = 0
            for i in range(input_idx, len(inputs)):
                name = f'var_positional_{input_parameter.name}{var_positional_idx}'
                var_positional_idx += 1
                inp = inputs[i]
                if inp is not None and (onnx_graph is None or name in onnx_graph_input_names):
                    if inp.requires_grad:
                        # input_names_require_grad holds all input tensors that have requires_grad
                        input_names_require_grad.append(name)

                    input_names.append(name)
                    dynamic_axes[name] = {}
                    for dim_idx in range(len(inp.shape)):
                        dynamic_axes[name].update({dim_idx : f'input{input_idx}_dim{dim_idx}'})

                    input_shape.append(list(inp.size()))
        else:
            name = input_parameter.name
            inp = None
            if input_idx < len(inputs) and inputs[input_idx] is not None:
                inp = inputs[input_idx]
            elif name in kwargs and kwargs[name] is not None:
                inp = kwargs[name]
            if inp is not None and (onnx_graph is None or name in onnx_graph_input_names):
                if inp.requires_grad:
                    # input_names_require_grad holds all input tensors that have requires_grad
                    input_names_require_grad.append(name)

                input_names.append(name)
                dynamic_axes[name] = {}
                for dim_idx in range(len(inp.shape)):
                    dynamic_axes[name].update({dim_idx : f'input{input_idx}_dim{dim_idx}'})

                input_shape.append(list(inp.size()))
    return input_names, dynamic_axes, input_names_require_grad, input_shape

def parse_outputs_for_onnx_export_and_extract_output_schema(module, inputs, kwargs):

    #   Do an inference to grab outputs
    is_train_mode = module.training
    module.eval()
    output_names = None
    output_dynamic_axes = None
    with torch.no_grad():
        # Deepcopy inputs, since input values may change after model run.
        sample_inputs_copy, sample_kwargs_copy = deepcopy_model_input(*inputs, **kwargs)
        try:
            # Deepcopy model, in case model is stateful and changes after model run.
            model_copy = copy.deepcopy(module)
        except Exception:
            model_copy = module
            warnings.warn("This model cannot be deep copied (or pickled), which is a required step for stateful models to be properly exported to ONNX."
                            " Compute will continue, but unexpected results may occur!")

        sample_outputs = model_copy(*sample_inputs_copy, **sample_kwargs_copy)

        # Parse the output and extract the output_names and output_dynamic_axes to be used for onnx export
        output_names, output_dynamic_axes = \
            _parse_outputs_and_extract_names_and_dynamic_axes(sample_outputs)
    if is_train_mode:
        module.train()

    # Return output names, output dynamic axes and output schema
    return output_names, output_dynamic_axes, _extract_output_schema(sample_outputs)
