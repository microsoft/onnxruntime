from collections import abc
import copy
import functools
import torch

class _TensorStub:
    # Stub for a torch.Tensor value to be used to formulate the output schema
    pass

def populate_user_output_from_schema_and_outputs(output_schema, output_names, outputs):
    """Follows the schema to generate an output that is expected by the user"""

    def replace_stub_with_tensor_value(user_output, outputs, output_idx):
        # Recursively traverse across user_output and replace all _TensorStub
        # with torch.Tensor values from outputs following output_idx

        if isinstance(user_output, _TensorStub):
            output_idx[0] += 1
            return outputs[output_idx[0]-1]

        if isinstance(user_output, abc.Sequence):
            sequence_type = type(user_output)
            user_output = list(user_output)
            for idx in range(len(user_output)):
                user_output[idx] = replace_stub_with_tensor_value(user_output[idx], outputs, output_idx)
            user_output = sequence_type(user_output)
        elif isinstance(user_output, abc.Mapping):
            for key in sorted(user_output):
                user_output[key] = replace_stub_with_tensor_value(user_output[key], outputs, output_idx)
        else:
            raise TypeError(f'ORTModule does not support the following model output type {type(user_output)}.')

        return user_output

    # Order the outputs according to the names so that the traversal order is consistent
    outputs = [x for _, x in sorted(zip(output_names, outputs))]

    # Replace every _TensorStub value in the schema with the torch.Tensor outputs calculated
    output_schema_copy = copy.deepcopy(output_schema)
    output_idx = [0]
    user_output = replace_stub_with_tensor_value(output_schema_copy, outputs, output_idx)

    return user_output

def extract_output_schema(output):
    """Extract the output schema by replacing every torch.Tensor value with _TensorStub"""

    # Depth first traversal to iterate over the output to replace every tensor with a stub
    if isinstance(output, torch.Tensor):
        return _TensorStub()

    if isinstance(output, abc.Sequence):
        sequence_type = type(output)
        output = list(output)
        for idx in range(len(output)):
            output[idx] = extract_output_schema(output[idx])
        output = sequence_type(output)
    elif isinstance(output, abc.Mapping):
        for key in sorted(output):
            output[key] = extract_output_schema(output[key])
    else:
        raise TypeError(f'ORTModule does not support the following model output type {type(output)}')

    return output

def parse_outputs_and_extract_names_and_dynamic_axes(output, output_names, output_dynamic_axes, output_idx):
    """Populate output_names and output_dynamic axes"""

    # Depth first traversal to traverse through the entire output collecting output names and dynamic axes
    if isinstance(output, torch.Tensor):
        output_name = f'output{output_idx[0]}'
        output_idx[0] += 1
        output_names.append(output_name)
        output_dynamic_axes[output_name] = {}
        for dim_idx in range(len(output.shape)):
            output_dynamic_axes[output_name].update({dim_idx: f'{output_name}_dim{dim_idx}'})
        return

    if isinstance(output, abc.Sequence):
        for value in output:
            parse_outputs_and_extract_names_and_dynamic_axes(value, output_names, output_dynamic_axes, output_idx)
    elif isinstance(output, abc.Mapping):
        for _, value in sorted(output.items()):
            parse_outputs_and_extract_names_and_dynamic_axes(value, output_names, output_dynamic_axes, output_idx)
    else:
        raise TypeError(f'ORTModule does not support the following model output type {type(output)}')

def get_flattened_output_module(original_module):
    """Returns a torch.nn.Module that flattens the output of the original module in its forward method"""

    def _transform_output_to_flat_tuple(output):
        """Converts the output to a flat tuple by iterating over the entire output structure"""

        def _flatten_output(output, flat_output):
            # Recursively traverse over the output and populate the flat_output with torch.Tensors

            if isinstance(output, torch.Tensor):
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
