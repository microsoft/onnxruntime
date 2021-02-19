import torch
from collections import abc, OrderedDict

def _process_output_name(output_name, output_index = None):
    processed_name = output_name.replace('*', '**')
    return processed_name if output_index is None else processed_name + '*{}'.format(output_index)

def _unprocess_output_name(processed_name):
    index_appended = processed_name.count('*') % 2 == 1
    output_index = None
    if index_appended:
        output_index = int(processed_name[processed_name.rfind('*')+1:])
        processed_name = processed_name[:processed_name.rfind('*')]
    return processed_name.replace('**', '*'), output_index

def create_output_dim_names_from_mapping(output):
    output_names, dynamic_axes, use_derived_module = [], {}, False
    for name, value in output.items():
        if isinstance(value, torch.Tensor):
            processed_name = _process_output_name(name)
            output_names.append(processed_name)
            dynamic_axes[processed_name] = {}
            for dim_idx in range(len(value.shape)):
                dynamic_axes[processed_name].update({dim_idx: '{}_dim{}'.format(processed_name, dim_idx)})
        elif isinstance(value, abc.Sequence):
            use_derived_module = True
            for i, sequence_value in enumerate(value):
                if not isinstance(sequence_value, torch.Tensor):
                    raise TypeError('ORTModule does not support the following model output type {} \
                        within a Sequence within a Mapping'.format(type(output)))
                processed_name = _process_output_name(name, i)
                dynamic_axes[processed_name] = {}
                output_names.append(processed_name)
                for dim_idx in range(len(sequence_value.shape)):
                    dynamic_axes[processed_name].update({dim_idx: '{}_dim{}'.format(processed_name, dim_idx)})
        else:
            raise TypeError('ORTModule does not support the following model output type {} within a Mapping'.format(type(value)))

    return output_names, dynamic_axes, use_derived_module

def create_output_dim_names(output, output_idx):
    output_names, dynamic_axes, use_derived_module = [], {}, False
    name = 'output{}'.format(output_idx)
    if isinstance(output, torch.Tensor):
        processed_name = _process_output_name(name)
        output_names.append(processed_name)
        dynamic_axes[processed_name] = {}
        for dim_idx in range(len(output.shape)):
            dynamic_axes[processed_name].update({dim_idx : '{}_dim{}'.format(processed_name, dim_idx)})
    elif isinstance(output, abc.Sequence):
        use_derived_module = True
        for i, sequence_value in enumerate(output):
            if not isinstance(sequence_value, torch.Tensor):
                raise TypeError('ORTModule does not support the following model output type {} \
                    within a Sequence within a Sequence'.format(type(output)))
            processed_name = _process_output_name(name, i)
            dynamic_axes[processed_name] = {}
            output_names.append(processed_name)
            for dim_idx in range(len(sequence_value.shape)):
                dynamic_axes[processed_name].update({dim_idx: '{}_dim{}'.format(processed_name, dim_idx)})
    else:
        raise TypeError('ORTModule does not support the following model output type {} within a Sequence'.format(type(output)))
    return output_names, dynamic_axes, use_derived_module

def populate_user_output(user_output_type, user_output_names, user_outputs):
    def _key_value_pairs_from_output(output_names, outputs):
        key_value_pairs = []
        for i, name in enumerate(user_output_names):
            output_name, output_index = _unprocess_output_name(name)
            if output_index is None:
                key_value_pairs.append((output_name, user_outputs[i]))
            elif output_index == 0:
                key_value_pairs.append((output_name, [user_outputs[i]]))
            else:
                key_value_pairs[-1][1].append(user_outputs[i])
        for i, _ in enumerate(key_value_pairs):
            if isinstance(key_value_pairs[i][1], abc.Sequence):
                key_value_pairs[i] = (key_value_pairs[i][0], tuple(key_value_pairs[i][1]))
        return key_value_pairs

    key_value_pairs = _key_value_pairs_from_output(user_output_names, user_outputs)
    if issubclass(user_output_type, abc.Mapping):
        return user_output_type(key_value_pairs)
    elif issubclass(user_output_type, abc.Sequence):
        try:
            # Try constructing the user named tuple from the output tuple
            return user_output_type(*[user_output for _, user_output in key_value_pairs])
        except TypeError:
            # The expected output type is not a namedtuple, but is a regular tuple type
            pass

    return key_value_pairs[0][1] if len(key_value_pairs) == 1 \
        else tuple(user_output for _, user_output in key_value_pairs)

def _transform_to_flat_structure(output):
    def _transform_output_from_mapping(output_dict):
        transformed_output = OrderedDict()
        for key, value in output_dict.items():
            if isinstance(value, torch.Tensor):
                processed_key = _process_output_name(key)
                transformed_output[processed_key] = value
            elif isinstance(value, abc.Sequence):
                for i, sequence_value in enumerate(value):
                    if not isinstance(sequence_value, torch.Tensor):
                        raise TypeError('ORTModule does not support the following output type {} \
                            within a Sequence within a Mapping'.format(type(sequence_value)))
                    processed_key = _process_output_name(key, i)
                    transformed_output[processed_key] = sequence_value
            else:
                raise TypeError('ORTModule does not support the following output type {} within a Mapping'.format(type(value)))
        return transformed_output

    def _transform_output_from_sequence(output_sequence):
        transformed_output = []
        for value in output_sequence:
            if isinstance(value, torch.Tensor):
                transformed_output.append(value)
            elif isinstance(value, abc.Sequence):
                for sequence_value in value:
                    if not isinstance(sequence_value, torch.Tensor):
                        raise TypeError('ORTModule does not support the following output type {} \
                            within a Sequence within a Sequence'.format(type(sequence_value)))
                    transformed_output.append(sequence_value)
            else:
                raise TypeError('ORTModule does not support the following output type {} within a Sequence'.format(type(value)))
        return tuple(transformed_output)

    output_structure = None
    if isinstance(output, abc.Mapping):
        output_structure = _transform_output_from_mapping(output)
    elif isinstance(output, abc.Sequence):
        output_structure = _transform_output_from_sequence(output)
    return output_structure

class DerivedModule(torch.nn.Module):
    def __init__(self, module):
        super(DerivedModule, self).__init__()
        self._base_module = module

    def forward(self, *args, **kwargs):
        return _transform_to_flat_structure(self._base_module(*args, **kwargs))
