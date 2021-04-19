from collections import abc
import copy
import inspect
import torch
import warnings


class _InputInfo(object):
    def __init__(self,
                 names,
                 shape,
                 require_grad_names=None,
                 dynamic_axes=None,
                 schema=None,
                 num_positionals=0,
                 keyword_names=None):
        self.names = names
        self.shape = shape
        self.require_grad_names = require_grad_names if require_grad_names else []
        self.dynamic_axes = dynamic_axes if dynamic_axes else {}
        self.schema = schema if schema else []
        self.num_positionals = num_positionals
        self.keyword_names = keyword_names

    def __repr__(self) -> str:
        return f'''_InputInfo class:
            \tNames:            {self.names}
            \tShape:            {self.shape}
            \tRequire gradient: {self.require_grad_names}
            \tDynamic axes:     {self.dynamic_axes}
            \tSchema:           {self.schema}
            \t#Positionals:     {self.num_positionals}
            \tKeyword names:    {self.keyword_names}'''

    def flatten(self, args, kwargs):
        ret = list(args)
        for _, kwarg in kwargs.items():
            ret.append(kwarg)
        return tuple(ret)

    def unflatten(self, flat_args):
        args = tuple(flat_args[:self.num_positionals])
        kwargs = {kwarg_name: arg for kwarg_name, arg in zip(self.keyword_names, flat_args[self.num_positionals:])}
        return args, kwargs

def _convert_input_to_list(param_names, user_input_names, buffer_names, inputs, kwargs):
    '''Creates forward `*inputs` list from user input and PyTorch initializers

    ONNX Runtime forward requires an ordered list of:
        * User input: computed from forward InferenceSession
        * Initializers: computed from original PyTorch model parameters
    '''

    # User inputs
    non_none_inputs = [inp for inp in inputs if inp is not None]
    named_buffers_iter = iter(buffer_names)
    result = []
    for input_idx, name in enumerate(user_input_names):
        inp = None
        if input_idx < len(non_none_inputs):
            inp = non_none_inputs[input_idx]
        elif name in kwargs and kwargs[name] is not None:
            inp = kwargs[name]
        elif input_idx >= len(non_none_inputs):
            # Registered buffers are translated to user_input+initializer in ONNX
            buffer_name, inp = next(named_buffers_iter)
            assert buffer_name == name, f'Input name {name} expected, but {buffer_name} found!'

        if inp is not None:
            result.append(inp)
        else:
            raise RuntimeError(f'Input is present in ONNX graph but not provided: {name}.')
    # Initializers
    for param in param_names:
        result.append(param[1])
    return result


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


class _TensorStub(object):
    '''Tensor stub class used to represent model's input or output'''

    def __init__(self, name=None, dtype=None, shape=None, shape_dims=None):
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.shape_dims = shape_dims

    def __repr__(self) -> str:
        result = '_TensorStub('
        if self.name is not None:
            result += f'name={self.name}'
        if self.dtype is not None:
            if result[-1] != '(':
                result += ', '
            result += f'dtype={self.dtype}'
        if self.shape is not None:
            if result[-1] != '(':
                result += ', '
            result += f'shape={self.shape}'
        if self.shape_dims is not None:
            if result[-1] != '(':
                result += ', '
            result += f'shape_dims={self.shape_dims}'
        result += ')'
        return result

    def __eq__(self, other):
        if not isinstance(other, _TensorStub):
            raise NotImplemented('_TensorStub must only be compared to another _TensorStub instance!')
        elif not other:
            return False
        elif self.name != other.name:
            return False
        elif self.dtype != other.dtype:
            return False
        elif self.shape != other.shape:
            return False
        elif self.shape_dims != other.shape_dims:
            return False
        return True


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


def _extract_schema(data):
    """Extract the data schema by replacing every torch.Tensor value with _TensorStub"""

    if data is None:
        return None
    # Depth first traversal to iterate over the data to replace every tensor with a stub
    elif isinstance(data, torch.Tensor):
        return _TensorStub(dtype=str(data.dtype), shape_dims=len(data.size()))

    if isinstance(data, abc.Sequence):
        sequence_type = type(data)
        data = list(data)
        for idx in range(len(data)):
            data[idx] = _extract_schema(data[idx])
        try:
            # namedtuple can be created by passing the list sequence to method _make
            data = sequence_type._make(data)
        except AttributeError:
            # If attribute error encountered, create the sequence directly
            data = sequence_type(data)
    elif isinstance(data, abc.Mapping):
        for key in sorted(data):
            data[key] = _extract_schema(data[key])
    else:
        raise TypeError(f'ORTModule does not support the following model data type {type(data)}')
    return data


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


def _transform_output_to_flat_tuple(data):
    """Converts the data to a flat tuple by iterating over the entire data structure"""

    def _flatten_data(data, flat_data):
        # Recursively traverse over the data and populate the flat_data with torch.Tensors

        if data is None:
            return
        elif isinstance(data, torch.Tensor):
            flat_data.append(data)
        elif isinstance(data, abc.Sequence):
            for value in data:
                _flatten_data(value, flat_data)
        elif isinstance(data, abc.Mapping):
            for _, value in sorted(data.items()):
                _flatten_data(value, flat_data)
        else:
            raise TypeError(f'ORTModule does not support the following data type {type(data)}.')

    flat_data = []
    _flatten_data(data, flat_data)
    return tuple(flat_data)


class _FlattenedModule(torch.nn.Module):
    def __init__(self, original_module):
        super(_FlattenedModule, self).__init__()
        self._original_module = original_module

        # Before `forward` is called, _ort_module must be assigned
        # Updated input info is needed to expand args into *args, **kwargs
        self._input_info = None

    def forward(self, *args):
        # TODO: Convert *args into *args + **kwars andd call _original_module with it
        new_args, new_kwargs = self._input_info.unflatten(args)
        return _transform_output_to_flat_tuple(self._original_module(*new_args, **new_kwargs))


def parse_inputs_for_onnx_export(all_input_parameters, onnx_graph, inputs, kwargs):

    def _add_dynamic_shape(name, input):
        dynamic_axes[name] = {}
        for dim_idx in range(len(input.shape)):
            dynamic_axes[name].update({dim_idx: f'{name}_dim{dim_idx}'})
        return dynamic_axes

    def _add_input(name, input, onnx_graph, onnx_graph_input_names):
        if input is not None and (onnx_graph is None or name in onnx_graph_input_names):
            if input.requires_grad:
                input_names_require_grad.append(name)
            input_names.append(name)
            dynamic_axes.update(_add_dynamic_shape(name, input))
            input_shape.append(list(input.size()))

    # Ignore optional inputs explicitly specified as None
    # ONNX exporter may remove unused inputs
    onnx_graph_input_names = []
    if onnx_graph is not None:
        onnx_graph_input_names = set([inp.name for inp in onnx_graph.graph.input])

    input_names = []
    dynamic_axes = {}
    input_names_require_grad = []
    input_shape = []
    var_positional_idx = 0

    for input_idx, input_parameter in enumerate(all_input_parameters):
        if input_parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            # VAR_POSITIONAL parameter carries all *args parameters from original forward method

            for args_i in range(input_idx, len(inputs)):
                name = f'{input_parameter.name}_{var_positional_idx}'
                var_positional_idx += 1
                inp = inputs[args_i]
                _add_input(name, inp, onnx_graph, onnx_graph_input_names)
        elif input_parameter.kind == inspect.Parameter.POSITIONAL_ONLY or\
             input_parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD or\
             input_parameter.kind == inspect.Parameter.KEYWORD_ONLY:
            # All positional non-*args and non-**kwargs are processed here
            name = input_parameter.name
            inp = None
            input_idx += var_positional_idx
            if input_idx < len(inputs) and inputs[input_idx] is not None:
                inp = inputs[input_idx]
            elif name in kwargs and kwargs[name] is not None:
                inp = kwargs[name]
            _add_input(name, inp, onnx_graph, onnx_graph_input_names)
        elif input_parameter.kind == inspect.Parameter.VAR_KEYWORD:
            # **kwargs is always the last argument of forward()
            for name,inp in kwargs.items():
                if name not in input_names:
                    _add_input(name, inp, onnx_graph, onnx_graph_input_names)

    # Shallow copy is ok as we need the data structure, not the content
    schema = _extract_schema({'args': copy.copy(inputs), 'kwargs': copy.copy(kwargs)})

    return _InputInfo(names=input_names,
                      shape=input_shape,
                      require_grad_names=input_names_require_grad,
                      dynamic_axes=dynamic_axes,
                      schema=schema,
                      num_positionals=len(inputs),
                      keyword_names=kwargs.keys())


def parse_outputs_for_onnx_export_and_extract_schema(module, inputs, kwargs):

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
            warnings.warn("This model cannot be deep copied (or pickled), "
                          "which is a required step for stateful models to be properly exported to ONNX."
                          " Compute will continue, but unexpected results may occur!")

        sample_outputs = model_copy(*sample_inputs_copy, **sample_kwargs_copy)

        # Parse the output and extract the output_names and output_dynamic_axes to be used for onnx export
        output_names, output_dynamic_axes = _parse_outputs_and_extract_names_and_dynamic_axes(sample_outputs)
    if is_train_mode:
        module.train()

    # Return output names, output dynamic axes and output schema
    return output_names, output_dynamic_axes, _extract_schema(sample_outputs)
