# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import copy
import gc
import inspect
import warnings
from collections import OrderedDict, abc
from functools import partial
from logging import Logger
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from onnxruntime.training.utils import (
    ORTModelInputOutputSchemaType,
    ORTModelInputOutputType,
    PrimitiveType,
    extract_data_and_schema,
    unflatten_data_using_schema,
)
from onnxruntime.training.utils.torch_io_helper import _TensorStub

from ._fallback import ORTModuleIOError, wrap_exception


class _OutputIdentityOp(torch.autograd.Function):
    """Internal class used to prepend Identity ops in model's outputs

    This class is required to support ONNX models which passthrough [some of] the models's inputs
    directly to the graph output. This is an issue because ONNX Runtime cannot build proper
    gradient graph based on this pattern.

    Adding a direct Identity Op to the user model doesn't work as the ONNX exporter would optimize it away,
    resulting in the same issue.

    Therefore a custom Autograd function was introduced to add an Identity right before the output
    in a way the ONNX exporter will not optimize it away.

    Given the model below

    .. code-block:: python

        class PassthroughNet(torch.nn.Module):
            def __init__(self, input_size, hidden_size, num_classes):
                super(PassthroughNet, self).__init__()
                self.fc1_1 = torch.nn.Linear(input_size, hidden_size)
                self.relu1 = torch.nn.ReLU()
                self.fc1_2 = torch.nn.Linear(hidden_size, num_classes)
            def forward(self, input1, passthrough_input):
                out1 = self.fc1_2(self.relu1(self.fc1_1(input1)))
                # use shape from passthrough_input
                out1 = out1.view(passthrough_input.size()[0], -1)
                return out1, passthrough_input

    We can see `passthrough_input` is part of both model input and output and the resulting
    ONNX subgraph would contain something like `output2 -> output2`.

    By prepending each model output to an :class:`_OutputIdentityOp` op, the resulting
    onnx subgraph for this example would be  `passthrough_input -> Identity -> output2`.

    TODO: Remove once PyTorch 1.8.2 or newer is released
    """

    @staticmethod
    def forward(ctx, input):
        return torch.nn.Identity()(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

    @staticmethod
    def symbolic(g, self):
        return g.op("Identity", self)


def deepcopy_model_input(
    *args, **kwargs
) -> Tuple[Sequence[ORTModelInputOutputType], Mapping[str, ORTModelInputOutputType]]:
    def extract_tensor(value):
        if isinstance(value, torch.Tensor):
            if value.requires_grad:
                return value.data.requires_grad_()
            else:
                return value.data
        else:
            return value

    sample_args_copy: Sequence[ORTModelInputOutputType] = [extract_tensor(value) for value in args]
    sample_args_copy = copy.deepcopy(tuple(sample_args_copy))

    sample_kwargs_copy: Mapping[str, ORTModelInputOutputType] = {}
    for name, value in kwargs.items():
        sample_kwargs_copy[name] = extract_tensor(value)
    sample_kwargs_copy = copy.deepcopy(sample_kwargs_copy)

    return sample_args_copy, sample_kwargs_copy


def _extract_schema(
    data: ORTModelInputOutputType, device
) -> Tuple[Sequence[ORTModelInputOutputType], ORTModelInputOutputSchemaType]:
    try:
        flatten_data, schema = extract_data_and_schema(data, constant_as_tensor=True, device=device)
        return flatten_data, schema
    except TypeError as e:
        raise wrap_exception(ORTModuleIOError, TypeError(f"ORTModule fails to extract schema from data: {e}")) from None


class _FlattenedModule(torch.nn.Module):
    def __init__(self, original_module: torch.nn.Module):
        super().__init__()
        self._original_module: torch.nn.Module = original_module

        # Before ONNX export, we flatten the args and kwargs into a 1-D list of tensors to make torch.export happy.
        # As a result, we need to unflatten the args and kwargs back to the original structure before calling the
        # original module's forward function.
        # So we need set those information that are needed to unflatten the args and kwargs, before calling the
        # torch.export.
        self._device: Optional[torch.device] = None
        self._args_schema: Optional[ORTModelInputOutputSchemaType] = None
        self._kwargs_schema: Optional[ORTModelInputOutputSchemaType] = None
        self._num_positionals: Optional[int] = None

        # Similarly, to make torch.export happy, we need to flatten the original module's outputs into a 1-D list of tensors.
        # Need to keep the output schema to unflatten the outputs back to the original structure.
        # Then those code depends on the original structure of the outputs can work properly.
        self._output_schema: Optional[ORTModelInputOutputSchemaType] = None

    def forward(self, *args):
        new_args = unflatten_data_using_schema(args[: self._num_positionals], self._args_schema)
        new_kwargs = unflatten_data_using_schema(args[self._num_positionals :], self._kwargs_schema)

        original_outputs = self._original_module(*new_args, **new_kwargs)

        # Flatten the outputs
        flatten_outputs, self._output_schema = _extract_schema(original_outputs, self._device)

        # Append _OutputIdentityOp to the outputs to support passthrough outputs
        final_flatten_outputs = []
        for output in flatten_outputs:
            final_flatten_outputs.append(_OutputIdentityOp.apply(output))

        return final_flatten_outputs


class ModelInfoForExport:
    def __init__(
        self,
        onnx_graph_input_names: List[str],
        onnx_graph_input_names_require_grad: List[str],
        onnx_graph_input_dynamic_axes_map: Dict[str, Dict[int, str]],
        onnx_graph_input_shapes: List[List[int]],
        onnx_graph_input_data_accessor: Optional[Dict[str, callable]] = None,
        onnx_graph_input_arg_schema: Optional[Dict[str, ORTModelInputOutputSchemaType]] = None,
        onnx_graph_input_kwarg_schema: Optional[Dict[str, ORTModelInputOutputSchemaType]] = None,
        num_positional_args: int = 0,
        export_mode: Optional[int] = None,
        export_extra_kwargs: Optional[Dict[str, any]] = None,
    ):
        # Value can be either torch.onnx.TrainingMode.TRAINING or torch.onnx.TrainingMode.EVAL
        self.export_mode = export_mode

        # Exporter can take extra arguments for ORTModule extensions
        # It cannot overlap with required/immutable arguments (validated in runtime)
        self.export_extra_kwargs = export_extra_kwargs

        # Input names parsed and then flatten from the model's forward function signature.
        # This should contains ONLY the user defined input names
        # Be noted: some of the input might not be used by the model for its compute.
        self.onnx_graph_input_names: List[str] = onnx_graph_input_names

        # A subset of onnx_graph_input_names.
        # Input names that require gradient parsed and then flatten from the model's forward function signature
        # This should contains ONLY the user defined input names
        # Be noted: some of the input might not be used by the model for its compute.
        self.onnx_graph_input_names_require_grad: List[str] = onnx_graph_input_names_require_grad

        # Create symbolic names for each dimension of the graph input (e.g. onnx_graph_input_names).
        # The key is the input name, the value is a dict of {dim_index: symbolic_dim_name}
        # e.g. {"input1": {0: "input1_dim0", 1: "input1_dim1"}, "input2": {0: "input2_dim0"}}
        self.onnx_graph_input_dynamic_axes_map: Dict[str, Dict[int, str]] = onnx_graph_input_dynamic_axes_map

        self.onnx_graph_input_shapes: List[List[int]] = onnx_graph_input_shapes

        # The input args schema for the original model's forward function.
        # Only contains the schema for those inputs used by the model for its compute (e.g. as the inputs
        # of the export model).
        self.onnx_graph_input_arg_schema: Dict[str, ORTModelInputOutputSchemaType] = onnx_graph_input_arg_schema

        # The input kwargs schema for the original model's forward function.
        # Only contains the schema for those inputs used by the model for its compute (e.g. as the inputs
        # of the export model).
        self.onnx_graph_input_kwarg_schema: Dict[str, ORTModelInputOutputSchemaType] = onnx_graph_input_kwarg_schema

        self.num_positional_args: int = num_positional_args

        # A function to access the input data from the args and kwargs.
        # If it is not None, the length is same as onnx_graph_input_names.
        # For i-th input name, we can use the i-th function to get the input data from args and kwargs.
        self.onnx_graph_input_data_accessor: Optional[Dict[str, callable]] = onnx_graph_input_data_accessor

    def __str__(self) -> str:
        return f"""ModelInfoForExport class:
            \tExport mode:                      {self.export_mode}
            \tExport extra kwargs:              {self.export_extra_kwargs}
            \tInput names:                      {self.onnx_graph_input_names}
            \tInput names require grad:         {self.onnx_graph_input_names_require_grad}
            \tInput dynamic axes:               {self.onnx_graph_input_dynamic_axes_map}
            \tInput shapes:                     {self.onnx_graph_input_shapes}"""

    def __repr__(self) -> str:
        return self.__str__()


def _arg_access_with_index_func(arg_index, args, kwargs):
    return args[arg_index]


def _kwarg_access_with_name_func(name, args, kwargs):
    return kwargs[name]


class SkipRetValue:
    """A placeholder class to indicate that the return value of a function should be skipped"""

    pass


def parse_inputs_for_onnx_export(
    all_input_parameters: List[inspect.Parameter],
    args: Sequence[ORTModelInputOutputType],
    kwargs: Mapping[str, ORTModelInputOutputType],
    constant_as_tensor: bool,
    device: torch.device,
    export_mode: int,
    export_extra_kwargs: Optional[Dict[str, any]] = None,
) -> ModelInfoForExport:
    """Parses through the model inputs and returns _InputInfo.

    Loop through all input parameters, try to flatten them into a 1-D list of inputs. For nested data in the inputs,
    construct the name in hierarchical order.

    Example 1, arg is a list, kwarg is a dict:
        args = [arg1, arg2], kwargs = {"a": 4, "b": 5},
        input_names = ["arg1", "arg2",  "a", "b"].

    Example 2, arg is a list, kwarg is a dict of mixed list and scalar:
        args = [arg1, arg2], kwargs = {"a": [4, 5], "b": 6},
        input_names = ["arg1", "arg2",  "a_0", "a_1", "b"].

    Example 3, arg is a list, kwarg is a dict of mixed dict and scalar:
        args = [arg1, arg2], kwargs = {"a": {"c": 4, "d": 5}, "b": 6},
        input_names = ["arg1", "arg2",  "a_c", "a_d", "b"].

    Args:
        all_input_parameters: All inspected input parameters from the original model forward function signature.
        args: The positional arguments of the model.
        kwargs: The keyword arguments of the model.
        constant_as_tensor: Whether to treat constant inputs as tensors.
        device: The device to be used for constant inputs.

    """

    tensor_idx = [-1]

    def _add_dynamic_shape(name, input) -> Dict[str, Dict[int, str]]:
        dynamic_axes[name] = {}
        for dim_idx in range(len(input.shape)):
            dynamic_axes[name].update({dim_idx: f"{name}_dim{dim_idx}"})
        return dynamic_axes

    def _warn_of_constant_inputs(data):
        warnings.warn(f"Received input of type {type(data)} is treated as a constant by ORT by default.")

    def _add_input(name: str, input_value, onnx_graph_input_names: List[str], cur_func: Callable):
        """Returns number of expanded non none inputs that _add_input processed"""

        # in case the input is already handled.
        if name in visited_input_names:
            return SkipRetValue()

        visited_input_names.append(name)

        value = input_value
        if value is None:
            _warn_of_constant_inputs(value)
            return value
        elif isinstance(value, str):
            _warn_of_constant_inputs(value)
            return value
        elif PrimitiveType.is_primitive_type(value):
            if constant_as_tensor:
                value = PrimitiveType.get_tensor(value, device)
            else:
                _warn_of_constant_inputs(value)
                return value
        elif isinstance(value, abc.Sequence):
            sequence_type = type(value)
            stubbed_schema = []

            # If the input is a sequence (like a list), expand the list so that
            # each element of the list is an input by itself.
            for i, val in enumerate(value):
                # Name each input with the index appended to the original name of the
                # argument.

                def _access_func(i, cur_func, args, kwargs):
                    return cur_func(args, kwargs)[i]

                input_schema = _add_input(
                    f"{name}_{i}",
                    val,
                    onnx_graph_input_names,
                    partial(_access_func, i, cur_func),
                )

                if not isinstance(input_schema, SkipRetValue):
                    stubbed_schema.append(input_schema)

            # Return here since the list by itself is not a valid input.
            # All the elements of the list have already been added as inputs individually.

            try:
                # namedtuple can be created by passing the list sequence to method _make
                stubbed_schema = sequence_type._make(stubbed_schema)
            except AttributeError:
                # If attribute error is encountered, create the sequence directly
                stubbed_schema = sequence_type(stubbed_schema)
            return stubbed_schema

        elif isinstance(value, abc.Mapping):
            dict_type = type(value)
            stubbed_schema = OrderedDict()

            # If the input is a mapping (like a dict), expand the dict so that
            # each element of the dict is an input by itself.
            for key, val in value.items():

                def _access_func(key, cur_func, args, kwargs):
                    return cur_func(args, kwargs)[key]

                input_schema = _add_input(
                    f"{name}_{key}",
                    val,
                    onnx_graph_input_names,
                    partial(_access_func, key, cur_func),
                )

                if not isinstance(input_schema, SkipRetValue):
                    stubbed_schema[key] = input_schema

            # Return here since the dict by itself is not a valid input.
            # All the elements of the dict have already been added as inputs individually.

            stubbed_schema = dict_type(**stubbed_schema)
            return stubbed_schema

        if isinstance(value, torch.Tensor):
            onnx_graph_input_names.append(name)
            data_accessors[name] = cur_func
            if value.requires_grad:
                input_names_require_grad.append(name)
            dynamic_axes.update(_add_dynamic_shape(name, value))
            input_shape.append(list(value.size()))
            tensor_idx[0] += 1
            return _TensorStub(
                tensor_idx[0],
                dtype=str(value.dtype),
                shape_dims=len(value.size()),
                name=name,
            )

    visited_input_names: List[str] = []

    onnx_graph_input_names: List[str] = []
    dynamic_axes: Dict[str, Dict[int, str]] = {}
    input_names_require_grad: List[str] = []
    input_shape: List[List[int]] = []
    input_arg_schema: Dict[str, ORTModelInputOutputSchemaType] = OrderedDict()
    input_kwarg_schema: Dict[str, ORTModelInputOutputSchemaType] = OrderedDict()
    data_accessors: Dict[str, Callable] = OrderedDict()
    num_positional_args: int = 0

    var_positional_idx = 0

    # Be noted, all_input_parameters is a list of inspect.Parameters parsed from the original module's forward method.
    # While the execution manager's forward function will map all given model inputs to *args and **kwargs, so it is
    # possible the input parameter list cannot represent the real model inputs given here (e.g., *args, **kwargs).
    # But it is still fine to use all_input_parameters to make sure all model inputs are covered.
    #
    # Here is an example caused by the mismatch between all_input_parameters and real model inputs.
    #   def foo(*args, named_kwargs, **kwargs):
    #       ... print("foo")
    # From inspection,
    #   > ('args', <_ParameterKind.VAR_POSITIONAL: 2>)
    #   > ('named_kwargs', <_ParameterKind.KEYWORD_ONLY: 3>)
    #   > ('kwargs', <_ParameterKind.VAR_KEYWORD: 4>)
    #
    # At this point, 'named_kwargs' exists in **kwargs as a result of ORTModule's forward parse all original
    # model inputs in to *args and **kwargs.
    # When we loop `all_input_parameters``, for the `named_kwargs`, we will try to handle it in KEYWORD_ONLY branch.
    # Additionally in VAR_KEYWORD branch, we will get the `named_kwargs` value again, because its name exists in the
    # `kwargs`. So _add_input avoids handling the `named_kwargs` twice, check test case `test_named_kwargs_dict_input`
    # for the details.
    for input_idx, input_parameter in enumerate(all_input_parameters):
        if input_parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            # VAR_POSITIONAL parameter carries all *args parameters from original forward method

            for args_i in range(input_idx, len(args)):
                name = f"{input_parameter.name}_{var_positional_idx}"
                var_positional_idx += 1
                num_positional_args += 1
                inp = args[args_i]
                schema = _add_input(name, inp, onnx_graph_input_names, partial(_arg_access_with_index_func, args_i))
                if not isinstance(schema, SkipRetValue):
                    input_arg_schema[name] = schema
        elif (
            input_parameter.kind == inspect.Parameter.POSITIONAL_ONLY
            or input_parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            or input_parameter.kind == inspect.Parameter.KEYWORD_ONLY
        ):
            # All positional non-*args and non-**kwargs are processed here
            name = input_parameter.name
            inp = None
            input_idx += var_positional_idx  # noqa: PLW2901
            access_func = None
            schema_to_write = None
            if input_idx < len(args) and args[input_idx] is not None:
                inp = args[input_idx]
                num_positional_args += 1
                access_func = partial(_arg_access_with_index_func, input_idx)
                schema_to_write = input_arg_schema
            elif name in kwargs and kwargs[name] is not None:
                inp = kwargs[name]
                access_func = partial(_kwarg_access_with_name_func, name)
                schema_to_write = input_kwarg_schema
            else:
                continue

            schema = _add_input(name, inp, onnx_graph_input_names, access_func)
            if not isinstance(schema, SkipRetValue):
                schema_to_write[name] = schema

        elif input_parameter.kind == inspect.Parameter.VAR_KEYWORD:
            # **kwargs is always the last argument of forward()
            for name, inp in kwargs.items():
                schema = _add_input(
                    name,
                    inp,
                    onnx_graph_input_names,
                    partial(_kwarg_access_with_name_func, name),
                )
                if not isinstance(schema, SkipRetValue):
                    input_kwarg_schema[name] = schema

    exported_graph = ModelInfoForExport(
        onnx_graph_input_names=onnx_graph_input_names,
        onnx_graph_input_names_require_grad=input_names_require_grad,
        onnx_graph_input_dynamic_axes_map=dynamic_axes,
        onnx_graph_input_shapes=input_shape,
        onnx_graph_input_data_accessor=data_accessors,
        onnx_graph_input_arg_schema=input_arg_schema,
        onnx_graph_input_kwarg_schema=input_kwarg_schema,
        num_positional_args=num_positional_args,
        export_mode=export_mode,
        export_extra_kwargs=export_extra_kwargs,
    )

    return exported_graph


def calculate_total_parameter_size_in_bytes(module: torch.nn.Module) -> int:
    """Calculate the total parameter size in bytes"""
    total_size = 0
    for p in module.parameters():
        total_size += p.numel() * p.element_size()
    return total_size


def can_module_be_deep_cloned(module: torch.nn.Module, device: Optional[torch.device]) -> bool:
    """Check if the module can be cloned

    If the 2 times total module parameter size >= device memory, the module cannot be cloned.
    > Initially there is one set of parameters;
    >  parse_outputs_for_onnx_export_and_extract_schema want to clone the full module including the frozen weight;
    > PyTorch ONNX exporter will clone the trainable parameters;

    So as long as the module can be cloned in parse_outputs_for_onnx_export_and_extract_schema, it is safe
    to export the model without OOM. Here we return whether can clone the module in
    parse_outputs_for_onnx_export_and_extract_schema.

    Args:
        module: The module to be cloned.
        device: The device to be used for cloning.
    """

    if device is None or device.type != "cuda":
        return True

    total_size = calculate_total_parameter_size_in_bytes(module)
    return total_size * 2 < torch.cuda.get_device_properties(device).total_memory * 0.90  # give a 10% buffer


def parse_outputs_for_onnx_export_and_extract_schema(
    module,
    flatten_args: Sequence[ORTModelInputOutputType],
    logger: Logger,
    clone_module: bool,
):
    # Perform a forward call to grab outputs
    output_names = None
    output_dynamic_axes = None
    deep_copied = False
    kwargs = {}
    logger.info("Running model forward to infer output schema and dynamic axes...")
    with torch.no_grad():
        # Deepcopy inputs, since input values may change after model run.
        sample_args_copy, sample_kwargs_copy = deepcopy_model_input(*flatten_args, **kwargs)
        try:
            if clone_module:
                # Deepcopy model, in case model is stateful and changes after model run.
                model_copy = copy.deepcopy(module)
                deep_copied = True
            else:
                model_copy = module
        except Exception:
            model_copy = module
            logger.warning(
                "This model cannot be deep copied (or pickled), "
                "which is a required step for stateful models to be properly exported to ONNX."
                " Compute will continue, but unexpected results may occur!"
            )

        sample_outputs = model_copy(*sample_args_copy, **sample_kwargs_copy)

        # Parse the output and extract the output_names and output_dynamic_axes to be used for onnx export
        output_names: List[str] = []
        output_dynamic_axes: Dict[str, Dict[int, str]] = {}
        for output_idx, output in enumerate(sample_outputs):
            output_name = f"output-{output_idx}"
            output_names.append(output_name)
            output_dynamic_axes[output_name] = {}
            for dim_idx in range(len(output.shape)):
                output_dynamic_axes[output_name].update({dim_idx: f"{output_name}_dim{dim_idx}"})

        original_module_output_schema = model_copy._output_schema

    if deep_copied:
        del model_copy
        gc.collect()
        if torch.cuda.is_available():
            # Trigger python GC is not enough.
            # Release the memory cached by torch.
            torch.cuda.empty_cache()
    # Return output names, output dynamic axes and output schema
    return output_names, output_dynamic_axes, original_module_output_schema
