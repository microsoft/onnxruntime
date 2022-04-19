# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import sys
import torch
import torch.utils.checkpoint
import warnings
from torch.onnx import symbolic_helper

from onnxruntime.capi._pybind_state import register_torch_autograd_function
from ._fallback import _FallbackManager, ORTModuleONNXModelException, ORTModuleTorchModelException, wrap_exception
from . import _logger

# Some autograd.Function's shouldn't be exported as PythonOp.
# If CheckpointFunction is exported as PythonOp, the checkpointed computation
# may be computed by Pytorch, not ORT. This situation is especially important
# for big models such as GPT-2. Exporting CheckpointFunction as PythonOp means
# every transformer would be computed by Pytorch and ORT doesn't contribute
# at all.
BANNED_AUTOGRAD_FUNCTION_NAMES = set(
    [torch.utils.checkpoint.CheckpointFunction.__name__])


def _export_pt_1_10(g, n, *args, **kwargs):
    '''
    This function exports PythonOp (input: "n") into a graph
    node in "g". "args" and "kwargs" are inputs to that PythonOp.
    A PythonOp represents a call to autograd.Function.
    '''
    try:
        name = kwargs['name']
        if name in BANNED_AUTOGRAD_FUNCTION_NAMES:
            raise Exception(f'The autograd.Function {name} should not be exported to ONNX. '
                            'Please replace ORTModule with HierarchalORTModule to only'
                            'wrap exportable sub-nn.Module\'s as ORTModule.')
        inplace = kwargs['inplace']
        training_mode = symbolic_helper._training_mode
        cconv = n.cconv()
        input_tensor_types = []
        input_requires_grads = []
        input_tensor_ranks = []

        input_int_scalars = []
        input_int_scalar_positions = []

        input_float_scalars = []
        input_float_scalar_positions = []

        input_int_tuples = []
        input_int_tuple_positions = []
        input_int_tuple_begins = []

        input_float_tuples = []
        input_float_tuple_positions = []
        input_float_tuple_begins = []

        input_pointer_scalars = []
        input_pointer_scalar_positions = []

        tensor_args = []
        # Encode inputs to autograd.Function.
        for i, arg, call_type in zip(range(len(args)), args, cconv):
            if call_type == 'd':
                # Got a tensor variable.
                tensor_args.append(arg)

                requires_grad = 1 if arg.requires_grad() else 0
                input_requires_grads.append(requires_grad)

                scalar_type = int(symbolic_helper.cast_pytorch_to_onnx[arg.type(
                ).scalarType()])
                input_tensor_types.append(scalar_type)
                input_tensor_ranks.append(arg.type().dim())
            elif call_type == 'c':
                # Got a non-tensor variable.
                # Non-tensor can't have gradient.
                input_requires_grads.append(0)
                if isinstance(arg, float):
                    # A float.
                    input_float_scalar_positions.append(i)
                    input_float_scalars.append(arg)
                elif isinstance(arg, int):
                    # A int.
                    input_int_scalar_positions.append(i)
                    input_int_scalars.append(arg)
                elif isinstance(arg, tuple):
                    assert len(arg) > 0
                    # A tuple of int or float.
                    if all(isinstance(ele, int) for ele in arg):
                        # A tuple of ints.
                        input_int_tuple_positions.append(i)
                        input_int_tuple_begins.append(len(input_int_tuples))
                        input_int_tuples.extend(list(arg))
                    elif all(isinstance(ele, float) for ele in arg):
                        # A tuple of floats.
                        input_float_tuple_positions.append(i)
                        input_float_tuple_begins.append(
                            len(input_float_tuples))
                        input_float_tuples.extend(list(arg))
                    else:
                        raise wrap_exception(ORTModuleONNXModelException,
                                             Exception(f'Unknown argument type found: {type(arg)}.'))
                else:
                    # All other inputs are accessed via "pointers".
                    input_pointer_scalar_positions.append(i)
                    input_pointer_scalars.append(id(arg))
            else:
                raise wrap_exception(ORTModuleONNXModelException,
                                     Exception(f'Unknown calling convention found: {i}. Only \'d\' and \'c\' are supported'))

        output_tensor_types = []
        output_tensor_ranks = []
        output_tensor_requires_grads = []
        for arg in n.outputs():
            # Type of tensor's elements.
            scalar_type = int(symbolic_helper.cast_pytorch_to_onnx[arg.type(
            ).scalarType()])
            output_tensor_types.append(scalar_type)
            output_tensor_ranks.append(arg.type().dim())
            # If output has gradient.
            requires_grad = 1 if arg.requires_grad() else 0
            output_tensor_requires_grads.append(requires_grad)

        # TODO: add fully-qualified name.
        attrs = {
            'name_s': name,
            'inplace_i': inplace,
            'input_convention_s': cconv,
            'outputs': n.outputsSize(),
            'input_tensor_types_i': input_tensor_types,
            'input_tensor_ranks_i': input_tensor_ranks,
            'input_requires_grads_i': input_requires_grads,
            'output_tensor_types_i': output_tensor_types,
            'output_tensor_ranks_i': output_tensor_ranks,
            'output_tensor_requires_grads_i': output_tensor_requires_grads,
            'training_mode_i': 1 if training_mode else 0
        }

        if len(input_int_scalars) > 0:
            attrs['input_int_scalars_i'] = input_int_scalars
            attrs['input_int_scalar_positions_i'] = input_int_scalar_positions
        if len(input_float_scalars) > 0:
            attrs['input_float_scalars_f'] = input_float_scalars
            attrs['input_float_scalar_positions_i'] = input_float_scalar_positions
        if len(input_int_tuples) > 0:
            attrs['input_int_tuples_i'] = input_int_tuples
            attrs['input_int_tuple_positions_i'] = input_int_tuple_positions
            attrs['input_int_tuple_begins_i'] = input_int_tuple_begins
        if len(input_float_tuples) > 0:
            attrs['input_float_tuples_f'] = input_float_tuples
            attrs['input_float_tuple_positions_i'] = input_float_tuple_positions
            attrs['input_float_tuple_begins_i'] = input_float_tuple_begins
        if len(input_pointer_scalars) > 0:
            attrs['input_pointer_scalars_i'] = input_pointer_scalars
            attrs['input_pointer_scalar_positions_i'] = input_pointer_scalar_positions

        returned_args = g.op("com.microsoft::PythonOp", *tensor_args, **attrs)

        return returned_args
    except Exception as e:
        sys.stdout.flush()
        sys.stderr.flush()
        raise wrap_exception(ORTModuleONNXModelException, e)

# Starting from PyTorch 1.11, there has been a change to symbolic function signature
# in terms of how additional context is accessed. More info at
# https://github.com/pytorch/pytorch/blob/6b02648479d3615fa3260961e24f38dd0f22da94/torch/onnx/symbolic_helper.py#L48
# This code can be cleaned up once support for PyTorch version < 1.11 is dropped.
try:
    from torch.onnx import SymbolicContext
    def _export(ctx: SymbolicContext, g, *args, **kwargs):
        n = ctx.cur_node
        return _export_pt_1_10(g, n, *args, **kwargs)
except ImportError:
    _export = _export_pt_1_10

def _post_process_after_export(exported_model, enable_custom_autograd_function, log_level):
    if enable_custom_autograd_function:
        return _post_process_enabling_autograd_fallback(exported_model)

    is_pythonop_needed = False
    for node in exported_model.graph.node:
        if node.domain == 'com.microsoft' and node.op_type in ["PythonOp"]:
            is_pythonop_needed = True
            break

    if is_pythonop_needed and log_level <= _logger.LogLevel.WARNING:
        warnings.warn('Detected autograd functions usage in current model, the run will fail \
                      without enabling \'_enable_custom_autograd_function\'. Please enable it with: \
                      \'module._execution_manager(is_training_mode)._enable_custom_autograd_function = True\'',
                      UserWarning)

    return exported_model


def _post_process_enabling_autograd_fallback(exported_model):
    registered_name_mappings = {}
    for kclass in torch.autograd.Function.__subclasses__():
        # Collect mapping of class names to full qualified class names.
        if kclass.__name__ not in registered_name_mappings:
            registered_name_mappings[kclass.__name__] = []
        full_qualified_name = kclass.__module__ + '.' + kclass.__qualname__
        registered_name_mappings[kclass.__name__].append(full_qualified_name)

        # Register function with class names.
        register_torch_autograd_function(kclass.__name__, kclass)

    index = 0
    for node in exported_model.graph.node:
        if node.domain == 'com.microsoft' and node.op_type in ["PythonOp"]:
            output_names = list(node.output)
            del node.output[:]
            node.output.append(output_names[0] + '_ctx')
            node.output.extend(output_names)
            for attr in node.attribute:
                if attr.name == 'name':
                    kclass_name = attr.s.decode('utf-8') if isinstance(attr.s, bytes) else attr.s
                    # If the duplicated function is used in ONNX graph, we will fail in case of a wrong function call.
                    # Todo: remove this trick once exporter can support fully qualified name for PythonOp.
                    if kclass_name in registered_name_mappings and len(registered_name_mappings[kclass_name]) > 1:
                        error_msg = 'More than one torch.autograd.Function named {}, but probabbly in different namespace. ' \
                                    'The conflicting autograd.Functions are: {}. Currently torch exporter cannot ' \
                                    'differentiate them with full qualified name, so there is a risk exported PythonOp calls a ' \
                                    'wrong autograd.Function.'.format(kclass_name, ','.join(registered_name_mappings[kclass_name]))
                        raise wrap_exception(ORTModuleONNXModelException, RuntimeError(error_msg))

                    break

        if not node.name:
            node.name = node.op_type + "_id_" + str(index)
            index += 1

    return exported_model
