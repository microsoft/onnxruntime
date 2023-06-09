# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import sys
import warnings

import torch
import torch.utils.checkpoint
from packaging import version
from torch.onnx import symbolic_helper

from onnxruntime.capi._pybind_state import register_miscellaneous_const_input, register_torch_autograd_function
from onnxruntime.training import ortmodule

from ._custom_op_symbolic_registry import pytorch_type_to_onnx, wrap_custom_export_function
from ._fallback import ORTModuleONNXModelException, wrap_exception
from ._utils import get_runtime_pytorch_version

# Some autograd.Function's shouldn't be exported as PythonOp.
# If CheckpointFunction is exported as PythonOp, the checkpointed computation
# may be computed by Pytorch, not ORT. This situation is especially important
# for big models such as GPT-2. Exporting CheckpointFunction as PythonOp means
# every transformer would be computed by Pytorch and ORT doesn't contribute
# at all.
BANNED_AUTOGRAD_FUNCTION_NAMES = frozenset([torch.utils.checkpoint.CheckpointFunction.__name__])


def _full_name(klass):
    module = klass.__module__
    if module == "builtins":
        return klass.__qualname__  # avoid outputs like 'builtins.str'
    return module + "." + klass.__qualname__


def _export_pt_1_10(g, n, *args, **kwargs):
    """
    This function exports PythonOp (input: "n") into a graph
    node in "g". "args" and "kwargs" are inputs to that PythonOp.
    A PythonOp represents a call to autograd.Function.
    """
    try:
        name = kwargs["name"]
        if name in BANNED_AUTOGRAD_FUNCTION_NAMES and (
            not ortmodule._defined_from_envvar("ORTMODULE_ALLOW_AUTOGRAD_CHECKPOINT", 0)
            or name != torch.utils.checkpoint.CheckpointFunction.__name__
        ):
            raise Exception(
                f"The autograd.Function {name} should not be exported to ONNX. "
                "Please replace ORTModule with HierarchalORTModule to only"
                "wrap exportable sub-nn.Module's as ORTModule."
            )
        inplace = kwargs["inplace"]
        # TODO move to public API once exporter team exposes that
        training_mode = None
        if get_runtime_pytorch_version() >= version.parse("1.12"):
            # FIXME: using privated modules
            from torch.onnx import _globals

            # before https://github.com/pytorch/pytorch/commit/c8b9b6266b505328e503b12f6a42fd88c56374f9,
            # training_mode is still a bool type
            if isinstance(_globals.GLOBALS.training_mode, bool):
                training_mode = _globals.GLOBALS.training_mode
            else:
                if _globals.GLOBALS.training_mode not in [
                    torch.onnx.TrainingMode.EVAL,
                    torch.onnx.TrainingMode.TRAINING,
                ]:
                    raise Exception(f"Unexpected training mode {_globals.GLOBALS.training_mode}")
                training_mode = _globals.GLOBALS.training_mode == torch.onnx.TrainingMode.TRAINING
        else:
            training_mode = symbolic_helper._training_mode
        cconv = n.cconv()
        input_tensor_types = []
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
        debug_comment = ""
        # Encode inputs to autograd.Function.
        for i, arg, call_type in zip(range(len(args)), args, cconv):
            if call_type == "d":
                # Got a tensor variable.
                tensor_args.append(arg)
                scalar_type = pytorch_type_to_onnx(arg.type().scalarType())
                input_tensor_types.append(scalar_type)
                input_tensor_ranks.append(arg.type().dim())
            elif call_type == "c":
                # Got a non-tensor variable.
                # Non-tensor can't have gradient.
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
                        input_float_tuple_begins.append(len(input_float_tuples))
                        input_float_tuples.extend(list(arg))
                    else:
                        raise wrap_exception(
                            ORTModuleONNXModelException, Exception(f"Unknown argument type found: {type(arg)}.")
                        )
                else:
                    if name == "_InspectActivation" and isinstance(arg, str):
                        # _InspectActivation is a special case where the first argument is a string
                        # that is used to determine the activation name to be inspected.
                        debug_comment += arg

                    # All other inputs are accessed via "pointers".
                    input_pointer_scalar_positions.append(i)
                    input_pointer_scalars.append(id(arg))

                    # For pointer (for example, ProcessGroup passed to PythonOp) needed for PythonOp execution,
                    # we append it into a global store to hold a reference (in case it is released after module exported).
                    register_miscellaneous_const_input(arg)
            else:
                raise wrap_exception(
                    ORTModuleONNXModelException,
                    Exception(f"Unknown calling convention found: {i}. Only 'd' and 'c' are supported"),
                )

        output_tensor_types = []
        output_tensor_ranks = []
        for arg in n.outputs():
            # Type of tensor's elements.
            scalar_type = pytorch_type_to_onnx(arg.type().scalarType())
            output_tensor_types.append(scalar_type)
            output_tensor_ranks.append(arg.type().dim())

        # TODO: add fully-qualified name.
        attrs = {
            "name_s": name,
            "inplace_i": inplace,
            "input_convention_s": cconv,
            "outputs": n.outputsSize(),
            "input_tensor_types_i": input_tensor_types,
            "input_tensor_ranks_i": input_tensor_ranks,
            "output_tensor_types_i": output_tensor_types,
            "output_tensor_ranks_i": output_tensor_ranks,
            "training_mode_i": 1 if training_mode else 0,
            "comment_s": debug_comment,
        }

        if len(input_int_scalars) > 0:
            attrs["input_int_scalars_i"] = input_int_scalars
            attrs["input_int_scalar_positions_i"] = input_int_scalar_positions
        if len(input_float_scalars) > 0:
            attrs["input_float_scalars_f"] = input_float_scalars
            attrs["input_float_scalar_positions_i"] = input_float_scalar_positions
        if len(input_int_tuples) > 0:
            attrs["input_int_tuples_i"] = input_int_tuples
            attrs["input_int_tuple_positions_i"] = input_int_tuple_positions
            attrs["input_int_tuple_begins_i"] = input_int_tuple_begins
        if len(input_float_tuples) > 0:
            attrs["input_float_tuples_f"] = input_float_tuples
            attrs["input_float_tuple_positions_i"] = input_float_tuple_positions
            attrs["input_float_tuple_begins_i"] = input_float_tuple_begins
        if len(input_pointer_scalars) > 0:
            attrs["input_pointer_scalars_i"] = input_pointer_scalars
            attrs["input_pointer_scalar_positions_i"] = input_pointer_scalar_positions

        returned_args = g.op("com.microsoft::PythonOp", *tensor_args, **attrs)

        return returned_args
    except Exception as e:
        sys.stdout.flush()
        sys.stderr.flush()
        raise wrap_exception(ORTModuleONNXModelException, e)  # noqa: B904


_export = wrap_custom_export_function(_export_pt_1_10)


def _post_process_after_export(exported_model, enable_custom_autograd_function):
    if enable_custom_autograd_function:
        return _post_process_enabling_autograd_fallback(exported_model)

    is_pythonop_needed = False
    for node in exported_model.graph.node:
        if node.domain == "com.microsoft" and node.op_type in ["PythonOp"]:
            is_pythonop_needed = True
            break

    if is_pythonop_needed:
        warnings.warn(
            "Detected autograd functions usage in current model, the run will fail \
                      without enabling '_enable_custom_autograd_function'. Please enable it with: \
                      'module._execution_manager(is_training_mode)._enable_custom_autograd_function = True'",
            UserWarning,
        )

    return exported_model


def _post_process_enabling_autograd_fallback(exported_model):
    registered_name_mappings = {}
    skipped_autograd_function_list = ortmodule._defined_from_envvar("ORTMODULE_SKIPPED_AUTOGRAD_FUNCTIONS", "").split(
        ","
    )
    for kclass in torch.autograd.Function.__subclasses__():
        full_qualified_name = _full_name(kclass)
        if full_qualified_name in skipped_autograd_function_list:
            continue
        # Collect mapping of class names to full qualified class names.
        if kclass.__name__ not in registered_name_mappings:
            registered_name_mappings[kclass.__name__] = []
        registered_name_mappings[kclass.__name__].append(full_qualified_name)

        # Register function with class names.
        register_torch_autograd_function(kclass.__name__, kclass)

    index = 0
    for node in exported_model.graph.node:
        if node.domain == "com.microsoft" and node.op_type in ["PythonOp"]:
            output_names = list(node.output)
            del node.output[:]
            node.output.append(output_names[0] + "_ctx")
            node.output.extend(output_names)
            for attr in node.attribute:
                if attr.name == "name":
                    kclass_name = attr.s.decode("utf-8") if isinstance(attr.s, bytes) else attr.s
                    # If the duplicated function is used in ONNX graph, we will fail in case of a wrong function call.
                    # Todo: remove this trick once exporter can support fully qualified name for PythonOp.
                    if kclass_name in registered_name_mappings and len(registered_name_mappings[kclass_name]) > 1:
                        error_msg = (
                            "More than one torch.autograd.Function named {}, but probabbly in different namespace. "
                            "The conflicting autograd.Functions are: {}. Currently torch exporter cannot "
                            "differentiate them with full qualified name, so there is a risk exported PythonOp calls a "
                            "wrong autograd.Function.".format(
                                kclass_name, ",".join(registered_name_mappings[kclass_name])
                            )
                        )
                        raise wrap_exception(ORTModuleONNXModelException, RuntimeError(error_msg))

                    break

        if not node.name:
            node.name = node.op_type + "_id_" + str(index)
            index += 1

    return exported_model
