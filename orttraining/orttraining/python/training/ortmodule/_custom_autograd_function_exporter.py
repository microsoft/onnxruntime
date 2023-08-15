# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import sys
from typing import Callable, ClassVar, Dict, Optional

import onnx
import torch
import torch.utils.checkpoint
from packaging import version
from torch.onnx import symbolic_helper

from onnxruntime.capi._pybind_state import register_miscellaneous_const_input, register_torch_autograd_function
from onnxruntime.training import ortmodule

from ._custom_op_symbolic_registry import pytorch_type_to_onnx, wrap_custom_export_function
from ._fallback import ORTModuleONNXModelException, wrap_exception
from ._utils import get_fully_qualified_class_name, get_runtime_pytorch_version


class PythonOpShapeInferStore:
    """A class to store shape inference functions for torch.autograd.Function."""

    _CLASS_MAP: ClassVar[Dict[str, Callable]] = {}

    @classmethod
    def register(cls, kclass: torch.autograd.Function) -> None:
        """Register a shape inference function for a torch.autograd.Function if there is staticmethod "infer_shape" defined.

        The signature of the shape inference function should be:
            @staticmethod
            def infer_shape(
                node: onnx.NodeProto,
                tensor_input_shapes: List[Optional[List[Union[int, str]]]],
                tensor_input_dtypes: List[torch.onnx.TensorProtoDataType],
            ) -> Tuple[List[Optional[List[Union[int, str]]]], List[torch.onnx.TensorProtoDataType]]:
                tensor_output_shapes = []
                tensor_output_dtypes = []
                ...
                return tensor_output_shapes, tensor_output_dtypes

        The tensor_input_shapes and tensor_input_dtypes are lists of shapes and dtypes of the input tensors.
        The tensor_output_shapes and tensor_output_dtypes are lists of shapes and dtypes of the output tensors.
        Be noted: we only pass in tensor inputs, and return tensor outputs, non-tensor inputs/outputs are ignored.

        """
        kclass_name = get_fully_qualified_class_name(kclass)
        if hasattr(kclass, "infer_shape") and kclass_name not in cls._CLASS_MAP:
            cls._CLASS_MAP[kclass_name] = kclass.infer_shape

    @classmethod
    def get_shape_infer(cls, name: str) -> Optional[Callable]:
        return cls._CLASS_MAP.get(name, None)


"""
Defines a list of names of torch.autograd.Function, for checkpoint activation purposes.

Note:
    If CheckpointFunction is exported as PythonOp, the checkpoint-ed computation
    (applied on every N transformer layer) may be computed by PyTorch, not ORT.
    This situation should be especially noted for large language models such as GPT-2.

As alternatives to using checkpoint activation:
1. Users could leverage HierarchalORTModule to wrap the model, which only wrap exportable
sub-nn.Module's as ORTModule. In this way, ideally, ORT could cover most of the model computation,
other than dispatching them to PyTorch.
2. Users could disable the check by setting the environment variable ORTMODULE_ALLOW_AUTOGRAD_CHECKPOINT=1.
This may imply that the exported model is not fully running on ORT, users should be aware of the potential
performance impact.
3. Users could also leverage ORT's memory optimization feature to achieve a similar effect as checkpointing
activations. Turn off PyTorch's checkpointing activation, then refer to env var ORTMODULE_MEMORY_OPT_CONFIG
to enable ORT's recomputation feature.

"""
_UNSUPPORTED_CKPT_FUNC_NAMES = frozenset(
    [
        # Full qualified name.
        "torch.utils.checkpoint.CheckpointFunction",
        "deepspeed.checkpointing.CheckpointFunction",
    ]
)


def _export_pt_1_10(g, n, *args, **kwargs):
    """Export torch.autograd.Function in ORT PythonOp.

    Exports PythonOp (input: "n") into a graph node in "g", and registers the PythonOp's autograd.Function in ORT backend.

    Args:
        g (jit_utils.GraphContext): The graph to export to.
        n (torch._C.Node): The PythonOp node to export, its property "pyobj" can be used to retrieve the
            torch.autograd.Function class.
            https://github.com/pytorch/pytorch/blob/68cb854d73458a14684d584c25c22b17eb79dfca/torch/csrc/jit/python/python_ir.cpp#L797
        args (list): The inputs.
        kwargs (dict): The keyword arguments.

    """
    try:
        func_class = n.pyobj().__self__
        func_full_qual_name = get_fully_qualified_class_name(func_class)

        # Check if the checkpointing activation is allowed.
        is_ckpt_activation_allowed = ortmodule._defined_from_envvar("ORTMODULE_ALLOW_AUTOGRAD_CHECKPOINT", 0) == 1
        if is_ckpt_activation_allowed is False and func_full_qual_name in _UNSUPPORTED_CKPT_FUNC_NAMES:
            raise Exception(
                f"The torch.autograd.Function {func_full_qual_name} should not be exported to ONNX. "
                "Please replace ORTModule with HierarchalORTModule to only"
                "wrap exportable sub-nn.Module's as ORTModule."
            )

        inplace = kwargs["inplace"]
        # TODO move to public API once the exporter team exposes that
        training_mode = None
        if get_runtime_pytorch_version() >= version.parse("1.12"):
            # FIXME: using private modules
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

        input_bool_scalars = []
        input_bool_scalar_positions = []

        input_int_scalars = []
        input_int_scalar_positions = []

        input_float_scalars = []
        input_float_scalar_positions = []

        input_bool_tuples = []
        input_bool_tuple_positions = []
        input_bool_tuple_begins = []

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
        # Encode inputs to torch.autograd.Function.
        for i, arg, call_type in zip(range(len(args)), args, cconv):
            if call_type == "d":
                # Got a tensor variable.
                tensor_args.append(arg)
                scalar_type = pytorch_type_to_onnx(arg.type().scalarType())
                input_tensor_types.append(scalar_type)
                input_tensor_ranks.append(arg.type().dim())
                continue

            if call_type != "c":
                raise wrap_exception(
                    ORTModuleONNXModelException,
                    Exception(f"Unknown calling convention found: {i}. Only 'd' and 'c' are supported"),
                )

            # Got a non-tensor variable.
            # Non-tensor can't have gradient.
            if isinstance(arg, float):
                # A float.
                input_float_scalar_positions.append(i)
                input_float_scalars.append(arg)
                continue
            # bool check MUST be before int check since bool is a subclass of int
            elif isinstance(arg, bool):
                # A bool.
                input_bool_scalar_positions.append(i)
                input_bool_scalars.append(int(arg))
                continue
            elif isinstance(arg, int):
                # A int.
                input_int_scalar_positions.append(i)
                input_int_scalars.append(arg)
                continue

            is_bool_tuple = False
            is_int_tuple = False
            is_float_tuple = False
            if isinstance(arg, tuple) and len(arg) > 0:
                # bool check MUST be before int check since bool is a subclass of int.
                is_bool_tuple = all(isinstance(ele, bool) for ele in arg)
                is_int_tuple = not is_bool_tuple and all(isinstance(ele, int) for ele in arg)
                is_float_tuple = not is_bool_tuple and not is_int_tuple and all(isinstance(ele, float) for ele in arg)

            # Only support tuple of bool, int or float, for other types, handle it as a pointer.
            if is_bool_tuple:
                # A tuple of bool.
                input_bool_tuple_positions.append(i)
                input_bool_tuple_begins.append(len(input_bool_tuples))
                input_bool_tuples.extend([int(ele) for ele in arg])
                continue
            elif is_int_tuple:
                # A tuple of ints.
                input_int_tuple_positions.append(i)
                input_int_tuple_begins.append(len(input_int_tuples))
                input_int_tuples.extend(list(arg))
                continue
            elif is_float_tuple:
                # A tuple of floats.
                input_float_tuple_positions.append(i)
                input_float_tuple_begins.append(len(input_float_tuples))
                input_float_tuples.extend(list(arg))
                continue

            is_inspect_activation = (
                func_full_qual_name == "onnxruntime.training.utils.hooks._subscriber_manager._InspectActivation"
            )
            if is_inspect_activation and isinstance(arg, str):
                # _InspectActivation is a special case where the first argument is a string
                # that is used to determine the activation name to be inspected.
                debug_comment += arg

            # All other inputs are accessed via "pointers".
            input_pointer_scalar_positions.append(i)
            input_pointer_scalars.append(id(arg))

            # For pointer (for example, ProcessGroup passed to PythonOp) needed for PythonOp execution,
            # we append it into a global store to hold a reference (in case it is released after module exported).
            register_miscellaneous_const_input(arg)

        output_tensor_types = []
        output_tensor_ranks = []
        for arg in n.outputs():
            # Type of tensor's elements.
            scalar_type = pytorch_type_to_onnx(arg.type().scalarType())
            output_tensor_types.append(scalar_type)
            output_tensor_ranks.append(arg.type().dim())

        attrs = {
            "func_name_s": func_full_qual_name,
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

        if len(input_bool_scalars) > 0:
            attrs["input_bool_scalars_i"] = input_bool_scalars
            attrs["input_bool_scalar_positions_i"] = input_bool_scalar_positions
        if len(input_int_scalars) > 0:
            attrs["input_int_scalars_i"] = input_int_scalars
            attrs["input_int_scalar_positions_i"] = input_int_scalar_positions
        if len(input_float_scalars) > 0:
            attrs["input_float_scalars_f"] = input_float_scalars
            attrs["input_float_scalar_positions_i"] = input_float_scalar_positions
        if len(input_bool_tuples) > 0:
            attrs["input_bool_tuples_i"] = input_bool_tuples
            attrs["input_bool_tuple_positions_i"] = input_bool_tuple_positions
            attrs["input_bool_tuple_begins_i"] = input_bool_tuple_begins
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

        # Register function with class names.
        register_torch_autograd_function(func_full_qual_name, func_class)

        # Register shape inference function.
        PythonOpShapeInferStore.register(func_class)
        return returned_args
    except Exception as e:
        sys.stdout.flush()
        sys.stderr.flush()
        raise wrap_exception(ORTModuleONNXModelException, e)  # noqa: B904


_export = wrap_custom_export_function(_export_pt_1_10)


def _post_process_after_export(
    exported_model: onnx.ModelProto, enable_custom_autograd_function: bool
) -> onnx.ModelProto:
    """Post process the exported model."""
    if enable_custom_autograd_function:
        exported_model = _post_process_enabling_autograd_function(exported_model)
    return exported_model


def _post_process_enabling_autograd_function(exported_model: onnx.ModelProto) -> onnx.ModelProto:
    # Loop all PythonOp, append "_ctx" as the first output.
    index = 0
    for node in exported_model.graph.node:
        op_name_prefix = node.op_type
        if node.domain == "com.microsoft" and node.op_type == "PythonOp":
            output_names = list(node.output)
            del node.output[:]
            node.output.append(output_names[0] + "_ctx")
            node.output.extend(output_names)
            for attr in node.attribute:
                if attr.name == "func_name":
                    kclass_name = attr.s.decode("utf-8") if isinstance(attr.s, bytes) else attr.s
                    op_name_prefix = kclass_name
                    break

        if not node.name:
            node.name = f"{op_name_prefix}_id_{index}"
            index += 1

    return exported_model
