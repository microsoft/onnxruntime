# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

import sys
from typing import ClassVar

import torch
import torch.utils.checkpoint
from onnx import ModelProto
from packaging import version
from torch.onnx import symbolic_helper

from onnxruntime.capi._pybind_state import (
    register_input_alias_function,
    register_miscellaneous_const_input,
    register_shape_inference_function,
    register_torch_autograd_function,
)
from onnxruntime.training import ortmodule
from onnxruntime.training.utils import pytorch_scalar_type_to_pytorch_dtype, pytorch_type_to_onnx_dtype

from ._custom_op_symbolic_registry import wrap_custom_export_function
from ._fallback import ORTModuleONNXModelException, wrap_exception
from ._logger import LogColor
from ._utils import get_fully_qualified_class_name, get_runtime_pytorch_version


class _SpecialCustomFunctionHandler:
    """A class to handle high priority export of torch.autograd.Function.
    `register_high_priority_handler` can be used as function decorator to register a handler for a torch.autograd.Function.
    """

    _HIGH_PRIORITY_EXPORT_HANDLER_MAP: ClassVar[dict[str, callable]] = {}

    @staticmethod
    def add_handler(func_name: str, handler: callable) -> None:
        """Add a handler for a function name.

        Args:
            func_name (str): The function name.
            handler (callable): The handler.

        """
        _SpecialCustomFunctionHandler._HIGH_PRIORITY_EXPORT_HANDLER_MAP[func_name] = handler

    @staticmethod
    def get_handler(func_name: str) -> callable | None:
        """Get the handler for a function name.

        Args:
            func_name (str): The function name.

        Returns:
            callable | None: The handler.

        """
        return _SpecialCustomFunctionHandler._HIGH_PRIORITY_EXPORT_HANDLER_MAP.get(func_name, None)


def register_high_priority_handler(func_name):
    """Register a handler for a torch.autograd.Function using its full qualified class name."""

    def symbolic_wrapper(fn):
        _SpecialCustomFunctionHandler.add_handler(func_name, fn)
        return fn

    return symbolic_wrapper


def register_custom_function_schema_supplementary(kclass: torch.autograd.Function) -> None:
    """Register schema summplementaries, for example custom shape inference function and
     alias input function for a custom autograd.Function.

    1. The signature of the shape inference function should be:
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


    2. The signature of the alias input function should be:
        @staticmethod
        def alias_input(node_proto_str: str) -> Tuple[List[int], List[int]]:
            fw_alias_map = [1, -1, -1]
            bw_alias_map = [-1, 0]
            return fw_alias_map, bw_alias_map

    The alias input function should return a tuple of two lists:
    - The first list is the forward alias map, its length is equal to the number of all outputs of the node.
    - The second list is the backward alias map, its length is equal to the number of all inputs
        (tensor and non-tensor) of the node.

    """
    kclass_name = get_fully_qualified_class_name(kclass)
    if hasattr(kclass, "infer_shape"):
        register_shape_inference_function(kclass_name, kclass.infer_shape)

    if hasattr(kclass, "alias_input"):
        register_input_alias_function(kclass_name, kclass.alias_input)


def _get_training_mode() -> bool:
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

    return bool(training_mode)


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

        # Check if the function is handled by high priority exporter.
        hi_pri_handler = _SpecialCustomFunctionHandler.get_handler(func_full_qual_name)
        if hi_pri_handler:
            try_export = hi_pri_handler(g, n, *args, **kwargs)
            if try_export is not None:
                return try_export

        output_tensor_types = []
        output_tensor_ranks = []
        for arg in n.outputs():
            # Type of tensor's elements.
            scalar_type = pytorch_type_to_onnx_dtype(arg.type().scalarType())
            output_tensor_types.append(scalar_type)
            output_tensor_ranks.append(arg.type().dim())
        # Fall back to common exporter if not handled by high priority exporter.
        return _default_export(
            g,
            func_full_qual_name,
            func_class,
            n.cconv(),
            n.outputsSize(),
            output_tensor_types,
            output_tensor_ranks,
            *args,
            **kwargs,
        )

    except Exception as e:
        sys.stdout.flush()
        sys.stderr.flush()
        raise wrap_exception(ORTModuleONNXModelException, e)  # noqa: B904


def _default_export(
    g, func_full_qual_name, func_class, cconv, output_size, output_tensor_types, output_tensor_ranks, *args, **kwargs
):
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
    assert len(args) == len(cconv), "Number of arguments does not match calling convention"

    # Encode inputs to torch.autograd.Function.
    for i, arg, call_type in zip(range(len(args)), args, cconv):
        if call_type == "d":
            # Got a tensor variable.
            tensor_args.append(arg)
            scalar_type = pytorch_type_to_onnx_dtype(arg.type().scalarType())
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

        from onnxruntime.training.utils.hooks._statistics_subscriber import _InspectActivation

        is_inspect_activation = func_full_qual_name == get_fully_qualified_class_name(_InspectActivation)
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

    attrs = {
        "func_name_s": func_full_qual_name,
        "input_convention_s": cconv,
        "outputs": output_size,
        "input_tensor_types_i": input_tensor_types,
        "input_tensor_ranks_i": input_tensor_ranks,
        "output_tensor_types_i": output_tensor_types,
        "output_tensor_ranks_i": output_tensor_ranks,
        "training_mode_i": 1 if _get_training_mode() else 0,
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

    register_custom_function_schema_supplementary(func_class)

    return returned_args


_export = wrap_custom_export_function(_export_pt_1_10)


def post_process_enabling_autograd_function(exported_model: ModelProto) -> ModelProto:
    # Loop all PythonOp, append "_ctx" as the first output.
    for index, node in enumerate(exported_model.graph.node):
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

            node.name = f"{op_name_prefix}_id_{index}"

    return exported_model


@register_high_priority_handler("torch.utils.checkpoint.CheckpointFunction")
@register_high_priority_handler("deepspeed.checkpointing.CheckpointFunction")
def _gradient_checkpointing_export(g, n, *args, **kwargs):
    """
    Register specialized exporter for torch.autograd.Function(s) used for checkpoint activation purposes.

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
    activations. Turn off PyTorch's checkpointing activation, then refer to env var ORTMODULE_MEMORY_OPT_LEVEL
    to enable ORT's recomputation feature.

    """
    # Check if the checkpointing activation is allowed.
    is_ckpt_activation_allowed = ortmodule._defined_from_envvar("ORTMODULE_ALLOW_AUTOGRAD_CHECKPOINT", 0) == 1
    if is_ckpt_activation_allowed is False:
        is_layerwise_recompute_enabled = ortmodule._defined_from_envvar("ORTMODULE_MEMORY_OPT_LEVEL", 0) == 1
        if not is_layerwise_recompute_enabled:
            raise Exception(
                f"{LogColor.RED}"
                "Model uses gradient checkpointing (via {func_full_qual_name}), "
                "which is not supported for export. \n"
                "Consider these alternatives:\n"
                "1) Enable ORTModule's gradient checkpointing for similar or better "
                "memory efficiency with `export ORTMODULE_MEMORY_OPT_LEVEL=1`.\n"
                "2) Allow gradient checkpointing export by setting the environment "
                "variable `ORTMODULE_ALLOW_AUTOGRAD_CHECKPOINT=1`, though subsequent "
                "execution may fail."
                "3) Replace ORTModule with HierarchalORTModule to wrap exportable "
                "sub-nn.Module's as ORTModule.\n"
                f"{LogColor.ENDC}"
            )

        # Hitting this branch means the user has enabled layerwise recompute, but _override_gradient_checkpoint didn't
        # catch the checkpointing function. This is usually because model code is importing torch.utils.checkpoint
        # earlier than ORTModule. We should tolerantly allow this case to happen.
        raise Exception(
            f"{LogColor.RED}"
            "Model uses gradient checkpointing (via {func_full_qual_name}), which is not "
            "supported for export. \n"
            "Consider these alternatives:\n"
            "1) `ORTMODULE_MEMORY_OPT_LEVEL=1` is set but checkpoint functions in the model "
            "are not overridden during onnxruntime.training.ortmodule import, consider importing "
            "onnxruntime.training.ortmodule earlier before any model code loaded.\n"
            "2) To allow gradient checkpointing export, set `ORTMODULE_ALLOW_AUTOGRAD_CHECKPOINT=1`. "
            "Subsequent execution may fail.\n"
            "3) Replace ORTModule with HierarchalORTModule to wrap exportable sub-nn.Module's as "
            "ORTModule.\n"
            f"{LogColor.ENDC}"
        )
    else:
        return None  # Let the common exporter handle the checkpointing function


@register_high_priority_handler("bitsandbytes.autograd._functions.MatMul4Bit")
def _matmul4bit_export(g, n, *args, **kwargs):
    cconv = n.cconv()
    can_converted = (
        len(cconv) >= 5
        and cconv[0] == "d"
        and cconv[1] == "d"
        and cconv[2] == "c"
        and cconv[3] == "c"
        and cconv[4] == "c"
    )
    can_converted = can_converted and (args[2] is None and args[3] is None and args[4] is not None)
    if not can_converted:
        return None

    quant_state = args[4]
    if isinstance(quant_state, list):
        # version <= 0.41.1
        absmax, shape, dtype, blocksize, compressed_stats, quant_type, data_type = quant_state
        nested = compressed_stats is not None
    else:
        # version > 0.41.1
        absmax = quant_state.absmax
        shape = quant_state.shape
        blocksize = quant_state.blocksize
        nested = quant_state.nested
        quant_type = quant_state.quant_type

    # MatMulBnb4's blocksize needs to be a power of 2 and not smaller than 16
    if blocksize < 16 or blocksize & (blocksize - 1) != 0:
        return None

    # MatMulBnb4 does not support double de-quantization (e.g. absmax is int, needs to be dequantized too)
    if nested:
        return None

    # The PyTorch linear weight shape is [out_feature, in_feature]
    in_feature = shape[1]
    out_feature = shape[0]
    if quant_type == "fp4":
        quant_type = 0
    elif quant_type == "nf4":
        quant_type = 1
    else:
        return None
    attrs = {
        "K_i": in_feature,
        "N_i": out_feature,
        "block_size_i": blocksize,
        "quant_type_i": quant_type,
        "training_mode_i": 1 if _get_training_mode() else 0,
    }

    # Make sure the quant weight can be flatten to 1D tensor safely, which com.microsoft::MatMulBnb4 requires.
    found_dim1 = any(v == 1 for v in args[1].type().sizes())
    if not found_dim1:
        return None

    absmax = g.op(
        "Constant",
        value_t=torch.tensor(absmax, dtype=pytorch_scalar_type_to_pytorch_dtype(args[0].type().scalarType())),
    )
    quant_weight = g.op(
        "Reshape", args[1], g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64))
    )  # flatten to 1D
    tensor_args = [args[0], quant_weight, absmax]
    return g.op("com.microsoft::MatMulBnb4", *tensor_args, **attrs)


class DetermisticWrapper(torch.autograd.Function):
    """
    A wrapper for run autograd function in a deterministic way. This is required for PythonOp that needs
    recompute support.
    """

    @staticmethod
    def forward(ctx, autograd_function: torch.autograd.Function, cpu_rng_state, device_rng_state, *args):
        """For normal forward run, both cpu_rng_state and device_rng_state are None.
        For recompute run, cpu_rng_state and device_rng_state are provided, and from the normal forward run results.

        If device_rng_state does not exist (in pure CPU training for example), we still return cpu_rng_state
        as device_rng_state to avoid the exporter to handle the case where we return None as forward outputs.
        """
        original_cpu_rng_state = None
        original_cuda_rng_state = None

        if cpu_rng_state is None:
            assert device_rng_state is None, "device_rng_state must be None if cpu_rng_state is None"
            cpu_rng_state = torch.get_rng_state()
            fwd_devices = list(
                {arg.device for arg in args if isinstance(arg, torch.Tensor) and arg.device.type != "cpu"}
            )
            if len(fwd_devices) > 0:
                assert len(fwd_devices) == 1, "Only support single device for now"
                assert fwd_devices[0].type == "cuda", "Only support cuda device for now"
                device_rng_state = torch.cuda.get_rng_state()
            else:
                # Pass CPU RNG state as device RNG state if device RNG state is not provided.
                # This is to workaround the tricky case where we return None|Tensor as forward outputs.
                device_rng_state = cpu_rng_state
        else:
            assert device_rng_state is not None, "device_rng_state must be provided if cpu_rng_state is provided"
            original_cpu_rng_state = torch.get_rng_state()
            torch.set_rng_state(cpu_rng_state)

            if device_rng_state.data_ptr() != cpu_rng_state.data_ptr():
                original_cuda_rng_state = torch.cuda.get_rng_state()
                torch.cuda.set_rng_state(device_rng_state)

        outputs = autograd_function.forward(ctx, *args)

        # Append the RNG states to the outputs in the beginning.
        updated_outputs = []
        updated_outputs.append(cpu_rng_state)
        updated_outputs.append(device_rng_state)
        if isinstance(outputs, torch.Tensor):
            updated_outputs.append(outputs)
        elif isinstance(outputs, tuple):
            updated_outputs.extend(outputs)
        else:
            raise ValueError(f"Unsupported outputs type: {type(outputs)}")

        ctx.autograd_function = autograd_function

        if original_cpu_rng_state is not None:
            torch.set_rng_state(original_cpu_rng_state)

        if original_cuda_rng_state is not None:
            torch.cuda.set_rng_state(original_cuda_rng_state)

        return tuple(updated_outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        # Skip the first two RNG states grad, which should be None.
        outputs = ctx.autograd_function.backward(ctx, *grad_outputs[2:])

        updated_outputs = [None, None, None]
        if isinstance(outputs, torch.Tensor):
            updated_outputs.append(outputs)
        elif isinstance(outputs, tuple):
            updated_outputs.extend(outputs)
        else:
            raise ValueError(f"Unsupported outputs type: {type(outputs)}")

        return tuple(updated_outputs)


@register_high_priority_handler("flash_attn.bert_padding.IndexFirstAxis")
@register_high_priority_handler("flash_attn.bert_padding.IndexPutFirstAxis")
@register_high_priority_handler("flash_attn.flash_attn_interface.FlashAttnFunc")
@register_high_priority_handler("flash_attn.flash_attn_interface.FlashAttnVarlenFunc")
@register_high_priority_handler(
    "orttraining_test_ortmodule_autograd.test_determistic_pythonop_export.<locals>.TestFunction"
)
@register_high_priority_handler(
    "orttraining_test_ortmodule_api.test_layerwise_recompute_determinstic.<locals>.DropoutFunction"
)
def _determinstic_exporter(g, n, *args, **kwargs):
    """
    Export torch.autograd.Function in ORT PythonOp with deterministic wrapper. This is required for PythonOp that needs
    recompute support.

    Here, we will insert 3 inputs before the actual inputs:
    1. The first input is a constant pointer, which is not a tensor. It points to the real autograd function to execute.
    2. The second input is a tensor, which is the CPU RNG state (during export, we assign None; in memory optimizer,
      the recomputed PythonOp will take the CPU RNG state output from normal forward PythonOp node).
    3. The third input is a tensor, which is the device RNG state (during export, we assign None; in memory optimizer,
      the recomputed PythonOp will take the CUDA RNG state output from normal forward PythonOp node).

    """
    # The first input is a constant pointer, which is not a tensor. The second input rng_state is a tensor.
    cconv = "ccc" + n.cconv()
    func_class = n.pyobj().__self__
    updated_args = [func_class, None, None]
    if isinstance(args, (tuple, list)):
        updated_args.extend(args)
    else:
        updated_args.append(args)

    output_tensor_types = []
    output_tensor_ranks = []
    for arg in n.outputs():
        # Type of tensor's elements.
        scalar_type = pytorch_type_to_onnx_dtype(arg.type().scalarType())
        output_tensor_types.append(scalar_type)
        output_tensor_ranks.append(arg.type().dim())

    for _ in range(2):
        output_tensor_types.insert(0, pytorch_type_to_onnx_dtype(torch.uint8))
        output_tensor_ranks.insert(0, 1)

    func_full_qual_name = get_fully_qualified_class_name(func_class)
    default_op_outputs = _default_export(
        g,
        func_full_qual_name,
        DetermisticWrapper,
        cconv,
        n.outputsSize() + 2,
        output_tensor_types,
        output_tensor_ranks,
        *updated_args,
        **kwargs,
    )

    return default_op_outputs[2:]
