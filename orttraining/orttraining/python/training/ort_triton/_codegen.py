# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
Generate code for each IR node.
Mostly, Nodes are classified into two categories:
    1. ElementwiseKernelNode: compute a tensor from other tensors, e.g. ElementwiseKernelNode
    2. ReduceKernelNode: perform a reduction computation on a tensor, e.g. reduce_sum/max/min
        one or more axes are supported

"""

from typing import Tuple

import numpy as np
import sympy
import torch
from sympy.codegen.rewriting import create_expand_pow_optimization

from ._common import CodeBuffer, CodegenContext, NodeVisitor
from ._ir import (
    ComputeNode,
    DropoutNode,
    ElementwiseKernelNode,
    IONode,
    IRNode,
    KernelNode,
    ModuleNode,
    OffsetCalculator,
    ReduceForLoopEnd,
    ReduceForLoopStart,
    ReduceKernelNode,
    ReduceNode,
)
from ._lowering import lower
from ._sorted_graph import SortedGraph
from ._sympy_utils import parse_shape, sympy_dot
from ._utils import may_add_brackets


class TritonCodegen(NodeVisitor):
    """
    Specialized codegen for Triton backend.
    """

    def __init__(self):
        super().__init__()

    def codegen(self, node: IRNode, context: CodegenContext, code_buffer: CodeBuffer, indent: int):
        func = getattr(self, node.__class__.__name__)
        assert func is not None, "unimplemented node: %s" % node.__class__.__name__
        func(node, context, code_buffer, indent)

    def _get_elementwise_offset_mask(self, offset_calc: OffsetCalculator, arg_name: str) -> Tuple[str, str]:
        if offset_calc.is_x_reduced(arg_name):
            return "", ""
        if offset_calc.is_same_x_shape(arg_name):
            return "xindex", "xmask" if offset_calc.requires_x_mask else ""
        strides = offset_calc.get_input_strides(arg_name)
        idx_var = [f"x{idx}" for idx in range(len(strides))]
        expand_opt = create_expand_pow_optimization(6)
        offset_str = str(expand_opt(sympy_dot(parse_shape(idx_var), strides)))
        return offset_str, "xmask" if offset_calc.requires_x_mask else ""

    def _get_reduce_offset_mask(self, offset_calc: OffsetCalculator, arg_name: str) -> Tuple[str, str]:
        offset_strs = []
        mask_strs = []
        if not offset_calc.is_x_reduced(arg_name):
            x_strides = offset_calc.get_x_input_strides(arg_name)
            if offset_calc.is_same_x_shape(arg_name):
                xindex_str = "xindex" if x_strides[-1] == sympy.Integer(1) else f"xindex * {x_strides[-1]}"
            else:
                idx_var = [f"x{idx}" for idx in range(len(x_strides))]
                expand_opt = create_expand_pow_optimization(6)
                xindex_str = str(expand_opt(sympy_dot(parse_shape(idx_var), x_strides)))
            offset_strs.append(xindex_str)
            if offset_calc.requires_x_mask:
                mask_strs.append("xmask")

        if not offset_calc.is_r_reduced(arg_name):
            r_strides = offset_calc.get_r_input_strides(arg_name)
            if offset_calc.is_same_r_shape(arg_name):
                rindex_str = "rindex" if r_strides[-1] == sympy.Integer(1) else f"rindex * {r_strides[-1]}"
            else:
                idx_var = [f"r{idx}" for idx in range(len(r_strides))]
                expand_opt = create_expand_pow_optimization(6)
                rindex_str = str(expand_opt(sympy_dot(parse_shape(idx_var), r_strides)))
            offset_strs.append(rindex_str)
            if offset_calc.requires_r_mask:
                mask_strs.append("rmask")

        return " + ".join(offset_strs), " & ".join(mask_strs)

    def _get_offset_mask(self, node: OffsetCalculator, arg_name: str) -> Tuple[str, str]:
        return (
            self._get_reduce_offset_mask(node, arg_name)
            if node.is_reduction
            else self._get_elementwise_offset_mask(node, arg_name)
        )

    def IONode(self, node: IONode, context: CodegenContext, code_buffer: CodeBuffer, indent: int):  # noqa: N802
        space_indent = " " * indent
        name = node.tensor_arg.name
        var_name = context.get_variable_name(name)
        internal_var_name = context.get_internal_variable_name(name)
        assert (
            var_name != internal_var_name
        ), f"variable name {var_name} and its internal variable name should not be the same."

        offset_str, mask_str = self._get_offset_mask(node.offset_calc, node.tensor_arg.name)
        if offset_str:
            offset_str = f" + {offset_str}"
        if mask_str:
            mask_str = f", {mask_str}"
        if node.is_load and mask_str:
            mask_str += ", other=0.0"

        if node.is_load:
            code_buffer += f"{space_indent}{internal_var_name} = tl.load({var_name}{offset_str}{mask_str})\n"
        else:
            code_buffer += f"{space_indent}tl.store({var_name}{offset_str}, {internal_var_name}{mask_str})\n"

    def _gen_kernel_signature(self, node: KernelNode, context: CodegenContext, code_buffer: CodeBuffer, indent: int):
        is_reduction = node.offset_calc.is_reduction
        space_indent = " " * indent
        autotune_configs_str = ""
        for config in node.offset_calc.autotune_configs.configs:
            if is_reduction:
                autotune_configs_str += (
                    f'{space_indent}        triton.Config({{"XBLOCK": {config[0]}, "RBLOCK": {config[1]}}}, '
                    f"num_warps={config[2]}),\n"
                )
            else:
                autotune_configs_str += (
                    f'{space_indent}        triton.Config({{"XBLOCK": {config[0]}}}, num_warps={config[2]}),\n'
                )
        keys_str = '"xnumel", "rnumel"' if is_reduction else '"xnumel"'
        input_args = [context.get_variable_name(input.name) for input in node.inputs]
        input_args_str = ", ".join(input_args)
        if input_args_str:
            input_args_str += ", "

        output_args = [context.get_variable_name(output.name) for output in node.outputs]
        output_args_str = ", ".join(output_args) + ", "

        other_input_args = "seed_cuda, " if node.has_dropout else ""
        # Support symbolic shape if any.
        symbolic_shape_args_str = ", ".join(node.symbolic_shape_variables)
        if symbolic_shape_args_str:
            other_input_args += f"{symbolic_shape_args_str}, "

        blocks_str = (
            "xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr"
            if is_reduction
            else "xnumel, XBLOCK: tl.constexpr"
        )

        code_buffer += (
            f"{space_indent}@triton.autotune(\n"
            f"{space_indent}    configs=[\n"
            f"{autotune_configs_str}"
            f"{space_indent}    ],\n"
            f"{space_indent}    key=[{keys_str}],\n"
            f"{space_indent})\n"
            f"{space_indent}@triton.jit\n"
            f"{space_indent}def {node.name}({input_args_str}{output_args_str}{other_input_args}{blocks_str}):\n"
        )

    def ElementwiseKernelNode(  # noqa: N802
        self, node: ElementwiseKernelNode, context: CodegenContext, code_buffer: CodeBuffer, indent: int
    ):
        self._gen_kernel_signature(node, context, code_buffer, indent)
        offset_calc = node.offset_calc
        indent += 4
        space_indent = " " * indent
        code_buffer += (
            f"{space_indent}xnumel = {offset_calc.x_numel}\n"
            f"{space_indent}xoffset = tl.program_id(0) * XBLOCK\n"
            f"{space_indent}xindex = xoffset + tl.arange(0, XBLOCK)\n"
        )
        if offset_calc.requires_x_mask:
            code_buffer += f"{space_indent}xmask = xindex < xnumel\n"
        for idx in range(offset_calc.x_rank):
            if idx in offset_calc.x_compute_dims:
                div_str = (
                    f" // {may_add_brackets(str(offset_calc.x_strides[idx]))}" if idx != offset_calc.x_rank - 1 else ""
                )
                mod_str = f" % {may_add_brackets(str(offset_calc.x_dims[idx]))}" if idx != 0 else ""
                code_buffer += f"{space_indent}x{idx} = xindex{div_str}{mod_str}\n"
        code_buffer += "\n"

        if node.has_dropout:
            code_buffer += (
                f"{space_indent}t_seed_cuda = tl.load(seed_cuda)\n"
                f"{space_indent}t_seed_cuda = tl.broadcast_to(t_seed_cuda, [XBLOCK])\n"
            )

        for ir_node in node.sub_nodes:
            ir_node.codegen(self, context, code_buffer, indent)

    def ReduceKernelNode(  # noqa: N802
        self, node: ReduceKernelNode, context: CodegenContext, code_buffer: CodeBuffer, indent: int
    ):
        self._gen_kernel_signature(node, context, code_buffer, indent)
        offset_calc = node.offset_calc
        indent += 4
        space_indent = " " * indent
        code_buffer += (
            f"{space_indent}xnumel = {offset_calc.x_numel}\n"
            f"{space_indent}rnumel = {offset_calc.r_numel}\n"
            f"{space_indent}xoffset = tl.program_id(0) * XBLOCK\n"
            f"{space_indent}xindex = xoffset + tl.arange(0, XBLOCK)[:, None]\n"
            f"{space_indent}rbase = tl.arange(0, RBLOCK)[None, :]\n"
        )
        if offset_calc.requires_x_mask:
            code_buffer += f"{space_indent}xmask = xindex < xnumel\n"
        for idx in range(offset_calc.x_rank):
            if idx in offset_calc.x_compute_dims:
                div_str = (
                    f" // {may_add_brackets(str(offset_calc.x_strides[idx]))}" if idx != offset_calc.x_rank - 1 else ""
                )
                mod_str = f" % {may_add_brackets(str(offset_calc.x_dims[idx]))}" if idx != 0 else ""
                code_buffer += f"{space_indent}x{idx} = xindex{div_str}{mod_str}\n"
        code_buffer += "\n"

        if node.has_dropout:
            code_buffer += f"{space_indent}t_seed_cuda = tl.load(seed_cuda)\n"
            code_buffer += f"{space_indent}t_seed_cuda = tl.broadcast_to(t_seed_cuda, [XBLOCK, RBLOCK])\n"

        if not offset_calc.autotune_configs.requires_for_loop:
            code_buffer += f"{space_indent}rindex = rbase\n"
            if offset_calc.requires_r_mask:
                code_buffer += f"{space_indent}rmask = rindex < rnumel\n"
            for idx in range(offset_calc.r_rank):
                if idx in offset_calc.r_compute_dims:
                    div_str = (
                        f" // {may_add_brackets(str(offset_calc.r_strides[idx]))}"
                        if idx != offset_calc.r_rank - 1
                        else ""
                    )
                    mod_str = f" % {may_add_brackets(str(offset_calc.r_dims[idx]))}" if idx != 0 else ""
                    code_buffer += f"{space_indent}r{idx} = rindex{div_str}{mod_str}\n"

        for ir_node in node.sub_nodes:
            ir_node.codegen(self, context, code_buffer, indent)
            if isinstance(ir_node, ReduceForLoopStart):
                indent += 4
            elif isinstance(ir_node, ReduceForLoopEnd):
                indent -= 4

    _COMPUTE_CODE_TEMPLATES = {  # noqa: RUF012
        "Add": "{indent}{o0} = {i0} + {i1}\n",
        "Sub": "{indent}{o0} = {i0} - {i1}\n",
        "Mul": "{indent}{o0} = {i0} * {i1}\n",
        "Div": "{indent}{o0} = {i0} / {i1}\n",
        "Relu": "{indent}{o0} = tl.maximum({i0}, 0.0)\n",
        "Pow": "{indent}{o0} = tl.math.pow({i0}, {i1})\n",
        "Pow2": "{indent}{o0} = {i0} * {i0}\n",
        "Pow3": "{indent}{o0} = {i0} * {i0} * {i0}\n",
        "Sqrt": "{indent}{o0} = tl.sqrt({i0})\n",
        "Rsqrt": "{indent}{o0} = 1.0 / tl.sqrt({i0})\n",
        "Cast": "{indent}{o0} = {i0}.to(tl.{dtype})\n",
        "CastBool": "{indent}{o0} = {i0} != 0\n",
        "Erf": "{indent}{o0} = tl.libdevice.erf({i0})\n",
        "Gelu": "{indent}{o0} = (tl.libdevice.erf({i0} / 1.41421356237) + 1.0) * 0.5\n",
        "Exp": "{indent}{o0} = tl.exp({i0})\n",
        "Tanh": "{indent}{o0} = tl.libdevice.tanh({i0})\n",
        "Where": "{indent}{o0} = tl.where({i0}, {i1}, {i2})\n",
        "Sigmoid": "{indent}{o0} = tl.sigmoid({i0})\n",
        "Log": "{indent}{o0} = tl.log({i0})\n",
        "DropoutGrad": "{indent}p = 1 - {i2}\n{indent}{o0} = tl.where({i1}, {i0} / p, 0.0)\n",
        "Identity": "{indent}{o0} = {i0}\n",
    }

    def ComputeNode(  # noqa: N802
        self, node: ComputeNode, context: CodegenContext, code_buffer: CodeBuffer, indent: int
    ):
        space_indent = " " * indent
        kwargs = {}
        for idx, input in enumerate(node.inputs):
            kwargs[f"i{idx}"] = context.get_internal_variable_name(input.name)
        for idx, output in enumerate(node.outputs):
            kwargs[f"o{idx}"] = context.get_internal_variable_name(output.name)

        op_type = node.op_type
        if op_type == "Pow":
            if kwargs["i1"] == 2:
                op_type = "Pow2"
            elif kwargs["i1"] == 3:
                op_type = "Pow3"
            elif kwargs["i1"] == 0.5:
                op_type = "Sqrt"

        if op_type == "Cast":
            from_dtype = node.inputs[0].dtype.type
            to_dtype = node.outputs[0].dtype.type
            if from_dtype == to_dtype:
                op_type = "Identity"
            elif to_dtype == np.bool_:
                op_type = "CastBool"
            else:
                kwargs["dtype"] = to_dtype.__name__

        if op_type == "Sum":
            output_var = kwargs["o0"]
            formula = " + ".join([kwargs[f"i{idx}"] for idx in range(len(node.inputs))])
            code_buffer += f"{space_indent}{output_var} = {formula}\n"
            return

        code_buffer += TritonCodegen._COMPUTE_CODE_TEMPLATES[op_type].format(indent=space_indent, **kwargs)

    def ReduceNode(self, node: ReduceNode, context: CodegenContext, code_buffer: CodeBuffer, indent: int):  # noqa: N802
        space_indent = " " * indent
        input_var_name = context.get_internal_variable_name(node.inputs[0].name)
        output_var_name = context.get_internal_variable_name(node.outputs[0].name)
        masks = []
        if node.offset_calc.requires_x_mask:
            masks.append("xmask")
        if node.offset_calc.requires_r_mask:
            masks.append("rmask")
        if len(masks) > 0:
            masks_str = " & ".join(masks)
            code_buffer += (
                f"{space_indent}{input_var_name} = tl.where({masks_str}, {input_var_name}, {node.default_value})\n"
            )
        code_buffer += f"{space_indent}{output_var_name} = {node.triton_func}({input_var_name}, axis=1)[:, None]\n"

    def ReduceForLoopStart(  # noqa: N802
        self, node: ReduceForLoopStart, context: CodegenContext, code_buffer: CodeBuffer, indent: int
    ):
        space_indent = " " * indent
        offset_calc = node.offset_calc
        for reduce_node in node.reduce_nodes:
            tmp_var_name = "tmp_" + context.get_internal_variable_name(reduce_node.outputs[0].name)
            code_buffer += (
                f"{space_indent}{tmp_var_name} = "
                f"tl.zeros([XBLOCK, RBLOCK], tl.float32) + {reduce_node.default_value}\n"
            )
        code_buffer += (
            f"{space_indent}for roffset in range(0, rnumel, RBLOCK):\n{space_indent}    rindex = rbase + roffset\n"
        )
        if offset_calc.requires_r_mask:
            code_buffer += f"{space_indent}    rmask = rindex < rnumel\n"
        for idx in range(offset_calc.r_rank):
            if idx in offset_calc.r_compute_dims:
                div_str = (
                    f" // {may_add_brackets(str(offset_calc.r_strides[idx]))}" if idx != offset_calc.r_rank - 1 else ""
                )
                mod_str = f" % {may_add_brackets(str(offset_calc.r_dims[idx]))}" if idx != 0 else ""
                code_buffer += f"{space_indent}    r{idx} = rindex{div_str}{mod_str}\n"

    def ReduceForLoopEnd(  # noqa: N802
        self, node: ReduceForLoopEnd, context: CodegenContext, code_buffer: CodeBuffer, indent: int
    ):
        space_indent = " " * indent
        offset_calc = node.offset_calc
        masks = []
        if offset_calc.requires_x_mask:
            masks.append("xmask")
        if offset_calc.requires_r_mask:
            masks.append("rmask")
        masks_str = " & ".join(masks)
        for reduce_node in node.reduce_nodes:
            input_var_name = context.get_internal_variable_name(reduce_node.inputs[0].name)
            output_var_name = context.get_internal_variable_name(reduce_node.outputs[0].name)
            tmp_output_var_name = "tmp_" + output_var_name
            if reduce_node.op_type == "ReduceSum":
                if not masks_str:
                    code_buffer += f"{space_indent}{tmp_output_var_name} = {tmp_output_var_name} + {input_var_name}\n"
                else:
                    code_buffer += (
                        f"{space_indent}{tmp_output_var_name} = "
                        f"tl.where({masks_str}, {tmp_output_var_name} + {input_var_name}, {tmp_output_var_name})\n"
                    )
            else:
                op_str = " < " if reduce_node.op_type == "ReduceMax" else " > "
                if masks_str:
                    masks_str += " & "
                code_buffer += (
                    f"{space_indent}{tmp_output_var_name} = tl.where("
                    f"{masks_str}({tmp_output_var_name}{op_str}{input_var_name}), "
                    f"{input_var_name}, {tmp_output_var_name})\n"
                )
        space_indent_outside = " " * (indent - 4)
        for reduce_node in node.reduce_nodes:
            output_var_name = context.get_internal_variable_name(reduce_node.outputs[0].name)
            input_var_name = "tmp_" + output_var_name
            code_buffer += (
                f"{space_indent_outside}{output_var_name} = "
                f"{reduce_node.triton_func}({input_var_name}, axis=1)[:, None]\n"
            )

    def DropoutNode(  # noqa: N802
        self, node: DropoutNode, context: CodegenContext, code_buffer: CodeBuffer, indent: int
    ):
        space_indent = " " * indent
        input_var_name = context.get_internal_variable_name(node.inputs[0].name)
        p_var_name = context.get_internal_variable_name(node.inputs[1].name)
        output_var_name = context.get_internal_variable_name(node.outputs[0].name)
        mask_var_name = (
            context.get_internal_variable_name(node.outputs[1].name)
            if len(node.outputs) >= 2
            else "dropout_mask_output"
        )
        offset_str = f"{node.global_offset} + " if node.global_offset != sympy.Integer(0) else ""
        offset_str += self._get_offset_mask(node.offset_calc, node.inputs[0].name)[0]
        code_buffer += (
            f"{space_indent}p = 1 - {p_var_name}\n"
            f"{space_indent}random = tl.rand(t_seed_cuda, {offset_str})\n"
            f"{space_indent}{mask_var_name} = random < p\n"
            f"{space_indent}{output_var_name} = tl.where({mask_var_name}, {input_var_name} / p, 0.0)\n"
        )

    def ModuleNode(self, node: ModuleNode, context: CodegenContext, code_buffer: CodeBuffer, indent: int):  # noqa: N802
        space_indent = " " * indent
        code_buffer += (
            f"{space_indent}import triton\n"
            f"{space_indent}import triton.language as tl\n"
            f"{space_indent}import torch\n"
        )

        for kernel_node in node.kernels:
            code_buffer += "\n\n"
            kernel_node.codegen(self, CodegenContext(kernel_node.var_map), code_buffer, indent)

        input_args = ", ".join([context.get_variable_name(input.name) for input in node.inputs])
        code_buffer += f"\n\n{space_indent}def {node.func_name}({input_args}):\n"

        indent += 4
        space_indent = " " * indent

        if node.has_dropout:
            code_buffer += (
                f'{space_indent}seed_cuda = torch.randint(2**31, size=(), dtype=torch.int64, device="cuda")\n\n'
            )

        for idx, kernel_node in enumerate(node.kernels):
            if idx != 0:
                code_buffer += "\n"
            # Allocate output tensor.
            for output in kernel_node.outputs:
                torch_dtype = torch.from_numpy(np.zeros(1, dtype=output.dtype)).dtype
                # Workaround for DLPack which doesn't support bool.
                if torch_dtype == torch.bool:
                    torch_dtype = torch.uint8
                code_buffer += (
                    f"{space_indent}{context.get_variable_name(output.name)} = "
                    f'torch.empty({tuple(output.shape)}, dtype={torch_dtype}, device="cuda")\n'
                )
            kernel_args_str = ", ".join([context.get_variable_name(input.name) for input in kernel_node.inputs])
            if kernel_args_str:
                kernel_args_str += ", "
            kernel_args_str += ", ".join([context.get_variable_name(output.name) for output in kernel_node.outputs])
            # TODO: support other kinds of variable args, such as symbolic shape variable.
            if kernel_node.has_dropout:
                kernel_args_str += ", seed_cuda"

            if isinstance(kernel_node, ReduceKernelNode):
                code_buffer += (
                    f"{space_indent}x_numel = {kernel_node.offset_calc.x_numel}\n"
                    f"{space_indent}r_numel = {kernel_node.offset_calc.r_numel}\n"
                    f'{space_indent}grid = lambda meta: (triton.cdiv(x_numel, meta["XBLOCK"]),)\n'
                    f"{space_indent}{kernel_node.name}[grid]({kernel_args_str}, x_numel, r_numel)\n"
                )
            else:
                code_buffer += (
                    f"{space_indent}n_elements = {kernel_node.offset_calc.x_numel}\n"
                    f'{space_indent}grid = lambda meta: (triton.cdiv(n_elements, meta["XBLOCK"]),)\n'
                    f"{space_indent}{kernel_node.name}[grid]({kernel_args_str}, n_elements)\n"
                )

            for name in node.cross_kernel_args_to_delete[idx]:
                code_buffer += f"{space_indent}del {name}\n"

        return_output_str = ", ".join([context.get_variable_name(output.name) for output in node.outputs])
        code_buffer += f"\n{space_indent}return {return_output_str}\n"


def codegen(func_name: str, sorted_graph: SortedGraph) -> str:
    module_node = lower(func_name, sorted_graph)
    code_buffer = CodeBuffer()
    module_node.codegen(TritonCodegen(), CodegenContext(module_node.var_map), code_buffer)
    return str(code_buffer)
