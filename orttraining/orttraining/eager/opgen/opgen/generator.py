# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# pylint: disable=missing-docstring, too-many-public-methods

import json
import sys
from typing import Dict, List, Optional, Union

from opgen import ast, lexer, parser
from opgen import writer as opgenwriter


class Outputs:
    def __init__(self, count: int):
        self.count = count
        self.name = None

    def __str__(self):
        return self.name if self.name else f"<unbound output>"


class AttrType:
    FLOAT = "at::ScalarType::Float"
    FLOATS = "<unsupported:FLOATS>"
    INT = "at::ScalarType::Int"
    INTS = "<unsupported:INTS>"
    STRING = "const char*"
    STRINGS = "<unsupported:STRINGS>"
    TENSOR = "at::Tensor"
    LONG = "at::ScalarType::Long"


class ONNXAttr:
    def __init__(self, value, type: AttrType = None):
        self.value = value
        self.type = type


class ONNXOpEvalContext:
    ops: List["ONNXOp"]

    def __init__(self):
        self.ops = []

    def prepare_outputs(self):
        for i, op in enumerate(self.ops):
            op.outputs.name = f"ort_outputs_{i}_{op.name}"


class ONNXOp:
    def __init__(
        self,
        name: str,
        outputs: int,
        input_types: List,
        *inputs: Union[str, Outputs],
        **attributes: Optional[Union[str, Outputs]],
    ):
        self.name = name
        self.outputs = Outputs(outputs)
        self.inputs = inputs
        self.attributes = attributes
        self.domain = None
        self.input_types = input_types

    def eval(self, ctx: ONNXOpEvalContext):
        evaluated_inputs = []

        for i in self.inputs:
            if isinstance(i, ONNXOp):
                i = i.eval(ctx)
            evaluated_inputs.append(i)

        self.inputs = evaluated_inputs

        ctx.ops.append(self)

        return self.outputs


class SignatureOnly(ONNXOp):
    def __init__(self):
        super().__init__(None, 0, [])


class MakeTorchFallback(ONNXOp):
    def __init__(self):
        super().__init__(None, 0, [])


class FunctionGenerationError(NotImplementedError):
    def __init__(self, cpp_func: ast.FunctionDecl, message: str):
        super().__init__(f"{message} ({cpp_func.identifier})")


class MappedOpFunction:
    def __init__(
        self,
        op_namespace: str,
        mapped_op_name: str,
        onnx_op: ONNXOp,
        cpp_func: ast.FunctionDecl,
        signature_only: bool,
        make_torch_fallback: bool,
    ):
        self.op_namespace = op_namespace
        self.mapped_op_name = mapped_op_name
        self.onnx_op = onnx_op
        self.cpp_func = cpp_func
        self.signature_only = signature_only
        self.make_torch_fallback = make_torch_fallback


class ORTGen:
    _mapped_ops: Dict[str, ONNXOp]
    _custom_ops: bool

    def __init__(
        self,
        ops: Optional[Dict[str, ONNXOp]] = None,
        custom_ops: bool = False,
        type_promotion_ops: List = (),
        aten_output_type: Dict = (),
    ):
        self._mapped_ops = {}
        if ops:
            self.register_many(ops)
        self._custom_ops = custom_ops
        self.type_promotion_ops = type_promotion_ops
        self.aten_output_type = aten_output_type

    def register(self, aten_name: str, onnx_op: ONNXOp):
        self._mapped_ops[aten_name] = onnx_op

    def register_many(self, ops: Dict[str, ONNXOp]):
        for k, v in ops.items():
            self.register(k, v)

    def run(self, cpp_parser: parser.CPPParser, writer: opgenwriter.SourceWriter):
        self._write_file_prelude(writer)

        generated_funcs = []
        current_ns = None

        for mapped_func in self._parse_mapped_function_decls(cpp_parser):
            del self._mapped_ops[mapped_func.mapped_op_name]
            generated_funcs.append(mapped_func)

            ns = mapped_func.op_namespace
            if current_ns and current_ns != ns:
                current_ns = None
                writer.pop_namespace()
            if ns != current_ns:
                current_ns = ns
                writer.writeline()
                writer.push_namespace(ns)

            writer.writeline()

            self._write_function_signature(writer, mapped_func.cpp_func)
            if mapped_func.signature_only:
                writer.writeline(";")
            else:
                writer.writeline(" {")
                writer.push_indent()
                self._write_function_body(writer, mapped_func)
                writer.pop_indent()
                writer.writeline("}")

        if current_ns:
            current_ns = None
            writer.pop_namespace()

        if not self._custom_ops:
            self._write_function_registrations(writer, generated_funcs)
        else:
            self._write_custom_ops_registrations(writer, generated_funcs)
        self._write_file_postlude(writer)

        if len(self._mapped_ops) > 0:
            raise Exception(
                "Torch operation(s) could not be parsed for mapping: "
                + ", ".join([f"'{o}'" for o in self._mapped_ops.keys()])
            )

    def _write_file_prelude(self, writer: opgenwriter.SourceWriter):
        writer.writeline("// AUTO-GENERATED CODE! - DO NOT EDIT!")
        writer.writeline(f'// $ python {" ".join(sys.argv)}')
        writer.writeline()
        writer.writeline('#include "python/onnxruntime_pybind_state_common.h"')
        writer.writeline()
        writer.writeline("#include <torch/extension.h>")
        writer.writeline("#include <ATen/native/CPUFallback.h>")
        writer.writeline()
        writer.writeline("#include <core/providers/dml/OperatorAuthorHelper/Attributes.h>")
        writer.writeline()
        writer.writeline('#include "ort_tensor.h"')
        writer.writeline('#include "ort_aten.h"')
        writer.writeline('#include "ort_log.h"')
        writer.writeline()
        writer.push_namespace("torch_ort")
        writer.push_namespace("eager")
        writer.writeline()
        writer.writeline("using namespace at;")
        writer.writeline("using NodeAttributes = onnxruntime::NodeAttributes;")

    def _write_file_postlude(self, writer: opgenwriter.SourceWriter):
        writer.pop_namespaces()

    def _write_function_signature(self, writer: opgenwriter.SourceWriter, cpp_func: ast.FunctionDecl):
        if cpp_func.torch_func:
            writer.writeline(f"// {cpp_func.torch_func.torch_schema}")
        cpp_func.return_type.write(writer)
        writer.write(f" {cpp_func.identifier.value}(")
        writer.push_indent()
        for param_list_member in cpp_func.parameters:
            writer.writeline()
            if isinstance(param_list_member.member.parameter_type, ast.KWArgsSentinelType):
                writer.write("// ")
            param_list_member.write(writer)
        writer.pop_indent()
        writer.write(")")

    def _write_cpu_fall_back(self, writer: opgenwriter.SourceWriter, mapped_func: MappedOpFunction):
        onnx_op, cpp_func = mapped_func.onnx_op, mapped_func.cpp_func
        # return at::native::call_fallback_fn<
        #  &at::native::cpu_fallback,
        #  ATEN_OP(eq_Tensor)>::call(self, other);
        writer.writeline("return native::call_fallback_fn<")
        writer.push_indent()
        writer.writeline("&native::cpu_fallback,")
        writer.write("ATEN_OP(")
        writer.write(cpp_func.identifier.value)
        writer.write(")>::call(")

        params = ", ".join([p.member.identifier.value for p in cpp_func.parameters if p.member.identifier])
        writer.write(params)
        writer.writeline(");")
        writer.pop_indent()

    # Generates a log line for method entry, with the function parameters.
    def _write_function_body_entry_logging(self, writer, func_parameters):
        log_params = ", ".join([p.member.identifier.value for p in func_parameters if p.member.identifier])
        writer.writeline(f"ORT_LOG_FN({log_params});")
        writer.writeline()

    # Generates code to resize a passed in output tensor to self.size().
    # TODO: allow resizing to other sizes.
    def _write_function_body_resize_output(self, writer):
        writer.writeline("// resize the output and then create output ort value to be updated.")
        writer.writeline(
            "resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), self.sizes());"
        )
        writer.writeline("auto ort_input_out = create_ort_value(invoker, out);")
        writer.writeline()

    # Generates code to do type promotion via casting for a single input for an onnx op.
    def _write_function_body_onnx_op_input_type_promotion(self, writer, cpp_param, onnx_op_index, op_input):
        type_func_str = (
            "type()" if cpp_param.parameter_type.desugar().identifier_tokens[0].value == "Scalar" else "scalar_type()"
        )
        writer.write(f"if ({op_input}.{type_func_str} != *promoted_type)")
        writer.writeline("{")
        writer.push_indent()
        writer.writeline(
            f"ort_input_{onnx_op_index}_{op_input} = CastToType(invoker, ort_input_{onnx_op_index}_{op_input}, *promoted_type);"
        )
        writer.pop_indent()
        writer.writeline("}")

    # Generates code to declare and populate node attributes for one onnx operator.
    def _write_function_body_onnx_op_node_attributes(self, writer, onnx_op, attrs, attrs_arg):
        writer.writeline()
        writer.writeline(f"NodeAttributes {attrs_arg}({len(attrs)});")

        for attr_name, attr in attrs.items():
            writer.write(f'{attrs_arg}["{attr_name}"] = ')
            writer.writeline("create_ort_attribute(")
            writer.push_indent()
            writer.write(f'"{attr_name}", {attr.value}')
            if attr.type.startswith("at::ScalarType::"):
                writer.write(f", {attr.type}")
            elif attr.type == AttrType.TENSOR:
                writer.write(f", true")
            elif attr.type != AttrType.STRING:
                raise FunctionGenerationError(
                    cpp_func,
                    f'Unsure how how to map ONNX op "{onnx_op.name}" attribute '
                    + f'"{attr_name}" of type "{attr.type}" to a call to '
                    + "create_ort_attribute. Please teach generator.py.",
                )
            writer.writeline(");")
            writer.pop_indent()

    # Generates code which assigns an inplace (ort wrapped) input parameter to this onnx op's corresponding output
    def _write_function_body_assign_onnx_op_outputs_to_inplace_params(
        self, writer, in_place_params, cpp_func, return_info, onnx_op, onnx_op_index
    ):
        for input_index, op_input in enumerate(onnx_op.inputs):
            if isinstance(op_input, Outputs):
                continue

            # See if this input is aliased as an in-place tensor
            cpp_param = cpp_func.get_parameter(op_input)
            if not cpp_param:
                continue

            for torch_p in cpp_param.torch_param:
                if isinstance(return_info, ast.TupleType):
                    for output_index, output_param in enumerate(return_info.elements):
                        assert isinstance(output_param.member, ast.TupleMemberType)
                        if self._is_inplace(output_param.member.element_type, torch_p):
                            writer.writeline(
                                f"{onnx_op.outputs}[{output_index}] = ort_input_{onnx_op_index}_{onnx_op.inputs[input_index]};"
                            )
                            in_place_params[output_index] = cpp_param.identifier.value
                            break
                elif isinstance(return_info, ast.ArrayType):
                    if self._is_inplace(return_info, torch_p):
                        writer.writeline(f"for (int i = 0; i < {onnx_op.outputs.count}; i++) {{")
                        writer.push_indent()
                        writer.writeline(
                            f"{onnx_op.outputs}[i] = ort_input_{onnx_op_index}_{onnx_op.inputs[input_index]}[i];"
                        )
                        writer.pop_indent()
                        writer.writeline("}")
                        in_place_params[0] = cpp_param.identifier.value
                        break
                elif self._is_inplace(return_info, torch_p):
                    writer.writeline(f"{onnx_op.outputs}[0] = ort_input_{onnx_op_index}_{onnx_op.inputs[input_index]};")
                    in_place_params[0] = cpp_param.identifier.value
                    break

    # Generates Onnx 'Invoke' call for one onnx op, with parameters, node attributes and output.
    def _write_function_body_onnx_op_invocation(self, writer, onnx_op, onnx_op_index, cpp_func, attrs_arg_ptr):
        # Perform the invocation
        writer.writeline()
        if onnx_op_index == 0:
            writer.write("auto ")
        writer.writeline(f'status = invoker.Invoke("{onnx_op.name}", {{')
        writer.push_indent()
        for op_input in onnx_op.inputs:
            if isinstance(op_input, Outputs):
                if op_input.count != 1:
                    raise FunctionGenerationError(cpp_func, "multiple outputs not supported")
                op_input = f"{op_input}[0]"
            else:
                op_input = f"ort_input_{onnx_op_index}_{op_input}"
            writer.writeline(f"std::move({op_input}),")
        writer.pop_indent()
        writer.write(f"}}, {onnx_op.outputs}, {attrs_arg_ptr}")
        if onnx_op.domain:
            writer.write(f", {onnx_op.domain}")
        writer.writeline(");")
        writer.writeline("CHECK_STATUS(status);")
        writer.writeline()

        return onnx_op.outputs

    # Generates code to assign the aten "out" parameter (now in ort_input_out) to
    # the Onnx Operator output so that it will populate when the onnx op is invoked.
    # In the case that type casting is needed, _write_function_body_return_no_inplace
    # will cast the operator output and assign the out param after invocation.
    def _write_function_body_assign_onnx_op_output_to_out_param(
        self,
        writer,
        num_in_place_params,
        set_out_tensor,
        is_last_onnx_op,
        need_type_promotion_without_cast,
        onnx_op,
        return_info,
    ):
        # if no in_place_params found and there is an out input to set
        # and this is the last onnx op, we set the out to be written to
        if num_in_place_params == 0 and set_out_tensor and is_last_onnx_op:
            if need_type_promotion_without_cast:
                writer.writeline("if (*promoted_type == out.scalar_type()) {")
                writer.push_indent()
                writer.writeline(f"{onnx_op.outputs}[0] = ort_input_out;")
                writer.pop_indent()
                writer.writeline("}")
            else:
                writer.writeline(f"{onnx_op.outputs}[0] = ort_input_out;")

        if num_in_place_params != 0 and num_in_place_params != (
            len(return_info.elements) if isinstance(return_info, ast.TupleType) else 1
        ):
            raise Exception("Cannot mix and match inplace with non-inplace parameters.")

    # Generates a non-ref return (the value returned is not one of the function's parameters).
    def _write_function_body_return_no_inplace(
        self,
        writer,
        need_type_promotion,
        first_param,
        mapped_func,
        cpp_func,
        return_outputs,
    ):
        # TODO: revisit the hardcoded use of TensorList.
        writer.write(f"at::TensorOptions tensor_options = {first_param.identifier.value}")
        if first_param.parameter_type.desugar().identifier_tokens[0].value == "TensorList":
            writer.write("[0]")
        writer.write(".options()")
        if need_type_promotion:
            writer.write(".dtype(*promoted_type)")

        # do we need to set type on the returned value
        if mapped_func.mapped_op_name in self.aten_output_type:
            writer.write(f".dtype({self.aten_output_type[mapped_func.mapped_op_name]})")

        writer.writeline(";")

        writer.writeline("return aten_tensor_from_ort(")
        writer.push_indent()
        if (
            isinstance(cpp_func.return_type, ast.TemplateType)
            and cpp_func.return_type.identifier_tokens[-1].value == "std::vector"
        ):
            writer.writeline(f"{return_outputs},")
            writer.writeline("tensor_options);")
        else:
            writer.writeline(f"std::move({return_outputs}[0]),")
            writer.writeline("tensor_options);")
        writer.pop_indent()

    # Generates the return statement when a tuple is returned.
    def _write_function_body_return_multiple(self, writer, cpp_func, in_place_params):
        if not (
            isinstance(cpp_func.return_type, ast.TemplateType)
            and cpp_func.return_type.identifier_tokens[-1].value == "std::tuple"
        ):
            raise Exception(f"")
        tensorRef = "Tensor&," * len(in_place_params)
        tensorRef = tensorRef[: len(tensorRef) - 1]
        writer.write(f"return std::tuple<{tensorRef}>(")
        for index, key in enumerate(sorted(in_place_params)):
            if index > 0:
                writer.write(", ")
            writer.write(in_place_params[key])
        writer.writeline(");")

    # determines if the "out" param exists and needs to be set.
    # "out", while similar to other modifiable (e.g. 'in_place') params, is treated
    # differently because the intent is to put result data into it, not modify data
    def _should_set_out_tensor(self, first_param, last_param, return_info):
        # if the torch func has a return ref tensor, out is the last param, and self param is the first input
        # then we need to update and return out. Record this need in set_out_tensor.
        # TODO: make this more general to handle cases where the first param is not self such as
        # - cat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
        # - complex.out(Tensor real, Tensor imag, *, Tensor(a!) out) -> Tensor(a!)

        if (
            return_info
            and last_param
            and last_param.identifier.value == "out"
            and first_param.identifier.value == "self"
        ):
            # output_alias is how the return tensor is marked, normally (a! -> a)
            output_alias = self._get_alias_info(return_info)

            for torch_p in last_param.torch_param:
                # Confirm we have an output alias and that it is writable (!)
                # and the current param (torch_p/last_param) is marked with the output alias
                if output_alias and output_alias.is_writable and self._get_alias_info(torch_p) == output_alias:
                    return True
        return False

    # Evals the outer ONNX op to produce a topologically ordered list of ops (in case of nested ops).
    def _get_onnx_ops_eval_context(self, op):
        ctx = ONNXOpEvalContext()
        op.eval(ctx)
        ctx.prepare_outputs()
        return ctx

    # Generates an assert to verify the first param is a tensor of size > 0.
    def _write_function_body_first_param_assert(self, writer, first_param):
        # Check if the first parameter is tensorlist and if yes it's size should be > 0
        if first_param.parameter_type.desugar().identifier_tokens[0].value == "TensorList":
            writer.write("assert(")
            writer.write(first_param.identifier.value)
            writer.writeline(".size()>0);")
        if (
            not isinstance(first_param.parameter_type.desugar(), ast.ConcreteType)
            or "Tensor" not in first_param.parameter_type.desugar().identifier_tokens[0].value
        ):
            raise FunctionGenerationError(cpp_func, "First parameter must be an at::Tensor")

    # Generates code to get an ORT Invoker for the device from the first param.
    # The Invoker will be used and reused in _write_function_body_onnx_op_invocation.
    def _write_function_body_get_invoker(self, writer, first_param):
        writer.write("auto& invoker = GetORTInvoker(")
        writer.write(first_param.identifier.value)
        if first_param.parameter_type.desugar().identifier_tokens[0].value == "TensorList":
            writer.write("[0]")
        writer.writeline(".device());")
        writer.writeline()

    # Generates code to declare a vector which receives the full output for one onnx op.
    def _write_function_body_onnx_op_output_vector(self, writer, onnx_op):
        writer.writeline()
        writer.write(f"std::vector<OrtValue> {onnx_op.outputs}")
        writer.writeline(f"({onnx_op.outputs.count});")

    # Generates code to create ORT values from each torch function parameter that needs to be passed to this onnx op.
    def _write_function_body_onnx_op_inputs(self, writer, onnx_op, onnx_op_index, need_type_promotion, cpp_func):
        for op_input in onnx_op.inputs:
            if isinstance(op_input, Outputs):
                continue
            cpp_param = cpp_func.get_parameter(op_input)
            writer.write(f"auto ort_input_{onnx_op_index}_{op_input} = ")
            writer.writeline(f"create_ort_value(invoker, {op_input});")
            if need_type_promotion:
                self._write_function_body_onnx_op_input_type_promotion(writer, cpp_param, onnx_op_index, op_input)

    # Ends the function by writing a return statement (or not for void).
    def _write_function_body_return(
        self,
        writer,
        cpp_func,
        in_place_params,
        set_out_tensor,
        need_type_promotion,
        impl_uses_cast,
        onnx_op_outputs,
        last_param,
        first_param,
        mapped_func,
        return_outputs,
    ):
        if cpp_func.return_type.desugar().identifier_tokens[0].value == "void":
            pass
        elif len(in_place_params) == 0:
            if set_out_tensor:
                if need_type_promotion and not impl_uses_cast:
                    writer.writeline("if (*promoted_type != out.scalar_type()) {")
                    writer.push_indent()
                    writer.writeline(
                        f"CastToType_out(invoker, {onnx_op_outputs}[0], ort_input_out, out.scalar_type());"
                    )
                    writer.pop_indent()
                    writer.writeline("}")

                writer.writeline(f"return {last_param.identifier.value};")
            else:
                self._write_function_body_return_no_inplace(
                    writer, need_type_promotion, first_param, mapped_func, cpp_func, return_outputs
                )
        elif len(in_place_params) == 1:
            writer.writeline(f"return {in_place_params[0]};")
        else:
            self._write_function_body_return_multiple(writer, cpp_func, in_place_params)

    # Generates all of the code for a single onnx op call, including mapping inputs, outputs and attributes.
    def _write_function_body_onnx_op(
        self,
        writer,
        onnx_op,
        onnx_op_index,
        need_type_promotion,
        cpp_func,
        return_info,
        set_out_tensor,
        ctx,
        impl_uses_cast,
    ):
        self._write_function_body_onnx_op_inputs(writer, onnx_op, onnx_op_index, need_type_promotion, cpp_func)

        # Torch kwargs -> ORT attributes
        attrs = {k: v for k, v in onnx_op.attributes.items() if v and v.value is not None}
        attrs_arg_ptr = "nullptr"
        if len(attrs) > 0:
            attrs_arg = f"attrs_{onnx_op_index}"
            attrs_arg_ptr = f"&{attrs_arg}"
            self._write_function_body_onnx_op_node_attributes(writer, onnx_op, attrs, attrs_arg)

        self._write_function_body_onnx_op_output_vector(writer, onnx_op)

        in_place_params = {}

        if return_info:
            self._write_function_body_assign_onnx_op_outputs_to_inplace_params(
                writer, in_place_params, cpp_func, return_info, onnx_op, onnx_op_index
            )
            self._write_function_body_assign_onnx_op_output_to_out_param(
                writer,
                len(in_place_params),
                set_out_tensor,
                onnx_op_index == (len(ctx.ops) - 1),
                need_type_promotion and not impl_uses_cast,
                onnx_op,
                return_info,
            )

        # We'll potentially return back to Torch from this op
        return_outputs = self._write_function_body_onnx_op_invocation(
            writer, onnx_op, onnx_op_index, cpp_func, attrs_arg_ptr
        )

        return in_place_params, return_outputs

    # Generates code for the entire body of the function (everything between { and }.)
    # TODO: Pick the right "out" Torch parameter; do not assume the first one
    # TODO: Handle multiple results
    # TODO: Assert return type
    # TODO: warn if we have not consumed all torch parameters (either as an ORT input or ORT attribute).
    def _write_function_body(self, writer: opgenwriter.SourceWriter, mapped_func: MappedOpFunction):
        full_onnx_op, cpp_func = mapped_func.onnx_op, mapped_func.cpp_func
        assert len(cpp_func.parameters) > 0

        self._write_function_body_entry_logging(writer, cpp_func.parameters)

        if mapped_func.make_torch_fallback:
            return self._write_cpu_fall_back(writer, mapped_func)

        first_param = cpp_func.parameters[0].member  # FIXME: Find first at::Tensor param instead of first param only?
        self._write_function_body_first_param_assert(writer, first_param)

        ctx = self._get_onnx_ops_eval_context(full_onnx_op)
        need_type_promotion = self._write_type_promotion(writer, mapped_func, cpp_func, ctx)
        return_info = cpp_func.torch_func.return_type if cpp_func.torch_func else None
        last_param = cpp_func.parameters[-1].member
        set_out_tensor = self._should_set_out_tensor(first_param, last_param, return_info)
        impl_uses_cast = self._write_type_check(writer, mapped_func, cpp_func, ctx, need_type_promotion, set_out_tensor)

        self._write_function_body_get_invoker(writer, first_param)

        if set_out_tensor:
            self._write_function_body_resize_output(writer)

        for onnx_op_index, onnx_op in enumerate(ctx.ops):
            in_place_params, return_outputs = self._write_function_body_onnx_op(
                writer,
                onnx_op,
                onnx_op_index,
                need_type_promotion,
                cpp_func,
                return_info,
                set_out_tensor,
                ctx,
                impl_uses_cast,
            )

        self._write_function_body_return(
            writer,
            cpp_func,
            in_place_params,
            set_out_tensor,
            need_type_promotion,
            impl_uses_cast,
            full_onnx_op.outputs,
            last_param,
            first_param,
            mapped_func,
            return_outputs,
        )

    def _write_type_check(self, writer, mapped_func, cpp_func, ctx, need_type_promotion, set_out_tensor):
        impl_uses_cast = False
        need_type_check = False
        if not self._custom_ops:
            for onnx_op in ctx.ops:
                for op_input in onnx_op.inputs:
                    if not isinstance(op_input, Outputs):
                        need_type_check = True
                        break
        if need_type_check:
            writer.write("if (")
            i = 0
            for onnx_op in ctx.ops:
                # track is the CAST op was explicitly used
                if onnx_op.name == "Cast":
                    impl_uses_cast = True
                for idx, op_input in enumerate(onnx_op.inputs):
                    if isinstance(op_input, Outputs):
                        continue
                    writer.writeline(" || " if i > 0 else "")
                    if i == 0:
                        writer.push_indent()
                    cpp_param = cpp_func.get_parameter(op_input)
                    supported_types = ",".join(sorted(list(onnx_op.input_types[idx])))
                    writer.write(f"!IsSupportedType({cpp_param.identifier.value}, {{{supported_types}}})")
                    i += 1
            # if we have type promotion and need to set the out tensor and CAST op not explictily listed,
            # then we confirm the promotion type is castable to the out type.
            if need_type_promotion and set_out_tensor and not impl_uses_cast:
                writer.writeline(" || ")
                writer.write("!c10::canCast(*promoted_type, out.scalar_type())")
            writer.writeline(") {")
            self._write_cpu_fall_back(writer, mapped_func)
            writer.pop_indent()
            writer.writeline("}")
        return impl_uses_cast

    def _write_type_promotion(self, writer, mapped_func, cpp_func, ctx):
        need_type_promotion = False
        if mapped_func.mapped_op_name in self.type_promotion_ops:
            types_from_tensor = []
            types_from_scalar = []
            for onnx_op in ctx.ops:
                for op_input in onnx_op.inputs:
                    if isinstance(op_input, Outputs):
                        continue
                    cpp_param = cpp_func.get_parameter(op_input)
                    if cpp_param:
                        if cpp_param.parameter_type.desugar().identifier_tokens[0].value == "Tensor":
                            types_from_tensor.append(f"{op_input}.scalar_type()")
                        elif cpp_param.parameter_type.desugar().identifier_tokens[0].value == "Scalar":
                            types_from_scalar.append(f"{op_input}.type()")
            if len(types_from_tensor) > 0 or len(types_from_scalar) > 0:
                need_type_promotion = True
                writer.writeline(
                    "auto promoted_type = PromoteScalarTypesWithCategory({%s}, {%s});"
                    % (",".join(types_from_tensor), ",".join(types_from_scalar))
                )
                writer.writeline()
        return need_type_promotion

    def _write_function_registrations(self, writer: opgenwriter.SourceWriter, generated_funcs: List[MappedOpFunction]):
        writer.writeline()
        writer.writeline("TORCH_LIBRARY_IMPL(aten, ORT, m) {")
        writer.push_indent()

        for mapped_func in generated_funcs:
            cpp_func, torch_func = mapped_func.cpp_func, mapped_func.cpp_func.torch_func

            if mapped_func.op_namespace:
                reg_function_arg = f"{mapped_func.op_namespace}::"
            else:
                reg_function_arg = ""
            reg_function_arg += cpp_func.identifier.value

            writer.write("m.impl(")
            reg_function_arg = f"TORCH_FN({reg_function_arg})"

            writer.writeline(f'"{torch_func.identifier.value}", {reg_function_arg});')

        writer.pop_indent()
        writer.writeline("}")
        writer.writeline()

    def _write_custom_ops_registrations(
        self, writer: opgenwriter.SourceWriter, generated_funcs: List[MappedOpFunction]
    ):
        writer.writeline()
        writer.writeline("TORCH_LIBRARY(ort, m) {")
        writer.push_indent()

        for mapped_func in generated_funcs:
            cpp_func = mapped_func.cpp_func
            writer.write("m.def(")
            writer.writeline(f'"{cpp_func.identifier.value}", &{cpp_func.identifier.value});')

        writer.pop_indent()
        writer.writeline("}")
        writer.writeline()

    def _get_alias_info(self, torch_type_or_param: Union[ast.Type, ast.ParameterDecl]):
        if isinstance(torch_type_or_param, ast.ParameterDecl):
            torch_type = torch_type_or_param.parameter_type
        else:
            torch_type = torch_type_or_param
        return getattr(torch_type.desugar(), "alias_info", None)

    def _parse_mapped_function_decls(self, cpp_parser: parser.CPPParser):
        for cpp_func in self._parse_function_decls(cpp_parser):
            torch_func = cpp_func.torch_func
            if not torch_func:
                op_namespace = None
                op_name = cpp_func.identifier.value
            else:
                op_name = torch_func.identifier.value

                try:
                    op_namespace = op_name[0 : op_name.index("::")]
                    op_namewithoutnamespace = op_name[len(op_namespace) + 2 :]
                except:
                    op_namespace = None
                    op_namewithoutnamespace = op_name

                cpp_func.identifier.value = op_namewithoutnamespace.replace(".", "_")

            onnx_op = self._mapped_ops.get(op_name)
            if not onnx_op:
                continue

            yield MappedOpFunction(
                op_namespace,
                op_name,
                onnx_op,
                cpp_func,
                isinstance(onnx_op, SignatureOnly),
                isinstance(onnx_op, MakeTorchFallback),
            )

    def _parse_function_decls(self, cpp_parser: parser.CPPParser):
        # Parse the C++ declarations
        tu = cpp_parser.parse_translation_unit()

        # Parse the Torch schema from the JSON comment that follows each C++ decl
        # and link associated Torch and C++ decls (functions, parameters, returns)
        for cpp_func in tu:
            hasSchema = False
            if cpp_func.semicolon and cpp_func.semicolon.trailing_trivia:
                for trivia in cpp_func.semicolon.trailing_trivia:
                    if trivia.kind == lexer.TokenKind.SINGLE_LINE_COMMENT:
                        yield self._parse_and_link_torch_function_decl(cpp_func, trivia)
                        hasSchema = True
                        break

            if not hasSchema:
                # customops might not have torch schema
                cpp_func.torch_func = None
                yield cpp_func

    def _parse_and_link_torch_function_decl(self, cpp_func: ast.FunctionDecl, torch_schema_comment_trivia: lexer.Token):
        metadata = json.loads(torch_schema_comment_trivia.value.lstrip("//"))
        schema = metadata["schema"]

        schema_parser = parser.torch_create_from_string(schema)
        schema_parser.set_source_location(cpp_func.semicolon.location)
        torch_func = schema_parser.parse_function()

        torch_func.torch_schema = schema
        torch_func.torch_dispatch = metadata["dispatch"] == "True"
        torch_func.torch_default = metadata["default"] == "True"

        cpp_func.torch_func = torch_func

        if cpp_func.return_type:
            cpp_func.return_type.torch_type = torch_func.return_type

        # Synthesize KWArgsSentinelType in the C++ declaration if we have one
        for i, torch_param in enumerate([p.member for p in torch_func.parameters]):
            if isinstance(torch_param.parameter_type, ast.KWArgsSentinelType):
                cpp_func.parameters.members.insert(
                    i, ast.SyntaxListMember(torch_param, lexer.Token(None, lexer.TokenKind.COMMA, ","))
                )
                break

        # Link Torch parameters to their C++ counterparts, special casing
        # TensorOptions parameters
        for i, cpp_param in enumerate([p.member for p in cpp_func.parameters]):
            if not getattr(cpp_param, "torch_param", None):
                cpp_param.torch_param = []

            torch_param_range = 1
            if isinstance(cpp_param.parameter_type.desugar(), ast.TensorOptionsType):
                torch_param_range = 4

            for j in range(torch_param_range):
                torch_param = torch_func.parameters[i + j].member
                cpp_param.torch_param.append(torch_param)

        return cpp_func

    def _is_inplace(self, element_type, torch_p):
        output_alias = self._get_alias_info(element_type)
        return output_alias and self._get_alias_info(torch_p) == output_alias and output_alias.is_writable
