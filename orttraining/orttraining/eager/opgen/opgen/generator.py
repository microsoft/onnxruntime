# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import Optional, Dict, List, Union

import sys
import json

import opgen.lexer as lexer
import opgen.parser as parser
import opgen.ast as ast
import opgen.writer as writer

class Outputs:
  def __init__(self, count: int):
    self.count = count
    self.name = None

  def __str__(self):
    return self.name if self.name else f'<unbound output>'

class AttrType:
  FLOAT = 'at::ScalarType::Float'
  FLOATS = '<unsupported:FLOATS>'
  INT = 'at::ScalarType::Int'
  INTS = '<unsupported:INTS>'
  STRING = 'const char*'
  STRINGS = '<unsupported:STRINGS>'
  TENSOR = 'at::Tensor'
  LONG = 'at::ScalarType::Long'

class ONNXAttr:
  def __init__(self, value, type: AttrType=None):
    self.value = value
    self.type = type

class ONNXOpEvalContext:
  ops: List['ONNXOp']

  def __init__(self):
    self.ops = []

  def prepare_outputs(self):
    for i, op in enumerate(self.ops):
      op.outputs.name = f'ort_outputs_{i}_{op.name}'

class ONNXOp:
  def __init__(self,
    name: str,
    outputs: int,
    input_types: List,
    *inputs: Union[str, Outputs],
    **attributes: Optional[Union[str, Outputs]]):
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
  def __init__(self): super().__init__(None, 0, [])

class MakeTorchFallback(ONNXOp):
  def __init__(self): super().__init__(None, 0, [])

class FunctionGenerationError(NotImplementedError):
  def __init__(self, cpp_func: ast.FunctionDecl, message: str):
    super().__init__(f'{message} ({cpp_func.identifier})')

class MappedOpFunction:
  def __init__(
    self,
    op_namespace: str,
    mapped_op_name: str,
    onnx_op: ONNXOp,
    cpp_func: ast.FunctionDecl,
    signature_only: bool,
    make_torch_fallback: bool):
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
    custom_ops : bool = False,
    type_promotion_ops : List = ()):
    self._mapped_ops = {}    
    if ops:
      self.register_many(ops)
    self._custom_ops = custom_ops  
    self.type_promotion_ops = type_promotion_ops    

  def register(self, aten_name: str, onnx_op: ONNXOp):
    self._mapped_ops[aten_name] = onnx_op

  def register_many(self, ops: Dict[str, ONNXOp]):
    for k, v in ops.items():
      self.register(k, v)

  def run(self, cpp_parser: parser.CPPParser, writer: writer.SourceWriter):
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
      if mapped_func.cpp_func.torch_func:
        writer.writeline(f'// {mapped_func.cpp_func.torch_func.torch_schema}')

      self._write_function_signature(writer, mapped_func.cpp_func)
      if mapped_func.signature_only:
        writer.writeline(';')
      else:
        writer.writeline(' {')
        writer.push_indent()
        self._write_function_body(writer, mapped_func)
        writer.pop_indent()
        writer.writeline('}')

    if current_ns:
      current_ns = None
      writer.pop_namespace()

    if not self._custom_ops:
      self._write_function_registrations(writer, generated_funcs)
    else:
      self._write_custom_ops_registrations(writer, generated_funcs)
    self._write_file_postlude(writer)

    if len(self._mapped_ops) > 0:
      raise Exception('Torch operation(s) could not be parsed for mapping: ' + \
        ', '.join([f'\'{o}\'' for o in self._mapped_ops.keys()]))

  def _write_file_prelude(self, writer: writer.SourceWriter):
    writer.writeline('// AUTO-GENERATED CODE! - DO NOT EDIT!')
    writer.writeline(f'// $ python {" ".join(sys.argv)}')
    writer.writeline()
    writer.writeline('#include "python/onnxruntime_pybind_state_common.h"')
    writer.writeline()
    writer.writeline('#include <torch/extension.h>')
    writer.writeline('#include <ATen/native/CPUFallback.h>')
    writer.writeline()
    writer.writeline('#include <core/providers/dml/OperatorAuthorHelper/Attributes.h>')
    writer.writeline()
    writer.writeline('#include "ort_tensor.h"')
    writer.writeline('#include "ort_aten.h"')
    writer.writeline('#include "ort_log.h"')
    writer.writeline()
    writer.push_namespace('torch_ort')
    writer.push_namespace('eager')
    writer.writeline()
    writer.writeline('using namespace at;')
    writer.writeline('using NodeAttributes = onnxruntime::NodeAttributes;')

  def _write_file_postlude(self, writer: writer.SourceWriter):
    writer.pop_namespaces()

  def _write_function_signature(
    self,
    writer: writer.SourceWriter,
    cpp_func: ast.FunctionDecl):
    cpp_func.return_type.write(writer)
    writer.write(f' {cpp_func.identifier.value}(')
    writer.push_indent()
    for param_list_member in cpp_func.parameters:
      writer.writeline()
      if isinstance(
        param_list_member.member.parameter_type,
        ast.KWArgsSentinelType):
        writer.write('// ')
      param_list_member.write(writer)
    writer.pop_indent()
    writer.write(')')

  def _write_cpu_fall_back(self, 
                           writer: writer.SourceWriter,
                           mapped_func: MappedOpFunction):
      onnx_op, cpp_func = mapped_func.onnx_op, mapped_func.cpp_func
      #return at::native::call_fallback_fn<
      #  &at::native::cpu_fallback,
      #  ATEN_OP(eq_Tensor)>::call(self, other);
      writer.writeline('return native::call_fallback_fn<')
      writer.push_indent()
      writer.writeline('&native::cpu_fallback,')
      writer.write('ATEN_OP(')
      writer.write(cpp_func.identifier.value)
      writer.write(')>::call(')

      params = ', '.join([p.member.identifier.value for p \
        in cpp_func.parameters if p.member.identifier])
      writer.write(params)
      writer.writeline(');')
      writer.pop_indent()


  def _write_function_body(
    self,
    writer: writer.SourceWriter,
    mapped_func: MappedOpFunction):
    onnx_op, cpp_func = mapped_func.onnx_op, mapped_func.cpp_func

    assert(len(cpp_func.parameters) > 0)

    # Debug Logging
    log_params = ', '.join([p.member.identifier.value for p \
      in cpp_func.parameters if p.member.identifier])
    writer.writeline(f'ORT_LOG_FN({log_params});')
    writer.writeline()

    if mapped_func.make_torch_fallback:
      return self._write_cpu_fall_back(writer, mapped_func)

    return_alias_info = self._get_alias_info(cpp_func.torch_func.return_type) if cpp_func.torch_func else None
    if return_alias_info and not return_alias_info.is_writable:
      return_alias_info = None
    in_place_param: ast.ParameterDecl = None

    # Eval the outer ONNX op to produce a topologically ordered list of ops
    ctx = ONNXOpEvalContext()
    onnx_op.eval(ctx)
    ctx.prepare_outputs()

    # Fetch the ORT invoker from an at::Tensor.device()
    # FIXME: find the first at::Tensor param anywhere in the signature
    # instead of simply the first parameter?
    first_param = cpp_func.parameters[0].member
    # Check if the first parameter is tensorlist and if yes it's size should be > 0
    if first_param.parameter_type.desugar().identifier_tokens[0].value == 'TensorList':
      writer.write('assert(')
      writer.write(first_param.identifier.value)
      writer.writeline('.size()>0);')

    # generate the type check
    need_type_check = False
    if not self._custom_ops:
      for onnx_op_index, onnx_op in enumerate(ctx.ops):
        for op_input in onnx_op.inputs:
          if not isinstance(op_input, Outputs):
            need_type_check = True
            break
    if need_type_check:
      writer.write('if (')
      i = 0
      for onnx_op_index, onnx_op in enumerate(ctx.ops):
        for idx, op_input in enumerate(onnx_op.inputs):
          if isinstance(op_input, Outputs):
            continue
          writer.writeline(' || ' if i > 0 else '')
          if i == 0:
            writer.push_indent()
          cpp_param = cpp_func.get_parameter(op_input)
          supported_types = ','.join([type for type in onnx_op.input_types[idx]])
          writer.write('!IsSupportedType(%s, {%s})' % (cpp_param.identifier.value, supported_types))
          i += 1
      writer.writeline(') {')
      self._write_cpu_fall_back(writer, mapped_func)
      writer.pop_indent()
      writer.writeline('}')      

    if not isinstance(
      first_param.parameter_type.desugar(),
      ast.ConcreteType) or 'Tensor' not in first_param.parameter_type.desugar().identifier_tokens[0].value:
      raise FunctionGenerationError(
        cpp_func,
        'First parameter must be an at::Tensor')

    writer.write('auto& invoker = GetORTInvoker(')
    writer.write(first_param.identifier.value)
    if first_param.parameter_type.desugar().identifier_tokens[0].value == 'TensorList':
      writer.write('[0]')
    writer.writeline('.device());')
    writer.writeline()

    # FIXME: warn if we have not consumed all torch parameters (either as
    # an ORT input or ORT attribute).

    # Perform kernel fission on the ATen op to yield a chain of ORT Invokes
    # e.g. aten::add(x, y, α) -> onnx::Add(x, onnx::Mul(α, y))
    
    # whether need type promotion
    need_type_promotion = False
    if mapped_func.mapped_op_name in self.type_promotion_ops:
      types_from_tensor = []
      types_from_scalar = []
      for onnx_op_index, onnx_op in enumerate(ctx.ops):
        for op_input in onnx_op.inputs:
          if isinstance(op_input, Outputs):
            continue
        cpp_param = cpp_func.get_parameter(op_input)
        if cpp_param:
          if cpp_param.parameter_type.desugar().identifier_tokens[0].value == 'Tensor':
            types_from_tensor.append(f'{op_input}.scalar_type()')
          elif cpp_param.parameter_type.desugar().identifier_tokens[0].value == 'Scalar':
            types_from_scalar.append(f'{op_input}.type()')
      if len(types_from_tensor) > 0 or len(types_from_scalar) > 0 :
        need_type_promotion = True
        writer.writeline('auto promoted_type = PromoteScalarTypesWithCategory({%s}, {%s});'
                         % (','.join(types_from_tensor), ','.join(types_from_scalar)))
        writer.writeline()

    for onnx_op_index, onnx_op in enumerate(ctx.ops):
      # Torch -> ORT inputs
      for op_input in onnx_op.inputs:
        if isinstance(op_input, Outputs):
          continue
        # See if this input is aliased as an in-place tensor
        cpp_param = cpp_func.get_parameter(op_input)
        if return_alias_info and cpp_param:
          for torch_p in cpp_param.torch_param:
            if self._get_alias_info(torch_p) == return_alias_info:
              in_place_param = cpp_param

        writer.write(f'auto ort_input_{op_input} = ')
        writer.writeline(f'create_ort_value(invoker, {op_input});')
        if need_type_promotion:
          type_func_str = 'type()' if cpp_param.parameter_type.desugar().identifier_tokens[0].value == 'Scalar' else 'scalar_type()'
          writer.write(f'if ({op_input}.{type_func_str} != *promoted_type)')
          writer.writeline('{')
          writer.push_indent()
          writer.writeline(f'ort_input_{op_input} = CastToType(invoker, ort_input_{op_input}, *promoted_type);')
          writer.pop_indent()
          writer.writeline('}')

      # Torch kwargs -> ORT attributes
      attrs = { k:v for k, v in onnx_op.attributes.items() if v and v.value }
      if len(attrs) > 0:
        attrs_arg = 'attrs'
        writer.writeline()
        writer.writeline(f'NodeAttributes {attrs_arg}({len(attrs)});')

        for attr_name, attr in attrs.items():
          writer.write(f'{attrs_arg}["{attr_name}"] = ')
          writer.writeline('create_ort_attribute(')
          writer.push_indent()
          writer.write(f'"{attr_name}", {attr.value}')
          if attr.type.startswith('at::ScalarType::'):
            writer.write(f', {attr.type}')
          elif attr.type != AttrType.STRING:
            raise FunctionGenerationError(
              cpp_func,
              f'Unsure how how to map ONNX op "{onnx_op.name}" attribute ' + 
              f'"{attr_name}" of type "{attr.type}" to a call to ' +
              'create_ort_attribute. Please teach generator.py.')
          writer.writeline(');')
          writer.pop_indent()
        attrs_arg = f'&{attrs_arg}'
      else:
        attrs_arg = 'nullptr'

      # Outputs vector
      writer.writeline()
      writer.write(f'std::vector<OrtValue> {onnx_op.outputs}')
      writer.writeline(f'({onnx_op.outputs.count});')

      if in_place_param:
        assert(onnx_op.outputs.count == 1)
        # TODO: This assumes that the first output corresponds to the first input.
        # This may not work for more complicated ops.
        writer.writeline(f'{onnx_op.outputs}[0] = ort_input_{onnx_op.inputs[0]};')

      # Perform the invocation
      writer.writeline()
      if onnx_op_index == 0:
        writer.write('auto ')
      writer.writeline(f'status = invoker.Invoke("{onnx_op.name}", {{')
      writer.push_indent()
      for op_input in onnx_op.inputs:
        if isinstance(op_input, Outputs):
          if op_input.count != 1:
            raise FunctionGenerationError(
              cpp_func,
              'multiple outputs not supported')
          op_input = f'{op_input}[0]'
        else:
          op_input = f'ort_input_{op_input}'
        writer.writeline(f'std::move({op_input}),')
      writer.pop_indent()
      writer.write(f'}}, {onnx_op.outputs}, {attrs_arg}')
      if onnx_op.domain:
        writer.write(f', {onnx_op.domain}')
      writer.writeline(');')
      writer.writeline()

      # Assert invocation
      writer.writeline('if (!status.IsOK())')
      writer.push_indent()
      writer.writeline('throw std::runtime_error(')
      writer.push_indent()
      writer.writeline('"ORT return failure status:" + status.ErrorMessage());')
      writer.pop_indent()
      writer.pop_indent()
      writer.writeline()

      # We'll potentially return back to Torch from this op
      return_outputs = onnx_op.outputs

    # TODO: Pick the right "out" Torch parameter; do not assume the first one
    # TODO: Handle mutliple results
    # TODO: Assert return type

    if not return_alias_info:     
      # tensor options
      writer.write(f'at::TensorOptions tensor_options = {first_param.identifier.value}')
      if first_param.parameter_type.desugar().identifier_tokens[0].value == 'TensorList':
        writer.write('[0]')
      writer.write('.options()')
      if need_type_promotion:
        writer.write('.dtype(*promoted_type)')
      writer.writeline(';')

      writer.writeline('return aten_tensor_from_ort(')
      writer.push_indent()
      if isinstance(cpp_func.return_type, ast.TemplateType) and cpp_func.return_type.identifier_tokens[-1].value == 'std::vector':
        writer.writeline(f'{return_outputs},')
        writer.writeline('tensor_options);')
      else:
        writer.writeline(f'std::move({return_outputs}[0]),')
        writer.writeline('tensor_options);')
      writer.pop_indent()
      return

    if not in_place_param:
      raise Exception(f'"{cpp_func.torch_func.torch_schema}" ' +
        'has alias info on its return type but no associated parameter')

    writer.writeline(f'return {in_place_param.identifier.value};')

  def _write_function_registrations(
    self,
    writer: writer.SourceWriter,
    generated_funcs: List[MappedOpFunction]):
    writer.writeline()
    writer.writeline('TORCH_LIBRARY_IMPL(aten, ORT, m) {')
    writer.push_indent()

    for mapped_func in generated_funcs:
      cpp_func, torch_func = mapped_func.cpp_func, mapped_func.cpp_func.torch_func

      
      if mapped_func.op_namespace:
        reg_function_arg = f'{mapped_func.op_namespace}::'
      else:
        reg_function_arg = ''
      reg_function_arg += cpp_func.identifier.value

      writer.write('m.impl(')
      reg_function_arg = f'TORCH_FN({reg_function_arg})'

      writer.writeline(f'"{torch_func.identifier.value}", {reg_function_arg});')

    writer.pop_indent()
    writer.writeline('}')
    writer.writeline()

  def _write_custom_ops_registrations(
    self,
    writer: writer.SourceWriter,
    generated_funcs: List[MappedOpFunction]):
    writer.writeline()
    writer.writeline('void GenerateCustomOpsBindings(pybind11::module_ m) {')
    writer.push_indent()
    writer.writeline('ORT_LOG_INFO << "GenerateCustomOpsBindings init";')

    for mapped_func in generated_funcs:
      cpp_func = mapped_func.cpp_func
      writer.write('m.def(')
      writer.writeline(f'"{cpp_func.identifier.value}", &{cpp_func.identifier.value});')

    writer.pop_indent()
    writer.writeline('}')
    writer.writeline()

  def _get_alias_info(self, torch_type_or_param: Union[ast.Type, ast.ParameterDecl]):
    if isinstance(torch_type_or_param, ast.ParameterDecl):
      torch_type = torch_type_or_param.parameter_type
    else:
      torch_type = torch_type_or_param
    return getattr(torch_type.desugar(), 'alias_info', None)

  def _parse_mapped_function_decls(self, cpp_parser: parser.CPPParser):
    for cpp_func in self._parse_function_decls(cpp_parser):
      torch_func = cpp_func.torch_func
      if not torch_func:
        op_namespace = None
        op_name = cpp_func.identifier.value
      else:       
        op_name = torch_func.identifier.value

        try:
          op_namespace = op_name[0:op_name.index('::')]
          op_namewithoutnamespace = op_name[len(op_namespace) + 2:]
        except:
          op_namespace = None
          op_namewithoutnamespace = op_name

        cpp_func.identifier.value = op_namewithoutnamespace.replace('.', '_')

      onnx_op = self._mapped_ops.get(op_name)
      if not onnx_op:
        continue

      yield MappedOpFunction(
        op_namespace,
        op_name,
        onnx_op,
        cpp_func,
        isinstance(onnx_op, SignatureOnly),
        isinstance(onnx_op, MakeTorchFallback))

  def _parse_function_decls(self, cpp_parser: parser.CPPParser):
    # Parse the C++ declarations
    tu = cpp_parser.parse_translation_unit()

    # Parse the Torch schema from the JSON comment that follows each C++ decl
    # and link associated Torch and C++ decls (functions, parameters, returns)
    for cpp_func in tu:
      if self._custom_ops == True:
        # customops don't have torch schema
        cpp_func.torch_func = None
        yield cpp_func
      elif cpp_func.semicolon and cpp_func.semicolon.trailing_trivia:
        for trivia in cpp_func.semicolon.trailing_trivia:
          if trivia.kind == lexer.TokenKind.SINGLE_LINE_COMMENT:
            yield self._parse_and_link_torch_function_decl(cpp_func, trivia)
            break

  def _parse_and_link_torch_function_decl(
    self,
    cpp_func: ast.FunctionDecl,
    torch_schema_comment_trivia: lexer.Token):
    metadata = json.loads(torch_schema_comment_trivia.value.lstrip('//'))
    schema = metadata['schema']

    schema_parser = parser.torch_create_from_string(schema)
    schema_parser.set_source_location(cpp_func.semicolon.location)
    torch_func = schema_parser.parse_function()

    torch_func.torch_schema = schema
    torch_func.torch_dispatch = metadata['dispatch'] == 'True'
    torch_func.torch_default = metadata['default'] == 'True'

    cpp_func.torch_func = torch_func

    if cpp_func.return_type:
      cpp_func.return_type.torch_type = torch_func.return_type

    # Synthesize KWArgsSentinelType in the C++ declaration if we have one
    for i, torch_param in enumerate([p.member for p in torch_func.parameters]):
      if isinstance(torch_param.parameter_type, ast.KWArgsSentinelType):
        cpp_func.parameters.members.insert(i, ast.SyntaxListMember(
          torch_param,
          lexer.Token(None, lexer.TokenKind.COMMA, ',')))
        break

    # Link Torch parameters to their C++ counterparts, special casing
    # TensorOptions parameters
    for i, cpp_param in enumerate([p.member for p in cpp_func.parameters]):
      if not getattr(cpp_param, 'torch_param', None):
        cpp_param.torch_param = []

      torch_param_range = 1
      if isinstance(cpp_param.parameter_type.desugar(), ast.TensorOptionsType):
        torch_param_range = 4

      for j in range(torch_param_range):
        torch_param = torch_func.parameters[i + j].member
        cpp_param.torch_param.append(torch_param)

    return cpp_func