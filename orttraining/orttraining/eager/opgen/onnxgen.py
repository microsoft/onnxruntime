#!/usr/bin/env python3
#
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os.path as path
from sys import argv
from onnx import defs
from onnx import AttributeProto

out_file = path.join(
  path.dirname(path.realpath(__file__)),
  'opgen',
  'onnxops.py')

#generate the onnx op eager mode frontend code
onnx_op_eager_file = path.join(
  path.dirname(path.realpath(__file__)),
  'opgen',
  'onnx_op_eager.py')

onnx_op_declaration_file = path.join(
  path.dirname(path.realpath(__file__)),
  'opgen',
  'OnnxOpDeclarations.h')

onnx_ops = {}
for schema in defs.get_all_schemas_with_history():
  key = schema.name.lower()
  if schema.deprecated:
    continue
  if key not in onnx_ops or \
    onnx_ops[key].since_version < schema.since_version:
    onnx_ops[key] = schema

def convert_to_aten_type(onnx_type_strs):
  type_map = {'tensor(float16)' : 'at::kHalf',
              'tensor(float)' : 'at::kFloat',
              'tensor(double)' : 'at::kDouble',
              'tensor(bfloat16)' : 'at::kBFloat16',
              'tensor(int32)' : 'at::kInt',
              'tensor(int16)' : 'at::kShort',
              'tensor(int8)' : 'at::kByte',
              'tensor(int64)' : 'at::kLong',
              'tensor(bool)' : 'at::kBool',
             }
  result = set({})
  for onnx_type in onnx_type_strs:
    # ONNX has more types, like tensor(string), ignore those types at this momemnt
    if onnx_type in type_map:
      result.add(type_map[onnx_type])
  return result

def write(f, s): 
  f.write(s)
def writeline(f, s = ''): 
  f.write(s + '\n')

def write_onnxops_header(fp):
  writeline(fp, f'# AUTO-GENERATED CODE! - DO NOT EDIT!')
  writeline(fp, f'# $ python {" ".join(argv)}')
  writeline(fp)

  writeline(fp, 'from opgen.generator import ONNXAttr, ONNXOp, AttrType')
  writeline(fp)

def write_onnx_op_eager_header(fp):
  writeline(fp, f'# AUTO-GENERATED CODE! - DO NOT EDIT!')
  writeline(fp, f'# $ python {" ".join(argv)}')
  writeline(fp)

  writeline(fp, 'from opgen.onnxops import *')
  writeline(fp)

def write_onnx_op_mapping(fp, schema):
  inputs = [f'\'{input.name}\'' for input in schema.inputs]
  for _, attr in schema.attributes.items():
    inputs.append(f'{attr.name}=\'{attr.name}\'')
  inputs_str = ','.join(inputs)
  writeline(fp, f'    \'{schema.name}\': {schema.name}({inputs_str}),')

def is_supported_onnx_eager(schema):
  if len(schema.outputs) > 1:
    return False
  if len(schema.inputs) == 0:
    return False
  for _, attr in schema.attributes.items():
    if attr.type not in (AttributeProto.AttributeType.FLOAT,
                         AttributeProto.AttributeType.INT):
      return False
  return True

def write_onnx_op_declarations(fp, schema):
  # more than 1 output is not supported
  if len(schema.outputs) > 1:
    return
  output = 'Tensor'
  op_name = schema.name
  #TODO: in-place invoke
  args = [f'const Tensor& {input.name}' for input in schema.inputs]
  for _, attr in schema.attributes.items():
    if attr.type == AttributeProto.AttributeType.FLOAT:
      args.append(f'float {attr.name}')
    elif attr.type == AttributeProto.AttributeType.INT:
      args.append(f'int {attr.name}')
    else:
      return
  args_str = ', '.join(args)
  writeline(fp, f'{output} {op_name}({args_str});') 

with open(out_file, 'wt') as fp,  \
     open(onnx_op_eager_file, 'wt') as onnx_fp, \
     open(onnx_op_declaration_file, 'wt') as onnx_header_fp:
  # write onnxops.py header
  write_onnxops_header(fp)
  # write onnx_op_eager.py header
  write_onnx_op_eager_header(onnx_fp)

  #start the ops declaration in onnx_op_eager.py
  writeline(onnx_fp, 'ops = {')

  for op_name, schema in sorted(onnx_ops.items()):
    writeline(fp, f'class {schema.name}(ONNXOp):')
    writeline(fp, f'  """')
    doc_str = schema.doc.strip('\r\n')
    for doc_line in str.splitlines(doc_str, keepends=False):
      writeline(fp, f'  {doc_line}')
    writeline(fp, f'  """')
    writeline(fp, )
    write(fp, '  def __init__(self')

    for input in schema.inputs:
      write(fp, f', {input.name}')

    if len(schema.attributes) > 0:
      writeline(fp, ',')
      for i, (k, attr) in enumerate(schema.attributes.items()):
        write(fp, f'    {attr.name}=None')
        if i < len(schema.attributes) - 1:
          writeline(fp, ', ')

    writeline(fp, '):')
    write(fp, f'    super().__init__(\'{schema.name}\', {len(schema.outputs)}')
    writeline(fp, ',')
    write(fp, '      ')
    input_types = []
    for input in schema.inputs:
      input_types.append(convert_to_aten_type(input.types))
    write(fp, str(input_types))
    if len(schema.inputs) > 0:
      writeline(fp, ',')
      input_names = ','.join([input.name for input in schema.inputs])
      write(fp, f'      {input_names}')
    

    if len(schema.attributes) > 0:
      writeline(fp, ',')
      for i, (k, attr) in enumerate(schema.attributes.items()):
        write(fp, f'      {attr.name}=ONNXAttr({attr.name}, {attr.type})')
        if i < len(schema.attributes) - 1:
          writeline(fp, ', ')

    writeline(fp, ')')
    writeline(fp)
    if is_supported_onnx_eager(schema):
      write_onnx_op_declarations(onnx_header_fp, schema)
      write_onnx_op_mapping(onnx_fp, schema)

  writeline(fp, 'onnx_ops = {')
  for i, (op_name, schema) in enumerate(onnx_ops.items()):
    writeline(fp, f'  \'{op_name}\': {schema.name},')
  write(fp, '}')

  writeline(onnx_fp, '}')
  writeline(onnx_fp, 'type_promotion_ops = {}')

print(f'File updated: {out_file}')