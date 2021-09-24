#!/usr/bin/env python3
#
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os.path as path
from sys import argv
from onnx import defs

out_file = path.join(
  path.dirname(path.realpath(__file__)),
  'opgen',
  'onnxops.py')

onnx_ops = {}
for schema in defs.get_all_schemas_with_history():
  key = schema.name.lower()
  if schema.deprecated:
    continue
  if key not in onnx_ops or \
    onnx_ops[key].since_version < schema.since_version:
    onnx_ops[key] = schema

with open(out_file, 'wt') as fp:
  def write(s): fp.write(s)
  def writeline(s = ''): fp.write(s + '\n')

  writeline(f'# AUTO-GENERATED CODE! - DO NOT EDIT!')
  writeline(f'# $ python {" ".join(argv)}')
  writeline()

  writeline('from opgen.generator import ONNXAttr, ONNXOp, AttrType')
  writeline()

  for op_name, schema in sorted(onnx_ops.items()):
    writeline(f'class {schema.name}(ONNXOp):')
    writeline(f'  """')
    doc_str = schema.doc.strip('\r\n')
    for doc_line in str.splitlines(doc_str, keepends=False):
      writeline(f'  {doc_line}')
    writeline(f'  """')
    writeline()
    write('  def __init__(self')

    for input in schema.inputs:
      write(f', {input.name}')

    if len(schema.attributes) > 0:
      writeline(',')
      for i, (k, attr) in enumerate(schema.attributes.items()):
        write(f'    {attr.name}=None')
        if i < len(schema.attributes) - 1:
          writeline(', ')

    writeline('):')
    write(f'    super().__init__(\'{schema.name}\', {len(schema.outputs)}')

    for input in schema.inputs:
      write(f', {input.name}')

    if len(schema.attributes) > 0:
      writeline(',')
      for i, (k, attr) in enumerate(schema.attributes.items()):
        write(f'      {attr.name}=ONNXAttr({attr.name}, {attr.type})')
        if i < len(schema.attributes) - 1:
          writeline(', ')

    writeline(')')
    writeline()

  writeline('onnx_ops = {')
  for i, (op_name, schema) in enumerate(onnx_ops.items()):
    writeline(f'  \'{op_name}\': {schema.name},')
  write('}')

print(f'File updated: {out_file}')