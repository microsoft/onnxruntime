#!/usr/bin/env python3
#
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import argparse

from pathlib import Path

from copy import deepcopy

from opgen.generator import \
  ORTGen as ORTGen, \
  ONNXOp as ONNXOp, \
  SignatureOnly as SignatureOnly, \
  MakeFallthrough as MakeFallthrough

from opgen.onnxops import *

kMSDomain = 'onnxruntime::kMSDomain'

parser = argparse.ArgumentParser(description='Generate ORT ATen operations')
parser.add_argument('--output_file', default=None, type=str, help='Output file [default to std out]')
parser.add_argument('--use_preinstalled_torch', action='store_true', help='Use pre-installed torch from the python environment')

args = parser.parse_args()


class ReluGrad(ONNXOp):
  def __init__(self, dY, X):
    super().__init__('ReluGrad', 1, dY, X)
    self.domain = kMSDomain

class Gelu(ONNXOp):
  def __init__(self, X):
    super().__init__('Gelu', 1, X)
    self.domain = kMSDomain

class GeluGrad(ONNXOp):
  def __init__(self, dY, X):
    super().__init__('GeluGrad', 1, dY, X)
    self.domain = kMSDomain

ops = {
  # Hand-Implemented Ops
  'aten::empty.memory_format': SignatureOnly(),
  'aten::empty_strided': SignatureOnly(),
  'aten::zero_': SignatureOnly(),
  'aten::copy_': SignatureOnly(),
  'aten::reshape': SignatureOnly(),
  'aten::view': SignatureOnly(),

  'aten::addmm': Gemm('mat1', 'mat2', 'self', alpha='alpha', beta='beta'),
  'aten::t': Transpose('self'),
  'aten::mm': MatMul('self', 'mat2'),
  'aten::zeros_like': ConstantOfShape(Shape('self')), #the default constant is 0, so don't need to speicify attribute

  'aten::sum.dim_IntList': ReduceSum('self', 'dim', keepdims='keepdim'),
  'aten::threshold_backward': ReluGrad('grad_output', 'self'),

  'aten::fmod.Scalar': Mod('self', 'other', fmod=1),
  'aten::fmod.Tensor': Mod('self', 'other', fmod=1),

  'aten::softshrink': Shrink('self', bias='lambd', lambd='lambd'), #yes, bias is set to 'lambd'
  'aten::hardshrink': Shrink('self', bias=0, lambd='lambd'),
  'aten::gelu' : Gelu('self'),
  'aten::gelu_backward' : GeluGrad('grad', 'self')
}

for binary_op, onnx_op in {
  'add': Add('self', Mul('alpha', 'other')),
  'sub': Sub('self', Mul('alpha', 'other')),
  'mul': Mul('self', 'other'),
  'div': Div('self', 'other')}.items():
  for dtype in ['Tensor', 'Scalar']:
    for variant in ['', '_']:
      ops[f'aten::{binary_op}{variant}.{dtype}'] = deepcopy(onnx_op)

for unary_op in [
  'abs','acos','acosh', 'asinh', 'atanh', 'asin', 'atan', 'ceil', 'cos',
  'cosh', 'erf', 'exp', 'floor', 'isnan', 'log', 'reciprocal', 'neg', 'round',
  'relu', 'selu', 'sigmoid', 'sin', 'sinh', 'sqrt', 'tan', 'tanh', 'nonzero',
  'sign', 'min', 'max', 'hardsigmoid', 'isinf', 'det']:
  aten_name = f'aten::{unary_op}'
  onnx_op = onnx_ops[unary_op]('self')
  ops[aten_name] = onnx_op
  # produce the in-place variant as well for ops that support it
  if unary_op not in ['isnan', 'nonzero', 'min', 'max', 'isinf', 'det']:
    ops[f'{aten_name}_'] = onnx_op

ortgen = ORTGen(ops)

import os
import sys

from opgen.parser import cpp_create_from_file as CPPParser
from opgen.writer import SourceWriter as SourceWriter

if args.use_preinstalled_torch:
  import torch
  regdecs_path = Path(torch.__file__).parent.joinpath('include/ATen/RegistrationDeclarations.h')
else:
  regdecs_path = os.path.realpath(os.path.join(
    os.path.dirname(__file__),
    '..',
    '..',
    '..',
    'external',
    'pytorch',
    'build',
    'aten',
    'src',
    'ATen',
    'RegistrationDeclarations.h'))

print(f"INFO: Using ATen RegistrationDeclations from: {regdecs_path}")
output = sys.stdout
if not args.output_file is None:
  output = open(args.output_file, 'wt')

with CPPParser(regdecs_path) as parser, SourceWriter(output) as writer:
  ortgen.run(parser, writer)
