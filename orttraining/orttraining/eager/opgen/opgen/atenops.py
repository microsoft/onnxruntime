from copy import deepcopy

from opgen.generator import \
  ORTGen as ORTGen, \
  ONNXOp as ONNXOp, \
  SignatureOnly as SignatureOnly, \
  MakeFallthrough as MakeFallthrough

from opgen.onnxops import *

kMSDomain = 'onnxruntime::kMSDomain'

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
