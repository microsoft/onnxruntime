from copy import deepcopy

from opgen.generator import \
  ORTGen as ORTGen, \
  ONNXOp as ONNXOp, \
  SignatureOnly as SignatureOnly, \
  MakeTorchFallback as MakeTorchFallback

from opgen.onnxops import *

kMSDomain = 'onnxruntime::kMSDomain'

class ReluGrad(ONNXOp):
  def __init__(self, dY, X):
    super().__init__('ReluGrad', 1, [{'at::kHalf', 'at::kFloat', 'at::kBFloat16'}, {'at::kHalf', 'at::kFloat', 'at::kBFloat16'}], dY, X)
    self.domain = kMSDomain

class Gelu(ONNXOp):
  def __init__(self, X):
    super().__init__('Gelu', 1, [{'at::kHalf', 'at::kFloat', 'at::kBFloat16'}], X)
    self.domain = kMSDomain

class GeluGrad(ONNXOp):
  def __init__(self, dY, X):
    super().__init__('GeluGrad', 1, [{'at::kHalf', 'at::kFloat', 'at::kBFloat16'}, {'at::kHalf', 'at::kFloat', 'at::kBFloat16'}], dY, X)
    self.domain = kMSDomain

ops = {}

for binary_op, onnx_op in {
  'add': Add('self', Mul('alpha', 'other')),
  'sub': Sub('self', Mul('alpha', 'other')),
  'mul': Mul('self', 'other'),
  'div': Div('self', 'other')}.items():
  for dtype in ['Tensor', 'Scalar']:
    for variant in ['', '_']:
      name = f'aten::{binary_op}{variant}.{dtype}'
      if name not in ops:
        ops[f'aten::{binary_op}{variant}.{dtype}'] = deepcopy(onnx_op)

for unary_op in [
  'abs','acos','acosh', 'asinh', 'atanh', 'asin', 'atan', 'ceil', 'cos',
  'cosh', 'erf', 'exp', 'floor', 'isnan', 'log', 'reciprocal', 'neg', 'round',
  'relu', 'selu', 'sigmoid', 'sin', 'sinh', 'sqrt', 'tan', 'tanh', 'nonzero',
  'sign', 'hardsigmoid', 'isinf', 'det']:
  aten_name = f'aten::{unary_op}'
  onnx_op = onnx_ops[unary_op]('self')
  ops[aten_name] = onnx_op
  # produce the in-place variant as well for ops that support it
  if unary_op not in ['isnan', 'nonzero', 'min', 'max', 'isinf', 'det']:
    ops[f'{aten_name}_'] = onnx_op

hand_implemented = {
  'aten::empty.memory_format': SignatureOnly(),
  'aten::empty_strided': SignatureOnly(),
  'aten::zero_': SignatureOnly(),
  'aten::copy_': SignatureOnly(),
  'aten::_reshape_alias': SignatureOnly(),
  'aten::view': SignatureOnly(),
  'aten::_copy_from_and_resize' : SignatureOnly(),
  'aten::addmm': Gemm('mat1', 'mat2', 'self', alpha='alpha', beta='beta'),
  'aten::add_.Tensor': SignatureOnly(),
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
  'aten::gelu_backward' : GeluGrad('grad', 'self'),
  'aten::max' : ReduceMax('self', keepdims=1),
  'aten::min' : ReduceMin('self', keepdims=1),
  'aten::slice.Tensor' : Slice('self', 'start', 'end', 'dim', 'step'),
  'aten::_cat': Concat('tensors', 'dim'),

  'aten::ne.Scalar':MakeTorchFallback(),
  'aten::ne.Scalar_out': MakeTorchFallback(),
  'aten::ne.Tensor_out': MakeTorchFallback(),
  'aten::eq.Tensor': MakeTorchFallback(),
  'aten::eq.Tensor_out':MakeTorchFallback(),
  'aten::bitwise_and.Tensor_out' : MakeTorchFallback(),
  'aten::masked_select' : MakeTorchFallback(),
  'aten::as_strided' : MakeTorchFallback(),
  'aten::_local_scalar_dense' : MakeTorchFallback(),
  'aten::gt.Scalar_out' : MakeTorchFallback(),
}

ops = {**ops, **hand_implemented} 
