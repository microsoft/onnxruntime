from copy import deepcopy

from opgen.generator import AttrType, ONNXAttr

from opgen.generator import \
  ORTGen as ORTGen, \
  ONNXOp as ONNXOp, \
  SignatureOnly as SignatureOnly, \
  MakeFallthrough as MakeFallthrough

from opgen.onnxops import *

ops = {
    'gemm': Gemm('A', 'B', 'C', 'alpha', 'beta', 'transA', 'transB')
}
