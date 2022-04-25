from copy import deepcopy

from opgen.generator import AttrType
from opgen.generator import MakeTorchFallback as MakeTorchFallback
from opgen.generator import ONNXAttr
from opgen.generator import ONNXOp as ONNXOp
from opgen.generator import ORTGen as ORTGen
from opgen.generator import SignatureOnly as SignatureOnly
from opgen.onnxops import *

ops = {
    "gemm": Gemm("A", "B", "C", "alpha", "beta", "transA", "transB"),
    "batchnorm_inplace": BatchNormalization("X", "scale", "B", "input_mean", "input_var", "epsilon", "momentum", 1),
}

type_promotion_ops = {}
