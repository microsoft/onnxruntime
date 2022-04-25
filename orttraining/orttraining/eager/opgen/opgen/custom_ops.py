from copy import deepcopy

from opgen.generator import AttrType, ONNXAttr

from opgen.generator import (
    ORTGen as ORTGen,
    ONNXOp as ONNXOp,
    SignatureOnly as SignatureOnly,
    MakeTorchFallback as MakeTorchFallback,
)

from opgen.onnxops import *

ops = {
    "gemm": Gemm("A", "B", "C", "alpha", "beta", "transA", "transB"),
    "batchnorm_inplace": BatchNormalization("X", "scale", "B", "input_mean", "input_var", "epsilon", "momentum", 1),
}

type_promotion_ops = {}
