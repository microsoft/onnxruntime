from opgen.onnxops import BatchNormalization, Gemm

ops = {
    "gemm": Gemm("A", "B", "C", "alpha", "beta", "transA", "transB"),
    "batchnorm_inplace": BatchNormalization("X", "scale", "B", "input_mean", "input_var", "epsilon", "momentum", 1),
}

type_promotion_ops = {}
aten_output_type = {}
