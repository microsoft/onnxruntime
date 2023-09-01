# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
from dataclasses import dataclass

import numpy as np
from benchmark import BenchmarkOp, add_arguments


@dataclass
class OpParam:
    dim1: int
    dim2: int
    dim3: int
    data_type: type


@dataclass
class ModelParam:
    batch_size: int
    seq_len: int
    inter_dim: int
    data_type: type


class BenchmarkFastGelu(BenchmarkOp):
    def __init__(self, args):
        BenchmarkOp.__init__(self, args)

    @classmethod
    def create_inputs_outputs(cls, op_param):
        np.random.seed(0)
        a = np.random.rand(op_param.dim1, op_param.dim2, op_param.dim3).astype(op_param.data_type)
        b = np.random.rand(op_param.dim3).astype(op_param.data_type)
        c = np.random.rand(op_param.dim1, op_param.dim2, op_param.dim3).astype(op_param.data_type)
        inputs = {"A": a, "B": b}
        outputs = {"return_val": c}
        return inputs, outputs

    def create_cases(self):
        model = "models/fast_gelu_fp16.onnx" if self.args.precision == "fp16" else "models/fast_gelu_fp32.onnx"
        data_type = np.float16 if self.args.precision == "fp16" else np.float32
        # bert-large
        model_param = ModelParam(1, 384, 1024 * 4, data_type)
        op_param = OpParam(
            model_param.batch_size,
            model_param.seq_len,
            model_param.inter_dim,
            model_param.data_type,
        )
        self.add_case(op_param, model)

    @classmethod
    def case_profile(cls, op_param, time):
        profile = f"(dim1 dim2 dim3) = ({op_param.dim1} {op_param.dim2} {op_param.dim3}), {time:7.4f} ms"
        return profile


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    bm = BenchmarkFastGelu(args)
    bm.benchmark()


if __name__ == "__main__":
    main()
