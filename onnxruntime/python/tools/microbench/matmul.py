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
    b1: int
    b2: int
    m: int
    k: int
    n: int
    data_type: type


@dataclass
class ModelParam:
    batch_size: int
    seq_len: int
    hidden_size: int
    inter_dim: int
    num_heads: int
    data_type: type


class BenchmarkMatMul(BenchmarkOp):
    def __init__(self, args):
        BenchmarkOp.__init__(self, args)

    @classmethod
    def create_inputs_outputs(cls, op_param):
        np.random.seed(0)
        a = np.random.rand(op_param.b1, op_param.b2, op_param.m, op_param.k).astype(op_param.data_type)
        b = np.random.rand(op_param.b1, op_param.b2, op_param.k, op_param.n).astype(op_param.data_type)
        c = np.random.rand(op_param.b1, op_param.b2, op_param.m, op_param.n).astype(op_param.data_type)
        inputs = {"A": a, "B": b}
        outputs = {"return_val": c}
        return inputs, outputs

    def add_model_cases(self, mp, model):
        self.add_case(
            OpParam(
                1,
                mp.batch_size,
                mp.seq_len,
                mp.hidden_size,
                mp.hidden_size,
                mp.data_type,
            ),
            model,
        )
        self.add_case(
            OpParam(1, mp.batch_size, mp.seq_len, mp.inter_dim, mp.hidden_size, mp.data_type),
            model,
        )
        self.add_case(
            OpParam(1, mp.batch_size, mp.seq_len, mp.hidden_size, mp.inter_dim, mp.data_type),
            model,
        )
        self.add_case(
            OpParam(
                mp.batch_size,
                mp.num_heads,
                mp.seq_len,
                mp.seq_len,
                int(mp.hidden_size / mp.num_heads),
                mp.data_type,
            ),
            model,
        )

    def create_cases(self):
        model = "models/matmul_fp16.onnx" if self.args.precision == "fp16" else "models/matmul_fp32.onnx"
        data_type = np.float16 if self.args.precision == "fp16" else np.float32
        # bert-large
        model_param = ModelParam(1, 384, 1024, 1024 * 4, 16, data_type)
        self.add_model_cases(model_param, model)
        # bert-base
        model_param = ModelParam(1, 384, 768, 768 * 4, 12, data_type)
        self.add_model_cases(model_param, model)

    @classmethod
    def case_profile(cls, op_param, time):
        tflops = op_param.b1 * op_param.b2 * op_param.m * op_param.k * op_param.n * 2 / time / 1000000000
        profile = f"(b1 b2 m k n) = ({op_param.b1} {op_param.b2} {op_param.m} {op_param.k} {op_param.n}), {time:7.4f} ms, {tflops:4.2f} tflops"
        return profile


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    bm = BenchmarkMatMul(args)
    bm.benchmark()


if __name__ == "__main__":
    main()
