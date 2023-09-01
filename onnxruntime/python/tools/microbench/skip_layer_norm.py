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
    batch_size: int
    seq_len: int
    hidden_size: int
    data_type: type


class BenchmarkSkipLayerNorm(BenchmarkOp):
    def __init__(self, args):
        BenchmarkOp.__init__(self, args)

    @classmethod
    def create_inputs_outputs(cls, op_param):
        np.random.seed(0)
        input_data = np.random.rand(op_param.batch_size, op_param.seq_len, op_param.hidden_size).astype(
            op_param.data_type
        )
        skip = np.random.rand(op_param.batch_size, op_param.seq_len, op_param.hidden_size).astype(op_param.data_type)
        gamma = np.random.rand(op_param.hidden_size).astype(op_param.data_type)
        beta = np.random.rand(op_param.hidden_size).astype(op_param.data_type)
        bias = np.random.rand(op_param.hidden_size).astype(op_param.data_type)
        output_data = np.random.rand(op_param.batch_size, op_param.seq_len, op_param.hidden_size).astype(
            op_param.data_type
        )

        inputs = {
            "INPUT": input_data,
            "SKIP": skip,
            "GAMMA": gamma,
            "BETA": beta,
            "BIAS": bias,
        }
        outputs = {"return_val": output_data}

        return inputs, outputs

    def create_cases(self):
        model = (
            "models/skip_layer_norm_fp16.onnx" if self.args.precision == "fp16" else "models/skip_layer_norm_fp32.onnx"
        )
        data_type = np.float16 if self.args.precision == "fp16" else np.float32
        # bert-large
        op_param = OpParam(1, 384, 1024, data_type)
        self.add_case(op_param, model)

    @classmethod
    def case_profile(cls, op_param, time):
        profile = f"(batch seq_len hidden_size) = ({op_param.batch_size} {op_param.seq_len} {op_param.hidden_size}), {time:7.4f} ms"
        return profile


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    bm = BenchmarkSkipLayerNorm(args)
    bm.benchmark()


if __name__ == "__main__":
    main()
