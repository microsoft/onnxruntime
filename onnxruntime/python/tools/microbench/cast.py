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
    x: int
    y: int
    m: int
    n: int
    input_data_type: type
    output_data_type: type


@dataclass
class ModelParam:
    token_type_ids_dim0: int
    input_ids_dim1: int


class BenchmarkCast(BenchmarkOp):
    def __init__(self, args):
        BenchmarkOp.__init__(self, args)

    def create_inputs_outputs(cls, op_param):
        np.random.seed(0)
        input_data = np.random.rand(op_param.x, op_param.y, op_param.m, op_param.n).astype(op_param.input_data_type)
        output_data = np.random.rand(op_param.x, op_param.y, op_param.m, op_param.n).astype(op_param.output_data_type)
        inputs = {"X": input_data}
        outputs = {"Y": output_data}
        return inputs, outputs

    def add_model_cases(self, mp, model, input_data_type, output_data_type):
        self.add_case(
            OpParam(
                1,
                mp.token_type_ids_dim0,
                mp.input_ids_dim1,
                1024,
                input_data_type,
                output_data_type,
            ),
            model,
        )
        self.add_case(
            OpParam(
                1,
                mp.token_type_ids_dim0,
                mp.input_ids_dim1,
                1,
                input_data_type,
                output_data_type,
            ),
            model,
        )
        self.add_case(
            OpParam(
                16,
                mp.token_type_ids_dim0,
                mp.input_ids_dim1,
                mp.input_ids_dim1,
                input_data_type,
                output_data_type,
            ),
            model,
        )

    def create_cases(self):
        model = "models/cast_fp16tofp32.onnx" if self.args.precision == "fp16" else "models/cast_fp32tofp16.onnx"
        input_data_type = np.float16 if self.args.precision == "fp16" else np.float32
        output_data_type = np.float32 if self.args.precision == "fp16" else np.float16
        # huggingface bert-large
        self.add_case(OpParam(1, 1, 1, 1024, input_data_type, output_data_type), model)
        self.add_case(OpParam(1, 1, 1024, 1024, input_data_type, output_data_type), model)
        self.add_case(OpParam(1, 1, 1024, 4096, input_data_type, output_data_type), model)
        self.add_case(OpParam(1, 1, 1024, 30522, input_data_type, output_data_type), model)
        # huggingface bert-large with default dims
        model_param = ModelParam(8, 512)
        self.add_model_cases(model_param, model, input_data_type, output_data_type)
        # huggingface bert-large with large input dims
        model_param = ModelParam(32, 1024)
        self.add_model_cases(model_param, model, input_data_type, output_data_type)

    def case_profile(cls, op_param, time):
        profile = f"(x y m n input_data_type) = ({op_param.x} {op_param.y} {op_param.m} {op_param.n} {op_param.input_data_type}), {time:7.4f} ms"
        return profile


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    bm = BenchmarkCast(args)
    bm.benchmark()


if __name__ == "__main__":
    main()
