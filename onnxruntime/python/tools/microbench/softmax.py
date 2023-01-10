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


class BenchmarkSoftmax(BenchmarkOp):
    def __init__(self, args):
        BenchmarkOp.__init__(self, args)

    @classmethod
    def create_inputs_outputs(cls, op_param):
        np.random.seed(0)
        input_data = np.random.rand(op_param.dim1, op_param.dim2, op_param.dim3).astype(op_param.data_type)
        softmax_output = np.random.rand(op_param.dim1, op_param.dim2, op_param.dim3).astype(op_param.data_type)
        inputs = {"input": input_data}
        outputs = {"softmax": softmax_output}
        return inputs, outputs

    def create_cases(self):
        # attributes of model : axis=-1
        model = "models/softmax_fp16.onnx" if self.args.precision == "fp16" else "models/softmax_fp32.onnx"
        data_type = np.float16 if self.args.precision == "fp16" else np.float32

        # change here to test your data shape
        self.add_case(OpParam(1, 32, 32, data_type), model)

    @classmethod
    def case_profile(cls, op_param, time):
        profile = f"(dim1 dim2 dim3) = ({op_param.dim1} {op_param.dim2} {op_param.dim3}), {time * 1000:7.4f} us"
        return profile


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    bm = BenchmarkSoftmax(args)
    bm.benchmark()


if __name__ == "__main__":
    main()
