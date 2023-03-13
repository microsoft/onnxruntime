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
    n: int
    cout: int
    cin: int
    h: int
    w: int

    data_type: type


class BenchmarkNhwcConv(BenchmarkOp):
    def __init__(self, args):
        BenchmarkOp.__init__(self, args)

    @classmethod
    def create_inputs_outputs(cls, op_param):
        np.random.seed(0)
        input_data = np.random.rand(op_param.n, op_param.h, op_param.w, op_param.cin).astype(op_param.data_type)
        weight = np.random.rand(op_param.cout, 3, 3, op_param.cin).astype(op_param.data_type)
        bias = np.random.rand(op_param.cout).astype(op_param.data_type)
        output = np.random.rand(op_param.n, op_param.h, op_param.w, op_param.cout).astype(op_param.data_type)
        inputs = {"input": input_data, "weight": weight, "bias": bias}
        outputs = {"conv": output}
        return inputs, outputs

    def create_cases(self):
        # attributes of model : kernel_shape(3,3), group(1), pads(1,1), strides(1,1), dilations(1,1)
        model = "models/nhwcConv_fp16.onnx" if self.args.precision == "fp16" else "models/nhwcConv_fp32.onnx"
        data_type = np.float16 if self.args.precision == "fp16" else np.float32

        # change here to test your data shape
        self.add_case(OpParam(2, 320, 320, 64, 64, data_type), model)

    @classmethod
    def case_profile(cls, op_param, time):
        profile = f"( n cout cin h w ) = ( {op_param.n} {op_param.cout} {op_param.cin} {op_param.h} {op_param.w} ), {time * 1000:7.4f} us"
        return profile


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    bm = BenchmarkNhwcConv(args)
    bm.benchmark()


if __name__ == "__main__":
    main()
