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
    k: int

    data_type: type


class BenchmarkTopK(BenchmarkOp):
    def __init__(self, args):
        BenchmarkOp.__init__(self, args)

    @classmethod
    def create_inputs_outputs(cls, op_param):
        np.random.seed(0)
        input_data = np.random.rand(op_param.dim1, op_param.dim2, op_param.dim3).astype(op_param.data_type)
        dim_k = np.array([op_param.k]).astype(np.int64)
        values = np.random.rand(op_param.dim1, op_param.dim2, op_param.k).astype(op_param.data_type)
        indices = np.random.rand(op_param.dim1, op_param.dim2, op_param.k).astype(np.int64)
        inputs = {"input": input_data, "k": dim_k}
        outputs = {"values": values, "indices": indices}
        return inputs, outputs

    def create_cases(self):
        # attributes of model: axis=-1, largest=1, sorted=1
        model = "models/topk_fp16.onnx" if self.args.precision == "fp16" else "models/topk_fp32.onnx"
        data_type = np.float16 if self.args.precision == "fp16" else np.float32

        # change here to test your data shape
        self.add_case(OpParam(1, 32, 32, 2, data_type), model)

    @classmethod
    def case_profile(cls, op_param, time):
        profile = f"(dim1 dim2 dim3) = ({op_param.dim1} {op_param.dim2} {op_param.dim3}), k = {op_param.k}, {time * 1000:7.4f} us"
        return profile


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    bm = BenchmarkTopK(args)
    bm.benchmark()


if __name__ == "__main__":
    main()
