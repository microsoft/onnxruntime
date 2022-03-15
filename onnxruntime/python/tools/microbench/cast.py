import argparse
from dataclasses import dataclass
from pydoc import ispackage
import numpy as np
from benchmark import BenchmarkOp, add_arguments


@dataclass
class OpParam:
    x : int
    y : int
    m : int
    n : int
    input_data_type : type
    output_data_type : type
    


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

    def create_cases(self):
        test_data = self.create_test_input_shape()
        for (x, y, m, n, is_fp16) in test_data:
            model = "models/cast_fp16tofp32.onnx" if is_fp16 else "models/cast_fp32tofp16.onnx"
            input_data_type = np.float16 if is_fp16 else np.float32
            output_data_type = np.float32 if is_fp16 else np.float16
            # bert-base
            op_param = OpParam(x, y, m, n, input_data_type, output_data_type)
            self.add_case(op_param, model)

    def case_profile(cls, op_param, time):
        profile = f"(x y m n input_data_type) = ({op_param.x} {op_param.y} {op_param.m} {op_param.n} {op_param.input_data_type}), {time:7.4f} ms"
        return profile
    
    def create_test_input_shape(self):
        test_input_shape = [
            (1, 384, 768, 768, 0),
            (1, 384, 768, 768, 1),
            (1, 1, 4, 512, 0),
            (1, 1, 4, 512, 1),
            (1, 4, 512, 32128, 0),
            (1, 4, 512, 32128, 1),
            (1, 4, 512, 512, 0),
            (1, 4, 512, 512, 1),
            (1, 4, 512, 2048, 0),
            (1, 4, 512, 2048, 1),
            (16, 1920, 5120, 1, 0),
            (16, 5120, 640, 1, 0),
            (16, 2560, 5120, 1, 0)
        ]
        return test_input_shape


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    bm = BenchmarkCast(args)
    bm.benchmark()


if __name__ == "__main__":
    main()
