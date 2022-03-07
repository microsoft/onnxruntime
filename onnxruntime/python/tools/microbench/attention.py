import argparse
from dataclasses import dataclass
import numpy as np
from benchmark import BenchmarkOp, add_arguments


@dataclass
class ModelParam:
    batch_size: int
    seq_len: int
    hidden_size: int
    length: int
    data_type: type


class BenchmarkAttention(BenchmarkOp):
    def __init__(self, args):
        BenchmarkOp.__init__(self, args)

    def create_inputs_outputs(cls, model_param):
        np.random.seed(0)
        input_data = np.random.rand(model_param.batch_size, model_param.seq_len, model_param.hidden_size).astype(model_param.data_type)
        weight = np.random.rand(model_param.hidden_size, model_param.length).astype(model_param.data_type)
        bias = np.random.rand(model_param.length).astype(model_param.data_type)
        mask_index = np.random.rand(model_param.batch_size, model_param.seq_len).astype(np.int32)
        output_data = np.random.rand(model_param.batch_size, model_param.seq_len, model_param.hidden_size).astype(model_param.data_type)
        inputs = {"INPUT": input_data, "WEIGHT": weight, "BIAS": bias, "MASK_INDEX": mask_index}
        outputs = {"return_val": output_data}
        return inputs, outputs

    def create_cases(self):
        model = "models/attention_fp16.onnx" if self.args.precision == "fp16" else "models/attention_fp32.onnx"
        data_type = np.float16 if self.args.precision == "fp16" else np.float32
        # bert-base
        model_param = ModelParam(1, 384, 768, 768 * 3, data_type)
        self.add_case(model_param, model)

    def case_profile(cls, model_param, time):
        profile = f"(batch_size seq_len length) = ({model_param.batch_size} {model_param.seq_len} {model_param.length}), {time:7.4f} ms"
        return profile


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    bm = BenchmarkAttention(args)
    bm.benchmark()


if __name__ == "__main__":
    main()
