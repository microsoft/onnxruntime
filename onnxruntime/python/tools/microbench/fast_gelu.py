import argparse
import numpy as np
from benchmark import benchmark, add_arguments


def create_inputs_outputs(batch, seq_len, intermediate_dimension, data_type):
    np.random.seed(0)
    a = np.random.rand(batch, seq_len, intermediate_dimension).astype(data_type)
    b = np.random.rand(intermediate_dimension).astype(data_type)
    c = np.random.rand(batch, seq_len, intermediate_dimension).astype(data_type)

    inputs = {"A": a, "B": b}
    outputs = {"return_val": c}
 
    return inputs, outputs


def add_benchmark_case(benchmark_cases, batch_size, seq_len, intermediate_dimension, data_type, model):
    benchmark_cases += [
        (batch_size, seq_len, intermediate_dimension, data_type, model),
    ]


def create_benchmark_cases(precision):
    benchmark_cases = []
    if precision == "fp16":
      model = "models/fast_gelu_fp16.onnx"
      data_type = np.float16
    else:
      model = "models/fast_gelu_fp32.onnx"
      data_type = np.float32

    # bert-large
    hidden_size = 1024
    seq_len = 384
    batch_size = 1
    intermediate_dimension = hidden_size * 4
    add_benchmark_case(benchmark_cases, batch_size, seq_len, intermediate_dimension, data_type, model)

    return benchmark_cases


def benchmark_fast_gelu(batch, seq_len, intermediate_dimension, data_type, onnx_file, args):
    inputs, outputs = create_inputs_outputs(batch, seq_len, intermediate_dimension, data_type)
    time = benchmark(onnx_file, inputs, outputs, args)
    return time


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    for (batch, seq_len, intermediate_dimension, data_type, onnx_file) in create_benchmark_cases(args.precision):
        time = benchmark_fast_gelu(batch, seq_len, intermediate_dimension, data_type, onnx_file, args)
        print(f"(batch seq_len inter_dim) = ({batch} {seq_len} {intermediate_dimension}), {time:7.4f} ms")


if __name__ == "__main__":
    main()
