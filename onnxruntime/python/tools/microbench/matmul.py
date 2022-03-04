import numpy as np
from benchmark import benchmark 


def create_inputs_outputs(b1, b2, m, k, n, data_type):
    np.random.seed(0)
    a = np.random.rand(b1, b2, m, k).astype(data_type)
    b = np.random.rand(b1, b2, k, n).astype(data_type)
    c = np.random.rand(b1, b2, m, n).astype(data_type)

    inputs = {"A": a, "B": b}
    outputs = {"return_val": c}
 
    return inputs, outputs


def add_benchmark_case(benchmark_cases, batch_size, seq_len, hidden_size, intermediate_dimension, num_heads, data_type, model):
    benchmark_cases += [
        (1, batch_size, seq_len, hidden_size, hidden_size, data_type, model),
        (1, batch_size, seq_len, intermediate_dimension, hidden_size, data_type, model),
        (1, batch_size, seq_len, hidden_size, intermediate_dimension, data_type, model),
        (batch_size, num_heads, seq_len, seq_len, int(hidden_size / num_heads), data_type, model),
    ]


def create_benchmark_cases(precision="fp16"):
    benchmark_cases = []
    if precision == "fp16":
      model = "models/matmul_fp16.onnx"
      data_type = np.float16
    else:
      model = "models/matmul_fp32.onnx"
      data_type = np.float32

    # bert-large
    hidden_size = 1024
    seq_len = 384
    num_heads = 16
    batch_size = 1
    intermediate_dimension = hidden_size * 4
    add_benchmark_case(benchmark_cases, batch_size, seq_len, hidden_size, intermediate_dimension, num_heads, data_type, model)

    # bert-base
    hidden_size = 768
    seq_len = 384
    num_heads = 12
    batch_size = 1
    intermediate_dimension = hidden_size * 4
    add_benchmark_case(benchmark_cases, batch_size, seq_len, hidden_size, intermediate_dimension, num_heads, data_type, model)

    return benchmark_cases


def benchmark_matmul(b1, b2, m, k, n, data_type, onnx_file):
    inputs, outputs = create_inputs_outputs(b1, b2, m, k, n, data_type)
    time = benchmark(onnx_file, inputs, outputs, "rocm")
    return time


def main():
    for (b1, b2, m, k, n, data_type, onnx_file) in create_benchmark_cases(): 
        time = benchmark_matmul(b1, b2, m, k, n, data_type, onnx_file)
        tflops = b1 * b2 * m * k * n * 2 / time / 1000000000
        print(f"(b1 b2 m k n) = ({b1} {b2} {m} {k} {n}), {time:7.4f} ms, {tflops:4.2f} tflops")


if __name__ == "__main__":
    main()
