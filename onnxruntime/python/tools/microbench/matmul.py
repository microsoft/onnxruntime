import numpy as np
import torch
from benchmark import create_session, benchmark 


def create_io_binding(sess, b1, b2, m, k, n, data_type):
    np.random.seed(0)
    a = np.random.rand(b1, b2, m, k).astype(data_type)
    b = np.random.rand(b1, b2, k, n).astype(data_type)
    c = np.random.rand(b1, b2, m, n).astype(data_type)

    io_binding = sess.io_binding()
    device = "cuda"

    a_d = torch.from_numpy(a).to(device)
    b_d = torch.from_numpy(b).to(device)
    c_d = torch.from_numpy(c).to(device)

    io_binding.bind_input("A", a_d.device.type, 0, data_type, a_d.shape, a_d.data_ptr())
    io_binding.bind_input("B", b_d.device.type, 0, data_type, b_d.shape, b_d.data_ptr())
    io_binding.bind_output("return_val", c_d.device.type, 0, data_type, c_d.shape, c_d.data_ptr())

    return io_binding


def create_benchmark_cases():
    benchmark_cases = []
    model_fp32 = "models/matmul_fp32.onnx"
    model_fp16 = "models/matmul_fp16.onnx"

    # bert-large
    hidden_size = 1024
    seq_len = 384
    num_heads = 16
    batch_size = 1
    intermediate_dimension = hidden_size * 4
    benchmark_cases += [
        (1, batch_size, seq_len, hidden_size, hidden_size, np.float32, model_fp32),
        (1, batch_size, seq_len, intermediate_dimension, hidden_size, np.float32, model_fp32),
        (1, batch_size, seq_len, hidden_size, intermediate_dimension, np.float32, model_fp32),
        (batch_size, num_heads, seq_len, seq_len, int(hidden_size / num_heads), np.float32, model_fp32),
        (1, batch_size, seq_len, hidden_size, hidden_size, np.float16, model_fp16),
        (1, batch_size, seq_len, intermediate_dimension, hidden_size, np.float16, model_fp16),
        (1, batch_size, seq_len, hidden_size, intermediate_dimension, np.float16, model_fp16),
        (batch_size, num_heads, seq_len, seq_len, int(hidden_size / num_heads), np.float16, model_fp16),
    ]

    # bert-base
    hidden_size = 768
    seq_len = 384
    num_heads = 12
    batch_size = 1
    intermediate_dimension = hidden_size * 4
    benchmark_cases += [
        (1, batch_size, seq_len, hidden_size, hidden_size, np.float32, model_fp32),
        (1, batch_size, seq_len, intermediate_dimension, hidden_size, np.float32, model_fp32),
        (1, batch_size, seq_len, hidden_size, intermediate_dimension, np.float32, model_fp32),
        (batch_size, num_heads, seq_len, seq_len, int(hidden_size / num_heads), np.float32, model_fp32),
        (1, batch_size, seq_len, hidden_size, hidden_size, np.float16, model_fp16),
        (1, batch_size, seq_len, intermediate_dimension, hidden_size, np.float16, model_fp16),
        (1, batch_size, seq_len, hidden_size, intermediate_dimension, np.float16, model_fp16),
        (batch_size, num_heads, seq_len, seq_len, int(hidden_size / num_heads), np.float16, model_fp16),
    ]

    return benchmark_cases


def benchmark_matmul(b1, b2, m, k, n, data_type, onnx_file):
    sess = create_session(onnx_file, "rocm")
    io_binding = create_io_binding(sess, b1, b2, m, k, n, data_type)
    time = benchmark(sess, io_binding)
    return time


def main():
    for (b1, b2, m, k, n, data_type, onnx_file) in create_benchmark_cases(): 
        time = benchmark_matmul(b1, b2, m, k, n, data_type, onnx_file)
        tflops = b1 * b2 * m * k * n * 2 / time / 1000000000
        print(f"(b1 b2 m k n) = ({b1} {b2} {m} {k} {n}) {time:7.4f} ms {tflops:4.2f} tflops")


if __name__ == "__main__":
    main()
