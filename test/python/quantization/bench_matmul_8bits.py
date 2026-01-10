# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
Benchmark performance of MatMulNBits for CUDA in ONNX Runtime.
"""

import argparse
import csv
import math
import statistics
from datetime import datetime

import torch
from onnx import TensorProto, helper

from onnxruntime import InferenceSession, SessionOptions, get_available_providers
from onnxruntime.transformers.io_binding_helper import CudaSession


class MatMulNBitsConfig:
    """
    Configuration for the MatMulNBits benchmark.
    """

    def __init__(
        self,
        bits: int,
        m: int,
        n: int,
        k: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        has_zero_point: bool = True,
        has_bias: bool = False,
        enable_cuda_graph: bool = False,
    ):
        """
        Initializes the MatMulNBitsConfig.

        Args:
            bits (int): Number of bits for quantization (e.g., 4, 8).
            m (int): The M dimension of the MatMul operation (batch size).
            n (int): The N dimension of the MatMul operation (output features).
            k (int): The K dimension of the MatMul operation (input features).
            block_size (int): The block size used for quantization along the K dimension.
            device (torch.device): The device to run the benchmark on (e.g., torch.device('cuda:0')).
            dtype (torch.dtype, optional): The data type for floating-point inputs and outputs. Defaults to torch.float16.
            has_zero_point (bool, optional): Whether the quantized weights have a zero point. Defaults to True.
            has_bias (bool, optional): Whether the MatMul operation includes a bias term. Defaults to False.
            enable_cuda_graph (bool, optional): Whether to enable CUDA graph capture. Defaults to False.
        """
        self.operator = "MatMulNBits"
        self.bits = bits
        self.m = m
        self.n = n
        self.k = k
        self.block_size = block_size
        self.has_zero_point = has_zero_point
        self.is_int_zero_point = True
        self.has_bias = has_bias
        # This script is specifically for CUDA benchmarking
        self.use_cuda = True
        self.dtype = dtype
        self.device = device
        self.enable_cuda_graph = enable_cuda_graph

        if self.k % self.block_size != 0:
            raise ValueError(f"K ({self.k}) must be divisible by block_size ({self.block_size}).")

        if self.bits not in [4, 8]:
            raise ValueError(f"Bits must be 4 or 8, but got {self.bits}.")

    def __repr__(self):
        """
        Returns a string representation of the configuration.
        """
        return (
            f"{self.operator}(bits={self.bits}, m={self.m}, n={self.n}, k={self.k}, block_size={self.block_size}, "
            f"dtype={self.dtype}, has_zero_point={self.has_zero_point}, "
            f"has_bias={self.has_bias}, enable_cuda_graph={self.enable_cuda_graph}) "
        )

    def shape_dict(self) -> dict[str, tuple]:
        """
        Returns a dictionary mapping input/output names to their shapes.

        Based on the MatMulNBits operator definition, input 'b' (weights) is
        quantized and structured as (N, K/block_size, block_size).
        Scales and zero_points are (N, K/block_size).
        """
        k_blocks = self.k // self.block_size
        shapes: dict[str, tuple] = {
            "output": (self.m, self.n),
            "a": (self.m, self.k),
            "b": (self.n, k_blocks, self.block_size),  # Quantized weights
            "scales": (self.n, k_blocks),
        }
        if self.has_zero_point:
            shapes["zero_points"] = (self.n, k_blocks)

        if self.has_bias:
            shapes["bias"] = (self.n,)

        return shapes

    def random_inputs(self, seed: int = 123) -> dict[str, torch.Tensor]:
        """
        Generates random input tensors based on the configuration.

        Args:
            seed (int, optional): Random seed for reproducibility. Use 0 for no seed. Defaults to 123.

        Returns:
            dict[str, torch.Tensor]: A dictionary of input tensors.
        """
        device = self.device
        dtype = self.dtype

        shape_dict = self.shape_dict()

        if seed > 0:
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

        feeds = {
            # 'a' is the activation tensor (M, K)
            "a": torch.empty(shape_dict["a"], device=device, dtype=dtype).normal_(mean=0, std=0.1),
            # 'b' is the quantized weight tensor (N, K/block_size, block_size)
            # Values should be within the [0, 2^bits - 1] range, but MatMulNBits takes UINT8.
            # The actual range used by the kernel depends on 'bits'.
            # Generating [0, 255] and letting the kernel handle the bit interpretation.
            "b": torch.randint(0, 256, shape_dict["b"], device=device, dtype=torch.uint8),
            # 'scales' is the scale tensor (N, K/block_size)
            "scales": torch.empty(shape_dict["scales"], device=device, dtype=dtype).normal_(mean=0, std=10.0),
        }

        if self.has_zero_point:
            # 'zero_points' is the zero point tensor (N, K/block_size)
            # Assuming is_int_zero_point is True, dtype is uint8.
            # Values should be within [0, 2^bits - 1]. Generating [0, 255].
            feeds["zero_points"] = torch.randint(0, 256, shape_dict["zero_points"], device=device, dtype=torch.uint8)

        if self.has_bias:
            # 'bias' is the bias tensor (N,)
            feeds["bias"] = torch.empty(shape_dict["bias"], device=device, dtype=dtype).normal_(mean=0, std=0.1)

        return feeds

    def get_input_output_names(self) -> tuple[list[str], list[str]]:
        """
        Returns the list of input and output names for the ONNX model.
        """
        inputs = ["a", "b", "scales"]
        if self.has_zero_point:
            inputs.append("zero_points")
        if self.has_bias:
            inputs.append("bias")

        outputs = ["output"]

        return inputs, outputs


def create_matmul_nbits_onnx_model(config: MatMulNBitsConfig) -> bytes:
    """
    Creates an ONNX model with a single MatMulNBits node.

    Args:
        config (MatMulNBitsConfig): The configuration for the MatMulNBits node.

    Returns:
        bytes: The serialized ONNX model.
    """
    input_names, output_names = config.get_input_output_names()

    float_type = TensorProto.FLOAT16 if config.dtype == torch.float16 else TensorProto.FLOAT
    nodes = [
        helper.make_node(
            "MatMulNBits",
            input_names,
            output_names,
            "MatMulNBits_0",  # Node name
            bits=config.bits,
            block_size=config.block_size,
            K=config.k,
            N=config.n,
            domain="com.microsoft",
        ),
    ]

    shape_dict = config.shape_dict()
    # Input types based on ONNX MatMulNBits definition. 'a', 'scales', 'bias' are float types.
    # 'b' and 'zero_points' are UINT8.
    inputs = [
        helper.make_tensor_value_info(
            input_name,
            TensorProto.UINT8 if input_name in ["b", "zero_points"] else float_type,
            list(shape_dict[input_name]),
        )
        for input_name in input_names
        if input_name
    ]

    outputs = [
        helper.make_tensor_value_info(output_name, float_type, list(shape_dict[output_name]))
        for output_name in output_names
        if output_name
    ]

    graph = helper.make_graph(
        nodes,
        "MatMulNBits_Graph",
        inputs,
        outputs,
    )

    model = helper.make_model(graph, producer_name="onnxruntime.benchmarks")

    return model.SerializeToString()


def create_ort_session(
    config: MatMulNBitsConfig,
    session_options: SessionOptions = None,
    use_tf32: bool = False,
) -> InferenceSession:
    """
    Creates an ONNX Runtime InferenceSession for the MatMulNBits model.

    Args:
        config (MatMulNBitsConfig): The configuration for the session.
        session_options (SessionOptions, optional): ONNX Runtime session options. Defaults to None.
        use_tf32 (bool, optional): Whether to enable TF32 mode on CUDA. Defaults to False.

    Returns:
        InferenceSession: The created ONNX Runtime InferenceSession.
    """
    onnx_model_str = create_matmul_nbits_onnx_model(config)

    # Assuming CUDA execution provider for this script
    if "CUDAExecutionProvider" not in get_available_providers():
        raise RuntimeError("CUDAExecutionProvider is not available.")

    device_id = config.device.index if isinstance(config.device, torch.device) else 0
    provider_options = CudaSession.get_cuda_provider_options(device_id, config.enable_cuda_graph)
    provider_options["use_tf32"] = int(use_tf32)
    # Include CPU as fallback, though performance sensitive tests should target CUDA
    providers = [("CUDAExecutionProvider", provider_options), "CPUExecutionProvider"]

    ort_session = InferenceSession(onnx_model_str, session_options, providers=providers)
    return ort_session


def create_session(
    config: MatMulNBitsConfig, session_options: SessionOptions = None, use_tf32: bool = False
) -> CudaSession:
    """
    Creates a CudaSession with pre-allocated buffers.

    Args:
        config (MatMulNBitsConfig): The configuration for the session.
        session_options (SessionOptions, optional): ONNX Runtime session options. Defaults to None.
        use_tf32 (bool, optional): Whether to enable TF32 mode on CUDA. Defaults to False.

    Returns:
        CudaSession: The created CudaSession.
    """
    ort_session = create_ort_session(config, session_options, use_tf32=use_tf32)
    cuda_session = CudaSession(ort_session, config.device, config.enable_cuda_graph)
    shape_dict = config.shape_dict()
    cuda_session.allocate_buffers(shape_dict)
    return cuda_session


def measure_latency(cuda_session: CudaSession, input_dict: dict[str, torch.Tensor]) -> float:
    """
    Measures the inference latency of a single run using CUDA events.

    Args:
        cuda_session (CudaSession): The CudaSession to benchmark.
        input_dict (dict[str, torch.Tensor]): The input data for inference.

    Returns:
        float: The latency in seconds.
    """
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Synchronize before starting the timed event
    torch.cuda.synchronize()
    start_event.record()

    cuda_session.infer(input_dict, synchronize=False)  # Infer without synchronizing inside

    end_event.record()
    # Synchronize after the timed event to get accurate duration
    torch.cuda.synchronize()

    latency_ms = start_event.elapsed_time(end_event)  # Latency in milliseconds
    return latency_ms / 1000.0  # Return latency in seconds


def flops(m: int, n: int, k: int) -> int:
    """
    Calculates the number of floating-point operations (FLOPs) for a MatMul (M, K) @ (K, N).
    """
    # MatMul (M, K) @ (K, N) performs M*N*K multiplications and M*N*(K-1) additions.
    # For simplicity, often approximated as 2 * M * N * K.
    return 2 * m * n * k


def tflops_per_second(flop: int, time_seconds: float) -> float:
    """
    Calculates TFLOPS (Tera Floating-point Operations Per Second).

    Args:
        flop (int): The number of FLOPs.
        time_seconds (float): The time taken in seconds.

    Returns:
        float: The TFLOPS/second, or 0.0 if time is non-positive or NaN.
    """
    if time_seconds > 0 and not math.isnan(time_seconds):
        return (flop / time_seconds) / 1e12
    return 0.0


def get_test_configs(args: argparse.Namespace) -> list[tuple]:
    """
    Generates a list of test configurations (m, n, k, block_size, bits).

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        list[tuple]: A list of tuples, each representing a configuration (m, n, k, block_size, bits).
    """
    if args.phi4:
        configs = []
        # Predefined configurations inspired by large language models.
        phi_weight_shapes = [
            # (N, K) of MatMul weights in phi4-mini model.
            (5120, 3072),
            (8192, 3072),
            (3072, 8192),
            (200064, 3072),
        ]

        for bits in [4, 8]:
            for m in [1, 256, 1024]:
                for block_size in [32, 128]:
                    for n, k in phi_weight_shapes:
                        if k % block_size == 0:
                            configs.append((m, n, k, block_size, bits))

        configs = sorted(configs)

    else:
        # Single configuration from command line arguments
        configs = [
            (
                args.m,
                args.n,
                args.k,
                args.block_size,
                args.bits,
            ),
        ]

    return configs


def get_compute_capability() -> str:
    """
    Gets the CUDA compute capability of the current device.

    Returns:
        str: The compute capability in 'major.minor' format, or 'N/A' if CUDA is not available.
    """
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return f"{major}.{minor}"
    return "N/A"


def run_tflops_test(
    csv_writer: csv.DictWriter,
    args: argparse.Namespace,
):
    """
    Runs the TFLOPS benchmark for the specified configurations.

    Args:
        csv_writer (csv.DictWriter): CSV writer object to write results.
        args (argparse.Namespace): Parsed command-line arguments.
    """
    assert torch.cuda.is_available()
    assert "CUDAExecutionProvider" in get_available_providers()

    enable_cuda_graph: bool = not args.disable_cuda_graph
    intra_op_num_threads: int = args.intra_op_num_threads
    repeats: int = args.repeats
    num_warmup_runs: int = args.warmup_runs

    device_id = torch.cuda.current_device()
    device = torch.device("cuda", device_id)

    configs = get_test_configs(args)

    # Print header to console
    print("-" * 120)
    print(
        f"Benchmarking MatMulNBits on {torch.cuda.get_device_name(device_id)} (Compute Capability: {get_compute_capability()})"
    )
    print("-" * 120)
    # Updated header format to match CSV columns and improve readability
    print(
        f"{'CUDA Graph':<12} | {'M':<8} | {'N':<8} | {'K':<8} | {'Bits':<6} | {'Block Size':<10} | {'Threads':<8} | {'Latency (us)':<15} | {'StdDev (us)':<12} | {'TFLOPS':<10}"
    )
    print("-" * 120)

    for m, n, k, block_size, bits in configs:
        config_str = f"(m={m}, n={n}, k={k}, block_size={block_size}, bits={bits})"
        try:
            config = MatMulNBitsConfig(
                bits=bits,
                m=m,
                n=n,
                k=k,
                block_size=block_size,
                device=device,
                dtype=torch.float16,  # Assuming float16 for CUDA performance tests
                has_zero_point=True,  # Assuming zero point for MatMulNBits
                has_bias=False,  # Not including bias in these benchmarks by default
                enable_cuda_graph=enable_cuda_graph,
            )

            sess_options = SessionOptions()
            sess_options.intra_op_num_threads = intra_op_num_threads
            session = create_session(config, sess_options, use_tf32=args.use_tf32)
            input_dict = config.random_inputs()

            # Warm-up runs
            for _ in range(num_warmup_runs):
                measure_latency(session, input_dict)  # Latency is measured, but result is discarded
            torch.cuda.synchronize()  # Ensure warm-up completes before timing

            # Measure repeats
            latency_list_seconds = []
            for _ in range(repeats):
                latency = measure_latency(session, input_dict)
                latency_list_seconds.append(latency)

            # Explicitly delete session to release GPU memory before processing results
            del session

            if not latency_list_seconds:
                average_latency_seconds = float("nan")
                stddev_latency_seconds = float("nan")
            else:
                average_latency_seconds = statistics.mean(latency_list_seconds)
                stddev_latency_seconds = statistics.stdev(latency_list_seconds) if repeats > 1 else 0.0

            # compute TFLOPS per second
            speed = tflops_per_second(
                flops(m, n, k),
                average_latency_seconds,
            )

            average_latency_us = average_latency_seconds * 1_000_000
            stddev_latency_us = stddev_latency_seconds * 1_000_000

            row = {
                "use_gpu": True,  # Hardcoded to True as this is a CUDA benchmark script
                "cuda_graph": enable_cuda_graph,
                "m": m,
                "n": n,
                "k": k,
                "bits": bits,
                "block_size": block_size,
                "intra_op_num_threads": intra_op_num_threads,
                "latency_seconds": average_latency_seconds,
                "latency_microseconds": average_latency_us,
                "latency_stddev_seconds": stddev_latency_seconds,
                "latency_stddev_microseconds": stddev_latency_us,
                "tflops": speed,
            }
            csv_writer.writerow(row)

            speed_str = f"{speed:.3f}" if speed is not None and not math.isnan(speed) else "NA"
            # Print results to console
            print(
                f"{enable_cuda_graph!s:<12} | {m:<8} | {n:<8} | {k:<8} | {bits:<6} | {block_size:<10} | {intra_op_num_threads:<8} | {average_latency_us:<15.1f} | {stddev_latency_us:<12.1f} | {speed_str:<10}"
            )

        except ValueError as e:
            print(f"Skipping invalid configuration {config_str} - {e}")
            # Optionally write a skipped row to CSV? For now, just skip.
            continue
        except Exception as e:
            print(f"Error running benchmark for config {config_str}: {e}")
            # Write a row with error info to CSV? Or just skip? Let's just skip for now.
            continue

    print("-" * 120)


def run_tflops_tests(args):
    """
    Sets up the CSV file and runs the benchmark tests.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    csv_filename = "{}{}.csv".format(
        args.csv_filename_prefix,
        datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    print(f"Writing results to {csv_filename}")

    # Use 'w' mode to create a new file for each run
    with open(csv_filename, mode="w", newline="") as csv_file:
        column_names = [
            "use_gpu",
            "cuda_graph",
            "m",
            "n",
            "k",
            "bits",
            "block_size",
            "intra_op_num_threads",
            "latency_seconds",
            "latency_microseconds",
            "latency_stddev_seconds",
            "latency_stddev_microseconds",
            "tflops",
        ]
        csv_writer = csv.DictWriter(csv_file, fieldnames=column_names)
        csv_writer.writeheader()

        # The script is specifically for CUDA now
        run_tflops_test(csv_writer, args)


def _parse_arguments():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark MatMulNBits performance for ONNX Runtime CUDAExecutionProvider. "
        "Supports both single configurations and predefined Phi-like shapes."
    )

    parser.add_argument(
        "--disable_cuda_graph",
        action="store_true",
        help="Disable CUDA graph capture in ONNX Runtime.",
    )

    parser.add_argument(
        "--intra_op_num_threads",
        type=int,
        choices=[0, 1, 2, 4, 8, 16],  # Common thread counts, 0 means default.
        default=0,
        help="intra_op_num_threads for ONNX Runtime session options. 0 means default.",
    )

    # Arguments for a single configuration
    parser.add_argument(
        "--m",
        type=int,
        default=1,
        help="The M dimension of the MatMul operation (batch size). Used when --phi4 is not set.",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=200064,  # This default seems unusually large, but kept from original.
        help="The N dimension of the MatMul operation (output features). Used when --phi4 is not set.",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=3072,
        help="The K dimension of the MatMul operation (input features). Used when --phi4 is not set.",
    )

    parser.add_argument(
        "--block_size",
        type=int,
        default=32,
        help="The block size used for quantization along the K dimension. Used when --phi4 is not set.",
    )

    parser.add_argument(
        "--bits",
        type=int,
        choices=[4, 8],
        default=8,
        help="Number of bits for quantization (4 or 8). Used when --phi4 is not set.",
    )

    parser.add_argument(
        "-r",
        "--repeats",
        type=int,
        default=10000,  # Default repeats for measurement
        help="Number of repeats for performance measurement of each configuration.",
    )

    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=10,  # Default warmup runs
        help="Number of warm-up runs before performance measurement for each configuration.",
    )

    parser.add_argument(
        "--phi4",
        action="store_true",
        help="Run a predefined set of configurations based on Phi4-mini model shapes, overriding --m, --n, --k, --block_size, --bits.",
    )

    parser.add_argument(
        "--csv_filename_prefix",
        type=str,
        default="benchmark_matmulnbits_cuda_",
        help="Prefix for the output CSV filename.",
    )

    parser.add_argument(
        "--use_tf32",
        action="store_true",
        help="Enable TF32 mode on CUDA. May affect precision and performance on compatible GPUs (Ampere+).",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = _parse_arguments()
    print(f"Parsed arguments: {args}")

    # Check for CUDA availability early
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This script requires a CUDA-enabled GPU.")
        exit(1)

    if "CUDAExecutionProvider" not in get_available_providers():
        print("Error: CUDAExecutionProvider is not available in your ONNX Runtime installation.")
        print("Please ensure you have installed the onnxruntime-gpu package (`pip install onnxruntime-gpu`).")
        exit(1)

    # Check if k is divisible by block_size for the single config case
    if not args.phi4 and args.k % args.block_size != 0:
        print(
            f"Error: For the single configuration (--phi4 not set), K ({args.k}) must be divisible by block_size ({args.block_size})."
        )
        exit(1)

    run_tflops_tests(args)
