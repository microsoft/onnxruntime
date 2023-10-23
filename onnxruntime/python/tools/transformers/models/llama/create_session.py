import onnxruntime as ort

import argparse
import gc
import itertools
import logging
import os
import sys
import time

import numpy as np
import onnx
import psutil
from benchmark_helper import setup_logger

from transformers import LlamaConfig

import onnxruntime as ort
from onnxruntime.transformers.benchmark_helper import measure_memory

logger = logging.getLogger(__name__)


# Create past_key_values
def get_sample_past_kv_inputs(config: LlamaConfig, batch_size: int, past_seq_len: int, use_fp16: bool):
    num_heads, head_size = config.num_key_value_heads, config.hidden_size // config.num_key_value_heads
    np_dtype = np.float16 if use_fp16 else np.float32
    rng = np.random.default_rng()
    past_kv = [
        (
            rng.standard_normal(size=(batch_size, num_heads, past_seq_len, head_size)).astype(np_dtype),
            rng.standard_normal(size=(batch_size, num_heads, past_seq_len, head_size)).astype(np_dtype),
        )
        for _ in range(config.num_hidden_layers)
    ]
    return past_kv


# Get position_ids from attention_mask
def get_position_ids(attention_mask, use_past_kv: bool):
    position_ids = attention_mask.cumsum(-1) - 1
    # print(position_ids)
    # position_ids.masked_fill_(attention_mask == 0, 1)
    if use_past_kv:
        position_ids = np.expand_dims(position_ids[:, -1], -1)
    return position_ids


# Inputs for all passes with past_key_values
def get_merged_sample_with_past_kv_inputs(
    config: LlamaConfig,
    batch_size: int,
    seq_len: int,
    past_seq_len: int,
    use_fp16: bool = False,
    return_dict: bool = False,
):
    input_ids = np.random.randint(low=0, high=config.vocab_size, size=(batch_size, seq_len), dtype=np.int64)
    attention_mask = np.ones((batch_size, past_seq_len + seq_len), dtype=np.int64)
    # position_ids is of shape (batch_size, seq_len) for prompt generation, (batch_size, 1) for token generation
    position_ids = get_position_ids(attention_mask, use_past_kv=(past_seq_len != 0))
    past_kv = get_sample_past_kv_inputs(config, batch_size, past_seq_len, use_fp16)

    if not return_dict:
        return (input_ids, attention_mask, position_ids, past_kv)

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_values": past_kv,
    }
    return inputs


# Convert list of past_kv to dict of past_key and past_value
def flatten_past_kv_inputs(past_key_values, use_fp16: bool):
    past_kv = {}
    for i, (past_k, past_v) in enumerate(past_key_values):
        past_kv[f"past_key_values.{i}.key"] = past_k
        past_kv[f"past_key_values.{i}.value"] = past_v
    return past_kv


# Format numpy inputs to ONNX Runtime inputs
def convert_inputs_for_ort(
    pt_inputs: dict,
    use_fp16: bool,
    use_buffer_share: bool = False,
    past_seq_len: int = 0,
    max_seq_len: int = 2048,
    device: str = "",
    device_id: int = -1,
):
    ort_inputs = {}
    for k, v in pt_inputs.items():
        if k == "past_key_values":
            ort_inputs.update(flatten_past_kv_inputs(v, use_fp16))
        elif k == "attention_mask" and use_fp16 and use_buffer_share:
            # Skip because FP16 model has GroupQueryAttention, uses buffer sharing,
            # and GQA supports a causal mask by default

            # Instead, add the past sequence length input for GQA
            ort_inputs["past_sequence_length"] = np.array([past_seq_len], dtype=np.int64)
        else:
            ort_inputs[k] = v

    # Enable past-present-share-buffer by using device memory directly
    if use_buffer_share and device != "" and device != "cpu" and device_id > -1:
        for k, v in ort_inputs.items():
            new_v = v
            # Allocate new buffers with max_sequence_length for GQA
            if "cache" in k or "past_key_values" in k:
                # Copy v (BxSxPxH) into new_v (BxSxMxH)
                batch_size, num_heads, _, head_size = v.shape
                new_v = np.zeros((batch_size, num_heads, max_seq_len, head_size), dtype=v.dtype)
                new_v[:batch_size, :num_heads, :past_seq_len, :head_size] = v
                # print(f"{k}:{new_v.shape}")
            ort_inputs[k] = ort.OrtValue.ortvalue_from_numpy(new_v, device_type=device, device_id=device_id)

    return ort_inputs


def get_inputs(args: argparse.Namespace):
    init_inputs, iter_inputs = None, None

    # For past_present_share_buffer:
    # Set max_seq_len to 2048 for Hugging Face model since that is the default value
    # Set max_seq_len to 2048 for Microsoft model since that is the max value currently supported
    max_seq_len = 2048

    # Microsoft export from convert_to_onnx
    init_inputs = get_merged_sample_with_past_kv_inputs(
        args.config,
        args.batch_size,
        seq_len=args.sequence_length,
        past_seq_len=0,
        use_fp16=args.use_fp16,
        return_dict=True,
    )
    iter_inputs = get_merged_sample_with_past_kv_inputs(
        args.config,
        args.batch_size,
        seq_len=1,
        past_seq_len=args.sequence_length,
        use_fp16=args.use_fp16,
        return_dict=True,
    )
    init_inputs = convert_inputs_for_ort(
        init_inputs,
        use_fp16=args.use_fp16,
        use_buffer_share=args.past_present_share_buffer,
        past_seq_len=0,
        max_seq_len=max_seq_len,
        device=args.device,
        device_id=args.device_id,
    )
    iter_inputs = convert_inputs_for_ort(
        iter_inputs,
        use_fp16=args.use_fp16,
        use_buffer_share=args.past_present_share_buffer,
        past_seq_len=args.sequence_length,
        max_seq_len=max_seq_len,
        device=args.device,
        device_id=args.device_id,
    )

    return init_inputs, iter_inputs


def get_model(args: argparse.Namespace):
    model, sess_options = None, None
    start_time, end_time = None, None

    sess_options = ort.SessionOptions()
    sess_options.enable_profiling = args.profile
    if args.verbose:
        sess_options.log_verbosity_level = 1
        sess_options.log_severity_level = 1

    logger.info(f"Loading model from {args.ort_model_path}")
    start_time = time.time()
    model = ort.InferenceSession(
        args.ort_model_path,
        sess_options,
        providers=[args.execution_provider],
    )
    end_time = time.time()

    logger.info(f"Loaded model in {end_time - start_time} s")
    return model


def time_fn(args, fn, inputs):
    # Warm up
    warmup_range = (
        range(args.warmup_runs)
        if args.benchmark_type in {"ort-msft", "ort-convert-to-onnx"}
        else trange(args.warmup_runs, file=sys.stdout, desc="Warm up")
    )

    if args.verbose:
        outputs = fn(inputs)
        logger.info(outputs)

    for _ in warmup_range:
        fn(inputs)

    # Benchmark
    start_time = time.time()

    print(args.num_runs)
    for _ in range(args.num_runs):
        fn(inputs)

    end_time = time.time()

    latency = (end_time - start_time) / args.num_runs
    throughput = args.batch_size / latency

    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Sequence Length: {args.sequence_length}")
    logger.info(f"Latency: {latency} s")
    logger.info(f"Throughput: {throughput} tps")
    return


def measure_fn(args, fn, inputs):
    # Measure CPU usage
    pid = os.getpid()
    process = psutil.Process(pid)
    process.cpu_percent(interval=0.1)

    # fn(inputs)
    # logger.info(f"CPU usage: {process.cpu_percent(interval=None)}%")

    # Measure memory usage
    gc.collect()
    measure_memory(is_gpu=(args.device != "cpu"), func=lambda: fn(inputs))

    # Flush output so memory usage is printed
    sys.stdout.flush()


def run_inference(args, init_inputs, iter_inputs, model):
    def prepare_ort_inputs(inputs):
        # Check that all model inputs will be provided
        model_inputs = set(map(lambda model_input: model_input.name, model.get_inputs()))
        user_inputs = set(inputs.keys())
        missing_inputs = model_inputs - user_inputs
        if len(missing_inputs):
            logger.error(f"The following model inputs are missing: {missing_inputs}")
            raise Exception("There are missing inputs to the model. Please add them and try again.")

        # Remove unnecessary inputs from model inputs
        unnecessary_inputs = user_inputs - model_inputs
        if len(unnecessary_inputs):
            for unnecessary_input in unnecessary_inputs:
                logger.info(f"Removing unnecessary input '{unnecessary_input}' from user provided inputs")
                del inputs[unnecessary_input]

        # Add IO bindings for non-CPU execution providers
        if args.device != "cpu":
            io_binding = model.io_binding()

            for k, v in inputs.items():
                if args.past_present_share_buffer:
                    # Bind all OrtValue inputs to device
                    io_binding.bind_ortvalue_input(k, v)
                else:
                    io_binding.bind_cpu_input(k, v)

            for output in model.get_outputs():
                name = output.name
                if args.past_present_share_buffer and ("out" in name or "present" in name):
                    # Bind present KV cache outputs to OrtValue with buffer sharing
                    io_binding.bind_ortvalue_output(
                        name, inputs[name.replace("out", "cache").replace("present", "past_key_values")]
                    )
                else:
                    io_binding.bind_output(name, device_type=args.device, device_id=args.device_id)

            return io_binding

        return inputs

    def with_io_binding(io_binding):
        # Inference pass with IO binding
        model.run_with_iobinding(io_binding)

    def without_io_binding(inputs):
        # Inference pass without IO binding
        outputs = model.run(None, inputs)
        return outputs

    generate_fn = with_io_binding if args.device != "cpu" else without_io_binding

    # ORT evaluations
    logger.info("\nEvaluating `model(inputs)` step to get past_key_values")
    # ort_init_inputs = prepare_ort_inputs(init_inputs)

    # time_fn(args, generate_fn, ort_init_inputs)
    # measure_fn(args, generate_fn, ort_init_inputs)

    logger.info("\nEvaluating `model(inputs)` step with past_key_values")
    ort_iter_inputs = prepare_ort_inputs(iter_inputs)
    time_fn(args, generate_fn, ort_iter_inputs)
    measure_fn(args, generate_fn, ort_iter_inputs)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-bt",
        "--benchmark-type",
        type=str,
        required=True,
        choices=["hf-pt-eager", "hf-pt-compile", "hf-ort", "ort-msft", "ort-convert-to-onnx"],
    )
    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
        required=True,
        help="Hugging Face name of model (e.g. 'meta-llama/Llama-2-7b-hf')",
    )
    parser.add_argument(
        "-a", "--auth", default=False, action="store_true", help="Use Hugging Face authentication token to access model"
    )

    # Args for choosing the model
    parser.add_argument(
        "-p",
        "--precision",
        required=True,
        type=str,
        default="fp32",
        choices=["int4", "int8", "fp16", "fp32"],
        help="Precision for model. For ONNX models, the model's precision should be set before running this script.",
    )
    parser.add_argument(
        "--ort-model-path",
        type=str,
        default="",
        help="Path to ONNX model",
    )

    # Args for running and evaluating the model
    parser.add_argument(
        "-b",
        "--batch-sizes",
        default="1 2",
    )
    parser.add_argument(
        "-s",
        "--sequence-lengths",
        default="8 16 32 64 128 256 512",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda", "rocm"],
    )
    parser.add_argument("-id", "--device-id", type=int, default=0)
    parser.add_argument("-w", "--warmup-runs", type=int, default=5)
    parser.add_argument("-n", "--num-runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2)

    # Args for decoding logic
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--num-return-sequences", type=int, default=1)

    # Args for accessing detailed info
    parser.add_argument("--profile", default=False, action="store_true")
    parser.add_argument(
        "--pt-filter-by", type=str, default="self_cpu_time_total", help="What to filter PyTorch profiler by"
    )
    parser.add_argument("--pt-num-rows", type=int, default=1000, help="Number of rows for PyTorch profiler to display")
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--log-folder", type=str, default=os.path.join("."), help="Folder to cache log files")

    args = parser.parse_args()

    # Set seed properties
    np.random.seed(args.seed)

    # Set runtime properties
    if "ort" in args.benchmark_type:
        setattr(args, "execution_provider", f"{args.device.upper()}ExecutionProvider")  # noqa: B010
        if args.execution_provider == "CUDAExecutionProvider":
            args.execution_provider = (args.execution_provider, {"device_id": args.device_id})
        elif args.execution_provider == "ROCMExecutionProvider":
            args.execution_provider = (args.execution_provider, {"device_id": args.device_id})
            args.device = "cuda"

    # Check that paths have been specified for any benchmarking with ORT
    if args.benchmark_type == "hf-ort":
        assert args.hf_ort_dir_path, "Please specify a path to `--hf-ort-dir-path`"
    if args.benchmark_type in {"ort-msft", "ort-convert-to-onnx"}:
        assert args.ort_model_path, "Please specify a path to `--ort-model-path`"

    args.batch_sizes = args.batch_sizes.split(" ")
    args.sequence_lengths = args.sequence_lengths.split(" ")

    # Use FP32 precision for FP32 and INT8 models, use FP16 precision for FP16 and INT4 models
    args.precision = "fp32" if args.precision in {"int8", "fp32"} else "fp16"

    # Check that only one (batch_size, sequence_length) combination is set for profiling
    if args.profile:
        assert (
            len(args.batch_sizes) == 1 and len(args.sequence_lengths) == 1
        ), "Please provide only one (batch_size, sequence_length) combination for profiling"

    return args


def main():
    args = get_args()
    setup_logger(args.verbose)
    logger.info(args.__dict__)

    config = LlamaConfig.from_pretrained(args.model_name)
    target_device = f"cuda:{args.device_id}" if args.device != "cpu" else args.device
    use_fp16 = args.precision == "fp16"

    setattr(args, "config", config)  # noqa: B010
    setattr(args, "target_device", target_device)  # noqa: B010
    setattr(args, "use_fp16", use_fp16)  # noqa: B010

    # Get model and model info
    model = get_model(args)

    # Check if past_present_share_buffer can be enabled (only for FP16 models with GQA)
    onnx_model = onnx.load_model(args.ort_model_path, load_external_data=False)
    gqa_nodes = list(filter(lambda node: node.op_type == "GroupQueryAttention", onnx_model.graph.node))

    use_buffer_share = use_fp16 and len(gqa_nodes) > 0 and args.device != "cpu"
    setattr(args, "past_present_share_buffer", use_buffer_share)  # noqa: B010

    # Measure prompt cost (init_inputs) and generated token cost (iter_inputs)
    for batch_size, sequence_length in itertools.product(args.batch_sizes, args.sequence_lengths):
        logger.info(f"\nBatch size = {batch_size} and sequence length = {sequence_length}...")
        setattr(args, "batch_size", int(batch_size))  # noqa: B010
        setattr(args, "sequence_length", int(sequence_length))  # noqa: B010

        init_inputs, iter_inputs = get_inputs(args)
        run_inference(args, init_inputs, iter_inputs, model)


if __name__ == "__main__":
    main()
