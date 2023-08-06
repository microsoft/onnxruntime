########################
# Benchmark LLaMA model
########################

import argparse
import datetime
import gc
import logging
import os
import sys
import time

import numpy as np
import psutil
import torch
from optimum.onnxruntime import ORTModelForCausalLM
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm import trange
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

import onnxruntime as ort
from onnxruntime.transformers.benchmark_helper import measure_memory

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from benchmark_helper import setup_logger  # noqa: E402

PRECISION = {
    "fp32": (torch.float32, np.float32),
    "fp16": (torch.float16, np.float16),
    "int8": (torch.int8, np.int8),
}

logger = logging.getLogger(__name__)


def get_position_ids(attention_mask, use_past_key_values=True):
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    if use_past_key_values:
        position_ids = position_ids[:, -1].unsqueeze(-1)
    return position_ids


def get_inputs(args, config, tokenizer):
    prompt = [args.prompt for _ in range(args.batch_size)]

    if args.benchmark_type in {"hf-pt", "hf-pt2", "hf-ort"}:
        tokenizer_outputs = tokenizer(prompt, return_tensors="pt").to(args.device)

        inputs = {
            "input_ids": tokenizer_outputs.input_ids,
            "attention_mask": tokenizer_outputs.attention_mask,
            "max_new_tokens": args.max_length,
        }
        if args.benchmark_type == "hf-ort":
            inputs["position_ids"] = get_position_ids(inputs["attention_mask"])

        exclude_list = inputs.keys()

    elif args.benchmark_type == "ort":
        # Microsoft export from https://github.com/microsoft/Llama-2-Onnx
        batch_size = args.batch_size
        max_seq_len = args.sequence_length
        head_size = config.hidden_size // config.num_attention_heads
        inputs = {
            "x": np.random.rand(batch_size, 1, config.hidden_size),
            "attn_mask": -10000.0
            * np.triu(np.ones((batch_size, config.hidden_size // 2, config.hidden_size // 2)), k=1),
            "k_cache": np.random.rand(
                batch_size, config.num_hidden_layers, max_seq_len, config.num_attention_heads, head_size
            ),
            "v_cache": np.random.rand(
                batch_size, config.num_hidden_layers, max_seq_len, config.num_attention_heads, head_size
            ),
            "pos": np.array(max_seq_len, dtype=np.int64),
        }

        exclude_list = ["pos"]

    else:
        raise Exception("Unable to auto-detect inputs for provided model")

    return set_inputs(args, inputs, exclude_list)


def set_inputs(args, input_dict, exclude_list):
    # Cast certain inputs to another dtype
    precision_dest = "fp32" if args.precision == "int8" else args.precision

    for k, v in input_dict.items():
        if k in exclude_list:
            continue

        if isinstance(v, torch.Tensor):
            input_dict[k] = v.to(PRECISION[precision_dest][0])
        elif isinstance(v, np.ndarray):
            input_dict[k] = v.astype(PRECISION[precision_dest][1])

    return input_dict


def get_vars(args):
    config = LlamaConfig.from_pretrained(args.model_name)
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name)
    inputs = get_inputs(args, config, tokenizer)
    model = None

    # There are multiple sources that the model could come from:
    # 1) Benchmark LLaMA from unofficial source on Hugging Face
    # 2) Benchmark LLaMA from official source on Hugging Face, which requires an authentication token
    # 3) Benchmark LLaMA from local download of model
    torch_dtype = PRECISION[args.precision][0] if args.precision != "int8" else PRECISION["fp32"][0]
    target_device = f"{args.device}:{args.device_id}" if args.device == "cuda" else args.device

    if args.benchmark_type in {"hf-pt", "hf-pt2"}:
        source = args.hf_pt_model_path if args.hf_pt_model_path else args.model_name
        start_time = time.time()
        model = LlamaForCausalLM.from_pretrained(
            source,
            torch_dtype=torch_dtype,
            use_auth_token=args.auth,
            use_cache=True,
        ).to(target_device)
        end_time = time.time()

    elif args.benchmark_type == "hf-ort":
        # Optimum export
        source = args.hf_ort_model_path if args.hf_ort_model_path else args.model_name
        start_time = time.time()
        model = ORTModelForCausalLM.from_pretrained(
            source,
            use_auth_token=args.auth,
            use_io_binding=True,
        ).to(target_device)
        end_time = time.time()

    elif args.benchmark_type == "ort":
        # Microsoft export
        sess_options = ort.SessionOptions()
        sess_options.enable_profiling = args.profile
        if args.verbose:
            sess_options.log_verbosity_level = 3
            sess_options.log_severity_level = 1
        logger.info(f"Loading model from {args.ort_model_path}")
        start_time = time.time()
        model = ort.InferenceSession(args.ort_model_path, providers=[args.execution_provider])
        end_time = time.time()

    else:
        raise Exception(f"Cannot recognize {args.benchmark_type}")

    logger.info(f"Loaded model in {end_time - start_time} s")

    if "pt2" in args.benchmark_type:
        model = torch.compile(model)

    return inputs, tokenizer, model


def time_fn(args, fn, inputs):
    init_range = range(args.warmup_runs) if args.benchmark_type == "ort" else trange(args.warmup_runs, file=sys.stdout)
    inf_range = range(args.num_runs) if args.benchmark_type == "ort" else trange(args.num_runs, file=sys.stdout)

    # Warm up
    for _ in init_range:
        outputs = fn(inputs)

    if args.verbose:
        logger.info(outputs)

    # Benchmark
    if args.device == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()

    for _ in inf_range:
        outputs = fn(inputs)

    if args.device == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()

    latency = (end_time - start_time) / args.num_runs
    throughput = args.batch_size / latency
    metrics = (args.batch_size, latency, throughput)

    # Newline print after trange in order to print metrics on new line without progress bar on same line
    if args.benchmark_type != "ort":
        logger.info("\n")

    return outputs, metrics


def run_hf_inference(args, inputs, tokenizer, model):
    def gen_and_dec():
        predicted_ids = model.generate(**inputs)
        transcription = []
        for bs in range(args.batch_size):
            for rs in range(args.num_return_sequences):
                transcription.append(
                    tokenizer.batch_decode(
                        predicted_ids[bs * args.num_return_sequences + rs], skip_special_tokens=True
                    )[0]
                )

    if args.profile:
        # Profile kernels
        with profile(  # noqa: SIM117
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True
        ) as prof:
            with record_function("model_inference"):
                gen_and_dec()
        prof_data = prof.key_averages(group_by_stack_n=5).table(sort_by=args.pt_filter_by, row_limit=args.pt_num_rows)

        # Filename format example: "hf_pt2_gen_and_dec_<current-time>.txt"
        filename = f"{args.benchmark_type.lower().replace(' + ', '_')}_gen_and_dec_{datetime.datetime.now():%Y-%m-%d_%H:%M:%S}.txt"
        with open(filename, "w") as f:
            f.write(prof_data)

        # Measure CPU usage
        pid = os.getpid()
        process = psutil.Process(pid)
        process.cpu_percent(interval=0.1)

        gen_and_dec()
        logger.info(f"CPU percentage = {process.cpu_percent(interval=None)}%")

        # Measure memory usage
        gc.collect()
        torch.cuda.empty_cache()
        measure_memory(is_gpu=(args.device == "cuda"), func=lambda: gen_and_dec())

        return

    logger.info("Evaluating `model.generate` step")
    generate_fn = lambda inputs: model.generate(**inputs)  # noqa: E731
    predicted_ids, metrics = time_fn(args, generate_fn, inputs)
    logger.info(f"Batch size = {metrics[0]}, latency = {metrics[1]} s, throughput = {metrics[2]} qps")

    logger.info("Evaluating `tokenizer.batch_decode` step")
    transcription_fn = lambda pred_ids: tokenizer.batch_decode(  # noqa: E731
        pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    transcription, metrics = time_fn(args, transcription_fn, predicted_ids)
    logger.info(f"Batch size = {metrics[0]}, latency = {metrics[1]} s, throughput = {metrics[2]} qps")


def run_ort_inference(args, inputs, model):
    # Check that all model inputs will be provided
    model_inputs = set(map(lambda model_input: model_input.name, model.get_inputs()))
    user_inputs = set(inputs.keys())
    missing_inputs = model_inputs - user_inputs
    if len(missing_inputs):
        logger.error(f"The following model inputs are missing: {missing_inputs}")
        raise Exception("There are missing inputs to the model. Please add them to `get_inputs`.")

    # Remove unnecessary inputs from model inputs
    unnecessary_inputs = user_inputs - model_inputs
    if len(unnecessary_inputs):
        for unnecessary_input in unnecessary_inputs:
            logger.info(f"Removing unnecessary input '{unnecessary_input}' from user provided inputs")
            del inputs[unnecessary_input]

    # Add IO binding for non-CPU inputs
    if args.device != "cpu":
        io_binding = model.io_binding()
        for k, v in inputs.items():
            io_binding.bind_cpu_input(k, v)
        for output in model.get_outputs():
            io_binding.bind_output(output.name)

    if args.profile:
        # Measure CPU usage
        pid = os.getpid()
        process = psutil.Process(pid)
        process.cpu_percent(interval=0.1)

        model.run(None, inputs) if args.device == "cpu" else model.run_with_iobinding(io_binding)
        logger.info(f"CPU percentage = {process.cpu_percent(interval=None)}%")

        # Turn profiling off to stop generating logs
        args.profile = False
        model.end_profiling()

        # Measure memory usage
        gc.collect()
        torch.cuda.empty_cache()
        measure_memory(is_gpu=(args.device == "cuda"), func=lambda: model.run(None, inputs))

        return

    if args.device == "cpu":
        generate_fn = lambda inputs: model.run(None, inputs)  # noqa: E731
        outputs, metrics = time_fn(args, generate_fn, inputs)
    else:
        generate_fn = lambda io_binding: model.run_with_iobinding(io_binding)  # noqa: E731
        outputs, metrics = time_fn(args, generate_fn, io_binding)
    logger.info(f"Batch size = {metrics[0]}, latency = {metrics[1]} s, throughput = {metrics[2]} qps")


def run_inference(args, inputs, tokenizer, model):
    if args.benchmark_type in {"hf-pt", "hf-pt2", "hf-ort"}:
        run_hf_inference(args, inputs, tokenizer, model)
    elif args.benchmark_type == "ort":
        run_ort_inference(args, inputs, model)
    else:
        raise Exception(f"Cannot recognize {args.benchmark_type}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-bt", "--benchmark-type", type=str, required=True, choices=["hf-pt", "hf-pt2", "hf-ort", "ort"]
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
    parser.add_argument("-ms", "--model-size", required=True, type=str, default="7b", choices=["7b", "13b", "70b"])
    parser.add_argument(
        "-p",
        "--precision",
        required=True,
        type=str,
        default="fp32",
        choices=["int8", "fp16", "fp32"],
        help="Precision for model and inputs. For PyTorch models, this sets the model's precision. \
                              For ONNX models, the model's precision should be set before running this script.",
    )
    parser.add_argument(
        "--hf-pt-model-path",
        type=str,
        default="",
        help="Path to directory containing all PyTorch files (e.g. tokenizer, PyTorch model)",
    )
    parser.add_argument(
        "--hf-ort-model-path",
        type=str,
        default="",
        help="Path to directory containing all ONNX files (e.g. tokenizer, encoder, decoder, decoder_with_past)",
    )
    parser.add_argument("--ort-model-path", type=str, default="", help="Path to ONNX model")

    parser.add_argument("-b", "--batch-size", type=int, default=1)
    parser.add_argument("-s", "--sequence-length", type=int, default=512)
    parser.add_argument("--prompt", type=str, default="Hey, are you conscious? Can you talk to me?")

    # Args for decoding logic
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--num-return-sequences", type=int, default=1)

    # Args for running and evaluating the model
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda", "rocm"],
    )
    parser.add_argument("-id", "--device-id", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("-w", "--warmup-runs", type=int, default=5)
    parser.add_argument("-n", "--num-runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2)

    # Args for accessing detailed info
    parser.add_argument("--profile", default=False, action="store_true")
    parser.add_argument(
        "--pt-filter-by", type=str, default="self_cpu_time_total", help="What to filter PyTorch profiler by"
    )
    parser.add_argument("--pt-num-rows", type=int, default=1000, help="Number of rows for PyTorch profiler to display")
    parser.add_argument("--verbose", default=False, action="store_true")

    args = parser.parse_args()

    # Set seed properties
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set runtime properties
    if "ort" in args.benchmark_type:
        setattr(args, "execution_provider", f"{args.device.upper()}ExecutionProvider")  # noqa: B010
        if args.execution_provider == "CUDAExecutionProvider":
            args.execution_provider = (args.execution_provider, {"device_id": args.device_id})
        elif args.execution_provider == "ROCMExecutionProvider":
            args.execution_provider = (args.execution_provider, {"device_id": args.device_id})
            args.device = "cuda"

    return args


def main():
    args = get_args()
    setup_logger(args.verbose)
    logger.info(args.__dict__)
    torch.backends.cudnn.benchmark = True
    inputs, tokenizer, model = get_vars(args)
    run_inference(args, inputs, tokenizer, model)


if __name__ == "__main__":
    main()
