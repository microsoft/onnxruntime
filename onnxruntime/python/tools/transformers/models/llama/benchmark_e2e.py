# This is an end-to-end benchmarking script for the Hugging Face LLaMA-2 model.
#
# Prerequisites:
# 1) Install `huggingface-cli`:
#
# $ pip install huggingface_hub
#
# 2) Authenticate with Hugging Face's CLI:
#
# $ huggingface-cli login
#
# 3) Accept Meta's license in Hugging Face to access the models at https://huggingface.co/meta-llama/
#
# 4) Install the latest ONNX Runtime version
#
# $ pip install onnxruntime-gpu

from __future__ import annotations

from llama_inputs import (
    add_io_bindings_as_tensors,
    get_initial_inputs_and_outputs,
)
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

import datetime
import gc
import itertools
import logging
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
import time

logger = logging.getLogger(__name__)


def get_model(args):
    if args.benchmark_type in {"hf-pt-eager", "hf-pt-compile"}:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir,
            torch_dtype=args.torch_dtype,
            use_auth_token=args.auth,
            use_cache=True,
        ).to(args.target_device)

        if args.benchmark_type == "hf-pt-compile":
            model = torch.compile(model)

    else:
        sess_options = ort.SessionOptions()
        ep = ("CUDAExecutionProvider", {"device_id": args.device_id}) if args.device == "cuda" else "CPUExecutionProvider"
        model = ort.InferenceSession(onnx_model_path, sess_options=sess_options, providers=[ep])

    return model


def run_inference(args, model, runs, inputs, outputs):
    # Synchronize inputs
    io_binding = None
    if args.benchmark_type in {"hf-pt-eager", "hf-pt-compile"}:
        torch.cuda.synchronize(args.target_device)
    else:
        io_binding = add_io_bindings_as_tensors(model, inputs, outputs, args.use_fp16, args.use_buffer_share)
        io_binding.synchronize_inputs()

    # Run inference
    start = time.perf_counter()
    for _ in range(runs):
        if args.benchmark_type in {"hf-pt-eager", "hf-pt-compile"}:
            _ = model(**inputs)
            torch.cuda.synchronize(args.target_device)
        else:
            model.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()

    end = time.perf_counter()
    avg = (end - start) / runs
    return avg


def prepare_model_for_inference(args, model, config, tokenizer, prompt_length, prompt):
    clear_cache()
    inputs, outputs = get_initial_inputs_and_outputs(config, tokenizer, prompt_length, prompt, args.target_device, args.use_fp16, args.use_buffer_share, args.engine)
    run_inference(args, model, args.warmup_runs, inputs, outputs)
    return inputs, outputs


def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()


def save_results(results, filename, gen_length):
    df = pd.DataFrame(
        results,
        columns=[
            "Batch Size",
            "Prompt Length",
            "Prompt Processing Latency (ms)",
            "Prompt Processing Throughput (tps)",
            "Sampling Latency (ms)",
            "Sampling Throughput (tps)",
            "First Token Generated Latency (ms)",
            "First Token Generated Throughput (tps)",
            f"First {gen_length // 2} Tokens Generated Avg Latency (ms)",
            f"First {gen_length // 2} Tokens Generated Avg Throughput (tps)",
            f"First {gen_length} Tokens Generated Avg Latency (ms)",
            f"First {gen_length} Tokens Generated Avg Throughput (tps)",
            "Wall-Clock Latency (s)",
            "Wall-Clock Throughput (tps)",
        ],
    )

    df.to_csv(filename, index=False)
    logger.info(f"Results saved in {filename}!")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-bt",
        "--benchmark-type",
        type=str,
        required=True,
        choices=["hf-pt-eager", "hf-pt-compile", "ort-convert-to-onnx"],
    )

    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
        required=True,
        help="Hugging Face name of model (e.g. 'meta-llama/Llama-2-7b-hf')",
    )

    parser.add_argument(
        "-a",
        "--auth",
        default=False,
        action="store_true",
        help="Use Hugging Face authentication token to access model",
    )

    parser.add_argument(
        "-c",
        "--cache-dir",
        type=str,
        default="./model_cache",
        help="Path to directory containing all Hugging Face files (e.g. config, tokenizer, PyTorch model)",
    )

    parser.add_argument(
        "-o",
        "--ort-model-path",
        required=False,
        help="Path to ONNX model",
    )

    parser.add_argument(
        "-p",
        "--prompts-file",
        required=True,
        default="prompts.json",
        help="JSON file containing entries in the format 'prompt length: prompt' where prompt length = tokenized length of prompt",
    )

    parser.add_argument(
        "--use_fp16",
        default=False,
        action="store_true",
        help="Use float16 precision for inputs and outputs",
    )

    parser.add_argument(
        "--use_buffer_share",
        default=False,
        action="store_true",
        help="Use when GroupQueryAttention (GQA) is in ONNX model",
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
        default="32 64 128 256 512",
    )
    parser.add_argument(
        "-g",
        "--generation-length",
        type=int,
        default=256,
        help="Number of new tokens to generate",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument("-id", "--device-id", type=int, default=0)
    parser.add_argument("-w", "--warmup-runs", type=int, default=5)
    parser.add_argument("-n", "--num-runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2)

    args = parser.parse_args()

    # Set seed properties
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set runtime properties
    if "ort" in args.benchmark_type:
        setattr(args, "execution_provider", f"{args.device.upper()}ExecutionProvider")  # noqa: B010
        if args.execution_provider == "CUDAExecutionProvider":
            args.execution_provider = (args.execution_provider, {"device_id": rank})

    # Check that paths have been specified for any benchmarking with ORT
    if args.benchmark_type == "ort-convert-to-onnx":
        assert args.ort_model_path, "Please specify a path to `--ort-model-path`"

    args.batch_sizes = args.batch_sizes.split(" ")
    args.sequence_lengths = args.sequence_lengths.split(" ")

    # Use FP32 precision for FP32, INT8, INT4 CPU models, use FP16 precision for FP16 and INT4 GPU models
    args.precision = (
        "fp32" if args.precision in {"int8", "fp32"} or (args.precision == "int4" and args.device == "cpu") else "fp16"
    )

    target_device = f"cuda:{args.device_id}" if args.device != "cpu" else args.device
    torch_dtype = torch.float16 if args.use_fp16 else torch.float32
    engine = "ort" if args.benchmark_type == "ort-convert-to-onnx" else "pt"
    setattr(args, "target_device", target_device)  # noqa: B010
    setattr(args, "torch_dtype", torch_dtype)  # noqa: B010
    setattr(args, "engine", engine)  # noqa: B010

    return args


def main():
    args = get_args()

    # Get prompts and prompt sizes
    size_to_prompt = None
    with open(args.prompts_file, "r") as f:
        size_to_prompt = json.load(f, object_hook=lambda d: {int(k): v for k, v in d.items()})

    # Get config, tokenizer, and model
    config = AutoConfig.from_pretrained(args.model_name, cache_dir=args.cache_dir, use_auth_token=args.auth)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir, use_auth_token=args.auth)
    model = get_model(args)

    all_csv_metrics = []
    for (batch_size, prompt_length) in itertools.product(batch_sizes, prompt_lengths):
        logger.info(f"Running batch size = {batch_size}, prompt length = {prompt_length}")
        max_length = prompt_length + args.generation_length
        prompt = [size_to_prompt[prompt_length]] * batch_size
        csv_metrics = [batch_size, prompt_length]

        try:
            # Measure prompt processing
            logger.info("Measuring prompt processing...")
            inputs, outputs = prepare_model_for_inference(args, model, config, tokenizer, prompt_length, prompt)
            accelerator_prompt_latency_s = run_inference(args, model, args.num_runs, inputs, outputs)

            # Calculate prompt metrics
            accelerator_prompt_latency_ms = accelerator_prompt_latency_s * 1000
            accelerator_prompt_thrpt = batch_size * (prompt_length / accelerator_prompt_latency_s)
            logger.info(f"Accelerator Prompt Processing Latency: {accelerator_prompt_latency_s * 1000} ms")
            logger.info(f"Accelerator Prompt Processing Throughput: {batch_size * (prompt_length / accelerator_prompt_latency_s)} tps")
            csv_metrics.extend([accelerator_prompt_latency_ms, accelerator_prompt_thrpt])

            # Measure token generation
            logger.info("Measuring token generation...")
            inputs, outputs = prepare_model_for_inference(args, model, config, tokenizer, prompt)

            all_token_ids = inputs["input_ids"].clone()
            batch_size, sequence_length = all_token_ids.shape
            num_heads = config.num_key_value_heads
            head_size = head_size = config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads

            current_length = sequence_length
            assert current_length == prompt_length
            has_eos = torch.zeros(batch_size, device=device, dtype=torch.bool)

            accelerator_times = []  # 0th entry will have prompt accelerator time, 1st entry onwards will have token generation accelerator time
            sampling_times = []  # cost to sample after each model run
            wall_clock_start_time = time.perf_counter()
            while current_length <= max_length:
                # Run inference
                accelerator_time_latency_s = run_inference(args, model, 1, inputs, outputs)
                accelerator_time_latency_ms = accelerator_time_latency_s * 1000
                accelerator_times.append(accelerator_time_latency_ms)

                # Sample with argmax (greedy search)
                sampling_start_time = time.perf_counter()
                if outputs["logits"].shape[1] > 1:
                    prompt_end_indices = inputs["attention_mask"].sum(1) - 1
                    idxs = prompt_end_indices.unsqueeze(dim=1).repeat(1, config.vocab_size).view(batch_size, 1, config.vocab_size)
                    next_token_logits = torch.gather(outputs["logits"], 1, idxs).squeeze()
                else:
                    next_token_logits = outputs["logits"][:, -1, :]
                next_tokens = torch.argmax(next_token_logits, dim=-1)

                # Check if we previously reached EOS token id or if generated token id is EOS token id
                has_eos = has_eos | next_tokens == tokenizer.eos_token_id

                # Determine which new tokens to add to list of all token ids
                # Add EOS token ids for batch entries that ended early (ragged batching scenario where some batch entries ended early and some haven't)
                tokens_to_add = next_tokens.masked_fill(has_eos, tokenizer.eos_token_id).reshape([batch_size, 1])
                sampling_end_time = time.perf_counter()
                sampling_times.append(sampling_end_time - sampling_start_time)

                all_token_ids = torch.cat([all_token_ids, tokens_to_add], dim=-1)

                # Return early if all batch entries have reached EOS token id
                current_length += 1
                if torch.all(has_eos) or current_length > max_length:
                    break

                # Update inputs for next inference run
                inputs["input_ids"] = tokens_to_add
                inputs["position_ids"] = torch.max(inputs["position_ids"], dim=1)[0].reshape(batch_size, 1) + 1
                inputs["attention_mask"] = torch.cat([inputs["attention_mask"], (~has_eos).to(torch.int64).reshape(batch_size, 1)], 1)

                # Set logits to zeros for next inference run and re-use memory buffer
                if outputs["logits"].shape[1] != 1:
                    outputs["logits"] = outputs["logits"][:, :1, :].contiguous()
                outputs["logits"].zero_()

                # Update KV caches for next inference run
                if args.engine == "pt":
                    # Update KV caches for PyTorch
                    inputs["past_key_values"] = outputs["past_key_values"]
                elif not use_buffer_share:
                    # Update KV caches for ONNX Runtime if buffer sharing is not used
                    for i in range(config.num_hidden_layers):
                        inputs[f"past_key_values.{i}.key"] = outputs[f"present.{i}.key"]
                        inputs[f"past_key_values.{i}.value"] = outputs[f"present.{i}.value"]

                    new_sequence_length = inputs["attention_mask"].shape[1]
                    for i in range(config.num_hidden_layers):
                        present_key = torch.zeros(batch_size, num_heads, new_sequence_length, head_size, device=args.target_device, dtype=args.torch_dtype)
                        present_value = torch.zeros(batch_size, num_heads, new_sequence_length, head_size, device=args.targe_device, dtype=args.torch_dtype)
                        outputs.update({
                            f"present.{i}.key": present_key.contiguous(),
                            f"present.{i}.value": present_value.contiguous()
                        })

            wall_clock_end_time = time.perf_counter()
            wall_clock_latency_s = wall_clock_end_time - wall_clock_start_time

            if len(accelerator_times) > 0:
                # Calculate sampling metrics
                avg_sampling_latency_s = sum(sampling_times) / len(sampling_times)
                avg_sampling_latency_ms = avg_sampling_latency_s * 1000
                avg_sampling_thrpt = batch_size * (1 / avg_sampling_latency_s)
                logger.info(f"Average Sampling Latency: {avg_sampling_latency_s * 1000} ms")
                logger.info(f"Average Sampling Throughput: {batch_size * (1 / avg_sampling_latency_s)} tps")

                # Calculate first token generated metrics
                avg_accelerator_token_latency_s = accelerator_times[1] / 1
                avg_accelerator_token_latency_ms = avg_accelerator_token_latency_s * 1000
                avg_accelerator_token_thrpt = batch_size * (1 / avg_accelerator_token_latency_s)
                logger.info(f"First Token Average Accelerator Token Generation Latency: {avg_accelerator_token_latency_s * 1000} ms")
                logger.info(f"First Token Average Accelerator Token Generation Throughput: {batch_size * (1 / avg_accelerator_token_latency_s)} tps")

                csv_metrics.extend([avg_sampling_latency_ms, avg_sampling_thrpt, avg_accelerator_token_latency_ms, avg_accelerator_token_thrpt])

            halfway_idx = 1 + (args.generation_length // 2)  # +1 is for prompt entry

            if len(accelerator_times) >= halfway_idx:
                # Calculating average of first `halfway` tokens generated metrics
                avg_accelerator_token_latency_s = sum(accelerator_times[1 : halfway_idx]) / len(accelerator_times[1 : halfway_idx])
                avg_accelerator_token_latency_ms = avg_accelerator_token_latency_s * 1000
                avg_accelerator_token_thrpt = batch_size * (1 / avg_accelerator_token_latency_s)
                logger.info(f"First {args.generation_length // 2} Tokens Average Accelerator Token Generation Latency: {avg_accelerator_token_latency_s * 1000} ms")
                logger.info(f"First {args.generation_length // 2} Tokens Average Accelerator Token Generation Throughput: {batch_size * (1 / avg_accelerator_token_latency_s)} tps")

                csv_metrics.extend([avg_accelerator_token_latency_ms, avg_accelerator_token_thrpt])

            if len(accelerator_times) == args.generation_length + 1:  # +1 is for prompt entry
                avg_accelerator_token_latency_s = sum(accelerator_times[1:]) / len(accelerator_times[1:])
                avg_accelerator_token_latency_ms = avg_accelerator_token_latency_s * 1000
                avg_accelerator_token_thrpt = batch_size * (1 / avg_accelerator_token_latency_s)
                logger.info(f"First {args.generation_length} Tokens Average Accelerator Token Generation Latency: {avg_accelerator_token_latency_s * 1000} ms")
                logger.info(f"First {args.generation_length} Tokens Average Accelerator Token Generation Throughput: {batch_size * (1 / avg_accelerator_token_latency_s)} tps")

                # Calculate wall-clock metrics
                wall_clock_thrpt = batch_size * ((prompt_length + args.generation_length) / wall_clock_latency_s)
                logger.info(f"Wall-Clock Latency: {wall_clock_latency_s} s")
                logger.info(f"Wall-Clock Throughput: {batch_size * ((prompt_length + args.generation_length) / wall_clock_latency_s)} tps")

                csv_metrics.extend([avg_accelerator_token_latency_ms, avg_accelerator_token_thrpt, wall_clock_latency_s, wall_clock_thrpt])

            # Add metrics to CSV
            if len(csv_metrics) == 14:
                logger.info("Adding results to CSV")
                all_csv_metrics.append(csv_metrics)

                # Batch decoding at end of generation
                # logger.info("-------------")
                # logger.info(tokenizer.batch_decode(all_token_ids, skip_special_tokens=True))
                # logger.info("-------------")
            else:
                logger.info(f"Could not process token generation at batch size = {batch_size}, prompt length = {prompt_length}")
                continue

        except:
            logger.info(f"Could not benchmark at batch size = {batch_size}, prompt length = {prompt_length}")

    filename = f"benchmark_{args.engine}_e2e_{datetime.datetime.now():%Y-%m-%d_%H:%M:%S}.csv"
    save_results(all_csv_metrics, filename, args.generation_length)


if __name__ == "__main__":
    main()
