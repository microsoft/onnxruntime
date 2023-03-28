import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "models", "gpt2"))
sys.path.append(os.path.join(os.path.dirname(__file__), "models", "t5"))

import argparse
import logging
import time
from pathlib import Path

import numpy as np
from benchmark_helper import get_latency_result
from gpt2_helper import PRETRAINED_GPT2_MODELS  # noqa: E402
from t5_helper import PRETRAINED_MT5_MODELS, PRETRAINED_T5_MODELS  # noqa: E402
from transformers import GPT2Tokenizer  # , T5Tokenizer

import onnxruntime

logger = logging.getLogger("")


def perf_gpt_model(args: argparse.Namespace):
    assert args.model_type == "gpt2"
    eos_token_id, pad_token_id, vocab_size = (-1, -1, -1)
    if args.model_name is not None:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
        tokenizer.pad_token = tokenizer.eos_token
        eos_token_id, pad_token_id, vocab_size = (tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.vocab_size)

    if args.eos_token_id >= 0:
        eos_token_id = args.eos_token_id
    if args.pad_token_id >= 0:
        pad_token_id = args.pad_token_id
    if args.vocab_size >= 0:
        vocab_size = args.vocab_size

    assert eos_token_id >= 0
    assert pad_token_id >= 0
    assert vocab_size > max(pad_token_id, eos_token_id)

    # TODO: put into args
    lowest_token_id = 1
    dummy_token_id = 6000
    bad_words_ids = [10000, 20000]

    logger.debug(f"Creating ort session form {args.onnx_model}......")
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    execution_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]  # onnxruntime.get_available_providers()
    logger.info(f"Using providers {execution_providers}......")
    ort_session = onnxruntime.InferenceSession(args.onnx_model, sess_options, providers=execution_providers)

    model_input_names = [model_input.name for model_input in ort_session.get_inputs()]

    logger.info(f"Generating inputs of length {args.context_length}......")
    batch_size = len(args.context_length)
    max_context_len = max(args.context_length)
    input_ids = np.random.randint(lowest_token_id, vocab_size - 1, size=(batch_size, max_context_len), dtype=np.int32)
    # Remove accident padding_id first, then pad from left
    np.place(input_ids, input_ids == pad_token_id, [dummy_token_id])
    for i, length in enumerate(args.context_length):
        input_ids[i, 0 : (max_context_len - length)] = pad_token_id

    inputs = {
        "input_ids": input_ids,
        "max_length": np.array([args.max_length + max_context_len], dtype=np.int32),
        "min_length": np.array([args.min_length + max_context_len], dtype=np.int32),
        "repetition_penalty": np.array([args.repetition_penalty], dtype=np.float32),
    }

    if "attention_mask" in model_input_names:
        attention_mask = np.ones(input_ids.shape, dtype=np.int32)
        np.place(attention_mask, attention_mask == pad_token_id, [0])
        inputs["attention_mask"] = attention_mask

    if "num_beams" in model_input_names:
        inputs.update({"num_beams": np.array([args.num_beams], dtype=np.int32)})
    if "num_return_sequences" in model_input_names:
        inputs.update({"num_return_sequences": np.array([args.num_return_sequences], dtype=np.int32)})
    if "length_penalty" in model_input_names:
        inputs.update({"length_penalty": np.array([args.length_penalty], dtype=np.float32)})
    if "repetition_penalty" in model_input_names:
        inputs.update({"repetition_penalty": np.array([args.repetition_penalty], dtype=np.float32)})

    if "vocab_mask" in model_input_names:
        vocab_mask = np.ones((vocab_size), dtype=np.int32)
        for bad_word_id in bad_words_ids:
            vocab_mask[bad_word_id] = 0
        inputs["vocab_mask"] = vocab_mask

    if "prefix_vocab_mask" in model_input_names:
        logger.info("Use prefix vocab mask with all ones in ORT, but no corresponding setting for Torch model.")
        prefix_vocab_mask = np.ones((batch_size, vocab_size), dtype=np.int32)
        inputs["prefix_vocab_mask"] = prefix_vocab_mask

    if args.save_test_data:
        test_data_dir = Path(args.onnx_model).parent.as_posix()
        logger.debug("test_data_dir", test_data_dir)
        from bert_test_data import output_test_data

        logger.info(f"Saving test_data to {test_data_dir}/test_data_set_0 ...")
        dir = os.path.join(test_data_dir, "test_data_set_0")
        output_test_data(dir, inputs)

    logger.debug("ORT inputs", inputs)

    logger.info("Warmup ort session......")
    result = ort_session.run(None, inputs)

    # Test performance
    logger.info("Testing ort session......")
    try:
        latency = []
        for _ in range(args.total_runs):
            start = time.time()
            _ = ort_session.run(None, inputs)
            latency.append(time.time() - start)
        output = get_latency_result(latency, batch_size)

        print("      --> ORT perf result:", output)
        return output
    except:
        return None


def parse_arguments(argv) -> argparse.Namespace:
    parser = argparse.ArgumentParser("perf_single_generative.py")

    parser.add_argument(
        "-m",
        "--model_name",
        required=False,
        type=str,
        default=None,
        help="Pytorch model checkpoint path, or pretrained model name in the list: "
        + ", ".join(PRETRAINED_GPT2_MODELS + PRETRAINED_T5_MODELS + PRETRAINED_MT5_MODELS),
    )

    parser.add_argument(
        "--model_type",
        required=False,
        type=str,
        default="gpt2",
        choices=["gpt2", "t5", "mt5"],
        help="Model type (currently only support gpt2) in the list: " + ", ".join(["gpt2", "t5", "mt5"]),
    )

    parser.add_argument(
        "--cache_dir",
        required=False,
        type=str,
        default=os.path.join(".", "cache_models"),
        help="Directory to cache pre-trained models",
    )

    parser.add_argument(
        "--onnx_model",
        required=True,
        type=str,
        help="onnx model path (exported from convert_generation.py) for test",
    )

    parser.add_argument(
        "--total_runs",
        required=False,
        type=int,
        default=10,
        help="Number of times of inference for latency measurement",
    )

    parser.add_argument(
        "--save_test_data",
        default=False,
        action="store_true",
        help="save test data for onnxruntimer_perf_test tool",
    )

    parser.add_argument("--batch_size", type=int, required=True, help="batch_size")
    parser.add_argument(
        "--context_length",
        type=int,
        nargs="+",
        required=True,
        help="Initial context length of batch_size, if less than batch_size, it will tiled and truncate to batch_size",
    )
    parser.add_argument(
        "--min_length", type=int, required=False, default=1, help="Min output sequence length, (default 1)"
    )
    parser.add_argument(
        "--max_length", type=int, required=False, default=50, help="Max output sequence length (default 50)"
    )
    parser.add_argument("--num_beams", type=int, required=False, default=4, help="Beam size (default 4)")
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        required=False,
        default=1,
        help="Number of return sequence <= num_beams, default 1",
    )

    parser.add_argument(
        "--length_penalty",
        type=float,
        required=False,
        default=1,
        help="Positive. >1 to penalize and <1 to encourage short sentence.",
    )

    parser.add_argument(
        "--repetition_penalty",
        type=float,
        required=False,
        default=1,
        help="Positive. >1 to penalize and <1 to encourage.",
    )

    parser.add_argument(
        "--vocab_size",
        type=int,
        required=False,
        default=-1,
        help="Vocab_size of the underlying model used to decide the shape of vocab mask. Default from tokenizer",
    )

    parser.add_argument(
        "--eos_token_id",
        type=int,
        required=False,
        default=-1,
        help="custom eos_token_id for generating model with existing onnx encoder/decoder. Default from tokenizer",
    )

    parser.add_argument(
        "--pad_token_id",
        type=int,
        required=False,
        default=-1,
        help="custom pad_token_id for generating model with existing onnx encoder/decoder. Default from tokenizer",
    )

    args = parser.parse_args(argv)
    if len(args.context_length) < args.batch_size:
        dup = (args.batch_size + len(args.context_length) - 1) // len(args.context_length)
        args.context_length = args.context_length * dup
    args.context_length = args.context_length[0 : args.batch_size]
    return args


def parse_perf_single_generative_model(argv):
    args = parse_arguments(argv=argv)
    config_map = {
        param: getattr(args, param)
        for param in [
            "model_name",
            "model_type",
            "batch_size",
            "context_length",
            "min_length",
            "max_length",
            "num_beams",
            "num_return_sequences",
            "length_penalty",
            "repetition_penalty",
            "vocab_size",
        ]
    }
    print("      --> Perfing with config:", config_map)
    perf_result = perf_gpt_model(args)
    return perf_result, config_map


if __name__ == "__main__":
    parse_perf_single_generative_model(sys.argv[1:])
