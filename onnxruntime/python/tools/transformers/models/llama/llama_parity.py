import argparse
import logging
import os
import sys
from typing import List

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from benchmark_helper import create_onnxruntime_session, setup_logger  # noqa: E402

logger = logging.getLogger("")


def get_position_ids(attention_mask: torch.Tensor, use_past_kv: bool):
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    if use_past_kv:
        position_ids = position_ids[:, -1].unsqueeze(-1)
    return position_ids


def verify_parity(args: argparse.Namespace, config: LlamaConfig, tokenizer: LlamaTokenizer, pt_model: LlamaForCausalLM):
    # Get model inputs and properties
    prompt = "Hey, are you conscious? Can you talk to me?"
    inputs = tokenizer(prompt, return_tensors="pt")

    batch_size, seq_len = inputs["input_ids"].shape
    num_heads, head_size = config.num_attention_heads, int(config.hidden_size / config.num_attention_heads)

    # Run inference with PyTorch
    inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "position_ids": get_position_ids(inputs["attention_mask"], args.use_past_kv),
    }
    if args.use_past_kv:
        inputs["input_ids"] = inputs["input_ids"][:, -1:]

        inputs.update(
            {
                "past_key_values": [
                    (
                        torch.rand(batch_size, num_heads, seq_len - 1, head_size),
                        torch.rand(batch_size, num_heads, seq_len - 1, head_size),
                    )
                    for _ in range(config.num_hidden_layers)
                ]
            }
        )

    pt_outputs = pt_model(**inputs).logits.detach().cpu().numpy()

    # Run inference with ORT
    ort_model = create_onnxruntime_session(
        args.onnxruntime,  # onnx_model_path
        args.execution_provider != "cpu",  # use_gpu
        provider=args.execution_provider,
        verbose=args.verbose,
    )

    for k, v in inputs.items():
        if k == "past_key_values":
            continue
        else:
            inputs[k] = v.detach().cpu().numpy()

    if "past_key_values" in inputs:
        np_dtype = np.float16 if "fp16" in args.onnxruntime else np.float32
        for i, (past_k, past_v) in enumerate(inputs["past_key_values"]):
            inputs[f"past_key_values.{i}.key"] = past_k.detach().cpu().numpy().astype(np_dtype)
            inputs[f"past_key_values.{i}.value"] = past_v.detach().cpu().numpy().astype(np_dtype)
        del inputs["past_key_values"]

    ort_outputs = ort_model.run(None, inputs)[0][0]

    # Compare PyTorch and ONNX Runtime accuracy
    tol = 1e-3 if "fp32" in args.onnxruntime else 1e-2 if "fp16" in args.onnxruntime else 1e2
    parity = np.allclose(pt_outputs, ort_outputs, rtol=tol, atol=tol)
    logger.warning(f"Are PyTorch and ONNX Runtime results close? {parity}")
    if not parity:
        logger.warning(f"Max diff: {np.max(pt_outputs - ort_outputs)}")


def get_args(argv: List[str]):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        help="Model name in Hugging Face",
    )

    parser.add_argument(
        "-p",
        "--pytorch",
        required=False,
        default=os.path.join("."),
        help="Directory to PyTorch model and associated files if saved on disk",
    )

    parser.add_argument(
        "-o",
        "--onnxruntime",
        required=True,
        default=os.path.join("."),
        help="Directory to ONNX Runtime model and associated files saved on disk",
    )

    parser.add_argument(
        "-ep",
        "--execution_provider",
        required=False,
        default="cpu",
        choices=["cpu", "cuda", "rocm"],
        help="Execution provider to verify parity with",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose logs",
    )
    parser.set_defaults(verbose=False)

    parser.add_argument(
        "-kv",
        "--use_past_kv",
        action="store_true",
        help="Use past key and past value as inputs to the model. Necessary for decoder_with_past_model.onnx models.",
    )
    parser.set_defaults(use_past_kv=False)

    args = parser.parse_args() if argv == [] else parser.parse_args(argv)
    return args


def main(argv: List[str] = []):  # noqa: B006
    args = get_args(argv)
    setup_logger(args.verbose)
    logger.info(f"Arguments: {args}")

    # Load model, tokenizer, and config
    use_auth_token = args.pytorch == os.path.join(".")
    config = LlamaConfig.from_pretrained(
        args.model_name if use_auth_token else args.pytorch, use_auth_token=use_auth_token
    )
    tokenizer = LlamaTokenizer.from_pretrained(
        args.model_name if use_auth_token else args.pytorch, use_auth_token=use_auth_token
    )
    llama = LlamaForCausalLM.from_pretrained(
        args.model_name if use_auth_token else args.pytorch, use_auth_token=use_auth_token, use_cache=True
    )

    verify_parity(args, config, tokenizer, llama)


if __name__ == "__main__":
    main()
