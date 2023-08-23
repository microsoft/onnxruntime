import argparse
import logging
import os
from typing import List

import numpy as np
import torch
from benchmark_helper import create_onnxruntime_session, setup_logger
from llama_inputs import convert_inputs_for_ort, get_sample_inputs, get_sample_with_past_kv_inputs
from transformers import LlamaConfig, LlamaForCausalLM

logger = logging.getLogger("")


def verify_parity(args: argparse.Namespace, config: LlamaConfig, pt_model: LlamaForCausalLM):
    # Dummy values for parity
    batch_size, sequence_length = 2, 8
    device = torch.device("cpu")

    # Run inference with PyTorch
    inputs = (
        get_sample_inputs(config, device, batch_size, sequence_length, return_dict=True)
        if not args.use_past_kv
        else get_sample_with_past_kv_inputs(
            config, device, batch_size, sequence_length, use_fp16=(args.precision == "fp16"), return_dict=True
        )
    )
    pt_outputs = pt_model(**inputs).logits.detach().cpu().numpy()

    # Run inference with ORT
    inputs = convert_inputs_for_ort(inputs, use_fp16=(args.precision == "fp16"))
    ort_model = create_onnxruntime_session(
        args.onnx_model_path,
        args.execution_provider != "cpu",  # use_gpu
        provider=args.execution_provider,
        verbose=args.verbose,
    )
    ort_outputs = ort_model.run(None, inputs)[0]

    # Compare PyTorch and ONNX Runtime accuracy
    tol = 1e-3 if args.precision == "fp32" else 1e-2 if args.precision == "fp16" else 1e2
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
        "-t",
        "--torch_model_directory",
        required=False,
        default=os.path.join("."),
        help="Path to folder containing PyTorch model and associated files if saved on disk",
    )

    parser.add_argument(
        "-o",
        "--onnx_model_path",
        required=True,
        default=os.path.join("."),
        help="Path to ONNX model (with external data files saved in the same folder as the model)",
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
        "-p",
        "--use_past_kv",
        action="store_true",
        help="Use past key and past value as inputs to the model. Necessary for decoder_with_past_model.onnx models.",
    )
    parser.set_defaults(use_past_kv=False)

    parser.add_argument(
        "-fp",
        "--precision",
        required=True,
        choices=["int8", "fp16", "fp32"],
        help="Precision of model",
    )

    args = parser.parse_args() if argv == [] else parser.parse_args(argv)
    return args


def main(argv: List[str] = []):  # noqa: B006
    args = get_args(argv)
    setup_logger(args.verbose)
    logger.info(f"Arguments: {args}")

    # Load model and config
    use_auth_token = args.torch_model_directory == os.path.join(".")
    location = args.model_name if use_auth_token else args.torch_model_directory

    config = LlamaConfig.from_pretrained(location, use_auth_token=use_auth_token)
    llama = LlamaForCausalLM.from_pretrained(
        location,
        torch_dtype=(torch.float16 if args.precision == "fp16" else torch.float32),
        use_auth_token=use_auth_token,
        use_cache=True,
    )

    verify_parity(args, config, llama)


if __name__ == "__main__":
    main()
