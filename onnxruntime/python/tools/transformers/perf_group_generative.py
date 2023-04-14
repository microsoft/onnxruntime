import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from perf_single_generative import parse_perf_single_generative_model
from transformers import GPT2Tokenizer

import onnxruntime

general_gpt2_exporting_args = [
    "-b",  # no block operator
    "--use_external_data_format",
    "--use_gpu",
    "--disable_parity",
    "--disable_perf_test",
    "--total_runs=1",
]

custom_gpt2_exporting_args = {
    "greedy": [
        "--top_p=0",
        "--past_present_share_buffer",
        "--use_decoder_masked_self_attention",
    ],
    "topp": [
        "--top_p=0.6",
        "--past_present_share_buffer",
    ],
    "beam": [
        "--top_p=0",
        "--past_present_share_buffer",
        "--use_decoder_masked_self_attention",
    ],
}

gpt2_running_variants = {
    "small_context": {"max_length": 32, "context_length": [128], "batch_size": [1, 2, 4, 8, 16, 32, 64]},
    "middle_context": {"max_length": 32, "context_length": [512], "batch_size": [1, 2, 4, 8, 16, 32]},
    "large_context": {"max_length": 32, "context_length": [1024], "batch_size": [1, 2, 4, 8, 16]},
    "different_length_context": {
        "max_length": 32,
        "context_length": [32, 64, 99, 128, 160, 192, 227, 256],
        "batch_size": [4, 8, 16, 32],
    },
}


def parse_arguments(argv):
    parser = argparse.ArgumentParser("perf_group_generative.py")

    parser.add_argument(
        "--model_type",
        required=False,
        type=str,
        default="gpt2",
        choices=["gpt2", "t5", "mt5"],
        help="Model type (currently only support gpt2) in the list: " + ", ".join(["gpt2", "t5", "mt5"]),
    )

    parser.add_argument(
        "--model_names",
        required=False,
        nargs="*",
        type=str,
        default=["gpt2"],
        choices=["gpt2", "gpt2-large", "distilgpt2", "gpt2-medium", "gpt2-xl"],
        help="Model names to test. default list is ['gpt2'].",
    )

    parser.add_argument(
        "--search_type",
        required=True,
        type=str,
        choices=["greedy", "topp", "beam"],
        help="Type of onnx model exported, (greedy, topp, beam).",
    )

    parser.add_argument(
        "-p",
        "--precision",
        required=False,
        type=str,
        default="fp16",
        choices=["fp16", "fp32"],
        help="using fp16(default) model or fp32 model",
    )

    parser.add_argument(
        "--workspace",
        required=False,
        type=str,
        default=os.path.join(".", "workspace"),
        help="Directory to save and perf various models and test result, final result is saved here as perf_result.txt",
    )

    parser.add_argument(
        "--cache_dir",
        required=False,
        type=str,
        default=os.path.join(".", "cache_models"),
        help="Directory to cache pre-trained models",
    )

    parser.add_argument(
        "-f",
        "--force_overwrite",
        required=False,
        action="store_true",
        help="Overwrite existing models to be exported",
    )

    parser.add_argument(
        "-s",
        "--skip_exist_exported",
        required=False,
        action="store_true",
        help="Skip is target model to export already exists",
    )

    parser.add_argument(
        "--debug",
        required=False,
        action="store_true",
        help="In debug mode, only first 1 batch_size config will be run for each test variants group",
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        required=False,
        default=1,
        help="Beam size (default 1), only effective at beam search is used",
    )

    parser.add_argument(
        "--num_return_sequences",
        type=int,
        required=False,
        default=1,
        help="Number of return sequence <= num_beams, default 1, only effective at beam search is used",
    )

    # if all below three are specified in command line, only run this config
    # {"max_length": 32, "context_length": [128, 256, 333], "batch_size": [1, 2, 4, 8, 16, 32, 64]},
    parser.add_argument(
        "--max_length",
        type=int,
        required=False,
        default=-1,
        help="Max_lenght of the output (no counting the context length), default -1 means not use command line args",
    )

    parser.add_argument(
        "--context_length",
        type=int,
        nargs="*",
        default=[],
        help="context_length of different sequence in batch. Empty means not use command line args",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="*",
        default=[],
        help="batch_size to test. if not specified, do not using command line args. ",
    )

    parser.add_argument(
        "-o",
        "--output_report",
        type=str,
        default=None,
        help="specify output report file name. if not specified, {workspace}/all_test_result.txt will be used",
    )

    parser.add_argument(
        "--custom_name",
        type=str,
        default=None,
        help="specify variant name if command line specify batch_size, context_length and max_length",
    )

    args = parser.parse_args(argv)
    return args


def report_message(freport, msg: str):
    print(msg)
    freport.write(msg)
    freport.write("\n")


def perform_group_perf(args):
    assert args.model_type == "gpt2"

    Path(args.workspace).mkdir(parents=True, exist_ok=True)
    result_perf_file = os.path.join(args.workspace, "all_test_result.txt")
    if args.output_report is not None:
        result_perf_file = args.output_report
    with open(result_perf_file, "w") as freport:
        for model_name in args.model_names:
            report_message(freport, f"====> Model name: {model_name}")
            exporting_cmd = ["python", "convert_generation.py"]
            exporting_cmd.extend(general_gpt2_exporting_args)
            exporting_cmd.extend(custom_gpt2_exporting_args[args.search_type])
            exporting_cmd.extend(
                [f"--num_beams={args.num_beams}", f"--num_return_sequences={args.num_return_sequences}"]
            )
            output_model_dir = os.path.join(args.workspace, model_name)
            output_model_path = os.path.join(output_model_dir, f"model_{args.precision}.onnx")
            exporting_cmd.extend([f"-m={model_name}", f"--cache_dir={args.cache_dir}"])
            exporting_cmd.extend([f"--output={output_model_path}", f"-p={args.precision}"])

            eos_token_id, pad_token_id, vocab_size = (-1, -1, -1)
            tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=args.cache_dir)
            tokenizer.pad_token = tokenizer.eos_token
            eos_token_id, pad_token_id, vocab_size = (tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.vocab_size)
            print(f"      ----Using eos_token_id:{eos_token_id} pad_token_id:{pad_token_id} vocab_size:{vocab_size}")
            exporting_cmd.extend(
                [f"--eos_token_id={eos_token_id}", f"--pad_token_id={pad_token_id}", f"--vocab_size={vocab_size}"]
            )

            report_message(freport, f"  ==> {exporting_cmd}")
            report_message(freport, f"  ==> {' '.join(exporting_cmd)}")

            Path(output_model_dir).mkdir(parents=True, exist_ok=True)
            if os.path.exists(output_model_path):
                if args.force_overwrite:
                    os.remove(output_model_path)
                elif args.skip_exist_exported:
                    report_message(freport, f"  ==> [WARNING] Skip exporting existing onnx model {output_model_path}")
                else:
                    raise RuntimeError(
                        f"Onnx model [{output_model_path}] existed, please use -f/--force_overwrite to rewrite it or -s/--skip_exist_exported!"
                    )

            if not os.path.exists(output_model_path):
                subprocess.run(exporting_cmd)
            if not os.path.exists(output_model_path):
                raise RuntimeError(f"Onnx model [{output_model_path}] not found, convert_generate error?")

            # Start running perf with various parameter combinations
            base_run_conf = ["--min_length=1"]
            base_run_conf.extend([f"-m={model_name}", f"--cache_dir={args.cache_dir}"])
            base_run_conf.extend([f"--onnx_model={output_model_path}"])
            base_run_conf.extend(
                [f"--num_beams={args.num_beams}", f"--num_return_sequences={args.num_return_sequences}"]
            )
            report_message(freport, f"  ==> Common running args:{base_run_conf}")

            effective_variants = gpt2_running_variants
            if args.max_length > 0 and len(args.batch_size) > 0 and len(args.context_length) > 0:
                custom_name = args.custom_name if args.custom_name is not None else "command_line_variant"
                effective_variants = {
                    custom_name: {
                        "max_length": args.max_length,
                        "context_length": args.context_length,
                        "batch_size": args.batch_size,
                    },
                }

            for perf_var_name, perf_variant in effective_variants.items():
                report_message(freport, f"  ==> Perf test group: {perf_var_name}")
                for batch_size in perf_variant["batch_size"]:
                    # {"max_length": 32, "context_length": [128], "batch_size": [1, 2, 4, 8, 16, 32, 64]},
                    simple_perf_args = [f"--batch_size={batch_size}", f"--max_length={perf_variant['max_length']}"]
                    simple_perf_args.extend(["--context_length"])
                    simple_perf_args.extend([str(len) for len in perf_variant["context_length"]])

                    perf_args = base_run_conf
                    perf_args.extend(simple_perf_args)
                    result, _ = parse_perf_single_generative_model(perf_args)

                    if result is None:
                        report_message(freport, f"    ====> Average_latency_ms: FAILED, {simple_perf_args}")
                    else:
                        report_message(
                            freport, f"    ====> Average_latency_ms:{result['average_latency_ms']}, {simple_perf_args}"
                        )

                    if args.debug:
                        break


if __name__ == "__main__":
    # Sample usage:
    # Test on greedy
    #   python perf_group_generative.py --workspace ~/gpt2_greedy --cache_dir ~/cache_models --search_type greedy
    # Test on topp:
    #   python perf_group_generative.py --workspace ~/gpt2_topp --cache_dir ~/cache_models --search_type topp
    # Test on beam search 4:
    #   python perf_group_generative.py --workspace ~/gpt2_beam4 --cache_dir ~/cache_models --search_type beam --num_beams=4 --num_return_sequences=4
    #
    os.environ["ORT_ENABLE_FUSED_CAUSAL_ATTENTION"] = "1"
    args = parse_arguments(sys.argv[1:])
    perform_group_perf(args)
