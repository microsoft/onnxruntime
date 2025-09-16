#!/usr/bin/env python3

"""
Script to generate an EP context model for any ONNX model. It also prints compile time and load time for EP context as well as regular ONNX model for comparison

Usage:
python compile_ep_context_model.py -i /path/to/model.onnx -o /path/to/model_ctx.onnx -e <embed_mode> -p <EP>

Dependencies:
- ONNXRuntime python wheel installed
"""

import argparse
import os
import time

import onnxruntime as ort

ort.set_default_logger_severity(3)

# EPs
TRT_RTX_EP = "NvTensorRTRTXExecutionProvider"


class KeyValueAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            try:
                key, val = value.split("=", 1)
                getattr(namespace, self.dest)[key] = val
            except ValueError:
                raise argparse.ArgumentError(self, f"Invalid argument format: {value}. Expected key=value.")


def compile(input_path, output_path, provider, provider_options, embed_mode=False):
    if os.path.exists(output_path):
        os.remove(output_path)

    session_options = ort.SessionOptions()
    session_options.add_provider(provider, provider_options)

    model_compiler = ort.ModelCompiler(session_options, input_path, embed_compiled_data_into_model=embed_mode)

    start = time.perf_counter()
    model_compiler.compile_to_file(output_path)
    stop = time.perf_counter()

    if os.path.exists(output_path):
        print("> Compiled successfully!")
        print(f"> Compile time: {stop - start: .3f} sec")
        print(f"> Compiled model saved at {output_path}")


def load_session(model_path, provider, provider_options):
    providers = [(provider, provider_options)]
    start = time.perf_counter()
    _ = ort.InferenceSession(model_path, providers=providers)
    stop = time.perf_counter()
    print(f"> Session load time: {stop - start: .3f} sec")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile ONNX model with ONNX Runtime")
    parser.add_argument("-i", "--model_path", type=str, required=True, help="Path to the ONNX model file")
    parser.add_argument("-o", "--output_path", type=str, help="Path to save the compiled EP context model")
    parser.add_argument("-p", "--provider", default=TRT_RTX_EP, type=str, help="Execution Provider")
    parser.add_argument(
        "-po",
        "--provider_options",
        nargs="+",
        action=KeyValueAction,
        help="Provider options as key=value pairs. E.g., ep_provider_option=value",
    )
    parser.add_argument("-e", "--embed", default=False, type=bool, help="Binary data embedded within EP context node")
    args = parser.parse_args()

    if not args.output_path:
        parent_dir = os.path.dirname(args.model_path)
        args.output_path = os.path.join(parent_dir, "model_ctx.onnx")

    provider_options = args.provider_options if args.provider_options else {}

    print("Available execution provider(s):", ort.get_available_providers())
    print()

    print(f"> Using Execution Provider: {args.provider}")
    if provider_options:
        print(f"> Using Provider Options: {provider_options}")

    print("> Loading regular onnx...")
    load_session(args.model_path, args.provider, provider_options)

    print("> Compiling model...")
    compile(args.model_path, args.output_path, args.provider, provider_options, args.embed)

    print("> Loading EP context model...")
    load_session(args.output_path, args.provider, provider_options)
