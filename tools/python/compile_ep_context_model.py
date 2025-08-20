#!/usr/bin/env python3

"""
Script to generate an EP context model for any ONNX model. It also prints compile time and load time for EP context as well as regular ONNX model for comparison

Usage:
python compile_ep_context_model.py -i /path/to/model.onnx -o /path/to/model_ctx.onnx -e <embed_mode> -p <EP>

Dependencies:
- ONNXRuntime python wheel installed
"""

import os
import sys
import time
import argparse
import logging

import onnxruntime as ort

ort.set_default_logger_severity(3)

# EPs
TRT_RTX_EP = "NvTensorRTRTXExecutionProvider"


def compile(input_path, output_path, provider, embed_mode=False):
    if os.path.exists(output_path):
        os.remove(output_path)

    session_options = ort.SessionOptions()
    ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    session_options.add_provider(provider, {})

    model_compiler = ort.ModelCompiler(session_options, input_path, embed_compiled_data_into_model=embed_mode)

    start = time.perf_counter()
    model_compiler.compile_to_file(output_path)
    stop = time.perf_counter()

    if os.path.exists(output_path):
        print("> Compiled successfully!")
        print(f"> Comile time: {stop - start: .3f} sec")
        print(f"> Compiled model saved at {output_path}")


def load_session(model_path, provider):
    providers = [(provider, {})]
    start = time.perf_counter()
    session = ort.InferenceSession(model_path, providers=providers)
    stop = time.perf_counter()
    print(f"> Session load time: {stop-start: .3f} sec")
    return


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Compile ONNX model with ONNX Runtime")
    parser.add_argument("-i", "--model_path", type=str, help="Path to the ONNX model file")
    parser.add_argument("-o", "--output_path", type=str, help="Path to save the compiled EP context model")
    parser.add_argument("-p", "--provider", default=TRT_RTX_EP, type=str, help="Execution Provider")
    parser.add_argument("-e", "--embed", default=False, type=bool, help="Binary data embedded within EP context node")
    args = parser.parse_args()

    print("Available execution provider(s):", ort.get_available_providers())
    print()

    print(f"> Using Execution Provider: {args.provider}")

    print("> Loading regular onnx...")
    load_session(args.model_path, args.provider)

    print("> Compiling model...")
    compile(args.model_path, args.output_path, args.provider, args.embed)

    print("> Loading EP context model...")
    load_session(args.output_path, args.provider)
