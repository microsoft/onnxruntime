#!/usr/bin/env python
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import onnxruntime as ort
import os
import re


def convert(model: str):

    if not model.endswith('.onnx'):
        raise ValueError("Model filename must end in .onnx.")

    onnx_target_path = re.sub('.onnx$', '.optimized.onnx', model)
    ort_target_path = re.sub('.onnx$', '.ort', model)

    so = ort.SessionOptions()
    so.optimized_model_filepath = onnx_target_path
    so.add_session_config_entry('session.save_model_format', 'ONNX')
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED  # Skip NCHWc optimizations

    print("Optimizing ONNX model {} and saving in ONNX format to {}".format(model, onnx_target_path))
    # creating the session will result in the optimized model being saved
    _ = ort.InferenceSession(model, sess_options=so)

    # Second, convert optimized ONNX model to ORT format
    so.optimized_model_filepath = ort_target_path
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL  # Convert model as-is so we don't change the kernels in this step # noqa

    so.add_session_config_entry('session.save_model_format', 'ORT')

    print("Converting optimized ONNX model {} to ORT format model {}".format(onnx_target_path, ort_target_path))
    _ = ort.InferenceSession(onnx_target_path, sess_options=so)

    orig_size = os.path.getsize(onnx_target_path)
    new_size = os.path.getsize(ort_target_path)
    print("Serialized {} to {}. Sizes: orig={} new={} diff={} new:old={:.4f}:1.0".format(
        onnx_target_path, ort_target_path, orig_size, new_size, new_size - orig_size, new_size / orig_size))


def parse_args():
    parser = argparse.ArgumentParser(os.path.basename(__file__),
                                     description='''Convert an onnx model -> optimized onnx model -> ORT format model.
                                     Expects a .onnx file as input. Optimized onnx model will be saved in the same
                                     directory with an extension of .optimized.onnx.
                                     An ORT format model will be created from the optimized onnx model.
                                     The optimized onnx model should be used as input to a minimal build so that
                                     any post-optimization kernels are included in the build.'''
                                     )
    parser.add_argument('model', help='Provide path to ONNX model to convert. Must have .onnx extension.')
    return parser.parse_args()


def main():
    args = parse_args()
    convert(args.model)


if __name__ == '__main__':
    main()
