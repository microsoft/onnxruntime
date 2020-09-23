#!/usr/bin/env python
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import glob
import os
import re
import sys
import tempfile

import onnxruntime as ort


def create_config_file(optimized_model_path, config_file_path):
    script_path = os.path.dirname(os.path.realpath(__file__))
    ci_build_py_path = os.path.abspath(os.path.join(script_path, '..', 'ci_build'))
    sys.path.append(ci_build_py_path)

    # create config file from all the optimized models
    print("Creating configuration file for operators required by optimized models in {}".format(config_file_path))
    from exclude_unused_ops import exclude_unused_ops  # tools/ci_build/exclude_unused_ops.py
    exclude_unused_ops(optimized_model_path, config_path=None, ort_root=None, output_config_path=config_file_path)


def convert(model_path: str):
    models = glob.glob(os.path.join(model_path, '**', '*.onnx'), recursive=True)

    if len(models) == 0:
        raise ValueError("No .onnx files were found in " + model_path)

    # create temp directory to create optimized onnx format models in. currently we need this to create the
    # config file with required operators. long term we could potentially do this from the ORT format model,
    # however that requires a lot of infrastructure to be able to parse the flatbuffers schema for those files
    with tempfile.TemporaryDirectory() as tmpdirname:
        for model in models:
            model_filename = os.path.basename(model)
            # create .optimized.onnx file in temp dir
            onnx_target_path = os.path.join(tmpdirname, re.sub('.onnx$', '.optimized.onnx', model_filename))
            # create .ort file in same dir as original onnx model
            ort_target_path = re.sub('.onnx$', '.ort', model)

            so = ort.SessionOptions()
            so.optimized_model_filepath = onnx_target_path
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED  # Skip NCHWc optimizations

            print("Optimizing ONNX model {}".format(model))
            # creating the session will result in the optimized model being saved
            _ = ort.InferenceSession(model, sess_options=so)

            # Second, convert optimized ONNX model to ORT format
            so.optimized_model_filepath = ort_target_path
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL  # Convert model as-is so we don't change the kernels in this step # noqa
            so.add_session_config_entry('session.save_model_format', 'ORT')

            print("Converting optimized ONNX model to ORT format model {}".format(ort_target_path))
            _ = ort.InferenceSession(onnx_target_path, sess_options=so)

            # orig_size = os.path.getsize(onnx_target_path)
            # new_size = os.path.getsize(ort_target_path)
            # print("Serialized {} to {}. Sizes: orig={} new={} diff={} new:old={:.4f}:1.0".format(
            #     onnx_target_path, ort_target_path, orig_size, new_size, new_size - orig_size, new_size / orig_size))

        # now that all models are converted create the config file before the temp dir is deleted
        create_config_file(tmpdirname, os.path.join(model_path, 'required_operators.config'))


def parse_args():
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description='''Convert the ONNX format model/s in the provided directory to ORT format models.
        All files with a `.onnx` extension will be processed. For each one, an ORT format model will be created in the
        same directory. A configuration file will also be created called `required_operators.config`, and will contain
        the list of required operators for all converted models.
        This configuration file should be used as input to the minimal build'''
    )

    parser.add_argument('model_path', help='Provide path to directory containing ONNX model/s to convert. '
                                           'Files with .onnx extension will be processed.')
    return parser.parse_args()


def main():
    args = parse_args()
    convert(args.model_path)


if __name__ == '__main__':
    main()
