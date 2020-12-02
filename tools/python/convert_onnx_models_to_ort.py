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


def convert(model_path: str, optimization_level: ort.GraphOptimizationLevel, use_nnapi: bool):
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
            so.graph_optimization_level = optimization_level

            print("Optimizing ONNX model {}".format(model))
            # creating the session will result in the optimized model being saved. we use just the CPU EP for this step
            providers = ['CPUExecutionProvider']
            _ = ort.InferenceSession(model, sess_options=so, providers=providers)

            # special case if we're enabling a compiling EP like NNAPI. we don't currently have a way to read the
            # required ops from an ORT format model, so we need an ONNX model that is only optimized to 'basic' level
            # to ensure all the nodes that NNAPI may take still exist. we can merge the required operators from that
            # with the required operators from an ONNX model optimized to a higher level (if the user requested that).
            # we must use this model with creating the ORT format model to maximize the nodes that NNAPI can potentially
            # take, so replace onnx_target_path with the new path.
            if use_nnapi and \
                (optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED or
                 optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_ALL):
                onnx_target_path = os.path.join(tmpdirname, re.sub('.onnx$', '.optimized.basic.onnx', model_filename))
                so.optimized_model_filepath = onnx_target_path
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
                _ = ort.InferenceSession(model, sess_options=so, providers=providers)

            # Second, convert optimized ONNX model to ORT format
            # we enable the compiling EPs when we generate the ORT format model so that we preserve the nodes it may
            # take, but allow optimization on any others
            if use_nnapi:
                # providers are priority based, so register NNAPI first
                providers.insert(0, 'NnapiExecutionProvider')

            so.optimized_model_filepath = ort_target_path
            # Use original optimization level so that if NNAPI is enabled we optimize nodes it is not taking
            so.graph_optimization_level = optimization_level
            so.add_session_config_entry('session.save_model_format', 'ORT')

            print("Converting optimized ONNX model to ORT format model {}".format(ort_target_path))
            _ = ort.InferenceSession(onnx_target_path, sess_options=so, providers=providers)

            # orig_size = os.path.getsize(onnx_target_path)
            # new_size = os.path.getsize(ort_target_path)
            # print("Serialized {} to {}. Sizes: orig={} new={} diff={} new:old={:.4f}:1.0".format(
            #     onnx_target_path, ort_target_path, orig_size, new_size, new_size - orig_size, new_size / orig_size))

        # now that all models are converted create the config file before the temp dir is deleted
        create_config_file(tmpdirname, os.path.join(model_path, 'required_operators.config'))


def _get_optimization_level(level):
    if level == 'disable':
        return ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    if level == 'basic':
        # Constant folding and other optimizations that only use ONNX operators
        return ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    if level == 'extended':
        # Optimizations using custom operators, excluding NCHWc optimizations
        return ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    if level == 'all':
        # all optimizations, including NCHWc (which has hardware specific logic)
        print('WARNING: Enabling layout optimizations is not recommended unless the ORT format model will be executed '
              'on the same hardware used to create the model.')
        return ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    raise ValueError('Invalid optimization level of ' + level)


def parse_args():
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description='''Convert the ONNX format model/s in the provided directory to ORT format models.
        All files with a `.onnx` extension will be processed. For each one, an ORT format model will be created in the
        same directory. A configuration file will also be created called `required_operators.config`, and will contain
        the list of required operators for all converted models.
        This configuration file should be used as input to the minimal build'''
    )

    parser.add_argument('--use_nnapi', action='store_true',
                        help='Enable the NNAPI Execution Provider when creating models and determining required '
                             'operators. Note that this will limit the optimizations possible on nodes that the '
                             'NNAPI execution provider takes, in order to preserve those nodes in the ORT format '
                             'model.')

    parser.add_argument('--optimization_level', default='extended',
                        choices=['disable', 'basic', 'extended', 'all'],
                        help="Level to optimize ONNX model with, prior to converting to ORT format model. "
                             "These map to the onnxruntime.GraphOptimizationLevel values. "
                             "NOTE: It is NOT recommended to use 'all' unless you are creating the ORT format model on "
                             "the device you will run it on, as the generated model may not be valid on other hardware."
                        )

    parser.add_argument('model_path', help='Provide path to directory containing ONNX model/s to convert. '
                                           'Files with .onnx extension will be processed.')

    return parser.parse_args()


def main():
    args = parse_args()
    optimization_level = _get_optimization_level(args.optimization_level)
    convert(args.model_path, optimization_level, args.use_nnapi)


if __name__ == '__main__':
    main()
