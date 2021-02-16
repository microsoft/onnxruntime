#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import glob
import os
import pathlib
import re
import tempfile

import onnxruntime as ort


def _create_config_file_from_ort_models(model_path, enable_type_reduction: bool):
    if model_path.is_file():
        model_path = model_path.parent

    filename = 'required_operators_and_types.config' if enable_type_reduction else 'required_operators.config'
    config_file_path = os.path.join(model_path, filename)
    print("Creating configuration file for operators required by ORT format models in {}.".format(config_file_path))

    from util.ort_format_model import create_config_from_models
    create_config_from_models(model_path, config_file_path, enable_type_reduction)


def _create_session_options(optimization_level: ort.GraphOptimizationLevel,
                            output_model_path: pathlib.Path,
                            custom_op_library: pathlib.Path):
    so = ort.SessionOptions()
    so.optimized_model_filepath = str(output_model_path)
    so.graph_optimization_level = optimization_level

    if custom_op_library:
        so.register_custom_ops_library(str(custom_op_library))

    return so


def _convert(model_path: pathlib.Path, optimization_level: ort.GraphOptimizationLevel, use_nnapi: bool,
             custom_op_library: pathlib.Path, create_optimized_onnx_model: bool):
    if model_path.is_file():
        models = [str(model_path)]
    else:
        models = glob.glob(os.path.join(model_path, '**', '*.onnx'), recursive=True)

    if len(models) == 0:
        raise ValueError("No .onnx files were found in " + model_path)

    providers = ['CPUExecutionProvider']
    if use_nnapi:
        # providers are priority based, so register NNAPI first
        providers.insert(0, 'NnapiExecutionProvider')

    for model in models:
        # create .ort file in same dir as original onnx model
        ort_target_path = re.sub('.onnx$', '.ort', model)

        if create_optimized_onnx_model:
            # Create an ONNX file with the same optimizations that will be used for the ORT format file.
            # This allows the ONNX equivalent of the ORT format model to be easily viewed in Netron.
            optimized_target_path = re.sub('.onnx$', '.optimized.onnx', model)
            so = _create_session_options(optimization_level, optimized_target_path, custom_op_library)

            print("Saving optimized ONNX model {}".format(model))
            _ = ort.InferenceSession(model, sess_options=so, providers=providers)

        # Load ONNX model, optimize, and save to ORT format
        so = _create_session_options(optimization_level, ort_target_path, custom_op_library)
        so.add_session_config_entry('session.save_model_format', 'ORT')

        print("Converting optimized ONNX model to ORT format model {}".format(ort_target_path))
        _ = ort.InferenceSession(model, sess_options=so, providers=providers)

        # orig_size = os.path.getsize(onnx_target_path)
        # new_size = os.path.getsize(ort_target_path)
        # print("Serialized {} to {}. Sizes: orig={} new={} diff={} new:old={:.4f}:1.0".format(
        #     onnx_target_path, ort_target_path, orig_size, new_size, new_size - orig_size, new_size / orig_size))


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
        This configuration file should be used as input to the minimal build via the `--include_ops_by_config`
        parameter.
        '''
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

    parser.add_argument('--enable_type_reduction', action='store_true',
                        help='Add operator specific type information to the configuration file to potentially reduce '
                             'the types supported by individual operator implementations.')

    parser.add_argument('--custom_op_library', type=pathlib.Path, default=None,
                        help='Provide path to shared library containing custom operator kernels to register.')

    parser.add_argument('--save_optimized_onnx_model', action='store_true',
                        help='Save the optimized version of each ONNX model. '
                             'This will have the same optimizations applied as the ORT format model.')

    parser.add_argument('model_path', type=pathlib.Path,
                        help='Provide path to ONNX model/s to convert. Can be a specific model or a directory. '
                             'All files with a .onnx extension, including in subdirectories, will be processed '
                             'if the path is a directory.')

    return parser.parse_args()


def main():
    args = parse_args()

    model_path = args.model_path.resolve()
    custom_op_library = args.custom_op_library.resolve() if args.custom_op_library else None

    if custom_op_library and not custom_op_library.is_file():
        raise FileNotFoundError("Unable to find custom operator library '{0}'".format(custom_op_library))

    optimization_level = _get_optimization_level(args.optimization_level)
    _convert(model_path, optimization_level, args.use_nnapi, custom_op_library, args.save_optimized_onnx_model)
    _create_config_file_from_ort_models(model_path, args.enable_type_reduction)


if __name__ == '__main__':
    main()
