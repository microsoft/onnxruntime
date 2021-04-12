#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import pathlib
import typing

import onnxruntime as ort


def _path_match_suffix_ignore_case(path: typing.Union[pathlib.Path, str], suffix: str):
    if not isinstance(path, str):
        path = str(path)
    return path.casefold().endswith(suffix.casefold())


def _onnx_model_path_to_ort_model_path(onnx_model_path: pathlib.Path):
    assert onnx_model_path.is_file() and _path_match_suffix_ignore_case(onnx_model_path, ".onnx")
    return onnx_model_path.with_suffix(".ort")


def _create_config_file_from_ort_models(onnx_model_path_or_dir: pathlib.Path, enable_type_reduction: bool):
    if onnx_model_path_or_dir.is_dir():
        # model directory
        model_path_or_dir = onnx_model_path_or_dir
        config_path = None  # default path in model directory
    else:
        # single model
        model_path_or_dir = _onnx_model_path_to_ort_model_path(onnx_model_path_or_dir)
        config_suffix = ".{}".format(
            'required_operators_and_types.config' if enable_type_reduction else 'required_operators.config')
        config_path = model_path_or_dir.with_suffix(config_suffix)

    from util.ort_format_model import create_config_from_models
    create_config_from_models(model_path_or_dir=str(model_path_or_dir),
                              output_file=str(config_path) if config_path is not None else None,
                              enable_type_reduction=enable_type_reduction)


def _create_session_options(optimization_level: ort.GraphOptimizationLevel,
                            output_model_path: pathlib.Path,
                            custom_op_library: pathlib.Path):
    so = ort.SessionOptions()
    so.optimized_model_filepath = str(output_model_path)
    so.graph_optimization_level = optimization_level

    if custom_op_library:
        so.register_custom_ops_library(str(custom_op_library))

    return so


def _convert(model_path_or_dir: pathlib.Path, optimization_level: ort.GraphOptimizationLevel, use_nnapi: bool,
             custom_op_library: pathlib.Path, create_optimized_onnx_model: bool):
    models = []
    if model_path_or_dir.is_file() and _path_match_suffix_ignore_case(model_path_or_dir, ".onnx"):
        models.append(model_path_or_dir)
    elif model_path_or_dir.is_dir():
        for root, _, files in os.walk(model_path_or_dir):
            for file in files:
                if _path_match_suffix_ignore_case(file, ".onnx"):
                    models.append(pathlib.Path(root, file))

    if len(models) == 0:
        raise ValueError("No .onnx files were found in '{}'".format(model_path_or_dir))

    providers = ['CPUExecutionProvider']
    if use_nnapi:
        # providers are priority based, so register NNAPI first
        providers.insert(0, 'NnapiExecutionProvider')

    # if the optimization level is 'all' we manually exclude the NCHWc transformer. It's not applicable to ARM
    # devices, and creates a device specific model which won't run on all hardware.
    # If someone really really really wants to run it they could manually create an optimized onnx model first,
    # or they could comment out this code.
    optimizer_filter = None
    if optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_ALL:
        optimizer_filter = ['NchwcTransformer']

    for model in models:
        # ignore any files with an extension of .optimized.onnx which are presumably from previous executions
        # of this script
        if _path_match_suffix_ignore_case(model, ".optimized.onnx"):
            print("Ignoring '{}'".format(model))
            continue

        # create .ort file in same dir as original onnx model
        ort_target_path = _onnx_model_path_to_ort_model_path(model)

        if create_optimized_onnx_model:
            # Create an ONNX file with the same optimizations that will be used for the ORT format file.
            # This allows the ONNX equivalent of the ORT format model to be easily viewed in Netron.
            optimized_target_path = model.with_suffix(".optimized.onnx")
            so = _create_session_options(optimization_level, optimized_target_path, custom_op_library)

            print("Saving optimized ONNX model {} to {}".format(model, optimized_target_path))
            _ = ort.InferenceSession(str(model), sess_options=so, providers=providers,
                                     disabled_optimizers=optimizer_filter)

        # Load ONNX model, optimize, and save to ORT format
        so = _create_session_options(optimization_level, ort_target_path, custom_op_library)
        so.add_session_config_entry('session.save_model_format', 'ORT')

        print("Converting optimized ONNX model to ORT format model {}".format(ort_target_path))
        _ = ort.InferenceSession(str(model), sess_options=so, providers=providers,
                                 disabled_optimizers=optimizer_filter)

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
        # Optimizations using custom operators, excluding NCHWc and NHWC layout optimizers
        return ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    if level == 'all':
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

    parser.add_argument('--optimization_level', default='all',
                        choices=['disable', 'basic', 'extended', 'all'],
                        help="Level to optimize ONNX model with, prior to converting to ORT format model. "
                             "These map to the onnxruntime.GraphOptimizationLevel values. "
                             "If the level is 'all' the NCHWc transformer is manually disabled as it contains device "
                             "specific logic, so the ORT format model must be generated on the device it will run on. "
                             "Additionally, the NCHWc optimizations are not applicable to ARM devices."
                        )

    parser.add_argument('--enable_type_reduction', action='store_true',
                        help='Add operator specific type information to the configuration file to potentially reduce '
                             'the types supported by individual operator implementations.')

    parser.add_argument('--custom_op_library', type=pathlib.Path, default=None,
                        help='Provide path to shared library containing custom operator kernels to register.')

    parser.add_argument('--save_optimized_onnx_model', action='store_true',
                        help='Save the optimized version of each ONNX model. '
                             'This will have the same optimizations applied as the ORT format model.')

    parser.add_argument('model_path_or_dir', type=pathlib.Path,
                        help='Provide path to ONNX model or directory containing ONNX model/s to convert. '
                             'All files with a .onnx extension, including in subdirectories, will be processed.')

    return parser.parse_args()


def main():
    args = parse_args()

    model_path_or_dir = args.model_path_or_dir.resolve()
    custom_op_library = args.custom_op_library.resolve() if args.custom_op_library else None

    if not model_path_or_dir.is_dir() and not model_path_or_dir.is_file():
        raise FileNotFoundError("Model path '{}' is not a file or directory.".format(model_path_or_dir))

    if custom_op_library and not custom_op_library.is_file():
        raise FileNotFoundError("Unable to find custom operator library '{}'".format(custom_op_library))

    optimization_level = _get_optimization_level(args.optimization_level)
    _convert(model_path_or_dir, optimization_level, args.use_nnapi, custom_op_library, args.save_optimized_onnx_model)
    _create_config_file_from_ort_models(model_path_or_dir, args.enable_type_reduction)


if __name__ == '__main__':
    main()
