#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import onnx
import pathlib
import sys


def _extract_ops_from_onnx_graph(graph, operators, domain_opset_map):
    '''Extract ops from an ONNX graph and all subgraphs'''

    for operator in graph.node:
        # empty domain is used as an alias for 'ai.onnx'
        domain = operator.domain if operator.domain else 'ai.onnx'

        if domain not in operators or domain not in domain_opset_map:
            continue

        operators[domain][domain_opset_map[domain]].add(operator.op_type)

        for attr in operator.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:  # process subgraph
                _extract_ops_from_onnx_graph(attr.g, operators, domain_opset_map)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                # Currently no ONNX operators use GRAPHS.
                # Fail noisily if we encounter this so we can implement support
                raise RuntimeError('Unexpected attribute proto of GRAPHS')


def _process_onnx_model(model_path, required_ops):
    model = onnx.load(model_path)

    # create map of domain to opset for the model
    domain_opset_map = {}
    for opset in model.opset_import:
        # empty domain == ai.onnx
        domain = opset.domain if opset.domain else 'ai.onnx'
        domain_opset_map[domain] = opset.version

        if domain not in required_ops:
            required_ops[domain] = {opset.version: set()}
        elif opset.version not in required_ops[domain]:
            required_ops[domain][opset.version] = set()

    # check the model imports at least one opset. if it does not it's an unexpected edge case that we have to ignore
    # as we don't know what opset nodes in the graph belong to.
    if domain_opset_map:
        _extract_ops_from_onnx_graph(model.graph, required_ops, domain_opset_map)


def _extract_ops_from_onnx_model(model_path_or_dir):
    '''Extract ops from a single ONNX model, or all ONNX models found by recursing model_path_or_dir'''

    if not os.path.exists(model_path_or_dir):
        raise ValueError('Path to model/s does not exist: {}'.format(model_path_or_dir))

    required_ops = {}

    if os.path.isfile(model_path_or_dir):
        _process_onnx_model(model_path_or_dir, required_ops)
    else:
        for root, _, files in os.walk(model_path_or_dir):
            for file in files:
                if file.lower().endswith('.onnx'):
                    model_path = os.path.join(root, file)
                    _process_onnx_model(model_path, required_ops)

    return required_ops


def create_config_from_onnx_models(model_path_or_dir: str, output_file: str):

    required_ops = _extract_ops_from_onnx_model(model_path_or_dir)

    directory, filename = os.path.split(output_file)
    if not filename:
        raise RuntimeError("Invalid output path for configuation: {}".format(output_file))

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(output_file, 'w') as out:
        out.write("# Generated from ONNX models path of {}\n".format(model_path_or_dir))

        for domain in sorted(required_ops.keys()):
            for opset in sorted(required_ops[domain].keys()):
                ops = required_ops[domain][opset]
                if ops:
                    out.write("{};{};{}\n".format(domain, opset, ','.join(sorted(ops))))


def main():
    argparser = argparse.ArgumentParser(
        'Script to create a reduced build config file from either ONNX or ORT format model/s. '
        'See /docs/Reduced_Operator_Kernel_build.md for more information on the configuration file format.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument('-f', '--format', choices=['ONNX', 'ORT'], default='ONNX',
                           help='Format of model/s to process.')
    argparser.add_argument('-t', '--enable_type_reduction', action='store_true',
                           help='Enable tracking of the specific types that individual operators require. '
                                'Operator implementations MAY support limiting the type support included in the build '
                                'to these types. Only possible with ORT format models.')
    argparser.add_argument('model_path_or_dir', type=pathlib.Path,
                           help='Path to a single model, or a directory that will be recursively searched '
                                'for models to process.')

    argparser.add_argument('config_path', nargs='?', type=pathlib.Path, default=None,
                           help='Path to write configuration file to. Default is to write to required_operators.config '
                                'or required_operators_and_types.config in the same directory as the models.')

    args = argparser.parse_args()

    if args.enable_type_reduction and args.format == 'ONNX':
        print('Type reduction requires model format to be ORT.', file=sys.stderr)
        sys.exit(-1)

    model_path_or_dir = args.model_path_or_dir.resolve()
    if args.config_path:
        config_path = args.config_path.resolve()
    else:
        config_path = model_path_or_dir if model_path_or_dir.is_dir() else model_path_or_dir.parent

    if config_path.is_dir():
        filename = 'required_operators_and_types.config' if args.enable_type_reduction else 'required_operators.config'
        config_path = config_path.joinpath(filename)

    if args.format == 'ONNX':
        create_config_from_onnx_models(model_path_or_dir, config_path)
    else:
        from util.ort_format_model import create_config_from_models as create_config_from_ort_models
        create_config_from_ort_models(model_path_or_dir, config_path, args.enable_type_reduction)

        # Debug code to validate that the config parsing matches
        # from util import parse_config
        # required_ops, op_type_usage_processor, _ = parse_config(args.config_path, True)
        # op_type_usage_processor.debug_dump()


if __name__ == "__main__":
    main()
