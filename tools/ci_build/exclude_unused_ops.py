'''exclude unused ops from build'''
# !/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sys
import os
import argparse
import shutil
import onnx
from onnx import AttributeProto as AP
from logger import log


domain_map = {'': 'kOnnxDomain',
              'ai.onnx': 'kOnnxDomain',
              'ai.onnx.ml': 'kMLDomain',
              'ai.onnx.training': 'ai.onnx.training',  # we don't have a constant for the training domains currently
              'ai.onnx.preview.training': 'ai.onnx.preview.training',
              'com.microsoft': 'kMSDomain',
              'com.microsoft.nchwc': 'kMSNchwcDomain',
              'com.microsoft.mlfeaturizers': 'kMSFeaturizersDomain',
              'com.microsoft.dml': 'kMSDmlDomain',
              'com.intel.ai': 'kNGraphDomain',
              'com.xilinx': 'kVitisAIDomain'}


def _map_domain(domain):

    if domain in domain_map:
        return domain_map[domain]

    log.warning("Attempt to map unknown domain of {}".format(domain))
    return 'UnknownDomain'


def _extract_ops_from_config(file_path, required_ops):
    '''extract ops from config file of format: domain;opset;op1,op2...'''

    if not file_path:
        return required_ops

    if not os.path.isfile(file_path):
        # exit. to continue may result in unexpectedly disabling all kernels.
        log.error('Configuration file {} does not exist'.format(file_path))
        sys.exit(-1)

    with open(file_path, 'r') as file_to_read:

        for stripped_line in [line.strip() for line in
                              file_to_read.readlines()]:

            if not stripped_line:  # skip empty lines
                continue

            if stripped_line.startswith("#"):  # skip comments
                continue

            raw_domain, raw_opset, raw_ops =\
                [segment.strip() for segment in stripped_line.split(';')]

            domain = _map_domain(raw_domain)
            opset = int(raw_opset)
            operators = set([raw_op.strip() for raw_op in raw_ops.split(',')])

            if domain not in required_ops:
                required_ops[domain] = {opset: operators}

            elif opset not in required_ops[domain]:
                required_ops[domain][opset] = operators

            else:
                required_ops[domain][opset].update(operators)

    return required_ops  # end of extract_ops_from_file(...)


def _extract_ops_from_model(model_path, required_ops):
    '''extract ops from models under model_path and return a diction'''

    if not model_path:
        return required_ops

    if not os.path.isdir(model_path):
        # exit. to continue may result in unexpectedly disabling all kernels.
        log.error('Directory containing models {} does not exist'.format(model_path))
        sys.exit(-1)

    def extract_ops_from_graph(graph, operators, domain_opset_map):
        '''extract ops from graph and all subgraphs'''

        for operator in graph.node:

            mapped_domain = _map_domain(operator.domain)

            if mapped_domain not in operators or\
               mapped_domain not in domain_opset_map:
                continue

            operators[mapped_domain][domain_opset_map[mapped_domain]].add(operator.op_type)

            for attr in operator.attribute:
                if attr.type == AP.GRAPH:  # process subgraph
                    extract_ops_from_graph(attr.g, operators, domain_opset_map)

    # end of extract_ops_from_graph(...)

    for root, _, files in os.walk(model_path):
        for file in files:

            if file.endswith('.onnx'):
                model_path = os.path.join(root, file)
                model = onnx.load(model_path)
                domain_opset_map = {}

                if len(model.opset_import) == 0:
                    continue

                for opset in model.opset_import:

                    mapped_domain = _map_domain(opset.domain)
                    domain_opset_map[mapped_domain] = opset.version

                    if mapped_domain not in required_ops:
                        required_ops[mapped_domain] = {opset.version: set()}

                    elif opset.version not in required_ops[mapped_domain]:
                        required_ops[mapped_domain][opset.version] = set()

                extract_ops_from_graph(model.graph, required_ops, domain_opset_map)

    return required_ops  # end of extract_ops_from_model(...)


def _exclude_unused_ops_in_provider(operators, provider_path):
    '''rewrite provider file to exclude unused ops'''

    if not os.path.isfile(provider_path):
        log.warning('File {} does not exist'.format(provider_path))
        return

    log.info("Processing {}".format(provider_path))
    onnx_op = 'ONNX_OPERATOR_KERNEL_CLASS_NAME'
    onnx_op_len = len(onnx_op)
    onnx_typed_op = 'ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME'
    onnx_typed_op_len = len(onnx_typed_op)
    onnx_versioned_op = 'ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME'
    onnx_versioned_op_len = len(onnx_versioned_op)
    onnx_versioned_typed_op = 'ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME'
    onnx_versioned_typed_op_len = len(onnx_versioned_typed_op)
    end_marks = tuple([');', ')>', ')>,', ')>,};', ')>};'])

    def should_exclude_op(domain, op_type, opset_from, opset_to=None):
        '''check if should exclude the op from build'''

        if domain not in operators:
            return True

        for opset in operators[domain]:
            if opset >= int(opset_from) and (opset_to is None or opset <= int(opset_to)):
                if op_type in operators[domain][opset]:
                    return False  # found a match, do not exclude

        return True  # end of should_exclude_op(...)

    def process_lines(lines, offset):
        '''extract op info from a logic code line start from offset to the line end
           with any of end_marks, then trigger callback(op_type, opset_from, opset_to, domain)
           return next line offset and whether current lines are disabled
        '''

        end_mark = ''
        lines_to_process = []

        while True:  # collect the logical code line

            lines_to_process.append(lines[offset])
            stripped = lines[offset].strip()
            line_end = False

            for mark in end_marks:
                if stripped.endswith(mark):
                    end_mark = mark
                    line_end = True
                    break

            if line_end:
                break

            offset += 1  # end of while

        code_line = ''.join([line.strip() for line in lines_to_process])

        trim_at = 0
        should_exclude = False
        if onnx_op in code_line:
            # e.g. class ONNX_OPERATOR_KERNEL_CLASS_NAME(
            #          kCpuExecutionProvider, kOnnxDomain, 1, Transpose);
            # e.g. BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(
            #          kCpuExecutionProvider, kOnnxDomain, 7, Cos)>,
            trim_at = code_line.index(onnx_op) + onnx_op_len + 1
            *_, domain, opset, op_type =\
                [arg.strip() for arg in code_line[trim_at: -len(end_mark)].split(',')]
            should_exclude = should_exclude_op(domain, op_type, opset, None)

        elif onnx_typed_op in code_line:
            # e.g. class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
            #          kCpuExecutionProvider, kOnnxDomain, 8, float, Expand);
            # e.g. BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
            #          kCpuExecutionProvider, kOnnxDomain, 7, double, Sin)>,
            trim_at = code_line.index(onnx_typed_op) + onnx_typed_op_len + 1
            *_, domain, opset, _, op_type =\
                [arg.strip() for arg in code_line[trim_at: -len(end_mark)].split(',')]
            should_exclude = should_exclude_op(domain, op_type, opset, None)

        elif onnx_versioned_op in code_line:
            # e.g. class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
            #          kCpuExecutionProvider, kOnnxDomain, 1, 10, Unsqueeze);
            # e.g. BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
            #          kCpuExecutionProvider, kOnnxDomain, 1, 10, Hardmax)>,
            trim_at = code_line.index(onnx_versioned_op) + onnx_versioned_op_len + 1
            *_, domain, opset_from, opset_to, op_type =\
                [arg.strip() for arg in code_line[trim_at: -len(end_mark)].split(',')]
            should_exclude = should_exclude_op(domain, op_type, opset_from, opset_to)

        elif onnx_versioned_typed_op in code_line:
            # e.g. class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
            #          kCpuExecutionProvider, kOnnxDomain, 7, 9, float, Upsample);
            # e.g. BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
            #          kCpuExecutionProvider, kOnnxDomain, 1, 10, float, LogSoftmax)>,
            trim_at = code_line.index(onnx_versioned_typed_op) + onnx_versioned_typed_op_len + 1
            *_, domain, opset_from, opset_to, _, op_type =\
                [arg.strip() for arg in code_line[trim_at: -len(end_mark)].split(',')]
            should_exclude = should_exclude_op(domain, op_type, opset_from, opset_to)

        if should_exclude:
            log.info('Disabling: {}'.format(code_line[trim_at: -len(end_mark)]))

        return offset + 1, should_exclude  # end of process_lines(...)

    lines = []
    with open(provider_path, 'r') as file_to_read:
        lines = file_to_read.readlines()

    backup_path = provider_path + '~'
    if not os.path.isfile(backup_path):
        shutil.move(provider_path, backup_path)

    with open(provider_path, 'w') as file_to_write:
        line_offset = 0

        while line_offset < len(lines):

            line = lines[line_offset]
            stripped = line.strip()

            if stripped.startswith('class ONNX_OPERATOR') or\
               stripped.startswith('BuildKernelCreateInfo<ONNX'):

                next_line_offset, disabled = process_lines(lines,
                                                           line_offset)

                for index in range(line_offset, next_line_offset):
                    if disabled:  # comment out unused
                        if lines[index].rstrip().endswith('};'):
                            file_to_write.write('/*' + lines[index].rstrip() + '*/};\n')
                        else:
                            file_to_write.write('//' + lines[index])

                    else:  # leave as it is
                        file_to_write.write(lines[index])

                line_offset = next_line_offset

            else:  # leave as it is
                file_to_write.write(line)
                line_offset += 1

    # end of rewrite_cpu_provider(...)


def _exclude_unused_ops_in_providers(required_operators, provider_paths):
    '''rewrite multiple provider files'''

    for provider_path in provider_paths:
        _exclude_unused_ops_in_provider(required_operators, provider_path)


def _get_provider_paths(ort_root=None, use_cuda=False):
    '''return paths to cpu and cuda providers'''

    if not ort_root:
        ort_root = os.path.dirname(os.path.abspath(__file__)) + '/../..'

    provider_path = ort_root + '/onnxruntime/core/providers/{ep}/{ep}_execution_provider.cc'
    contrib_provider_path = ort_root + '/onnxruntime/contrib_ops/{ep}/{ep}_contrib_kernels.cc'
    provider_paths = [provider_path.format(ep='cpu'),
                      contrib_provider_path.format(ep='cpu')]

    if use_cuda:
        provider_paths.append(provider_path.format(ep='cuda'))
        provider_paths.append(contrib_provider_path.format(ep='cuda'))

    return provider_paths  # end of get_provider_paths


def _create_config_file_with_required_ops(required_operators, model_path, config_path, output_file):

    directory, filename = os.path.split(output_file)
    if not filename:
        log.error("Invalid path to write final config to. {}".format(output_file))
        sys.exit(-1)

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(output_file, 'w') as out:
        model_path_info = '--model_path {} '.format(model_path) if model_path else ''
        config_path_info = '--config_path {}'.format(config_path) if config_path else ''
        out.write("# Generated from {}{}\n".format(model_path_info, config_path_info))

        for domain in sorted(required_operators.keys()):
            if domain == 'UnknownDomain':
                continue

            # reverse the mapping of the domain. entry must exist given we created the required_operators dictionary.
            # also need to handle ai.onnx being special-cased as an empty string
            orig_domain = [key for (key, value) in domain_map.items() if value == domain][0]
            if not orig_domain:
                orig_domain = 'ai.onnx'

            for opset in sorted(required_operators[domain].keys()):
                ops = required_operators[domain][opset]
                if ops:
                    out.write("{};{};{}\n".format(orig_domain, opset, ','.join(sorted(ops))))

    log.info("Wrote set of required operators to {}".format(output_file))


def exclude_unused_ops(models_path, config_path, ort_root=None, use_cuda=True, output_config_path=None):
    '''Determine operators that are used, and either exclude them or create a configuration file that will.
    Note that this called directly from build.py'''

    if not models_path and not config_path:
        log.error('Please specify model_path and/or config_path.')
        sys.exit(-1)

    if not ort_root and not output_config_path:
        log.info('ort_root was not specified. Inferring ONNX Runtime repository root from location of this script.')

    required_ops = _extract_ops_from_config(config_path, _extract_ops_from_model(models_path, {}))

    if not output_config_path:
        _exclude_unused_ops_in_providers(required_ops, _get_provider_paths(ort_root, use_cuda))
    else:
        _create_config_file_with_required_ops(required_ops, models_path, config_path, output_config_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script to exclude unused operator kernels by disabling their registration in ONNXRuntime."
                    "See /docs/Reduced_Operator_Kernel_build.md for more information",
        usage="Provide model_path, config_path, or both, to disable kernel registration of unused kernels.")

    parser.add_argument(
        "--model_path", type=str, help="Path to folder containing one or more ONNX models")

    parser.add_argument(
        "--config_path", type=str, help="Path to configuration file with format of 'domain;opset;op1,op2...'")

    parser.add_argument(
        "--ort_root", type=str, help="Path to ONNXRuntime repository root. "
                                     "Inferred from the location of this script if not provided.")

    parser.add_argument(
        "--write_combined_config_to", type=str,
        help="Optional path to create a configuration file with the combined set of required kernels "
             "from processing --model_path and/or --config_path. If provided, a configuration file will be created "
             "and NO updates will be made to the kernel registrations."
    )

    args = parser.parse_args()

    models_path = os.path.abspath(args.model_path) if args.model_path else ''
    config_path = os.path.abspath(args.config_path) if args.config_path else ''
    ort_root = os.path.abspath(args.ort_root) if args.ort_root else ''

    if not models_path and not config_path:
        log.error('Please specify at least either model path or file path.')
        parser.print_help()
        sys.exit(-1)

    exclude_unused_ops(models_path, config_path, ort_root, use_cuda=True,
                       output_config_path=args.write_combined_config_to)
