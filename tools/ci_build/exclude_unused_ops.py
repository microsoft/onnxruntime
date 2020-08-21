'''exclude unused ops from build'''
# !/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import argparse
import shutil
import onnx
from onnx import AttributeProto as AP
from logger import log


domain_map = {'': 'kOnnxDomain',
              'ai.onnx': 'kOnnxDomain',
              'ai.onnx.ml': 'kMLDomain',
              'com.microsoft': 'kMSDomain',
              'com.microsoft.nchwc': 'kMSNchwcDomain',
              'com.microsoft.mlfeaturizers': 'kMSFeaturizersDomain',
              'com.microsoft.dml': 'kMSDmlDomain',
              'com.intel.ai': 'kNGraphDomain',
              'com.xilinx': 'kVitisAIDomain'}


def map_domain(domain):

    if domain in domain_map:
        return domain_map[domain]

    return 'UnknownDomain'


def extract_ops_from_file(file_path, referred_ops):
    '''extract ops from file of format: domain;opset;op1,op2...'''

    if not file_path:
        return referred_ops

    if not os.path.isfile(file_path):
        log.warning('File {} does not exist'.format(file_path))
        return referred_ops

    with open(file_path, 'r') as file_to_read:

        for stripped_line in [line.strip() for line in
                              file_to_read.readlines()]:

            if stripped_line.startswith("#"):  # skip comments
                continue

            raw_domain, raw_opset, raw_ops =\
                [segment.strip() for segment in stripped_line.split(';')]

            domain = map_domain(raw_domain)
            opset = int(raw_opset)
            operators = set([raw_op.strip() for raw_op in raw_ops.split(',')])

            if domain not in referred_ops:
                referred_ops[domain] = {opset: operators}

            elif opset not in referred_ops[domain]:
                referred_ops[domain][opset] = operators

            else:
                referred_ops[domain][opset].update(operators)

    return referred_ops  # end of extract_ops_from_file(...)


def extract_ops_from_model(model_path, referred_ops):
    '''extract ops from models under model_path and return a diction'''

    if not model_path:
        return referred_ops

    if not os.path.isdir(model_path):
        log.warning('Directory {} does not exist'.format(model_path))
        return referred_ops

    def extract_ops_from_graph(graph, operators, domain_opset_map):
        '''extract ops from graph and all subgraphs'''

        for operator in graph.node:

            mapped_domain = map_domain(operator.domain)
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

                for opset in model.opset_import:

                    mapped_domain = map_domain(opset.domain)
                    domain_opset_map[mapped_domain] = opset.version

                    if mapped_domain not in referred_ops:
                        referred_ops[mapped_domain] = {opset.version: set()}

                    elif opset.version not in referred_ops[mapped_domain]:
                        referred_ops[mapped_domain][opset.version] = set()

                extract_ops_from_graph(model.graph, referred_ops, domain_opset_map)

    return referred_ops  # end of extract_ops_from_model(...)


def exclude_unused_ops(model_path, file_path, provider_paths):
    '''rewrite multiple provider files'''

    operators = extract_ops_from_file(file_path, extract_ops_from_model(model_path, {}))
    for provider_path in provider_paths:
        exclude_unused_ops_in_provider(operators, provider_path)

    # end of disable_ops_in_providers(...)


def exclude_unused_ops_in_provider(operators, provider_path):
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


def get_provider_path(ort_root='', use_cuda=False):
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


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(
        description="provider rewriter",
        usage="""
        --model_path <path to model(s) folder>
        --file_path <path to file whose line formated like 'domain;opset;op1,op2...'>
        --ort_root <path to ort root with current as default>
        """)

    PARSER.add_argument(
        "--model_path", type=str, help="path to model(s) folder")

    PARSER.add_argument(
        "--file_path", type=str, help="path to file of ops")

    PARSER.add_argument(
        "--ort_root", type=str, help="path to ort root with current as default")

    ARGS = PARSER.parse_args()

    model_path = os.path.abspath(ARGS.model_path) if ARGS.model_path else ''
    file_path = os.path.abspath(ARGS.file_path) if ARGS.file_path else ''
    ort_root = os.path.abspath(ARGS.ort_root) if ARGS.ort_root else ''

    if not model_path and not file_path:
        log.warning('Please specify at least either model path or file path.')

    if not ort_root:
        log.info('ort root not specified, taking current as root')

    exclude_unused_ops(model_path, file_path,
                       get_provider_path(ort_root, use_cuda=True))
