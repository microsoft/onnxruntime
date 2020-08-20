'''rewrite execution providers to disable ops'''
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

            raw_domain, opset, raw_ops =\
                [segment.strip() for segment in stripped_line.split(';')]

            domain = map_domain(raw_domain)
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

    def extract_ops_from_graph(graph, operators):
        '''extract ops from graph and all subgraphs'''

        for operator in graph.node:
            operators.add(operator.op_type)
            for attr in operator.attribute:
                if attr.type == AP.GRAPH:  # process subgraph
                    extract_ops_from_graph(attr.g, operators)

    # end of extract_ops_from_graph(...)

    for root, _, files in os.walk(model_path):
        for file in files:

            if file.endswith('.onnx'):
                model_path = os.path.join(root, file)
                model = onnx.load(model_path)

                all_ops = set()
                extract_ops_from_graph(model.graph, all_ops)

                for opset in model.opset_import:

                    opset_version = str(opset.version)
                    mapped_domain = map_domain(opset.domain)

                    if mapped_domain not in referred_ops:
                        referred_ops[mapped_domain] = {opset_version: all_ops}

                    elif opset_version not in referred_ops[mapped_domain]:
                        referred_ops[mapped_domain][opset_version] = all_ops

                    else:
                        referred_ops[mapped_domain][opset_version].update(all_ops)

    return referred_ops  # end of extract_ops_from_model(...)


def disable_ops_in_providers(model_path, file_path, ep_paths):
    '''rewrite multiple provider files'''

    operators = extract_ops_from_file(file_path, extract_ops_from_model(model_path, {}))
    for ep_path in ep_paths:
        disable_ops_in_provider(operators, ep_path)

    # end of disable_ops_in_providers(...)


def disable_ops_in_provider(operators, ep_path):
    '''rewrite provider file to exclude unused ops'''

    log.info("Rewriting {}".format(ep_path))
    onnx_op = 'ONNX_OPERATOR_KERNEL_CLASS_NAME'
    onnx_op_len = len(onnx_op)
    onnx_typed_op = 'ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME'
    onnx_typed_op_len = len(onnx_typed_op)
    onnx_versioned_op = 'ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME'
    onnx_versioned_op_len = len(onnx_versioned_op)
    onnx_versioned_typed_op = 'ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME'
    onnx_versioned_typed_op_len = len(onnx_versioned_typed_op)
    version_map = {}  # {domain:{op:[v1, v2, v3 ...]}}

    def fill_version_map(op_type, opset_from, opset_to, domain):
        '''callback func to register op in version_map'''

        opset_from = int(opset_from)
        opset_to = int(opset_to)

        if domain not in version_map:
            version_map[domain] = {}

        if op_type not in version_map[domain]:
            version_map[domain][op_type] =\
                [opset_from, opset_to] if opset_from != opset_to else [opset_from]

        else:
            if opset_from not in version_map[domain][op_type]:
                version_map[domain][op_type].append(opset_from)

            if opset_to not in version_map[domain][op_type]:
                version_map[domain][op_type].append(opset_to)

        version_map[domain][op_type].sort()  # make sure it goes up
        return True  # end of fill_version_map(...)

    def disable_op(op_type, opset_from, opset_to, domain):
        '''callback func to check if the op is in ops'''

        def find_first_bigger(vector, elem):
            '''return index of first element that is bigger than a'''
            start_at = 0
            end_at = len(vector)
            while start_at < end_at:
                mid_at = start_at + (end_at-start_at >> 1)
                if vector[mid_at] > elem:
                    end_at = mid_at
                else:
                    start_at = mid_at + 1
            return end_at  # end of find_first_bigger(...)

        if domain not in operators or\
           op_type not in version_map[domain]:
            log.info("Disable {0} {1} {2} {3}".format(domain, opset_from, opset_to, op_type))
            return True

        opset_from = int(opset_from)
        raw_opset_to = opset_to
        opset_to = int(opset_to)

        first_bigger_at = find_first_bigger(version_map[domain][op_type], opset_to)
        if first_bigger_at == len(version_map[domain][op_type]):
            opset_to = 999
        elif opset_from == opset_to or opset_to not in version_map[domain][op_type]:
            opset_to = version_map[domain][op_type][first_bigger_at]

        for opset in [str(i) for i in range(opset_from, opset_to+1)]:
            if opset in operators[domain] and\
               op_type in operators[domain][opset]:
                return False

        log.info("Disable {0} {1} {2} {3}".format(domain, opset_from, raw_opset_to, op_type))
        return True  # end of disable_op(...)

    def process_lines(lines, offset, end_marks, call_back):
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

        call_back_ret = False
        if onnx_op in code_line:
            # e.g. class ONNX_OPERATOR_KERNEL_CLASS_NAME(
            #          kCpuExecutionProvider, kOnnxDomain, 1, Transpose);
            # e.g. BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(
            #          kCpuExecutionProvider, kOnnxDomain, 7, Cos)>,
            trim_at = code_line.index(onnx_op) + onnx_op_len
            *_, domain, opset, op_type =\
                [arg.strip() for arg in code_line[trim_at: -len(end_mark)].split(',')]
            call_back_ret = call_back(op_type, opset, opset, domain)

        elif onnx_typed_op in code_line:
            # e.g. class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
            #          kCpuExecutionProvider, kOnnxDomain, 8, float, Expand);
            # e.g. BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
            #          kCpuExecutionProvider, kOnnxDomain, 7, double, Sin)>,
            trim_at = code_line.index(onnx_typed_op) + onnx_typed_op_len
            *_, domain, opset, _, op_type =\
                [arg.strip() for arg in code_line[trim_at: -len(end_mark)].split(',')]
            call_back_ret = call_back(op_type, opset, opset, domain)

        elif onnx_versioned_op in code_line:
            # e.g. class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
            #          kCpuExecutionProvider, kOnnxDomain, 1, 10, Unsqueeze);
            # e.g. BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
            #          kCpuExecutionProvider, kOnnxDomain, 1, 10, Hardmax)>,
            trim_at = code_line.index(onnx_versioned_op) + onnx_versioned_op_len
            *_, domain, opset_from, opset_to, op_type =\
                [arg.strip() for arg in code_line[trim_at: -len(end_mark)].split(',')]
            call_back_ret = call_back(op_type, opset_from, opset_to, domain)

        elif onnx_versioned_typed_op in code_line:
            # e.g. class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
            #          kCpuExecutionProvider, kOnnxDomain, 7, 9, float, Upsample);
            # e.g. BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
            #          kCpuExecutionProvider, kOnnxDomain, 1, 10, float, LogSoftmax)>,
            trim_at = code_line.index(onnx_versioned_typed_op) + onnx_versioned_typed_op_len
            *_, domain, opset_from, opset_to, _, op_type =\
                [arg.strip() for arg in code_line[trim_at: -len(end_mark)].split(',')]
            call_back_ret = call_back(op_type, opset_from, opset_to, domain)

        return offset + 1, call_back_ret  # end of process_lines(...)

    lines = []
    with open(ep_path, 'r') as file_to_read:
        lines = file_to_read.readlines()

    backup_path = ep_path + '~'
    if not os.path.isfile(backup_path):
        shutil.move(ep_path, backup_path)

    with open(ep_path, 'w') as file_to_write:
        line_offset = 0

        while line_offset < len(lines):

            line = lines[line_offset]
            stripped = line.strip()

            if stripped.startswith('class ONNX_OPERATOR'):
                # collect op versions

                next_line_offset, _ = process_lines(lines,
                                                    line_offset,
                                                    tuple([');']),
                                                    fill_version_map)

                for index in range(line_offset, next_line_offset):
                    file_to_write.write(lines[index])  # leave as it was

                line_offset = next_line_offset

            elif stripped.startswith('BuildKernelCreateInfo<ONNX'):
                # comment out unused ops

                next_line_offset, disabled = process_lines(lines,
                                                           line_offset,
                                                           tuple([')>', ')>,', ')>,};', ')>};']),
                                                           disable_op)

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


def get_ep_paths(ort_root='', use_cuda=False):
    '''return paths to cpu and cuda providers'''

    if not ort_root:
        ort_root = os.path.dirname(os.path.abspath(__file__)) + '/../..'

    ep_path = ort_root + '/onnxruntime/core/providers/{ep}/{ep}_execution_provider.cc'
    contrib_ep_path = ort_root + '/onnxruntime/contrib_ops/{ep}/{ep}_contrib_kernels.cc'
    ep_paths = [ep_path.format(ep='cpu'),
                contrib_ep_path.format(ep='cpu')]

    if use_cuda:
        ep_paths.append(ep_path.format(ep='cuda'))
        ep_paths.append(contrib_ep_path.format(ep='cuda'))

    return ep_paths  # end of get_ep_paths


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
    ort_root = ARGS.ort_root if ARGS.ort_root else ''

    if not model_path and not file_path:
        log.warning('Please specify at least either model path or file path.')

    if not ort_root:
        log.info('ort root not specified, taking current as root')

    disable_ops_in_providers(model_path,
                             file_path,
                             get_ep_paths(ort_root, use_cuda=True))
