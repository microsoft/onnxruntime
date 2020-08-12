#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
import onnx
from onnx import AttributeProto as AP
from logger import log

#pylint: disable=no-member,too-many-locals,too-many-statements

def extract_ops_from_file(file_path, referred_ops):
    '''extract ops from csv file with format:
        op_type,domain,opset'''

    if not os.path.isfile(file_path):
        log.warning('File {} does not exist'.format(file_path))
        return referred_ops

    with open(file_path, 'r') as csv_to_read:

        for line in csv_to_read.readlines():
            op_type, domain, raw_opset = line.strip().split(',')
            opset = int(raw_opset)

            if op_type not in referred_ops:
                referred_ops[op_type] = {domain:[opset]}

            elif domain not in referred_ops[op_type]:
                referred_ops[op_type][domain] = [opset]

            elif opset not in referred_ops[op_type][domain]:
                referred_ops[op_type][domain].append(opset)

    return referred_ops #end of extract_ops_from_file(...)


def extract_ops_from_model(model_path, referred_ops):
    '''extract ops from models under model_path and return a diction'''

    if not os.path.isdir(model_path):
        log.warning('Directory {} does not exist'.format(model_path))
        return referred_ops

    def map_domain(domain):
        if domain == 'ai.onnx.ml':
            return 'kMLDomain'
        if domain == 'com.microsoft':
            return 'kMSDomain'
        return 'kOnnxDomain'

    def extract_ops_from_graph(graph, opsets, operators):
        '''extract ops from graph and all subgraphs'''

        for operator in graph.node:

            if operator.op_type not in operators:
                operators[operator.op_type] = {}

            for opset in opsets:
                mapped_domain = map_domain(opset.domain)
                if mapped_domain not in operators[operator.op_type]:
                    operators[operator.op_type][mapped_domain] = []
                if opset.version not in operators[operator.op_type][mapped_domain]:
                    operators[operator.op_type][mapped_domain].append(opset.version)

            for attr in operator.attribute:

                if attr.type == AP.GRAPH: #process subgraph
                    extract_ops_from_graph(attr.g, opsets, operators)

    #end of extract_ops_from_graph(...)


    for root, _, files in os.walk(model_path):
        for file in files:
            if file.endswith('.onnx'):
                model_path = os.path.join(root, file)
                model = onnx.load(model_path)
                extract_ops_from_graph(model.graph, model.opset_import, referred_ops)

    return referred_ops #end of extract_ops_from_model(...)


def rewrite_provider(model_path, file_path, ep_path):
    '''rewrite provider file to exclude unused ops'''

    onnx_op = 'ONNX_OPERATOR_KERNEL_CLASS_NAME'
    onnx_op_len = len(onnx_op)
    onnx_typed_op = 'ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME'
    onnx_typed_op_len = len(onnx_typed_op)
    onnx_versioned_op = 'ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME'
    onnx_versioned_op_len = len(onnx_versioned_op)
    onnx_versioned_typed_op = 'ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME'
    onnx_versioned_typed_op_len = len(onnx_versioned_typed_op)
    version_map = {} #{op:{domain:[opset1, opset2, opset3 ...]}
    operators = extract_ops_from_file(file_path, extract_ops_from_model(model_path, {}))

    def fill_version_map(op_type, opset_from, opset_to, domain):
        '''callback func to register op in version_map'''

        if op_type in version_map:
            if domain in version_map[op_type]:

                if opset_from not in version_map[op_type][domain]:
                    version_map[op_type][domain].append(opset_from)

                if opset_to not in version_map[op_type][domain]:
                    version_map[op_type][domain].append(opset_to)
            else:
                version_map[op_type][domain] =\
                    [opset_from, opset_to] if opset_from < opset_to else [opset_from]

        else:
            version_map[op_type] =\
                {domain: [opset_from, opset_to] if opset_from < opset_to else [opset_from]}

        return True #end of fill_version_map(...)


    def disable_op(op_type, opset_from, opset_to, domain):
        '''callback func to check if the op is in ops'''

        if op_type in operators and domain in operators[op_type]:
            opset_to_index = version_map[op_type][domain].index(opset_to)

            if opset_to_index == len(version_map[op_type][domain]) - 1:
                opset_to = 9999 #if is the latest, extend to unlimited

            elif opset_from == opset_to:
                opset_to = version_map[op_type][domain][opset_to_index + 1]

            for opset in range(opset_from, opset_to + 1):
                if opset in operators[op_type][domain]:
                    return False #do not comment

        return True #end of disable_op(...)


    def process_lines(lines, offset, end_mark, call_back):
        '''extract op info from a logic code line start from offset to the line end
           with end_mark, then trigger callback(op_type, opset_from, opset_to, domain)
           return next line offset and whether current lines are disabled
        '''

        lines_to_process = []
        while True: #collect the logical code line
            lines_to_process.append(lines[offset])
            stripped = lines[offset].strip()
            if stripped.endswith(end_mark):
                break
            offset += 1
        code_line = ''.join([line.strip() for line in lines_to_process])

        call_back_ret = False
        if onnx_op in code_line:
            #e.g. class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, Transpose);
            #e.g. BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, Cos)>,
            trim_at = code_line.index(onnx_op) + onnx_op_len
            *_, domain, opset, op_type =\
                [arg.strip() for arg in code_line[trim_at: -len(end_mark)].split(',')]
            call_back_ret = call_back(op_type, int(opset), int(opset), domain)

        elif onnx_typed_op in code_line:
            #e.g. class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, float, Expand);
            #e.g. BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, double, Sin)>,
            trim_at = code_line.index(onnx_typed_op) + onnx_typed_op_len
            *_, domain, opset, _, op_type =\
                [arg.strip() for arg in code_line[trim_at: -len(end_mark)].split(',')]
            call_back_ret = call_back(op_type, int(opset), int(opset), domain)

        elif onnx_versioned_op in code_line:
            #e.g. class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, Unsqueeze);
            #e.g. BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, Hardmax)>,
            trim_at = code_line.index(onnx_versioned_op) + onnx_versioned_op_len
            *_, domain, opset_from, opset_to, op_type =\
                [arg.strip() for arg in code_line[trim_at: -len(end_mark)].split(',')]
            call_back_ret = call_back(op_type, int(opset_from), int(opset_to), domain)

        elif onnx_versioned_typed_op in code_line:
            #e.g. class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 9, float, Upsample);
            #e.g. BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, float, LogSoftmax)>,
            trim_at = code_line.index(onnx_versioned_typed_op) + onnx_versioned_typed_op_len
            *_, domain, opset_from, opset_to, _, op_type =\
                [arg.strip() for arg in code_line[trim_at: -len(end_mark)].split(',')]
            call_back_ret = call_back(op_type, int(opset_from), int(opset_to), domain)

        return offset + 1, call_back_ret #end of process_lines(...)


    lines = []
    with open(ep_path, 'r') as file_to_read:
        lines = file_to_read.readlines()

    backup_path = ep_path + '.bak'
    if not os.path.isfile(backup_path):
        shutil.move(ep_path, backup_path)

    with open(ep_path, 'w') as file_to_write:
        line_offset = 0

        while line_offset < len(lines):

            line = lines[line_offset]
            stripped = line.strip()

            if stripped.startswith('class ONNX_OPERATOR'):
                #collection versions of ops

                next_line_offset, _ = process_lines(lines,
                                                    line_offset,
                                                    ');',
                                                    fill_version_map)

                for index in range(line_offset, next_line_offset):
                    file_to_write.write(lines[index]) #leave as it was

                line_offset = next_line_offset

            elif stripped.startswith('BuildKernelCreateInfo<ONNX'):
                #comment out unused ops

                next_line_offset, disabled = process_lines(lines,
                                                           line_offset,
                                                           ')>,',
                                                           disable_op)

                for index in range(line_offset, next_line_offset):
                    if disabled: #comment out unused
                        file_to_write.write('//' + lines[index])

                    else: #leave as it was
                        file_to_write.write(lines[index])

                line_offset = next_line_offset

            else: #leave as it was
                file_to_write.write(line)
                line_offset += 1

    #end of rewrite_cpu_provider(...)
