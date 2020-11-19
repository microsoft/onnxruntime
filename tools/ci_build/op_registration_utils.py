# !/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
'''
Utilities to help process files containing kernel registrations.
'''

import os
import sys
import typing
import re

from logger import get_logger

log = get_logger("op_registration_utils")

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


def map_domain(domain):

    if domain in domain_map:
        return domain_map[domain]

    log.warning("Attempt to map unknown domain of {}".format(domain))
    return 'UnknownDomain'


def get_kernel_registration_files(ort_root=None, include_cuda=False):
    '''
    Return paths to files containing kernel registrations for CPU and CUDA providers.
    :param ort_root: ORT repository root directory. Inferred from the location of this script if not provided.
    :param include_cuda: Include the CUDA registrations in the list of files.
    :return: list[str] containing the kernel registration filenames.
    '''

    if not ort_root:
        ort_root = os.path.dirname(os.path.abspath(__file__)) + '/../..'

    provider_path = ort_root + '/onnxruntime/core/providers/{ep}/{ep}_execution_provider.cc'
    contrib_provider_path = ort_root + '/onnxruntime/contrib_ops/{ep}/{ep}_contrib_kernels.cc'
    provider_paths = [provider_path.format(ep='cpu'),
                      contrib_provider_path.format(ep='cpu')]

    if include_cuda:
        provider_paths.append(provider_path.format(ep='cuda'))
        provider_paths.append(contrib_provider_path.format(ep='cuda'))

    provider_paths = [os.path.abspath(p) for p in provider_paths]

    return provider_paths


def get_nnapi_registration_files(ort_root=None):
    '''
    Return paths to files containing operator registrations for NNAPI EP.
    :param ort_root: ORT repository root directory. Inferred from the location of this script if not provided.
    :return: list[str] containing the operator registration filenames.
    '''
    if not ort_root:
        ort_root = os.path.dirname(os.path.abspath(__file__)) + '/../..'

    nnapi_path = ort_root + '/onnxruntime/core/providers/nnapi/nnapi_builtin/builders/'
    nnapi_provider_paths = [nnapi_path + 'op_builder.cc',
                            nnapi_path + 'op_support_checker.cc']

    nnapi_provider_paths = [os.path.abspath(p) for p in nnapi_provider_paths]
    return nnapi_provider_paths


class RegistrationProcessor:
    '''
    Class to process lines that are extracted from a kernel registration file.
    For each kernel registration, process_registration is called.
    For all other lines, process_other_line is called.
    '''

    def process_registration(self, lines: typing.List[str], domain: str, operator: str,
                             start_version: int, end_version: int = None, input_type: str = None):
        '''
        Process lines that contain a kernel registration.
        :param lines: Array containing the original lines containing the kernel registration.
        :param domain: Domain for the operator
        :param operator: Operator type
        :param start_version: Start version
        :param end_version: End version or None if unversioned registration
        :param input_type: Type if this is a typed registration
        '''
        pass

    def process_other_line(self, line):
        '''
        Process a line that does not contain a kernel registation
        :param line: Original line
        '''
        pass

    def ok(self):
        '''
        Get overall status for processing
        :return: True if successful. False if not. Error will be logged as the registrations are processed.
        '''
        return False  # return False as the derived class must override to report the real status


def _process_lines(lines: typing.List[str], offset: int, registration_processor: RegistrationProcessor):
    '''
    Process one or more lines that contain a kernel registration.
    Merge lines if split over multiple, and call registration_processor.process_registration with the original lines
    and the registration information.
    :return: Offset for first line that was not consumed.
    '''

    onnx_op = 'ONNX_OPERATOR_KERNEL_CLASS_NAME'
    onnx_op_len = len(onnx_op)
    onnx_typed_op = 'ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME'
    onnx_typed_op_len = len(onnx_typed_op)
    onnx_versioned_op = 'ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME'
    onnx_versioned_op_len = len(onnx_versioned_op)
    onnx_versioned_typed_op = 'ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME'
    onnx_versioned_typed_op_len = len(onnx_versioned_typed_op)
    end_marks = tuple([');', ')>', ')>,', ')>,};', ')>};'])

    end_mark = ''
    lines_to_process = []

    # merge line if split over multiple.
    # original lines will be in lines_to_process. merged and stripped line will be in code_line
    while True:
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

        offset += 1
        if offset > len(lines):
            log.error('Past end of input lines looking for line terminator.')
            sys.exit(-1)

    code_line = ''.join([line.strip() for line in lines_to_process])

    if onnx_op in code_line:
        # e.g. BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(
        #          kCpuExecutionProvider, kOnnxDomain, 7, Cos)>,
        trim_at = code_line.index(onnx_op) + onnx_op_len + 1
        *_, domain, start_version, op_type = \
            [arg.strip() for arg in code_line[trim_at: -len(end_mark)].split(',')]

        registration_processor.process_registration(lines_to_process, domain, op_type,
                                                    int(start_version), None, None)

    elif onnx_typed_op in code_line:
        # e.g. BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
        #          kCpuExecutionProvider, kOnnxDomain, 7, double, Sin)>,
        trim_at = code_line.index(onnx_typed_op) + onnx_typed_op_len + 1
        *_, domain, start_version, input_type, op_type = \
            [arg.strip() for arg in code_line[trim_at: -len(end_mark)].split(',')]
        registration_processor.process_registration(lines_to_process, domain, op_type,
                                                    int(start_version), None, input_type)

    elif onnx_versioned_op in code_line:
        # e.g. BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
        #          kCpuExecutionProvider, kOnnxDomain, 1, 10, Hardmax)>,
        trim_at = code_line.index(onnx_versioned_op) + onnx_versioned_op_len + 1
        *_, domain, start_version, end_version, op_type = \
            [arg.strip() for arg in code_line[trim_at: -len(end_mark)].split(',')]
        registration_processor.process_registration(lines_to_process, domain, op_type,
                                                    int(start_version), int(end_version), None)

    elif onnx_versioned_typed_op in code_line:
        # e.g. BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
        #          kCpuExecutionProvider, kOnnxDomain, 1, 10, float, LogSoftmax)>,
        trim_at = code_line.index(onnx_versioned_typed_op) + onnx_versioned_typed_op_len + 1
        *_, domain, start_version, end_version, input_type, op_type = \
            [arg.strip() for arg in code_line[trim_at: -len(end_mark)].split(',')]
        registration_processor.process_registration(lines_to_process, domain, op_type,
                                                    int(start_version), int(end_version), input_type)

    return offset + 1


def _process_nnapi_lines(lines: typing.List[str], offset: int, registration_processor: RegistrationProcessor):
    end_mark = ';'
    lines_to_process = []

    # merge line if split over multiple.
    # original lines will be in lines_to_process. merged and stripped line will be in code_line
    while True:
        lines_to_process.append(lines[offset])
        stripped = lines[offset].strip()

        if stripped.endswith(end_mark):
            break

        offset += 1
        if offset > len(lines):
            log.error('Past end of input lines looking for line terminator.')
            sys.exit(-1)

    code_line = ''.join([line.strip() for line in lines_to_process])

    line_match_pattern = ('^NNAPI_EP_ADD_(SHARED_OP_BUILDER|SINGLE_OP_BUILDER|'
                          'SHARED_OP_SUPPORT_CHECKER|SINGLE_OP_SUPPORT_CHECKER)\\(\\"\\s*(\\w+)\\"')

    print(line_match_pattern)

    match = re.match(line_match_pattern, code_line)
    if match:
        op_type = match.group(2)
        registration_processor.process_registration(lines_to_process, None, op_type, None, None, None)

    return offset + 1


def process_kernel_registration_file(filename: str, registration_processor: RegistrationProcessor, use_nnapi=False):
    '''
    Process a kernel registration file using registration_processor.
    :param filename: Path to file containing kernel registrations.
    :param registration_processor: Processor to be used.
    :param use_nnapi: Process nnapi registration files if set true, otherwise process cpu[/cuda] registration files
    :return True if processing was successful.
    '''

    if not os.path.isfile(filename):
        log.error('File not found: {}'.format(filename))
        return False

    lines = []
    with open(filename, 'r') as file_to_read:
        lines = file_to_read.readlines()

    offset = 0

    if use_nnapi:
        line_start = 'NNAPI_EP_ADD_'
    else:
        line_start = 'BuildKernelCreateInfo<ONNX'

    while offset < len(lines):

        line = lines[offset]
        stripped = line.strip()

        if stripped.startswith(line_start):
            if use_nnapi:
                offset = _process_nnapi_lines(lines, offset, registration_processor)
            else:
                offset = _process_lines(lines, offset, registration_processor)

        else:
            registration_processor.process_other_line(line)
            offset += 1
