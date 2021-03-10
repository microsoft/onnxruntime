# !/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import op_registration_utils
import os
import shutil
import sys
import typing

from logger import get_logger

# add the path to /tools/python so we can import the config parsing and type reduction processing
script_path = os.path.dirname(os.path.realpath(__file__))
ort_root = os.path.abspath(os.path.join(script_path, '..', '..', ))
ort_tools_py_path = os.path.abspath(os.path.join(ort_root, 'tools', 'python'))
sys.path.append(ort_tools_py_path)

from util import parse_config  # noqa
from util.ort_format_model.operator_type_usage_processors import OpTypeImplFilterInterface  # noqa

log = get_logger("reduce_op_kernels")


class _ExcludingRegistrationProcessor(op_registration_utils.RegistrationProcessor):
    '''Registration processor that excludes registrations and writes the result to an output file.'''
    def __init__(self, required_ops: typing.Optional[dict],
                 op_type_impl_filter: typing.Optional[OpTypeImplFilterInterface],
                 output_file: str):
        self._required_ops = required_ops
        self._op_type_impl_filter = op_type_impl_filter
        self._output_file = output_file

    def _is_op_required(self, domain: str, operator: str,
                        start_version: int, end_version: typing.Optional[int]) -> typing.Tuple[bool, str]:
        '''See if an op is required.'''
        if self._required_ops is None:
            return True

        if domain not in self._required_ops:
            return False

        for opset in self._required_ops[domain]:
            if opset >= start_version and (end_version is None or opset <= end_version):
                if operator in self._required_ops[domain][opset]:
                    return True

        return False

    def process_registration(self, lines: typing.List[str], constant_for_domain: str, operator: str,
                             start_version: int, end_version: typing.Optional[int] = None,
                             type: typing.Optional[str] = None):
        registration_identifier = '{}:{}({}){}'.format(constant_for_domain, operator, start_version,
                                                       '<{}>'.format(type) if type else '')

        # convert from the ORT constant name to the domain string used in the config
        domain = op_registration_utils.map_ort_constant_to_domain(constant_for_domain)

        exclude = False
        reason = ""

        if domain is not None:
            if not self._is_op_required(domain, operator, start_version, end_version):
                exclude = True
                reason = "Entire op is not required."

            if not exclude and type is not None and self._op_type_impl_filter is not None:
                if not self._op_type_impl_filter.is_typed_registration_needed(domain, operator, type):
                    exclude = True
                    reason = "Specific typed registration is not required."
        else:
            log.warning('Keeping {} registration from unknown domain: {}'
                        .format(registration_identifier, constant_for_domain))

        if exclude:
            log.info('Disabling {} registration: {}'.format(registration_identifier, reason))
            for line in lines:
                self._output_file.write('// ' + line)

            # edge case of last entry in table where we still need the terminating }; to not be commented out
            if lines[-1].rstrip().endswith('};'):
                self._output_file.write('};\n')
        else:
            for line in lines:
                self._output_file.write(line)

    def process_other_line(self, line):
        self._output_file.write(line)

    def ok(self):
        return True


def _process_provider_registrations(
        ort_root: str, use_cuda: bool,
        required_ops: typing.Optional[dict],
        op_type_impl_filter: typing.Optional[OpTypeImplFilterInterface]):
    '''Rewrite provider registration files.'''
    kernel_registration_files = op_registration_utils.get_kernel_registration_files(ort_root, use_cuda)

    for kernel_registration_file in kernel_registration_files:
        if not os.path.isfile(kernel_registration_file):
            raise ValueError('Kernel registration file {} does not exist'.format(kernel_registration_file))

        log.info("Processing {}".format(kernel_registration_file))

        backup_path = kernel_registration_file + '~'
        shutil.move(kernel_registration_file, backup_path)

        # read from backup and overwrite original with commented out lines for any kernels that are not required
        with open(kernel_registration_file, 'w') as file_to_write:
            processor = _ExcludingRegistrationProcessor(required_ops, op_type_impl_filter, file_to_write)

            op_registration_utils.process_kernel_registration_file(backup_path, processor)

            if not processor.ok():
                # error should have already been logged so just exit
                sys.exit(-1)


def _insert_type_control_cpp_code(ort_root: str, cpp_lines: typing.Sequence[str]):
    '''
    Insert the C++ code to specify operator type requirements.
    :param ort_root: Root of the ONNX Runtime repository
    :param cpp_lines: The C++ code to insert
    '''
    if not cpp_lines:
        return

    target = os.path.join(ort_root, 'onnxruntime', 'core', 'providers', 'op_kernel_type_control_overrides.inc')
    if not os.path.exists(target) or not os.path.isfile(target):
        log.warning('Could not find {}. Skipping generation of C++ code to reduce the types supported by operators.'
                    .format(target))
        return

    # copy existing content to use as input
    src = target + '.tmp'
    shutil.copyfile(target, src)

    # find the insertion block and replace any existing content in it
    inserted = False
    with open(src, 'r') as input, open(target, 'w') as output:
        inside_insertion_block = False
        for line in input.readlines():
            if '@@insertion_point_begin(allowed_types)@@' in line:
                inside_insertion_block = True
                output.write(line)
                [output.write('{}\n'.format(code_line)) for code_line in cpp_lines]
                inserted = True
                continue
            elif inside_insertion_block:
                if '@@insertion_point_end(allowed_types)@@' in line:
                    inside_insertion_block = False
                else:
                    # we ignore any old lines within the insertion block
                    continue

            output.write(line)

    os.remove(src)

    if not inserted:
        raise RuntimeError('Insertion point was not found in {}'.format(target))


def reduce_ops(config_path: str, enable_type_reduction: bool = False, use_cuda: bool = True):
    '''
    Reduce op kernel implementations.
    :param config_path: Path to configuration file that specifies the ops to include
    :param enable_type_reduction: Whether per operator type reduction is enabled
    :param use_cuda: Whether to reduce op kernels for the CUDA provider
    '''
    required_ops, op_type_impl_filter = parse_config(config_path, enable_type_reduction)

    _process_provider_registrations(ort_root, use_cuda, required_ops, op_type_impl_filter)

    type_control_cpp_code = op_type_impl_filter.get_cpp_entries() if op_type_impl_filter is not None else []

    _insert_type_control_cpp_code(ort_root, type_control_cpp_code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reduces operator kernel implementations in ONNX Runtime. "
                    "Entire op implementations or op implementations for specific types may be pruned.")

    parser.add_argument("config_path", type=str,
                        help="Path to configuration file. "
                             "Create with <ORT root>/tools/python/create_reduced_build_config.py and edit if needed. "
                             "See /docs/ONNX_Runtime_Format_Model_Usage.md for more information.")

    args = parser.parse_args()
    config_path = os.path.abspath(args.config_path)
    reduce_ops(config_path, enable_type_reduction=True, use_cuda=True)
