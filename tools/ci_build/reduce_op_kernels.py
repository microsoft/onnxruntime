# !/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import op_registration_utils
import os
import re
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
from util.ort_format_model.operator_type_usage_processors import OperatorTypeUsageManager  # noqa

log = get_logger("reduce_op_kernels")


# valid C++ scalar types that can be specified as globally allowed types
_valid_allowed_types = {
    "bool",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "int8_t", "int16_t", "int32_t", "int64_t",
    "MLFloat16", "BFloat16",  # in onnxruntime namespace
    "float", "double",
    "string",  # in std namespace
}


def _validated_globally_allowed_types(globally_allowed_types: typing.Collection[str]) -> typing.Set[str]:
    '''Return a valid set of globally allowed types.'''
    # ensure globally_allowed_types is a set
    if not isinstance(globally_allowed_types, set):
        globally_allowed_types = set(globally_allowed_types)

    if not globally_allowed_types <= _valid_allowed_types:
        raise ValueError(
            "Globally allowed types must be a subset of valid allowed types. Actual: {}, valid: {}".format(
                globally_allowed_types, sorted(_valid_allowed_types)))

    return globally_allowed_types


def _type_re_from_globally_allowed_types(globally_allowed_types: typing.Set[str]) -> typing.re.Pattern:
    '''Return a regular expression to match type registration strings to a set of globally allowed types.'''
    # to keep a registration, the type should match patterns like:
    # 1. T0
    # 2. T0_T1_T2
    # where Ti is a member of globally_allowed_types and multiple Ti's are delimited by "_"
    # this covers both the common case (1) and special cases like OneHot registration (2)
    allowed_type_subpattern = \
        "(?:" + "|".join(re.escape(allowed_type) for allowed_type in sorted(globally_allowed_types)) + ")"
    return re.compile("^{0}(?:_{0})*$".format(allowed_type_subpattern))


class _ExcludingRegistrationProcessor(op_registration_utils.RegistrationProcessor):
    '''Registration processor that excludes registrations and writes the result to an output file.'''
    def __init__(self, required_ops: dict, op_type_usage_manager: typing.Optional[OperatorTypeUsageManager],
                 globally_allowed_types: typing.Optional[typing.Set[str]], output_file: str):
        self._required_ops = required_ops

        if op_type_usage_manager is not None and globally_allowed_types is not None:
            raise ValueError("At most one of op_type_usage_manager and globally_allowed_types may be provided.")

        self._op_type_usage_manager = op_type_usage_manager

        self._enable_all_ops = globally_allowed_types is not None and not required_ops
        if self._enable_all_ops:
            log.info("No required ops were specified but globally allowed types were specified. "
                     "Globally allowed types will be used to exclude op implementations.")

        self._globally_allowed_types_re = \
            _type_re_from_globally_allowed_types(globally_allowed_types) \
            if globally_allowed_types is not None else None

        self._output_file = output_file

    def _is_op_required(self, domain: str, operator: str,
                        start_version: int, end_version: typing.Optional[int]) -> typing.Tuple[bool, str]:
        '''See if an op should be excluded because it is not required.'''
        if self._enable_all_ops:
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

            if not exclude and type is not None:
                if self._op_type_usage_manager is not None:
                    if not self._op_type_usage_manager.is_typed_registration_needed(domain, operator, type):
                        exclude = True
                        reason = "Specific typed registration is not required."

                elif self._globally_allowed_types_re is not None:
                    if not self._globally_allowed_types_re.match(type):
                        exclude = True
                        reason = "Specific typed registration does not contain globally allowed types."

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
        required_ops: dict,
        op_type_usage_manager: typing.Optional[OperatorTypeUsageManager],
        globally_allowed_types: typing.Optional[typing.Set[str]]):
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
            processor = _ExcludingRegistrationProcessor(
                required_ops, op_type_usage_manager, globally_allowed_types, file_to_write)

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
    required_ops, op_type_usage_manager, globally_allowed_types = parse_config(config_path, enable_type_reduction)

    if globally_allowed_types is not None:
        globally_allowed_types = _validated_globally_allowed_types(globally_allowed_types)

    _process_provider_registrations(ort_root, use_cuda, required_ops, op_type_usage_manager, globally_allowed_types)

    if op_type_usage_manager is not None:
        type_control_cpp_code = op_type_usage_manager.get_cpp_entries()
    elif globally_allowed_types is not None:
        type_control_cpp_code = ["ORT_SPECIFY_OP_KERNEL_GLOBAL_ALLOWED_TYPES({});".format(
            ", ".join(sorted(globally_allowed_types)))]
    else:
        type_control_cpp_code = []

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
