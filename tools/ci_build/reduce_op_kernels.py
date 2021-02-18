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


def _validated_globally_allowed_types(
        globally_allowed_types: typing.Optional[typing.Collection[str]]) -> typing.Set[str]:
    '''Return a valid set of globally allowed types.'''
    # if unspecified, allow all valid types
    if globally_allowed_types is None:
        globally_allowed_types = _valid_allowed_types.copy()

    # ensure globally_allowed_types is a set
    if not isinstance(globally_allowed_types, set):
        globally_allowed_types = set(globally_allowed_types)

    if not globally_allowed_types <= _valid_allowed_types:
        raise ValueError(
            "Globally allowed types must be a subset of valid allowed types. Actual: {}, valid: {}".format(
                globally_allowed_types, sorted(_valid_allowed_types)))

    return globally_allowed_types


class _ExcludingRegistrationProcessor(op_registration_utils.RegistrationProcessor):
    '''Registration processor that excludes registrations and writes the result to an output file.'''
    def __init__(self, output_file: str):
        '''
        Constructor.
        :param output_file: Output file path
        '''
        self._output_file = output_file

    def should_exclude_registration(self, constant_for_domain: str, operator: str,
                                    start_version: int, end_version: typing.Optional[int] = None,
                                    type: typing.Optional[str] = None) -> typing.Tuple[bool, str]:
        '''
        Indicate whether the registration should be excluded. Derived classes should implement this.
        :param domain: Domain for the operator
        :param operator: Operator type
        :param start_version: Start version
        :param end_version: End version or None if unversioned registration
        :param type: Type used in registration, if this is a typed registration
        :return: Tuple of (whether to exclude: bool, reason for exclusion: str)
        '''
        raise NotImplementedError()

    def process_registration(self, lines: typing.List[str], constant_for_domain: str, operator: str,
                             start_version: int, end_version: typing.Optional[int] = None,
                             type: typing.Optional[str] = None):
        registration_identifier = '{}:{}({}){}'.format(constant_for_domain, operator, start_version,
                                                       '<{}>'.format(type) if type else '')

        exclude, reason = self.should_exclude_registration(
            constant_for_domain, operator, start_version, end_version, type)

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
        create_processor_fn: typing.Callable[[str], op_registration_utils.RegistrationProcessor]):
    '''
    Rewrite provider registration files.
    :param ort_root: Root of the ONNX Runtime repository
    :use_cuda: Whether to process registrations for the CUDA provider
    :create_processor_fn: Function that accepts an output file path and returns a RegistrationProcessor to use
    '''
    kernel_registration_files = op_registration_utils.get_kernel_registration_files(ort_root, use_cuda)

    for kernel_registration_file in kernel_registration_files:
        if not os.path.isfile(kernel_registration_file):
            raise ValueError('Kernel registration file {} does not exist'.format(kernel_registration_file))

        log.info("Processing {}".format(kernel_registration_file))

        backup_path = kernel_registration_file + '~'
        shutil.move(kernel_registration_file, backup_path)

        # read from backup and overwrite original with commented out lines for any kernels that are not required
        with open(kernel_registration_file, 'w') as file_to_write:
            processor = create_processor_fn(file_to_write)

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


class _ExcludeFromConfigRegistrationProcessor(_ExcludingRegistrationProcessor):
    '''Registration processor that excludes registrations based on configuration file info.'''
    def __init__(self, required_ops, op_type_usage_manager, output_file):
        super().__init__(output_file)
        self._required_ops = required_ops
        self._op_type_usage_manager = op_type_usage_manager

    def _should_exclude_op(self, domain, operator, start_version, end_version):
        if domain not in self._required_ops:
            return True

        for opset in self._required_ops[domain]:
            if opset >= start_version and (end_version is None or opset <= end_version):
                if operator in self._required_ops[domain][opset]:
                    return False  # found a match, do not exclude

        return True

    def should_exclude_registration(self, constant_for_domain: str, operator: str,
                                    start_version: int, end_version: typing.Optional[int] = None,
                                    type: typing.Optional[str] = None) -> typing.Tuple[bool, str]:
        # convert from the ORT constant name to the domain string used in the config
        domain = op_registration_utils.map_ort_constant_to_domain(constant_for_domain)
        exclude = False
        reason = ""

        if domain:
            # see if entire op is excluded
            if self._should_exclude_op(domain, operator, start_version, end_version):
                exclude = True
                reason = "Entire op is excluded by configuration."

            # see if a specific typed registration can be excluded
            if not exclude and type and self._op_type_usage_manager \
                    and not self._op_type_usage_manager.is_typed_registration_needed(domain, operator, type):
                exclude = True
                reason = "Specific typed registration is excluded by configuration."

        return exclude, reason


def exclude_unused_ops_and_types(config_path: str, enable_type_reduction: bool = False, use_cuda: bool = True):
    '''
    Exclude op kernel implementations based on a configuration file.
    :param config_path: Configuration file path
    :param enable_type_reduction: Whether per operator type reduction is enabled
    :param use_cuda: Whether to reduce op kernels for the CUDA provider
    '''
    required_ops, op_type_usage_manager = parse_config(config_path, enable_type_reduction)

    def create_processor(file_to_write):
        return _ExcludeFromConfigRegistrationProcessor(required_ops,
                                                       op_type_usage_manager,
                                                       file_to_write)

    _process_provider_registrations(ort_root, use_cuda, create_processor)

    _insert_type_control_cpp_code(ort_root, op_type_usage_manager.get_cpp_entries())


class _ExcludeFromGloballyAllowedTypesRegistrationProcessor(_ExcludingRegistrationProcessor):
    '''Registration processor that excludes registrations based on a set of globally allowed types.'''
    def __init__(self, globally_allowed_types: typing.Set[str], output_file: str):
        # to keep a registration, the type should match patterns like:
        # 1. T0
        # 2. T0_T1_T2
        # where Ti is a member of globally_allowed_types and multiple Ti's are delimited by "_"
        # this covers both the common case (1) and special cases like OneHot registration (2)
        allowed_type_subpattern = \
            "(?:" + "|".join(re.escape(allowed_type) for allowed_type in sorted(globally_allowed_types)) + ")"
        self._type_pattern_re = re.compile("^{0}(?:_{0})*$".format(allowed_type_subpattern))

        super().__init__(output_file)

    def should_exclude_registration(self, constant_for_domain: str, operator: str,
                                    start_version: int, end_version: typing.Optional[int] = None,
                                    type: typing.Optional[str] = None) -> typing.Tuple[bool, str]:
        exclude = False
        reason = ""

        if type is not None and not self._type_pattern_re.match(type):
            exclude = True
            reason = "Specific typed registration is excluded by globally allowed types."

        return exclude, reason


def constrain_ops_to_globally_allowed_types(
        globally_allowed_types: typing.Optional[typing.Collection[str]],
        use_cuda: bool = True):
    '''
    Constrain op kernel implementations to the specified globally allowed types.
    :param globally_allowed_types: The allowed types for which op kernel implementations are kept
    :param use_cuda: Whether to reduce op kernels for the CUDA provider
    '''
    globally_allowed_types = _validated_globally_allowed_types(globally_allowed_types)

    def create_processor(file_to_write):
        return _ExcludeFromGloballyAllowedTypesRegistrationProcessor(globally_allowed_types,
                                                                     file_to_write)

    _process_provider_registrations(ort_root, use_cuda, create_processor)

    _insert_type_control_cpp_code(
        ort_root,
        ["ORT_SPECIFY_OP_KERNEL_GLOBAL_ALLOWED_TYPES({});".format(
            ", ".join(sorted(globally_allowed_types)))])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reduces operator kernel implementations in ONNX Runtime. "
                    "Entire op implementations or op implementations for specific types may be pruned. "
                    "Two modes are supported: reduction may be specified via a configuration file or via a set of "
                    "globally allowed types.")

    parser.add_argument("--config-path", type=str,
                        help="Path to configuration file. "
                             "Create with <ORT root>/tools/python/create_reduced_build_config.py and edit if needed. "
                             "See /docs/ONNX_Runtime_Format_Model_Usage.md for more information.")

    parser.add_argument("--globally-allowed-types", nargs="*", choices=sorted(_valid_allowed_types),
                        help="Specifies the globally allowed C++ scalar types.")

    args = parser.parse_args()

    if (args.config_path is None) == (args.globally_allowed_types is None):
        raise ValueError("Exactly one of --config-file or --globally-allowed-types must be specified.")

    if args.config_path is not None:
        config_path = os.path.abspath(args.config_path)
        exclude_unused_ops_and_types(config_path, enable_type_reduction=True, use_cuda=True)
    else:
        constrain_ops_to_globally_allowed_types(args.globally_allowed_types, use_cuda=True)
