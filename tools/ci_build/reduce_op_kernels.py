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


'''
Callable that determines whether a registration should be excluded.
:param domain: Domain for the operator
:param operator: Operator type
:param start_version: Start version
:param end_version: End version or None if unversioned registration
:param type: Type used in registration, if this is a typed registration
:return: Tuple of (whether to exclude: bool, reason for exclusion: str)
'''
_ShouldExcludeRegistrationFn = \
    typing.Callable[
        [str,  # domain
         str,  # operator
         int,  # start_version
         typing.Optional[int],  # end_version
         typing.Optional[str],  # type
         ],
        typing.Tuple[bool, str]]


class _ExcludingRegistrationProcessor(op_registration_utils.RegistrationProcessor):
    '''Registration processor that excludes registrations and writes the result to an output file.'''
    def __init__(self, should_exclude_registration_fn: _ShouldExcludeRegistrationFn, output_file: str):
        '''
        Constructor.
        :should_exclude_registration_fn: Callable that determines whether a registration should be excluded
        :param output_file: Output file path
        '''
        self._should_exclude_registration_fn = should_exclude_registration_fn
        self._output_file = output_file

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
            exclude, reason = self._should_exclude_registration_fn(
                domain, operator, start_version, end_version, type)
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
        ort_root: str, use_cuda: bool, should_exclude_registration_fn: _ShouldExcludeRegistrationFn):
    '''
    Rewrite provider registration files.
    :param ort_root: Root of the ONNX Runtime repository
    :use_cuda: Whether to process registrations for the CUDA provider
    :should_exclude_registration_fn: Callable that determines whether a registration should be excluded
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
            processor = _ExcludingRegistrationProcessor(should_exclude_registration_fn, file_to_write)

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


def _should_exclude_op(required_ops: dict, domain: str, operator: str,
                       start_version: int, end_version: typing.Optional[int]) -> typing.Tuple[bool, str]:
    '''See if an op should be excluded because it is not required.'''
    def is_op_required():
        if domain not in required_ops:
            return False

        for opset in required_ops[domain]:
            if opset >= start_version and (end_version is None or opset <= end_version):
                if operator in required_ops[domain][opset]:
                    return True

        return False

    return (False, "") if is_op_required() else (True, "Entire op is not required.")


def reduce_ops(config_path: str, enable_type_reduction: bool = False, use_cuda: bool = True):
    '''
    Reduce op kernel implementations.
    :param config_path: Path to configuration file that specifies the ops to include
    :param enable_type_reduction: Whether per operator type reduction is enabled
    :param use_cuda: Whether to reduce op kernels for the CUDA provider
    '''
    required_ops, op_type_usage_manager = parse_config(config_path, enable_type_reduction)

    def should_exclude_registration(domain: str, operator: str,
                                    start_version: int, end_version: typing.Optional[int],
                                    type: typing.Optional[str]) -> typing.Tuple[bool, str]:
        exclude, reason = _should_exclude_op(required_ops, domain, operator, start_version, end_version)

        # see if a specific typed registration can be excluded
        if not exclude and op_type_usage_manager is not None and type is not None and \
                not op_type_usage_manager.is_typed_registration_needed(domain, operator, type):
            exclude = True
            reason = "Specific typed registration is not required."

        return exclude, reason

    _process_provider_registrations(ort_root, use_cuda, should_exclude_registration)

    if op_type_usage_manager is not None:
        _insert_type_control_cpp_code(ort_root, op_type_usage_manager.get_cpp_entries())


def reduce_ops_with_globally_allowed_types(
        config_path: str,
        globally_allowed_types: typing.Optional[typing.Collection[str]],
        use_cuda: bool = True):
    '''
    Reduce op kernel implementations with type reduction to globally allowed types.
    :param config_path: Path to configuration file that specifies the ops to include
    :param globally_allowed_types: The allowed types for which op kernel implementations are included
    :param use_cuda: Whether to reduce op kernels for the CUDA provider
    '''
    globally_allowed_types = _validated_globally_allowed_types(globally_allowed_types)

    required_ops, _ = parse_config(config_path, enable_type_reduction=False)

    # to keep a registration, the type should match patterns like:
    # 1. T0
    # 2. T0_T1_T2
    # where Ti is a member of globally_allowed_types and multiple Ti's are delimited by "_"
    # this covers both the common case (1) and special cases like OneHot registration (2)
    allowed_type_subpattern = \
        "(?:" + "|".join(re.escape(allowed_type) for allowed_type in sorted(globally_allowed_types)) + ")"
    type_pattern_re = re.compile("^{0}(?:_{0})*$".format(allowed_type_subpattern))

    def should_exclude_registration(domain: str, operator: str,
                                    start_version: int, end_version: typing.Optional[int] = None,
                                    type: typing.Optional[str] = None) -> typing.Tuple[bool, str]:
        exclude, reason = _should_exclude_op(required_ops, domain, operator, start_version, end_version)

        if not exclude and type is not None and not type_pattern_re.match(type):
            exclude = True
            reason = "Specific typed registration is excluded by globally allowed types."

        return exclude, reason

    _process_provider_registrations(ort_root, use_cuda, should_exclude_registration)

    _insert_type_control_cpp_code(
        ort_root,
        ["ORT_SPECIFY_OP_KERNEL_GLOBAL_ALLOWED_TYPES({});".format(
            ", ".join(sorted(globally_allowed_types)))])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reduces operator kernel implementations in ONNX Runtime. "
                    "Entire op implementations or op implementations for specific types may be pruned.")

    parser.add_argument("config_path", type=str,
                        help="Path to configuration file. "
                             "Create with <ORT root>/tools/python/create_reduced_build_config.py and edit if needed. "
                             "See /docs/ONNX_Runtime_Format_Model_Usage.md for more information.")

    parser.add_argument("--globally-allowed-types", nargs="*", metavar="type", choices=sorted(_valid_allowed_types),
                        help="Specifies the globally allowed C++ scalar types.")

    args = parser.parse_args()

    config_path = os.path.abspath(args.config_path)

    if args.globally_allowed_types is not None:
        reduce_ops_with_globally_allowed_types(config_path, args.globally_allowed_types, use_cuda=True)
    else:
        reduce_ops(config_path, enable_type_reduction=True, use_cuda=True)
