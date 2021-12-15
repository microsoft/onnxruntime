# !/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import op_registration_utils
import shutil
import sys
import typing

from logger import get_logger
from pathlib import Path

# directory containing the reduced op files, relative to the build directory
OP_REDUCTION_DIR = "op_reduction.generated"

# add the path to /tools/python so we can import the config parsing and type reduction processing
SCRIPT_DIR = Path(__file__).parent.resolve()
ORT_ROOT = SCRIPT_DIR.parents[1]
sys.path.append(str(ORT_ROOT / 'tools' / 'python'))

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


def _get_op_reduction_file_path(ort_root: Path, build_dir: Path, original_path: typing.Optional[Path] = None):
    '''
    Return the op reduction file path corresponding to `original_path` or the op reduction file root if unspecified.
    Op reduction files are in a subdirectory of `build_dir` but otherwise share the same components of `original_path`
    relative to `ort_root`.
    '''
    op_reduction_root = Path(build_dir, OP_REDUCTION_DIR)
    return (op_reduction_root / original_path.relative_to(ort_root)) if original_path is not None \
        else op_reduction_root


def _generate_provider_registrations(
        ort_root: Path, build_dir: Path, use_cuda: bool,
        required_ops: typing.Optional[dict],
        op_type_impl_filter: typing.Optional[OpTypeImplFilterInterface]):
    '''Generate provider registration files.'''
    kernel_registration_files = [Path(f) for f in
                                 op_registration_utils.get_kernel_registration_files(str(ort_root), use_cuda)]

    for kernel_registration_file in kernel_registration_files:
        if not kernel_registration_file.is_file():
            raise ValueError(f'Kernel registration file does not exist: {kernel_registration_file}')

        log.info("Processing {}".format(kernel_registration_file))

        reduced_path = _get_op_reduction_file_path(ort_root, build_dir, kernel_registration_file)

        reduced_path.parent.mkdir(parents=True, exist_ok=True)

        # read from original and create the reduced kernel def file with commented out lines for any kernels that are
        # not required
        with open(reduced_path, 'w') as file_to_write:
            processor = _ExcludingRegistrationProcessor(required_ops, op_type_impl_filter, file_to_write)

            op_registration_utils.process_kernel_registration_file(kernel_registration_file, processor)

            if not processor.ok():
                # error should have already been logged so just exit
                sys.exit(-1)


def _generate_type_control_overrides(ort_root: Path, build_dir: Path, cpp_lines: typing.Sequence[str]):
    '''
    Generate type control overrides. Insert applicable C++ code to specify operator type requirements.
    :param ort_root: Root of the ONNX Runtime repository
    :param build_dir: Path to the build directory
    :param cpp_lines: The C++ code to insert
    '''
    src = Path(ort_root, 'onnxruntime', 'core', 'providers', 'op_kernel_type_control_overrides.inc')

    if not src.is_file():
        raise ValueError(f"Op kernel type control overrides file does not exist: {src}")

    # create a copy of op_kernel_type_control_overrides.inc
    target = _get_op_reduction_file_path(ort_root, build_dir, src)

    target.parent.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(src, target)

    if cpp_lines:
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

        if not inserted:
            raise RuntimeError('Insertion point was not found in {}'.format(target))


def reduce_ops(config_path: str, build_dir: str, enable_type_reduction: bool = False, use_cuda: bool = True):
    '''
    Reduce op kernel implementations.
    :param config_path: Path to configuration file that specifies the ops to include
    :param build_dir: Path to the build directory. The op reduction files will be generated under the build directory.
    :param enable_type_reduction: Whether per operator type reduction is enabled
    :param use_cuda: Whether to reduce op kernels for the CUDA provider
    '''
    build_dir = Path(build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)

    required_ops, op_type_impl_filter = parse_config(config_path, enable_type_reduction)

    # delete any existing generated files first
    op_reduction_root = _get_op_reduction_file_path(ORT_ROOT, build_dir)
    if op_reduction_root.is_dir():
        log.info(f"Deleting existing op reduction file root directory: {op_reduction_root}")
        shutil.rmtree(op_reduction_root)

    _generate_provider_registrations(ORT_ROOT, build_dir, use_cuda, required_ops, op_type_impl_filter)

    type_control_cpp_code = op_type_impl_filter.get_cpp_entries() if op_type_impl_filter is not None else []
    _generate_type_control_overrides(ORT_ROOT, build_dir, type_control_cpp_code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reduces operator kernel implementations in ONNX Runtime. "
                    "Entire op implementations or op implementations for specific types may be pruned.")

    parser.add_argument("config_path", type=str,
                        help="Path to configuration file. "
                             "Create with <ORT root>/tools/python/create_reduced_build_config.py and edit if needed. "
                             "See /docs/ONNX_Runtime_Format_Model_Usage.md for more information.")

    parser.add_argument("--cmake_build_dir", type=str, required=True,
                        help="Path to the build directory. "
                             "The op reduction files will be generated under the build directory.")

    parser.add_argument("--enable_type_reduction", action="store_true",
                        help="Whether per operator type reduction is enabled.")

    parser.add_argument("--use_cuda", action="store_true",
                        help="Whether to reduce op kernels for the CUDA provider.")

    args = parser.parse_args()

    reduce_ops(config_path=args.config_path,
               build_dir=args.cmake_build_dir,
               enable_type_reduction=args.enable_type_reduction,
               use_cuda=args.use_cuda)
