# !/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import io
import re
import shutil
import sys
import typing
from pathlib import Path

import op_registration_utils
from logger import get_logger

# directory containing the reduced op files, relative to the build directory
OP_REDUCTION_DIR = "op_reduction.generated"

# add the path to tools/python so we can import the config parsing and type reduction processing
SCRIPT_DIR = Path(__file__).parent.resolve()
ORT_ROOT = SCRIPT_DIR.parents[1]
sys.path.append(str(ORT_ROOT / "tools" / "python"))

from util import parse_config  # noqa: E402
from util.ort_format_model.operator_type_usage_processors import OpTypeImplFilterInterface  # noqa: E402

log = get_logger("reduce_op_kernels")


def _adapt_filters_for_extended_minimal_build(
    base_required_ops: typing.Optional[dict], base_op_type_impl_filter: typing.Optional[OpTypeImplFilterInterface]
):
    """
    Adapts the values returned by parse_config() for an extended minimal build or higher.
    In particular:
    - Includes ONNX ops needed by layout transformation
    - Includes MS ops needed by NHWC optimizer
    """
    # graph transformations in an extended minimal build require certain ops to be available
    extended_minimal_build_required_op_ids = set()  # set of (domain, optype, opset)
    with open(
        ORT_ROOT / "onnxruntime/core/optimizer/layout_transformation/layout_transformation_potentially_added_ops.h",
    ) as f:
        region_boundary_pattern = re.compile(r"@@region_(begin|end)\(extended_minimal_build_required_kernels\)@@")
        op_id_pattern = re.compile(
            r'OpIdentifierWithStringViews{(?P<domain>\w+),\s+"(?P<optype>\w+)",\s+(?P<opset>\d+)}'
        )
        in_region = False
        for line in f:
            region_boundary_match = region_boundary_pattern.search(line)
            if region_boundary_match:
                in_region = region_boundary_match.group(1) == "begin"
                continue

            if not in_region:
                continue

            op_id_match = op_id_pattern.search(line)
            if op_id_match:
                domain = op_registration_utils.map_ort_constant_to_domain(
                    op_id_match.group("domain"), allow_unknown_constant=False
                )
                optype = op_id_match.group("optype")
                opset = int(op_id_match.group("opset"))
                extended_minimal_build_required_op_ids.add((domain, optype, opset))

    adapted_required_ops = None
    if base_required_ops is not None:
        adapted_required_ops = base_required_ops.copy()
        for domain, optype, opset in extended_minimal_build_required_op_ids:
            adapted_required_ops.setdefault(domain, dict()).setdefault(opset, set()).add(optype)

    adapted_op_type_impl_filter = None
    if base_op_type_impl_filter is not None:

        class _AdaptedFilter(OpTypeImplFilterInterface):
            def __init__(
                self,
                filter_to_adapt: OpTypeImplFilterInterface,
                required_domain_and_optypes: typing.Set[typing.Tuple[str, str]],
            ):
                self.filter_to_adapt = filter_to_adapt
                self.required_domain_and_optypes = required_domain_and_optypes

            def is_typed_registration_needed(self, domain: str, optype: str, type_registration_str: str):
                # Always require registration for ops in self.required_domain_and_optypes.
                if (domain, optype) in self.required_domain_and_optypes:
                    return True
                return self.filter_to_adapt.is_typed_registration_needed(domain, optype, type_registration_str)

            def get_cpp_entries(self):
                # The required types for ops in self.required_optypes must be specified in the C++ implementation.
                # Doing that also accounts for globally allowed types.
                # We don't need to do anything special with the allowed type overrides here.
                return self.filter_to_adapt.get_cpp_entries()

        adapted_op_type_impl_filter = _AdaptedFilter(
            base_op_type_impl_filter,
            {(domain, optype) for (domain, optype, opset) in extended_minimal_build_required_op_ids},
        )

    return (adapted_required_ops, adapted_op_type_impl_filter)


class _ExcludingRegistrationProcessor(op_registration_utils.RegistrationProcessor):
    """Registration processor that excludes registrations and writes the result to an output file."""

    def __init__(
        self,
        required_ops: typing.Optional[dict],
        op_type_impl_filter: typing.Optional[OpTypeImplFilterInterface],
        output_file: io.TextIOWrapper,
    ):
        self._required_ops = required_ops
        self._op_type_impl_filter = op_type_impl_filter
        self._output_file = output_file

    def _is_op_required(
        self, domain: str, operator: str, start_version: int, end_version: typing.Optional[int]
    ) -> bool:
        """See if an op is required."""
        if self._required_ops is None:
            return True

        if domain not in self._required_ops:
            return False

        for opset in self._required_ops[domain]:
            if opset >= start_version and (end_version is None or opset <= end_version):
                if operator in self._required_ops[domain][opset]:
                    return True

        return False

    def process_registration(
        self,
        lines: typing.List[str],
        constant_for_domain: str,
        operator: str,
        start_version: int,
        end_version: typing.Optional[int] = None,
        type: typing.Optional[str] = None,
    ):
        registration_identifier = "{}:{}({}){}".format(
            constant_for_domain, operator, start_version, f"<{type}>" if type else ""
        )

        # convert from the ORT constant name to the domain string used in the config
        domain = op_registration_utils.map_ort_constant_to_domain(constant_for_domain, allow_unknown_constant=False)

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
            log.warning(f"Keeping {registration_identifier} registration from unknown domain: {constant_for_domain}")

        if exclude:
            log.info(f"Disabling {registration_identifier} registration: {reason}")
            for line in lines:
                self._output_file.write("// " + line)

            # edge case of last entry in table where we still need the terminating }; to not be commented out
            if lines[-1].rstrip().endswith("};"):
                self._output_file.write("};\n")
        else:
            for line in lines:
                self._output_file.write(line)

    def process_other_line(self, line):
        self._output_file.write(line)

    def ok(self):
        return True


def _get_op_reduction_root(build_dir: Path):
    """
    Return the op reduction root directory which is a subdirectory of `build_dir`.
    """
    return Path(build_dir, OP_REDUCTION_DIR)


def _get_op_reduction_file_path(ort_root: Path, build_dir: Path, original_path: Path):
    """
    Return the op reduction file path corresponding to `original_path`.
    Op reduction files are in the op reduction root but otherwise share the same components of `original_path`
    relative to `ort_root`.
    """
    return _get_op_reduction_root(build_dir) / original_path.relative_to(ort_root)


def _generate_provider_registrations(
    ort_root: Path,
    build_dir: Path,
    use_cuda: bool,
    required_ops: typing.Optional[dict],
    op_type_impl_filter: typing.Optional[OpTypeImplFilterInterface],
):
    """Generate provider registration files."""
    kernel_registration_files = [
        Path(f) for f in op_registration_utils.get_kernel_registration_files(str(ort_root), use_cuda)
    ]

    for kernel_registration_file in kernel_registration_files:
        if not kernel_registration_file.is_file():
            raise ValueError(f"Kernel registration file does not exist: {kernel_registration_file}")

        log.info(f"Processing {kernel_registration_file}")

        reduced_path = _get_op_reduction_file_path(ort_root, build_dir, kernel_registration_file)

        reduced_path.parent.mkdir(parents=True, exist_ok=True)

        # read from original and create the reduced kernel def file with commented out lines for any kernels that are
        # not required
        with open(reduced_path, "w") as file_to_write:
            processor = _ExcludingRegistrationProcessor(required_ops, op_type_impl_filter, file_to_write)

            op_registration_utils.process_kernel_registration_file(kernel_registration_file, processor)

            if not processor.ok():
                # error should have already been logged so just exit
                sys.exit(-1)


def _generate_type_control_overrides(ort_root: Path, build_dir: Path, cpp_lines: typing.Sequence[str]):
    """
    Generate type control overrides. Insert applicable C++ code to specify operator type requirements.
    :param ort_root: Root of the ONNX Runtime repository
    :param build_dir: Path to the build directory
    :param cpp_lines: The C++ code to insert
    """
    src = Path(ort_root, "onnxruntime", "core", "providers", "op_kernel_type_control_overrides.inc")

    if not src.is_file():
        raise ValueError(f"Op kernel type control overrides file does not exist: {src}")

    # create a copy of op_kernel_type_control_overrides.inc
    target = _get_op_reduction_file_path(ort_root, build_dir, src)

    target.parent.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(src, target)

    if cpp_lines:
        # find the insertion block and replace any existing content in it
        inserted = False
        with open(src) as input, open(target, "w") as output:
            inside_insertion_block = False
            for line in input.readlines():
                if "@@insertion_point_begin(allowed_types)@@" in line:
                    inside_insertion_block = True
                    output.write(line)
                    [output.write(f"{code_line}\n") for code_line in cpp_lines]
                    inserted = True
                    continue
                elif inside_insertion_block:
                    if "@@insertion_point_end(allowed_types)@@" in line:
                        inside_insertion_block = False
                    else:
                        # we ignore any old lines within the insertion block
                        continue

                output.write(line)

        if not inserted:
            raise RuntimeError(f"Insertion point was not found in {target}")


def reduce_ops(
    config_path: str,
    build_dir: str,
    enable_type_reduction: bool,
    use_cuda: bool,
    is_extended_minimal_build_or_higher: bool,
):
    """
    Reduce op kernel implementations.
    :param config_path: Path to configuration file that specifies the ops to include
    :param build_dir: Path to the build directory. The op reduction files will be generated under the build directory.
    :param enable_type_reduction: Whether per operator type reduction is enabled
    :param use_cuda: Whether to reduce op kernels for the CUDA provider
    :param is_extended_minimal_build_or_higher: Whether this build has at least the features of an extended minimal
                                                build enabled.
    """
    build_dir_path = Path(build_dir).resolve()
    build_dir_path.mkdir(parents=True, exist_ok=True)

    required_ops, op_type_impl_filter = parse_config(config_path, enable_type_reduction)
    if is_extended_minimal_build_or_higher:
        required_ops, op_type_impl_filter = _adapt_filters_for_extended_minimal_build(required_ops, op_type_impl_filter)

    # delete any existing generated files first
    op_reduction_root = _get_op_reduction_root(build_dir_path)
    if op_reduction_root.is_dir():
        log.info(f"Deleting existing op reduction file root directory: {op_reduction_root}")
        shutil.rmtree(op_reduction_root)

    _generate_provider_registrations(ORT_ROOT, build_dir_path, use_cuda, required_ops, op_type_impl_filter)

    type_control_cpp_code = op_type_impl_filter.get_cpp_entries() if op_type_impl_filter is not None else []
    _generate_type_control_overrides(ORT_ROOT, build_dir_path, type_control_cpp_code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reduces operator kernel implementations in ONNX Runtime. "
        "Entire op implementations or op implementations for specific types may be pruned."
    )

    parser.add_argument(
        "config_path",
        type=str,
        help="Path to configuration file. "
        "Create with <ORT root>/tools/python/create_reduced_build_config.py and edit if needed. "
        "See https://onnxruntime.ai/docs/reference/operators/reduced-operator-config-file.html for more "
        "information.",
    )

    parser.add_argument(
        "--cmake_build_dir",
        type=str,
        required=True,
        help="Path to the build directory. The op reduction files will be generated under the build directory.",
    )

    parser.add_argument(
        "--is_extended_minimal_build_or_higher",
        action="store_true",
        help="Whether this build has at least the features of an extended minimal build enabled.",
    )

    parser.add_argument(
        "--enable_type_reduction", action="store_true", help="Whether per operator type reduction is enabled."
    )

    parser.add_argument("--use_cuda", action="store_true", help="Whether to reduce op kernels for the CUDA provider.")

    args = parser.parse_args()

    reduce_ops(
        config_path=args.config_path,
        build_dir=args.cmake_build_dir,
        enable_type_reduction=args.enable_type_reduction,
        use_cuda=args.use_cuda,
        is_extended_minimal_build_or_higher=args.is_extended_minimal_build_or_higher,
    )
