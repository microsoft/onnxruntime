# !/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Validate ORT kernel registrations.
"""

import argparse
import dataclasses
import itertools
import os
import sys
import typing

import op_registration_utils
from logger import get_logger

log = get_logger("op_registration_validator")

# deprecated ops where the last registration should have an end version.
# value for each entry is the opset when it was deprecated. end version of last registration should equal value - 1.
deprecated_ops = {
    "kOnnxDomain:Scatter": 11,
    "kOnnxDomain:Upsample": 10,
    # LayerNormalization, MeanVarianceNormalization and ThresholdedRelu were in contrib ops and incorrectly registered
    # using the kOnnxDomain. They became official ONNX operators later and are registered there now. That leaves
    # entries in the contrib ops registrations with end versions for when the contrib op was 'deprecated'
    # and became an official op.
    "kOnnxDomain:LayerNormalization": 17,
    "kOnnxDomain:MeanVarianceNormalization": 9,
    "kOnnxDomain:ThresholdedRelu": 10,
}


@dataclasses.dataclass
class RegistrationInfo:
    domain: str
    operator: str
    start_version: int
    end_version: typing.Optional[int]
    lines: typing.List[str]

    def domain_and_op_str(self):
        return f"{self.domain}:{self.operator}"


def _log_registration_error(r: RegistrationInfo, message: str):
    log.error("Invalid registration for %s. %s\n%s", r.domain_and_op_str(), message, "".join(r.lines))


class RegistrationValidator(op_registration_utils.RegistrationProcessor):
    def __init__(self):
        self.all_registrations: typing.List[RegistrationInfo] = []

    def process_registration(
        self,
        lines: typing.List[str],
        domain: str,
        operator: str,
        start_version: int,
        end_version: typing.Optional[int] = None,
        type: typing.Optional[str] = None,
    ):
        self.all_registrations.append(
            RegistrationInfo(
                domain=domain, operator=operator, start_version=start_version, end_version=end_version, lines=lines
            )
        )

    def ok(self):
        num_invalid_registrations = self._validate_all_registrations()
        if num_invalid_registrations > 0:
            log.error(f"Found {num_invalid_registrations} invalid registration(s).")
            return False

        return True

    def _validate_all_registrations(self) -> int:
        """
        Validates all registrations added by `process_registration()` and returns the number of invalid ones.
        """

        def registration_info_sort_key(r: RegistrationInfo):
            return (
                r.domain,
                r.operator,
                r.start_version,
                1 if r.end_version is None else 0,  # unspecified end_version > specified end_version
                r.end_version,
            )

        def domain_and_op_key(r: RegistrationInfo):
            return (r.domain, r.operator)

        sorted_registrations = sorted(self.all_registrations, key=registration_info_sort_key)

        num_invalid_registrations = 0
        for _, registration_group in itertools.groupby(sorted_registrations, key=domain_and_op_key):
            num_invalid_registrations += self._validate_registrations_for_domain_and_op(registration_group)

        return num_invalid_registrations

    def _validate_registrations_for_domain_and_op(self, registrations: typing.Iterator[RegistrationInfo]) -> int:
        """
        Validates registrations in sorted order for a single domain and op and returns the number of invalid ones.
        """
        num_invalid_registrations = 0
        r = next(registrations, None)
        while r is not None:
            next_r = next(registrations, None)
            if not self._validate_registration(r, next_r):
                num_invalid_registrations += 1
            r = next_r

        return num_invalid_registrations

    def _validate_registration(self, r: RegistrationInfo, next_r: typing.Optional[RegistrationInfo]) -> bool:
        """
        Validates a registration, `r`, with the next one in sorted order for a single domain and op, `next_r`, and
        returns whether it is valid.
        """
        if not (r.end_version is None or r.start_version <= r.end_version):
            _log_registration_error(
                r, f"Start version ({r.start_version}) is greater than end version ({r.end_version})."
            )
            return False

        if next_r is None:
            return self._validate_last_registration(r)

        # It is valid to match next registration start and end versions exactly.
        # This is expected if there are multiple registrations for an opset (e.g., typed registrations).
        if (r.start_version, r.end_version) == (next_r.start_version, next_r.end_version):
            return True

        # This registration has no end version but it should have one if the next registration has different versions.
        if r.end_version is None:
            _log_registration_error(
                r,
                f"Registration for opset {r.start_version} has no end version but was superseded by version "
                f"{next_r.start_version}.",
            )
            return False

        # This registration's end version is not adjacent to the start version of the next registration.
        if r.end_version != next_r.start_version - 1:
            _log_registration_error(
                r,
                f"Registration end version is not adjacent to the next registration's start version. "
                f"Current start and end versions: {(r.start_version, r.end_version)}. "
                f"Next start and end versions: {(next_r.start_version, next_r.end_version)}.",
            )
            return False

        return True

    def _validate_last_registration(self, last_r: RegistrationInfo) -> bool:
        """
        Validates the last registration in sorted order for a single domain and op and returns whether it is valid.
        """
        # make sure we have an unversioned last entry for each operator unless it's deprecated

        # TODO If the operator is deprecated, validation is more lax. I.e., it doesn't require a versioned registration.
        # This could be tightened up but we would need to handle the deprecated contrib ops registered in the ONNX
        # domain that have newer registrations in a non-contrib op file differently. They should only be considered
        # deprecated as contrib ops.
        domain_and_op_str = last_r.domain_and_op_str()
        deprecation_version = deprecated_ops.get(domain_and_op_str)

        allow_missing_unversioned_registration = (
            deprecation_version is not None and last_r.end_version == deprecation_version - 1
        )

        # special handling for ArgMin/ArgMax, which CUDA EP doesn't yet support for opset 12+
        # TODO remove once CUDA EP supports ArgMin/ArgMax for opset 12+
        ops_with_incomplete_support = ["kOnnxDomain:ArgMin", "kOnnxDomain:ArgMax"]
        if domain_and_op_str in ops_with_incomplete_support:
            log.warning(
                f"Allowing missing unversioned registration for op with incomplete support: {domain_and_op_str}."
            )
            allow_missing_unversioned_registration = True

        if last_r.end_version is not None and not allow_missing_unversioned_registration:
            log.error(f"Missing unversioned registration for {domain_and_op_str}.")
            return False

        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to validate operator kernel registrations.")

    parser.add_argument(
        "--ort_root",
        type=str,
        help="Path to ONNXRuntime repository root. Inferred from the location of this script if not provided.",
    )

    args = parser.parse_args()

    ort_root = os.path.abspath(args.ort_root) if args.ort_root else None
    include_cuda = True  # validate CPU and CUDA EP registrations

    registration_files = op_registration_utils.get_kernel_registration_files(ort_root, include_cuda)

    def validate_registration_file(file: str) -> bool:
        log.info(f"Processing {file}")

        processor = RegistrationValidator()
        op_registration_utils.process_kernel_registration_file(file, processor)

        return processor.ok()

    validation_successful = all(
        # Validate each file first by storing the validation results in a list.
        # Otherwise, all() will exit early when it encounters the first invalid file.
        list(map(validate_registration_file, registration_files))
    )

    log.info(f"Op kernel registration validation {'succeeded' if validation_successful else 'failed'}.")
    sys.exit(0 if validation_successful else 1)
