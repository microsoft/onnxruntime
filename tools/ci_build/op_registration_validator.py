# !/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
'''
Validate ORT kernel registrations.
'''

import argparse
import op_registration_utils
import os
import sys
import typing

from logger import get_logger

log = get_logger("op_registration_validator")

# deprecated ops where the last registration should have an end version.
# value for each entry is the opset when it was deprecated. end version of last registration should equal value - 1.
deprecated_ops = {'kOnnxDomain:Scatter': 11,
                  'kOnnxDomain:Upsample': 10,
                  # MeanVarianceNormalization and ThresholdedRelu were in contrib ops and incorrectly registered using
                  # kOnnxDomain. They became official ONNX operators later and are registered there now. That leaves
                  # entries in the contrib ops registrations with end versions for when the contrib op was 'deprecated'
                  # and became an official op.
                  'kOnnxDomain:MeanVarianceNormalization': 9,
                  'kOnnxDomain:ThresholdedRelu': 10}


class RegistrationValidator(op_registration_utils.RegistrationProcessor):
    def __init__(self):
        self.last_op_registrations = {}
        self.failed = False

    def process_registration(self, lines: typing.List[str], domain: str, operator: str,
                             start_version: int, end_version: typing.Optional[int] = None,
                             type: typing.Optional[str] = None):
        key = domain + ':' + operator
        prev_start, prev_end = self.last_op_registrations[key] if key in self.last_op_registrations else (None, None)

        if prev_start:
            # a typed registration where the to/from matches for each entry so nothing to update
            if prev_start == start_version and prev_end == end_version:
                return

            # previous registration was unversioned but should have been if we are seeing another registration
            if not prev_end:
                log.error("Invalid registration for {}. Registration for opset {} has no end version but was "
                          "superceeded by version {}."
                          .format(key, prev_start, start_version))
                self.failed = True
                return

            # previous registration end opset is not adjacent to the start of the next registration
            if prev_end != start_version - 1:
                log.error("Invalid registration for {}. Registration for opset {} should have end version of {}"
                          .format(key, prev_start, start_version - 1))
                self.failed = True
                return

        self.last_op_registrations[key] = (start_version, end_version)

    def ok(self):
        return not self.failed

    def validate_last_registrations(self):
        # make sure we have an unversioned last entry for each operator unless it's deprecated
        for entry in self.last_op_registrations.items():
            key, value = entry
            opset_from, opset_to = value

            deprecated = key in deprecated_ops and opset_to == deprecated_ops[key] - 1
            if opset_to and not deprecated:
                log.error('Missing unversioned registration for {}'.format(key))
                self.failed = True


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script to validate operator kernel registrations.")

    parser.add_argument(
        "--ort_root", type=str, help="Path to ONNXRuntime repository root. "
                                     "Inferred from the location of this script if not provided.")

    args = parser.parse_args()

    ort_root = os.path.abspath(args.ort_root) if args.ort_root else ''
    include_cuda = True  # validate CPU and CUDA EP registrations

    registration_files = op_registration_utils.get_kernel_registration_files(ort_root, include_cuda)

    for file in registration_files:
        log.info("Processing {}".format(file))

        processor = RegistrationValidator()
        op_registration_utils.process_kernel_registration_file(file, processor)
        processor.validate_last_registrations()

        if not processor.ok():
            sys.exit(-1)
