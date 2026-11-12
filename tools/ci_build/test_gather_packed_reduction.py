#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import io
import json
import re
import sys
import unittest
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import op_registration_utils
from reduce_op_kernels import ORT_ROOT, _ExcludingRegistrationProcessor
from util.ort_format_model import GloballyAllowedTypesOpTypeImplFilter, OperatorTypeUsageManager


class GatherPackedReductionTest(unittest.TestCase):
    output_types = ("float", "MLFloat16", "BFloat16")
    index_types = ("int32_t", "int64_t")

    def _enabled_cpu_registrations(self, type_filter):
        output = io.StringIO()
        processor = _ExcludingRegistrationProcessor({"com.microsoft": {1: {"GatherPacked"}}}, type_filter, output)
        op_registration_utils.process_kernel_registration_file(
            ORT_ROOT / "onnxruntime/contrib_ops/cpu/cpu_contrib_kernels.cc", processor
        )
        registrations = [
            line.strip()
            for line in output.getvalue().splitlines()
            if "BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(" in line and ", GatherPacked)>" in line
        ]
        self.assertEqual(len(registrations), 6)
        return {
            re.search(r",\s*(\w+),\s*(\w+),\s*GatherPacked\)>", line).groups()
            for line in registrations
            if not line.startswith("//")
        }

    def test_global_filter_keeps_all_supported_pairs(self):
        allowed = {"uint8_t", *self.output_types, *self.index_types}
        type_filter = GloballyAllowedTypesOpTypeImplFilter(allowed)
        self.assertEqual(
            self._enabled_cpu_registrations(type_filter), set(product(self.output_types, self.index_types))
        )

    def test_global_filter_checks_output_and_index_types_separately(self):
        for output_type, index_type in product(self.output_types, self.index_types):
            with self.subTest(output_type=output_type, index_type=index_type):
                type_filter = GloballyAllowedTypesOpTypeImplFilter({"uint8_t", output_type, index_type})
                self.assertEqual(self._enabled_cpu_registrations(type_filter), {(output_type, index_type)})

    def test_global_filter_rejects_pairs_with_a_disallowed_type(self):
        for allowed in ({"uint8_t", *self.output_types}, {"uint8_t", *self.index_types}):
            with self.subTest(allowed=allowed):
                self.assertEqual(self._enabled_cpu_registrations(GloballyAllowedTypesOpTypeImplFilter(allowed)), set())

    def test_model_filter_preserves_packed_storage_and_selected_pair(self):
        for output_type, index_type in product(self.output_types, self.index_types):
            with self.subTest(output_type=output_type, index_type=index_type):
                entry = json.dumps({"inputs": {"0": ["uint8_t"], "1": [index_type]}, "outputs": {"0": [output_type]}})
                manager = OperatorTypeUsageManager()
                manager.restore_from_config_entry("com.microsoft", "GatherPacked", entry)
                type_filter = manager.make_op_type_impl_filter()
                self.assertEqual(self._enabled_cpu_registrations(type_filter), {(output_type, index_type)})
                self.assertEqual(
                    json.loads(manager.get_config_entry("com.microsoft", "GatherPacked")), json.loads(entry)
                )
                self.assertIn(
                    "ORT_SPECIFY_OP_KERNEL_ARG_ALLOWED_TYPES(kMSDomain, GatherPacked, Input, 0, uint8_t);",
                    type_filter.get_cpp_entries(),
                )

    def test_model_filter_without_type_information_keeps_all_pairs(self):
        self.assertEqual(
            self._enabled_cpu_registrations(OperatorTypeUsageManager().make_op_type_impl_filter()),
            set(product(self.output_types, self.index_types)),
        )


if __name__ == "__main__":
    unittest.main()
