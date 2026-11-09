# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools.python.generate_matmul_nbits_cases import Case, generate_cases
from tools.python.run_matmul_nbits_experiments import (
    CaseError,
    load_manifest,
    logical_reference,
    run_experiments,
    validate_report,
)


def report_fixture(case: Case) -> dict:
    reference = logical_reference(case)
    logical = [
        {
            "component": "matmul_core",
            "metric": "dense_equivalent_ops",
            "convention_version": "core_arithmetic_v1",
            "availability": "available",
            "operation_count": reference["core_dense_equivalent_ops"],
        },
        {
            "component": "bias",
            "metric": "floating_point_ops",
            "convention_version": "bias_add_v1",
            "availability": "available",
            "operation_count": reference["bias_float_ops"],
        },
    ]
    expected = [
        dict(component, metric="floating_point_ops", convention_version="cpu_expected_flops_v1")
        for component in logical
    ]
    expected.append(
        {
            "component": "dequantization",
            "metric": "floating_point_ops",
            "convention_version": "cpu_expected_flops_v1",
            "availability": "available",
            "operation_count": str(2 * case.n * ((case.k + case.block_size - 1) // case.block_size) * case.block_size),
        }
    )

    def summaries(components):
        return [
            {
                "component": component["component"],
                "estimated_component_count": 1,
                "unavailable_component_count": 0,
                "subtotal_availability": "available",
                "known_operation_count": component["operation_count"],
                "complete_for_captured_component_scope": True,
            }
            for component in components
        ]

    total = str(sum(int(component["operation_count"]) for component in expected))
    return {
        "schema_version": 3,
        "report_kind": "runtime_workload",
        "run_state": "succeeded",
        "collection": {
            "collection_complete": True,
            "retained_record_count": 1,
            "record_limit_drops": 0,
            "payload_limit_drops": 0,
            "allocation_failure_drops": 0,
            "metadata_failure_drops": 0,
            "run_reuse_rejections": 0,
        },
        "component_summaries": summaries(logical),
        "expected_flops": {
            "whole_operator_work_complete": False,
            "component_summaries": summaries(expected),
        },
        "invocations": [
            {
                "provider": "CPUExecutionProvider",
                "domain": "com.microsoft",
                "op_type": "MatMulNBits",
                "execution_basis": "successful_kernel_execution",
                "input_shape": list(case.a_shape),
                "output_shape": [*case.a_shape[:-1], case.n],
                "backend_geometry": {"m": case.m, "n": case.n, "k": case.k, "batch_count": 1},
                "bias_present": case.bias,
                "input_element_type": 1 if case.dtype == "float32" else 10,
                "bits": case.bits,
                "block_size": case.block_size,
                "route": "packed_qnbit",
                "compute_mode": "sqnbit_fp32",
                "components": logical,
                "expected_flops": {
                    "model": "packed_float_single_padded_reconstruction_v1",
                    "availability": "available",
                    "total_availability": "available",
                    "expected_flops": total,
                    "known_flops_subtotal": total,
                    "components": expected,
                },
            }
        ],
    }


def fake_runtime(
    report: dict | None,
    *,
    difference: float = 0,
    initialization_report: bool = False,
    duplicate_report: bool = False,
    execution_error: bool = False,
):
    class Options:
        def __init__(self):
            self.prefix = None

        def add_session_config_entry(self, key, value):
            if key != "session.compute_estimate_report_path":
                raise ValueError(f"unexpected config key: {key}")
            self.prefix = Path(value)

    class Session:
        def __init__(self, path, sess_options, providers):
            if providers != ["CPUExecutionProvider"]:
                raise ValueError("unexpected provider")
            self.prefix = sess_options.prefix
            if initialization_report and self.prefix:
                self.prefix.with_name("estimate_initialization.json").write_text("{}")

        def run(self, names, feeds):
            if execution_error:
                raise RuntimeError("controlled runtime error")
            activation = feeds["A"]
            n = report["invocations"][0]["backend_geometry"]["n"] if report else 32
            output = np.zeros((*activation.shape[:-1], n), dtype=activation.dtype)
            if self.prefix:
                output += difference
                if report:
                    self.prefix.with_name("estimate_session-1_run-1.json").write_text(json.dumps(report))
                    if duplicate_report:
                        self.prefix.with_name("estimate_session-1_run-2.json").write_text(json.dumps(report))
            return [output]

    return SimpleNamespace(
        SessionOptions=Options,
        GraphOptimizationLevel=SimpleNamespace(ORT_DISABLE_ALL=0),
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL=0),
        InferenceSession=Session,
        get_available_providers=lambda: ["CPUExecutionProvider"],
        get_build_info=lambda: "test fixture; not a branch-wheel validation",
        __version__="test",
        __file__="test_fixture",
    )


class TestRunMatMulNBitsExperiments(unittest.TestCase):
    def setUp(self):
        self.case = Case((2, 3, 65), 32, bias=True)

    def test_valid_report_and_batched_geometry(self):
        report = report_fixture(self.case)
        validate_report(report, self.case)
        report["invocations"][0]["backend_geometry"].update(m=3, batch_count=2)
        validate_report(report, self.case)

    def test_report_failures_are_explicit(self):
        mutations = (
            ("report_invalid", lambda r: r.update(schema_version=2)),
            ("incomplete_capture", lambda r: r.update(invocations=[])),
            ("incomplete_capture", lambda r: r["collection"].update(record_limit_drops=1)),
            ("incomplete_capture", lambda r: r["collection"].update(collection_complete=False)),
            ("incomplete_capture", lambda r: r["collection"].pop("metadata_failure_drops")),
            ("metadata_mismatch", lambda r: r["invocations"][0].update(provider="CUDAExecutionProvider")),
            ("metadata_mismatch", lambda r: r["invocations"][0].update(input_shape=[6, 65])),
            ("metadata_mismatch", lambda r: r["invocations"][0].pop("compute_mode")),
            ("count_mismatch", lambda r: r["invocations"][0]["components"][0].update(operation_count="1")),
            ("report_invalid", lambda r: r["invocations"][0]["components"][0].update(operation_count=24960)),
            ("report_invalid", lambda r: r["component_summaries"][0].update(known_operation_count="0")),
            ("estimator_unavailable", lambda r: r.pop("expected_flops")),
            ("estimator_unavailable", lambda r: r["invocations"][0].pop("expected_flops")),
            ("report_invalid", lambda r: r["invocations"][0]["expected_flops"].update(expected_flops="0")),
            ("report_invalid", lambda r: r["expected_flops"].update(whole_operator_work_complete=True)),
            ("report_invalid", lambda r: r["invocations"][0]["components"][0].update(component=[])),
            ("metadata_mismatch", lambda r: r["invocations"][0].update(route=[])),
        )
        for status, mutate in mutations:
            with self.subTest(status=status, mutate=mutate):
                report = report_fixture(self.case)
                mutate(report)
                with self.assertRaises(CaseError) as caught:
                    validate_report(report, self.case)
                self.assertEqual(caught.exception.status, status)

    def test_unavailable_estimates_are_not_zero_or_pass(self):
        report = report_fixture(self.case)
        invocation = report["invocations"][0]
        invocation.update(route="packed_lut", compute_mode="lut")
        expected = invocation["expected_flops"]
        expected.pop("model")
        expected.pop("expected_flops")
        expected.update(
            availability="partial",
            total_availability="unavailable",
            known_flops_subtotal=str(self.case.m * self.case.n),
        )
        for index in (0, 2):
            component = expected["components"][index]
            component.pop("operation_count")
            component.update(availability="unavailable", unavailable_reason="unsupported_route")
            report["expected_flops"]["component_summaries"][index].update(
                estimated_component_count=0,
                unavailable_component_count=1,
                known_operation_count="0",
                complete_for_captured_component_scope=False,
            )
        with self.assertRaises(CaseError) as caught:
            validate_report(report, self.case)
        self.assertEqual(caught.exception.status, "estimate_unavailable")

    def test_manifest_validation_precedes_output_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generate_cases([self.case], root / "cases", 123)
            manifest_path = root / "cases/manifest.json"
            original = json.loads(manifest_path.read_text())
            for field, value in (
                ("case_id", "../escape"),
                ("inputs", "../outside.npz"),
                ("logical_reference", {}),
                ("output_shape", [6, 32]),
            ):
                manifest = copy.deepcopy(original)
                manifest["cases"][0][field] = value
                manifest_path.write_text(json.dumps(manifest))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    run_experiments(manifest_path, root / "results", fake_runtime(report_fixture(self.case)))
                self.assertFalse((root / "results").exists())
            original["cases"] = []
            manifest_path.write_text(json.dumps(original))
            with self.assertRaises(ValueError):
                load_manifest(manifest_path.resolve())

    def test_runner_persists_results_outputs_and_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generate_cases([self.case], root / "cases", 123)
            with redirect_stdout(io.StringIO()):
                result = run_experiments(
                    root / "cases/manifest.json",
                    root / "results",
                    fake_runtime(report_fixture(self.case)),
                    runtime_label="fixture",
                )
            self.assertTrue(result["complete"])
            self.assertEqual(result["status_counts"], {"pass": 1})
            self.assertEqual(result["executed_fp_accuracy"], "not_validated")
            record = result["cases"][0]
            self.assertTrue((root / "results" / record["report"]).is_file())
            with np.load(root / "results" / record["outputs"], allow_pickle=False) as outputs:
                np.testing.assert_array_equal(outputs["reporting_disabled"], outputs["reporting_enabled"])
            with (root / "results/summary.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["status"], "pass")
            self.assertEqual(rows[0]["expected_flops"], record["expected_flops"]["expected_flops"])
            self.assertEqual(json.loads((root / "results/results.json").read_text())["status_counts"], {"pass": 1})
            with self.assertRaises(FileExistsError):
                run_experiments(root / "cases/manifest.json", root / "results", fake_runtime(report_fixture(self.case)))

    def test_runner_failure_statuses_and_continuation(self):
        old_report = report_fixture(self.case)
        old_report.pop("expected_flops")
        old_report["invocations"][0].pop("expected_flops")
        for status, runtime in (
            ("report_missing", fake_runtime(None)),
            ("report_missing", fake_runtime(report_fixture(self.case), duplicate_report=True)),
            ("output_mismatch", fake_runtime(report_fixture(self.case), difference=1)),
            ("report_invalid", fake_runtime(report_fixture(self.case), initialization_report=True)),
            ("execution_failure", fake_runtime(report_fixture(self.case), execution_error=True)),
            ("estimator_unavailable", fake_runtime(old_report)),
        ):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                generate_cases([self.case], root / "cases", 123)
                with redirect_stderr(io.StringIO()):
                    result = run_experiments(root / "cases/manifest.json", root / "results", runtime, (RuntimeError,))
                self.assertEqual(result["status_counts"], {status: 1})
                self.assertTrue((root / "results/summary.csv").is_file())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generate_cases([self.case, Case((64,), 32)], root / "cases", 123)
            with redirect_stderr(io.StringIO()):
                result = run_experiments(
                    root / "cases/manifest.json",
                    root / "results",
                    fake_runtime(report_fixture(self.case), execution_error=True),
                    (RuntimeError,),
                )
            self.assertEqual(result["status_counts"], {"execution_failure": 2})

    def test_tolerances_are_explicit_and_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generate_cases([self.case], root / "cases", 123)
            for value in (float("nan"), float("inf"), -1):
                with self.assertRaises(ValueError):
                    run_experiments(
                        root / "cases/manifest.json",
                        root / "invalid",
                        fake_runtime(report_fixture(self.case)),
                        atol=value,
                    )
            with redirect_stdout(io.StringIO()):
                result = run_experiments(
                    root / "cases/manifest.json",
                    root / "tolerated",
                    fake_runtime(report_fixture(self.case), difference=0.001),
                    atol=0.002,
                )
            self.assertEqual(result["status_counts"], {"pass": 1})
            self.assertEqual(result["execution"]["atol"], 0.002)


if __name__ == "__main__":
    unittest.main()
