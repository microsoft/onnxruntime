# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Run CPU MatMulNBits capture/report checks, not executed-FLOP accuracy measurements."""

import argparse
import csv
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from google.protobuf.message import DecodeError

if __package__:
    from .generate_matmul_nbits_cases import Case
else:
    from generate_matmul_nbits_cases import Case


class CaseError(Exception):
    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


def require(condition: bool, message: str, status: str = "report_invalid"):
    if not condition:
        raise CaseError(status, message)


def artifact_path(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("artifact paths must be nonempty relative paths")
    path = (root / value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"artifact missing or outside manifest directory: {value}")
    return path


def logical_reference(case: Case) -> dict:
    return {
        "convention": "matmul_nbits_logical_v1",
        "m": str(case.m),
        "n": str(case.n),
        "k": str(case.k),
        "core_dense_equivalent_ops": str(2 * case.m * case.n * case.k),
        "bias_float_ops": str(case.m * case.n if case.bias else 0),
    }


def load_manifest(path: Path) -> list[tuple[dict, Case]]:
    path = path.resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int:
        raise ValueError("manifest must be a schema-versioned object")
    if manifest["schema_version"] != 1 or manifest.get("generator") != "generate_matmul_nbits_cases":
        raise ValueError("expected a schema-v1 generate_matmul_nbits_cases manifest")
    records = manifest.get("cases")
    if not isinstance(records, list) or not records:
        raise ValueError("manifest cases must be a nonempty list")
    cases = []
    seen = set()
    for index, record in enumerate(records):
        try:
            if not isinstance(record, dict) or not isinstance(record.get("settings"), dict):
                raise ValueError("expected a case object with settings")
            settings = record["settings"]
            if not isinstance(settings.get("a_shape"), list):
                raise ValueError("settings.a_shape must be a list")
            case = Case(**(settings | {"a_shape": tuple(settings["a_shape"])}))
            case_id = f"matmul_nbits_{case.identity()}"
            if record.get("case_id") != case_id or case_id in seen:
                raise ValueError("invalid or duplicate case_id")
            seen.add(case_id)
            if record.get("logical_reference") != logical_reference(case):
                raise ValueError("logical_reference does not match independently recomputed counts")
            if record.get("input_names") != ["A"] or record.get("output_names") != ["Y"]:
                raise ValueError("expected input A and output Y")
            if record.get("output_shape") != [*case.a_shape[:-1], case.n]:
                raise ValueError("output_shape does not match settings")
            if (
                type(record.get("expected_invocations_per_run")) is not int
                or record["expected_invocations_per_run"] != 1
            ):
                raise ValueError("expected exactly one invocation per run")
            for name in ("model", "inputs"):
                artifact_path(path.parent, record.get(name))
        except (TypeError, ValueError) as error:
            raise ValueError(f"manifest case {index}: {error}") from error
        cases.append((record, case))
    return cases


def components_by_name(values, names: tuple[str, ...]) -> dict:
    require(isinstance(values, list), "components must be a list")
    require(
        all(isinstance(value, dict) and isinstance(value.get("component"), str) for value in values),
        "component entries must be objects with string names",
    )
    require(
        len(values) == len(names) and {value.get("component") for value in values} == set(names),
        f"expected exactly these components: {names}",
    )
    return {value["component"]: value for value in values}


def decimal_count(value) -> int:
    require(isinstance(value, str) and value.isascii() and value.isdecimal(), "count must be a decimal string")
    count = int(value)
    require(count <= (1 << 64) - 1, "reported count exceeds uint64")
    return count


def check_summaries(values, components: dict):
    summaries = components_by_name(values, tuple(components))
    for name, component in components.items():
        available = component.get("availability") == "available"
        summary = summaries[name]
        require(
            all(type(summary.get(key)) is int for key in ("estimated_component_count", "unavailable_component_count")),
            f"{name}: summary counts must be integers",
        )
        require(summary.get("estimated_component_count") == int(available), f"{name}: incorrect estimated count")
        require(
            summary.get("unavailable_component_count") == int(not available), f"{name}: incorrect unavailable count"
        )
        require(summary.get("subtotal_availability") == "available", f"{name}: subtotal unavailable")
        require(summary.get("complete_for_captured_component_scope") is available, f"{name}: incorrect completeness")
        expected = decimal_count(component.get("operation_count")) if available else 0
        require(decimal_count(summary.get("known_operation_count")) == expected, f"{name}: subtotal mismatch")


def validate_report(report: dict, case: Case):
    """Validate capture and logical counts; do not reimplement route-aware FLOP formulas."""
    require(isinstance(report, dict), "report must be an object")
    require(type(report.get("schema_version")) is int and report["schema_version"] == 3, "expected schema version 3")
    require(report.get("report_kind") == "runtime_workload", "expected runtime_workload report")
    require(report.get("run_state") == "succeeded", "report run did not succeed", "incomplete_capture")
    collection = report.get("collection")
    require(isinstance(collection, dict), "collection must be an object")
    require(collection.get("collection_complete") is True, "collection incomplete", "incomplete_capture")
    require(collection.get("retained_record_count") == 1, "expected one retained record", "incomplete_capture")
    for key in (
        "record_limit_drops",
        "payload_limit_drops",
        "allocation_failure_drops",
        "metadata_failure_drops",
        "run_reuse_rejections",
    ):
        require(type(collection.get(key)) is int and collection[key] == 0, f"{key} must be zero", "incomplete_capture")
    invocations = report.get("invocations")
    require(isinstance(invocations, list) and len(invocations) == 1, "expected one invocation", "incomplete_capture")
    invocation = invocations[0]
    require(isinstance(invocation, dict), "invocation must be an object")
    expected_scope = report.get("expected_flops")
    expected = invocation.get("expected_flops")
    require(
        isinstance(expected_scope, dict) and isinstance(expected, dict),
        "missing expected_flops: install a current branch-built wheel",
        "estimator_unavailable",
    )
    expected_metadata = {
        "provider": "CPUExecutionProvider",
        "domain": "com.microsoft",
        "op_type": "MatMulNBits",
        "execution_basis": "successful_kernel_execution",
        "input_shape": list(case.a_shape),
        "output_shape": [*case.a_shape[:-1], case.n],
        "bias_present": case.bias,
        "input_element_type": 1 if case.dtype == "float32" else 10,
        "bits": case.bits,
        "block_size": case.block_size,
    }
    for key, value in expected_metadata.items():
        require(invocation.get(key) == value, f"captured {key} mismatch", "metadata_mismatch")
    geometry = invocation.get("backend_geometry")
    require(isinstance(geometry, dict), "backend_geometry must be an object")
    require(geometry.get("n") == case.n and geometry.get("k") == case.k, "backend N/K mismatch", "metadata_mismatch")
    require(
        all(type(geometry.get(key)) is int and geometry[key] > 0 for key in ("m", "batch_count")),
        "invalid backend M/batch_count",
        "metadata_mismatch",
    )
    require(geometry["m"] * geometry["batch_count"] == case.m, "backend M/batch mismatch", "metadata_mismatch")
    modes = {
        "unpacked": ("float32_gemm",),
        "packed_qnbit": ("sqnbit_fp32", "hqnbit_fp16", "sqnbit_int8", "hqnbit_int8", "bhqnbit_bf16"),
        "packed_lut": ("lut",),
    }
    require(
        isinstance(invocation.get("route"), str)
        and invocation["route"] in modes
        and invocation.get("compute_mode") in modes[invocation["route"]],
        "missing or inconsistent CPU route/compute_mode",
        "metadata_mismatch",
    )
    logical = components_by_name(invocation.get("components"), ("matmul_core", "bias"))
    reference = logical_reference(case)
    for name, metric, convention, count in (
        ("matmul_core", "dense_equivalent_ops", "core_arithmetic_v1", reference["core_dense_equivalent_ops"]),
        ("bias", "floating_point_ops", "bias_add_v1", reference["bias_float_ops"]),
    ):
        component = logical[name]
        require(component.get("availability") == "available", f"{name}: logical estimate unavailable", "count_mismatch")
        require(
            component.get("metric") == metric and component.get("convention_version") == convention,
            f"{name}: logical convention mismatch",
        )
        require(
            decimal_count(component.get("operation_count")) == int(count), f"{name}: logical mismatch", "count_mismatch"
        )
    check_summaries(report.get("component_summaries"), logical)
    require(
        expected_scope.get("whole_operator_work_complete") is False,
        "expected_flops must not claim whole-operator completeness",
    )
    components = components_by_name(expected.get("components"), ("matmul_core", "bias", "dequantization"))
    available = 0
    known = 0
    for name, component in components.items():
        require(
            component.get("metric") == "floating_point_ops"
            and component.get("convention_version") == "cpu_expected_flops_v1",
            f"{name}: expected-FLOP convention mismatch",
        )
        if component.get("availability") == "available":
            available += 1
            known += decimal_count(component.get("operation_count"))
        else:
            require(
                component.get("availability") == "unavailable"
                and isinstance(component.get("unavailable_reason"), str)
                and bool(component["unavailable_reason"])
                and "operation_count" not in component,
                f"{name}: invalid unavailable component",
            )
    require(decimal_count(expected.get("known_flops_subtotal")) == known, "expected-FLOP known subtotal mismatch")
    require(
        expected.get("availability") == ("available" if available == 3 else "partial" if available else "unavailable"),
        "expected-FLOP availability mismatch",
    )
    check_summaries(expected_scope.get("component_summaries"), components)
    require(
        components["bias"].get("availability") == "available"
        and decimal_count(components["bias"].get("operation_count")) == int(reference["bias_float_ops"]),
        "expected-FLOP bias mismatch",
        "count_mismatch",
    )
    if available != 3:
        require(
            expected.get("total_availability") == "unavailable" and "expected_flops" not in expected,
            "partial estimate must not have a total",
        )
        reasons = {
            name: component["unavailable_reason"]
            for name, component in components.items()
            if component["availability"] != "available"
        }
        raise CaseError("estimate_unavailable", f"route-aware components unavailable: {reasons}")
    require(isinstance(expected.get("model"), str) and bool(expected["model"]), "missing expected-FLOP model")
    require(expected.get("total_availability") == "available", "expected-FLOP total unavailable")
    require(decimal_count(expected.get("expected_flops")) == known, "expected-FLOP total mismatch")


def load_activation(model_path: Path, input_path: Path, case: Case) -> np.ndarray:
    model = onnx.load(model_path)
    require(len(model.graph.node) == 1, "model must have exactly one node", "input_failure")
    node = model.graph.node[0]
    require(node.domain == "com.microsoft" and node.op_type == "MatMulNBits", "incorrect operator", "input_failure")
    attributes = {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute}
    for name, value in (("K", case.k), ("N", case.n), ("bits", case.bits), ("block_size", case.block_size)):
        require(attributes.get(name) == value, f"model {name} mismatch", "input_failure")
    require(
        attributes.get("accuracy_level", 0) == case.accuracy_level, "model accuracy_level mismatch", "input_failure"
    )
    for index, present in ((3, case.zero_points), (5, case.bias)):
        require(
            bool(node.input[index] if len(node.input) > index else "") == present,
            f"model optional input {index} mismatch",
            "input_failure",
        )
    with np.load(input_path, allow_pickle=False) as inputs:
        require(inputs.files == ["A"], "expected only input A", "input_failure")
        activation = inputs["A"]
    require(
        activation.shape == case.a_shape and activation.dtype == np.dtype(case.dtype),
        "activation shape/dtype mismatch",
        "input_failure",
    )
    require(bool(np.isfinite(activation).all()), "nonfinite activation", "input_failure")
    return activation


def run_case(
    record: dict,
    case: Case,
    root: Path,
    directory: Path,
    ort,
    runtime_errors: tuple,
    threads: int,
    rtol: float,
    atol: float,
) -> dict:
    result = {
        "case_id": record["case_id"],
        "settings": record["settings"],
        "logical_reference": logical_reference(case),
        "executed_fp_accuracy": "not_validated",
    }
    directory.mkdir()
    stage = "input_failure"
    try:
        activation = load_activation(artifact_path(root, record["model"]), artifact_path(root, record["inputs"]), case)

        def create_session(reporting: bool):
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            if reporting:
                options.add_session_config_entry("session.compute_estimate_report_path", str(directory / "estimate"))
            return ort.InferenceSession(
                str(root / record["model"]), sess_options=options, providers=["CPUExecutionProvider"]
            )

        stage = "execution_failure"
        baseline = create_session(False)
        reference = baseline.run(["Y"], {"A": activation})[0]
        require(not list(directory.glob("estimate_*.json")), "disabled reporting produced a report")
        captured = create_session(True)
        require(not list(directory.glob("estimate_*.json")), "initialization produced a report")
        observed = captured.run(["Y"], {"A": activation})[0]
        for value in (reference, observed):
            require(
                list(value.shape) == record["output_shape"] and value.dtype == np.dtype(case.dtype),
                "output shape/dtype mismatch",
                "output_mismatch",
            )
            require(bool(np.isfinite(value).all()), "nonfinite output", "output_mismatch")
        np.savez(directory / "outputs.npz", reporting_disabled=reference, reporting_enabled=observed)
        result["outputs"] = f"{record['case_id']}/outputs.npz"
        result["max_abs_output_difference"] = float(np.max(np.abs(observed.astype(np.float64) - reference)))
        reports = list(directory.glob("estimate_*.json"))
        require(len(reports) == 1, f"expected one report, found {len(reports)}", "report_missing")
        result["report"] = f"{record['case_id']}/{reports[0].name}"
        stage = "report_invalid"
        report = json.loads(reports[0].read_text(encoding="utf-8"))
        if isinstance(report, dict) and isinstance(report.get("invocations"), list) and len(report["invocations"]) == 1:
            invocation = report["invocations"][0]
            if isinstance(invocation, dict):
                for key in ("route", "compute_mode"):
                    result[key] = invocation.get(key) if isinstance(invocation.get(key), str) else None
                result["expected_flops"] = invocation.get("expected_flops")
        require(
            bool(np.allclose(observed, reference, rtol=rtol, atol=atol, equal_nan=False)),
            "reporting-enabled outputs differ from reporting-disabled outputs",
            "output_mismatch",
        )
        validate_report(report, case)
        result.update(status="pass", message="capture, logical counts and reporting-on/off outputs validated")
    except CaseError as error:
        result.update(status=error.status, message=str(error))
    except runtime_errors as error:
        result.update(status="execution_failure", message=f"{type(error).__name__}: {error}")
    except (OSError, ValueError, DecodeError) as error:
        result.update(status=stage, message=f"{type(error).__name__}: {error}")
    return result


def run_experiments(
    manifest_path: Path,
    output: Path,
    ort,
    runtime_errors: tuple = (),
    threads: int = 1,
    rtol: float = 0.0,
    atol: float = 0.0,
    runtime_label: str = "",
) -> dict:
    manifest_path = manifest_path.resolve()
    output = output.resolve()
    if type(threads) is not int or threads <= 0:
        raise ValueError("threads must be a positive integer")
    if not all(math.isfinite(value) and value >= 0 for value in (rtol, atol)):
        raise ValueError("rtol and atol must be finite and nonnegative")
    cases = load_manifest(manifest_path)
    if "CPUExecutionProvider" not in ort.get_available_providers():
        raise ValueError("CPUExecutionProvider is unavailable")
    output.mkdir(parents=True, exist_ok=False)
    results = {
        "schema_version": 1,
        "complete": False,
        "manifest": str(manifest_path),
        "scope": "cpu_capture_logical_counts_and_reporting_equivalence_only",
        "executed_fp_accuracy": "not_validated",
        "runtime": {
            "label": runtime_label,
            "onnxruntime_version": ort.__version__,
            "onnxruntime_file": ort.__file__,
            "onnxruntime_build_info": ort.get_build_info(),
            "available_providers": ort.get_available_providers(),
            "python": sys.version,
            "python_executable": sys.executable,
            "numpy_version": np.__version__,
            "onnx_version": onnx.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "execution": {
            "provider": "CPUExecutionProvider",
            "threads": threads,
            "rtol": rtol,
            "atol": atol,
            "graph_optimization": "disabled",
            "mode": "sequential",
            "runs_per_session": 1,
        },
        "requested_cases": len(cases),
        "cases": [],
    }
    result_path = output / "results.json"
    result_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for record, case in cases:
        result = run_case(
            record, case, manifest_path.parent, output / record["case_id"], ort, runtime_errors, threads, rtol, atol
        )
        results["cases"].append(result)
        result_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        stream = sys.stdout if result["status"] == "pass" else sys.stderr
        print(f"{record['case_id']}: {result['status']}: {result['message']}", file=stream)
    results["status_counts"] = dict(Counter(case["status"] for case in results["cases"]))
    results["route_counts"] = dict(Counter(case.get("route") or "unrecorded" for case in results["cases"]))
    columns = (
        "case_id",
        "status",
        "route",
        "compute_mode",
        "expected_flops",
        "max_abs_output_difference",
        "report",
        "message",
    )
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for result in results["cases"]:
            row = {key: result.get(key, "") for key in columns}
            expected = result.get("expected_flops")
            row["expected_flops"] = expected.get("expected_flops", "") if isinstance(expected, dict) else ""
            writer.writerow(row)
    results["complete"] = True
    result_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New results directory; existing paths are rejected")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--rtol", type=float, default=0.0, help="Reporting-on/off relative tolerance (default: exact)")
    parser.add_argument("--atol", type=float, default=0.0, help="Reporting-on/off absolute tolerance (default: exact)")
    parser.add_argument(
        "--runtime-label", default="", help="User-supplied wheel/build provenance; not independently verified"
    )
    args = parser.parse_args()
    try:
        from onnxruntime.capi.onnxruntime_pybind11_state import (  # noqa: PLC0415
            EngineError,
            EPFail,
            Fail,
            InvalidArgument,
            InvalidGraph,
            NoSuchFile,
            NotImplemented,
            RuntimeException,
        )

        import onnxruntime as ort  # noqa: PLC0415
    except ImportError as error:
        parser.exit(2, f"Cannot import ORT: {error}. Use the branch-built wheel and avoid source-package shadowing.\n")
    try:
        results = run_experiments(
            args.manifest,
            args.output,
            ort,
            (EPFail, EngineError, Fail, InvalidArgument, InvalidGraph, NoSuchFile, NotImplemented, RuntimeException),
            args.threads,
            args.rtol,
            args.atol,
            args.runtime_label,
        )
    except (OSError, ValueError) as error:
        parser.exit(2, f"Experiment setup/output failed: {error}\n")
    print(f"Results: {args.output / 'results.json'}; statuses: {results['status_counts']}")
    if any(case["status"] != "pass" for case in results["cases"]):
        parser.exit(1)


if __name__ == "__main__":
    main()
