# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from tools.python.generate_matmul_nbits_cases import Case, generate_cases
from tools.python.profile_matmul_nbits_cuda import run
from tools.python.run_matmul_nbits_experiments import CaseError, validate_report
from tools.python.test_run_matmul_nbits_experiments import report_fixture


def cuda_fixture(case, route="cuda_fused"):
    report = report_fixture(case)
    invocation = report["invocations"][0]
    invocation.update(
        provider="CUDAExecutionProvider",
        execution_basis="successful_host_submission",
        route=route,
        compute_mode="cuda_floating_point",
        reconstruction_k=case.k,
    )
    invocation["expected_flops"]["model"] = "cuda_fused_nominal_reconstruction_v1"
    for component in invocation["expected_flops"]["components"]:
        component["convention_version"] = "cuda_expected_flops_v1"
    return report


class TestCudaMatMulNBitsCollector(unittest.TestCase):
    def test_cuda_routes_and_cpu_rejection(self):
        case = Case((128, 64), 32)
        for route in (
            "cuda_fused",
            "cuda_fpa_intb_gemv",
            "cuda_fpa_intb_gemm",
            "cuda_mixed_fpa_intb",
            "cuda_dequantize_gemm",
        ):
            report = cuda_fixture(case, route)
            validate_report(report, case, "CUDAExecutionProvider")
            with self.assertRaises(CaseError):
                validate_report(report, case)
        report["invocations"][0]["provider"] = "CPUExecutionProvider"
        with self.assertRaises(CaseError):
            validate_report(report, case, "CUDAExecutionProvider")

    def fake_runtime(self, case, report, *, fallback=False, mismatch=False):
        runtime = MagicMock()
        runtime.__version__ = "test"
        runtime.__file__ = "test-runtime"
        runtime.get_build_info.return_value = "test build"
        runtime.get_available_providers.return_value = ["CUDAExecutionProvider"]
        sessions = []

        def create_session(model, sess_options, providers):
            session = MagicMock()
            session.get_providers.return_value = ["CPUExecutionProvider" if fallback else "CUDAExecutionProvider"]
            options = dict(call.args for call in sess_options.add_session_config_entry.call_args_list)
            prefix = options.get("session.compute_estimate_report_path")
            output = np.zeros((*case.a_shape[:-1], case.n), dtype=case.dtype)
            if prefix and mismatch:
                output += 1
            session.io_binding.return_value.copy_outputs_to_cpu.return_value = [output]

            def execute(binding):
                if prefix:
                    Path(prefix + "_session-1_run-1.json").write_text(json.dumps(report))

            session.run_with_iobinding.side_effect = execute
            sessions.append((session, options, providers))
            return session

        runtime.InferenceSession.side_effect = create_session
        return runtime, sessions

    def test_capture_and_profile_flow(self):
        for mode in ("capture", "profile"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                case = Case((128, 64), 32)
                manifest = generate_cases([case], root / "cases", 123)
                runtime, sessions = self.fake_runtime(case, cuda_fixture(case))
                args = SimpleNamespace(
                    manifest=root / "cases/manifest.json",
                    case_id=manifest["cases"][0]["case_id"],
                    output=root / "result",
                    device=0,
                    warmup=2,
                    repeats=3,
                    mode=mode,
                    nvtx=False,
                )
                with (
                    patch.dict("sys.modules", {"onnxruntime": runtime}),
                    patch("tools.python.profile_matmul_nbits_cuda.subprocess.check_output", return_value="GPU fixture"),
                ):
                    run(args)
                metadata = json.loads((root / "result/metadata.json").read_text())
                self.assertTrue(metadata["complete"])
                self.assertEqual(len(sessions), 2 if mode == "capture" else 1)
                for session, options, providers in sessions:
                    self.assertEqual(options["session.disable_cpu_ep_fallback"], "1")
                    self.assertEqual(providers[0][1]["enable_cuda_graph"], "0")
                    session.disable_fallback.assert_called_once()
                    session.io_binding.return_value.synchronize_outputs.assert_called()
                if mode == "profile":
                    self.assertEqual(sessions[0][0].run_with_iobinding.call_count, 5)
                    self.assertFalse(list((root / "result").glob("estimate_*.json")))
                else:
                    self.assertEqual(metadata["invocation"]["provider"], "CUDAExecutionProvider")

    def test_fallback_and_output_mismatch_fail_explicitly(self):
        for fallback, mismatch in ((True, False), (False, True)):
            with self.subTest(fallback=fallback), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                case = Case((128, 64), 32)
                manifest = generate_cases([case], root / "cases", 123)
                runtime, _ = self.fake_runtime(case, cuda_fixture(case), fallback=fallback, mismatch=mismatch)
                args = SimpleNamespace(
                    manifest=root / "cases/manifest.json",
                    case_id=manifest["cases"][0]["case_id"],
                    output=root / "result",
                    device=0,
                    warmup=1,
                    repeats=1,
                    mode="capture",
                    nvtx=False,
                )
                with (
                    patch.dict("sys.modules", {"onnxruntime": runtime}),
                    patch("tools.python.profile_matmul_nbits_cuda.subprocess.check_output", return_value="GPU fixture"),
                    self.assertRaises(ValueError),
                ):
                    run(args)
                self.assertFalse(json.loads((root / "result/metadata.json").read_text())["complete"])


if __name__ == "__main__":
    unittest.main()
