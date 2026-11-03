import copy
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import qualify_qsa_hist_rank as qualification
import run_hc_up_parallel2_model as runner
from build_qsa_hist_rank import FLAG, HISTOGRAM, UNIFORM_PROGRAM


def windows(ratio=1.005, wins=4):
    return {
        "geometric_mean_matched_window_ratio": ratio,
        "matched_window_pairs": 16,
        "all_request_output_tokens_match": True,
        "per_request": [
            {"repetition": i, "qsa_hist_rank0_median_window_tps": 70.0,
             "qsa_hist_rank1_median_window_tps": 70.4 if i < wins else 69.8}
            for i in range(4)],
    }


class HistogramQualificationTests(unittest.TestCase):
    def test_standard_verifier_uses_actual_chunk_and_preserves_original_default(self):
        raw = {"load_experiment_flags": {FLAG: "1"}}
        with patch.object(qualification, "verify_standard", return_value={"passed": True}) as verify:
            result = qualification.verify_histogram_standard(raw, {}, {}, {}, [], 32768, False)
        self.assertEqual(result, {"passed": True})
        verify.assert_called_once_with(raw, {}, {}, {}, [], 32768, False, chunk=1024)
        self.assertEqual(inspect.signature(runner.verify_standard).parameters["chunk"].default, 256)
        raw["load_experiment_flags"][FLAG] = "0"
        with self.assertRaises(AssertionError):
            qualification.verify_histogram_standard(raw, {}, {}, {}, [], 32768, False)

    def test_exact_screen_bound_and_repeated_long_positive_requirement(self):
        qualification.verify_window_result(windows(0.99, 0), False)
        for ratio in (0.989999, float("nan"), float("inf")):
            with self.assertRaises(AssertionError):
                qualification.verify_window_result(windows(ratio), False)
        qualification.verify_window_result(windows(wins=3), True)
        for result in (windows(1.0), windows(wins=2)):
            with self.assertRaises(AssertionError):
                qualification.verify_window_result(result, True)
        for key, bad_value in (("all_request_output_tokens_match", False),
                               ("matched_window_pairs", 15)):
            result = windows()
            result[key] = bad_value
            with self.assertRaises(AssertionError):
                qualification.verify_window_result(result, False)
        result = windows()
        result["per_request"][0]["qsa_hist_rank1_median_window_tps"] = float("nan")
        with self.assertRaises(AssertionError):
            qualification.verify_window_result(result, False)

    def test_standard_requires_both_repeats_and_all_short_modes(self):
        def proof(native, context, phase, full, attempt=1, sources=None):
            if phase == "profile":
                result = {"candidate_program_dispatches": {
                    UNIFORM_PROGRAM: 0, HISTOGRAM: 0}, "original_program_dispatches": 12}
            else:
                result = windows()
            return {"result": result}

        with patch.object(qualification, "read_proof", side_effect=proof) as read:
            qualification.verify_prerequisites({}, "standard")
        calls = [(call.args[1], call.args[2], call.args[3],
                  call.args[4] if len(call.args) > 4 else 1) for call in read.call_args_list]
        self.assertEqual(len(calls), 10)
        for attempt in (1, 2):
            for full in (True, False):
                self.assertIn((32768, "windows", full, attempt), calls)
        for context in (2048, 8192):
            self.assertIn((context, "profile", True, 1), calls)
            for full in (True, False):
                self.assertIn((context, "windows", full, 1), calls)
        with patch.object(qualification, "read_proof", side_effect=FileNotFoundError):
            with self.assertRaises(FileNotFoundError):
                qualification.verify_prerequisites({}, "standard")

    def test_short_profile_must_prove_original_and_zero_candidate_dispatches(self):
        good = {"result": windows()}
        bad = {"result": {"candidate_program_dispatches": {
            UNIFORM_PROGRAM: 12, HISTOGRAM: 12}, "original_program_dispatches": 0}}
        with patch.object(qualification, "read_proof",
                          side_effect=[copy.deepcopy(good) for _ in range(4)] + [bad]):
            with self.assertRaises(AssertionError):
                qualification.verify_prerequisites({}, "standard")

    def test_generic_long_only_runner_stays_blocked(self):
        with patch("sys.argv", ["runner", "--phase", "standard"]):
            with self.assertRaises(SystemExit):
                runner.main(variant="qsa-hist-rank-v2", initial_context=32768)
        with patch("sys.argv", ["runner"]):
            with self.assertRaises(ValueError):
                runner.main(histogram_qualification=True)

    def test_persisted_short_profile_reports_actual_counts(self):
        evidence = Path(__file__).parent
        native_name = "qsa-hist-rank-v2-native-gpu7-20261031.qualification.json"
        parent_name = "qsa-prefill-score16-v1-contexts-20261031.qualification.json"
        baseline = "qsa-prefill-score16-v1-chunk1024-gpu7-full-standard2048-20261031"
        raw = json.loads((evidence / (baseline + ".json")).read_text())
        raw["load_experiment_flags"][FLAG] = "1"
        raw["profiling_scope"] = "decode_only"
        raw["profile_decode_after_tokens"] = 35
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in (native_name, parent_name, baseline + ".guard.json"):
                (root / name).write_bytes((evidence / name).read_bytes())
            (root / "int4-pending64-standard-contexts.json").write_text(
                json.dumps({"samples": [{"context": 2048}]}))
            prefix = root / "qsa-hist-rank-v2-chunk1024-gpu7-full-profile2048-20261031"

            def execute(command, **kwargs):
                benchmark_command = command[command.index("--") + 1:]
                prefix.with_suffix(".json").write_text(json.dumps(raw))
                prefix.with_suffix(".guard.json").write_text(json.dumps({"command": benchmark_command}))
                prefix.with_suffix(".log").write_text("")
                counts = {"SparseAttentionIndexerQsaParallelRank": 12,
                          "DynamicSparseAttentionQueryCache": 12, "MatMulNBitsBlock32F16": 159}
                events = [{"cat": "Kernel", "name": name}
                          for name, count in counts.items() for _ in range(count)]
                prefix.with_name(prefix.name + "-profile-ctx2048-rep0-configured_mock.json").write_text(
                    json.dumps(events))

            with patch.object(runner, "ROOT", root), \
                    patch.object(qualification, "verify_prerequisites"), \
                    patch.object(runner, "prepare_model_cache", return_value={
                        "evidence_path": str(root / "cache.json")}), \
                    patch.object(runner, "snapshot_model_cache", return_value=[]), \
                    patch.object(runner.subprocess, "run", side_effect=execute), \
                    patch.object(runner, "verify_metadata"), patch.object(runner, "verify_tokens"), \
                    patch("sys.argv", ["runner", "--context", "2048", "--phase", "profile",
                                       "--full-validation"]):
                runner.main(variant="qsa-hist-rank-v2", experiment_flag=FLAG, stamp="20261031",
                            parent_variant="qsa-prefill-score16-v1", parent_stamp="20261031",
                            chunk=1024, parent_standard_prefix=baseline,
                            parent_provider="/home/sayanshaw/repos/qwen38-int4-qsa-prefill-score16/build-v1/"
                                            "libonnxruntime_providers_webgpu.so",
                            program_name=UNIFORM_PROGRAM,
                            original_program_name="SparseAttentionIndexerQsaParallelRank",
                            dispatches=12, additional_programs=(HISTOGRAM,), initial_context=32768,
                            histogram_qualification=True)
            result = json.loads(prefix.with_suffix(".qualification.json").read_text())["result"]
            self.assertEqual(result["candidate_program_dispatches"], {UNIFORM_PROGRAM: 0, HISTOGRAM: 0})
            self.assertEqual(result["original_program_dispatches"], 12)


if __name__ == "__main__":
    unittest.main()
