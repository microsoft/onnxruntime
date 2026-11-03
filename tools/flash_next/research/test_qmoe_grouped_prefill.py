from pathlib import Path
import json
import random
import tempfile
import unittest
from unittest.mock import patch

from build_qmoe_grouped_prefill import FLAG, transform
from run_qmoe_prefill256_native import verify_additional_dispatches
from run_qsa_prefill_score16_model import grouped_program_count, model_command
from run_qmoe_grouped_prefill_native import PROGRAMS
import run_qsa_prefill_score16_model as model_runner


class GroupedPrefillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill1024/build-v1/qmoe.cc").read_text()

    def test_permutation_and_scatter_restore_every_original_rank(self):
        rng = random.Random(3102)
        for tokens in (512, 1024):
            experts = [rng.randrange(512) for _ in range(tokens * 10)]
            rows = list(range(len(experts)))
            rng.shuffle(rows)
            rows.sort(key=lambda row: experts[row])
            self.assertEqual(sorted(rows), list(range(len(experts))))
            self.assertEqual([experts[row] for row in rows], sorted(experts))
            restored = [None] * len(rows)
            for sorted_row, original_row in enumerate(rows):
                restored[original_row] = (original_row // 10, experts[original_row], original_row % 10)
            self.assertEqual(restored, [(row // 10, expert, row % 10) for row, expert in enumerate(experts)])

    def test_original_gemm_activation_and_mix_arithmetic_are_unchanged(self):
        result = transform(self.source)
        self.assertEqual(result.count("ApplyMatMulNBits("), self.source.count("ApplyMatMulNBits("))
        self.assertEqual(result.count("projection_experts,"), 2)
        self.assertIn("(use_prefill512 || use_prefill1024)", result)
        self.assertIn("use_small_batch && !use_int4_cooperative", result)
        self.assertIn(f"{FLAG} must be 0 or 1", result)
        for name in ("QmoeFC1SwiGLUProgram", "QmoeFC2FinalMixProgram", "FusedFinalMix1TokenProgram"):
            self.assertEqual(result.count(name), self.source.count(name))
        self.assertIn("final_mix_input = &*restored_fc2", result)
        self.assertIn("projection_experts = use_grouped_prefill ? &*sorted_experts : &indirect_experts", result)

    def test_copy_kernels_only_move_existing_elements(self):
        source = Path(__file__).with_name("qmoe_grouped_prefill.inc").read_text()
        self.assertIn("(original_row / uniforms.k) * uniforms.hidden + column", source)
        self.assertIn("original_row * uniforms.hidden + column", source)
        self.assertIn("atomicStore(&cursor, 0u)", source)
        self.assertIn("atomicAdd(&cursor, 1u)", source)
        self.assertNotIn("dot(", source)
        self.assertNotIn("f16(", source)

    def test_profile_gate_rejects_missing_and_duplicate_grouping_programs(self):
        expected = {"QmoeGroupedHiddenGather": 1, "QmoeGroupedOutputScatter": 1}
        events = [{"cat": "Kernel", "name": "QMoE&" + name} for name in expected]
        self.assertEqual(verify_additional_dispatches(events, expected), expected)
        for invalid in (events[:1], events + events[:1]):
            with self.assertRaises(AssertionError):
                verify_additional_dispatches(invalid, expected)

    def test_model_profile_requires_all_five_grouping_programs(self):
        events = [{"cat": "Kernel", "name": "QMoE&" + name} for name in PROGRAMS]
        self.assertEqual(grouped_program_count(events), 1)
        for invalid in (events[:4], events + events[:1], []):
            with self.assertRaises(AssertionError):
                grouped_program_count(invalid)

    def test_control_and_candidate_preserve_qualified_score16_stack(self):
        old = ["bash", "-lc", "runner --tokens 256 --repetitions 3 --chunk 1024 "
               "--webgpu-validation-mode full --plugin /workspace/qwen38-int4-dense-block32/build-v2/"
               "libonnxruntime_providers_webgpu.so --output baseline.json STATS=baseline-stats STATIC=baseline-static"]
        provider = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-grouped-prefill/build-v1/"
                        "libonnxruntime_providers_webgpu.so")
        for mode in (0, 1):
            for profile in (False, True):
                actual = model_command(old, Path("/evidence/new"), "baseline", provider, mode, profile,
                                       grouped_prefill=True)
                self.assertIn(f"export {FLAG}={mode};", actual[-1])
                self.assertIn("export ORT_WEBGPU_QSA_PREFILL_SCORE16=1;", actual[-1])
                self.assertIn("--tokens " + ("2" if profile else "36"), actual[-1])
                self.assertIn("--repetitions 1 --chunk 1024", actual[-1])
                self.assertIn("--webgpu-validation-mode full", actual[-1])
                self.assertEqual("--profile-prefix" in actual[-1], profile)
                self.assertNotIn("baseline", actual[-1])
                self.assertIn("baseline", old[-1])

    def test_written_profile_selector_and_attempt_are_exact_for_both_variants(self):
        evidence = Path(__file__).parent
        for grouped in (False, True):
            variant = "qmoe-grouped-prefill-v1" if grouped else "qsa-prefill-score16-v1"
            selector = FLAG if grouped else "ORT_WEBGPU_QSA_PREFILL_SCORE16"
            parent = ("qsa-prefill-score16-v1-contexts-20261031" if grouped else
                      "dense-block32-v2-contexts-20261030")
            native_name = variant + "-native-gpu7-20261031.qualification.json"
            baseline = "dense-block32-v2-chunk1024-gpu7-full-standard2048-20261030.guard.json"
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                for name in (native_name, parent + ".qualification.json", baseline,
                             "int4-pending64-standard-contexts.json"):
                    (root / name).write_text((evidence / name).read_text())
                native = json.loads((root / native_name).read_text())
                parent_record = json.loads((root / (parent + ".qualification.json")).read_text())
                prefix = root / (variant + "-mode1-gpu7-full-profile2048-20261031-attempt1")

                def fake_run(arguments, **kwargs):
                    command = arguments[arguments.index("--") + 1:]
                    flags = {name: "1" for name in (
                        FLAG, "ORT_WEBGPU_QSA_PREFILL_SCORE16", "ORT_WEBGPU_QMOE_PREFILL1024",
                        "ORT_WEBGPU_DSA_QUERY_CACHE", "ORT_WEBGPU_DENSE_BLOCK32")}
                    prefix.with_suffix(".json").write_text(json.dumps({
                        "load_experiment_flags": flags, "profiling_scope": "session"}))
                    prefix.with_suffix(".guard.json").write_text(json.dumps({"command": command}))
                    prefix.with_suffix(".log").write_text("PASS\n")
                    if grouped:
                        events = [{"cat": "Kernel", "name": "QMoE&" + name}
                                  for name in PROGRAMS for _ in range(192)]
                    else:
                        events = [{"cat": "Kernel", "name": "QSA&" + model_runner.PROGRAM,
                                   "args": {"shapes": "inputs[0] = {1,1024,640}"}}
                                  for _ in range(48)]
                    (root / (prefix.name + "-profile_0.json")).write_text(json.dumps(events))

                def fake_sha(path):
                    return {"libonnxruntime_providers_webgpu.so": native["provider_sha256"],
                            "model.onnx": parent_record["graph_sha256"],
                            "genai_config.json": parent_record["config_sha256"]}.get(Path(path).name, "evidence")

                with patch.object(model_runner, "ROOT", root), \
                     patch.object(model_runner, "sha", side_effect=fake_sha), \
                     patch.object(model_runner, "prepare_model_cache",
                                  return_value={"evidence_path": str(root / "cache.json")}), \
                     patch.object(model_runner, "snapshot_model_cache", return_value=[]), \
                     patch.object(model_runner, "verify_metadata"), \
                     patch.object(model_runner, "verify_tokens"), \
                     patch.object(model_runner.subprocess, "run", side_effect=fake_run), \
                     patch("sys.argv", ["runner", "--phase", "profile", "--full-validation", "--attempt", "1"]):
                    model_runner.main(grouped_prefill=grouped)
                report = json.loads(prefix.with_suffix(".qualification.json").read_text())
                self.assertEqual(report["selector"], selector)
                self.assertEqual((report["selector_value"], report["attempt"]), (1, 1))
                field = "active_grouped_prefill_programs" if grouped else "active_prefill_score16_programs"
                self.assertEqual(report[field], 192 if grouped else 48)


if __name__ == "__main__":
    unittest.main()
