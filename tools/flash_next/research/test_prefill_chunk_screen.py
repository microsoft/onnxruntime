import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_prefill_chunk_screen as screen


class PrefillChunkScreenTests(unittest.TestCase):
    def run_screen(self, *, extended, tokens, repetitions, corrupt_after_prefix=False,
                   standard=False, dense=False, chunk=64, invalid_full_proof=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = "expert-bitcast-unpack-v2-gpu7-standard2048-20261029"
            original_provider = "/workspace/qwen38-int4-expert-bitcast-unpack/build-v2/libonnxruntime_providers_webgpu.so"
            command = ["bash", "-lc",
                       f"set -e; exec benchmark --plugin {original_provider} --contexts 2048 "
                       f"--tokens 256 --repetitions 3 --output {baseline}.json"]
            references = [dict(context=2048, prompt_sha256=f"prompt{i}", output_token_ids=list(range(256)))
                          for i in range(4)]
            inputs = {
                baseline + ".qualification.json": dict(provider_sha256="baseline", graph_sha256="graph"),
                baseline + ".guard.json": dict(command=command),
                "int4-pending64-standard-contexts.json": dict(config_sha256="config", samples=references),
                "qmoe-extended-prefill-v1-native-gpu7-20261029.qualification.json":
                    dict(provider_sha256="candidate", baseline_provider_sha256="baseline"),
                "prefill-dense-stack-v1-native-gpu7-20261029.qualification.json":
                    dict(provider_sha256="stack", baseline_provider_sha256="candidate",
                         status="combined_dense_and_prefill_native_qualified"),
            }
            for name, value in inputs.items():
                (root / name).write_text(json.dumps(value))
            if standard:
                for context in (2048, 8192, 32768):
                    variant = "prefill-dense-stack-v1" if dense else "qmoe-extended-prefill-v1"
                    name = f"{variant}-chunk{chunk}-gpu7-screen{context}-t256-r3-20261029.diagnostic.json"
                    (root / name).write_text(json.dumps(dict(
                        provider_sha256="stack" if dense else "candidate", original_output_prefixes_match=True,
                        context=context, chunk=chunk,
                        original_prompts_match=not (invalid_full_proof and context == 32768),
                        zero_major_faults=True, full_validation=True, output_tokens=256, repetitions=3)))

            def execute(args, check):
                self.assertFalse(invalid_full_proof, "GPU work must not start with a failed full-validation proof")
                self.assertTrue(check)
                prefix = Path(args[args.index("--output-prefix") + 1])
                actual_command = args[args.index("--") + 1:]
                prefix.with_suffix(".guard.json").write_text(json.dumps(dict(
                    status="completed", exit_code=0, command=actual_command, gpu_index=7)))
                samples = copy.deepcopy(references[:repetitions + 1])
                for sample in samples:
                    sample["output_token_ids"] = sample["output_token_ids"][:tokens]
                    sample["process_resource_usage"] = dict(
                        prefill=dict(major_page_faults=0), decode=dict(major_page_faults=0))
                if corrupt_after_prefix:
                    samples[-1]["output_token_ids"][63] = -1
                raw = dict(status="completed", plugin_sha256="stack" if dense else "candidate" if extended else "baseline",
                           graph_sha256="graph", config_sha256="config", chunk=chunk, provider="webgpu",
                           load_experiment_flags={"ORT_WEBGPU_DENSE_BITCAST_UNPACK": "1",
                                                  "ORT_WEBGPU_QMOE_EXTENDED_PREFILL": "1"},
                           webgpu_validation_mode_override=None if standard else "full",
                           samples=samples, summary={"2048:configured": {}})
                prefix.with_suffix(".json").write_text(json.dumps(raw))
                prefix.with_suffix(".log").write_text("completed")

            argv = ["screen", "--chunks", str(chunk), "--tokens", str(tokens), "--repetitions", str(repetitions)]
            if extended:
                argv.append("--extended-prefill")
            if standard:
                argv.append("--standard-validation")
            if dense:
                argv.append("--dense-stack")
            def hash_file(path):
                if path.suffix != ".so":
                    return "graph"
                if "prefill-dense-stack" in str(path):
                    return "stack"
                return "candidate" if "extended-prefill" in str(path) else "baseline"

            with patch.object(screen, "ROOT", root), patch.object(screen, "sha", side_effect=hash_file), \
                    patch.object(screen, "snapshot", return_value={}), \
                    patch.object(screen, "prepare_model_cache", return_value={"evidence_path": str(root / "cache.json")}), \
                    patch.object(screen, "snapshot_model_cache", return_value={}), \
                    patch.object(screen.subprocess, "run", side_effect=execute), \
                    patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                if invalid_full_proof:
                    with self.assertRaises(AssertionError):
                        screen.main()
                    return
                elif corrupt_after_prefix:
                    with self.assertRaisesRegex(RuntimeError, "failed original-prefix/fault gates"):
                        screen.main()
                else:
                    screen.main()
            reports = list(root.glob("*-standard-validation-*.diagnostic.json" if standard else "*.diagnostic.json"))
            self.assertEqual(len(reports), 1)
            report = json.loads(reports[0].read_text())
            self.assertEqual(report["output_tokens"], tokens)
            self.assertEqual(report["repetitions"], repetitions)
            self.assertTrue(report["original_32_token_prefixes_match"])
            self.assertEqual(report["original_output_prefixes_match"], not corrupt_after_prefix)
            self.assertFalse(report["headline_eligible"])
            self.assertEqual(report["full_validation"], not standard)

    def test_existing_short_protocol(self):
        self.run_screen(extended=False, tokens=32, repetitions=1)

    def test_extended_full_length_protocol(self):
        self.run_screen(extended=True, tokens=256, repetitions=3)

    def test_matching_first32_does_not_hide_later_divergence(self):
        self.run_screen(extended=True, tokens=256, repetitions=3, corrupt_after_prefix=True)

    def test_retained_validation_protocol_after_full_proof(self):
        self.run_screen(extended=True, tokens=256, repetitions=3, standard=True)

    def test_combined_stack_chunk128_full_protocol(self):
        self.run_screen(extended=True, tokens=256, repetitions=3, dense=True, chunk=128)

    def test_combined_stack_chunk128_retained_validation(self):
        self.run_screen(extended=True, tokens=256, repetitions=3, dense=True, chunk=128, standard=True)

    def test_combined_stack_chunk128_rejects_late_divergence(self):
        self.run_screen(extended=True, tokens=256, repetitions=3, dense=True, chunk=128,
                        corrupt_after_prefix=True)

    def test_retained_validation_rejects_failed_32k_prompt_proof(self):
        self.run_screen(extended=True, tokens=256, repetitions=3, dense=True, chunk=128,
                        standard=True, invalid_full_proof=True)


if __name__ == "__main__":
    unittest.main()
