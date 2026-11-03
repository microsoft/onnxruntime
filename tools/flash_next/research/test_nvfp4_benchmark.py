import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("benchmark", root.parent / "benchmark_webgpu_fp8_checkpoint.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class Config:
    def __init__(self, _):
        self.overlay_value = None

    def overlay(self, value):
        self.overlay_value = json.loads(value)


class Engine:
    def __init__(self, _):
        self.pending = False
        self.forwards = 0

    def get_capabilities(self):
        return types.SimpleNamespace(max_request_length=50000)

    def create_request(self):
        def begin_turn(*_):
            self.count = 0
            self.pending = True
        return types.SimpleNamespace(begin_turn=begin_turn, close=lambda: None)

    def create_event_buffer(self, _):
        return None

    def has_pending_requests(self):
        return self.pending

    def run(self, _):
        self.count += 1
        self.forwards += 1
        self.pending = self.count < 2
        return [types.SimpleNamespace(flags=0, error_code=0, token=self.count)]

    def get_speculative_stats(self):
        keys = ("completed_rounds", "mtp_failures", "standard_fallback_steps", "draft_tokens_proposed",
                "draft_tokens_evaluated", "draft_tokens_accepted", "draft_forward_passes", "target_forward_passes")
        return {key: self.forwards if key == "target_forward_passes" else 0 for key in keys}


class BenchmarkTest(unittest.TestCase):
    def invoke(self, width, pool_case=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "genai_config.json").write_text(json.dumps({
                "model": {"decoder": {"session_options": {}}, "mtp": {"enabled": False}},
                "engine": {"dynamic_batching": {"block_size": 256, "num_blocks": 160}},
            }))
            plugin = path / "provider.so"
            plugin.write_bytes(b"mock-provider")
            output = path / "result.json"
            module = types.ModuleType("onnxruntime_genai")
            module.Config = Config
            module.Model = lambda config: config
            module.Engine = Engine
            module.Tokenizer = lambda _: types.SimpleNamespace(encode=lambda _: list(range(45)), decode=lambda _: "test")
            module.TurnOptions = lambda _: types.SimpleNamespace(
                set_do_sample=lambda _: None, set_min_generated_tokens=lambda _: None,
                set_max_generated_tokens=lambda _: None,
            )
            module.EngineEventFlags = types.SimpleNamespace(FAILED=1, CAPACITY_BLOCKED=2)
            module.set_log_options = lambda **_: None
            module.register_execution_provider_library = lambda *_: None
            argv = ["benchmark", "--model", str(path), "--plugin", str(plugin), "--output", str(output),
                    "--width", str(width), "--expert-quantization", "nvfp4", "--tokens", "2", "--repetitions", "1"]
            if pool_case is not None:
                text = path / "pool.txt"
                text.write_text("frozen natural text")
                tokenizer = path / "tokenizer.json"
                tokenizer.write_text("frozen tokenizer")
                manifest = {
                    "pool_sha256": hashlib.sha256(text.read_bytes()).hexdigest(),
                    "tokenizer_sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
                    "tokens_sha256": benchmark.token_hash(list(range(45))),
                    "pool_tokens": 45,
                }
                field = {
                    "text": "pool_sha256", "tokenizer": "tokenizer_sha256",
                    "tokens": "tokens_sha256", "length": "pool_tokens",
                }.get(pool_case)
                if field is not None:
                    manifest[field] = 44 if field == "pool_tokens" else "incorrect"
                manifest_path = path / "manifest.json"
                manifest_path.write_text(json.dumps(manifest))
                argv.extend(["--context-tokens", "32", "--text-file", str(text),
                             "--prompt-pool-manifest", str(manifest_path)])
            with patch.dict(sys.modules, onnxruntime_genai=module), patch.object(sys, "argv", argv):
                with contextlib.redirect_stdout(io.StringIO()):
                    if width:
                        with self.assertRaisesRegex(RuntimeError, "did not complete any speculative round"):
                            benchmark.main()
                    elif pool_case not in (None, "valid"):
                        with self.assertRaises(ValueError):
                            benchmark.main()
                    else:
                        benchmark.main()
            return json.loads(output.read_text())

    def test_decoder_only_does_not_execute_mtp(self):
        result = self.invoke(0)
        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["config_overlay"]["model"]["mtp"]["enabled"])
        self.assertFalse(result["quality_qualified"])
        self.assertEqual(result["expert_quantization"], "nvfp4")
        self.assertIn("E2M1", result["limitations"][2])

    def test_mtp_reenabled_and_silent_decoder_fallback_rejected(self):
        result = self.invoke(2)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["config_overlay"]["model"]["mtp"]["enabled"])

    def test_frozen_pool_matches(self):
        result = self.invoke(0, "valid")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["text_pool_tokens"], 45)
        self.assertEqual(result["text_tokens_sha256"], result["prompt_pool_manifest"]["tokens_sha256"])

    def test_frozen_pool_mismatches_fail_explicitly(self):
        for case in ("text", "tokenizer", "tokens", "length"):
            with self.subTest(case=case):
                result = self.invoke(0, case)
                self.assertEqual(result["status"], "failed")


if __name__ == "__main__":
    unittest.main()
