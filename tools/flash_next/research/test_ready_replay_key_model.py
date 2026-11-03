import ast
from pathlib import Path
import unittest

from prepare_ready_replay_key_model import model_source, screen_source


ROOT = Path(__file__).parent


class ReadyReplayKeyModelTests(unittest.TestCase):
    def test_model_uses_original_control_and_full_process_mode_proof(self):
        source = model_source((ROOT / "run_dense_replay_scratch_model.py").read_text())
        ast.parse(source)
        self.assertIn('if args.mode == "control":\n        native = original_native', source)
        self.assertIn('native["finite_bitwise_comparisons"] == 233165568', source)
        self.assertIn('native["process_modes_tested"] == [0, 1]', source)
        self.assertIn('native["invalid_selector_rejected_in_fresh_process"]', source)
        self.assertIn('qwen38-int4-ready-replay-key-trim/build-v2', source)

    def test_coverage_reads_correct_static_and_actual_key_stats(self):
        source = model_source((ROOT / "run_dense_replay_scratch_model.py").read_text())
        self.assertIn('prefix.name + "-static-stats.jsonl"', source)
        self.assertIn('prefix.name + "-key-stats.jsonl"', source)
        self.assertIn('row.get("operator") == "MatMulNBitsExact"', source)
        self.assertIn('len(dense) == 171', source)
        self.assertIn('key_coverage["commands"] >= result["dense_replay_hits"]', source)
        self.assertIn('assert not key_path.exists()', source)

    def test_screen_is_original_provider_net_and_retains_escalation_gates(self):
        source = screen_source((ROOT / "run_dense_replay_scratch_screen.py").read_text())
        ast.parse(source)
        self.assertIn('providers["control"] != providers["trim"]', source)
        self.assertIn('0955e1d6cc30cac13044450f692100ffa5f33361aaef229e25ff0efa20a8c8d4', source)
        self.assertIn("bracketed_separate_load_original_provider_net_screen", source)
        self.assertIn('initial["status"] == "initial_trim_decode_gain_both_modes"', source)
        self.assertIn('len(measured) == 3', source)


if __name__ == "__main__":
    unittest.main()
