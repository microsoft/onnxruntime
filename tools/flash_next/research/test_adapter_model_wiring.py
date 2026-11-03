import unittest

from prepare_adapter_tensor_reuse_model import ROOT, base, model_source


class AdapterModelWiringTests(unittest.TestCase):
    def test_all_object_native_and_empty_cache_proofs_are_required(self):
        source = model_source((ROOT / "run_ready_replay_key_model.py").read_text())
        self.assertIn('native["finite_bitwise_comparisons"] == 237408768', source)
        self.assertIn('build["all_152_original_source_objects_byte_identical"]', source)
        self.assertIn('build["complete_sdk_copy_for_relative_include_wiring"]', source)
        self.assertIn('native["original_expanded_gdn_reference_tested"]', source)
        self.assertIn('row["retained_input_elements"] == row["retained_output_elements"] == 0', source)
        self.assertIn('row["requests"] == row["returns"]', source)
        self.assertIn('if args.mode == "control":', source)
        self.assertIn("qwen38-int4-qmoe-prefill-rows4/build-v2/", source)
        self.assertNotIn("ReadyReplayWithoutKey", source)
        self.assertNotIn("key_coverage", source)

    def test_original_provider_brackets_and_long_context_gates_remain(self):
        source = base((ROOT / "run_ready_replay_key_screen.py").read_text())
        self.assertIn('providers["control"] != providers["reuse"]', source)
        self.assertIn("0955e1d6cc30cac13044450f692100ffa5f33361aaef229e25ff0efa20a8c8d4", source)
        self.assertIn('supported = all(row["comparison"]["supported"] for row in rows)', source)
        model = model_source((ROOT / "run_ready_replay_key_model.py").read_text())
        self.assertIn("previous = 2048 if args.context == 8192 else 8192", model)
        self.assertIn("context{previous}-repeat-screen", model)
        self.assertIn("initial_reuse_decode_gain_both_modes", model)


if __name__ == "__main__":
    unittest.main()
