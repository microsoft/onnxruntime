import unittest

from prepare_adapter_inline_proofs import ROOT, base, model_source, native_source


class AdapterInlineWiringTests(unittest.TestCase):
    def test_native_requires_coherent_layout_and_separate_processes(self):
        source = native_source((ROOT / "run_adapter_tensor_reuse_native.py").read_text())
        self.assertIn('assert not build["no_object_layout_changes"]', source)
        self.assertIn("for mode in (0, 1):", source)
        self.assertIn('row["enabled_contexts"] == row["contexts"] * mode', source)
        self.assertIn('row["input_capacity"] == 8 and row["output_capacity"] == 5', source)
        self.assertIn("233165568 + 4243200", source)
        self.assertNotIn("AdapterTensorVectors", source)

    def test_statistics_are_absent_from_standard_commands(self):
        source = model_source((ROOT / "run_adapter_tensor_reuse_model.py").read_text())
        namespace = {}
        start = source.index("def model_command(")
        end = source.index("\ndef verify_coverage(", start)
        namespace["standard_command"] = lambda *args: ["runner", "--tokens 256 --repetitions 3"]
        namespace["FLAG"] = "ORT_WEBGPU_ADAPTER_INLINE_TENSORS"
        exec(source[start:end], namespace)
        for mode in ("control", "inline"):
            for phase in ("standard", "coverage"):
                command = namespace["model_command"](None, None, ROOT / "fixture", 2048, True, mode, phase)[-1]
                self.assertEqual("export ORT_WEBGPU_ADAPTER_INLINE_STATS_FILE=" in command,
                                 phase == "coverage" and mode == "inline")
                self.assertEqual("--tokens 36 --repetitions 1" in command, phase == "coverage")
        self.assertIn('native["complete_context_layout_rebuild"]', source)
        self.assertIn('if args.phase == "coverage" and args.mode == "inline":', source)
        self.assertIn("237408768", source)

    def test_original_provider_and_long_context_gates_are_preserved(self):
        source = base((ROOT / "run_adapter_tensor_reuse_screen.py").read_text())
        self.assertIn('providers["control"] != providers["inline"]', source)
        self.assertIn("0955e1d6cc30cac13044450f692100ffa5f33361aaef229e25ff0efa20a8c8d4", source)
        model = model_source((ROOT / "run_adapter_tensor_reuse_model.py").read_text())
        self.assertIn("previous = 2048 if args.context == 8192 else 8192", model)
        self.assertIn("context{previous}-repeat-screen", model)
        self.assertIn("initial_inline_decode_gain_both_modes", model)


if __name__ == "__main__":
    unittest.main()
