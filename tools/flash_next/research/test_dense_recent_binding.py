import unittest
from pathlib import Path

from build_dense_recent_binding import FLAG, transform
from build_dense_recent_bindings_proof import transform as bank_transform
from prepare_dense_recent_model import ROOT, model_source, transform as prepare_transform


class RecentBindingTests(unittest.TestCase):
    def source(self):
        return transform(Path("/home/sayanshaw/repos/qwen38-int4-dense-bitcast-unpack/"
                              "build-v1/overlay/static_node_replay.h").read_text())

    def test_default_validation_and_guard(self):
        source = self.source()
        self.assertIn(f'"{FLAG}", 0', source)
        self.assertIn("recent == 0 || recent == 1", source)
        self.assertIn("recent_binding_ < binding_entries_.size()", source)
        self.assertIn("previous.prototype_id == entry.id", source)
        self.assertIn("previous.buffers[position] == entry.bindings.uniform_buffer", source)
        fast = source[source.index("if (recent == 1"):]
        self.assertLess(fast.index("CheckBufferTensor(tensor"), fast.index("tensor->DataRaw()"))

    def test_fallback_and_eviction(self):
        source = self.source()
        self.assertIn("recent_binding_ = binding_entries_.size() - 1;", source)
        self.assertIn("bound - binding_entries_.begin()", source)
        self.assertIn("std::find_if(binding_entries_.begin()", source)
        self.assertIn("static constexpr size_t kCapacity = 64;", source)
        self.assertIn("context.PrepareNodeBinding(", source)
        self.assertNotIn("scratch_key", source)
        self.assertIn('"recent_binding_hits', source.replace("\\", ""))

    def test_candidate_uses_different_live_bank(self):
        source = bank_transform(Path("/home/sayanshaw/repos/qwen38-int4-dense-recent-binding/"
                                     "build-v1/native_compare.cpp").read_text())
        self.assertIn("candidate_bank = gpu_inputs[(round + 1)", source)
        self.assertIn("candidate_copy = env.CopyTensors(inputs, candidate_bank", source)
        self.assertIn("auto actual = session.Run(Ort::RunOptions{}, names.data(), candidate_bank.data()", source)

    def test_isolated_model_family_and_proofs(self):
        model = model_source((ROOT / "run_dense_replay_scratch_model.py").read_text())
        screen = prepare_transform((ROOT / "run_dense_replay_scratch_screen.py").read_text())
        self.assertNotIn("scratch", model.lower())
        self.assertNotIn("scratch", screen.lower())
        self.assertIn("from build_dense_recent_binding import FLAG", model)
        self.assertIn("dense_recent_binding_fast_path_changing_inputs_finite_bitwise_qualified", model)
        self.assertIn("dense_recent_binding_distinct_bank_miss_path_finite_bitwise_qualified", model)
        self.assertIn('"MatMulNBitsRecentExact"', model)
        self.assertIn('prefix.name + "-static-stats.jsonl"', model)
        self.assertIn("recent_binding_fraction=fast_hits / binding_hits", model)
        self.assertIn("from run_dense_recent_binding_model import", screen)


if __name__ == "__main__":
    unittest.main()
