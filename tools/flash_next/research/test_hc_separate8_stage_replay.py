from pathlib import Path
import unittest

from build_hc_separate8_stage_replay import FLAG, REPLAY, SOURCE, stage_replay, transform
from build_hc_separate8 import programs


class HCSeparate8StageReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(str(SOURCE).replace("/workspace/", "/home/sayanshaw/repos/")).read_text()
        cls.candidate = transform(cls.source)

    def test_single_command_stage_boundaries(self):
        norm = self.candidate[self.candidate.index("norm_replay_.Run"):
                              self.candidate.index("projection_replay_.Run")]
        projection = self.candidate[self.candidate.index("projection_replay_.Run"):
                                    self.candidate.index("    if (fused) {")]
        self.assertEqual(norm.count("context.RunProgram("), 1)
        self.assertEqual(projection.count("context.RunProgram("), 1)
        self.assertIn("{y, normalized}", norm)
        self.assertIn("{activated, tail}", projection)
        self.assertIn("}, 0, 0, {normalized});", projection)
        self.assertIn("mutable StaticNodeReplay norm_replay_", self.candidate)
        self.assertIn("mutable HCSeparate8StageReplay projection_replay_", self.candidate)
        self.assertIn(f'"{FLAG}"', self.candidate)

    def test_original_fused_and_fallback_preserved(self):
        start = self.source.index("    if (fused) {\n")
        end = self.source.index("\n protected:", start)
        self.assertIn(self.source[start:end], self.candidate)
        self.assertEqual(self.candidate.count("mutable StaticNodeReplay replay_;"), 1)
        self.assertEqual(programs(self.source), programs(self.candidate))

    def test_projection_extra_input_key_and_binding_order(self):
        source = Path(str(REPLAY).replace("/workspace/", "/home/sayanshaw/repos/")).read_text()
        header = stage_replay(source)
        self.assertIn("key.push_back(extra_inputs.size())", header)
        self.assertIn("append(input, false)", header)
        start = header.index("auto bindings =")
        self.assertLess(header.index("for (const auto* tensor : extra_inputs)", start),
                        header.index("for (int index : gpu_inputs_)", start))
        self.assertIn("entry.commands.size() == 1", header)
        self.assertIn("entry.bindings.buffers == expected", header)
        self.assertIn("PrepareNodeBinding", header)


if __name__ == "__main__":
    unittest.main()
