import random
import unittest
from pathlib import Path

from prepare_dense_columns8_double_buffer import builder_source
from test_hc_projection_inner_unroll import render


class Columns8DoubleBufferTests(unittest.TestCase):
    def test_rendered_shader_changes_only_staging_and_barriers(self):
        namespace = {"__name__": "double-buffer-test", "__file__": __file__}
        exec(compile(builder_source(), "builder", "exec"), namespace)
        candidate = render(namespace["PROGRAM"], 0, False, False, "", namespace["FLAG"],
                           class_name=namespace["CLASS"], constructor_args=())
        original_namespace = {"__name__": "original-test", "__file__": __file__}
        exec(compile(Path(__file__).with_name("build_dense_columns8.py").read_text(),
                     "original-builder", "exec"), original_namespace)
        reference = render(original_namespace["PROGRAM"], 0, False, False, "",
                           original_namespace["FLAG"],
                           class_name=original_namespace["CLASS"], constructor_args=())
        normalized = candidate.replace("tile_a: array<vec4<f16>, 512>",
                                       "tile_a: array<vec4<f16>, 256>")
        normalized = normalized.replace("(tile % 2u) * 256u + ", "")
        normalized = normalized.replace("}\nworkgroupBarrier();\nif (local_idx < 8u)",
                                        "  workgroupBarrier();\n}\nif (local_idx < 8u)")
        self.assertEqual(normalized, reference)
        self.assertEqual(candidate.count("workgroupBarrier();"), 2)
        self.assertEqual(namespace["STORAGE_BYTES"], 512 * 8 + 256 * 2)
        source = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/"
                      "build-v2/matmul_nbits.cc").read_text()
        transformed = namespace["transform"](source)
        self.assertIn("columns8 ? 7", transformed)
        self.assertIn("N_ >= 2560 && N_ % 8 == 0", transformed)
        self.assertIn("maxComputeWorkgroupStorageSize >= 4608", transformed)

    def test_randomized_warp_scheduling_preserves_buffers(self):
        for tiles in (1, 2, 3, 6, 10, 32):
            for seed in range(100):
                rng = random.Random(seed)
                buffers = [[None] * 4 for _ in range(2)]
                states = [(0, "load")] * 4
                finished = [False] * 4
                while not all(finished):
                    runnable = [warp for warp, (_, stage) in enumerate(states)
                                if stage != "barrier" and not finished[warp]]
                    if not runnable:
                        self.assertEqual(len({tile for tile, _ in states}), 1)
                        states = [(tile, "read") for tile, _ in states]
                        continue
                    warp = rng.choice(runnable)
                    tile, stage = states[warp]
                    if stage == "load":
                        buffers[tile % 2][warp] = tile
                        states[warp] = (tile, "barrier")
                    else:
                        self.assertEqual(buffers[tile % 2], [tile] * 4)
                        if tile + 1 == tiles:
                            finished[warp] = True
                            states[warp] = (tile, "barrier")
                        else:
                            states[warp] = (tile + 1, "load")


if __name__ == "__main__":
    unittest.main()
