import unittest
from pathlib import Path

from hc_tile_parallel2_transform import CLASS, FLAG, NAME, transform
from hc_tile_parallel_transform import transform as four_worker_transform
from prepare_hc_tile_parallel2 import ROOT, build_source, controller_source


ORIGINAL = Path(
    "/home/sayanshaw/repos/qwen38-int4-hc-post-norm-projection/build-v2/hc_post_norm_projection.cc")


class TileParallelTwoTests(unittest.TestCase):
    def test_only_thread_geometry_and_tile_schedule_differ_from_exact_four_worker_source(self):
        four = four_worker_transform(ORIGINAL.read_text())
        two = transform(ORIGINAL.read_text())
        normalized = two.replace("TILE_PARALLEL2", "TILE_PARALLEL4")
        normalized = normalized.replace("TileParallel2", "TileParallel4").replace("tile_parallel2", "tile_parallel4")
        normalized = normalized.replace("maxComputeWorkgroupSizeX >= 512", "maxComputeWorkgroupSizeX >= 1024")
        normalized = normalized.replace("maxComputeInvocationsPerWorkgroup >= 512",
                                        "maxComputeInvocationsPerWorkgroup >= 1024")
        normalized = normalized.replace("tile_parallel4 == 1 ? 512 : 256", "tile_parallel4 == 1 ? 1024 : 256")
        normalized = normalized.replace("tile < 10u; tile += 2u", "tile < 10u; tile += 4u")
        self.assertEqual(normalized, four)
        for token in (FLAG, CLASS, NAME, "maxComputeWorkgroupStorageSize >= 37888"):
            self.assertIn(token, two)

    def test_two_workers_own_all_full_words_once_with_even_five_tile_schedule(self):
        writers = {}
        for physical in range(512):
            logical, worker = physical % 256, physical // 256
            tiles = range(worker, 10, 2)
            self.assertEqual(len(tiles), 5)
            for tile in tiles:
                slot = tile * 256 + logical
                self.assertNotIn(slot, writers)
                writers[slot] = physical
        self.assertEqual(set(writers), set(range(2560)))

    def test_build_metadata_and_gpu_events_require_new_geometry(self):
        builder = build_source((ROOT / "build_hc_owned_pair34.py").read_text())
        self.assertIn("from hc_tile_parallel2_transform import transform", builder)
        self.assertIn('"physical_threads": 512', builder)
        self.assertIn('"independent_tile_workers": 2', builder)
        self.assertIn('"shared_storage_bytes": 37888', builder)
        source = controller_source((ROOT / "run_hc_owned_pair34_v1.py").read_text())
        namespace = {"__name__": "not_main", "__file__": str(ROOT / "run_hc_tile_parallel2_v1.py")}
        exec(compile(source, namespace["__file__"], "exec"), namespace)
        for selected, threads in ((False, 256), (True, 512)):
            name = namespace["NAME"] if selected else "HCPostNormProjectionF16"
            events = [{
                "cat": "Kernel", "name": "node&" + name, "pid": -1, "tid": -1,
                "ph": "X", "dur": 20, "args": {"cache_key": f":{threads},1,1:"},
            } for _ in range(144)]
            self.assertEqual(len(namespace["measured_gpu"](events, selected)), 128)
            events[0]["args"]["cache_key"] = ":1024,1,1:"
            with self.assertRaises(RuntimeError):
                namespace["measured_gpu"](events, selected)


if __name__ == "__main__":
    unittest.main()
