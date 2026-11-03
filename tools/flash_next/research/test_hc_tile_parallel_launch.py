import unittest

from prepare_hc_tile_parallel import ROOT, build_source, controller_source


class TileParallelLaunchTests(unittest.TestCase):
    def test_build_metadata_and_actual_shader_transform_are_not_held_layout(self):
        source = build_source((ROOT / "build_hc_owned_pair34.py").read_text())
        self.assertIn("from hc_tile_parallel_transform import transform", source)
        self.assertIn('"physical_threads": 1024', source)
        self.assertIn('"shared_storage_bytes": 37888', source)
        self.assertIn('"independent_tile_workers": 4', source)
        self.assertNotIn('"gates_computed_once_per_workgroup"', source)
        self.assertIn('qwen38-int4-fc2-private-a/build-v1', source)

    def test_gpu_filter_demands_selected_1024_and_original_256_threads(self):
        source = controller_source((ROOT / "run_hc_owned_pair34_v1.py").read_text())
        namespace = {"__name__": "not_main", "__file__": str(ROOT / "run_hc_tile_parallel4_v1.py")}
        exec(compile(source, namespace["__file__"], "exec"), namespace)
        measured_gpu = namespace["measured_gpu"]
        for selected, threads in ((False, 256), (True, 1024)):
            name = namespace["NAME"] if selected else "HCPostNormProjectionF16"
            events = [{
                "cat": "Kernel", "name": "node&" + name, "pid": -1, "tid": -1,
                "ph": "X", "dur": 20, "args": {"cache_key": f":{threads},1,1:"},
            } for _ in range(144)]
            self.assertEqual(len(measured_gpu(events, selected)), 128)
            events[0]["args"]["cache_key"] = f":{256 if selected else 1024},1,1:"
            with self.assertRaises(RuntimeError):
                measured_gpu(events, selected)


if __name__ == "__main__":
    unittest.main()
