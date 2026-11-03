import unittest

from prepare_dense_tile_parallel2 import ROOT, build_source, controller_source


class DenseTwoWorkerLaunchTests(unittest.TestCase):
    def test_geometry_provider_paths_and_strict_gpu_thresholds_are_wired_together(self):
        build = build_source((ROOT / "build_dense_tile_parallel3.py").read_text())
        controller = controller_source((ROOT / "run_dense_tile_parallel3_v1.py").read_text())
        compile(build, "two-worker-builder", "exec")
        scope = {"__name__": "two_worker_controller_test"}
        exec(compile(controller, "two-worker-controller", "exec"), scope)
        self.assertIn('"physical_threads": 256, "independent_tile_workers": 2', build)
        self.assertIn('ROOT / "dense_tile_parallel3_transform.py"', build)
        self.assertIn("qwen38-int4-dense-tile-parallel2/build-v1", build)
        event = {"cat": "Kernel", "name": "node&" + scope["NAME"], "pid": -1, "tid": -1,
                 "ph": "X", "dur": 10, "args": {"cache_key": "program:256,1,1:shape"}}
        self.assertEqual(scope["measured_gpu"]([event] * 144, True), [10] * 128)
        with self.assertRaises(RuntimeError):
            scope["measured_gpu"]([dict(event, args={"cache_key": "program:384,1,1:shape"})] * 144, True)
        match = scope["matched"]
        self.assertTrue(match([10.2] * 128, [10] * 128, [10.2] * 128)["operator_gate_pass"])
        self.assertFalse(match([10] * 128, [10] * 128, [10] * 128)["operator_gate_pass"])
        self.assertFalse(match([10.2] * 128, [10] * 119 + [11] * 9, [10.2] * 128)["operator_gate_pass"])
        shell = scope["command_for"]("capture", scope["BUILD"] / "libonnxruntime_providers_webgpu.so",
                                     scope["FIXTURES"] / "k2560-n7168.onnx",
                                     scope["DATA"] / "unused.bin", ROOT / "unused-two-profile", 2)[-1]
        self.assertIn("ORT_WEBGPU_DENSE_TILE_PARALLEL2=2", shell)
        self.assertIn("dense_compare capture /workspace/qwen38-int4-dense-tile-parallel2", shell)


if __name__ == "__main__":
    unittest.main()
