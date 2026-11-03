from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from build_dense_qkv_threads64 import FLAG, PROGRAM, transform
import run_hc_up_parallel2_model as model_runner


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2/matmul_nbits.cc")


class DenseQkvThreads64Tests(unittest.TestCase):
    def test_original_function_is_unchanged_and_private_geometry_is_scoped(self):
        original = SOURCE.read_text()
        actual = transform(original)
        start = original.rindex("Status ApplyDenseBlock32MatMulNBits(")
        end = original.index("\n}\n\n}  // namespace webgpu", start)
        self.assertIn(original[start:end], actual)
        start = actual.rindex("Status ApplyDenseQkvThreads64MatMulNBits(")
        end = actual.index("\n}\n\n}  // namespace webgpu", start)
        private = actual[start:end]
        self.assertIn("constexpr uint32_t workgroup_size = 64;", private)
        self.assertIn("constexpr uint32_t tile_size = 4;", private)
        self.assertIn('? 16u : 32u;', private)
        self.assertIn(f'Program{{"{PROGRAM}"}}', actual)
        self.assertIn(f"{FLAG} must be 0 or 1", actual)
        self.assertIn("K_ == 2560 && N_ == 16384", actual)
        self.assertIn("maxComputeWorkgroupStorageSize >= 2304", actual)
        self.assertIn("maxComputeWorkgroupsPerDimension >= 4096", actual)
        self.assertIn("qkv_threads64_route ? 21", actual)
        self.assertIn("!context.EnableMatmulFp32Accumulation()", actual)

    def test_two_columns_per_thread_cover_original_four_column_slots(self):
        slots = []
        outputs = []
        for thread in range(64):
            column_group, k_lane = divmod(thread, 32)
            for row_offset in (0, 2):
                slots.append((row_offset + column_group, k_lane))
                if k_lane == 0:
                    outputs.append(row_offset + column_group)
        self.assertEqual(sorted(outputs), list(range(4)))
        self.assertEqual(len(slots), 128)
        self.assertEqual(len(set(slots)), 128)
        self.assertEqual(sorted(block for lane in range(32) for block in range(lane, 80, 32)),
                         list(range(80)))

    def test_native_and_fp32_proofs_admit_profile_without_running_gpu(self):
        evidence = model_runner.ROOT
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(model_runner, "ROOT", Path(temporary)), \
                patch.object(model_runner, "prepare_model_cache",
                          side_effect=RuntimeError("preflight reached")), \
                patch("sys.argv", ["runner", "--phase", "profile", "--full-validation"]):
            names = [
                "dense-qkv-threads64-v1-native-gpu7-20261031.qualification.json",
                "dense-qkv-threads64-v1-fp32-native-gpu7-20261031.qualification.json",
                "qmoe-prefill-rows4-v2-contexts-20261031.qualification.json",
                "qmoe-prefill-rows4-v2-standard-contexts-20261031.qualification.json",
                "qsa-prefill-score16-v1-chunk1024-gpu7-full-standard2048-20261031.guard.json"]
            for name in names:
                (Path(temporary) / name).write_bytes((evidence / name).read_bytes())
                if name.startswith("dense-qkv-threads64"):
                    for source in json.loads((evidence / name).read_text())["source_sha256"]:
                        if source != "build-manifest.json":
                            (Path(temporary) / source).write_bytes((evidence / source).read_bytes())
            with self.assertRaisesRegex(RuntimeError, "preflight reached"):
                model_runner.main(
                    variant="dense-qkv-threads64-v1", experiment_flag=FLAG, stamp="20261031",
                    parent_variant="qmoe-prefill-rows4-v2", parent_stamp="20261031", chunk=1024,
                    parent_standard_prefix="qsa-prefill-score16-v1-chunk1024-gpu7-full-standard2048-20261031",
                    parent_provider="/home/sayanshaw/repos/qwen38-int4-qsa-prefill-score16/build-v1/"
                                    "libonnxruntime_providers_webgpu.so",
                    program_name=PROGRAM, original_program_name="MatMulNBitsBlock32F16",
                    dispatches=36, original_dispatches=123)
        self.assertIn("dense-qkv-threads64-v1", model_runner.ROW4_DECODE_VARIANTS)
        self.assertIn("dense-qkv-threads64-v1", model_runner.ROW4_REPEAT_VARIANTS)


if __name__ == "__main__":
    unittest.main()
