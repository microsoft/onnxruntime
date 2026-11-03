from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from build_dense_qkv_predecode8 import FLAG, PROGRAM, STATS, transform
from build_dense_predecode import predecoded_template
from run_dense_qkv_predecode8_native import verify_stats
import run_hc_up_parallel2_model as model_runner


REPOS = Path("/home/sayanshaw/repos")
SOURCE = REPOS / "qwen38-int4-qmoe-prefill-rows4/build-v2/matmul_nbits.cc"
HEADER = REPOS / "qwen38-int4-static-node-replay/build-v8/overlay/contrib_ops/webgpu/quantization/matmul_nbits.h"
TEMPLATE = REPOS / "qwen38-int4-dense-block32/block32-template-v1/dense_block32_template.inc"


class DenseQkvPredecode8Tests(unittest.TestCase):
    def test_original_method_is_unchanged_and_private_route_is_guarded(self):
        original = SOURCE.read_text()
        actual = transform(original, HEADER.read_text())
        signature = "Status MatMulNBits::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {"
        begin = original.index(signature)
        end = original.index("\n}\n", begin) + 3
        self.assertIn(original[begin:end], actual)
        self.assertIn(f"{FLAG} must be 0 or 1", actual)
        self.assertNotIn("ORT_WEBGPU_DENSE_PREDECODE", actual)
        for guard in (
                "K_ == 2560 && N_ == 16384", 'std::string_view{"nvidia"}',
                "maxComputeWorkgroupSizeX >= 256", "maxComputeInvocationsPerWorkgroup >= 256",
                "maxComputeWorkgroupStorageSize >= 2560", "maxComputeWorkgroupsPerDimension >= 2048",
                "weights_constant_", "!context.EnableMatmulFp32Accumulation()"):
            self.assertIn(guard, actual)

    def test_eight_columns_retain_32_lanes_and_owned_constant_cache(self):
        actual = transform(SOURCE.read_text(), HEADER.read_text())
        self.assertIn(f'Program{{"{PROGRAM}"}}', actual)
        self.assertIn(f'decoded_replay_{{"{STATS}", {{0, 2}}}}', actual)
        self.assertIn("program{8, 4, false, false, false, false, false, 32, false, false}", actual)
        self.assertIn("SetWorkgroupSize(256).SetDispatchGroupSize((N + 7u) / 8u)", actual)
        self.assertIn("K / 32u, (N + 7u) / 8u, 1u, 0u, 1u", actual)
        self.assertIn("info.TryGetConstantInput(1, &constant) && info.TryGetConstantInput(2, &constant)", actual)
        self.assertIn("std::call_once(decode_once_", actual)
        self.assertIn("ORT_THROW_IF_ERROR(context.RunProgram(program))", actual)
        self.assertIn("}, 0, 0, {decoded_weights_.get()});", actual)

    def test_weight_load_transform_preserves_exact_arithmetic(self):
        original = TEMPLATE.read_text()
        actual = predecoded_template(original)
        def arithmetic(text):
            return [line for line in text.splitlines()
                    if not line.lstrip().startswith("//") and
                    any(token in line for token in (
                        "sum += dot(", "inter_results[", "output_value +=",
                        "loadSHMA(", "a_offset +=", "workgroupBarrier()"))]
        self.assertEqual(arithmetic(original), arithmetic(actual))

    def test_columns_and_partial_slots_are_covered_once(self):
        outputs = []
        slots = []
        for lane in range(256):
            column, k_lane = divmod(lane, 32)
            slots.append((column, k_lane))
            if k_lane == 0:
                outputs.append(column)
        self.assertEqual(outputs, list(range(8)))
        self.assertEqual(len(set(slots)), 8 * 32)
        blocks = [block for lane in range(32) for block in range(lane, 80, 32)]
        self.assertEqual(sorted(blocks), list(range(80)))

    def test_statistics_require_one_target_and_eight_fallback_shapes(self):
        def row(operator, requests):
            captures = 1
            return dict(operator=operator, requests=requests, captures=captures,
                        hits=requests-captures, rebinds=captures,
                        binding_hits=requests-2*captures, entries=captures,
                        capacity=64, binding_entries=captures, binding_capacity=64)
        rows = [row("MatMulNBitsExact", 256) for _ in range(8)]
        rows += [row("MatMulNBitsExact", 128), row(STATS, 128)]
        verify_stats(rows)
        with self.assertRaises(AssertionError):
            verify_stats(rows[:-1])
        rows[0] = row("MatMulNBitsExact", 128)
        with self.assertRaises(AssertionError):
            verify_stats(rows)

    def test_native_and_fp32_proofs_admit_profile_without_running_gpu(self):
        evidence = model_runner.ROOT
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(model_runner, "ROOT", Path(temporary)), \
                patch.object(model_runner, "prepare_model_cache",
                          side_effect=RuntimeError("preflight reached")), \
                patch("sys.argv", ["runner", "--phase", "profile", "--full-validation"]):
            names = [
                "dense-qkv-predecode8-v1-native-gpu7-20261031.qualification.json",
                "dense-qkv-predecode8-v1-fp32-native-gpu7-20261031.qualification.json",
                "qmoe-prefill-rows4-v2-contexts-20261031.qualification.json",
                "qmoe-prefill-rows4-v2-standard-contexts-20261031.qualification.json",
                "qsa-prefill-score16-v1-chunk1024-gpu7-full-standard2048-20261031.guard.json"]
            for name in names:
                (Path(temporary) / name).write_bytes((evidence / name).read_bytes())
                if name.startswith("dense-qkv-predecode8"):
                    for source in json.loads((evidence / name).read_text())["source_sha256"]:
                        if source != "build-manifest.json":
                            (Path(temporary) / source).write_bytes((evidence / source).read_bytes())
            with self.assertRaisesRegex(RuntimeError, "preflight reached"):
                model_runner.main(
                    variant="dense-qkv-predecode8-v1", experiment_flag=FLAG, stamp="20261031",
                    parent_variant="qmoe-prefill-rows4-v2", parent_stamp="20261031", chunk=1024,
                    parent_standard_prefix="qsa-prefill-score16-v1-chunk1024-gpu7-full-standard2048-20261031",
                    parent_provider=str(REPOS / "qwen38-int4-qsa-prefill-score16/build-v1/"
                                        "libonnxruntime_providers_webgpu.so"),
                    program_name=PROGRAM, original_program_name="MatMulNBitsBlock32F16",
                    dispatches=36, original_dispatches=123)
        self.assertIn("dense-qkv-predecode8-v1", model_runner.ROW4_DECODE_VARIANTS)
        self.assertIn("dense-qkv-predecode8-v1", model_runner.ROW4_REPEAT_VARIANTS)


if __name__ == "__main__":
    unittest.main()
