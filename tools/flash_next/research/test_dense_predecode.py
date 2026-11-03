from pathlib import Path
import json
import re
import tempfile
import unittest
from unittest.mock import patch

from build_dense_predecode import FLAG, predecoded_template, transform
from build_hc_staged_projection import stage_replay
from run_dense_predecode_native import verify_predecoded_stats
import run_hc_up_parallel2_model as model_runner


BASE = Path("/home/sayanshaw/repos")
COMPONENT = BASE / "qwen38-int4-dense-block32/build-v1"
HEADER = BASE / "qwen38-int4-static-node-replay/build-v8/overlay/contrib_ops/webgpu/quantization/matmul_nbits.h"
TEMPLATE = BASE / "qwen38-int4-dense-block32/block32-template-v1/dense_block32_template.inc"


class DensePredecodeTests(unittest.TestCase):
    def test_template_changes_weight_loads_not_arithmetic_or_reductions(self):
        original = TEMPLATE.read_text()
        actual = predecoded_template(original)
        arithmetic = lambda text: [
            line for line in text.splitlines()
            if not line.lstrip().startswith("//") and
            any(token in line for token in ("sum += dot(", "inter_results[", "output_value +=",
                                            "loadSHMA(", "a_offset +=", "workgroupBarrier()"))]
        self.assertEqual(arithmetic(original), arithmetic(actual))
        self.assertNotIn("let b_value_lower =", actual)
        self.assertNotIn("let b_value_upper =", actual)
        self.assertEqual(actual.count('* 8u + i * 2u'), 2)
        self.assertIn("alias acc_element_t = output_element_t", actual)
        self.assertEqual(len(re.findall(r'__var_b->GetByOffset\("\(b_global', actual)), 4)

    def test_original_kernel_body_and_class_layout_are_not_modified(self):
        original, header = (COMPONENT / "matmul_nbits.cc").read_text(), HEADER.read_text()
        actual = transform(original, header)
        signature = "Status MatMulNBits::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {"
        start = original.index(signature)
        stop = original.index("\n}\n", start) + 3
        self.assertIn(original[start:stop], actual)
        self.assertIn("class PersistentDenseMatMulNBits final", actual)
        self.assertIn("    PersistentDenseMatMulNBits);", actual)
        self.assertEqual(header, HEADER.read_text())

    def test_constant_only_admission_and_owned_extra_binding_order(self):
        actual = transform((COMPONENT / "matmul_nbits.cc").read_text(), HEADER.read_text())
        self.assertIn("info.TryGetConstantInput(1, &constant) && info.TryGetConstantInput(2, &constant)", actual)
        self.assertIn("predecode_mode == 1 && dense_bitcast_route && block32_mode == 1 && weights_constant_", actual)
        self.assertIn(f"{FLAG} must be 0 or 1", actual)
        self.assertIn("mutable std::unique_ptr<Tensor> decoded_weights_", actual)
        self.assertIn('decoded_replay_{"DensePredecoded", {0, 2}}', actual)
        self.assertIn("}, 0, 0, {decoded_weights_.get()});", actual)
        replay = stage_replay((BASE / "qwen38-int4-dense-bitcast-unpack/build-v1/overlay/static_node_replay.h").read_text())
        self.assertLess(replay.index("for (const auto* tensor : extra_inputs)"),
                        replay.index("for (int index : gpu_inputs_)"))
        self.assertTrue("return context.RunProgram(program);" in actual)
        self.assertIn("ORT_THROW_IF_ERROR(context.RunProgram(program))", actual)

    def test_private_and_original_replay_statistics_require_actual_reuse(self):
        rows = [dict(operator=name, requests=128, rebinds=1, binding_hits=126, hits=127,
                     captures=1, entries=1, capacity=64, binding_entries=1, binding_capacity=64)
                for name in ("MatMulNBitsExact", "DensePredecoded") for _ in range(9)]
        verify_predecoded_stats(rows)
        with self.assertRaises(AssertionError):
            verify_predecoded_stats(rows[:-1])
        rows[0]["binding_hits"] = 0
        with self.assertRaises(AssertionError):
            verify_predecoded_stats(rows)

    def test_actual_native_and_fp32_proofs_pass_model_preflight_without_running_gpu(self):
        evidence = Path(__file__).parent
        baseline = "qsa-prefill-score16-v1-chunk1024-gpu7-full-standard2048-20261031"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("dense-predecode-v2-native-gpu7-20261031.qualification.json",
                         "dense-predecode-v2-fp32-native-gpu7-20261031.qualification.json",
                         "qsa-prefill-score16-v1-contexts-20261031.qualification.json",
                         baseline + ".guard.json"):
                (root / name).write_bytes((evidence / name).read_bytes())
                if name.startswith("dense-predecode"):
                    proof = json.loads((root / name).read_text())
                    for source in proof["source_sha256"]:
                        if source != "build-manifest.json":
                            (root / source).write_bytes((evidence / source).read_bytes())
            with patch.object(model_runner, "ROOT", root), \
                    patch.object(model_runner, "prepare_model_cache", side_effect=RuntimeError("preflight reached")), \
                    patch("sys.argv", ["runner", "--phase", "profile", "--full-validation"]):
                with self.assertRaisesRegex(RuntimeError, "preflight reached"):
                    model_runner.main(
                        variant="dense-predecode-v2", experiment_flag=FLAG, stamp="20261031",
                        parent_variant="qsa-prefill-score16-v1", parent_stamp="20261031", chunk=1024,
                        parent_standard_prefix=baseline,
                        parent_provider="/home/sayanshaw/repos/qwen38-int4-qsa-prefill-score16/build-v1/"
                                        "libonnxruntime_providers_webgpu.so",
                        program_name="MatMulNBitsPredecodedF16",
                        original_program_name="MatMulNBitsBlock32F16", dispatches=159)


if __name__ == "__main__":
    unittest.main()
