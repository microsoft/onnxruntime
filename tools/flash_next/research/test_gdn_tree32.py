import inspect
import json
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from build_gdn_tree32 import FLAG, PRIVATE_TEMPLATE, PROGRAM, transform, transform_compare, transform_template
from build_gdn_value2_capture import transform as capture_transform
from run_gdn_value2_native import main as native_main
from qsa_rank_decode_windows import GDN_TREE32_FLAG
import run_hc_up_parallel2_model as model_runner


BASE = Path("/home/sayanshaw/repos")
SOURCE = BASE / "qwen38-int4-gdn-decode-view/build-original/gated_delta_net.cc"
HEADER = BASE / "onnxruntime-webgpu-h200-perf/onnxruntime/contrib_ops/webgpu/bert/gated_delta_net.h"
TEMPLATE = HEADER.with_name("gated_delta_net.wgsl.template")


def reduce_tree(values, add, hybrid):
    values = list(values)
    for stride in (64, 32):
        for index in range(stride):
            values[index] = add(values[index], values[index + stride])
    if hybrid:
        values = values[:32]
    for stride in (16, 8, 4, 2, 1):
        snapshot = values[:]
        for index in range(stride):
            values[index] = add(snapshot[index], snapshot[index + stride])
    return values[0]


class GdnTree32Tests(unittest.TestCase):
    def test_fresh_norm_only_native_metadata_matches_dispatch_coverage(self):
        report = json.loads(Path(__file__).with_name(
            "gdn-norm32-v1-native-gpu7-20261031-attempt2.qualification.json").read_text())
        self.assertEqual(report["optimized_fixtures"], 1)
        self.assertEqual(report["finite_bitwise_comparisons"], 236531952)
        for pending in ("1", "64"):
            self.assertEqual(report["program_counts"][pending]["GatedDeltaNetNorm32"], 9)
            self.assertEqual(report["program_counts"][pending]["GatedDeltaNet"], 639)

    def test_model_preflight_accepts_only_the_fresh_norm_only_native_report(self):
        evidence = Path(__file__).parent
        baseline = "qsa-prefill-score16-v1-chunk1024-gpu7-full-standard2048-20261031"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("gdn-norm32-v1-native-gpu7-20261031-attempt2.qualification.json",
                         "qsa-prefill-score16-v1-contexts-20261031.qualification.json",
                         baseline + ".guard.json"):
                (root / name).write_bytes((evidence / name).read_bytes())
            with patch.object(model_runner, "ROOT", root), \
                    patch.object(model_runner, "prepare_model_cache",
                                 side_effect=RuntimeError("preflight reached")), \
                    patch("sys.argv", ["runner", "--phase", "profile", "--full-validation"]):
                with self.assertRaisesRegex(RuntimeError, "preflight reached"):
                    model_runner.main(
                        variant="gdn-norm32-v1", experiment_flag="ORT_WEBGPU_GDN_NORM32",
                        stamp="20261031", parent_variant="qsa-prefill-score16-v1",
                        parent_stamp="20261031", chunk=1024, parent_standard_prefix=baseline,
                        parent_provider="/home/sayanshaw/repos/qwen38-int4-qsa-prefill-score16/build-v1/"
                                        "libonnxruntime_providers_webgpu.so",
                        program_name="GatedDeltaNetNorm32", original_program_name="GatedDeltaNet",
                        dispatches=36)

    def test_original_tree32_transform_reproduces_native_qualified_source(self):
        built = BASE / "qwen38-int4-gdn-tree32/build-v1/gated_delta_net.cc"
        self.assertEqual(transform(SOURCE.read_text(), HEADER.read_text()), built.read_text())

    def test_norm_only_preserves_entire_original_dot_tree_and_original_fallback(self):
        original = TEMPLATE.read_text()
        actual = transform_template(original, norm_only=True)
        marker = ("    for (var value_offset = 0u; value_offset < VALUE_CHANNELS_PER_WORKGROUP;"
                  " value_offset++) {\n      let reduction_offset")
        start = original.index(marker)
        self.assertEqual(actual[actual.index(marker):], original[start:])
        self.assertEqual(actual.count("stride >= 32u"), 1)
        self.assertEqual(actual.count("subgroupShuffle("), 2)
        source = transform(SOURCE.read_text(), HEADER.read_text(), norm_only=True)
        self.assertIn('Program{"GatedDeltaNetNorm32"}', source)
        self.assertIn("qk_l2_norm_ && use_packed_qkv && dk == 128 && dv == 128", source)
        self.assertIn("ORT_WEBGPU_GDN_NORM32 must be 0 or 1", source)
        self.assertNotIn("ORT_WEBGPU_GDN_TREE32", source)
        compare = transform_compare(SOURCE.with_name("native_compare.cpp").read_text(), norm_only=True)
        self.assertIn("ORT_TEST_TOGGLE_GDN_NORM32", compare)
        self.assertNotIn("ORT_TEST_TOGGLE_GDN_TREE32", compare)

    def test_full_expression_tree_and_float32_materialization_order_match(self):
        original = reduce_tree(range(128), lambda left, right: (left, right), False)
        candidate = reduce_tree(range(128), lambda left, right: (left, right), True)
        self.assertEqual(original, candidate)
        rng = random.Random(310132)

        def add(left, right):
            return struct.unpack("f", struct.pack("f", left + right))[0]

        for _ in range(256):
            values = [struct.unpack("f", struct.pack("f", rng.uniform(-1e10, 1e10)))[0]
                      for _ in range(128)]
            self.assertEqual(struct.pack("f", reduce_tree(values, add, False)),
                             struct.pack("f", reduce_tree(values, add, True)))

    def test_only_two_ordered_reduction_blocks_change_in_template(self):
        original = TEMPLATE.read_text()
        actual = transform_template(original)
        self.assertEqual(actual.count("stride >= 32u"), 2)
        self.assertEqual(actual.count("subgroupShuffle("), 5)
        self.assertEqual(actual.count("workgroupBarrier();"), original.count("workgroupBarrier();") + 2)
        for expression in ("key_norm[key_index] += key_norm[key_index + stride]",
                           "retrieved[reduction_offset] += retrieved[reduction_offset + stride]",
                           "pre_output[reduction_offset] += pre_output[reduction_offset + stride]",
                           "key_query[key_index] += key_query[key_index + stride]"):
            self.assertIn(expression, actual)
        for marker in ("gate_scale = exp(gate);", "state[value_offset] *= gate_scale;",
                       "retrieved[reduction_offset] = state[value_offset] * key_value;",
                       "pre_output[reduction_offset] = state[value_offset] * query_value;"):
            self.assertIn(marker, actual)
        self.assertEqual(actual[:actual.index("    for (var stride =")],
                         original[:original.index("    for (var stride =")])
        self.assertEqual(actual[actual.index("    if (key_index == 0u) {\n      var beta_value"):],
                         original[original.index("    if (key_index == 0u) {\n      var beta_value"):])

    def test_private_program_only_and_original_layout_and_fallback_preserved(self):
        source = SOURCE.read_text()
        actual = transform(source, HEADER.read_text())
        start = source.index("Status GatedDeltaNetProgram::GenerateShaderCode")
        stop = source.index("Status GatedDeltaNetNormalizeProgram::GenerateShaderCode", start)
        self.assertIn(source[start:stop], actual)
        self.assertIn(f'Program{{"{PROGRAM}"}}', actual)
        self.assertIn(f'{FLAG} must be 0 or 1', actual)
        for admission in ("total_tokens == 1 && batch == 1", "dk == 128 && dv == 128",
                          "query->IsDataType<MLFloat16>()",
                          "context.HasFeature(wgpu::FeatureName::Subgroups)",
                          "subgroupMinSize == 32 && context.AdapterInfo().subgroupMaxSize == 32"):
            self.assertIn(admission, actual)
        self.assertIn('GatedDeltaNetProgram program{update_rule_', actual)
        self.assertNotIn("class GatedDeltaNet final", actual)
        self.assertNotIn("WGSL_TEMPLATE_PARAMETER(value_channels_per_workgroup, 2)", actual)

    def test_private_template_uses_existing_generator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            templates = root / "templates"
            templates.mkdir()
            (templates / PRIVATE_TEMPLATE).write_text(transform_template(TEMPLATE.read_text()))
            subprocess.run([
                "python3", str(BASE / "onnxruntime-webgpu-h200-perf/tools/python/wgsl_gen.py"),
                "-i", str(templates), "--output", str(root / "generated"),
                "--generator", "static-cpp-literal"], check=True, capture_output=True, text=True)
            index = (root / "generated/index.h").read_text()
            self.assertIn(f'TemplateParameter<"{PRIVATE_TEMPLATE}">', index)
            implementation = (root / "generated/generated/gdn_decode_tree32.h").read_text()
            self.assertIn("subgroupShuffle", implementation)
            self.assertNotIn("__str_", implementation)

    def test_capture_comparator_checks_both_outputs_and_old_native_stamp_default(self):
        source = SOURCE.with_name("native_compare.cpp").read_text()
        actual = capture_transform(transform_compare(source))
        self.assertIn("std::memcmp(", actual)
        self.assertIn("std::isfinite(av)", actual)
        self.assertIn("output < names.size()", actual)
        self.assertIn("ORT_TEST_TOGGLE_GDN_TREE32", actual)
        self.assertIn(f"{FLAG} must be 0 or 1", actual)
        self.assertNotIn("VALUE2", actual)
        self.assertEqual(inspect.signature(native_main).parameters["parent_stamp"].default, "20261030")

    def test_window_metadata_selector_and_cli_are_wired(self):
        self.assertEqual(GDN_TREE32_FLAG, FLAG)
        benchmark = Path(__file__).with_name("benchmark_webgpu_int4.py").read_text()
        for marker in ("--compare-gdn-tree32-windows", "args.compare_gdn_tree32_windows or",
                       "gdn_tree32_decode_windows", f'"{FLAG}",', "GDN_TREE32_FLAG,",
                       "GDN_TREE32_FLAG if args.compare_gdn_tree32_windows"):
            self.assertIn(marker, benchmark)


if __name__ == "__main__":
    unittest.main()
