import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from build_dsa_paired import paired_program, transform
from qualify_dsa_query_cache import main as qualify, qualify_windows, verify_flags
from qualify_quickgelu_mul_long_context import ROOT
from run_dsa_query_cache_native import main as native


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-dsa-query-cache/build-v2/dynamic_sparse_attention.cc")


class DsaPairedTests(unittest.TestCase):
    def test_existing_query_cache_standards_reproduce_without_writes(self):
        output = ROOT / "dsa-query-cache-v2-contexts-20261030.qualification.json"
        expected = json.loads(output.read_text())
        original_open, exists = Path.open, Path.exists
        captured = []

        @contextlib.contextmanager
        def capture(path, mode="r", *args, **kwargs):
            if path == output and mode == "x":
                stream = io.StringIO()
                yield stream
                captured.append(json.loads(stream.getvalue()))
            else:
                with original_open(path, mode, *args, **kwargs) as stream:
                    yield stream

        with patch("sys.argv", ["qualify_dsa_query_cache.py", "--phase", "standard"]), \
                patch.object(Path, "exists", lambda path: False if path == output else exists(path)), \
                patch.object(Path, "open", capture), contextlib.redirect_stdout(io.StringIO()):
            qualify()
        self.assertEqual(captured, [expected])

    def test_existing_query_cache_windows_reproduce_exactly(self):
        expected = json.loads((ROOT / "dsa-query-cache-v2-windows-contexts-20261030.qualification.json").read_text())
        actual, _, _, _ = qualify_windows()
        self.assertEqual(actual, expected)

    def test_native_cli_defaults_remain_unchanged(self):
        with patch("sys.argv", ["run_dsa_query_cache_native.py", "--inside"]), \
                patch("run_dsa_query_cache_native.inside") as inside:
            native()
        inside.assert_called_once_with(
            directory=Path("/workspace/qwen38-int4-dsa-query-cache/build-v2"),
            prefix_name="dsa-query-cache-v2-native-gpu7-20261030", parent_variant="prefill1024-v2",
            flag="ORT_WEBGPU_DSA_QUERY_CACHE", program_name="DynamicSparseAttentionQueryCache",
            extra_flags=None, report_status="dsa_query_cache_finite_bitwise_and_cpu_reference_qualified")

    def test_paired_metadata_requires_all_stacked_flags(self):
        flags = {flag: "1" for flag in (
            "ORT_WEBGPU_DSA_QUERY_CACHE", "ORT_WEBGPU_DSA_PAIRED", "ORT_WEBGPU_DENSE_BLOCK32",
            "ORT_WEBGPU_QMOE_PREFILL1024", "ORT_WEBGPU_DENSE_TILE4", "ORT_WEBGPU_HC_UP_PARALLEL2",
            "ORT_WEBGPU_HC_UP_PREMIX", "ORT_WEBGPU_QSA_SCORE16")}
        verify_flags({"load_experiment_flags": flags}, True)
        for flag in flags:
            with self.assertRaises(AssertionError):
                verify_flags({"load_experiment_flags": dict(flags, **{flag: "0"})}, True)

    def test_exact_logical_group_and_lane_mapping(self):
        for heads in (2, 4, 24):
            for partitions in (1, 2, 3, 32, 33, 65):
                logical = [(group * 2 + subgroup) for group in range(heads * partitions // 2)
                           for subgroup in range(2)]
                self.assertEqual(logical, list(range(heads * partitions)))
                for index in logical:
                    self.assertLess(index // partitions, heads)
                    for lane in range(32):
                        self.assertEqual([lane + slot * 32 for slot in range(8)],
                                         list(range(lane, 256, 32)))

    def test_original_shaders_merge_and_fallbacks_are_unchanged(self):
        source = SOURCE.read_text()
        actual = transform(source)
        start = source.index("class DynamicSparseAttentionQueryCacheProgram final")
        stop = source.index("Status DynamicSparseAttentionProgram::GenerateShaderCode", start)
        self.assertIn(source[start:stop], actual)
        start = source.index("Status DynamicSparseAttentionMergeProgram::GenerateShaderCode")
        stop = source.index("Status DynamicSparseAttention::ComputeInternal", start)
        self.assertIn(source[start:stop], actual)
        for condition in ("paired_mode == 0 || paired_mode == 1", "paired_mode == 1 && query_cache",
                          "parameters.batch_size == 1 && parameters.sequence_length == 1",
                          "parameters.num_heads % 2 == 0", "maxComputeWorkgroupSizeX >= 64",
                          "maxComputeInvocationsPerWorkgroup >= 64"):
            self.assertIn(condition, actual)
        self.assertIn("run_attention(attention_program, 2)", actual)

    def test_private_shader_keeps_order_and_removes_shared_length_communication(self):
        source = SOURCE.read_text()
        program = paired_program(source)
        self.assertNotIn("total_length_shared", program)
        self.assertNotIn("selected_count_shared", program)
        self.assertNotIn("workgroupUniformLoad", program)
        self.assertIn("workgroup_idx * 2u + subgroup_index", program)
        self.assertIn("let paired_lane = sg_id;", program)
        self.assertNotIn("if (logical_workgroup >= uniforms.num_workgroups)", program)
        start = source.index("  auto emit_candidate =", source.index("class DynamicSparseAttentionQueryCacheProgram"))
        stop = source.index("  if (local_plus_selected_)", start)
        self.assertIn(source[start:stop].replace("local_idx", "paired_lane"), program)
        self.assertIn("use_subgroups_ ? kSubgroupAttentionWorkgroupSize : kAttentionWorkgroupSize", program)


if __name__ == "__main__":
    unittest.main()
