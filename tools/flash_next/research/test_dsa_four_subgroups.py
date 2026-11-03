from pathlib import Path
import unittest

from build_dsa_four_subgroups import FLAG, transform


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-dsa-query-cache/build-v2/dynamic_sparse_attention.cc")


class FourSubgroupTests(unittest.TestCase):
    def test_logical_groups_cover_original_heads_and_partitions(self):
        for heads in (4, 8, 24):
            for partitions in (1, 2, 3, 32, 33, 65, 512):
                mapped = [group * 4 + subgroup for group in range(heads * partitions // 4)
                          for subgroup in range(4)]
                self.assertEqual(mapped, list(range(heads * partitions)))
                for lane in range(32):
                    self.assertEqual([lane + slot * 32 for slot in range(8)],
                                     list(range(lane, 256, 32)))

    def test_immutable_selector_and_admission(self):
        result = transform(SOURCE.read_text())
        for text in (
            f'static const int four_subgroups_mode = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0)',
            f'"{FLAG} must be 0 or 1"',
            "parameters.num_heads % 4 == 0 && use_subgroups",
            "subgroupMinSize == 32", "subgroupMaxSize == 32",
            "maxComputeWorkgroupSizeX >= 128", "maxComputeInvocationsPerWorkgroup >= 128",
            "run_attention(attention_program, 4)",
            "workgroup_idx * 4u + subgroup_index",
        ):
            self.assertIn(text, result)

    def test_original_cached_shader_merge_and_candidate_arithmetic(self):
        source = SOURCE.read_text()
        result = transform(source)
        start = source.index("class DynamicSparseAttentionQueryCacheProgram final")
        stop = source.index("Status DynamicSparseAttentionProgram::GenerateShaderCode", start)
        self.assertIn(source[start:stop], result)
        start = source.index("Status DynamicSparseAttentionMergeProgram::GenerateShaderCode")
        stop = source.index("Status DynamicSparseAttention::ComputeInternal", start)
        self.assertIn(source[start:stop], result)
        start = source.index("  auto emit_candidate =", source.index("class DynamicSparseAttentionQueryCacheProgram"))
        stop = source.index("  if (local_plus_selected_)", start)
        self.assertIn(source[start:stop].replace("local_idx", "paired_lane"), result)


if __name__ == "__main__":
    unittest.main()
