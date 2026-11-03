import json
from pathlib import Path

from build_dsa_paired import transform as paired_transform
from build_qmoe_node_replay import replace_once
from build_scoped_webgpu_source import build


FLAG = "ORT_WEBGPU_DSA_FOUR_SUBGROUPS"
NAME = "DynamicSparseAttentionFourSubgroups"
SOURCE = Path("/workspace/qwen38-int4-dsa-query-cache/build-v2/dynamic_sparse_attention.cc")
OUTPUT = Path("/workspace/qwen38-int4-dsa-four-subgroups/build-v1")


def transform(source):
    result = paired_transform(source).replace("ORT_WEBGPU_DSA_PAIRED", FLAG).replace(
        "DynamicSparseAttentionPaired", NAME).replace("paired_mode", "four_subgroups_mode")
    result = replace_once(result, f'const int four_subgroups_mode = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);',
                          f'static const int four_subgroups_mode = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);')
    result = replace_once(result, "workgroup_idx * 2u + subgroup_index",
                          "workgroup_idx * 4u + subgroup_index")
    result = replace_once(result, "parameters.num_heads % 2 == 0",
        'parameters.num_heads % 4 == 0 && use_subgroups &&\n'
        '      context.AdapterInfo().vendor == std::string_view{"nvidia"} &&\n'
        '      context.AdapterInfo().subgroupMinSize == 32 && context.AdapterInfo().subgroupMaxSize == 32')
    result = replace_once(result, "maxComputeWorkgroupSizeX >= 64", "maxComputeWorkgroupSizeX >= 128")
    result = replace_once(result, "maxComputeInvocationsPerWorkgroup >= 64", "maxComputeInvocationsPerWorkgroup >= 128")
    return replace_once(result, "run_attention(attention_program, 2)", "run_attention(attention_program, 4)")


if __name__ == "__main__":
    report = build(OUTPUT, SOURCE, transform, SOURCE.parent / "native_compare.cpp", FLAG,
        dict(model_eligible=False, physical_threads=128, logical_attention_groups=4,
             original_partition_size_preserved=64, original_dot_order_preserved=True,
             original_merge_unchanged=True, native_independent_reference_required=True))
    report.update(original_compute_and_fallbacks_unchanged=False,
                  original_default_compute_and_fallbacks_unchanged=True)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
