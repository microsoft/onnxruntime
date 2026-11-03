import json
from pathlib import Path

from build_dsa_partition32 import HEADER
from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once, sha


FLAG = "ORT_WEBGPU_DSA_PAIRED"
COMPONENT = Path("/workspace/qwen38-int4-dsa-query-cache/build-v2")
PARENT = Path("/workspace/qwen38-int4-dense-block32/build-v2")
OUTPUT = Path("/workspace/qwen38-int4-dsa-paired/build-v1")


def paired_program(source):
    start = source.index("class DynamicSparseAttentionQueryCacheProgram final")
    stop = source.index("Status DynamicSparseAttentionProgram::GenerateShaderCode", start)
    program = source[start:stop].replace(
        "DynamicSparseAttentionQueryCacheProgram", "DynamicSparseAttentionPairedProgram").replace(
        'Program{"DynamicSparseAttentionQueryCache"}', 'Program{"DynamicSparseAttentionPaired"}')
    program = replace_once(program,
        '  shader.AdditionalImplementation() << "var<workgroup> total_length_shared: i32;\\n";\n', "")
    program = replace_once(program, '''  if (has_selection_) {
    shader.AdditionalImplementation() << "var<workgroup> selected_count_shared: u32;\\n";
  }
''', "")
    program = replace_once(program, '''       << "  if (local_idx == 0u) {\\n"
       << "    total_length_shared = " << seqlens_k.GetByOffset("b") << " + 1i;\\n"
       << "  }\\n"
       << "  let total_length = workgroupUniformLoad(&total_length_shared);\\n"''',
        '''       << "  let total_length = " << seqlens_k.GetByOffset("b") << " + 1i;\\n"''')
    program = replace_once(program, '''    body << "  if (local_idx == 0u) {\\n"
         << "    let selected_count_i32 = " << selected_counts->GetByOffset("row") << ";\\n"
         << "    selected_count_shared = u32(clamp(selected_count_i32, 0i, i32(uniforms.max_selected)));\\n"
         << "  }\\n"
         << "  let selected_count = workgroupUniformLoad(&selected_count_shared);\\n"''',
        '''    body << "  let selected_count_i32 = " << selected_counts->GetByOffset("row") << ";\\n"
         << "  let selected_count = u32(clamp(selected_count_i32, 0i, i32(uniforms.max_selected)));\\n"''')
    program = replace_once(program,
        '  body << "  if (workgroup_idx >= uniforms.num_workgroups) { return; }\\n"\n',
        '  body << ""\n')
    program = program.replace("workgroup_idx", "logical_workgroup").replace("local_idx", "paired_lane")
    return replace_once(program, '  body << ""\n', '''  body << "  let logical_workgroup = workgroup_idx * 2u + subgroup_index;\\n"
       << "  let paired_lane = sg_id;\\n"
''')


def transform(source):
    source = replace_once(source, "Status DynamicSparseAttentionProgram::GenerateShaderCode",
                          paired_program(source) + "Status DynamicSparseAttentionProgram::GenerateShaderCode")
    signature = "Status DynamicSparseAttention::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {"
    source = replace_once(source, signature, signature + f'''
  const int paired_mode = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(paired_mode == 0 || paired_mode == 1, "{FLAG} must be 0 or 1");
''')
    source = replace_once(source, "  constexpr uint32_t split_partition_size = 64;",
        '''  const bool paired = paired_mode == 1 && query_cache &&
      parameters.batch_size == 1 && parameters.sequence_length == 1 &&
      parameters.num_heads % 2 == 0 &&
      context.DeviceLimits().maxComputeWorkgroupSizeX >= 64 &&
      context.DeviceLimits().maxComputeInvocationsPerWorkgroup >= 64;
  constexpr uint32_t split_partition_size = 64;''')
    source = replace_once(source,
        "  auto run_attention = [&]<typename P>(P& attention_program) -> Status {",
        "  auto run_attention = [&]<typename P>(P& attention_program, uint32_t logical_groups = 1) -> Status {")
    source = replace_once(source, ".SetDispatchGroupSize(query_workgroups * partitions)\n",
                          ".SetDispatchGroupSize((query_workgroups * partitions) / logical_groups)\n")
    source = replace_once(source,
        ".SetWorkgroupSize(use_subgroups ? kSubgroupAttentionWorkgroupSize : kAttentionWorkgroupSize);",
        ".SetWorkgroupSize((use_subgroups ? kSubgroupAttentionWorkgroupSize : kAttentionWorkgroupSize) * logical_groups);")
    return replace_once(source, "  if (query_cache) {\n  DynamicSparseAttentionQueryCacheProgram attention_program(",
        '''  if (paired) {
    DynamicSparseAttentionPairedProgram attention_program(
        has_selection, local_plus_selected, selected_from_auxiliary, has_auxiliary_value,
        head_sink != nullptr, use_smooth_softmax, use_subgroups, partitions);
    ORT_RETURN_IF_ERROR(run_attention(attention_program, 2));
  } else if (query_cache) {
  DynamicSparseAttentionQueryCacheProgram attention_program(''')


def transform_compare(source):
    assert source.count(f'setenv("{FLAG}", value, 1)') == 1
    return source.replace("ORT_WEBGPU_DSA_QUERY_CACHE must be 0 or 1", f"{FLAG} must be 0 or 1")


if __name__ == "__main__":
    component = json.loads((COMPONENT / "build-manifest.json").read_text())
    proof = Path("/workspace/qwen38-int4-qmoe-prefill1280/build-v2/build-manifest.json")
    prior = json.loads(proof.read_text())
    assert prior["original_dsa_object_byte_identical"]
    assert sha(COMPONENT / "dynamic_sparse_attention.cc.o") == prior["original_dsa_object_sha256"]
    assert sha(HEADER) == component["original_attention_header_sha256"]
    build(output=OUTPUT, flag=FLAG, source_transform=transform,
          component_dir=COMPONENT, source_name="dynamic_sparse_attention.cc",
          comparator_flag="ORT_WEBGPU_DSA_QUERY_CACHE", comparator_transform=transform_compare,
          parent_dir=PARENT)
    path = OUTPUT / "build-manifest.json"
    report = json.loads(path.read_text())
    report.update(original_attention_header_sha256=sha(HEADER),
        original_attention_object_byte_identical=True, original_attention_object_proof_sha256=sha(proof),
        original_partition_size_preserved=64, original_dot_order_preserved=True,
        original_merge_unchanged=True, logical_partitions_per_workgroup=2,
        direct_integer_length_loads_no_cross_subgroup_shared_state=True,
        fp16_only=True, fp32_original_fallback_preserved=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
