import json
from pathlib import Path

from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once, sha


FLAG = "ORT_WEBGPU_DSA_PARTITION32"
COMPONENT = Path("/workspace/qwen38-int4-qmoe-prefill512/build-v3")
OUTPUT = Path("/workspace/qwen38-int4-dsa-partition32/build-v1")
HEADER = Path("/workspace/onnxruntime-webgpu-h200-perf/onnxruntime/contrib_ops/webgpu/bert/dynamic_sparse_attention.h")


def private_program(source, header):
    start = header.index("class DynamicSparseAttentionProgram final")
    stop = header.index("class DynamicSparseAttentionMergeProgram", start)
    declaration = header[start:stop]
    start = source.index("Status DynamicSparseAttentionProgram::GenerateShaderCode")
    stop = source.index("Status DynamicSparseAttentionMergeProgram::GenerateShaderCode", start)
    method = source[start:stop]
    return ((declaration + "\n" + method)
            .replace("DynamicSparseAttentionProgram", "DynamicSparseAttentionPartition32Program")
            .replace('Program{"DynamicSparseAttention"}', 'Program{"DynamicSparseAttentionPartition32"}'))


def dispatch_private_attention(source, private_class, condition):
    start = source.index("  DynamicSparseAttentionProgram attention_program(")
    middle = source.index("  attention_program.CacheHint(", start)
    stop = source.index("  ORT_RETURN_IF_ERROR(context.RunProgram(attention_program));", middle)
    end = stop + len("  ORT_RETURN_IF_ERROR(context.RunProgram(attention_program));")
    constructor = source[start:middle]
    setup = source[middle:stop]
    replacement = (
        "  auto run_attention = [&]<typename P>(P& attention_program) -> Status {\n" +
        setup + "    return context.RunProgram(attention_program);\n  };\n"
        f"  if ({condition}) {{\n" +
        constructor.replace("DynamicSparseAttentionProgram", private_class) +
        "    ORT_RETURN_IF_ERROR(run_attention(attention_program));\n"
        "  } else {\n" + constructor +
        "    ORT_RETURN_IF_ERROR(run_attention(attention_program));\n  }")
    return source[:start] + replacement + source[end:]


def transform(source, header):
    program = private_program(source, header)
    source = replace_once(source, "Status DynamicSparseAttentionProgram::GenerateShaderCode",
                          program + "Status DynamicSparseAttentionProgram::GenerateShaderCode")
    signature = "Status DynamicSparseAttention::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {"
    source = replace_once(source, signature, signature + f'''
  const int partition32_mode = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(partition32_mode == 0 || partition32_mode == 1, "{FLAG} must be 0 or 1");
''')
    source = replace_once(source, "  constexpr uint32_t split_partition_size = 64;",
        '''  const bool partition32 = partition32_mode == 1 && split_kv &&
      static_cast<uint64_t>(query_workgroups) * ((candidates_limit + 31u) / 32u) <=
          context.DeviceLimits().maxComputeWorkgroupsPerDimension;
  const uint32_t split_partition_size = partition32 ? 32u : 64u;''')
    return dispatch_private_attention(source, "DynamicSparseAttentionPartition32Program", "partition32")


def transform_compare(source):
    assert source.count(f'setenv("{FLAG}", value, 1)') == 1
    return source


if __name__ == "__main__":
    original_header_sha256 = sha(HEADER)
    build(output=OUTPUT, flag=FLAG, source_transform=lambda source: transform(source, HEADER.read_text()),
          component_dir=COMPONENT, source_name="dynamic_sparse_attention.cc",
          comparator_flag="ORT_WEBGPU_QMOE_PREFILL512", comparator_transform=transform_compare,
          parent_dir=COMPONENT)
    path = OUTPUT / "build-manifest.json"
    report = json.loads(path.read_text())
    report.update(original_attention_header_sha256=original_header_sha256,
                  original_shader_expressions_preserved=True,
                  partition_grouping_changes_fp32_softmax_association=True,
                  native_cpu_reference_required=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
