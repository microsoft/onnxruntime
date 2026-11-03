import json
from pathlib import Path
import subprocess

from build_gdn_value2 import main as build, transform_compare as value2_compare
from build_gdn_value2_capture import main as build_capture
from build_qmoe_node_replay import replace_once, sha


FLAG = "ORT_WEBGPU_GDN_TREE32"
PROGRAM = "GatedDeltaNetTree32"
OUTPUT = Path("/workspace/qwen38-int4-gdn-tree32/build-v1")
TEMPLATE = Path("/workspace/onnxruntime-webgpu-h200-perf/onnxruntime/contrib_ops/webgpu/bert/"
                "gated_delta_net.wgsl.template")
PRIVATE_TEMPLATE = "gdn_decode_tree32.wgsl.template"


def transform_template(source, norm_only=False):
    norm = """    for (var stride = workgroup_size_x >> 1u; stride > 0u; stride = stride >> 1u) {
      if (key_index < stride) {
        key_norm[key_index] += key_norm[key_index + stride];
        query_norm[key_index] += query_norm[key_index + stride];
      }
      workgroupBarrier();
    }"""
    norm_tail = """
    if (subgroup_index == 0u) {
      var key_sum = key_norm[sg_id];
      var query_sum = query_norm[sg_id];
      for (var stride = 16u; stride > 0u; stride >>= 1u) {
        let other_key = subgroupShuffle(key_sum, (sg_id + stride) & 31u);
        let other_query = subgroupShuffle(query_sum, (sg_id + stride) & 31u);
        if (sg_id < stride) {
          key_sum += other_key;
          query_sum += other_query;
        }
      }
      if (sg_id == 0u) {
        key_norm[0] = key_sum;
        query_norm[0] = query_sum;
      }
    }
    workgroupBarrier();"""
    source = replace_once(source, norm, norm.replace("stride > 0u", "stride >= 32u") + norm_tail)
    if norm_only:
        return source
    dot = """    for (var stride = workgroup_size_x >> 1u; stride > 0u; stride = stride >> 1u) {
      if (key_index < stride) {
        for (var value_offset = 0u; value_offset < VALUE_CHANNELS_PER_WORKGROUP; value_offset++) {
          let reduction_offset = value_offset * workgroup_size_x + key_index;
          retrieved[reduction_offset] += retrieved[reduction_offset + stride];
          pre_output[reduction_offset] += pre_output[reduction_offset + stride];
        }
        key_query[key_index] += key_query[key_index + stride];
      }
      workgroupBarrier();
    }"""
    dot_tail = """
    if (subgroup_index == 0u) {
      for (var value_offset = 0u; value_offset < VALUE_CHANNELS_PER_WORKGROUP; value_offset++) {
        let reduction_offset = value_offset * workgroup_size_x + sg_id;
        var retrieved_sum = retrieved[reduction_offset];
        var output_sum = pre_output[reduction_offset];
        for (var stride = 16u; stride > 0u; stride >>= 1u) {
          let other_retrieved = subgroupShuffle(retrieved_sum, (sg_id + stride) & 31u);
          let other_output = subgroupShuffle(output_sum, (sg_id + stride) & 31u);
          if (sg_id < stride) {
            retrieved_sum += other_retrieved;
            output_sum += other_output;
          }
        }
        if (sg_id == 0u) {
          retrieved[value_offset * workgroup_size_x] = retrieved_sum;
          pre_output[value_offset * workgroup_size_x] = output_sum;
        }
      }
      var key_query_sum = key_query[sg_id];
      for (var stride = 16u; stride > 0u; stride >>= 1u) {
        let other_key_query = subgroupShuffle(key_query_sum, (sg_id + stride) & 31u);
        if (sg_id < stride) { key_query_sum += other_key_query; }
      }
      if (sg_id == 0u) { key_query[0] = key_query_sum; }
    }
    workgroupBarrier();"""
    return replace_once(source, dot, dot.replace("stride > 0u", "stride >= 32u") + dot_tail)


def transform(source, header, norm_only=False):
    output = Path("/workspace/qwen38-int4-gdn-norm32/build-v1") if norm_only else OUTPUT
    flag = "ORT_WEBGPU_GDN_NORM32" if norm_only else FLAG
    program = "GatedDeltaNetNorm32" if norm_only else PROGRAM
    private_class = "GatedDeltaNetNorm32Program" if norm_only else "GatedDeltaNetTree32Program"
    private_template = "gdn_decode_norm32.wgsl.template" if norm_only else PRIVATE_TEMPLATE
    source = replace_once(source, "#include <limits>", "#include <limits>\n#include <string_view>")
    source = replace_once(source, '#include "core/providers/webgpu/shader_helper.h"',
                          '#include "core/providers/webgpu/shader_helper.h"\n'
                          '#include "core/providers/webgpu/webgpu_utils.h"')
    start = header.index("class GatedDeltaNetProgram final")
    stop = header.index("\n};", start) + len("\n};")
    declaration = header[start:stop].replace("GatedDeltaNetProgram", private_class)
    declaration = replace_once(declaration, 'Program{"GatedDeltaNet"}', f'Program{{"{program}"}}')
    start = source.index("Status GatedDeltaNetProgram::GenerateShaderCode")
    stop = source.index("Status GatedDeltaNetNormalizeProgram::GenerateShaderCode", start)
    method = source[start:stop].replace("GatedDeltaNetProgram::", f"{private_class}::")
    method = replace_once(method, "bert/gated_delta_net.wgsl.template", private_template)
    generated = output / "generated"
    includes = f"""
namespace onnxruntime::webgpu::wgsl_gen {{
#define INCLUDED_BY_WGSL_GEN_HEADER
#include "{generated}/index.h"
#undef INCLUDED_BY_WGSL_GEN_HEADER
#define INCLUDED_BY_WGSL_GEN_IMPL
#include "{generated}/index_impl.h"
#undef INCLUDED_BY_WGSL_GEN_IMPL
}}
"""
    source = replace_once(source, "namespace onnxruntime::contrib::webgpu {",
                          includes + "\nnamespace onnxruntime::contrib::webgpu {")
    start = source.index("Status GatedDeltaNetProgram::GenerateShaderCode")
    source = source[:start] + declaration + "\n\n" + method + source[start:]
    signature = "Status GatedDeltaNet::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {"
    if signature not in source:
        signature = "Status GatedDeltaNet::ComputeInternal(ComputeContext& context) const {"
    source = replace_once(source, signature, signature + f"""
  const int tree32_mode = ParseEnvironmentVariableWithDefault<int>("{flag}", 0);
  ORT_RETURN_IF_NOT(tree32_mode == 0 || tree32_mode == 1, "{flag} must be 0 or 1");
""")
    start = source.index("  GatedDeltaNetProgram program{", source.index("const bool vectorized_value_io ="))
    end = source.index(";", start) + 1
    declaration = source[start:end]
    stop = source.index("  return context.RunProgram(program);", end) + len("  return context.RunProgram(program);")
    setup = source[end:stop]
    private = declaration.replace("GatedDeltaNetProgram", private_class)
    replacement = """
  const bool tree32 = tree32_mode == 1 && total_tokens == 1 && batch == 1 &&
      use_packed_qkv && dk == 128 && dv == 128 &&
      query->IsDataType<MLFloat16>() &&
      context.HasFeature(wgpu::FeatureName::Subgroups) &&
      context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
      context.AdapterInfo().subgroupMinSize == 32 && context.AdapterInfo().subgroupMaxSize == 32;
  auto run_program = [&](auto& program) -> Status {
""" + setup + """
  };
  if (tree32) {
""" + private + """
    return run_program(program);
  }
""" + declaration + """
  return run_program(program);"""
    if norm_only:
        replacement = replace_once(replacement, "      use_packed_qkv && dk == 128 && dv == 128",
                                   "      qk_l2_norm_ && use_packed_qkv && dk == 128 && dv == 128")
    return source[:start] + replacement + source[stop:]


def transform_compare(source, norm_only=False):
    flag = "ORT_WEBGPU_GDN_NORM32" if norm_only else FLAG
    toggle = "ORT_TEST_TOGGLE_GDN_NORM32" if norm_only else "ORT_TEST_TOGGLE_GDN_TREE32"
    return value2_compare(source).replace("ORT_WEBGPU_GDN_VALUE2", flag).replace(
        "ORT_TEST_TOGGLE_GDN_VALUE2", toggle).replace("value2", "tree32")


def main(norm_only=False):
    output = Path("/workspace/qwen38-int4-gdn-norm32/build-v1") if norm_only else OUTPUT
    private_template = "gdn_decode_norm32.wgsl.template" if norm_only else PRIVATE_TEMPLATE
    def generate(source, header):
        templates = output / "templates"
        templates.mkdir()
        original = TEMPLATE.read_text()
        frozen_template = Path("/workspace/onnxruntime-webgpu-fp8-checkpoint/onnxruntime/contrib_ops/"
                               "webgpu/bert/gated_delta_net.wgsl.template")
        assert sha(TEMPLATE) == sha(frozen_template)
        (templates / private_template).write_text(transform_template(original, norm_only=norm_only))
        subprocess.run([
            "python3", "/workspace/onnxruntime-webgpu-h200-perf/tools/python/wgsl_gen.py",
            "-i", str(templates), "--output", str(output / "generated"),
            "--generator", "static-cpp-literal"], check=True)
        return transform(source, header, norm_only=norm_only)

    build(parent_dir=Path("/workspace/qwen38-int4-qsa-prefill-score16/build-v1"),
          output=output, source_transform=generate,
          comparator_transform=lambda source: transform_compare(source, norm_only=norm_only))
    manifest_path = output / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["original_template_sha256"] = sha(TEMPLATE)
    manifest["fixed_subgroup_size"] = 32
    manifest["original_128_lane_tree_association_preserved"] = True
    manifest["only_five_intra_subgroup_tree_levels_replaced"] = True
    manifest["norm_tree_only"] = norm_only
    manifest["shader_generation"] = "repository_static_cpp_literal_generator"
    for path in sorted((output / "generated").rglob("*.h")):
        manifest["source_sha256"][str(path.relative_to(output))] = sha(path)
    manifest["source_sha256"][f"templates/{private_template}"] = sha(output / "templates" / private_template)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    build_capture(output)


if __name__ == "__main__":
    main()
