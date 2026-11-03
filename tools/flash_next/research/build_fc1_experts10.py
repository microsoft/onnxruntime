from pathlib import Path

from build_fc2_subgroup_fold import FLAG as FOLD_FLAG, OVERLAY, transform_comparator as fold_comparator
from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_QMOE_FC1_EXPERTS10"
PROGRAM = "QmoeFC1Experts10BitcastF16"


def transform_program(source):
    source = replace_once(source, "QmoeFC1SwiGLUProgram(bool bias, bool bitcast)",
                          "QmoeFC1SwiGLUProgram(bool bias, bool bitcast, bool multi_expert)")
    source = replace_once(source, 'Program{bitcast ? "QmoeFC1SwiGLUBitcastF16"',
                          'Program{multi_expert ? (bitcast ? "QmoeFC1Experts10BitcastF16" : '
                          '"QmoeFC1Experts10F16") : bitcast ? "QmoeFC1SwiGLUBitcastF16"')
    source = replace_once(source, "bias_{bias}, bitcast_{bitcast} {}",
                          "bias_{bias}, bitcast_{bitcast}, multi_expert_{multi_expert} {}")
    replacements = (
        ("var<workgroup> partials: array<array<f16, 32>, 8>;\\n",
         "var<workgroup> partials: array<array<f16, 32>, 20>;\\n"),
        ("let row = workgroup_idx / uniforms.tiles;\\n", "let row = local_idx / 32u;\\n"),
        ("let base_col = (workgroup_idx % uniforms.tiles) * 8u;\\n", "let base_col = workgroup_idx * 2u;\\n"),
        ("let group = local_idx / 32u;\\n", "let group = 0u;\\n"),
        ("  for (var id = local_idx; id < 256u; id += 128u) {\\n",
         "  for (var id = local_idx; id < 256u; id += 320u) {\\n"),
        ("  for (var offset = 0u; offset < 8u; offset += 4u) {\\n",
         "  for (var offset = 0u; offset < 2u; offset++) {\\n"),
        ("      partials[offset + group][lane] += sum;\\n",
         "      partials[row * 2u + offset][lane] += sum;\\n"),
        ("if (local_idx < 8u && base_col + local_idx < uniforms.cols * 2u) {\\n",
         "if (local_idx < 20u) {\\n"),
        ("  value += bias[expert * uniforms.cols * 2u + base_col + local_idx];\\n",
         "  value += bias[experts[local_idx / 2u] * uniforms.cols * 2u + base_col + local_idx % 2u];\\n"),
        ("if (local_idx < 4u) {\\n", "if (local_idx < 10u) {\\n"),
        ("  let col = base_col / 2u + local_idx;\\n", "  let col = base_col / 2u;\\n"),
    )
    for original, candidate in replacements:
        source = replace_once(source, f'<< "{original}"',
                              f'<< (multi_expert_ ? "{candidate}" : "{original}")')
    source = replace_once(source, 'output.SetByOffset("row * uniforms.cols + col",',
                          'output.SetByOffset(multi_expert_ ? "local_idx * uniforms.cols + col" : '
                          '"row * uniforms.cols + col",')
    return replace_once(source, "  bool bitcast_;", "  bool bitcast_;\n  bool multi_expert_;")


def transform(source, program_source=None):
    validation = ('  ORT_RETURN_IF_NOT(prefill1024 == 0 || prefill1024 == 1, '
                  '"ORT_WEBGPU_QMOE_PREFILL1024 must be 0 or 1");')
    assert source.count(validation) == 2
    source = source.replace(validation, validation + f'''
  const int fc1_experts10 = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(fc1_experts10 == 0 || fc1_experts10 == 1, "{FLAG} must be 0 or 1");''')
    source = replace_once(source, "  key.push_back(prefill1024);",
                          "  key.push_back(prefill1024);\n  key.push_back(fc1_experts10);")
    source = replace_once(source,
        "QmoeFC1SwiGLUProgram program{fc1_experts_bias_optional != nullptr, expert_bitcast == 1};",
        '''const bool use_fc1_experts10 = fc1_experts10 == 1 &&
          context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
          context.DeviceLimits().maxComputeWorkgroupSizeX >= 320 &&
          context.DeviceLimits().maxComputeInvocationsPerWorkgroup >= 320 &&
          context.DeviceLimits().maxComputeWorkgroupStorageSize >= 3328;
      QmoeFC1SwiGLUProgram program{fc1_experts_bias_optional != nullptr, expert_bitcast == 1,
                                 use_fc1_experts10};''')
    source = replace_once(source, ".SetWorkgroupSize(128).SetDispatchGroupSize(tiles, routed_rows)",
                          ".SetWorkgroupSize(use_fc1_experts10 ? 320 : 128)"
                          ".SetDispatchGroupSize(use_fc1_experts10 ? cols : tiles, use_fc1_experts10 ? 1 : routed_rows)")
    return replace_once(source, '#include "qmoe_fc1_activation.inc"',
                        transform_program(program_source if program_source is not None else
                                          (OVERLAY / "qmoe_fc1_activation.inc").read_text()))


def transform_comparator(source):
    return fold_comparator(source).replace(f'"{FOLD_FLAG}"', f'"{FLAG}"')


if __name__ == "__main__":
    build(output=Path("/workspace/qwen38-int4-fc1-experts10/build-v1"), flag=FLAG,
          source_transform=transform,
          component_dir=Path("/workspace/qwen38-int4-qmoe-grouped-prefill/build-v1"),
          source_name="qmoe.cc", comparator_flag="ORT_WEBGPU_QMOE_GROUPED_PREFILL",
          comparator_transform=transform_comparator,
          parent_dir=Path("/workspace/qwen38-int4-qmoe-prefill-rows4/build-v2"),
          extra_include_dirs=(OVERLAY,))
