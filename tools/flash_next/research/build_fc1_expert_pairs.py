from pathlib import Path

from build_fc2_subgroup_fold import FLAG as FOLD_FLAG, OVERLAY, transform_comparator as fold_comparator
from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_QMOE_FC1_EXPERT_PAIRS"
PROGRAM = "QmoeFC1ExpertPairsBitcastF16"


def transform_program(source):
    source = replace_once(source, "QmoeFC1SwiGLUProgram(bool bias, bool bitcast)",
                          "QmoeFC1SwiGLUProgram(bool bias, bool bitcast, bool expert_pairs)")
    source = replace_once(source, 'Program{bitcast ? "QmoeFC1SwiGLUBitcastF16"',
                          'Program{expert_pairs ? (bitcast ? "QmoeFC1ExpertPairsBitcastF16" : '
                          '"QmoeFC1ExpertPairsF16") : bitcast ? "QmoeFC1SwiGLUBitcastF16"')
    source = replace_once(source, "bias_{bias}, bitcast_{bitcast} {}",
                          "bias_{bias}, bitcast_{bitcast}, expert_pairs_{expert_pairs} {}")
    replacements = (
        ("var<workgroup> partials: array<array<f16, 32>, 8>;\\n",
         "var<workgroup> partials: array<array<f16, 32>, 16>;\\n"),
        ("let row = workgroup_idx / uniforms.tiles;\\n",
         "let row = (workgroup_idx / uniforms.tiles) * 2u + local_idx / 128u;\\n"),
        ("let group = local_idx / 32u;\\n", "let group = (local_idx % 128u) / 32u;\\n"),
        ("  for (var id = local_idx; id < 256u; id += 128u) {\\n",
         "  for (var id = local_idx; id < 256u; id += 256u) {\\n"),
        ("      partials[offset + group][lane] += sum;\\n",
         "      partials[(local_idx / 128u) * 8u + offset + group][lane] += sum;\\n"),
        ("if (local_idx < 8u && base_col + local_idx < uniforms.cols * 2u) {\\n",
         "if (local_idx % 128u < 8u && base_col + local_idx % 128u < uniforms.cols * 2u) {\\n"),
        ("  for (var i = 0u; i < 32u; i++) { value += partials[local_idx][i]; }\\n",
         "  for (var i = 0u; i < 32u; i++) { "
         "value += partials[(local_idx / 128u) * 8u + local_idx % 128u][i]; }\\n"),
        ("  value += bias[expert * uniforms.cols * 2u + base_col + local_idx];\\n",
         "  value += bias[expert * uniforms.cols * 2u + base_col + local_idx % 128u];\\n"),
        ("  partials[local_idx][0] = value;\\n",
         "  partials[(local_idx / 128u) * 8u + local_idx % 128u][0] = value;\\n"),
        ("if (local_idx < 4u) {\\n", "if (local_idx % 128u < 4u) {\\n"),
        ("  let col = base_col / 2u + local_idx;\\n", "  let col = base_col / 2u + local_idx % 128u;\\n"),
        ("    var gate_value = f32(partials[local_idx * 2u][0]);\\n",
         "    var gate_value = f32(partials[(local_idx / 128u) * 8u + (local_idx % 128u) * 2u][0]);\\n"),
        ("    var linear_value = f32(partials[local_idx * 2u + 1u][0]);\\n",
         "    var linear_value = f32(partials[(local_idx / 128u) * 8u + (local_idx % 128u) * 2u + 1u][0]);\\n"),
    )
    for original, candidate in replacements:
        source = replace_once(source, f'<< "{original}"',
                              f'<< (expert_pairs_ ? "{candidate}" : "{original}")')
    return replace_once(source, "  bool bitcast_;", "  bool bitcast_;\n  bool expert_pairs_;")


def transform(source, program_source=None):
    validation = ('  ORT_RETURN_IF_NOT(prefill1024 == 0 || prefill1024 == 1, '
                  '"ORT_WEBGPU_QMOE_PREFILL1024 must be 0 or 1");')
    assert source.count(validation) == 2
    source = source.replace(validation, validation + f'''
  const int fc1_expert_pairs = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(fc1_expert_pairs == 0 || fc1_expert_pairs == 1, "{FLAG} must be 0 or 1");''')
    source = replace_once(source, "  key.push_back(prefill1024);",
                          "  key.push_back(prefill1024);\n  key.push_back(fc1_expert_pairs);")
    source = replace_once(source,
        "QmoeFC1SwiGLUProgram program{fc1_experts_bias_optional != nullptr, expert_bitcast == 1};",
        '''const bool use_fc1_expert_pairs = fc1_expert_pairs == 1 && routed_rows == 10 &&
          context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
          context.DeviceLimits().maxComputeWorkgroupSizeX >= 256 &&
          context.DeviceLimits().maxComputeInvocationsPerWorkgroup >= 256 &&
          context.DeviceLimits().maxComputeWorkgroupStorageSize >= 3072;
      QmoeFC1SwiGLUProgram program{fc1_experts_bias_optional != nullptr, expert_bitcast == 1,
                                 use_fc1_expert_pairs};''')
    source = replace_once(source, ".SetWorkgroupSize(128).SetDispatchGroupSize(tiles, routed_rows)",
                          ".SetWorkgroupSize(use_fc1_expert_pairs ? 256 : 128)"
                          ".SetDispatchGroupSize(tiles, use_fc1_expert_pairs ? routed_rows / 2 : routed_rows)")
    return replace_once(source, '#include "qmoe_fc1_activation.inc"',
                        transform_program(program_source if program_source is not None else
                                          (OVERLAY / "qmoe_fc1_activation.inc").read_text()))


def transform_comparator(source):
    return fold_comparator(source).replace(f'"{FOLD_FLAG}"', f'"{FLAG}"')


if __name__ == "__main__":
    build(output=Path("/workspace/qwen38-int4-fc1-expert-pairs/build-v1"), flag=FLAG,
          source_transform=transform,
          component_dir=Path("/workspace/qwen38-int4-qmoe-grouped-prefill/build-v1"),
          source_name="qmoe.cc", comparator_flag="ORT_WEBGPU_QMOE_GROUPED_PREFILL",
          comparator_transform=transform_comparator,
          parent_dir=Path("/workspace/qwen38-int4-qmoe-prefill-rows4/build-v2"),
          extra_include_dirs=(OVERLAY,))
