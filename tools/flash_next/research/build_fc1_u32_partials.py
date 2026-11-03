from pathlib import Path

from build_fc2_subgroup_fold import FLAG as FOLD_FLAG, OVERLAY, transform_comparator as fold_comparator
from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_QMOE_FC1_U32_PARTIALS"
PROGRAM = "QmoeFC1U32PartialsBitcastF16"


def transform_program(source):
    source = replace_once(source, "QmoeFC1SwiGLUProgram(bool bias, bool bitcast)",
                          "QmoeFC1SwiGLUProgram(bool bias, bool bitcast, bool u32_partials)")
    source = replace_once(source, 'Program{bitcast ? "QmoeFC1SwiGLUBitcastF16"',
                          'Program{u32_partials ? (bitcast ? "QmoeFC1U32PartialsBitcastF16" : '
                          '"QmoeFC1U32PartialsF16") : bitcast ? "QmoeFC1SwiGLUBitcastF16"')
    source = replace_once(source, "bias_{bias}, bitcast_{bitcast} {}",
                          "bias_{bias}, bitcast_{bitcast}, u32_partials_{u32_partials} {}")
    source = replace_once(source, "  bool bitcast_;", "  bool bitcast_;\n  bool u32_partials_;")
    replacements = (
        ("var<workgroup> partials: array<array<f16, 32>, 8>;\\n",
         "var<workgroup> partials: array<array<u32, 32>, 8>;\\n"),
        ("      partials[offset + group][lane] += sum;\\n",
         "      partials[offset + group][lane] = bitcast<u32>(vec2<f16>("
         "bitcast<vec2<f16>>(partials[offset + group][lane]).x + sum, f16(0)));\\n"),
        ("  for (var i = 0u; i < 32u; i++) { value += partials[local_idx][i]; }\\n",
         "  for (var i = 0u; i < 32u; i++) { "
         "value += bitcast<vec2<f16>>(partials[local_idx][i]).x; }\\n"),
        ("  partials[local_idx][0] = value;\\n",
         "  partials[local_idx][0] = bitcast<u32>(vec2<f16>(value, f16(0)));\\n"),
        ("    var gate_value = f32(partials[local_idx * 2u][0]);\\n",
         "    var gate_value = f32(bitcast<vec2<f16>>(partials[local_idx * 2u][0]).x);\\n"),
        ("    var linear_value = f32(partials[local_idx * 2u + 1u][0]);\\n",
         "    var linear_value = f32(bitcast<vec2<f16>>(partials[local_idx * 2u + 1u][0]).x);\\n"),
    )
    for original, candidate in replacements:
        source = replace_once(source, f'<< "{original}"',
                              f'<< (u32_partials_ ? "{candidate}" : "{original}")')
    return source


def transform(source, program_source=None):
    validation = ('  ORT_RETURN_IF_NOT(prefill1024 == 0 || prefill1024 == 1, '
                  '"ORT_WEBGPU_QMOE_PREFILL1024 must be 0 or 1");')
    assert source.count(validation) == 2
    source = source.replace(validation, validation + f'''
  const int fc1_u32_partials = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(fc1_u32_partials == 0 || fc1_u32_partials == 1, "{FLAG} must be 0 or 1");''')
    source = replace_once(source, "  key.push_back(prefill1024);",
                          "  key.push_back(prefill1024);\n  key.push_back(fc1_u32_partials);")
    source = replace_once(source,
        "QmoeFC1SwiGLUProgram program{fc1_experts_bias_optional != nullptr, expert_bitcast == 1};",
        '''const bool use_fc1_u32_partials = fc1_u32_partials == 1 &&
          context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
          context.DeviceLimits().maxComputeWorkgroupStorageSize >= 3072;
      QmoeFC1SwiGLUProgram program{fc1_experts_bias_optional != nullptr, expert_bitcast == 1,
                                 use_fc1_u32_partials};''')
    return replace_once(source, '#include "qmoe_fc1_activation.inc"',
                        transform_program(program_source if program_source is not None else
                                          (OVERLAY / "qmoe_fc1_activation.inc").read_text()))


def transform_comparator(source):
    return fold_comparator(source).replace(f'"{FOLD_FLAG}"', f'"{FLAG}"')


if __name__ == "__main__":
    build(output=Path("/workspace/qwen38-int4-fc1-u32-partials/build-v1"), flag=FLAG,
          source_transform=transform,
          component_dir=Path("/workspace/qwen38-int4-qmoe-grouped-prefill/build-v1"),
          source_name="qmoe.cc", comparator_flag="ORT_WEBGPU_QMOE_GROUPED_PREFILL",
          comparator_transform=transform_comparator,
          parent_dir=Path("/workspace/qwen38-int4-qmoe-prefill-rows4/build-v2"),
          extra_include_dirs=(OVERLAY,))
