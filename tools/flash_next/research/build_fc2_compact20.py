from pathlib import Path

from build_fc2_subgroup_fold import FLAG as FOLD_FLAG, OVERLAY, transform_comparator as fold_comparator
from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_QMOE_FC2_COMPACT20"
PROGRAM = "QmoeFC2Compact20BitcastF16"


def transform_program(source):
    source = replace_once(source, "QmoeFC2FinalMixProgram(bool bias, bool bitcast)",
                          "QmoeFC2FinalMixProgram(bool bias, bool bitcast, bool compact)")
    source = replace_once(source, 'Program{bitcast ? "QmoeFC2FinalMixBitcastF16"',
                          'Program{compact ? (bitcast ? "QmoeFC2Compact20BitcastF16" : '
                          '"QmoeFC2Compact20F16") : bitcast ? "QmoeFC2FinalMixBitcastF16"')
    source = replace_once(source, "bias_{bias}, bitcast_{bitcast} {}",
                          "bias_{bias}, bitcast_{bitcast}, compact_{compact} {}")
    replacements = (
        ("let rank = local_idx / 32u;\\n", "let rank = local_idx / 20u;\\n"),
        ("let lane = local_idx % 32u;\\n", "let lane = local_idx % 20u;\\n"),
        ("let expert = experts[rank];\\n", ""),
        ("for (var id = local_idx; id < 1600u; id += 320u) {\\n",
         "for (var id = local_idx; id < 1600u; id += 256u) {\\n"),
        ("if (lane < 20u) {\\n",
         "if (local_idx < 200u) {\\n  let expert = experts[rank];\\n"),
    )
    for original, candidate in replacements:
        source = replace_once(source, f'<< "{original}"',
                              f'<< (compact_ ? "{candidate}" : "{original}")')
    return replace_once(source, "  bool bitcast_;", "  bool bitcast_;\n  bool compact_;")


def transform(source, program_source=None):
    validation = ('  ORT_RETURN_IF_NOT(prefill1024 == 0 || prefill1024 == 1, '
                  '"ORT_WEBGPU_QMOE_PREFILL1024 must be 0 or 1");')
    assert source.count(validation) == 2
    source = source.replace(validation, validation + f'''
  const int fc2_compact20 = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(fc2_compact20 == 0 || fc2_compact20 == 1, "{FLAG} must be 0 or 1");''')
    source = replace_once(source, "  key.push_back(prefill1024);",
                          "  key.push_back(prefill1024);\n  key.push_back(fc2_compact20);")
    source = replace_once(source,
        "QmoeFC2FinalMixProgram program{fc2_experts_bias_optional != nullptr, expert_bitcast == 1};",
        '''const bool use_fc2_compact20 = fc2_compact20 == 1 &&
          context.AdapterInfo().vendor == std::string_view{"nvidia"};
      QmoeFC2FinalMixProgram program{fc2_experts_bias_optional != nullptr, expert_bitcast == 1,
                                   use_fc2_compact20};''')
    source = replace_once(source, ".SetWorkgroupSize(320).SetDispatchGroupSize(1280)",
                          ".SetWorkgroupSize(use_fc2_compact20 ? 256 : 320).SetDispatchGroupSize(1280)")
    return replace_once(source, '#include "qmoe_fc2_final_mix.inc"',
                        transform_program(program_source if program_source is not None else
                                          (OVERLAY / "qmoe_fc2_final_mix.inc").read_text()))


def transform_comparator(source):
    return fold_comparator(source).replace(f'"{FOLD_FLAG}"', f'"{FLAG}"')


if __name__ == "__main__":
    build(output=Path("/workspace/qwen38-int4-fc2-compact20/build-v1"), flag=FLAG,
          source_transform=transform,
          component_dir=Path("/workspace/qwen38-int4-qmoe-grouped-prefill/build-v1"),
          source_name="qmoe.cc", comparator_flag="ORT_WEBGPU_QMOE_GROUPED_PREFILL",
          comparator_transform=transform_comparator,
          parent_dir=Path("/workspace/qwen38-int4-qmoe-prefill-rows4/build-v2"),
          extra_include_dirs=(OVERLAY,))
