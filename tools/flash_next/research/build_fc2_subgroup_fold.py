import json
from pathlib import Path

from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once, sha


FLAG = "ORT_WEBGPU_QMOE_FC2_SUBGROUP_FOLD"
PROGRAM = "QmoeFC2SubgroupFoldBitcastF16"
OVERLAY = Path("/workspace/qwen38-int4-qmoe-prefill256/build-v1/overlay/contrib_ops/webgpu/moe")
COMPARATOR = Path("/workspace/qwen38-int4-expert-bitcast-unpack/build-v2/native_compare.cpp")


def transform_program(source):
    start = source.index('         << "if (local_idx < 20u)')
    stop = source.index('         << "workgroupBarrier();\\n"\n'
                        '         << "if (local_idx < 2u)', start)
    original = source[start:stop]
    replacement = ''';
    if (subgroup_fold_) {
      code << "let packed_partial = bitcast<u32>(vec2<f16>(partials[rank * 2u][lane],\\n"
           << "                                                  partials[rank * 2u + 1u][lane]));\\n"
           << "var values = vec2<f16>(0);\\n";
      for (uint32_t source_lane = 0; source_lane < 32; source_lane++) {
        code << "values += bitcast<vec2<f16>>(subgroupBroadcast(packed_partial, "
             << source_lane << "u));\\n";
      }
      code << "if (lane == 0u) {\\n";
      if (bias_) {
        code << "  values += vec2<f16>(bias[experts[rank] * 2560u + base_col],\\n"
             << "                          bias[experts[rank] * 2560u + base_col + 1u]);\\n";
      }
      code << "  partials[rank * 2u][0] = values.x;\\n"
           << "  partials[rank * 2u + 1u][0] = values.y;\\n"
           << "}\\n";
    } else {
      code ''' + original + ''';
    }
    code '''
    source = source[:start] + replacement + source[stop:]
    source = replace_once(source, "QmoeFC2FinalMixProgram(bool bias, bool bitcast)",
                          "QmoeFC2FinalMixProgram(bool bias, bool bitcast, bool subgroup_fold)")
    source = replace_once(source, 'Program{bitcast ? "QmoeFC2FinalMixBitcastF16"',
                          'Program{subgroup_fold ? (bitcast ? "QmoeFC2SubgroupFoldBitcastF16" : '
                          '"QmoeFC2SubgroupFoldF16") : bitcast ? "QmoeFC2FinalMixBitcastF16"')
    source = replace_once(source, "bias_{bias}, bitcast_{bitcast} {}",
                          "bias_{bias}, bitcast_{bitcast}, subgroup_fold_{subgroup_fold} {}")
    return replace_once(source, "  bool bitcast_;", "  bool bitcast_;\n  bool subgroup_fold_;")


def transform(source, program_source=None):
    validation = ('  ORT_RETURN_IF_NOT(prefill1024 == 0 || prefill1024 == 1, '
                  '"ORT_WEBGPU_QMOE_PREFILL1024 must be 0 or 1");')
    assert source.count(validation) == 2
    source = source.replace(validation, validation + f'''
  const int fc2_subgroup_fold = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(fc2_subgroup_fold == 0 || fc2_subgroup_fold == 1, "{FLAG} must be 0 or 1");''')
    source = replace_once(source, "  key.push_back(prefill1024);",
                          "  key.push_back(prefill1024);\n  key.push_back(fc2_subgroup_fold);")
    source = replace_once(source,
        "QmoeFC2FinalMixProgram program{fc2_experts_bias_optional != nullptr, expert_bitcast == 1};",
        '''const bool use_fc2_subgroup_fold = fc2_subgroup_fold == 1 &&
          context.HasFeature(wgpu::FeatureName::Subgroups) &&
          context.AdapterInfo().vendor == std::string_view{"nvidia"};
      QmoeFC2FinalMixProgram program{fc2_experts_bias_optional != nullptr, expert_bitcast == 1,
                                   use_fc2_subgroup_fold};
      if (use_fc2_subgroup_fold && context.HasFeature(wgpu::FeatureName::SubgroupSizeControl)) {
        program.SetSubgroupSize(32);
      }''')
    return replace_once(source, '#include "qmoe_fc2_final_mix.inc"',
                        transform_program(program_source if program_source is not None else
                                          (OVERLAY / "qmoe_fc2_final_mix.inc").read_text()))


def transform_comparator(_source):
    manifest = json.loads((COMPARATOR.parent / "build-manifest.json").read_text())
    assert sha(COMPARATOR) == manifest["source_sha256"][COMPARATOR.name]
    source = COMPARATOR.read_text()
    assert source.count('"ORT_WEBGPU_EXPERT_BITCAST_UNPACK"') == 4
    return source.replace('"ORT_WEBGPU_EXPERT_BITCAST_UNPACK"', f'"{FLAG}"')


if __name__ == "__main__":
    build(output=Path("/workspace/qwen38-int4-fc2-subgroup-fold/build-v1"), flag=FLAG,
          source_transform=transform,
          component_dir=Path("/workspace/qwen38-int4-qmoe-grouped-prefill/build-v1"),
          source_name="qmoe.cc", comparator_flag="ORT_WEBGPU_QMOE_GROUPED_PREFILL",
          comparator_transform=transform_comparator,
          parent_dir=Path("/workspace/qwen38-int4-qmoe-prefill-rows4/build-v2"),
          extra_include_dirs=(OVERLAY,))
