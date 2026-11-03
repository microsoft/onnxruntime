from pathlib import Path

from build_fc2_subgroup_fold import FLAG as FOLD_FLAG, OVERLAY, transform_comparator as fold_comparator
from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_QMOE_FC1_PAIRED_DOT"
PROGRAM = "QmoeFC1PairedDotBitcastF16"


def transform_program(source):
    start = source.index('         << "  for (var offset = 0u; offset < 8u; offset += 4u)')
    stop = source.index('         << "  workgroupBarrier();\\n"\n         << "}\\n"', start)
    original = source[start:stop]
    paired = ''';
    if (paired_dot_) {
      code << "  let col0 = base_col + group;\\n"
           << "  let col1 = col0 + 4u;\\n"
           << "  let block = kidx / 32u + lane;\\n"
           << "  if (col1 < uniforms.cols * 2u && block < blocks) {\\n"
           << "    let index0 = (expert * uniforms.cols * 2u + col0) * blocks + block;\\n"
           << "    let index1 = index0 + 4u * blocks;\\n"
           << "    let packed0 = " << b.GetByOffset("index0") << ";\\n"
           << "    let packed1 = " << b.GetByOffset("index1") << ";\\n"
           << "    let scale0 = " << scales.GetByOffset("index0") << ";\\n"
           << "    let scale1 = " << scales.GetByOffset("index1") << ";\\n"
           << "    var sums = vec2<f16>(0);\\n"
           << "    for (var i = 0u; i < 4u; i++) {\\n";
      for (uint32_t column = 0; column < 2; column++) {
        code << "      let low" << column << " = ";
        if (bitcast_) code << "decode_low_nibbles(packed" << column << "[i]);\\n";
        else code << "vec4<f16>(unpack4xU8(packed" << column << "[i] & 0x0f0f0f0fu)) - vec4<f16>(8);\\n";
        code << "      let high" << column << " = ";
        if (bitcast_) code << "decode_low_nibbles(packed" << column << "[i] >> 4u);\\n";
        else code << "vec4<f16>(unpack4xU8((packed" << column << "[i] >> 4u) & 0x0f0f0f0fu)) - vec4<f16>(8);\\n";
        code << "      let w" << column << "0 = vec4<f16>(low" << column << ".x, high" << column
             << ".x, low" << column << ".y, high" << column << ".y) * scale" << column << ";\\n"
             << "      let w" << column << "1 = vec4<f16>(low" << column << ".z, high" << column
             << ".z, low" << column << ".w, high" << column << ".w) * scale" << column << ";\\n";
      }
      code << "      let a0 = tile_a[lane * 8u + i * 2u];\\n"
           << "      let a1 = tile_a[lane * 8u + i * 2u + 1u];\\n"
           << "      sums += vec2<f16>(dot(a0, w00) + dot(a1, w01), dot(a0, w10) + dot(a1, w11));\\n"
           << "    }\\n"
           << "    partials[group][lane] += sums.x;\\n"
           << "    partials[group + 4u][lane] += sums.y;\\n"
           << "  }\\n";
    } else {
      code ''' + original + ''';
    }
    code '''
    source = source[:start] + paired + source[stop:]
    source = replace_once(source, "QmoeFC1SwiGLUProgram(bool bias, bool bitcast)",
                          "QmoeFC1SwiGLUProgram(bool bias, bool bitcast, bool paired_dot)")
    source = replace_once(source, 'Program{bitcast ? "QmoeFC1SwiGLUBitcastF16"',
                          'Program{paired_dot ? (bitcast ? "QmoeFC1PairedDotBitcastF16" : '
                          '"QmoeFC1PairedDotF16") : bitcast ? "QmoeFC1SwiGLUBitcastF16"')
    source = replace_once(source, "bias_{bias}, bitcast_{bitcast} {}",
                          "bias_{bias}, bitcast_{bitcast}, paired_dot_{paired_dot} {}")
    return replace_once(source, "  bool bitcast_;", "  bool bitcast_;\n  bool paired_dot_;")


def transform(source, program_source=None):
    validation = ('  ORT_RETURN_IF_NOT(prefill1024 == 0 || prefill1024 == 1, '
                  '"ORT_WEBGPU_QMOE_PREFILL1024 must be 0 or 1");')
    assert source.count(validation) == 2
    source = source.replace(validation, validation + f'''
  const int fc1_paired_dot = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(fc1_paired_dot == 0 || fc1_paired_dot == 1, "{FLAG} must be 0 or 1");''')
    source = replace_once(source, "  key.push_back(prefill1024);",
                          "  key.push_back(prefill1024);\n  key.push_back(fc1_paired_dot);")
    source = replace_once(source,
        "QmoeFC1SwiGLUProgram program{fc1_experts_bias_optional != nullptr, expert_bitcast == 1};",
        """QmoeFC1SwiGLUProgram program{fc1_experts_bias_optional != nullptr, expert_bitcast == 1,
                                 fc1_paired_dot == 1 && moe_params.inter_size == 640};""")
    return replace_once(source, '#include "qmoe_fc1_activation.inc"',
                        transform_program(program_source if program_source is not None else
                                          (OVERLAY / "qmoe_fc1_activation.inc").read_text()))


def transform_comparator(source):
    return fold_comparator(source).replace(f'"{FOLD_FLAG}"', f'"{FLAG}"')


if __name__ == "__main__":
    build(output=Path("/workspace/qwen38-int4-fc1-paired-dot/build-v1"), flag=FLAG,
          source_transform=transform,
          component_dir=Path("/workspace/qwen38-int4-qmoe-grouped-prefill/build-v1"),
          source_name="qmoe.cc", comparator_flag="ORT_WEBGPU_QMOE_GROUPED_PREFILL",
          comparator_transform=transform_comparator,
          parent_dir=Path("/workspace/qwen38-int4-qmoe-prefill-rows4/build-v2"),
          extra_include_dirs=(OVERLAY,))
