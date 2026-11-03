from pathlib import Path

from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_QMOE_FC2_PAIRED_DOT"
PROGRAM = "QmoeFC2PairedDotBitcastF16"
OVERLAY = Path("/workspace/qwen38-int4-qmoe-prefill256/build-v1/overlay/contrib_ops/webgpu/moe")
COMPARATOR = Path("/workspace/qwen38-int4-expert-bitcast-unpack/build-v2/native_compare.cpp")


def transform_program(source):
    start = source.index('         << "  for (var c = 0u; c < 2u; c++)')
    stop = source.index('         << "}\\n"\n         << "workgroupBarrier();\\n"', start)
    original = source[start:stop]
    paired = ''';\n
    if (paired_dot_) {
      code << "  let index0 = (expert * 2560u + base_col) * 20u + lane;\\n"
           << "  let index1 = index0 + 20u;\\n"
           << "  let packed0 = " << b.GetByOffset("index0") << ";\\n"
           << "  let packed1 = " << b.GetByOffset("index1") << ";\\n"
           << "  let scale0 = " << scales.GetByOffset("index0") << ";\\n"
           << "  let scale1 = " << scales.GetByOffset("index1") << ";\\n"
           << "  var sums = vec2<f16>(0);\\n"
           << "  for (var i = 0u; i < 4u; i++) {\\n";
      for (uint32_t column = 0; column < 2; column++) {
        code << "    let low" << column << " = ";
        if (bitcast_) code << "decode_low_nibbles(packed" << column << "[i]);\\n";
        else code << "vec4<f16>(unpack4xU8(packed" << column << "[i] & 0x0f0f0f0fu)) - vec4<f16>(8);\\n";
        code << "    let high" << column << " = ";
        if (bitcast_) code << "decode_low_nibbles(packed" << column << "[i] >> 4u);\\n";
        else code << "vec4<f16>(unpack4xU8((packed" << column << "[i] >> 4u) & 0x0f0f0f0fu)) - vec4<f16>(8);\\n";
        code << "    let w" << column << "0 = vec4<f16>(low" << column << ".x, high" << column
             << ".x, low" << column << ".y, high" << column << ".y) * scale" << column << ";\\n"
             << "    let w" << column << "1 = vec4<f16>(low" << column << ".z, high" << column
             << ".z, low" << column << ".w, high" << column << ".w) * scale" << column << ";\\n";
      }
      code << "    let a0 = tile_a[rank * 160u + lane * 8u + i * 2u];\\n"
           << "    let a1 = tile_a[rank * 160u + lane * 8u + i * 2u + 1u];\\n"
           << "    sums += vec2<f16>(dot(a0, w00) + dot(a1, w01), dot(a0, w10) + dot(a1, w11));\\n"
           << "  }\\n"
           << "  partials[rank * 2u][lane] += sums.x;\\n"
           << "  partials[rank * 2u + 1u][lane] += sums.y;\\n";
    } else {
      code ''' + original + ''';\n
    }
    code '''
    source = source[:start] + paired + source[stop:]
    source = replace_once(source, "QmoeFC2FinalMixProgram(bool bias, bool bitcast)",
                          "QmoeFC2FinalMixProgram(bool bias, bool bitcast, bool paired_dot)")
    source = replace_once(source, 'Program{bitcast ? "QmoeFC2FinalMixBitcastF16"',
                          'Program{paired_dot ? (bitcast ? "QmoeFC2PairedDotBitcastF16" : '
                          '"QmoeFC2PairedDotF16") : bitcast ? "QmoeFC2FinalMixBitcastF16"')
    source = replace_once(source, "bias_{bias}, bitcast_{bitcast} {}",
                          "bias_{bias}, bitcast_{bitcast}, paired_dot_{paired_dot} {}")
    return replace_once(source, "  bool bitcast_;", "  bool bitcast_;\n  bool paired_dot_;")


def transform(source):
    validation = ('  ORT_RETURN_IF_NOT(prefill1024 == 0 || prefill1024 == 1, '
                  '"ORT_WEBGPU_QMOE_PREFILL1024 must be 0 or 1");')
    assert source.count(validation) == 2
    source = source.replace(validation, validation + f'''
  const int fc2_paired_dot = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(fc2_paired_dot == 0 || fc2_paired_dot == 1, "{FLAG} must be 0 or 1");''')
    source = replace_once(source, "  key.push_back(prefill1024);",
                          "  key.push_back(prefill1024);\n  key.push_back(fc2_paired_dot);")
    source = replace_once(source,
        "QmoeFC2FinalMixProgram program{fc2_experts_bias_optional != nullptr, expert_bitcast == 1};",
        "QmoeFC2FinalMixProgram program{fc2_experts_bias_optional != nullptr, expert_bitcast == 1, fc2_paired_dot == 1};")
    return replace_once(source, '#include "qmoe_fc2_final_mix.inc"',
                        transform_program((OVERLAY / "qmoe_fc2_final_mix.inc").read_text()))


def transform_comparator(_source):
    source = COMPARATOR.read_text()
    assert source.count('"ORT_WEBGPU_EXPERT_BITCAST_UNPACK"') == 4
    return source.replace('"ORT_WEBGPU_EXPERT_BITCAST_UNPACK"', f'"{FLAG}"')


if __name__ == "__main__":
    build(output=Path("/workspace/qwen38-int4-fc2-paired-dot/build-v1"), flag=FLAG,
          source_transform=transform,
          component_dir=Path("/workspace/qwen38-int4-qmoe-prefill1024/build-v1"),
          source_name="qmoe.cc", comparator_flag="ORT_WEBGPU_QMOE_PREFILL1024",
          comparator_transform=transform_comparator,
          parent_dir=Path("/workspace/qwen38-int4-qsa-prefill-score16/build-v1"),
          extra_include_dirs=(OVERLAY,))
