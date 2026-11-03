from pathlib import Path

from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_QMOE_FC1_WARP_FOLD"
PROGRAM = "QmoeFC1WarpFoldBitcastF16"
ROOT = Path(__file__).parent
COMPONENT = Path("/workspace/qwen38-int4-qmoe-prefill1024/build-v1")
PARENT = Path("/workspace/qwen38-int4-qsa-prefill-score16/build-v1")
OVERLAY = Path("/workspace/qwen38-int4-qmoe-prefill256/build-v1/overlay/contrib_ops/webgpu/moe")
COMPARATOR = Path("/workspace/qwen38-int4-expert-bitcast-unpack/build-v2/native_compare.cpp")


def transform_program(source, vector_fold=False):
    source = replace_once(source, "QmoeFC1SwiGLUProgram(bool bias, bool bitcast)",
                          "QmoeFC1SwiGLUProgram(bool bias, bool bitcast, bool warp_fold)")
    source = replace_once(source, 'Program{bitcast ? "QmoeFC1SwiGLUBitcastF16"',
                          'Program{warp_fold ? (bitcast ? "QmoeFC1WarpFoldBitcastF16" : '
                          '"QmoeFC1WarpFoldF16") : bitcast ? "QmoeFC1SwiGLUBitcastF16"')
    source = replace_once(source, "bias_{bias}, bitcast_{bitcast} {}",
                          "bias_{bias}, bitcast_{bitcast}, warp_fold_{warp_fold} {}")
    start = source.index('         << "if (local_idx < 8u && base_col + local_idx')
    stop = source.index('         << "workgroupBarrier();\\n"', start)
    original_fold = source[start:stop]
    replacement = ''';\n
    if (warp_fold_) {
      code << "if (lane == 0u) {\\n"
           << "  for (var offset = 0u; offset < 8u; offset += 4u) {\\n"
           << "    let fold_col = offset + group;\\n"
           << "    if (base_col + fold_col < uniforms.cols * 2u) {\\n"
           << "      var value = f16(0);\\n"
           << "      for (var i = 0u; i < 32u; i++) { value += partials[fold_col][i]; }\\n";
      if (bias_) code << "      value += bias[expert * uniforms.cols * 2u + base_col + fold_col];\\n";
      code << "      partials[fold_col][0] = value;\\n"
           << "    }\\n"
           << "  }\\n"
           << "}\\n";
    } else {
      code ''' + original_fold + ''';\n
    }
    code '''
    if vector_fold:
        replacement = ''';\n
    if (warp_fold_) {
      code << "if (lane == 0u) {\\n"
           << "  let fold_col = group * 2u;\\n"
           << "  if (base_col + fold_col + 1u < uniforms.cols * 2u) {\\n"
           << "    var values = vec2<f16>(0);\\n"
           << "    for (var i = 0u; i < 32u; i++) {\\n"
           << "      values += vec2<f16>(partials[fold_col][i], partials[fold_col + 1u][i]);\\n"
           << "    }\\n";
      if (bias_) code << "    values += vec2<f16>(bias[expert * uniforms.cols * 2u + base_col + fold_col], "
                        "bias[expert * uniforms.cols * 2u + base_col + fold_col + 1u]);\\n";
      code << "    partials[fold_col][0] = values.x;\\n"
           << "    partials[fold_col + 1u][0] = values.y;\\n"
           << "  }\\n"
           << "}\\n";
    } else {
      code ''' + original_fold + ''';\n
    }
    code '''
    source = source[:start] + replacement + source[stop:]
    if vector_fold:
        source = source.replace("QmoeFC1WarpFoldBitcastF16", "QmoeFC1WarpFoldVec2BitcastF16").replace(
            '"QmoeFC1WarpFoldF16"', '"QmoeFC1WarpFoldVec2F16"')
    return replace_once(source, "  bool bitcast_;", "  bool bitcast_;\n  bool warp_fold_;")


def transform(source, vector_fold=False):
    validation = ('  ORT_RETURN_IF_NOT(prefill1024 == 0 || prefill1024 == 1, '
                  '"ORT_WEBGPU_QMOE_PREFILL1024 must be 0 or 1");')
    assert source.count(validation) == 2
    source = source.replace(validation, validation + f'''
  const int fc1_warp_fold = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(fc1_warp_fold == 0 || fc1_warp_fold == 1, "{FLAG} must be 0 or 1");''')
    source = replace_once(source, "  key.push_back(prefill1024);",
                          "  key.push_back(prefill1024);\n  key.push_back(fc1_warp_fold);")
    source = replace_once(source,
        "QmoeFC1SwiGLUProgram program{fc1_experts_bias_optional != nullptr, expert_bitcast == 1};",
        "QmoeFC1SwiGLUProgram program{fc1_experts_bias_optional != nullptr, expert_bitcast == 1, fc1_warp_fold == 1};")
    return replace_once(source, '#include "qmoe_fc1_activation.inc"',
                        transform_program((OVERLAY / "qmoe_fc1_activation.inc").read_text(), vector_fold))


def transform_comparator(_source):
    source = COMPARATOR.read_text()
    assert source.count('"ORT_WEBGPU_EXPERT_BITCAST_UNPACK"') == 4
    return source.replace('"ORT_WEBGPU_EXPERT_BITCAST_UNPACK"', f'"{FLAG}"')


def main(output=Path("/workspace/qwen38-int4-fc1-warp-fold/build-v1"), vector_fold=False):
    build(output=output, flag=FLAG,
          source_transform=lambda source: transform(source, vector_fold),
          component_dir=COMPONENT, source_name="qmoe.cc",
          comparator_flag="ORT_WEBGPU_QMOE_PREFILL1024", comparator_transform=transform_comparator,
          parent_dir=PARENT, extra_include_dirs=(OVERLAY,))


if __name__ == "__main__":
    main()
