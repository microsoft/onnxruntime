from pathlib import Path

from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_HC_PROJECTION_SHUFFLE"


def shuffled_program(source):
    start = source.index("class HCPostNormProjectionProgram final")
    stop = source.index("class HCPostNormProjectionF16", start)
    program = source[start:stop].replace(
        "HCPostNormProjectionProgram", "HCPostNormProjectionShuffleProgram")
    program = replace_once(program, 'Program{"HCPostNormProjectionF16"}',
                           'Program{"HCPostNormProjectionShuffleF16"}')
    program = replace_once(program, '        "var<workgroup> partials: array<f16, 256>;\\n"\n', "")
    program = replace_once(program, "let lane = local_idx % 32u;", "let lane = sg_id;")
    program = replace_once(program, "let col = workgroup_idx * 8u + local_idx / 32u;",
                           "let col = workgroup_idx * 8u + subgroup_index;")
    original = '''         << "partials[local_idx] = total;\\n"
         << "workgroupBarrier();\\n"
         << "if (lane == 0u && col < 324u) {\\n"
         << "  var value = f16(0);\\n"
         << "  for (var i = 0u; i < 32u; i++) { value += partials[local_idx + i]; }\\n"'''
    replacement = '''         << "let shuffle_total = f32(total);\\n"
         << "var value = f16(0);\\n"
         << "for (var i = 0u; i < 32u; i++) {\\n"
         << "  value += f16(subgroupShuffle(shuffle_total, i));\\n"
         << "}\\n"
         << "if (lane == 0u && col < 324u) {\\n"'''
    return replace_once(program, original, replacement)


def transform(source):
    source = replace_once(source, "class HCPostNormProjectionF16",
                          shuffled_program(source) + "class HCPostNormProjectionF16")
    source = replace_once(source,
        "  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {",
        f'''  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {{
    const int shuffle_mode = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
    ORT_RETURN_IF_NOT(shuffle_mode == 0 || shuffle_mode == 1, "{FLAG} must be 0 or 1");
    const bool projection_shuffle = shuffle_mode == 1 &&
        context.HasFeature(wgpu::FeatureName::Subgroups) &&
        context.AdapterInfo().vendor == std::string_view{{"nvidia"}} &&
        context.AdapterInfo().subgroupMinSize == 32 && context.AdapterInfo().subgroupMaxSize == 32;''')
    start = source.index("      return replay_.Run(context, {y, normalized, activated, tail}, [&]() -> Status {")
    stop = source.index("\n    }\n    HCPostNormProjectionBaselineNormProgram", start)
    block = source[start:stop]
    setup = block[block.index("        program.CacheHint"):block.index("      });")]
    replacement = (
        "      auto run = [&]<typename P>(P& program) -> Status {\n" + setup +
        "      };\n"
        "      return replay_.Run(context, {y, normalized, activated, tail}, [&]() -> Status {\n"
        "        if (projection_shuffle) {\n"
        "          HCPostNormProjectionShuffleProgram program{shared_, shared_scale};\n"
        "          return run(program);\n"
        "        }\n"
        "        HCPostNormProjectionProgram program{shared_, shared_scale};\n"
        "        return run(program);\n"
        "      }, 0, projection_shuffle);")
    return source[:start] + replacement + source[stop:]


if __name__ == "__main__":
    build(output=Path("/workspace/qwen38-int4-hc-projection-shuffle/build-v1"),
          flag=FLAG, source_transform=transform,
          parent_dir=Path("/workspace/qwen38-int4-qmoe-prefill512/build-v3"))
