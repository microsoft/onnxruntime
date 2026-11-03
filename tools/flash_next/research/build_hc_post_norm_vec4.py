from pathlib import Path

from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_HC_POST_NORM_VEC4"


def vector_program(source):
    start = source.index("class HCPostNormProjectionProgram")
    stop = source.index("class HCPostNormProjectionF16", start)
    program = source[start:stop].replace("HCPostNormProjectionProgram", "HCPostNormProjectionVec4Program")
    replacements = (
        ('Program{"HCPostNormProjectionF16"}', 'Program{"HCPostNormProjectionVec4F16"}'),
        ('"var<workgroup> post_values: array<f16, 11520>;\\n"',
         '"var<workgroup> post_values: array<vec4<f16>, 2880>;\\n"\n'
         '        "fn load_post(index: u32) -> f16 { return post_values[index / 4u][index % 4u]; }\\n"\n'
         '        "fn store_post(index: u32, value: f16) { post_values[index / 4u][index % 4u] = value; }\\n"'),
        ('"post_values[address" << stream << "] = stored" << stream << ";\\n"',
         '"store_post(address" << stream << ", stored" << stream << ");\\n"'),
        ('"let v" << stream << " = f32(post_values[address" << stream << "]);\\n"',
         '"let v" << stream << " = f32(load_post(address" << stream << "));\\n"'),
        ('"let value" << stream << " = f16(f32(post_values[address" << stream << "]) * inv_rms"',
         '"let value" << stream << " = f16(f32(load_post(address" << stream << ")) * inv_rms"'),
        ('"post_values[address" << stream << "] = value" << stream << ";\\n"',
         '"store_post(address" << stream << ", value" << stream << ");\\n"'),
        ('"    let x0 = vec4<f16>(post_values[base], post_values[base + 1u], post_values[base + 2u], post_values[base + 3u]);\\n"',
         '"    let x0 = post_values[base / 4u];\\n"'),
        ('"    let x1 = vec4<f16>(post_values[base + 4u], post_values[base + 5u], post_values[base + 6u], post_values[base + 7u]);\\n"',
         '"    let x1 = post_values[base / 4u + 1u];\\n"'),
    )
    for original, replacement in replacements:
        program = replace_once(program, original, replacement)
    return program


def transform(source):
    return private_program_transform(source, vector_program(source), FLAG, "vec4_mode",
                                     "HCPostNormProjectionVec4Program")


def private_program_transform(source, program_source, flag, mode_variable, class_name):
    source = replace_once(source, "class HCPostNormProjectionF16",
                          program_source + "class HCPostNormProjectionF16")
    source = replace_once(source,
        "  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {",
        f'''  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {{
    const int {mode_variable} = ParseEnvironmentVariableWithDefault<int>("{flag}", 0);
    ORT_RETURN_IF_NOT({mode_variable} == 0 || {mode_variable} == 1, "{flag} must be 0 or 1");''')
    start = source.index("      return replay_.Run(context, {y, normalized, activated, tail}, [&]() -> Status {")
    stop = source.index("\n    }\n    HCPostNormProjectionBaselineNormProgram", start)
    block = source[start:stop]
    setup = block[block.index("        program.CacheHint"):block.index("      });")]
    replacement = (
        "      auto run = [&]<typename P>(P& program) -> Status {\n" + setup +
        "      };\n"
        "      return replay_.Run(context, {y, normalized, activated, tail}, [&]() -> Status {\n"
        f"        if ({mode_variable} == 1) {{\n"
        f"          {class_name} program{{shared_, shared_scale}};\n"
        "          return run(program);\n"
        "        }\n"
        "        HCPostNormProjectionProgram program{shared_, shared_scale};\n"
        "        return run(program);\n"
        f"      }}, 0, {mode_variable});")
    return source[:start] + replacement + source[stop:]


if __name__ == "__main__":
    build(output=Path("/workspace/qwen38-int4-hc-post-norm-vec4/build-v1"),
          flag=FLAG, source_transform=transform,
          parent_dir=Path("/workspace/qwen38-int4-dense-block32/build-v2"))
