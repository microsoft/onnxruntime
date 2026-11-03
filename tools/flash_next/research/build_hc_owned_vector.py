import json
from pathlib import Path

from build_hc_post_norm_vec4 import private_program_transform
from build_hc_projection_inner_unroll import SOURCE, program_parts
from build_qmoe_node_replay import replace_once
from build_scoped_webgpu_source import build


FLAG = "ORT_WEBGPU_HC_OWNED_VECTOR"
CLASS = "HCPostNormProjectionOwnedVectorProgram"
NAME = "HCPostNormProjectionOwnedVectorF16"
OUTPUT = Path("/workspace/qwen38-int4-hc-owned-vector/build-v1")


def owned_stage(program, normalized):
    loop = '    code << "for (var h = local_idx; h < uniforms.hidden; h += 256u) {\\n";'
    start = program.rindex(loop) if normalized else program.index(loop)
    body_start = start + len(loop)
    closing = '    code << "}\\nworkgroupBarrier();\\n"'
    end = program.index(closing, body_start)
    body = program[body_start:end]
    value = "value" if normalized else "stored"
    body = replace_once(body,
        f'"post_values[address" << stream << "] = {value}" << stream << ";\\n"',
        f'"quad_value" << stream << "[component] = {value}" << stream << ";\\n"')
    replacement = '''    code << "for (var quad = local_idx; quad < uniforms.hidden / 4u; quad += 256u) {\\n";
    for (int stream = 0; stream < 4; ++stream) {
      code << "var quad_value" << stream << " = vec4<f16>(0);\\n";
    }
    code << "for (var component = 0u; component < 4u; component++) {\\n"
         << "let h = quad * 4u + component;\\n";
''' + body + '''    code << "}\\n";
    for (int stream = 0; stream < 4; ++stream) {
      code << "let logical_vector" << stream << " = " << stream << "u * (uniforms.hidden / 4u) + quad;\\n"
           << "let vector_address" << stream << " = (logical_vector" << stream
           << " / 8u) * 9u + logical_vector" << stream << " % 8u;\\n"
           << "post_values[vector_address" << stream << "] = quad_value" << stream << ";\\n";
    }
''' + closing
    return program[:start] + replacement + program[end + len(closing):]


def make_program(source):
    _, _, program = program_parts(source)
    program = program.replace("HCPostNormProjectionProgram", CLASS)
    program = replace_once(program, 'Program{"HCPostNormProjectionF16"}', f'Program{{"{NAME}"}}')
    program = replace_once(program, '"var<workgroup> post_values: array<f16, 11520>;\\n"',
        '"var<workgroup> post_values: array<vec4<f16>, 2880>;\\n"\n'
        '        "fn load_post(index: u32) -> f16 { return post_values[index / 4u][index % 4u]; }\\n"')
    program = replace_once(program,
        '"let v" << stream << " = f32(post_values[address" << stream << "]);\\n"',
        '"let v" << stream << " = f32(load_post(address" << stream << "));\\n"')
    program = replace_once(program,
        '"let value" << stream << " = f16(f32(post_values[address" << stream << "]) * inv_rms"',
        '"let value" << stream << " = f16(f32(load_post(address" << stream << ")) * inv_rms"')
    program = owned_stage(program, True)
    program = owned_stage(program, False)
    program = replace_once(program,
        '"    let x0 = vec4<f16>(post_values[base], post_values[base + 1u], post_values[base + 2u], post_values[base + 3u]);\\n"',
        '"    let x0 = post_values[base / 4u];\\n"')
    return replace_once(program,
        '"    let x1 = vec4<f16>(post_values[base + 4u], post_values[base + 5u], post_values[base + 6u], post_values[base + 7u]);\\n"',
        '"    let x1 = post_values[base / 4u + 1u];\\n"')


def transform(source):
    result = private_program_transform(source, make_program(source), FLAG, "owned_vector", CLASS)
    result = replace_once(result,
        f'    const int owned_vector = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);\n'
        f'    ORT_RETURN_IF_NOT(owned_vector == 0 || owned_vector == 1, "{FLAG} must be 0 or 1");',
        f'    static const int selector = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);\n'
        f'    ORT_RETURN_IF_NOT(selector == 0 || selector == 1, "{FLAG} must be 0 or 1");\n'
        '    const int owned_vector = selector == 1 && context.AdapterInfo().vendor == std::string_view{"nvidia"};')
    return replace_once(result, "}, 0, owned_vector);", "});")


if __name__ == "__main__":
    report = build(OUTPUT, SOURCE, transform, SOURCE.parent / "native_compare.cpp", FLAG,
        dict(model_eligible=False, physical_threads=256, output_columns=8,
             original_rms_streams=256, original_rms_tree_preserved=True,
             original_fp16_projection_schedule_preserved=True,
             single_owner_vector_stores=True, shared_storage_bytes=27648,
             original_replay_api_unchanged=True))
    report.update(original_compute_and_fallbacks_unchanged=False,
                  original_default_compute_and_fallbacks_unchanged=True)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
