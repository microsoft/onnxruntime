import json
from pathlib import Path

from build_hc_post_norm_vec4 import private_program_transform
from build_hc_projection_inner_unroll import SOURCE, program_parts
from build_qmoe_node_replay import replace_once
from build_scoped_webgpu_source import build


FLAG = "ORT_WEBGPU_HC_PARALLEL16"
CLASS = "HCPostNormProjectionParallel16Program"
NAME = "HCPostNormProjectionParallel16F16"
OUTPUT = Path("/workspace/qwen38-int4-hc-parallel16/build-v1")
STORAGE_BYTES = 4096 + 11520 * 2 + 512 * 2


def make_program(source):
    _, _, program = program_parts(source)
    program = program.replace("HCPostNormProjectionProgram", CLASS)
    program = replace_once(program, 'Program{"HCPostNormProjectionF16"}', f'Program{{"{NAME}"}}')
    program = replace_once(program, "partials: array<f16, 256>", "partials: array<f16, 512>")
    loop = '    code << "for (var h = local_idx; h < uniforms.hidden; h += 256u) {\\n";'
    assert program.count(loop) == 3
    program = program.replace(loop,
        '    code << "if (local_idx < 256u) {\\n"\n'
        '         << "for (var h = local_idx; h < uniforms.hidden; h += 256u) {\\n";')
    closing = '    code << "}\\nworkgroupBarrier();\\n"'
    assert program.count(closing) == 2
    program = program.replace(closing, '    code << "}\\n}\\nworkgroupBarrier();\\n"')
    program = replace_once(program,
        '    code << "workgroupBarrier();\\n"\n'
        '         << "for (var width = 128u;',
        '    code << "}\\nworkgroupBarrier();\\n"\n'
        '         << "for (var width = 128u;')
    return replace_once(program,
        '"let col = workgroup_idx * 8u + local_idx / 32u;\\n"',
        '"let col = workgroup_idx * 16u + local_idx / 32u;\\n"')


def transform(source):
    result = private_program_transform(source, make_program(source), FLAG, "parallel16", CLASS)
    result = replace_once(result,
        f'    const int parallel16 = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);\n'
        f'    ORT_RETURN_IF_NOT(parallel16 == 0 || parallel16 == 1, "{FLAG} must be 0 or 1");',
        f'    static const int selector = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);\n'
        f'    ORT_RETURN_IF_NOT(selector == 0 || selector == 1, "{FLAG} must be 0 or 1");')
    result = replace_once(result, "    if (fused) {\n",
        "    if (fused) {\n"
        "      const bool parallel16 = selector == 1 &&\n"
        '          context.AdapterInfo().vendor == std::string_view{"nvidia"} &&\n'
        "          limits.maxComputeWorkgroupSizeX >= 512 &&\n"
        "          limits.maxComputeInvocationsPerWorkgroup >= 512 &&\n"
        f"          limits.maxComputeWorkgroupStorageSize >= {STORAGE_BYTES};\n")
    result = replace_once(result, ".SetWorkgroupSize(256).SetDispatchGroupSize(41);",
        ".SetWorkgroupSize(parallel16 ? 512 : 256).SetDispatchGroupSize(parallel16 ? 21 : 41);")
    return replace_once(result, "}, 0, parallel16);", "});")


if __name__ == "__main__":
    report = build(OUTPUT, SOURCE, transform, SOURCE.parent / "native_compare.cpp", FLAG,
        dict(model_eligible=False, physical_threads=512, output_columns=16,
             original_rms_streams=256, original_rms_tree_preserved=True,
             original_fp16_projection_schedule_preserved=True,
             staging_and_rms_store_threads=256, shared_storage_bytes=STORAGE_BYTES,
             original_replay_api_unchanged=True))
    report.update(original_compute_and_fallbacks_unchanged=False,
                  original_default_compute_and_fallbacks_unchanged=True)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
