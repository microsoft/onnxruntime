from build_hc_post_norm_vec4 import private_program_transform
from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_HC_TILE_PARALLEL4"
CLASS = "HCPostNormProjectionTileParallel4Program"
NAME = "HCPostNormProjectionTileParallel4F16"
SHARED_BYTES = 37888


def make_program(source):
    start = source.index("class HCPostNormProjectionProgram final")
    stop = source.index("class HCPostNormProjectionF16", start)
    program = source[start:stop].replace("HCPostNormProjectionProgram", CLASS)
    program = replace_once(program, 'Program{"HCPostNormProjectionF16"}',
                           f'Program{{"{NAME}"}}')
    program = replace_once(
        program, '"var<workgroup> partials: array<f16, 256>;\\n"',
        '"var<workgroup> partials: array<f16, 256>;\\n"\n'
        '        "var<workgroup> tile_sums: array<f32, 2560>;\\n"')
    loop = "for (var h = local_idx; h < uniforms.hidden; h += 256u)"
    if program.count(loop) != 3:
        raise RuntimeError("Unexpected original post/RMS/normalization writer loops")
    program = program.replace(
        loop, "for (var h = local_idx; local_idx < 256u && h < uniforms.hidden; h += 256u)")
    program = replace_once(
        program,
        '    for (int stream = 0; stream < 4; ++stream) {\n'
        '      code << "sums[" << stream * 256 << "u + local_idx] = sum" << stream << ";\\n";\n'
        '    }\n'
        '    code << "workgroupBarrier();\\n"',
        '    code << "if (local_idx < 256u) {\\n";\n'
        '    for (int stream = 0; stream < 4; ++stream) {\n'
        '      code << "sums[" << stream * 256 << "u + local_idx] = sum" << stream << ";\\n";\n'
        '    }\n'
        '    code << "}\\nworkgroupBarrier();\\n"')
    program = replace_once(program, '         << "  var sum = f16(0);\\n"\n', "")
    program = replace_once(
        program,
        '         << "let col = workgroup_idx * 8u + local_idx / 32u;\\n"\n'
        '         << "var total = f16(0);\\n"\n'
        '         << "if (col < 324u) {\\n"\n'
        '         << "for (var tile = 0u; tile < 10u; tile++) {\\n"',
        '         << "let logical_lane = local_idx % 256u;\\n"\n'
        '         << "let col = workgroup_idx * 8u + logical_lane / 32u;\\n"\n'
        '         << "let tile_worker = local_idx / 256u;\\n"\n'
        '         << "for (var tile = tile_worker; tile < 10u; tile += 4u) {\\n"\n'
        '         << "  var sum = f16(0);\\n"\n'
        '         << "if (col < 324u) {\\n"')
    program = replace_once(
        program,
        '         << "  total += sum;\\n"\n'
        '         << "}\\n}\\n"\n'
        '         << "partials[local_idx] = total;\\n"\n'
        '         << "workgroupBarrier();\\n"',
        '         << "}\\n"\n'
        '         << "tile_sums[tile * 256u + logical_lane] = f32(sum);\\n"\n'
        '         << "}\\nworkgroupBarrier();\\n"\n'
        '         << "if (local_idx < 256u) {\\n"\n'
        '         << "var total = f16(0);\\n"\n'
        '         << "for (var tile = 0u; tile < 10u; tile++) {\\n"\n'
        '         << "  total += f16(tile_sums[tile * 256u + local_idx]);\\n"\n'
        '         << "}\\npartials[local_idx] = total;\\n}\\n"\n'
        '         << "workgroupBarrier();\\n"')
    return replace_once(
        program, "if (lane == 0u && col < 324u)",
        "if (local_idx < 256u && lane == 0u && col < 324u)")


def transform(source):
    result = private_program_transform(
        source, make_program(source), FLAG, "tile_parallel4", CLASS)
    result = replace_once(
        result,
        f'    const int tile_parallel4 = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);\n'
        f'    ORT_RETURN_IF_NOT(tile_parallel4 == 0 || tile_parallel4 == 1, "{FLAG} must be 0 or 1");',
        f'    static const int selector = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);\n'
        f'    ORT_RETURN_IF_NOT(selector == 0 || selector == 1, "{FLAG} must be 0 or 1");\n'
        '    const int tile_parallel4 = selector == 1 &&\n'
        '        context.AdapterInfo().vendor == std::string_view{"nvidia"} &&\n'
        '        context.DeviceLimits().maxComputeWorkgroupSizeX >= 1024 &&\n'
        '        context.DeviceLimits().maxComputeInvocationsPerWorkgroup >= 1024 &&\n'
        f'        context.DeviceLimits().maxComputeWorkgroupStorageSize >= {SHARED_BYTES};')
    result = replace_once(
        result, ".SetWorkgroupSize(256).SetDispatchGroupSize(41);",
        ".SetWorkgroupSize(tile_parallel4 == 1 ? 1024 : 256).SetDispatchGroupSize(41);")
    return replace_once(result, "}, 0, tile_parallel4);", "});")
