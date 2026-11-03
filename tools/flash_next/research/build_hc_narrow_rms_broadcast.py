from pathlib import Path

from build_hc_narrow4 import narrow_program
from build_hc_post_norm_vec4 import private_program_transform
from build_hc_projection_inner_unroll import SOURCE
from build_qmoe_node_replay import replace_once
from build_scoped_webgpu_source import build


FLAG = "ORT_WEBGPU_HC_NARROW_RMS_BROADCAST"
CLASS = "HCPostNormProjectionNarrowRmsBroadcastProgram"
NAME = "HCPostNormProjectionNarrowRmsBroadcastF16"
OUTPUT = Path("/workspace/qwen38-int4-hc-narrow-rms-broadcast/build-v1")


def make_program(source):
    program = narrow_program(source)
    program = program.replace("HCPostNormProjectionNarrow4Program", CLASS)
    program = replace_once(program, 'Program{"HCPostNormProjectionNarrow4F16"}',
                           f'Program{{"{NAME}"}}')
    program = replace_once(
        program, '"for (var width = 128u; width > 0u; width >>= 1u) {\\n"',
        '"for (var width = 128u; width >= 32u; width >>= 1u) {\\n"')
    broadcast = ""
    for stream in range(4):
        broadcast += (
            f'    code << "var reduced{stream} = sums[{stream * 256}u + sg_id];\\n";\n')
    for width in (16, 8, 4, 2, 1):
        for stream in range(4):
            broadcast += (
                f'    code << "let other{stream}_{width} = subgroupShuffle(reduced{stream}, '
                f'(sg_id + {width}u) & 31u);\\n";\n')
        broadcast += f'    code << "if (sg_id < {width}u) {{\\n";\n'
        for stream in range(4):
            broadcast += f'    code << "reduced{stream} += other{stream}_{width};\\n";\n'
        broadcast += '    code << "}\\n";\n'
    for stream in range(4):
        broadcast += (
            f'    code << "let rms_total{stream} = subgroupShuffle(reduced{stream}, 0u);\\n";\n')
    marker = '    code << "}\\nworkgroupBarrier();\\n}\\n";\n'
    program = replace_once(program, marker, marker + broadcast)
    program = replace_once(
        program,
        '      code << "let inv_rms" << stream << " = inverseSqrt(sums[" << stream * 256\n'
        '           << "u] / f32(uniforms.hidden) + uniforms.epsilon);\\n";',
        '      code << "let inv_rms" << stream << " = inverseSqrt(rms_total" << stream\n'
        '           << " / f32(uniforms.hidden) + uniforms.epsilon);\\n";')
    return program


def transform(source):
    result = private_program_transform(source, make_program(source), FLAG, "rms_broadcast", CLASS)
    result = replace_once(
        result,
        f'    const int rms_broadcast = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);\n'
        f'    ORT_RETURN_IF_NOT(rms_broadcast == 0 || rms_broadcast == 1, "{FLAG} must be 0 or 1");',
        f'    static const int selector = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);\n'
        f'    ORT_RETURN_IF_NOT(selector == 0 || selector == 1, "{FLAG} must be 0 or 1");\n'
        '    const int rms_broadcast = selector == 1 && context.HasFeature(wgpu::FeatureName::Subgroups) &&\n'
        '        context.AdapterInfo().vendor == std::string_view{"nvidia"} &&\n'
        '        context.AdapterInfo().subgroupMinSize == 32 && context.AdapterInfo().subgroupMaxSize == 32;')
    return replace_once(
        result, ".SetWorkgroupSize(256).SetDispatchGroupSize(41);",
        ".SetWorkgroupSize(rms_broadcast == 1 ? 128 : 256)"
        ".SetDispatchGroupSize(rms_broadcast == 1 ? 81 : 41);")


if __name__ == "__main__":
    build(OUTPUT, SOURCE, transform, SOURCE.parent / "native_compare.cpp", FLAG,
          dict(diagnostic_only=False, model_eligible=False,
               physical_threads=128, original_rms_streams=256, output_columns=4,
               original_rms_tree_preserved=True, shared_rms_widths=[128, 64, 32],
               subgroup_rms_widths=[16, 8, 4, 2, 1],
               subgroup_result_broadcast=True, shared_storage_bytes=27392,
               original_provider_full_kernel_comparison_required=True))
