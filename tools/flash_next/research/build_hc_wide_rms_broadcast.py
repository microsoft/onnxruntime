import json
from pathlib import Path

from build_hc_narrow_rms_broadcast import make_program as narrow_broadcast_program
from build_hc_post_norm_vec4 import private_program_transform
from build_hc_projection_inner_unroll import SOURCE
from build_qmoe_node_replay import replace_once
from build_scoped_webgpu_source import build


FLAG = "ORT_WEBGPU_HC_WIDE_RMS_BROADCAST"
CLASS = "HCPostNormProjectionWideRmsBroadcastProgram"
NAME = "HCPostNormProjectionWideRmsBroadcastF16"
OUTPUT = Path("/workspace/qwen38-int4-hc-wide-rms-broadcast/build-v1")
STORAGE_BYTES = 27648


def make_program(source):
    program = narrow_broadcast_program(source)
    program = program.replace("HCPostNormProjectionNarrowRmsBroadcastProgram", CLASS)
    program = replace_once(program, 'Program{"HCPostNormProjectionNarrowRmsBroadcastF16"}',
                           f'Program{{"{NAME}"}}')
    local_loop = "for (var h = local_idx; h < uniforms.hidden; h += 128u)"
    if program.count(local_loop) != 2:
        raise ValueError("Unexpected post-mix/normalization staging loops")
    program = program.replace(local_loop, local_loop.replace("128u", "256u"))
    program = replace_once(program,
        '    code << "for (var virtual_idx = local_idx; virtual_idx < 256u; virtual_idx += 128u) {\\n";\n', "")
    program = replace_once(program,
        '    code << "}\\n";\n    code << "workgroupBarrier();\\n"',
        '    code << "workgroupBarrier();\\n"')
    program = program.replace("virtual_idx", "local_idx")
    program = replace_once(program, "workgroup_idx * 4u + local_idx / 32u",
                           "workgroup_idx * 8u + local_idx / 32u")
    return replace_once(program, "partials: array<f16, 128>", "partials: array<f16, 256>")


def transform(source):
    result = private_program_transform(source, make_program(source), FLAG, "rms_broadcast", CLASS)
    result = replace_once(result,
        f'    const int rms_broadcast = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);\n'
        f'    ORT_RETURN_IF_NOT(rms_broadcast == 0 || rms_broadcast == 1, "{FLAG} must be 0 or 1");',
        f'    static const int selector = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);\n'
        f'    ORT_RETURN_IF_NOT(selector == 0 || selector == 1, "{FLAG} must be 0 or 1");\n'
        '    const int rms_broadcast = selector == 1 && context.HasFeature(wgpu::FeatureName::Subgroups) &&\n'
        '        context.AdapterInfo().vendor == std::string_view{"nvidia"} &&\n'
        '        context.AdapterInfo().subgroupMinSize == 32 && context.AdapterInfo().subgroupMaxSize == 32;')
    return replace_once(result, "}, 0, rms_broadcast);", "});")


if __name__ == "__main__":
    report = build(OUTPUT, SOURCE, transform, SOURCE.parent / "native_compare.cpp", FLAG,
                   dict(diagnostic_only=False, model_eligible=False,
                        physical_threads=256, original_rms_streams=256, output_columns=8,
                        original_rms_tree_preserved=True, shared_rms_widths=[128, 64, 32],
                        subgroup_rms_widths=[16, 8, 4, 2, 1],
                        subgroup_result_broadcast=True, shared_storage_bytes=STORAGE_BYTES,
                        original_replay_api_unchanged=True,
                        original_provider_full_kernel_comparison_required=True))
    report.update(original_compute_and_fallbacks_unchanged=False,
                  original_default_compute_and_fallbacks_unchanged=True)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
