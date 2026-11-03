import json
from pathlib import Path

from build_hc_projection_inner_unroll import SOURCE
from build_qmoe_node_replay import replace_once
from build_scoped_webgpu_source import build


FLAG = "ORT_WEBGPU_HC_SEPARATE8"
OUTPUT = Path("/workspace/qwen38-int4-hc-separate8/build-v1")
NORM_CLASS = "HCSeparate8NormProgram"
PROJECTION_CLASS = "HCSeparate8ProjectionProgram"


def programs(source):
    start = source.index("class HCPostNormProjectionBaselineNormProgram")
    middle = source.index("class HCPostNormProjectionBaselineMatMulProgram", start)
    end = source.index("class HCPostNormProjectionProgram final", middle)
    norm = source[start:middle].replace("HCPostNormProjectionBaselineNormProgram", NORM_CLASS)
    norm = replace_once(norm, 'Program{"HCPostNormF16"}', 'Program{"HCSeparate8NormF16"}')
    projection = source[middle:end].replace("HCPostNormProjectionBaselineMatMulProgram", PROJECTION_CLASS)
    projection = replace_once(projection, 'Program{"HCProjectionSiLUF16"}',
                              'Program{"HCSeparate8ProjectionF16"}')
    projection = replace_once(projection, "partials: array<f16, 64>", "partials: array<f16, 256>")
    projection = replace_once(projection, "workgroup_idx * 2u + local_idx / 32u",
                              "workgroup_idx * 8u + local_idx / 32u")
    projection = replace_once(projection, '         << "for (var tile = 0u;',
                              '         << "if (col < 324u) {\\n"\n'
                              '         << "for (var tile = 0u;')
    projection = replace_once(projection,
        '         << "}\\n"\n         << "partials[local_idx] = total;',
        '         << "}\\n}\\n"\n         << "partials[local_idx] = total;')
    projection = replace_once(projection, "if (lane == 0u) {", "if (lane == 0u && col < 324u) {")
    return norm, projection


def transform(source):
    norm, projection = programs(source)
    result = replace_once(source, "class HCPostNormProjectionF16", norm + projection + "class HCPostNormProjectionF16")
    signature = "  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {"
    result = replace_once(result, signature, signature +
        f'\n    static const int separate8 = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);\n'
        f'    ORT_RETURN_IF_NOT(separate8 == 0 || separate8 == 1, "{FLAG} must be 0 or 1");')
    start = source.index("    HCPostNormProjectionBaselineNormProgram norm{")
    end = source.index("    auto intermediate =", start)
    setup = source[start:end]
    norm_setup, projection_setup = setup.split("    if (projection_fused) {\n")
    norm_setup = norm_setup.replace("HCPostNormProjectionBaselineNormProgram", NORM_CLASS)
    projection_setup = projection_setup.replace("HCPostNormProjectionBaselineMatMulProgram", PROJECTION_CLASS)
    assert projection_setup.endswith("    }\n")
    projection_setup = projection_setup[:-6]
    projection_setup = replace_once(projection_setup, ".SetWorkgroupSize(64).SetDispatchGroupSize(162);",
                                    ".SetWorkgroupSize(256).SetDispatchGroupSize(41);")
    branch = "        if (separate8 == 1) {\n" + norm_setup + projection_setup + "        }\n"
    return replace_once(result,
        "      return replay_.Run(context, {y, normalized, activated, tail}, [&]() -> Status {\n",
        "      return replay_.Run(context, {y, normalized, activated, tail}, [&]() -> Status {\n" + branch)


if __name__ == "__main__":
    report = build(OUTPUT, SOURCE, transform, SOURCE.parent / "native_compare.cpp", FLAG,
        dict(model_eligible=False, normalization_workgroups=4, normalization_threads=256,
             projection_workgroups=41, projection_threads=256, output_columns=8,
             projection_shared_storage_bytes=512,
             original_rms_streams=256, original_rms_tree_preserved=True,
             original_fp16_projection_schedule_preserved=True,
             both_stages_captured_in_original_replay=True,
             no_additional_persistent_buffers=True, original_replay_api_unchanged=True))
    report.update(original_compute_and_fallbacks_unchanged=False,
                  original_default_compute_and_fallbacks_unchanged=True)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
