import json
from pathlib import Path

from build_hc_owned_vector import make_program as vector_program
from build_hc_post_norm_vec4 import private_program_transform
from build_hc_projection_inner_unroll import SOURCE
from build_qmoe_node_replay import replace_once
from build_scoped_webgpu_source import build


FLAG = "ORT_WEBGPU_HC_OWNED_VECTOR_UNROLL"
CLASS = "HCPostNormProjectionOwnedVectorUnrollProgram"
NAME = "HCPostNormProjectionOwnedVectorUnrollF16"
OUTPUT = Path("/workspace/qwen38-int4-hc-owned-vector-unroll/build-v1")


def make_program(source):
    program = vector_program(source).replace("HCPostNormProjectionOwnedVectorProgram", CLASS)
    program = replace_once(program, "HCPostNormProjectionOwnedVectorF16", NAME)
    begin = ('    code << "for (var component = 0u; component < 4u; component++) {\\n"\n'
             '         << "let h = quad * 4u + component;\\n";\n')
    end = '    code << "}\\n";\n    for (int stream = 0; stream < 4; ++stream) {'
    if program.count(begin) != 2:
        raise ValueError("Unexpected owned staging component loops")
    while begin in program:
        start = program.index(begin)
        body_start = start + len(begin)
        stop = program.index(end, body_start)
        body = program[body_start:stop]
        replacement = '''    for (uint32_t component = 0; component < 4; ++component) {
      code << "{\\nlet component = " << component << "u;\\n"
           << "let h = quad * 4u + component;\\n";
''' + body + '''      code << "}\\n";
    }
'''
        program = program[:start] + replacement + program[stop + len('    code << "}\\n";\n'):]
    return program


def transform(source):
    result = private_program_transform(source, make_program(source), FLAG, "owned_vector_unroll", CLASS)
    result = replace_once(result,
        f'    const int owned_vector_unroll = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);\n'
        f'    ORT_RETURN_IF_NOT(owned_vector_unroll == 0 || owned_vector_unroll == 1, "{FLAG} must be 0 or 1");',
        f'    static const int selector = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);\n'
        f'    ORT_RETURN_IF_NOT(selector == 0 || selector == 1, "{FLAG} must be 0 or 1");\n'
        '    const int owned_vector_unroll = selector == 1 && context.AdapterInfo().vendor == std::string_view{"nvidia"};')
    return replace_once(result, "}, 0, owned_vector_unroll);", "});")


if __name__ == "__main__":
    report = build(OUTPUT, SOURCE, transform, SOURCE.parent / "native_compare.cpp", FLAG,
        dict(model_eligible=False, physical_threads=256, output_columns=8,
             original_rms_streams=256, original_rms_tree_preserved=True,
             original_fp16_projection_schedule_preserved=True,
             single_owner_vector_stores=True, staging_components_constant_unrolled=True,
             shared_storage_bytes=27648, original_replay_api_unchanged=True))
    report.update(original_compute_and_fallbacks_unchanged=False,
                  original_default_compute_and_fallbacks_unchanged=True)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
