from build_hc_post_norm_vec4 import private_program_transform
from build_hc_projection_inner_unroll import program_parts
from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_HC_GATE_STAGING"
CLASS = "HCPostNormProjectionGateStagingProgram"
NAME = "HCPostNormProjectionGateStagingF16"


def make_program(source):
    program = program_parts(source)[2].replace("HCPostNormProjectionProgram", CLASS)
    program = replace_once(program, 'Program{"HCPostNormProjectionF16"}', f'Program{{"{NAME}"}}')
    program = replace_once(program, '"var<workgroup> partials: array<f16, 256>;\\n"',
        '"var<workgroup> partials: array<f16, 256>;\\n"\n'
        '        "var<workgroup> staged_gates: array<f32, 5>;\\n"')
    begin = program.index("    auto& code = shader.MainFunctionBody();")
    end = program.index('    code << "for (var h = local_idx;', begin)
    staged = r'''    auto& code = shader.MainFunctionBody();
    code << "if (local_idx < 4u) {\n"
         << "let divided_gate = f16(" << gate.GetByOffset("local_idx") << " / f16(4.0));\n"
         << "let sigmoid_gate = f16(1.0 / (1.0 + exp(-divided_gate)));\n"
         << "let gate_value = f16(sigmoid_gate * f16(2.0));\n"
         << "staged_gates[local_idx] = f32(gate_value);\n}\n";
    if (shared_) {
      code << "if (local_idx == 4u) {\n"
           << "let shared_sigmoid = f16(1.0 / (1.0 + exp(-"
           << shared_gate->GetByOffset("0u") << ")));\n"
           << "staged_gates[4u] = f32(shared_sigmoid);\n}\n";
    }
    code << "workgroupBarrier();\n";
    if (shared_) {
      code << "let shared_sigmoid = f16(staged_gates[4u]);\n";
    }
    for (int stream = 0; stream < 4; ++stream) {
      code << "let gate_value" << stream << " = f16(staged_gates[" << stream << "u]);\n";
    }
'''
    return program[:begin] + staged + program[end:]


def transform(source):
    result = private_program_transform(source, make_program(source), FLAG, "gate_staging", CLASS)
    result = replace_once(result,
        f'    const int gate_staging = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);\n'
        f'    ORT_RETURN_IF_NOT(gate_staging == 0 || gate_staging == 1, "{FLAG} must be 0 or 1");',
        f'    static const int selector = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);\n'
        f'    ORT_RETURN_IF_NOT(selector == 0 || selector == 1, "{FLAG} must be 0 or 1");\n'
        '    const int gate_staging = selector == 1 && context.AdapterInfo().vendor == std::string_view{"nvidia"};')
    return replace_once(result, "}, 0, gate_staging);", "});")
