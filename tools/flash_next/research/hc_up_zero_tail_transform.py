from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_HC_UP_ZERO_TAIL"
NAME = "HCUpPreMixPacked2ZeroTailF16"


def selected_program(source):
    start = source.index("class HCUpPreMixPacked2Program final")
    stop = source.index("class HCUpPreMixF16 final", start)
    program = source[start:stop].replace("HCUpPreMixPacked2Program", "HCUpPreMixZeroTailProgram")
    program = replace_once(program, 'Program{"HCUpPreMixPacked2F16"}', f'Program{{"{NAME}"}}')
    return replace_once(
        program,
        '<< "  for (var i = 0u; i < 32u; i++) { value += partials[partial_idx + i]; }\\n"',
        '<< "  for (var i = 0u; i < 10u; i++) { value += partials[partial_idx + i]; }\\n"\n'
        '         << "  value += f16(0);\\n"')


def transform(source):
    source = replace_once(source, "class HCUpPreMixF16 final",
                          selected_program(source) + "class HCUpPreMixF16 final")
    source = replace_once(
        source,
        "  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {",
        f'''  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {{
    static const int zero_tail = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
    ORT_RETURN_IF_NOT(zero_tail == 0 || zero_tail == 1, "{FLAG} must be 0 or 1");''')
    source = replace_once(source,
                          '    const int enabled = ParseEnvironmentVariableWithDefault<int>',
                          '    const bool selected = zero_tail == 1 && parallel2 == 1;\n'
                          '    const int enabled = ParseEnvironmentVariableWithDefault<int>')
    source = replace_once(
        source, "        if (parallel2 == 1) {\n          HCUpPreMixPacked2Program program;",
        "        if (selected) {\n"
        "          HCUpPreMixZeroTailProgram program;\n"
        "          return run(program);\n"
        "        }\n"
        "        if (parallel2 == 1) {\n          HCUpPreMixPacked2Program program;")
    return replace_once(source, "}, 0, parallel2);", "}, 0, selected ? 2 : parallel2);")
