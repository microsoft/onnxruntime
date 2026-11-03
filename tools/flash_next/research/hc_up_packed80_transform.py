from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_HC_UP_PACKED80"
NAME = "HCUpPreMixPacked80F16"


def selected_program(source):
    start = source.index("class HCUpPreMixPacked2Program final")
    stop = source.index("class HCUpPreMixF16 final", start)
    program = source[start:stop].replace("HCUpPreMixPacked2Program", "HCUpPreMixPacked80Program")
    for original, replacement in (
            ('Program{"HCUpPreMixPacked2F16"}', f'Program{{"{NAME}"}}'),
            ("partials: array<f16, 256>", "partials: array<f16, 264>"),
            ("let lane = local_idx % 16u;", "let lane = local_idx % 10u;"),
            ("let branch = (local_idx / 16u) % 4u;", "let branch = (local_idx / 10u) % 4u;"),
            ("let column = local_idx / 64u;", "let column = local_idx / 40u;"),
            ("let partial_idx = column * 128u + branch * 32u + lane;",
             "let partial_idx = column * 132u + branch * 33u + lane;"),
            ('<< "partials[partial_idx + 16u] = f16(0);\\n"',
             '<< "for (var slot = lane + 10u; slot < 32u; slot += 10u) {\\n"\n'
             '         << "  partials[partial_idx + slot - lane] = f16(0);\\n"\n'
             '         << "}\\n"')):
        program = replace_once(program, original, replacement)
    return program


def transform(source):
    source = replace_once(source, "class HCUpPreMixF16 final",
                          selected_program(source) + "class HCUpPreMixF16 final")
    source = replace_once(
        source, "  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {",
        f'''  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {{
    static const int packed80 = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
    ORT_RETURN_IF_NOT(packed80 == 0 || packed80 == 1, "{FLAG} must be 0 or 1");''')
    source = replace_once(source, '    const int enabled = ParseEnvironmentVariableWithDefault<int>',
                          '    const bool selected = packed80 == 1 && parallel2 == 1;\n'
                          '    const int enabled = ParseEnvironmentVariableWithDefault<int>')
    source = replace_once(source, "program.SetWorkgroupSize(128).SetDispatchGroupSize",
                          "program.SetWorkgroupSize(selected ? 80 : 128).SetDispatchGroupSize")
    source = replace_once(
        source, "        if (parallel2 == 1) {\n          HCUpPreMixPacked2Program program;",
        "        if (selected) {\n"
        "          HCUpPreMixPacked80Program program;\n"
        "          return run(program);\n"
        "        }\n"
        "        if (parallel2 == 1) {\n          HCUpPreMixPacked2Program program;")
    return replace_once(source, "}, 0, parallel2);", "}, 0, selected ? 2 : parallel2);")
