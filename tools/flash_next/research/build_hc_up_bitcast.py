from pathlib import Path

from build_hc_bitcast_unpack import main as build
from build_mlp_bitcast_unpack import DECODE_IMPLEMENTATION
from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_HC_UP_BITCAST_UNPACK"


def transform(source):
    stop = source.index("class HCUpPreMixF16")
    program = source[:stop]
    program = replace_once(program, 'HCUpPreMixProgram() : Program{"HCUpPreMixF16"} {}',
        'HCUpPreMixProgram(bool bitcast) : Program{bitcast ? "HCUpPreMixBitcastF16" : "HCUpPreMixF16"}, bitcast_{bitcast} {}')
    program = replace_once(program, "    auto& code = shader.MainFunctionBody();", DECODE_IMPLEMENTATION)
    for name, argument, original in (
            ("low", "packed[i]", "packed[i] & 0x0f0f0f0fu"),
            ("high", "packed[i] >> 4u", "(packed[i] >> 4u) & 0x0f0f0f0fu")):
        literal = f'"    let {name} = vec4<f16>(unpack4xU8({original})) - vec4<f16>(8);\\n"'
        program = replace_once(program, literal,
            f'(bitcast_ ? "    let {name} = decode_low_nibbles({argument});\\n" : {literal})')
    program = replace_once(program,
        '      {"reduction_scale", ProgramUniformVariableDataType::Float32});\n};',
        '      {"reduction_scale", ProgramUniformVariableDataType::Float32});\n'
        ' private:\n  bool bitcast_;\n};')
    kernel = source[stop:]
    kernel = replace_once(kernel,
        "  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {",
        f'''  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {{
    const int bitcast = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
    ORT_RETURN_IF_NOT(bitcast == 0 || bitcast == 1, "{FLAG} must be 0 or 1");''')
    kernel = replace_once(kernel, "HCUpPreMixProgram program;", "HCUpPreMixProgram program{bitcast == 1};")
    kernel = replace_once(kernel, "return context.RunProgram(program);\n      });",
                          "return context.RunProgram(program);\n      }, 0, bitcast);")
    return program + kernel


if __name__ == "__main__":
    build(output=Path("/workspace/qwen38-int4-hc-up-bitcast-unpack/build-v1"),
          flag=FLAG, source_transform=transform,
          component_dir=Path("/workspace/qwen38-int4-hc-up-premix/build-v2"),
          source_name="hc_up_premix.cc", comparator_flag="ORT_WEBGPU_HC_UP_PREMIX")
