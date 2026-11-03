from pathlib import Path

from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_HC_UP_PARALLEL2"


def packed_program(source):
    start = source.index("class HCUpPreMixProgram")
    stop = source.index("class HCUpPreMixF16", start)
    program = source[start:stop].replace("HCUpPreMixProgram", "HCUpPreMixPacked2Program")
    for original, replacement in (
            ('Program{"HCUpPreMixF16"}', 'Program{"HCUpPreMixPacked2F16"}'),
            ("partials: array<f16, 128>", "partials: array<f16, 256>"),
            ("gates: array<f16, 4>", "gates: array<f16, 8>"),
            ("let lane = local_idx % 32u;", "let lane = local_idx % 16u;"),
            ('<< "let branch = local_idx / 32u;\\n"',
             '<< "let branch = (local_idx / 16u) % 4u;\\n"\n'
             '         << "let column = local_idx / 64u;\\n"\n'
             '         << "let output_col = workgroup_idx * 2u + column;\\n"\n'
             '         << "let partial_idx = column * 128u + branch * 32u + lane;\\n"'),
            ("let col = workgroup_idx + branch * 2560u;",
             "let col = output_col + branch * 2560u;"),
            ('<< "partials[local_idx] = total;\\n"',
             '<< "partials[partial_idx] = total;\\n"\n'
             '         << "partials[partial_idx + 16u] = f16(0);\\n"'),
            ("value += partials[local_idx + i]", "value += partials[partial_idx + i]"),
            ("gates[branch] =", "gates[column * 4u + branch] ="),
            ("if (local_idx == 0u)", "if (branch == 0u && lane == 0u)"),
            ('streams.GetByOffset("c * 2560u + workgroup_idx")',
             'streams.GetByOffset("c * 2560u + output_col")'),
            ("f32(gates[c])", "f32(gates[column * 4u + c])"),
            ('y.SetByOffset("workgroup_idx",', 'y.SetByOffset("output_col",')):
        program = replace_once(program, original, replacement)
    return program


def transform(source):
    source = replace_once(source, "class HCUpPreMixF16",
                          packed_program(source) + "class HCUpPreMixF16")
    source = replace_once(source,
        "  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {",
        f'''  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {{
    const int parallel2 = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
    ORT_RETURN_IF_NOT(parallel2 == 0 || parallel2 == 1, "{FLAG} must be 0 or 1");''')
    start = source.index("      return static_replay_.Run(context, {y}, [&]() -> Status {")
    stop = source.index("\n    }\n    auto projected", start)
    block = source[start:stop]
    setup = block[block.index("        program.SetWorkgroupSize"):block.index("      });")]
    setup = replace_once(setup, ".SetDispatchGroupSize(2560)",
                         ".SetDispatchGroupSize(parallel2 == 1 ? 1280 : 2560)")
    replacement = (
        "      auto run = [&]<typename P>(P& program) -> Status {\n" + setup +
        "      };\n"
        "      return static_replay_.Run(context, {y}, [&]() -> Status {\n"
        "        if (parallel2 == 1) {\n"
        "          HCUpPreMixPacked2Program program;\n"
        "          return run(program);\n"
        "        }\n"
        "        HCUpPreMixProgram program;\n"
        "        return run(program);\n"
        "      }, 0, parallel2);")
    return source[:start] + replacement + source[stop:]


if __name__ == "__main__":
    build(output=Path("/workspace/qwen38-int4-hc-up-parallel2/build-v1"),
          flag=FLAG, source_transform=transform,
          component_dir=Path("/workspace/qwen38-int4-hc-up-premix/build-v2"),
          source_name="hc_up_premix.cc", comparator_flag="ORT_WEBGPU_HC_UP_PREMIX",
          parent_dir=Path("/workspace/qwen38-int4-qsa-score16/build-v5"))
