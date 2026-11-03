from pathlib import Path

from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_HC_POST_NORM_WIDE16"
OUTPUT = Path("/workspace/qwen38-int4-hc-post-norm-wide16/build-v1")
STORAGE_BYTES = 4096 + 11520 * 2 + 512 * 2


def wide_program(source):
    start = source.index("class HCPostNormProjectionProgram final")
    stop = source.index("class HCPostNormProjectionF16", start)
    program = source[start:stop].replace("HCPostNormProjectionProgram", "HCPostNormProjectionWide16Program")
    program = replace_once(program, 'Program{"HCPostNormProjectionF16"}',
                           'Program{"HCPostNormProjectionWide16F16"}')
    program = replace_once(program, "partials: array<f16, 256>", "partials: array<f16, 512>")
    program = replace_once(program, '<< "let col = workgroup_idx * 8u + local_idx / 32u;\\n"',
        '<< "for (var slot = 0u; slot < 2u; slot++) {\\n"\n'
        '         << "let col = workgroup_idx * 16u + slot * 8u + local_idx / 32u;\\n"')
    program = replace_once(program, '<< "partials[local_idx] = total;\\n"',
        '<< "partials[slot * 256u + local_idx] = total;\\n"\n'
        '         << "}\\n"')
    program = replace_once(program, '<< "if (lane == 0u && col < 324u) {\\n"',
        '<< "for (var slot = 0u; slot < 2u; slot++) {\\n"\n'
        '         << "let col = workgroup_idx * 16u + slot * 8u + local_idx / 32u;\\n"\n'
        '         << "if (lane == 0u && col < 324u) {\\n"')
    program = replace_once(program, "value += partials[local_idx + i];",
                           "value += partials[slot * 256u + local_idx + i];")
    return replace_once(program, '<< "  }\\n}\\n";', '<< "  }\\n}\\n}\\n";')


def transform(source):
    source = replace_once(source, "class HCPostNormProjectionF16",
                          wide_program(source) + "class HCPostNormProjectionF16")
    source = replace_once(source,
        "  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {",
        f'''  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {{
    const int wide_mode = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
    ORT_RETURN_IF_NOT(wide_mode == 0 || wide_mode == 1, "{FLAG} must be 0 or 1");''')
    start = source.index("      return replay_.Run(context, {y, normalized, activated, tail}, [&]() -> Status {")
    stop = source.index("\n    }\n    HCPostNormProjectionBaselineNormProgram", start)
    block = source[start:stop]
    setup = block[block.index("        program.CacheHint"):block.index("      });")]
    setup = replace_once(setup, ".SetWorkgroupSize(256).SetDispatchGroupSize(41);",
                         ".SetWorkgroupSize(256).SetDispatchGroupSize(wide16 ? 21 : 41);")
    replacement = (
        f"      const bool wide16 = wide_mode == 1 && limits.maxComputeWorkgroupStorageSize >= {STORAGE_BYTES};\n"
        "      auto run = [&]<typename P>(P& program) -> Status {\n" + setup +
        "      };\n"
        "      return replay_.Run(context, {y, normalized, activated, tail}, [&]() -> Status {\n"
        "        if (wide16) {\n"
        "          HCPostNormProjectionWide16Program program{shared_, shared_scale};\n"
        "          return run(program);\n"
        "        }\n"
        "        HCPostNormProjectionProgram program{shared_, shared_scale};\n"
        "        return run(program);\n"
        "      }, 0, wide16);")
    return source[:start] + replacement + source[stop:]


if __name__ == "__main__":
    build(output=OUTPUT, flag=FLAG, source_transform=transform,
          parent_dir=Path("/workspace/qwen38-int4-dense-block32/build-v2"))
