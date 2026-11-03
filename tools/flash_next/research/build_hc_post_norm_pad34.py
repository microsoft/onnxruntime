from pathlib import Path

from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_HC_POST_NORM_PAD34"


def padded_program(source):
    start = source.index("class HCPostNormProjectionProgram")
    stop = source.index("class HCPostNormProjectionF16", start)
    program = source[start:stop].replace("HCPostNormProjectionProgram", "HCPostNormProjectionPad34Program")
    program = replace_once(program, 'Program{"HCPostNormProjectionF16"}',
                           'Program{"HCPostNormProjectionPad34F16"}')
    assert program.count("36u") == 4
    program = program.replace("36u", "34u")
    return replace_once(program, "post_values: array<f16, 11520>", "post_values: array<f16, 10880>")


def transform(source):
    source = replace_once(source, "class HCPostNormProjectionF16",
                          padded_program(source) + "class HCPostNormProjectionF16")
    source = replace_once(source,
        "  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {",
        f'''  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {{
    const int pad34 = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
    ORT_RETURN_IF_NOT(pad34 == 0 || pad34 == 1, "{FLAG} must be 0 or 1");''')
    start = source.index("      return replay_.Run(context, {y, normalized, activated, tail}, [&]() -> Status {")
    stop = source.index("\n    }\n    HCPostNormProjectionBaselineNormProgram", start)
    block = source[start:stop]
    setup = block[block.index("        program.CacheHint"):block.index("      });")]
    replacement = (
        "      auto run = [&]<typename P>(P& program) -> Status {\n" + setup +
        "      };\n"
        "      return replay_.Run(context, {y, normalized, activated, tail}, [&]() -> Status {\n"
        "        if (pad34 == 1) {\n"
        "          HCPostNormProjectionPad34Program program{shared_, shared_scale};\n"
        "          return run(program);\n"
        "        }\n"
        "        HCPostNormProjectionProgram program{shared_, shared_scale};\n"
        "        return run(program);\n"
        "      }, 0, pad34);")
    return source[:start] + replacement + source[stop:]


if __name__ == "__main__":
    build(output=Path("/workspace/qwen38-int4-hc-post-norm-pad34/build-v1"),
          flag=FLAG, source_transform=transform,
          parent_dir=Path("/workspace/qwen38-int4-hc-up-parallel2/build-v1"))
