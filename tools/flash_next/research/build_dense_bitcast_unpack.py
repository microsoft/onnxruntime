import json
import os
from pathlib import Path
import re
import subprocess

from build_mlp_bitcast_unpack import DECODE_IMPLEMENTATION
from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent
PARENT = Path("/workspace/qwen38-int4-expert-bitcast-unpack/build-v2")
COMPONENT = Path("/workspace/qwen38-int4-generic-matmul-replay/build-v1")
FROZEN = Path("/workspace/qwen38-int4-static-node-replay/build-v8")
GENERATED = Path("/workspace/onnxruntime-webgpu-fp8-checkpoint/build-webgpu-fp8/Release/wgsl_generated/wgsl_template_gen")
OUTPUT = Path("/workspace/qwen38-int4-dense-bitcast-unpack/build-v1")
FLAG = "ORT_WEBGPU_DENSE_BITCAST_UNPACK"


def transform_template(source, strings):
    source = replace_once(source, "#pragma once\n", "")
    source = replace_once(source,
        'template <>\nStatus ApplyTemplate<"quantization/matmul_nbits.wgsl.template">',
        "Status ApplyDenseBitcastTemplate")
    source = replace_once(source,
        'TemplateParameter<"quantization/matmul_nbits.wgsl.template">::type',
        'onnxruntime::webgpu::wgsl_gen::TemplateParameter<"quantization/matmul_nbits.wgsl.template">::type')
    literals = dict(re.findall(r'constexpr const char\* (__str_\d+) = (".*");', strings))
    for symbol, word, indent in (
            ("__str_3784", "b_value", "        "),
            ("__str_3785", "b_value >> 4u", "        "),
            ("__str_3793", "b_value[i]", "            "),
            ("__str_3794", "b_value[i] >> 4u", "            ")):
        name = "lower" if symbol in ("__str_3784", "__str_3793") else "upper"
        assert "unpack4xU8" in literals[symbol]
        source = replace_once(source, f"ss << {symbol};",
            f'ss << "{indent}let b_value_{name} = decode_low_nibbles({word});\\n";')
    source = re.sub(r"\b__str_\d+\b", lambda match: literals[match[0]], source)
    decoder = DECODE_IMPLEMENTATION.split("      shader.AdditionalImplementation() <<", 1)[1]
    decoder = decoder.split("\n    }\n", 1)[0]
    source = replace_once(source, "  // Extract parameters",
        "  shader_helper.AdditionalImplementation() <<" + decoder + "\n\n  // Extract parameters")
    source = replace_once(source, "MainFunctionStart();",
        "{ auto& ss = shader_helper.MainFunctionBody();")
    return replace_once(source, "MainFunctionEnd();", "}")


def transform_replay(source):
    source = replace_once(source, "int64_t row_elements = 0) {",
        "int64_t row_elements = 0, uint64_t variant = 0) {")
    return replace_once(source, "    key.reserve(96);",
        "    key.reserve(97);\n    key.push_back(variant);")


def transform(source, header):
    declaration = header[header.index("class MatMulNBitsProgram final"):
                         header.index("class MatMulNBits final")]
    declaration = declaration.replace("MatMulNBitsProgram", "DenseBitcastProgram")
    declaration = replace_once(declaration, 'Program{"MatMulNBits"}',
        'Program{"MatMulNBitsBitcastF16"}')
    start = source.index("Status MatMulNBitsProgram::GenerateShaderCode")
    end = source.index("Status MatMulNBits::ComputeInternal", start)
    method = source[start:end].replace("MatMulNBitsProgram::", "DenseBitcastProgram::")
    method = replace_once(method,
        'WGSL_TEMPLATE_APPLY(shader, "quantization/matmul_nbits.wgsl.template",',
        "ApplyDenseBitcastTemplate(shader, {")
    method = replace_once(method, "WGSL_TEMPLATE_VARIABLE(scales_b, scales_b));",
        "WGSL_TEMPLATE_VARIABLE(scales_b, scales_b)});")
    function_start = source.index("Status ApplyMatMulNBits(")
    function_end = source.index("\n}  // namespace webgpu", function_start)
    function = source[function_start:function_end]
    function = replace_once(function, "Status ApplyMatMulNBits(", "Status ApplyDenseBitcastMatMulNBits(")
    function = replace_once(function, "  MatMulNBitsProgram program{",
        "  DenseBitcastProgram program{")
    forward = header[header.index("Status ApplyMatMulNBits("):
                     header.index("\n}  // namespace webgpu")]
    forward = replace_once(forward, "Status ApplyMatMulNBits(", "Status ApplyDenseBitcastMatMulNBits(")
    addition = ('namespace {\n#include "dense_bitcast_template.inc"\n' +
                declaration + method + forward + "\n}\n\n")
    source = source[:start] + addition + source[start:]
    source = replace_once(source,
        "Status MatMulNBits::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {",
        f'''Status MatMulNBits::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {{
  const int dense_bitcast = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(dense_bitcast == 0 || dense_bitcast == 1, "{FLAG} must be 0 or 1");''')
    source = replace_once(source, "  auto compute = [&]() {",
        '''  const bool dense_bitcast_route =
      dense_bitcast == 1 &&
      context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
      bits_ == 4 && block_size_ == 32 && !has_zero_points && bias == nullptr &&
      !context.EnableMatmulFp32Accumulation() &&
      a->IsDataType<MLFloat16>() && scales->IsDataType<MLFloat16>() &&
      a->Shape().Size() == K_ && K_ % 32 == 0 &&
      ((K_ == 2560 && (N_ == 1 || N_ == 16384 || N_ == 7168 || N_ == 6144 ||
                      N_ == 10240 || N_ == 2560)) ||
       ((K_ == 6144 || K_ == 640) && N_ == 2560) ||
       (K_ == 10240 && N_ == 320));
  auto compute = [&]() {
    if (dense_bitcast_route) {
      return ApplyDenseBitcastMatMulNBits(a, b, scales, zero_points, bias,
          K_, N_, block_size_, accuracy_level_, bits_, context, y, 0);
    }''')
    source = replace_once(source, "return static_replay_.Run(context, {y}, compute);",
        "return static_replay_.Run(context, {y}, compute, 0, dense_bitcast_route);")
    return replace_once(source, "\n}  // namespace webgpu",
        "\nnamespace {\n" + function + "\n}\n\n}  // namespace webgpu")


def main():
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    component = json.loads((COMPONENT / "build-manifest.json").read_text())
    frozen = json.loads((FROZEN / "build-manifest.json").read_text())
    original = COMPONENT / "matmul_nbits.cc"
    assert sha(original) == component["source_sha256"][original.name]
    OUTPUT.mkdir(parents=True, exist_ok=False)
    scratch = OUTPUT / "scratch"
    scratch.mkdir()
    overlay = OUTPUT / "overlay"
    overlay.mkdir()
    environment = dict(os.environ, TMPDIR=str(scratch))
    library = OUTPUT / "libonnxruntime_providers_webgpu.so"
    link = parent["link_command"].copy()
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    assert sha(library) == parent["provider_sha256"], "Baseline relink mismatch"
    library.unlink()
    replay = FROZEN / "overlay/static_node_replay.h"
    (overlay / replay.name).write_text(transform_replay(replay.read_text()))
    template = GENERATED / "generated/quantization/matmul_nbits.h"
    strings = GENERATED / "string_table.h"
    (OUTPUT / "dense_bitcast_template.inc").write_text(
        transform_template(template.read_text(), strings.read_text()))
    header = FROZEN / "overlay/contrib_ops/webgpu/quantization/matmul_nbits.h"
    source = OUTPUT / "matmul_nbits.cc"
    source.write_text(transform(original.read_text(), header.read_text()))
    obj = OUTPUT / (source.name + ".o")
    command = component["compile_commands"][0].copy()
    command[1:1] = ["-I" + str(overlay)]
    command[command.index("-c") + 1] = str(source)
    command[command.index("-o") + 1] = str(obj)
    subprocess.run(command, cwd=parent["working_directory"], env=environment, check=True)
    assert link.count(str(COMPONENT / obj.name)) == 1
    link[link.index(str(COMPONENT / obj.name))] = str(obj)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    comparator = OUTPUT / "native_compare.cpp"
    comparator.write_text(replace_once((ROOT / "static_node_replay_compare.cpp").read_text(),
        'setenv("ORT_WEBGPU_STATIC_NODE_REPLAY", value, 1)', f'setenv("{FLAG}", value, 1)'))
    subprocess.run(["c++", "-O3", "-std=c++20", str(comparator),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(OUTPUT / "native_compare")], env=environment, check=True)
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
        baseline_provider_sha256=parent["provider_sha256"], baseline_byte_identical=True,
        compile_commands=[command], link_command=link, working_directory=parent["working_directory"],
        source_sha256={str(p.relative_to(OUTPUT)): sha(p) for p in
                       (source, comparator, overlay / replay.name, OUTPUT / "dense_bitcast_template.inc")},
        frozen_template_sha256=sha(template), frozen_string_table_sha256=sha(strings),
        comparator_sha256=sha(OUTPUT / "native_compare"), no_existing_object_layout_changes=True,
        standalone_matmul_only=True, replay_variant_keyed=True,
        quality_qualified=False, headline_eligible=False)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "provider_sha256", "baseline_byte_identical")}))


if __name__ == "__main__":
    main()
