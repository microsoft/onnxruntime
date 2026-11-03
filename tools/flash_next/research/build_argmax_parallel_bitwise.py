import json
from pathlib import Path
import subprocess

import build_argmax_parallel_v3 as parent
from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent
FLAG = "ORT_WEBGPU_ARGMAX_PARALLEL_BITWISE"
OUTPUT = Path("/workspace/qwen38-int4-argmax-parallel/build-v4")


def make_program():
    source = parent.PROGRAM.replace("array<f32, 128>", "array<u32, 128>")
    source = replace_once(source,
        '  ArgMaxParallelPartialProgram() : Program{"ArgMaxParallelPartial"} {}',
        '  ArgMaxParallelPartialProgram(bool fp16) : Program{"ArgMaxParallelPartial"}, fp16_{fp16} {}\n'
        ' private:\n  const bool fp16_;\n public:')
    source = replace_once(source,
        '    const auto& x = shader.AddInput("x", ShaderUsage::UseValueTypeAlias);',
        r'''    const auto& x = shader.AddInput("x", ShaderUsage::UseValueTypeAlias);
    const auto bits = [this](const std::string& value) {
      return fp16_ ? "((bitcast<u32>(vec2<f16>(" + value + ", f16(0)))) & 65535u)"
                   : "bitcast<u32>(" + value + ")";
    };
    shader.AdditionalImplementation()
        << "fn is_nan(bits: u32) -> bool { return (bits & "
        << (fp16_ ? "32767u" : "2147483647u") << ") > "
        << (fp16_ ? "31744u" : "2139095040u") << "; }\n"
        << "fn ordered_key(bits: u32) -> u32 {\n"
        << "  let sign = " << (fp16_ ? "32768u" : "2147483648u") << ";\n"
        << "  let normalized = select(bits, 0u, (bits & (sign - 1u)) == 0u);\n"
        << "  return select(normalized ^ sign, (~normalized) & "
        << (fp16_ ? "65535u" : "4294967295u") << ", (normalized & sign) != 0u);\n}\n";''')
    source = replace_once(source,
        '         << "var best = f32(" << x.GetByOffset("offset") << ");\\n"',
        '         << "let seed = " << bits(x.GetByOffset("offset")) << ";\\n"\n'
        '         << "let first_nan = is_nan(seed);\\n"\n'
        '         << "var best = ordered_key(seed);\\n"')
    source = replace_once(source,
        '         << "  let candidate = f32(" << x.GetByOffset("offset + i") << ");\\n"\n'
        '         << "  if (candidate > best) { best = candidate; index = i; }\\n"',
        '         << "  let raw = " << bits(x.GetByOffset("offset + i")) << ";\\n"\n'
        '         << "  let candidate = ordered_key(raw);\\n"\n'
        '         << "  if (!first_nan && !is_nan(raw) && candidate > best) { best = candidate; index = i; }\\n"')
    return source


PROGRAM = make_program()
PARENT_TRANSFORM = parent.transform


def transform(source):
    source = PARENT_TRANSFORM(source)
    source = replace_once(source, "DataTypeImpl::GetType<float>(), partial_shape",
                          "DataTypeImpl::GetType<uint32_t>(), partial_shape")
    return replace_once(source, "ArgMaxParallelPartialProgram partial;",
                        "ArgMaxParallelPartialProgram partial(input_tensor->IsDataType<MLFloat16>());")


if __name__ == "__main__":
    parent.FLAG, parent.OUTPUT, parent.PROGRAM, parent.transform = FLAG, OUTPUT, PROGRAM, transform
    parent.build()
    comparator = ROOT / "argmax_parallel_compare_bitwise.cpp"
    binary = OUTPUT / "bitwise_argmax_compare"
    subprocess.run(["c++", "-O3", "-std=c++20", str(comparator),
                    "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
                    "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
                    "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
                    "-lonnxruntime", "-o", str(binary)], check=True)
    path = OUTPUT / "build-manifest.json"
    report = json.loads(path.read_text())
    report["bitwise_comparator_sha256"] = sha(binary)
    report["integer_order_keys_preserve_subnormals"] = True
    report["source_sha256"].update({str(p): sha(p) for p in
                                   (Path(__file__), comparator, ROOT / "prepare_argmax_parallel_bitwise.py")})
    path.write_text(json.dumps(report, indent=2) + "\n")
