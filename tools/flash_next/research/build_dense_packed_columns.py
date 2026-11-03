import json
import os
from pathlib import Path
import subprocess

from build_dense_double_buffer import transform_compare
from build_dense_lane8 import private_parts as lane8_parts
from build_qmoe_node_replay import replace_once, sha
from build_scoped_webgpu_source import build


ROOT = Path(__file__).parent
FLAG = "ORT_WEBGPU_DENSE_PACKED_COLUMNS"
OUTPUT = Path("/workspace/qwen38-int4-dense-packed-columns/build-v2")
SOURCE = Path("/workspace/qwen38-int4-qmoe-prefill-rows4/build-v2/matmul_nbits.cc")
BASE_COMPONENT = Path("/workspace/qwen38-int4-dense-block32/build-v1")
TEMPLATE = BASE_COMPONENT.parent / "block32-template-v1/dense_block32_template.inc"
INCLUDES = OUTPUT.parent / "packed-columns-template-v2"
SELECTOR = f'''
  const int dense_packed_columns_mode = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(dense_packed_columns_mode == 0 || dense_packed_columns_mode == 1,
                    "{FLAG} must be 0 or 1");
  const bool dense_packed_columns = dense_packed_columns_mode == 1 &&
      !context.EnableMatmulFp32Accumulation() &&
      context.AdapterInfo().vendor == std::string_view{{"nvidia"}};
'''
DISPATCH = '''
    if (dense_bitcast_route && block32_mode == 1 && dense_packed_columns) {
      return ApplyDensePackedColumnsMatMulNBits(a, b, scales, zero_points, bias,
          K_, N_, block_size_, accuracy_level_, bits_, context, y, 0);
    }
'''


def transform_template(original):
    source = replace_once(original, "Status ApplyDenseBlock32Template(",
                          "Status ApplyDensePackedColumnsTemplate(")
    start = source.index('// 136 |')
    end = source.index('// 301 |', start)
    return source[:start] + (ROOT / "dense_packed_columns_body.cpp.in").read_text() + source[end:]


def private_parts(original):
    program, function = lane8_parts(original)
    program = program.replace("DenseLane8", "DensePackedColumns").replace(
        "MatMulNBitsLane8F16", "MatMulNBitsPackedColumnsF16")
    function = function.replace("DenseLane8", "DensePackedColumns")
    function = replace_once(function,
        "constexpr uint32_t workgroup_size = 128;\n  constexpr uint32_t tile_size = 4;",
        "constexpr uint32_t workgroup_size = 64;\n  constexpr uint32_t tile_size = 4;")
    return program, function


def transform_source(original):
    program, function = private_parts(original)
    result = replace_once(original, '#include "dense_block32_template.inc"',
        '#include "dense_block32_template.inc"\n' +
        f'#include "{INCLUDES / "dense_packed_columns_template.inc"}"')
    end = result.index("\n}\n\nStatus MatMulNBitsProgram::")
    result = result[:end] + "\n\n" + program + result[end:]
    end = result.index("\n}\n\n}  // namespace webgpu",
                       result.rindex("Status ApplyDenseBlock32MatMulNBits("))
    result = result[:end] + "\n\n" + function + result[end:]
    signature = "Status MatMulNBits::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {"
    result = replace_once(result, signature, signature + SELECTOR)
    result = replace_once(result, "  auto compute = [&]() {\n",
                          "  auto compute = [&]() {\n" + DISPATCH)
    return replace_once(result, "(block32_mode == 1 ? 3 : 1 + dense_tile4)",
        "(block32_mode == 1 ? (dense_packed_columns ? 4 : 3) : 1 + dense_tile4)")


def main():
    manifest = json.loads((BASE_COMPONENT / "build-manifest.json").read_text())
    assert sha(TEMPLATE) == manifest["source_sha256"]["../block32-template-v1/dense_block32_template.inc"]
    INCLUDES.mkdir(parents=True, exist_ok=False)
    generated = INCLUDES / "dense_packed_columns_template.inc"
    generated.write_text(transform_template(TEMPLATE.read_text()))
    report = build(OUTPUT, SOURCE, transform_source, SOURCE.parent / "native_compare.cpp", FLAG,
                   dict(paired_output_columns=True, workgroup_size=64,
                        logical_k_lanes_per_column=32, output_columns_per_group=4,
                        original_serial_fold_and_tile_order_unchanged=True))
    comparator = OUTPUT / "native_compare.cpp"
    source = (BASE_COMPONENT / "native_compare.cpp").read_text()
    assert sha(BASE_COMPONENT / "native_compare.cpp") == manifest["source_sha256"]["native_compare.cpp"]
    source = replace_once(source, 'setenv("ORT_WEBGPU_DENSE_BLOCK32", value, 1)',
                          f'setenv("{FLAG}", value, 1)')
    comparator.write_text(transform_compare(source, flag=FLAG))
    command = ["c++", "-O3", "-std=c++20", str(comparator),
               "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
               "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
               "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
               "-lonnxruntime", "-o", str(OUTPUT / "native_compare")]
    subprocess.run(command, check=True)
    report.update(comparator_sha256=sha(OUTPUT / "native_compare"),
                  original_compute_and_fallbacks_unchanged=False,
                  original_fallbacks_unchanged=True,
                  no_existing_kernel_object_layout_changes=True,
                  no_reduction_or_normalization_changes=True, replay_variant_keyed=True,
                  comparator_compile_command=command)
    report["source_sha256"]["native_compare.cpp"] = sha(comparator)
    for path in (generated, TEMPLATE, BASE_COMPONENT / "native_compare.cpp",
                 BASE_COMPONENT / "build-manifest.json", Path(__file__),
                 ROOT / "dense_packed_columns_body.cpp.in", ROOT / "build_dense_lane8.py",
                 ROOT / "build_dense_double_buffer.py"):
        key = (str(path).replace("/evidence/",
               "/home/sayanshaw/.copilot/session-state/a608fc15-5d78-4524-9638-186c85652628/files/", 1)
               if path.is_relative_to(Path("/evidence")) else os.path.relpath(path, OUTPUT))
        report["source_sha256"][key] = sha(path)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status="built_not_qualified", provider_sha256=report["provider_sha256"])))


if __name__ == "__main__":
    main()
