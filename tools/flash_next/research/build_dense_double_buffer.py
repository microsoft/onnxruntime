import json
from pathlib import Path

from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once, sha


FLAG = "ORT_WEBGPU_DENSE_DOUBLE_BUFFER"
OUTPUT = Path("/workspace/qwen38-int4-dense-double-buffer/build-v1")
COMPONENT = Path("/workspace/qwen38-int4-dense-tile4/build-v1")
TEMPLATE = Path("/workspace/qwen38-int4-dense-bitcast-unpack/build-v1")
STORAGE_BYTES = 2 * 1024 * 2 + 4 * 32 * 2


def double_buffer_template(source):
    source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("//")) + "\n"
    source = replace_once(source, "Status ApplyDenseBitcastTemplate(", "Status ApplyDenseDoubleBufferTemplate(")
    original = '''ss << "var<workgroup> tile_A : array<input_a_value_t, ";
ss << __param_a_length_per_tile;
ss << ">;\\n";'''
    replacement = '''ss << "var<workgroup> tile_A : array<array<input_a_value_t, ";
ss << __param_a_length_per_tile;
ss << ">, 2>;\\n";'''
    source = replace_once(source, original, replacement)
    source = source.replace("tile_A[", 'tile_A[(kidx / " << __param_tile_size_k << "u) & 1u][')
    return replace_once(source, 'ss << "    workgroupBarrier();\\n";\nss << "  }\\n";',
                        'ss << "  }\\n";\nss << "  workgroupBarrier();\\n";')


def private_program(source):
    start = source.index("class DenseTile4Program")
    stop = source.index("\n}\n\nStatus MatMulNBitsProgram::", start)
    return (source[start:stop]
            .replace("DenseTile4Program", "DenseDoubleBufferProgram")
            .replace("ApplyDenseTile4MatMulNBits", "ApplyDenseDoubleBufferMatMulNBits")
            .replace("MatMulNBitsTile4F16", "MatMulNBitsDoubleBufferF16")
            .replace("ApplyDenseBitcastTemplate", "ApplyDenseDoubleBufferTemplate"))


def private_function(source):
    start = source.rindex("Status ApplyDenseTile4MatMulNBits(")
    stop = source.index("\n}\n\n}  // namespace webgpu", start)
    return (source[start:stop]
            .replace("ApplyDenseTile4MatMulNBits", "ApplyDenseDoubleBufferMatMulNBits")
            .replace("DenseTile4Program", "DenseDoubleBufferProgram"))


def transform(source):
    program, function = private_program(source), private_function(source)
    source = replace_once(source, '#include "dense_bitcast_template.inc"',
                          '#include "dense_bitcast_template.inc"\n#include "dense_double_buffer_template.inc"')
    stop = source.index("\n}\n\nStatus MatMulNBitsProgram::")
    source = source[:stop] + "\n\n" + program + source[stop:]
    stop = source.index("\n}\n\n}  // namespace webgpu", source.rindex("Status ApplyDenseTile4MatMulNBits("))
    source = source[:stop] + "\n\n" + function + source[stop:]
    signature = "Status MatMulNBits::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {"
    source = replace_once(source, signature, signature + f'''
  const int double_buffer_mode = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(double_buffer_mode == 0 || double_buffer_mode == 1, "{FLAG} must be 0 or 1");
  const size_t double_buffer_storage_bytes = 4096 + 4 * 32 *
      (context.EnableMatmulFp32Accumulation() ? sizeof(float) : sizeof(MLFloat16));
  const bool double_buffer = double_buffer_mode == 1 &&
      context.DeviceLimits().maxComputeWorkgroupStorageSize >= double_buffer_storage_bytes;
''')
    source = replace_once(source, "  auto compute = [&]() {\n    if (dense_bitcast_route && dense_tile4 == 1) {",
        '''  auto compute = [&]() {
    if (dense_bitcast_route && double_buffer) {
      return ApplyDenseDoubleBufferMatMulNBits(a, b, scales, zero_points, bias,
          K_, N_, block_size_, accuracy_level_, bits_, context, y, 0);
    }
    if (dense_bitcast_route && dense_tile4 == 1) {''')
    return replace_once(source, "compute, 0, dense_bitcast_route ? 1 + dense_tile4 : 0);",
                        "compute, 0, dense_bitcast_route ? (double_buffer ? 3 : 1 + dense_tile4) : 0);")


def transform_compare(source, flag=FLAG):
    assert source.count(f'setenv("{flag}", value, 1)') == 1
    assert "expected_bits=" in source and "actual_bits=" in source
    source = replace_once(source, "    size_t comparisons = 0;",
        '''    const char* fp32 = std::getenv("ORT_TEST_DENSE_FP32_ACCUMULATION");
    if (fp32 == nullptr) fp32 = "0";
    if (std::strcmp(fp32, "0") != 0 && std::strcmp(fp32, "1") != 0)
      throw std::runtime_error("FP32 accumulation selector must be 0 or 1");
    std::cout << "FP32 accumulation selector=" << fp32 << std::endl;
    size_t comparisons = 0;''')
    return replace_once(source, 'env, devices, {{"enableInt64", "1"}, {"validationMode", "full"},',
                        'env, devices, {{"enableInt64", "1"}, {"validationMode", "full"},\n'
                        '                         {"enableMatmulFp32Accumulation", fp32},')


if __name__ == "__main__":
    frozen = json.loads((TEMPLATE / "build-manifest.json").read_text())
    original = TEMPLATE / "dense_bitcast_template.inc"
    assert sha(original) == frozen["source_sha256"][original.name]
    includes = OUTPUT.parent / "double-buffer-template-v1"
    includes.mkdir(parents=True, exist_ok=False)
    header = includes / "dense_double_buffer_template.inc"
    header.write_text(double_buffer_template(original.read_text()))
    build(output=OUTPUT, flag=FLAG, source_transform=transform,
          component_dir=COMPONENT, source_name="matmul_nbits.cc",
          comparator_flag="ORT_WEBGPU_DENSE_TILE4", comparator_transform=transform_compare,
          parent_dir=Path("/workspace/qwen38-int4-qmoe-prefill1024/build-v2"),
          extra_include_dirs=(TEMPLATE, includes))
    path = OUTPUT / "build-manifest.json"
    report = json.loads(path.read_text())
    report["source_sha256"]["../double-buffer-template-v1/dense_double_buffer_template.inc"] = sha(header)
    report["frozen_double_buffer_parent_sha256"] = sha(original)
    report["activation_shared_storage_bytes"] = 4096
    report["total_shared_storage_bytes_fp16"] = STORAGE_BYTES
    report["total_shared_storage_bytes_fp32_accumulation"] = 4608
    path.write_text(json.dumps(report, indent=2) + "\n")
