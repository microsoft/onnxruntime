import json
from pathlib import Path

from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once, sha


FLAG = "ORT_WEBGPU_DENSE_PRELOAD"
OUTPUT = Path("/workspace/qwen38-int4-dense-preload/build-v1")
COMPONENT = Path("/workspace/qwen38-int4-dense-tile4/build-v1")
TEMPLATE = Path("/workspace/qwen38-int4-dense-bitcast-unpack/build-v1")


def preload_template(source):
    source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("//")) + "\n"
    source = replace_once(source, "Status ApplyDenseBitcastTemplate(", "Status ApplyDensePreloadTemplate(")
    start = source.index('ss << "    for (var id = local_idx; id < ";')
    stop = source.index('ss << "    for (var local_row_offset', start)
    staging = source[start:stop]
    source = source[:start] + source[stop:]
    staging = replace_once(staging, "loadSHMA(batch, a_row, kidx, id)",
                           "loadSHMA(batch, a_row, 0u, id)")
    marker = 'ss << "  for (var kidx = 0u; kidx < uniforms.K; kidx += ";'
    source = replace_once(source, marker, staging + marker)
    source = replace_once(source, 'ss << "        var a_offset = idx * (8 / ";',
                          'ss << "        var a_offset = kidx / " << __param_component_a << "u + idx * (8 / ";')
    source = replace_once(source, 'ss << "    workgroupBarrier();\\n";\nss << "  }\\n";',
                          'ss << "  }\\n";\nss << "    workgroupBarrier();\\n";')
    return source


def private_program(source):
    start = source.index("class DenseTile4Program")
    stop = source.index("\n}\n\nStatus MatMulNBitsProgram::", start)
    program = (source[start:stop].replace("DenseTile4Program", "DensePreloadProgram")
               .replace("ApplyDenseTile4MatMulNBits", "ApplyDensePreloadMatMulNBits")
               .replace("MatMulNBitsTile4F16", "MatMulNBitsPreloadF16")
               .replace("ApplyDenseBitcastTemplate", "ApplyDensePreloadTemplate"))
    program = replace_once(program, "bool acc_f32 = false)\n",
                           "bool acc_f32 = false, uint32_t full_k = 0)\n")
    program = replace_once(program, "acc_f32_(acc_f32) {}",
                           "acc_f32_(acc_f32), full_k_(full_k) {}")
    program = replace_once(program, "  bool acc_f32_;\n", "  bool acc_f32_;\n  uint32_t full_k_;\n")
    return replace_once(program, "const uint32_t a_length_per_tile = tile_size_k / components_a;",
                        "const uint32_t a_length_per_tile = full_k_ / components_a;")


def private_function(source):
    start = source.rindex("Status ApplyDenseTile4MatMulNBits(")
    stop = source.index("\n}\n\n}  // namespace webgpu", start)
    function = (source[start:stop].replace("ApplyDenseTile4MatMulNBits", "ApplyDensePreloadMatMulNBits")
                .replace("DenseTile4Program", "DensePreloadProgram"))
    function = replace_once(function, "tile_size_k_vec, broadcast_a, acc_f32};",
                            "tile_size_k_vec, broadcast_a, acc_f32, K};")
    return replace_once(function,
                        "has_weight_idx_indirect, tile_size_k_vec, broadcast_a, acc_f32);",
                        "has_weight_idx_indirect, tile_size_k_vec, broadcast_a, acc_f32, K);")


def transform(source):
    program, function = private_program(source), private_function(source)
    source = replace_once(source, '#include "dense_bitcast_template.inc"',
                          '#include "dense_bitcast_template.inc"\n#include "dense_preload_template.inc"')
    stop = source.index("\n}\n\nStatus MatMulNBitsProgram::")
    source = source[:stop] + "\n\n" + program + source[stop:]
    stop = source.index("\n}\n\n}  // namespace webgpu", source.rindex("Status ApplyDenseTile4MatMulNBits("))
    source = source[:stop] + "\n\n" + function + source[stop:]
    source = replace_once(source,
        "Status MatMulNBits::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {",
        f'''Status MatMulNBits::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {{
  const int preload_mode = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(preload_mode == 0 || preload_mode == 1, "{FLAG} must be 0 or 1");
  const uint32_t storage_limit = context.DeviceLimits().maxComputeWorkgroupStorageSize;
  const bool preload = preload_mode == 1 && storage_limit >= 256 && K_ > 0 &&
      static_cast<uint64_t>(K_) <= (storage_limit - 256) / sizeof(MLFloat16);''')
    source = replace_once(source, "  auto compute = [&]() {\n    if (dense_bitcast_route && dense_tile4 == 1) {",
        '''  auto compute = [&]() {
    if (dense_bitcast_route && preload) {
      return ApplyDensePreloadMatMulNBits(a, b, scales, zero_points, bias,
          K_, N_, block_size_, accuracy_level_, bits_, context, y, 0);
    }
    if (dense_bitcast_route && dense_tile4 == 1) {''')
    return replace_once(source, "compute, 0, dense_bitcast_route ? 1 + dense_tile4 : 0);",
                        "compute, 0, dense_bitcast_route ? (preload ? 3 : 1 + dense_tile4) : 0);")


def transform_compare(source):
    assert source.count(f'setenv("{FLAG}", value, 1)') == 1
    assert "expected_bits=" in source and "actual_bits=" in source
    return source


if __name__ == "__main__":
    frozen = json.loads((TEMPLATE / "build-manifest.json").read_text())
    original = TEMPLATE / "dense_bitcast_template.inc"
    assert sha(original) == frozen["source_sha256"][original.name]
    includes = OUTPUT.parent / "preload-template-v1"
    includes.mkdir(parents=True, exist_ok=False)
    header = includes / "dense_preload_template.inc"
    header.write_text(preload_template(original.read_text()))
    build(output=OUTPUT, flag=FLAG, source_transform=transform,
          component_dir=COMPONENT, source_name="matmul_nbits.cc",
          comparator_flag="ORT_WEBGPU_DENSE_TILE4", comparator_transform=transform_compare,
          parent_dir=Path("/workspace/qwen38-int4-qmoe-prefill512/build-v3"),
          extra_include_dirs=(TEMPLATE, includes))
    path = OUTPUT / "build-manifest.json"
    report = json.loads(path.read_text())
    report["source_sha256"]["../preload-template-v1/dense_preload_template.inc"] = sha(header)
    report["frozen_preload_parent_sha256"] = sha(original)
    path.write_text(json.dumps(report, indent=2) + "\n")
