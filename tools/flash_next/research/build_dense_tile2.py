from pathlib import Path

from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_DENSE_TILE2"
COMPONENT = Path("/workspace/qwen38-int4-dense-tile4/build-v1")
TEMPLATE = Path("/workspace/qwen38-int4-dense-bitcast-unpack/build-v1")


def private_program(source):
    start = source.index("class DenseTile4Program")
    stop = source.index("\n}\n\nStatus MatMulNBitsProgram::", start)
    return (source[start:stop].replace("DenseTile4Program", "DenseTile2Program")
            .replace("ApplyDenseTile4MatMulNBits", "ApplyDenseTile2MatMulNBits")
            .replace('Program{"MatMulNBitsTile4F16"}', 'Program{"MatMulNBitsTile2F16"}'))


def private_function(source):
    start = source.rindex("Status ApplyDenseTile4MatMulNBits(")
    stop = source.index("\n}\n\n}  // namespace webgpu", start)
    function = (source[start:stop].replace("ApplyDenseTile4MatMulNBits", "ApplyDenseTile2MatMulNBits")
                .replace("DenseTile4Program", "DenseTile2Program"))
    return replace_once(function,
                        "constexpr uint32_t workgroup_size = 128;\n  constexpr uint32_t tile_size = 4;",
                        "constexpr uint32_t workgroup_size = 64;\n  constexpr uint32_t tile_size = 2;")


def transform(source):
    program = private_program(source)
    function = private_function(source)
    stop = source.index("\n}\n\nStatus MatMulNBitsProgram::")
    source = source[:stop] + "\n\n" + program + source[stop:]
    stop = source.index("\n}\n\n}  // namespace webgpu", source.rindex("Status ApplyDenseTile4MatMulNBits("))
    source = source[:stop] + "\n\n" + function + source[stop:]
    source = replace_once(source,
        "Status MatMulNBits::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {",
        f'''Status MatMulNBits::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {{
  const int dense_tile2 = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(dense_tile2 == 0 || dense_tile2 == 1, "{FLAG} must be 0 or 1");''')
    source = replace_once(source, "  auto compute = [&]() {\n    if (dense_bitcast_route && dense_tile4 == 1) {",
        '''  auto compute = [&]() {
    if (dense_bitcast_route && dense_tile2 == 1) {
      return ApplyDenseTile2MatMulNBits(a, b, scales, zero_points, bias,
          K_, N_, block_size_, accuracy_level_, bits_, context, y, 0);
    }
    if (dense_bitcast_route && dense_tile4 == 1) {''')
    return replace_once(source, "compute, 0, dense_bitcast_route ? 1 + dense_tile4 : 0);",
                        "compute, 0, dense_bitcast_route ? (dense_tile2 == 1 ? 3 : 1 + dense_tile4) : 0);")


def transform_compare(source):
    assert source.count(f'setenv("{FLAG}", value, 1)') == 1
    assert '" expected_bits="' in source and '" actual_bits="' in source
    return source


if __name__ == "__main__":
    build(output=Path("/workspace/qwen38-int4-dense-tile2/build-v2"), flag=FLAG,
          source_transform=transform, component_dir=COMPONENT, source_name="matmul_nbits.cc",
          comparator_flag="ORT_WEBGPU_DENSE_TILE4", comparator_transform=transform_compare,
          parent_dir=Path("/workspace/qwen38-int4-qmoe-prefill512/build-v3"),
          extra_include_dirs=(TEMPLATE,))
