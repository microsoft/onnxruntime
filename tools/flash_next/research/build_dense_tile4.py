from pathlib import Path

from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_DENSE_TILE4"
COMPONENT = Path("/workspace/qwen38-int4-dense-bitcast-unpack/build-v1")


def private_program(source):
    start = source.index("class DenseBitcastProgram")
    stop = source.index("\n}\n\nStatus MatMulNBitsProgram::", start)
    return (source[start:stop].replace("DenseBitcastProgram", "DenseTile4Program")
            .replace("ApplyDenseBitcastMatMulNBits", "ApplyDenseTile4MatMulNBits")
            .replace('Program{"MatMulNBitsBitcastF16"}', 'Program{"MatMulNBitsTile4F16"}'))


def private_function(source):
    start = source.rindex("Status ApplyDenseBitcastMatMulNBits(")
    stop = source.index("\n}\n\n}  // namespace webgpu", start)
    function = (source[start:stop].replace("ApplyDenseBitcastMatMulNBits", "ApplyDenseTile4MatMulNBits")
                .replace("DenseBitcastProgram", "DenseTile4Program"))
    return replace_once(function, "constexpr uint32_t tile_size = 8;", "constexpr uint32_t tile_size = 4;")


def transform(source):
    program = private_program(source)
    function = private_function(source)
    stop = source.index("\n}\n\nStatus MatMulNBitsProgram::")
    source = source[:stop] + "\n\n" + program + source[stop:]
    stop = source.index("\n}\n\n}  // namespace webgpu", source.rindex("Status ApplyDenseBitcastMatMulNBits("))
    source = source[:stop] + "\n\n" + function + source[stop:]
    source = replace_once(source,
        "Status MatMulNBits::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {",
        f'''Status MatMulNBits::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {{
  const int dense_tile4 = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(dense_tile4 == 0 || dense_tile4 == 1, "{FLAG} must be 0 or 1");''')
    source = replace_once(source, "  auto compute = [&]() {\n    if (dense_bitcast_route) {",
        '''  auto compute = [&]() {
    if (dense_bitcast_route && dense_tile4 == 1) {
      return ApplyDenseTile4MatMulNBits(a, b, scales, zero_points, bias,
          K_, N_, block_size_, accuracy_level_, bits_, context, y, 0);
    }
    if (dense_bitcast_route) {''')
    return replace_once(source, "compute, 0, dense_bitcast_route);",
                        "compute, 0, dense_bitcast_route ? 1 + dense_tile4 : 0);")


if __name__ == "__main__":
    build(output=Path("/workspace/qwen38-int4-dense-tile4/build-v1"), flag=FLAG,
          source_transform=transform, component_dir=COMPONENT, source_name="matmul_nbits.cc",
          comparator_flag="ORT_WEBGPU_DENSE_BITCAST_UNPACK",
          parent_dir=Path("/workspace/qwen38-int4-hc-up-parallel2/build-v1"),
          extra_include_dirs=(COMPONENT,))
