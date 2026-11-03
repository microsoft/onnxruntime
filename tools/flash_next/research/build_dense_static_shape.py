import json
from pathlib import Path
import re

from build_dense_double_buffer import transform_compare
from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once, sha


FLAG = "ORT_WEBGPU_DENSE_STATIC_SHAPE"
COMPONENT = Path("/workspace/qwen38-int4-dense-block32/build-v1")
PARENT = Path("/workspace/qwen38-int4-dense-block32/build-v2")
OUTPUT = Path("/workspace/qwen38-int4-dense-static-shape/build-v1")
DIMENSIONS = ("N", "K", "K_of_a", "K_of_b", "blocks_per_col")


def static_template(source):
    source = replace_once(source, "Status ApplyDenseBlock32Template(", "Status ApplyDenseStaticShapeTemplate(")
    return re.sub(r"\buniforms\.(N|K|K_of_a|K_of_b|blocks_per_col)\b", r"dense_static_\1", source)


def transform(source):
    start = source.index("class DenseBlock32Program")
    stop = source.index("\n}\n\nStatus MatMulNBitsProgram::", start)
    program = source[start:stop]
    start = source.rindex("Status ApplyDenseBlock32MatMulNBits(")
    stop = source.index("\n}\n\n}  // namespace webgpu", start)
    function = source[start:stop]
    for old, new in (("DenseBlock32Program", "DenseStaticShapeProgram"),
                     ("ApplyDenseBlock32", "ApplyDenseStaticShape"),
                     ("MatMulNBitsBlock32F16", "MatMulNBitsStaticShapeF16")):
        program = program.replace(old, new)
        function = function.replace(old, new)
    arguments = ", ".join(f"uint32_t static_{name}" for name in DIMENSIONS)
    program = replace_once(program, "DenseStaticShapeProgram(uint32_t tile_size,",
                          f"DenseStaticShapeProgram({arguments}, uint32_t tile_size,")
    members = ", ".join(f"static_{name}_{{static_{name}}}" for name in DIMENSIONS)
    program = replace_once(program, 'Program{"MatMulNBitsStaticShapeF16"},',
                          f'Program{{"MatMulNBitsStaticShapeF16"}}, {members},')
    declarations = "".join(f"  uint32_t static_{name}_;\n" for name in DIMENSIONS)
    program = replace_once(program, " private:\n", " private:\n" + declarations)
    constants = "".join(
        f'  shader.AdditionalImplementation() << "const dense_static_{name}: u32 = " '
        f'<< static_{name}_ << "u;\\n";\n' for name in DIMENSIONS)
    program = replace_once(program, "  return ApplyDenseStaticShapeTemplate(",
                          constants + "  return ApplyDenseStaticShapeTemplate(")
    function = replace_once(function, "DenseStaticShapeProgram program{tile_size,",
        "DenseStaticShapeProgram program{N, K, K / components_a, K_of_b, n_blocks_per_col, tile_size,")
    function = replace_once(function,
        "tile_size_k_vec, broadcast_a, acc_f32);\n  if (has_zero_points)",
        "tile_size_k_vec, broadcast_a, acc_f32, N, K, K / components_a, K_of_b, n_blocks_per_col);\n  if (has_zero_points)")
    source = replace_once(source, '#include "dense_block32_template.inc"',
                          '#include "dense_block32_template.inc"\n#include "dense_static_shape_template.inc"')
    stop = source.index("\n}\n\nStatus MatMulNBitsProgram::")
    source = source[:stop] + "\n\n" + program + source[stop:]
    stop = source.index("\n}\n\n}  // namespace webgpu", source.rindex("Status ApplyDenseBlock32MatMulNBits("))
    source = source[:stop] + "\n\n" + function + source[stop:]
    signature = "Status MatMulNBits::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {"
    source = replace_once(source, signature, signature + f'''
  const int static_shape_mode = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(static_shape_mode == 0 || static_shape_mode == 1, "{FLAG} must be 0 or 1");
''')
    source = replace_once(source, "  auto compute = [&]() {\n    if (dense_bitcast_route && block32_mode == 1) {",
        '''  auto compute = [&]() {
    if (dense_bitcast_route && block32_mode == 1 && static_shape_mode == 1) {
      return ApplyDenseStaticShapeMatMulNBits(a, b, scales, zero_points, bias,
          K_, N_, block_size_, accuracy_level_, bits_, context, y, 0);
    }
    if (dense_bitcast_route && block32_mode == 1) {''')
    return replace_once(source,
        "dense_bitcast_route ? (block32_mode == 1 ? 3 : 1 + dense_tile4) : 0);",
        "dense_bitcast_route ? (block32_mode == 1 ? (static_shape_mode == 1 ? 4 : 3) : 1 + dense_tile4) : 0);")


if __name__ == "__main__":
    original = COMPONENT.parent / "block32-template-v1/dense_block32_template.inc"
    component = json.loads((COMPONENT / "build-manifest.json").read_text())
    assert sha(original) == component["source_sha256"]["../block32-template-v1/dense_block32_template.inc"]
    includes = OUTPUT.parent / "static-shape-template-v1"
    includes.mkdir(parents=True, exist_ok=False)
    header = includes / "dense_static_shape_template.inc"
    header.write_text(static_template(original.read_text()))
    build(output=OUTPUT, flag=FLAG, source_transform=transform,
          component_dir=COMPONENT, source_name="matmul_nbits.cc", parent_dir=PARENT,
          comparator_flag="ORT_WEBGPU_DENSE_BLOCK32",
          comparator_transform=lambda source: transform_compare(source, flag=FLAG),
          extra_include_dirs=(Path("/workspace/qwen38-int4-dense-bitcast-unpack/build-v1"),
                              original.parent, includes))
    path = OUTPUT / "build-manifest.json"
    report = json.loads(path.read_text())
    report["source_sha256"]["../static-shape-template-v1/dense_static_shape_template.inc"] = sha(header)
    report.update(original_block32_template_sha256=sha(original),
                  specialized_integer_dimensions=list(DIMENSIONS),
                  original_floating_point_expressions_and_dispatch_preserved=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
