import json
from pathlib import Path

from build_dense_double_buffer import private_function, private_program
from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once, sha


FLAG = "ORT_WEBGPU_DENSE_BLOCK32"
OUTPUT = Path("/workspace/qwen38-int4-dense-block32/build-v1")
COMPONENT = Path("/workspace/qwen38-int4-dense-tile4/build-v1")
TEMPLATE = Path("/workspace/qwen38-int4-dense-bitcast-unpack/build-v1")


def block32_template(source):
    source = replace_once(source, "Status ApplyDenseBitcastTemplate(", "Status ApplyDenseBlock32Template(")
    source = replace_once(source, 'ss << ") / uniforms.block_size;\\n";', 'ss << ") / 32u;\\n";')
    source = replace_once(source,
        'ss << "  let batch = workgroup_idx / (uniforms.dispatch_M * uniforms.num_N_tile);\\n";',
        'ss << "  let batch = 0u;\\n";')
    source = replace_once(source,
        'ss << "  let a_global = (workgroup_idx / uniforms.num_N_tile) % uniforms.dispatch_M;\\n";',
        'ss << "  let a_global = 0u;\\n";')
    return replace_once(source,
        'ss << "  let b_global_base = (workgroup_idx % uniforms.num_N_tile) * ";',
        'ss << "  let b_global_base = workgroup_idx * ";')


def transform(source):
    program = private_program(source).replace("DoubleBuffer", "Block32")
    function = private_function(source).replace("DoubleBuffer", "Block32")
    source = replace_once(source, '#include "dense_bitcast_template.inc"',
        '#include "dense_bitcast_template.inc"\n#include "dense_block32_template.inc"')
    stop = source.index("\n}\n\nStatus MatMulNBitsProgram::")
    source = source[:stop] + "\n\n" + program + source[stop:]
    stop = source.index("\n}\n\n}  // namespace webgpu", source.rindex("Status ApplyDenseTile4MatMulNBits("))
    source = source[:stop] + "\n\n" + function + source[stop:]
    signature = "Status MatMulNBits::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {"
    source = replace_once(source, signature, signature + f'''
  const int block32_mode = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(block32_mode == 0 || block32_mode == 1, "{FLAG} must be 0 or 1");
''')
    source = replace_once(source, "  auto compute = [&]() {\n    if (dense_bitcast_route && dense_tile4 == 1) {",
        '''  auto compute = [&]() {
    if (dense_bitcast_route && block32_mode == 1) {
      return ApplyDenseBlock32MatMulNBits(a, b, scales, zero_points, bias,
          K_, N_, block_size_, accuracy_level_, bits_, context, y, 0);
    }
    if (dense_bitcast_route && dense_tile4 == 1) {''')
    return replace_once(source, "compute, 0, dense_bitcast_route ? 1 + dense_tile4 : 0);",
        "compute, 0, dense_bitcast_route ? (block32_mode == 1 ? 3 : 1 + dense_tile4) : 0);")


if __name__ == "__main__":
    original = TEMPLATE / "dense_bitcast_template.inc"
    assert sha(original) == json.loads((TEMPLATE / "build-manifest.json").read_text())["source_sha256"][original.name]
    includes = OUTPUT.parent / "block32-template-v1"
    includes.mkdir(parents=True, exist_ok=False)
    header = includes / "dense_block32_template.inc"
    header.write_text(block32_template(original.read_text()))
    build(output=OUTPUT, flag=FLAG, source_transform=transform,
          component_dir=COMPONENT, source_name="matmul_nbits.cc",
          comparator_flag="ORT_WEBGPU_DENSE_TILE4", comparator_transform=lambda source: source,
          parent_dir=Path("/workspace/qwen38-int4-qmoe-prefill1024/build-v2"),
          extra_include_dirs=(TEMPLATE, includes))
    path = OUTPUT / "build-manifest.json"
    report = json.loads(path.read_text())
    report["source_sha256"]["../block32-template-v1/dense_block32_template.inc"] = sha(header)
    report["canonical_single_row_block32_integer_indexing_specialized"] = True
    path.write_text(json.dumps(report, indent=2) + "\n")
