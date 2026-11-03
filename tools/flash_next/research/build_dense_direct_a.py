import json
from pathlib import Path
import re

from build_hc_bitcast_unpack import main as build
from build_dense_double_buffer import transform_compare
from build_qmoe_node_replay import replace_once, sha


FLAG = "ORT_WEBGPU_DENSE_DIRECT_A"
OUTPUT = Path("/workspace/qwen38-int4-dense-direct-a/build-v1")
PARENT = Path("/workspace/qwen38-int4-dense-block32/build-v2")
COMPONENT = Path("/workspace/qwen38-int4-dense-block32/build-v1")
BITCAST_TEMPLATE = Path("/workspace/qwen38-int4-dense-bitcast-unpack/build-v1")
BLOCK32_TEMPLATE = COMPONENT.parent / "block32-template-v1"


def direct_a_template(source):
    source = replace_once(source, "Status ApplyDenseBlock32Template(", "Status ApplyDenseDirectATemplate(")
    source = replace_once(source, "  auto& __param_a_length_per_tile = params.param_a_length_per_tile;\n", "")
    source = replace_once(source, '''ss << "var<workgroup> tile_A : array<input_a_value_t, ";
ss << __param_a_length_per_tile;
ss << ">;\\n";''', "")
    begin, end = source.index("//  94 | fn loadSHMA("), source.index("// 104 | $MAIN {")
    source = source[:begin] + '''ss << "fn read_dense_a(index: u32, kidx: u32) -> input_a_value_t {\\n"
   << "  return " << __var_a->GetByOffset("kidx / 4u + index") << ";\\n"
   << "}\\n";
''' + source[end:]
    begin, end = source.index("// 148 |     for (var id ="), source.index("// 153 |")
    source = source[:begin] + source[end:]
    source = replace_once(source, '// 298 |     workgroupBarrier();\nss << "    workgroupBarrier();\\n";\n// 299 |   }\nss << "  }\\n";',
                          'ss << "  }\\n";\nss << "  workgroupBarrier();\\n";')
    source, count = re.subn(r"tile_A\[(a_offset(?: \+ \d+)?)\]",
                           r"read_dense_a(\1, kidx)", source)
    assert count > 30
    actual = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("//"))
    assert "tile_A" not in actual and "loadSHMA" not in actual
    assert actual.count('workgroupBarrier();\\n') == 1
    return actual + "\n"


def transform(source, flag=FLAG, name="DirectA", suffix="direct_a", replay_variant=5):
    begin = source.index("class DenseBlock32Program")
    end = source.index("\n}\n\nStatus MatMulNBitsProgram::", begin)
    program = source[begin:end].replace("DenseBlock32", f"Dense{name}").replace(
        "MatMulNBitsBlock32F16", f"MatMulNBits{name}F16")
    begin = source.rindex("Status ApplyDenseBlock32MatMulNBits(")
    end = source.index("\n}\n\n}  // namespace webgpu", begin)
    function = source[begin:end].replace("DenseBlock32", f"Dense{name}")
    source = replace_once(source, '#include "dense_block32_template.inc"',
                          f'#include "dense_block32_template.inc"\n#include "dense_{suffix}_template.inc"')
    end = source.index("\n}\n\nStatus MatMulNBitsProgram::")
    source = source[:end] + "\n\n" + program + source[end:]
    end = source.index("\n}\n\n}  // namespace webgpu", source.rindex("Status ApplyDenseBlock32MatMulNBits("))
    source = source[:end] + "\n\n" + function + source[end:]
    signature = "Status MatMulNBits::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {"
    source = replace_once(source, signature, signature + f'''
  const int {suffix}_mode = ParseEnvironmentVariableWithDefault<int>("{flag}", 0);
  ORT_RETURN_IF_NOT({suffix}_mode == 0 || {suffix}_mode == 1, "{flag} must be 0 or 1");
''')
    anchor = "  auto compute = [&]() {\n    if (dense_bitcast_route && block32_mode == 1) {"
    source = replace_once(source, anchor, f'''  const bool {suffix}_route = dense_bitcast_route && block32_mode == 1 &&
      {suffix}_mode == 1 && a->IsDataType<MLFloat16>() && !context.EnableMatmulFp32Accumulation();
  auto compute = [&]() {{
    if ({suffix}_route) {{
      return ApplyDense{name}MatMulNBits(a, b, scales, zero_points, bias,
          K_, N_, block_size_, accuracy_level_, bits_, context, y, 0);
    }}
    if (dense_bitcast_route && block32_mode == 1) {{''')
    return replace_once(source,
        "compute, 0, dense_bitcast_route ? (block32_mode == 1 ? 3 : 1 + dense_tile4) : 0);",
        f"compute, 0, dense_bitcast_route ? ({suffix}_route ? {replay_variant} : block32_mode == 1 ? 3 : 1 + dense_tile4) : 0);")


if __name__ == "__main__":
    original = BLOCK32_TEMPLATE / "dense_block32_template.inc"
    frozen = json.loads((COMPONENT / "build-manifest.json").read_text())
    assert sha(original) == frozen["source_sha256"]["../block32-template-v1/dense_block32_template.inc"]
    includes = OUTPUT.parent / "direct-a-template-v1"
    includes.mkdir(parents=True, exist_ok=False)
    header = includes / "dense_direct_a_template.inc"
    header.write_text(direct_a_template(original.read_text()))
    build(output=OUTPUT, flag=FLAG, source_transform=transform,
          component_dir=COMPONENT, source_name="matmul_nbits.cc",
          comparator_flag="ORT_WEBGPU_DENSE_BLOCK32",
          comparator_transform=lambda source: transform_compare(source, FLAG),
          parent_dir=PARENT, extra_include_dirs=(BITCAST_TEMPLATE, BLOCK32_TEMPLATE, includes))
    manifest = OUTPUT / "build-manifest.json"
    report = json.loads(manifest.read_text())
    report["source_sha256"]["../direct-a-template-v1/dense_direct_a_template.inc"] = sha(header)
    report["fp16_canonical_direct_activation_reads"] = True
    report["original_shared_partial_rounding_and_serial_fold_preserved"] = True
    report["no_activation_shared_array"] = True
    report["one_final_workgroup_barrier"] = True
    manifest.write_text(json.dumps(report, indent=2) + "\n")
