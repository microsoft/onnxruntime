import json
from pathlib import Path

from build_dense_double_buffer import transform_compare
from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once, sha


FLAG = "ORT_WEBGPU_DENSE_LANE8"
OUTPUT = Path("/workspace/qwen38-int4-dense-lane8/build-v2")
BASE_COMPONENT = Path("/workspace/qwen38-int4-dense-block32/build-v1")
TEMPLATE = Path("/workspace/qwen38-int4-dense-block32/block32-template-v1")
PARENT = Path("/workspace/qwen38-int4-qmoe-prefill-rows4/build-v2")
INCLUDES = OUTPUT.parent / "lane8-template-v2"
LANE_EXPRESSION = "(local_idx % 4u) * 32u + (local_idx / 4u) % 8u + (local_idx / 32u) * 8u"
SELECTOR = f'''
  const int dense_lane8_mode = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(dense_lane8_mode == 0 || dense_lane8_mode == 1, "{FLAG} must be 0 or 1");
  const bool dense_lane8 = dense_lane8_mode == 1 &&
      !context.EnableMatmulFp32Accumulation() &&
      context.AdapterInfo().vendor == std::string_view{{"nvidia"}};
'''
DISPATCH = '''
    if (dense_bitcast_route && block32_mode == 1 && dense_lane8) {
      return ApplyDenseLane8MatMulNBits(a, b, scales, zero_points, bias,
          K_, N_, block_size_, accuracy_level_, bits_, context, y, 0);
    }
'''


def lane8_template(original):
    result = replace_once(original, "Status ApplyDenseBlock32Template(",
                          "Status ApplyDenseLane8Template(")
    result = replace_once(result, 'ss << "  let idx = local_idx % ";',
                          f'ss << "  let logical_lane = {LANE_EXPRESSION};\\n";\n'
                          'ss << "  let idx = logical_lane % ";')
    return replace_once(result, 'ss << "  let idy = local_idx / ";',
                        'ss << "  let idy = logical_lane / ";')


def private_parts(original):
    start = original.index("class DenseBlock32Program")
    end = original.index("\n}\n\nStatus MatMulNBitsProgram::", start)
    program = original[start:end].replace("DenseBlock32", "DenseLane8").replace(
        "MatMulNBitsBlock32F16", "MatMulNBitsLane8F16")
    start = original.rindex("Status ApplyDenseBlock32MatMulNBits(")
    end = original.index("\n}\n\n}  // namespace webgpu", start)
    function = original[start:end].replace("DenseBlock32", "DenseLane8")
    return program, function


def transform(original):
    program, function = private_parts(original)
    result = replace_once(original, '#include "dense_block32_template.inc"',
                          '#include "dense_block32_template.inc"\n#include "dense_lane8_template.inc"')
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
                        "(block32_mode == 1 ? (dense_lane8 ? 4 : 3) : 1 + dense_tile4)")


def main():
    template = TEMPLATE / "dense_block32_template.inc"
    manifest = json.loads((BASE_COMPONENT / "build-manifest.json").read_text())
    assert sha(template) == manifest["source_sha256"]["../block32-template-v1/dense_block32_template.inc"]
    INCLUDES.mkdir(parents=True, exist_ok=False)
    generated = INCLUDES / "dense_lane8_template.inc"
    generated.write_text(lane8_template(template.read_text()))
    build(output=OUTPUT, flag=FLAG, source_transform=transform,
          component_dir=PARENT, source_name="matmul_nbits.cc", comparator_dir=BASE_COMPONENT,
          comparator_flag="ORT_WEBGPU_DENSE_BLOCK32",
          comparator_transform=lambda source: transform_compare(source, flag=FLAG),
          parent_dir=PARENT, extra_include_dirs=(INCLUDES,))
    path = OUTPUT / "build-manifest.json"
    report = json.loads(path.read_text())
    report["source_sha256"]["../lane8-template-v2/dense_lane8_template.inc"] = sha(generated)
    report["frozen_block32_template_sha256"] = sha(template)
    report["physical_layout"] = dict(workgroup_size=128, outputs=4, k_lanes_per_output=32,
                                     physical_warp_outputs=4, physical_warp_k_lanes=8,
                                     logical_lane_expression=LANE_EXPRESSION)
    report["dot_and_partial_reduction_order_unchanged"] = True
    report["default_enabled"] = False
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
