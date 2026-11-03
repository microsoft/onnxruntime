import json
from pathlib import Path

from build_dense_direct_a import BITCAST_TEMPLATE, BLOCK32_TEMPLATE, COMPONENT, transform as clone
from build_dense_double_buffer import transform_compare
from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once, sha


FLAG = "ORT_WEBGPU_DENSE_QKV_THREADS64"
OUTPUT = Path("/workspace/qwen38-int4-dense-qkv-threads64/build-v1")
PARENT = Path("/workspace/qwen38-int4-qmoe-prefill-rows4/build-v2")
PROGRAM = "MatMulNBitsQkvThreads64F16"


def transform(source):
    source = clone(source, flag=FLAG, name="QkvThreads64", suffix="qkv_threads64", replay_variant=21)
    begin = source.rindex("Status ApplyDenseQkvThreads64MatMulNBits(")
    end = source.index("\n}\n\n}  // namespace webgpu", begin)
    function = replace_once(source[begin:end],
                            "constexpr uint32_t workgroup_size = 128;\n  constexpr uint32_t tile_size = 4;",
                            "constexpr uint32_t workgroup_size = 64;\n  constexpr uint32_t tile_size = 4;")
    source = source[:begin] + function + source[end:]
    return replace_once(source,
        "qkv_threads64_mode == 1 && a->IsDataType<MLFloat16>() && !context.EnableMatmulFp32Accumulation();",
        """qkv_threads64_mode == 1 && a->IsDataType<MLFloat16>() && !context.EnableMatmulFp32Accumulation() &&
      K_ == 2560 && N_ == 16384 &&
      context.DeviceLimits().maxComputeWorkgroupSizeX >= 64 &&
      context.DeviceLimits().maxComputeInvocationsPerWorkgroup >= 64 &&
      context.DeviceLimits().maxComputeWorkgroupStorageSize >= 2304 &&
      context.DeviceLimits().maxComputeWorkgroupsPerDimension >= 4096;""")


def main():
    original = BLOCK32_TEMPLATE / "dense_block32_template.inc"
    frozen = json.loads((COMPONENT / "build-manifest.json").read_text())
    assert sha(original) == frozen["source_sha256"]["../block32-template-v1/dense_block32_template.inc"]
    includes = OUTPUT.parent / "qkv-threads64-template-v1"
    includes.mkdir(parents=True, exist_ok=False)
    header = includes / "dense_qkv_threads64_template.inc"
    header.write_text(replace_once(original.read_text(), "Status ApplyDenseBlock32Template(",
                                  "Status ApplyDenseQkvThreads64Template("))
    build(output=OUTPUT, flag=FLAG, source_transform=transform,
          component_dir=PARENT, source_name="matmul_nbits.cc",
          comparator_dir=COMPONENT, comparator_flag="ORT_WEBGPU_DENSE_BLOCK32",
          comparator_transform=lambda source: transform_compare(source, FLAG),
          parent_dir=PARENT, extra_include_dirs=(BITCAST_TEMPLATE, BLOCK32_TEMPLATE, includes))
    path = OUTPUT / "build-manifest.json"
    manifest = json.loads(path.read_text())
    manifest["source_sha256"]["../qkv-threads64-template-v1/dense_qkv_threads64_template.inc"] = sha(header)
    manifest.update(original_block32_template_sha256=sha(original),
                    logical_shape={"M": 1, "K": 2560, "N": 16384},
                    workgroup_threads=64, output_columns_per_workgroup=4,
                    physical_column_groups=2, output_columns_per_thread=2,
                    k_lanes_per_column=32, original_partial_slots=32,
                    original_fp16_dot_expressions_lane_updates_and_serial_fold_preserved=True,
                    row4_prefill_preserved=True, compressed_weights_preserved=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
