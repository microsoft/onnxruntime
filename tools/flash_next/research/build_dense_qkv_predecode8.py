import json
from pathlib import Path

import build_dense_predecode as original
from build_dense_double_buffer import transform_compare
from build_hc_bitcast_unpack import main as build
from build_hc_staged_projection import REPLAY, stage_replay
from build_qmoe_node_replay import replace_once, sha


FLAG = "ORT_WEBGPU_DENSE_QKV_PREDECODE8"
OUTPUT = Path("/workspace/qwen38-int4-dense-qkv-predecode8/build-v1")
PARENT = Path("/workspace/qwen38-int4-qmoe-prefill-rows4/build-v2")
PROGRAM = "MatMulNBitsQkvPredecoded8F16"
STATS = "DenseQkvPredecoded8"
INCLUDES = OUTPUT.parent / "qkv-predecoded8-includes-v1"


def transform(source, header):
    source = original.transform(source, header)
    source = source.replace(original.FLAG, FLAG)
    source = replace_once(source, "MatMulNBitsPredecodedF16", PROGRAM)
    source = replace_once(source, '"DensePredecoded", {0, 2}', f'"{STATS}", {{0, 2}}')
    setup = replace_once(original.SETUP,
        "DensePredecodedProgram program{4, 4, false, false, false, false, false, 32, false, false};",
        "DensePredecodedProgram program{8, 4, false, false, false, false, false, 32, false, false};")
    setup = setup.replace("SetWorkgroupSize(128)", "SetWorkgroupSize(256)")
    setup = setup.replace("(N + 3u) / 4u", "(N + 7u) / 8u")
    source = replace_once(source, original.SETUP, setup)
    return replace_once(source,
        "predecode_mode == 1 && dense_bitcast_route && block32_mode == 1 && weights_constant_",
        """predecode_mode == 1 && dense_bitcast_route && block32_mode == 1 && weights_constant_ &&
      K_ == 2560 && N_ == 16384 &&
      context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
      context.DeviceLimits().maxComputeWorkgroupSizeX >= 256 &&
      context.DeviceLimits().maxComputeInvocationsPerWorkgroup >= 256 &&
      context.DeviceLimits().maxComputeWorkgroupStorageSize >= 2560 &&
      context.DeviceLimits().maxComputeWorkgroupsPerDimension >= 2048""")


def main():
    component = json.loads((original.COMPONENT / "build-manifest.json").read_text())
    assert sha(original.TEMPLATE) == component["source_sha256"]["../block32-template-v1/dense_block32_template.inc"]
    replay_manifest = json.loads((REPLAY.parent.parent / "build-manifest.json").read_text())
    assert sha(REPLAY) == replay_manifest["source_sha256"]["overlay/static_node_replay.h"]
    INCLUDES.mkdir(parents=True, exist_ok=False)
    template = INCLUDES / "dense_predecoded_template.inc"
    template.write_text(original.predecoded_template(original.TEMPLATE.read_text()))
    replay = INCLUDES / "dense_predecoded_replay.h"
    replay.write_text(stage_replay(REPLAY.read_text()).replace("HCStageReplay", "DensePredecodedReplay"))
    build(output=OUTPUT, flag=FLAG,
          source_transform=lambda source: transform(source, original.HEADER.read_text()),
          component_dir=PARENT, source_name="matmul_nbits.cc",
          comparator_dir=original.COMPONENT, comparator_flag="ORT_WEBGPU_DENSE_BLOCK32",
          comparator_transform=lambda source: transform_compare(source, flag=FLAG),
          parent_dir=PARENT, extra_include_dirs=(original.TEMPLATE.parent, INCLUDES))
    path = OUTPUT / "build-manifest.json"
    report = json.loads(path.read_text())
    report.update(
        original_matmul_kernel_header_sha256=sha(original.HEADER),
        constant_weights_and_scales_only=True,
        original_kernel_class_unchanged=True,
        private_kernel_owns_decoded_weights=True,
        original_dot_order_activation_staging_and_serial_fold_preserved=True,
        no_fp32_accumulation_or_tensorcore_changes=True,
        logical_shape={"M": 1, "K": 2560, "N": 16384},
        workgroup_threads=256, output_columns_per_workgroup=8,
        k_lanes_per_column=32, shared_bytes=2560,
        per_initializer_cache_bytes=2560 * 16384 * 2,
        row4_prefill_preserved=True)
    for item in (template, replay):
        report["source_sha256"]["../qkv-predecoded8-includes-v1/" + item.name] = sha(item)
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
