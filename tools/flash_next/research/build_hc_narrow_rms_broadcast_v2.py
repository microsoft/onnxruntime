from pathlib import Path

from build_hc_narrow_rms_broadcast import FLAG, SOURCE, transform as prototype_transform
from build_qmoe_node_replay import replace_once
from build_scoped_webgpu_source import build


OUTPUT = Path("/workspace/qwen38-int4-hc-narrow-rms-broadcast/build-v2")


def transform(source):
    return replace_once(prototype_transform(source),
                        "}, 0, rms_broadcast);", "});")


if __name__ == "__main__":
    build(OUTPUT, SOURCE, transform, SOURCE.parent / "native_compare.cpp", FLAG,
          dict(diagnostic_only=False, model_eligible=False,
               physical_threads=128, original_rms_streams=256, output_columns=4,
               original_rms_tree_preserved=True, shared_rms_widths=[128, 64, 32],
               subgroup_rms_widths=[16, 8, 4, 2, 1],
               subgroup_result_broadcast=True, shared_storage_bytes=27392,
               process_static_selector=True, original_replay_api_unchanged=True,
               original_provider_full_kernel_comparison_required=True))
