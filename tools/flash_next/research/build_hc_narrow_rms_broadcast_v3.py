import json
from pathlib import Path

from build_hc_narrow_rms_broadcast_v2 import FLAG, SOURCE, transform
from build_scoped_webgpu_source import build


OUTPUT = Path("/workspace/qwen38-int4-hc-narrow-rms-broadcast/build-v3")


if __name__ == "__main__":
    report = build(OUTPUT, SOURCE, transform, SOURCE.parent / "native_compare.cpp", FLAG,
                   dict(diagnostic_only=False, model_eligible=False,
                        physical_threads=128, original_rms_streams=256, output_columns=4,
                        original_rms_tree_preserved=True, shared_rms_widths=[128, 64, 32],
                        subgroup_rms_widths=[16, 8, 4, 2, 1],
                        subgroup_result_broadcast=True, shared_storage_bytes=27392,
                        original_replay_api_unchanged=True,
                        original_provider_full_kernel_comparison_required=True))
    report.update(original_compute_and_fallbacks_unchanged=False,
                  original_default_compute_and_fallbacks_unchanged=True)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
