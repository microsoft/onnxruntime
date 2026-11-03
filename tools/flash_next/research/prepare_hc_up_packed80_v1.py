import json
from pathlib import Path

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent


def build_source(source):
    source = source.replace("qwen38-int4-hc-up-zero-tail/build-v1", "qwen38-int4-hc-up-packed80/build-v1")
    source = source.replace("hc_up_zero_tail_transform", "hc_up_packed80_transform")
    source = source.replace("hc_up_zero_tail_native.cpp", "hc_up_packed80_native.cpp")
    source = replace_once(source, '"physical_threads": 128', '"physical_threads": 80')
    source = replace_once(source, '"shared_storage_bytes": 528', '"shared_storage_bytes": 544')
    return replace_once(source, '"positive_zero_tail_collapsed_22_to_1": True',
                         '"original_32_term_fold_preserved": True, "partial_row_stride": 33')


def controller_source(source):
    source = source.replace("qwen38-int4-hc-up-zero-tail/build-v1", "qwen38-int4-hc-up-packed80/build-v1")
    source = source.replace('DATA = BUILD.parent / "actual-screen-v3"',
                            'DATA = BUILD.parent / "actual-screen-v1"')
    source = source.replace('VARIANT = "hc-up-zero-tail-v3"', 'VARIANT = "hc-up-packed80-v1"')
    source = source.replace("hc_up_zero_tail_transform", "hc_up_packed80_transform")
    source = replace_once(source, 'manifest["physical_threads"] != 128', 'manifest["physical_threads"] != 80')
    source = replace_once(source, 'manifest["shared_storage_bytes"] != 528',
                          'manifest["shared_storage_bytes"] != 544')
    source = replace_once(source, '    name = NAME if selected else "HCUpPreMixPacked2F16"',
                          '    name = NAME if selected else "HCUpPreMixPacked2F16"\n'
                          '    threads = 80 if selected else 128')
    source = replace_once(source, '":128,1,1:" not in', 'f":{threads},1,1:" not in')
    return source.replace("hc_up_zero_tail_actual_operator_screen_completed",
                           "hc_up_packed80_actual_operator_screen_completed")


def main():
    parents = (ROOT / "build_hc_up_zero_tail_v1.py", ROOT / "run_hc_up_zero_tail_v3.py",
               ROOT / "hc_up_zero_tail_native.cpp")
    outputs = (ROOT / "build_hc_up_packed80_v1.py", ROOT / "run_hc_up_packed80_v1.py",
               ROOT / "hc_up_packed80_native.cpp")
    report = ROOT / "hc-up-packed80-v1-generation.json"
    if report.exists() or any(path.exists() for path in outputs):
        raise FileExistsError("Fresh 80-thread HC-up generation required")
    sources = (build_source(parents[0].read_text()), controller_source(parents[1].read_text()),
               parents[2].read_text().replace("ORT_WEBGPU_HC_UP_ZERO_TAIL", "ORT_WEBGPU_HC_UP_PACKED80")
               .replace("hc-up-zero-tail-native", "hc-up-packed80-native"))
    for output, source in zip(outputs, sources, strict=True):
        if output.suffix == ".py":
            compile(source, str(output), "exec")
        with output.open("x") as stream:
            stream.write(source)
    with report.open("x") as stream:
        json.dump({"status": "generated_not_qualified",
                   "reason": "Zero-tail pruning failed its GPU gate. Test distinct 80-thread all-active geometry with padded 33-half rows; retain all original 32 additions and reuse immutable actual-weight fixtures.",
                   "source_sha256": {str(path): sha(path) for path in
                                     (Path(__file__), ROOT / "hc_up_packed80_transform.py", *parents, *outputs)}},
                  stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
