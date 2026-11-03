import json
from pathlib import Path

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent


def build_source(source):
    source = source.replace("qwen38-int4-hc-up-packed80/build-v1", "qwen38-int4-hc-up-packed160/build-v1")
    source = source.replace("hc_up_packed80", "hc_up_packed160")
    source = replace_once(source, '"physical_threads": 80', '"physical_threads": 160')
    source = replace_once(source, '"shared_storage_bytes": 544', '"shared_storage_bytes": 1088')
    source = replace_once(source, '"output_columns": 2', '"output_columns": 4')
    return replace_once(source, 'ROOT / "hc_up_packed160_transform.py", native_source,',
                         'ROOT / "hc_up_packed160_transform.py", ROOT / "hc_up_packed80_transform.py", native_source,')


def controller_source(source):
    source = source.replace("qwen38-int4-hc-up-packed80/build-v1", "qwen38-int4-hc-up-packed160/build-v1")
    source = source.replace("hc-up-packed80-v1", "hc-up-packed160-v1")
    source = source.replace("hc_up_packed80", "hc_up_packed160")
    source = replace_once(source, "threads = 80 if selected else 128", "threads = 160 if selected else 128")
    source = replace_once(source, 'manifest["physical_threads"] != 80', 'manifest["physical_threads"] != 160')
    return replace_once(source, 'manifest["shared_storage_bytes"] != 544',
                         'manifest["shared_storage_bytes"] != 1088')


def main():
    parents = (ROOT / "build_hc_up_packed80_v1.py", ROOT / "run_hc_up_packed80_v1.py",
               ROOT / "hc_up_packed80_native.cpp")
    outputs = (ROOT / "build_hc_up_packed160_v1.py", ROOT / "run_hc_up_packed160_v1.py",
               ROOT / "hc_up_packed160_native.cpp")
    report = ROOT / "hc-up-packed160-v1-generation.json"
    if report.exists() or any(path.exists() for path in outputs):
        raise FileExistsError("Fresh 160-thread/four-column HC-up generation required")
    sources = (build_source(parents[0].read_text()), controller_source(parents[1].read_text()),
               parents[2].read_text().replace("ORT_WEBGPU_HC_UP_PACKED80", "ORT_WEBGPU_HC_UP_PACKED160")
               .replace("hc-up-packed80-native", "hc-up-packed160-native"))
    for output, source in zip(outputs, sources, strict=True):
        if output.suffix == ".py":
            compile(source, str(output), "exec")
        with output.open("x") as stream:
            stream.write(source)
    with report.open("x") as stream:
        json.dump({"status": "generated_not_qualified",
                   "reason": "80-thread/two-column screen gained 3.34% but only 78/128 calls improved. Test four columns in five complete active warps, versus six partly occupied warps for two 80-thread groups, retaining exact dot/fold and padded partial layout.",
                   "source_sha256": {str(path): sha(path) for path in
                                     (Path(__file__), ROOT / "hc_up_packed160_transform.py",
                                      ROOT / "hc_up_packed80_transform.py", *parents, *outputs)}},
                  stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
