import json
from pathlib import Path

from build_qmoe_node_replay import replace_once, sha
from prepare_dense_two_long_k_ns_v1 import controller_source


ROOT = Path(__file__).parent


def builder_source(source):
    source = replace_once(source, "long-k-timestamp-ns/build-v1", "long-k-timestamp-ns/build-v2")
    source = replace_once(source, 'OUTPUT / "original-control-relink.so"',
                          'OUTPUT / "original-control" / "libonnxruntime_providers_webgpu.so"')
    source = replace_once(source, 'OUTPUT / "unmodified-candidate-relink.so"',
                          'OUTPUT / "unmodified-candidate" / "libonnxruntime_providers_webgpu.so"')
    return replace_once(source, '        command = manifest["link_command"].copy()',
                        '        library.parent.mkdir(parents=True, exist_ok=True)\n'
                        '        command = manifest["link_command"].copy()')


def main():
    output = ROOT / "dense-two-long-k-ns-v2-generation.json"
    builder_parent = ROOT / "build_dense_two_long_k_ns_v1.py"
    controller_parent = ROOT / "run_dense_tile_parallel2_long_k_v1.py"
    builder = ROOT / "build_dense_two_long_k_ns_v2.py"
    controller = ROOT / "run_dense_two_long_k_ns_v2.py"
    if any(path.exists() for path in (output, builder, controller)):
        raise FileExistsError("Fresh corrected-link packaging generation required")
    sources = {
        builder: builder_source(builder_parent.read_text()),
        controller: controller_source(controller_parent.read_text()).replace(
            "long-k-timestamp-ns/build-v1", "long-k-timestamp-ns/build-v2").replace(
                "dense-tile-parallel2-long-k-ns-v1", "dense-tile-parallel2-long-k-ns-v2"),
    }
    for path, source in sources.items():
        compile(source, str(path), "exec")
    for path, source in sources.items():
        with path.open("x") as stream:
            stream.write(source)
    report = {
        "status": "generated_not_qualified",
        "v1_failure": "Original-control re-link used a different output basename; GNU ELF base-version dynamic string changed. No GPU launch or shader/precision failure.",
        "repair": "Retain exact original library basename in independent control/candidate proof subdirectories. Never weaken byte-identical relink requirement.",
        "source_sha256": {str(path): sha(path) for path in
                          (Path(__file__), builder_parent, controller_parent,
                           ROOT / "prepare_dense_two_long_k_ns_v1.py", builder, controller)},
    }
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
