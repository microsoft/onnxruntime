import json
import os
from pathlib import Path
import shutil
import subprocess

from build_gdn_value2_capture import main as build_capture
from build_qmoe_node_replay import sha


ROOT = Path(__file__).parent
HISTOGRAM = Path("/workspace/qwen38-int4-qsa-hist-rank/build-v2")
TREE = Path("/workspace/qwen38-int4-gdn-tree32/build-v1")
OUTPUT = Path("/workspace/qwen38-int4-gdn-tree32-hist/build-v1")


def stack_link(histogram, tree, output=OUTPUT):
    assert histogram["baseline_provider_sha256"] == tree["baseline_provider_sha256"]
    left, right = histogram["link_command"], tree["link_command"]
    assert len(left) == len(right)
    differences = [(a, b) for a, b in zip(left, right) if a != b]
    assert differences == [
        (str(HISTOGRAM / "libonnxruntime_providers_webgpu.so"),
         str(TREE / "libonnxruntime_providers_webgpu.so")),
        ("/workspace/qwen38-int4-gdn-decode-view/build-original/gated_delta_net.cc.o",
         str(TREE / "gated_delta_net.cc.o")),
        (str(HISTOGRAM / "sparse_attention_indexer.cc.o"),
         "/workspace/qwen38-int4-qsa-prefill-score16/build-v1/sparse_attention_indexer.cc.o"),
    ], "Stack inputs must differ only in the disjoint GDN and ranking objects"
    link = left.copy()
    link[link.index("-o") + 1] = str(output / "libonnxruntime_providers_webgpu.so")
    original = differences[1][0]
    assert link.count(original) == 1
    link[link.index(original)] = str(output / "gated_delta_net.cc.o")
    return link


def main():
    histogram = json.loads((HISTOGRAM / "build-manifest.json").read_text())
    tree = json.loads((TREE / "build-manifest.json").read_text())
    link = stack_link(histogram, tree)
    qualifications = {}
    for variant, directory in (("qsa-hist-rank-v2", HISTOGRAM), ("gdn-tree32-v1", TREE)):
        path = ROOT / f"{variant}-contexts-20261031.qualification.json"
        proof = json.loads(path.read_text())
        assert proof["all_original_sequences_match"] and proof["zero_major_faults"]
        assert proof["provider_sha256"] == sha(directory / "libonnxruntime_providers_webgpu.so")
        for name, digest in proof["source_sha256"].items():
            assert sha(ROOT / name) == digest
        qualifications[path.name] = sha(path)
    for directory, manifest in ((HISTOGRAM, histogram), (TREE, tree)):
        assert sha(directory / "libonnxruntime_providers_webgpu.so") == manifest["provider_sha256"]
        for name, digest in manifest["source_sha256"].items():
            assert sha(directory / name) == digest
    assert tree["baseline_byte_identical"] and tree["original_gdn_object_byte_identical"]
    assert sha(Path("/workspace/qwen38-int4-gdn-decode-view/build-original/gated_delta_net.cc.o")) == \
        tree["original_gdn_object_sha256"]
    assert sha(TREE / "native_compare") == tree["comparator_sha256"]
    OUTPUT.mkdir(parents=True, exist_ok=False)
    sources = {}
    for directory, manifest in ((HISTOGRAM, histogram), (TREE, tree)):
        for name in ("build-manifest.json", *manifest["source_sha256"]):
            path = directory / name
            sources[os.path.relpath(path, OUTPUT)] = sha(path)
    for name in ("gated_delta_net.cc.o", "native_compare.cpp", "native_compare"):
        shutil.copy2(TREE / name, OUTPUT / name)
    sources["native_compare.cpp"] = sha(OUTPUT / "native_compare.cpp")
    sources["gated_delta_net.cc.o"] = sha(OUTPUT / "gated_delta_net.cc.o")
    histogram_object = HISTOGRAM / "sparse_attention_indexer.cc.o"
    sources[os.path.relpath(histogram_object, OUTPUT)] = sha(histogram_object)
    original_link = histogram["link_command"].copy()
    library = OUTPUT / "libonnxruntime_providers_webgpu.so"
    original_link[original_link.index("-o") + 1] = str(library)
    subprocess.run(original_link, cwd=histogram["working_directory"], check=True)
    assert sha(library) == histogram["provider_sha256"], "Histogram parent relink mismatch"
    subprocess.run(link, cwd=histogram["working_directory"], check=True)
    report = dict(tree, status="built_not_qualified", provider_sha256=sha(library),
                  baseline_provider_sha256=histogram["provider_sha256"],
                  link_command=link, compile_commands=[], source_sha256=sources,
                  qualification_source_sha256=qualifications,
                  gdn_component_object_byte_identical=True,
                  histogram_component_object_unchanged=True,
                  only_disjoint_gdn_object_replaced=True,
                  stacked_histogram_and_gdn=True)
    with (OUTPUT / "build-manifest.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    build_capture(OUTPUT)
    print(json.dumps({key: report[key] for key in
                     ("status", "provider_sha256", "baseline_provider_sha256")}), flush=True)


if __name__ == "__main__":
    main()
