import json
import os
from pathlib import Path
import subprocess

from build_hc_bitcast_unpack import COMPONENT
from build_hc_post_norm_small_cta import FLAG, PARENT, STORAGE_BYTES, transform
from build_qmoe_node_replay import replace_once, sha


OUTPUT = Path("/workspace/qwen38-int4-hc-post-norm-small-cta/build-v2")
REPLAY = Path("/workspace/qwen38-int4-static-node-replay/build-v8/overlay/static_node_replay.h")


def private_replay(source):
    source = source.replace("StaticNodeReplay", "HcSmallCtaStaticNodeReplay")
    source = replace_once(source, "int64_t row_elements = 0) {",
                          "int64_t row_elements = 0, int64_t variant = 0) {")
    return replace_once(source, "    key.reserve(96);",
                        "    key.reserve(96);\n    key.push_back(variant);")


def candidate_source(source):
    source = transform(source)
    source = replace_once(source, '#include "static_node_replay.h"',
                          '#include "static_node_replay.h"\n#include "hc_small_cta_replay.h"')
    source = replace_once(source, "  mutable StaticNodeReplay replay_;",
                          "  mutable HcSmallCtaStaticNodeReplay replay_;")
    return replace_once(source, "using namespace onnxruntime::webgpu;",
        "using namespace onnxruntime::webgpu;\n"
        "static_assert(sizeof(HcSmallCtaStaticNodeReplay) == sizeof(StaticNodeReplay));")


def main():
    component = json.loads((COMPONENT / "build-manifest.json").read_text())
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    original = COMPONENT / "hc_post_norm_projection.cc"
    if sha(original) != component["source_sha256"][original.name]:
        raise RuntimeError("Frozen HC source changed")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    scratch = OUTPUT / "scratch"
    scratch.mkdir()
    environment = dict(os.environ, TMPDIR=str(scratch))
    command = next(command.copy() for command in component["compile_commands"]
                   if command[command.index("-c") + 1] == str(original))
    original_object = COMPONENT / (original.name + ".o")
    baseline_object = OUTPUT / "hc.baseline.cc.o"
    command[command.index("-o") + 1] = str(baseline_object)
    subprocess.run(command, cwd=parent["working_directory"], env=environment, check=True)
    if sha(baseline_object) != sha(original_object):
        raise RuntimeError("Exact original-header HC object proof failed; no GPU admission")
    library = OUTPUT / "libonnxruntime_providers_webgpu.so"
    link = list(parent["link_command"])
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    if sha(library) != parent["provider_sha256"]:
        raise RuntimeError("Combined parent relink proof failed")
    library.unlink()
    header = OUTPUT / "hc_small_cta_replay.h"
    header.write_text(private_replay(REPLAY.read_text()))
    source = OUTPUT / original.name
    source.write_text(candidate_source(original.read_text()))
    obj = OUTPUT / (source.name + ".o")
    command[command.index("-c") + 1] = str(source)
    command[command.index("-o") + 1] = str(obj)
    subprocess.run(command, cwd=parent["working_directory"], env=environment, check=True)
    if link.count(str(original_object)) != 1:
        raise RuntimeError("Combined parent does not contain exactly one original HC object")
    link[link.index(str(original_object))] = str(obj)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    comparator = OUTPUT / "native_compare.cpp"
    comparator.write_text(replace_once((COMPONENT / comparator.name).read_text(),
        'setenv("ORT_WEBGPU_HC_POST_NORM_PROJECTION", value, 1)', f'setenv("{FLAG}", value, 1)'))
    subprocess.run(["c++", "-O3", "-std=c++20", str(comparator),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(OUTPUT / "native_compare")], env=environment, check=True)
    sources = [source, header, comparator, original, REPLAY, Path(__file__),
               Path(__file__).with_name("build_hc_post_norm_small_cta.py")]
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
                  baseline_provider_sha256=parent["provider_sha256"], baseline_byte_identical=True,
                  original_hc_object_byte_identical=True, original_hc_object_sha256=sha(original_object),
                  compile_commands=[command], link_command=link, working_directory=parent["working_directory"],
                  flag=FLAG, default_enabled=False, declared_workgroup_bytes=STORAGE_BYTES,
                  no_public_layout_changes=True, private_replay_type_and_variant_key=True,
                  no_floating_point_reduction_order_changes=True,
                  source_sha256={str(path): sha(path) for path in sources},
                  comparator_sha256=sha(OUTPUT / "native_compare"),
                  quality_qualified=False, headline_eligible=False)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "provider_sha256", "original_hc_object_byte_identical")}))


if __name__ == "__main__":
    main()
