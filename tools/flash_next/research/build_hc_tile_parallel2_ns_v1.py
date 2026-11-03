import json
import os
from pathlib import Path
import subprocess

from build_qmoe_node_replay import sha


ROOT = Path(__file__).parent
OUTPUT = Path("/workspace/qwen38-int4-hc-tile-parallel2-timestamp-ns/build-v1")
CANDIDATE = Path("/workspace/qwen38-int4-hc-tile-parallel2/build-v1")
CONTROL = Path("/workspace/qwen38-int4-gpu-timestamp-ns/build-v1")
ORIGINAL_CONTEXT = "/workspace/qwen38-int4-static-node-replay/build-v8/webgpu_context.cc.o"


def main():
    candidate = json.loads((CANDIDATE / "build-manifest.json").read_text())
    control = json.loads((CONTROL / "build-manifest.json").read_text())
    qualification_path = ROOT / "gpu-timestamp-ns-v1-qualification.json"
    qualification = json.loads(qualification_path.read_text())
    original_sha = "3a283ecfb892613b19a6f49d15b726e928b41027c8b3d838bd8cd1dfed9939d3"
    if (candidate["baseline_provider_sha256"] != original_sha or
            control["baseline_provider_sha256"] != original_sha or
            not candidate["original_hc_object_byte_identical"] or
            not control["original_context_object_byte_identical"] or
            not qualification["precision_qualified"] or
            qualification["provider_sha256"] != control["provider_sha256"]):
        raise RuntimeError("Missing exact compute/metadata provenance and actual precision qualification")
    source_hashes = {}
    for manifest in (candidate, control):
        for path, digest in manifest["source_sha256"].items():
            if sha(Path(path)) != digest:
                raise RuntimeError(f"Changed precise-measurement input: {path}")
            source_hashes[path] = digest
    for directory, manifest in ((CANDIDATE, candidate), (CONTROL, control)):
        if sha(directory / "libonnxruntime_providers_webgpu.so") != manifest["provider_sha256"]:
            raise RuntimeError("Changed candidate or qualified precision control")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    scratch = OUTPUT / "scratch"
    scratch.mkdir()
    environment = dict(os.environ, TMPDIR=str(scratch))

    def relink(manifest, library, replacement=None):
        library.parent.mkdir(parents=True, exist_ok=True)
        command = manifest["link_command"].copy()
        command[command.index("-o") + 1] = str(library)
        if replacement is not None:
            if command.count(ORIGINAL_CONTEXT) != 1:
                raise RuntimeError("Original context object is not unique")
            command[command.index(ORIGINAL_CONTEXT)] = str(replacement)
        subprocess.run(command, cwd=manifest["working_directory"], env=environment, check=True)
        return command

    original_control = OUTPUT / "original-control" / "libonnxruntime_providers_webgpu.so"
    relink(control, original_control)
    if sha(original_control) != control["provider_sha256"]:
        raise RuntimeError("Qualified precise original-compute control does not reproduce")
    original_candidate = OUTPUT / "unmodified-candidate" / "libonnxruntime_providers_webgpu.so"
    relink(candidate, original_candidate)
    if sha(original_candidate) != candidate["provider_sha256"]:
        raise RuntimeError("Unmodified candidate does not reproduce")
    context = CONTROL / "webgpu_context.cc.o"
    library = OUTPUT / "libonnxruntime_providers_webgpu.so"
    link = relink(candidate, library, context)
    binary = Path("/workspace/qwen38-int4-dawn-compute-scope-reuse/native-v1/native_compare")
    for path in (Path(__file__), qualification_path, CANDIDATE / "build-manifest.json",
                 CONTROL / "build-manifest.json", context, binary, original_control, original_candidate):
        source_hashes[str(path)] = sha(path)
    report = {
        **candidate, "status": "built_not_qualified", "provider_sha256": sha(library),
        "baseline_provider_sha256": control["provider_sha256"],
        "original_compute_carrier_sha256": original_sha,
        "unmodified_candidate_provider_sha256": candidate["provider_sha256"],
        "control_definition": control["control_definition"],
        "all_candidate_compute_code_unchanged": True, "nanosecond_timing": True,
        "new_precision_measurement_not_unchanged_retry": True,
        "original_precise_control_byte_identical": True, "unmodified_candidate_byte_identical": True,
        "source_sha256": source_hashes, "link_command": link,
        "performance_qualified": False, "quality_qualified": False,
    }
    with (OUTPUT / "build-manifest.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in
                      ("status", "provider_sha256", "baseline_provider_sha256")}), flush=True)


if __name__ == "__main__":
    main()
