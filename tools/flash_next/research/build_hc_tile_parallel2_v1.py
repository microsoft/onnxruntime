import json
import os
from pathlib import Path
import subprocess

from hc_tile_parallel2_transform import transform
from build_hc_post_norm_vec4 import private_program_transform
from build_qmoe_node_replay import replace_once, sha


FLAG = "ORT_WEBGPU_HC_TILE_PARALLEL2"
CLASS = "HCPostNormProjectionTileParallel2Program"
NAME = "HCPostNormProjectionTileParallel2F16"
OUTPUT = Path("/workspace/qwen38-int4-hc-tile-parallel2/build-v1")
SOURCE = Path("/workspace/qwen38-int4-hc-post-norm-projection/build-v2/hc_post_norm_projection.cc")
PARENT = Path("/workspace/qwen38-int4-fc2-private-a/build-v1")
PROOF = Path("/workspace/qwen38-int4-adapter-inline-tensors/build-v3")


def main():
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    proof = json.loads((PROOF / "baseline.qualification.json").read_text())
    commands = json.loads((PROOF / "recovered-commands.json").read_text())
    if parent["provider_sha256"] != "3a283ecfb892613b19a6f49d15b726e928b41027c8b3d838bd8cd1dfed9939d3":
        raise RuntimeError("Not the supported original carrier")
    if sha(PARENT / "libonnxruntime_providers_webgpu.so") != parent["provider_sha256"]:
        raise RuntimeError("Original carrier changed")
    if not proof["all_152_original_objects_byte_identical"]:
        raise RuntimeError("Missing recovered original compile commands")
    for name, digest in proof["sdk_sha256"].items():
        if sha(PROOF / name) != digest:
            raise RuntimeError(f"Changed frozen SDK input: {name}")
    original, = [path for path, command in commands.items()
                 if command[command.index("-c") + 1] == str(SOURCE)]
    command = proof["compile_commands"][list(commands).index(original)].copy()
    link = list(parent["link_command"])
    if link.count(original) != 1:
        raise RuntimeError("Current carrier does not use the exact original HC object")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    scratch = OUTPUT / "scratch"
    scratch.mkdir()
    environment = dict(os.environ, TMPDIR=str(scratch))
    obj = OUTPUT / "hc_post_norm_projection.cc.o"
    command[command.index("-o") + 1] = str(obj)
    for option in ("-MF", "-MT"):
        if option in command:
            command[command.index(option) + 1] = str(obj) + ".d" if option == "-MF" else str(obj)
    subprocess.run(command, cwd=parent["working_directory"], env=environment, check=True)
    original_object = Path(original)
    if not original_object.is_absolute():
        original_object = Path(parent["working_directory"]) / original_object
    if sha(obj) != sha(original_object):
        raise RuntimeError("Original HC translation-unit rebuild differs")
    original_hash = sha(obj)
    library = OUTPUT / "libonnxruntime_providers_webgpu.so"
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    if sha(library) != parent["provider_sha256"]:
        raise RuntimeError("Supported original provider relink differs")
    candidate = OUTPUT / SOURCE.name
    candidate.write_text(transform(SOURCE.read_text()))
    command[command.index("-c") + 1] = str(candidate)
    command.insert(1, f"-fmacro-prefix-map={candidate}={SOURCE}")
    subprocess.run(command, cwd=parent["working_directory"], env=environment, check=True)
    link[link.index(original)] = str(obj)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    inputs = [Path(__file__), Path(__file__).with_name("hc_tile_parallel_transform.py"),
              Path(__file__).with_name("prepare_hc_tile_parallel.py"),
              Path(__file__).with_name("prepare_hc_gate_staging.py"), Path(__file__).with_name("hc_tile_parallel2_transform.py"),
              Path(__file__).with_name("prepare_hc_tile_parallel2.py"),
              Path(__file__).with_name("build_hc_owned_pair34.py"), Path(__file__).with_name("build_hc_owned_vector.py"),
              Path(__file__).with_name("build_hc_post_norm_vec4.py"), SOURCE, candidate,
              PROOF / "baseline.qualification.json", PROOF / "recovered-commands.json",
              *[PROOF / name for name in proof["sdk_sha256"]]]
    report = {
        "status": "built_not_qualified",
        "provider_sha256": sha(library),
        "baseline_provider_sha256": parent["provider_sha256"],
        "baseline_byte_identical": True,
        "original_hc_object_byte_identical": True,
        "original_hc_object_sha256": original_hash,
        "flag": FLAG,
        "default_enabled": False,
        "process_static_selector": True,
        "physical_threads": 512,
        "output_columns": 8,
        "groups": 41,
        "original_rms_tree_and_fp16_projection_order_preserved": True,
        "independent_tile_workers": 2,
        "tile_scratch_full_word_f32_slots": 2560,
        "finite_f16_f32_f16_storage_roundtrip": True,
        "padded_block_half_elements": 36,
        "padded_block_u32_words": 18,
        "shared_storage_bytes": 37888,
        "no_object_layout_changes": True,
        "compile_command": command,
        "link_command": link,
        "working_directory": parent["working_directory"],
        "source_sha256": {str(path): sha(path) for path in inputs},
        "performance_qualified": False,
        "quality_qualified": False,
    }
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("status", "provider_sha256", "original_hc_object_byte_identical")}), flush=True)


if __name__ == "__main__":
    main()
