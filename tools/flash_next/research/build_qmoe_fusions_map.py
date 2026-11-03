import json
import os
from pathlib import Path
import subprocess

from build_qmoe_node_replay import sha


ROOT = Path(__file__).parent
REPOS = Path("/workspace")
PARENT = REPOS / "qwen38-int4-qmoe-fc2-final-mix/build-v2"
MAP = REPOS / "qwen38-int4-qsa-parallel-map/build-v1"
OUTPUT = REPOS / "qwen38-int4-qmoe-fusions-map/build-v2"


def main():
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    mapping = json.loads((MAP / "build-manifest.json").read_text())
    OUTPUT.mkdir(parents=True, exist_ok=False)
    scratch = OUTPUT / "scratch"
    scratch.mkdir()
    environment = dict(os.environ, TMPDIR=str(scratch))
    library = OUTPUT / "libonnxruntime_providers_webgpu.so"
    map_directory = scratch / "map-component"
    map_directory.mkdir()
    map_baseline = map_directory / library.name
    map_link = mapping["link_command"].copy()
    map_link[map_link.index("-o") + 1] = str(map_baseline)
    subprocess.run(map_link, cwd=mapping["working_directory"], env=environment, check=True)
    assert sha(map_baseline) == mapping["provider_sha256"], "Map-component relink mismatch"
    map_baseline.unlink()
    link = parent["link_command"].copy()
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    assert sha(library) == parent["provider_sha256"], "FC2-parent relink mismatch"
    library.unlink()
    original = [p for p in link if p.endswith("contrib_ops__webgpu__bert__sparse_attention_indexer.cc.o")]
    replacement = str(MAP / "sparse_attention_indexer.cc.o")
    assert len(original) == 1 and map_link.count(replacement) == 1
    for name, digest in mapping["source_sha256"].items():
        assert sha(MAP / name) == digest, name
    link[link.index(original[0])] = replacement
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
        baseline_byte_identical=True, map_component_byte_identical=True,
        baseline_provider_sha256=parent["provider_sha256"],
        map_component_provider_sha256=mapping["provider_sha256"],
        map_object_sha256=sha(Path(replacement)),
        replaced_object_sha256=sha(Path(original[0])),
        link_command=link, working_directory=parent["working_directory"],
        no_object_layout_changes=True, quality_qualified=False, headline_eligible=False)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "provider_sha256", "baseline_byte_identical")}))


if __name__ == "__main__":
    main()
