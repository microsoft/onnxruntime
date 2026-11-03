import json
import os
from pathlib import Path
import subprocess

from build_qmoe_node_replay import sha


PREFILL = Path("/workspace/qwen38-int4-qmoe-extended-prefill/build-v1")
DENSE = Path("/workspace/qwen38-int4-dense-bitcast-unpack/build-v1")
OUTPUT = Path("/workspace/qwen38-int4-prefill-dense-stack/build-v1")


def main():
    prefill = json.loads((PREFILL / "build-manifest.json").read_text())
    dense = json.loads((DENSE / "build-manifest.json").read_text())
    assert prefill["baseline_provider_sha256"] == dense["baseline_provider_sha256"]
    assert prefill["no_existing_object_layout_changes"] and dense["no_existing_object_layout_changes"]
    OUTPUT.mkdir(parents=True, exist_ok=False)
    scratch = OUTPUT / "scratch"
    scratch.mkdir()
    environment = dict(os.environ, TMPDIR=str(scratch))
    library = OUTPUT / "libonnxruntime_providers_webgpu.so"
    link = prefill["link_command"].copy()
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=prefill["working_directory"], env=environment, check=True)
    assert sha(library) == prefill["provider_sha256"], "Prefill baseline relink mismatch"
    library.unlink()
    original = "/workspace/qwen38-int4-generic-matmul-replay/build-v1/matmul_nbits.cc.o"
    replacement = DENSE / "matmul_nbits.cc.o"
    assert link.count(original) == 1 and replacement.is_file()
    link[link.index(original)] = str(replacement)
    subprocess.run(link, cwd=prefill["working_directory"], env=environment, check=True)
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
        baseline_provider_sha256=prefill["provider_sha256"],
        retained_provider_sha256=prefill["baseline_provider_sha256"],
        baseline_byte_identical=True, working_directory=prefill["working_directory"],
        link_command=link, component_provider_sha256=dict(
            prefill=prefill["provider_sha256"], dense=dense["provider_sha256"]),
        component_object_sha256={str(replacement): sha(replacement),
                                str(PREFILL / "qmoe.cc.o"): sha(PREFILL / "qmoe.cc.o")},
        no_existing_object_layout_changes=True, quality_qualified=False, headline_eligible=False)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "provider_sha256", "baseline_byte_identical")}))


if __name__ == "__main__":
    main()
