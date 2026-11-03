import json
from pathlib import Path

from attribute_cpu_allocations_v2 import sha
from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent


def main():
    parent = ROOT / "run_literal_snapshot_v1.py"
    output = ROOT / "run_literal_snapshot_v2.py"
    generation = ROOT / "literal-snapshot-launch-v2-generation.json"
    if output.exists() or generation.exists():
        raise FileExistsError("Fresh launch repair required")
    text = parent.read_text().replace(
        'VARIANT = "webgpu-literal-snapshot-v1"',
        'VARIANT = "webgpu-literal-snapshot-v2"')
    anchor = '            resolved = host_path(path) if path.startswith(("/workspace/", "/evidence/")) else Path(path)\n'
    text = replace_once(
        text, anchor,
        anchor
        + "            if resolved.is_symlink() and (target := resolved.readlink()).is_absolute():\n"
        + '                if not str(target).startswith(("/workspace/", "/evidence/")):\n'
        + '                    raise RuntimeError(f"Unmapped absolute SDK input symlink: {target}")\n'
        + "                resolved = host_path(str(target))\n")
    text = replace_once(
        text, "        Path(__file__), Path(native_template.__file__),",
        '        Path(__file__), ROOT / "literal-snapshot-launch-v2-generation.json",\n'
        "        Path(native_template.__file__),")
    compile(text, str(output), "exec")
    with output.open("x") as stream:
        stream.write(text)
    with generation.open("x") as stream:
        json.dump({
            "reason": "SDK libonnxruntime.so is a container-absolute symlink; map its explicit target before checking the unchanged expected library hash.",
            "no_gpu_evidence_from_failed_preflight": True,
            "source_sha256": {str(path): sha(path) for path in
                              (Path(__file__), parent, ROOT / "webgpu-literal-snapshot-v1-run.log")},
            "generated_sha256": sha(output),
        }, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
