import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    hc = json.loads(Path("/workspace/qwen38-int4-hc-gdn-combined/build-v2/build-manifest.json").read_text())
    replay = json.loads(Path("/workspace/qwen38-int4-qmoe-node-replay/build-v5/build-manifest.json").read_text())
    output = Path("/workspace/qwen38-int4-hc-node-replay-combined/build-v1")
    output.mkdir(parents=True, exist_ok=False)
    scratch = output / "scratch"
    scratch.mkdir()
    environment = dict(os.environ, TMPDIR=str(scratch))
    for name, manifest in (("hc", hc), ("replay", replay)):
        command = manifest["link_command"].copy()
        directory = output / ("verify-" + name)
        directory.mkdir()
        temporary = directory / "libonnxruntime_providers_webgpu.so"
        command[command.index("-o") + 1] = str(temporary)
        subprocess.run(command, cwd=manifest["working_directory"], env=environment, check=True)
        assert sha(temporary) == manifest["provider_sha256"], name + " relink mismatch"
        temporary.unlink()
        directory.rmdir()
    link = hc["link_command"].copy()
    library = output / "libonnxruntime_providers_webgpu.so"
    link[link.index("-o") + 1] = str(library)
    replaced = {}
    for source in ("contrib_ops/webgpu/moe/qmoe.cc", "core/providers/webgpu/webgpu_context.cc",
                   "core/providers/webgpu/wgsl_templates/wgsl_gen.cc"):
        suffix = source.replace("/", "__") + ".o"
        old = [value for value in link if value.endswith(suffix)]
        assert len(old) == 1, old
        command = next(command for command in replay["compile_commands"]
                       if command[command.index("-c") + 1].endswith(source))
        new = command[command.index("-o") + 1]
        assert replay["link_command"].count(new) == 1
        link[link.index(old[0])] = new
        replaced[source] = sha(Path(new))
    subprocess.run(link, cwd=hc["working_directory"], env=environment, check=True)
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
                  component_provider_sha256=dict(hc=hc["provider_sha256"], replay=replay["provider_sha256"]),
                  component_relinks_byte_identical=True, link_command=link,
                  working_directory=hc["working_directory"], replaced_object_sha256=replaced,
                  gdn_object_sha256=hc["gdn_object_sha256"],
                  quality_qualified=False, headline_eligible=False)
    (output / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("status", "provider_sha256", "component_relinks_byte_identical")}))


if __name__ == "__main__":
    main()
