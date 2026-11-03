import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    hc = json.loads(Path("/workspace/qwen38-int4-hc-post-norm/build-v8/build-manifest.json").read_text())
    gdn = json.loads(Path("/workspace/qwen38-int4-gdn-decode-view/build-original/build-manifest.json").read_text())
    output = Path("/workspace/qwen38-int4-hc-gdn-combined/build-v2")
    output.mkdir(parents=True, exist_ok=False)
    scratch = output / "scratch"
    scratch.mkdir()
    environment = dict(os.environ, TMPDIR=str(scratch))
    for name, manifest in (("hc", hc), ("gdn", gdn)):
        command = manifest["link_command"].copy()
        verify_directory = output / ("verify-" + name)
        verify_directory.mkdir()
        temporary = verify_directory / "libonnxruntime_providers_webgpu.so"
        command[command.index("-o") + 1] = str(temporary)
        subprocess.run(command, cwd=manifest["working_directory"], env=environment, check=True)
        assert sha(temporary) == manifest["provider_sha256"], name + " relink mismatch"
        temporary.unlink()
        verify_directory.rmdir()
    link = hc["link_command"].copy()
    library = output / "libonnxruntime_providers_webgpu.so"
    link[link.index("-o") + 1] = str(library)
    matches = [p for p in link if p.endswith("contrib_ops__webgpu__bert__gated_delta_net.cc.o")]
    assert len(matches) == 1, matches
    object_path = gdn["compile_command"][gdn["compile_command"].index("-o") + 1]
    link[link.index(matches[0])] = object_path
    subprocess.run(link, cwd=hc["working_directory"], env=environment, check=True)
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
                  component_provider_sha256=dict(hc=hc["provider_sha256"], gdn=gdn["provider_sha256"]),
                  component_relinks_byte_identical=True, link_command=link,
                  working_directory=hc["working_directory"], gdn_object_sha256=sha(Path(object_path)),
                  quality_qualified=False, headline_eligible=False)
    (output / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "provider_sha256", "component_relinks_byte_identical")}))


if __name__ == "__main__":
    main()
