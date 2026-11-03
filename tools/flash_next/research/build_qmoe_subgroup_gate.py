import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    repo = Path("/workspace/onnxruntime-webgpu-h200-perf")
    prior = json.loads((repo / "build-int4-matmul-gemv-n96/build-manifest.json").read_text())
    output = repo / "build-int4-qmoe-subgroup-gate"
    output.mkdir(exist_ok=True)
    baseline = output / "baseline"
    baseline.mkdir(exist_ok=True)
    link = prior["link_command"].copy()
    baseline_library = baseline / "libonnxruntime_providers_webgpu.so"
    link[link.index("-o") + 1] = str(baseline_library)
    subprocess.run(link, cwd=prior["working_directory"], check=True)
    expected = prior["provider_sha256"]
    if digest(baseline_library) != expected:
        raise RuntimeError("Unchanged-object baseline relink did not reproduce the qualified provider")
    sources = output / "template-input/moe"
    sources.mkdir(parents=True, exist_ok=True)
    template = repo / "onnxruntime/contrib_ops/webgpu/moe/gate_1token.wgsl.template"
    shutil.copy2(template, sources / template.name)
    partial = output / "partial/wgsl_template_gen"
    subprocess.run(["python3", str(repo / "tools/python/wgsl_gen.py"),
                    "-i", str(sources.parent), "-o", str(partial),
                    "-g", "static-cpp-literal", "-I", "wgsl_template_gen/"], check=True)
    original = repo / "build-int4-uniform-arena/wgsl_template_gen"
    overlay = output / "overlay/wgsl_template_gen"
    shutil.copytree(original, overlay, dirs_exist_ok=True)
    pattern = (r'template <>\nstruct TemplateParameter<"moe/gate_1token\.wgsl\.template"> \{'
               r'.*?::type params\);')
    replacement, count = re.subn(pattern, lambda match: match.group(),
                                 (partial / "index.h").read_text(), flags=re.S)
    matches = re.findall(pattern, replacement, flags=re.S)
    if count != 1 or len(matches) != 1:
        raise RuntimeError("Expected one generated gate declaration")
    index, count = re.subn(pattern, lambda _: matches[0], (overlay / "index.h").read_text(), flags=re.S)
    if count != 1:
        raise RuntimeError("Expected one original gate declaration")
    (overlay / "index.h").write_text(index)
    header = Path("generated/moe/gate_1token.h")
    shutil.copy2(partial / header, overlay / header)
    changed = [str(p.relative_to(overlay)) for p in overlay.rglob("*") if p.is_file() and
               digest(p) != digest(original / p.relative_to(overlay))]
    if set(changed) != {"index.h", str(header)}:
        raise RuntimeError(f"Unexpected generated-template changes: {changed}")
    commands = []
    objects = {}
    for source in ("contrib_ops/webgpu/moe/qmoe.cc", "core/providers/webgpu/wgsl_templates/wgsl_gen.cc"):
        command = prior["compile_command"].copy()
        command.insert(1, "-I" + str(overlay.parent))
        command[command.index("-c") + 1] = str(repo / "onnxruntime" / source)
        obj = output / source.replace("/", "__").replace(".cc", ".cc.o")
        command[command.index("-o") + 1] = str(obj)
        subprocess.run(command, cwd=prior["working_directory"], check=True)
        commands.append(command)
        objects[source] = str(obj)
    link = prior["link_command"].copy()
    for source, obj in objects.items():
        old = "/workspace/onnxruntime-webgpu-h200-perf/build-int4-uniform-arena/" + source.replace("/", "__") + ".o"
        if link.count(old) != 1:
            raise RuntimeError(f"Expected one original link object: {old}")
        link[link.index(old)] = obj
    library = output / "libonnxruntime_providers_webgpu.so"
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=prior["working_directory"], check=True)
    report = {
        "status": "built_not_GPU_qualified", "baseline_byte_identical": True,
        "baseline_provider_sha256": expected, "provider_sha256": digest(library),
        "changed_generated_files": changed, "compile_commands": commands,
        "link_command": link, "working_directory": prior["working_directory"],
        "source_sha256": {str(repo / "onnxruntime" / source): digest(repo / "onnxruntime" / source)
                          for source in objects},
        "template_sha256": digest(template),
    }
    (output / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "baseline_byte_identical", "provider_sha256")}))


if __name__ == "__main__":
    main()
