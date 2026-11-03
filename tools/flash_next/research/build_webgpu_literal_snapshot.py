import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from attribute_cpu_allocations_v2 import sha


ROOT = Path(__file__).parent
PARENT = Path("/workspace/qwen38-int4-fc2-private-a/build-v1")
OUTPUT = Path("/workspace/qwen38-int4-webgpu-literal-snapshot/build-v1")
RECOVERY = Path("/workspace/qwen38-int4-adapter-inline-tensors/build-v3/recovered-commands.json")
DENSE = Path("/workspace/qwen38-int4-dense-columns8-double-buffer/build-v1/build-manifest.json")
CONTEXT = "/workspace/qwen38-int4-static-node-replay/build-v8/webgpu_context.cc.o"
FLAG = "ORT_WEBGPU_LITERAL_SNAPSHOT"
CALL = re.compile(
    r"\b(ParseEnvironmentVariableWithDefault|ParseEnvironmentVariable)"
    r'\s*<\s*int\s*>\s*\(\s*"(ORT_WEBGPU_[A-Z0-9_]+)"\s*([,)])')


def transform(source):
    output = []
    records = []
    index = 0
    while index < len(source):
        begin = index
        if source.startswith("//", index):
            end = source.find("\n", index + 2)
            index = len(source) if end < 0 else end
        elif source.startswith("/*", index):
            end = source.find("*/", index + 2)
            if end < 0:
                raise RuntimeError("Unterminated C++ comment")
            index = end + 2
        elif match := re.match(r'R"([^ ()\\\t\r\n]{0,16})\(', source[index:index + 20]):
            end = source.find(")" + match[1] + '"', index + match.end())
            if end < 0:
                raise RuntimeError("Unterminated C++ raw string")
            index = end + len(match[1]) + 2
        elif (source[index] == "'" and index > 0 and index + 1 < len(source)
              and source[index - 1].isalnum() and source[index + 1].isalnum()
              and re.search(r"\b[0-9][A-Za-z0-9_']*$", source[:index])):
            index += 1
        elif source[index] in ('"', "'"):
            quote = source[index]
            index += 1
            while index < len(source) and source[index] != quote:
                index += 2 if source[index] == "\\" else 1
            if index >= len(source):
                raise RuntimeError("Unterminated C++ quoted string")
            index += 1
        elif match := CALL.match(source, index):
            function, name, delimiter = match.groups()
            if function.endswith("WithDefault") and delimiter == ",":
                replacement = f'WebGpuLiteralSnapshotDefault<int, "{name}">('
            elif function == "ParseEnvironmentVariable" and delimiter == ")":
                replacement = f'WebGpuLiteralSnapshotOptional<int, "{name}">()'
            else:
                raise RuntimeError("Unexpected environment-call argument shape")
            output.append(replacement)
            records.append({"flag": name, "original_function": function, "offset": index})
            index = match.end()
            continue
        else:
            index += 1
        output.append(source[begin:index])
    return "".join(output), records


def main():
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    dense = json.loads(DENSE.read_text())
    recovered = json.loads(RECOVERY.read_text())
    commands = [
        list(parent["compile_command"]),
        list(dense["compile_command"]),
        list(recovered[CONTEXT]),
    ]
    link = list(parent["link_command"])
    if sha(PARENT / "libonnxruntime_providers_webgpu.so") != parent["provider_sha256"]:
        raise RuntimeError("Changed supported carrier")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    scratch = OUTPUT / "scratch"
    scratch.mkdir()
    environment = dict(os.environ, TMPDIR=str(scratch))
    helper = OUTPUT / "webgpu_literal_snapshot.h"
    shutil.copyfile(ROOT / "webgpu_literal_snapshot.h", helper)
    evidence = [Path(__file__), ROOT / "webgpu_literal_snapshot.h", helper,
                PARENT / "build-manifest.json", DENSE, RECOVERY]
    units = []
    for index, original_command in enumerate(commands):
        original = Path(original_command[original_command.index("-c") + 1])
        original_object = Path(original_command[original_command.index("-o") + 1])
        if link.count(str(original_object)) != 1:
            raise RuntimeError(f"Object is not uniquely in the supported link: {original_object}")
        directory = OUTPUT / f"unit{index}"
        directory.mkdir()
        baseline = directory / "baseline.o"
        command = original_command.copy()
        command[command.index("-o") + 1] = str(baseline)
        subprocess.run(command, cwd=parent["working_directory"], env=environment, check=True)
        if sha(baseline) != sha(original_object):
            raise RuntimeError(f"Original object no longer reproduces: {original_object}")
        transformed, rewrites = transform(original.read_text())
        if not rewrites:
            raise RuntimeError(f"No scoped literal selector calls found: {original}")
        source = directory / original.name
        source.write_text(f'#line 1 "{original}"\n' + transformed)
        units.append({
            "original": str(original), "original_object": str(original_object),
            "original_object_sha256": sha(original_object),
            "baseline_object": str(baseline), "source": str(source),
            "rewrites": rewrites, "command": command,
        })
        evidence.extend((original, original_object, baseline, source))
    library = OUTPUT / "libonnxruntime_providers_webgpu.so"
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    if sha(library) != parent["provider_sha256"]:
        raise RuntimeError("Original current-provider relink proof failed")
    library.unlink()
    for unit in units:
        command = unit["command"].copy()
        source = Path(unit["source"])
        obj = source.with_suffix(".cc.o")
        command.insert(1, "-I" + str(Path(unit["original"]).parent))
        position = command.index("-c")
        command[position:position] = ["-include", str(helper)]
        command[command.index("-c") + 1] = str(source)
        command[command.index("-o") + 1] = str(obj)
        subprocess.run(command, cwd=parent["working_directory"], env=environment, check=True)
        link[link.index(unit["original_object"])] = str(obj)
        unit["candidate_command"] = command
        unit["candidate_object"] = str(obj)
        evidence.append(obj)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    report = {
        "status": "built_not_qualified", "flag": FLAG, "default_enabled": False,
        "provider_sha256": sha(library),
        "baseline_provider_sha256": parent["provider_sha256"],
        "all_three_original_objects_byte_identical": True,
        "baseline_provider_byte_identical": True,
        "units": units, "link_command": link,
        "working_directory": parent["working_directory"],
        "scope": "literal_ORT_WEBGPU_int_calls_in_qmoe_dense_and_context_cc_only",
        "shared_lazy_process_values_fresh_process_required": True,
        "headers_dynamic_names_noninteger_and_other_environment_reads_unchanged": True,
        "no_shader_changes": True, "no_object_layout_changes": True,
        "no_persistent_maps_or_kernel_fields": True,
        "performance_qualified": False, "quality_qualified": False,
        "source_sha256": {str(path): sha(path) for path in evidence},
    }
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "provider_sha256": report["provider_sha256"],
        "rewrites_per_unit": [len(unit["rewrites"]) for unit in units],
        "three_original_objects_and_provider_reproduced": True,
    }))


if __name__ == "__main__":
    main()
