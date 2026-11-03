import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import subprocess

from build_dense_double_buffer import transform_compare
from build_qmoe_node_replay import replace_once, sha


REPOS = Path("/workspace")
PARENT = REPOS / "qwen38-int4-qmoe-prefill-rows4/build-v2"
NATIVE = REPOS / "onnxruntime-webgpu-fp8-checkpoint/build-webgpu-fp8/Release"
REPO = REPOS / "onnxruntime-webgpu-h200-perf"
SDK = REPOS / "onnxruntime-webgpu-fp8-checkpoint/include/onnxruntime"
HEADER = SDK / "ep/adapter/op_kernel.h"
OUTPUT = REPOS / "qwen38-int4-adapter-tensor-reuse/build-v1"
PROBE = REPOS / "qwen38-int4-adapter-tensor-reuse/probe-v3"
FLAG = "ORT_WEBGPU_ADAPTER_TENSOR_VECTOR_REUSE"
HELPERS = f'''inline bool AdapterTensorVectorReuseEnabled() {{
  static const int mode = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_ENFORCE(mode == 0 || mode == 1, "{FLAG} must be 0 or 1");
  return mode == 1;
}}

struct AdapterTensorVectorCache {{
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  size_t requests = 0;
  size_t nonempty_capacity_reuses = 0;
  size_t returns = 0;
  ~AdapterTensorVectorCache() {{
    const char* path = std::getenv("ORT_WEBGPU_ADAPTER_TENSOR_VECTOR_STATS_FILE");
    if (path == nullptr || requests == 0) return;
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    std::ofstream output(path, std::ios::app);
    output << "{{\\"operator\\":\\"AdapterTensorVectors\\",\\"requests\\":" << requests
           << ",\\"nonempty_capacity_reuses\\":" << nonempty_capacity_reuses
           << ",\\"returns\\":" << returns
           << ",\\"retained_input_elements\\":" << inputs.size()
           << ",\\"retained_output_elements\\":" << outputs.size() << "}}\\n";
    if (!output) {{
      LOGS_DEFAULT(ERROR) << "Cannot write adapter tensor-vector statistics to " << path;
    }}
  }}
}};

inline AdapterTensorVectorCache& GetAdapterTensorVectorCache() {{
  static thread_local AdapterTensorVectorCache cache;
  return cache;
}}

'''
TAKE = '''    if (AdapterTensorVectorReuseEnabled()) {
      auto& cache = GetAdapterTensorVectorCache();
      input_tensors_.swap(cache.inputs);
      output_tensors_.swap(cache.outputs);
      ++cache.requests;
      cache.nonempty_capacity_reuses += input_tensors_.capacity() > 0 &&
                                        output_tensors_.capacity() > 0;
    }
'''
RETURN = '''  ~OpKernelContext() {
    if (AdapterTensorVectorReuseEnabled()) {
      input_tensors_.clear();
      output_tensors_.clear();
      auto& cache = GetAdapterTensorVectorCache();
      input_tensors_.swap(cache.inputs);
      output_tensors_.swap(cache.outputs);
      ++cache.returns;
    }
  }

'''


def transform(source):
    source = replace_once(source, "#pragma once",
                          '#pragma once\n\n#include <fstream>\n#include <mutex>\n'
                          '#include "core/platform/env_var_utils.h"')
    source = replace_once(source, "struct OpKernelContext {", HELPERS + "struct OpKernelContext {")
    source = replace_once(source, "    input_tensors_.resize(context_.GetInputCount());",
                          TAKE + "    input_tensors_.resize(context_.GetInputCount());")
    return replace_once(source,
                        "    output_tensors_.resize(context_.GetOutputCount());\n  }\n",
                        "    output_tensors_.resize(context_.GetOutputCount());\n  }\n\n" + RETURN)


def make_commands():
    directory = NATIVE / "CMakeFiles/onnxruntime_providers_webgpu.dir"
    values = {}
    for line in (directory / "flags.make").read_text().splitlines():
        if line.startswith(("CXX_DEFINES =", "CXX_INCLUDES =", "CXX_FLAGS =")):
            key, value = line.split(" = ", 1)
            values[key] = value
    result = {}
    for line in (directory / "build.make").read_text().splitlines():
        if not line.startswith("\t/usr/bin/c++ ") or " -c " not in line:
            continue
        for key, value in values.items():
            line = line.replace("$(" + key + ")", value)
        if "$" in line:
            raise ValueError("Unexpanded generated compiler command")
        command = shlex.split(line)
        result[command[command.index("-o") + 1]] = command
    return result


def collect_commands(parent):
    objects = [item for item in parent["link_command"] if item.endswith(".o")]
    commands = make_commands()
    for directory in sorted({str(Path(p).parent) for p in objects if p.startswith("/workspace/")}):
        path = Path(directory) / "build-manifest.json"
        if not path.exists():
            continue
        manifest = json.loads(path.read_text())
        rows = manifest.get("compile_commands", [])
        if "compile_command" in manifest:
            rows = [*rows, manifest["compile_command"]]
        for command in rows:
            if isinstance(command, list) and "-o" in command:
                commands[command[command.index("-o") + 1]] = command.copy()
    ancestor = json.loads((REPO / "build-int4-bindgroup-cache/native-v4/build-manifest.json").read_text())
    for obj in objects:
        if obj in commands:
            continue
        if "/build-int4-uniform-arena/" not in obj:
            raise ValueError(f"Missing exact compile recipe: {obj}")
        relative = Path(obj).name.removesuffix(".o").replace("__", "/")
        source = REPO / "onnxruntime" / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        command = ancestor["compile_command"].copy()
        command[command.index("-c") + 1] = str(source)
        command[command.index("-o") + 1] = obj
        commands[obj] = command
    assert len(objects) == 152
    return {obj: commands[obj] for obj in objects}


def relocated(command, output, overlay=None, diagnostic_path_map=False):
    command = command.copy()
    if "-include" in command:
        index = command.index("-include")
        del command[index:index + 2]
    command[1:1] = ["-I" + str(SDK), "-I" + str(SDK / "core/session"),
                    "-include", str(SDK / "ep/adapters.h")]
    if diagnostic_path_map:
        command.insert(1,
            "-fmacro-prefix-map=/workspace/onnxruntime-webgpu-fp8-checkpoint/include/onnxruntime/"
            "=/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/")
    if "-MF" in command:
        command[command.index("-MF") + 1] = str(output) + ".d"
    if "-MT" in command:
        command[command.index("-MT") + 1] = str(output)
    command[command.index("-o") + 1] = str(output)
    if overlay is not None:
        command.insert(1, "-I" + str(overlay))
    return command


def original_object(path, parent):
    return Path(path) if Path(path).is_absolute() else Path(parent["working_directory"]) / path


def compile_original(obj, command, output, parent):
    subprocess.run(command, cwd=parent["working_directory"], check=True)
    actual = sha(output)
    expected = sha(original_object(obj, parent))
    if actual != expected:
        failure = dict(status="unchanged_source_object_not_byte_identical",
                       original_object=obj, source=command[command.index("-c") + 1],
                       actual_sha256=actual, expected_sha256=expected, compile_command=command)
        output.with_suffix(".failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise RuntimeError(f"Unchanged source rebuild mismatch: {obj}; evidence: {output.with_suffix('.failure.json')}")
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--diagnostic-path-map", action="store_true")
    args = parser.parse_args()
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    commands = collect_commands(parent)
    output = PROBE if args.probe_only else OUTPUT
    if args.diagnostic_path_map:
        output = output.with_name("probe-v5" if args.probe_only else "build-v2")
    output.mkdir(parents=True, exist_ok=False)
    (output / "recovered-commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    print(json.dumps(dict(status="compile_recipes_recovered", objects=len(commands))), flush=True)
    if args.probe_only:
        obj = next(p for p in commands if p.endswith("controlflow__if.cc.o"))
        target = output / "controlflow_if.cc.o"
        command = relocated(commands[obj], target, diagnostic_path_map=args.diagnostic_path_map)
        compile_original(obj, command, target, parent)
        proof = dict(status="unchanged_adapter_source_probe_byte_identical",
                     objects_recovered=len(commands), original_object=obj, object_sha256=sha(target),
                     compile_command=command, source_sha256=sha(Path(command[command.index("-c") + 1])))
        (output / "qualification.json").write_text(json.dumps(proof, indent=2) + "\n")
        print(json.dumps(proof), flush=True)
        return
    library = output / "libonnxruntime_providers_webgpu.so"
    link = parent["link_command"].copy()
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=parent["working_directory"], check=True)
    assert sha(library) == parent["provider_sha256"], "Parent relink mismatch"
    library.unlink()
    baseline = output / "baseline"
    baseline.mkdir()
    entries = list(commands.items())
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for index, (obj, command) in enumerate(entries):
            target = baseline / f"object-{index}.o"
            futures.append(executor.submit(compile_original, obj,
                relocated(command, target, diagnostic_path_map=args.diagnostic_path_map), target, parent))
        completed = False
        try:
            for index, future in enumerate(futures):
                future.result()
                print(f"Original object {index + 1}/152 byte-identical", flush=True)
            completed = True
        finally:
            if not completed:
                for future in futures:
                    future.cancel()
    overlay = output / "overlay"
    header = overlay / "ep/adapter/op_kernel.h"
    header.parent.mkdir(parents=True)
    header.write_text(transform(HEADER.read_text()))
    candidate_commands = []
    replacements = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for index, (obj, command) in enumerate(entries):
            target = output / f"object-{index}.o"
            changed = relocated(command, target, overlay, args.diagnostic_path_map)
            candidate_commands.append(changed)
            replacements[obj] = str(target)
            futures.append(executor.submit(subprocess.run, changed,
                                           cwd=parent["working_directory"], check=True))
        completed = False
        try:
            for future in futures:
                future.result()
            completed = True
        finally:
            if not completed:
                for future in futures:
                    future.cancel()
    link = [replacements.get(item, item) for item in link]
    subprocess.run(link, cwd=parent["working_directory"], check=True)
    assert FLAG.encode() in library.read_bytes(), "New adapter header was bypassed"
    comparator_source = REPOS / "qwen38-int4-dense-block32/build-v1/native_compare.cpp"
    comparator = output / "native_compare.cpp"
    comparator.write_text(transform_compare(comparator_source.read_text(),
                                           flag="ORT_WEBGPU_DENSE_BLOCK32"))
    comparator_command = ["c++", "-O3", "-std=c++20", str(comparator),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib", "-lonnxruntime",
        "-o", str(output / "native_compare")]
    subprocess.run(comparator_command, check=True)
    sources = [HEADER, header, comparator, comparator_source,
               *[Path(command[command.index("-c") + 1]) for command in candidate_commands]]
    manifest = dict(status="built_not_qualified", provider_sha256=sha(library),
        baseline_provider_sha256=parent["provider_sha256"], baseline_byte_identical=True,
        all_152_original_source_objects_byte_identical=True,
        comparator_sha256=sha(output / "native_compare"), compile_commands=candidate_commands,
        link_command=link, working_directory=parent["working_directory"],
        source_sha256={os.path.relpath(path, output): sha(path) for path in sources},
        process_static_selector=True, default_enabled=False, shader_and_arithmetic_unchanged=True,
        diagnostic_macro_path_map=args.diagnostic_path_map,
        no_object_layout_changes=True, quality_qualified=False, headline_eligible=False)
    (output / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(status=manifest["status"], provider_sha256=manifest["provider_sha256"])))


if __name__ == "__main__":
    main()
