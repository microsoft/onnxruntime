# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Relink the benchmark plugin with opt-in GPU timestamps, preserving kernel objects."""

import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

from build import HERE, ROOT, compile_copy, replace_once, sha256


def instrument_provider(text):
    text = replace_once(
        text,
        "namespace onnxruntime {",
        f'#include "{HERE / "gpu_profile.h"}"\n\nnamespace onnxruntime {{',
    )
    text = replace_once(
        text,
        "  if ((session_profiler_ && session_profiler_->Enabled()) || run_options.enable_profiling) {",
        """  if (const char* path = DiagnosticGpuProfilePath(); path != nullptr && *path) {
    context_.SetProfilingStartTime(std::chrono::high_resolution_clock::now());
    context_.StartProfiling();
  } else if ((session_profiler_ && session_profiler_->Enabled()) || run_options.enable_profiling) {""",
    )
    begin = text.index("Status WebGpuExecutionProvider::OnRunEnd(")
    before = "  if (session_profiler_ && session_profiler_->Enabled()) {"
    if before not in text[begin:]:
        raise ValueError("Missing OnRunEnd profiling collection point.")
    return text[:begin] + text[begin:].replace(
        before,
        """  if (const char* path = DiagnosticGpuProfilePath(); path != nullptr && *path) {
    profiling::Events events;
    context_.CollectProfilingData(events);
    WriteDiagnosticGpuProfile(this, events, context_.AdapterInfo());
  } else if (session_profiler_ && session_profiler_->Enabled()) {""",
        1,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build, output = args.build_dir.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    original = ROOT / "onnxruntime/core/providers/webgpu/webgpu_execution_provider.cc"
    copied = output / original.name
    copied.write_text(instrument_provider(original.read_text()))
    commands = subprocess.check_output(
        ["ninja", "-C", str(build), "-t", "commands", "onnxruntime_providers_webgpu"], text=True
    ).splitlines()
    commands = [line.replace(f" -c {original}", f" -I{original.parent} -c {original}") for line in commands]
    obj = output / "webgpu_execution_provider.cc.o"
    compile_command = compile_copy(commands, original, copied, obj, build)
    candidates = [line for line in commands if " -o libonnxruntime_providers_webgpu.so " in line]
    if len(candidates) != 1:
        raise ValueError("Expected one WebGPU plugin link command.")
    link = shlex.split(candidates[0])
    link = link[link.index(compile_command[0]) :]
    if "&&" in link:
        link = link[: link.index("&&")]
    original_objects = [i for i, value in enumerate(link) if value.endswith("/webgpu_execution_provider.cc.o")]
    if len(original_objects) != 1:
        raise ValueError("Expected one execution-provider object in the plugin link.")
    link[original_objects[0]] = str(obj)
    library = output / "libonnxruntime_providers_webgpu.so"
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=build, check=True)
    manifest = {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)).hexdigest(),
        "build_dir": str(build),
        "library": str(library),
        "library_sha256": sha256(library),
        "original_provider_sha256": sha256(original),
        "copied_provider_sha256": sha256(copied),
        "profile_header_sha256": sha256(HERE / "gpu_profile.h"),
        "kernel_sha256": sha256(ROOT / "onnxruntime/contrib_ops/webgpu/quantization/matmul_nbits_small_m_simt.h"),
        "compile_command": compile_command,
        "link_command": link,
        "profiling": "Inactive unless ORT_WEBGPU_DIAGNOSTIC_GPU_PROFILE is set. Off/on use this identical library.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(library, flush=True)


if __name__ == "__main__":
    main()
