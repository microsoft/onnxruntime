# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Build a diagnostic executable without editing production sources or objects."""

import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "onnxruntime/contrib_ops/webgpu/quantization/matmul_nbits.cc"
SHADER_HELPER = ROOT / "onnxruntime/core/providers/webgpu/shader_helper.cc"


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"Expected exactly one diagnostic insertion point: {before}")
    return text.replace(before, after, 1)


def instrument_source(source):
    source = replace_once(
        source,
        "namespace onnxruntime {",
        f'#include "{HERE / "diagnostic.h"}"\n\nnamespace onnxruntime {{',
    )
    marker = "  // apple|intel - Experimental dawn support for subgroup matrix matmul."
    insertion = """
  const char* diagnostic_text = std::getenv("ORT_WEBGPU_SMALL_M_DIAGNOSTIC");
  const std::string_view diagnostic_mode = diagnostic_text ? diagnostic_text : "auto";
  ORT_ENFORCE(diagnostic_mode == "auto" || diagnostic_mode == "regular" || diagnostic_mode == "simt",
              "ORT_WEBGPU_SMALL_M_DIAGNOSTIC must be auto, regular, or simt.");
  if (diagnostic_mode == "simt") {
    return RunDiagnosticSmallMSimt(a, b, scales, zero_points, bias, M, N, K, block_size,
                                   batch_count, weight_index, weight_index_indirect, dispatch_M,
                                   static_cast<uint32_t>(nbits), context, y);
  }
  const bool diagnostic_regular = diagnostic_mode == "regular";

"""
    source = replace_once(source, marker, insertion + marker)
    for name in (
        "CanApplySubgroupMatrixMatMulNBits",
        "CanApplyDP4AMatrixMatMulNBits",
        "CanApplyWideTileMatMulNBits",
    ):
        source = replace_once(source, f"if ({name}(", f"if (!diagnostic_regular && {name}(")
    return source


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_copy(commands, original, copied, obj, build):
    matches = [shlex.split(line) for line in commands if f" -c {original}" in line]
    if len(matches) != 1:
        raise ValueError(f"Expected one existing compile command for {original}, got {len(matches)}")
    command = matches[0]
    for flag, value in (("-c", copied), ("-o", obj), ("-MF", str(obj) + ".d"), ("-MT", obj)):
        command[command.index(flag) + 1] = str(value)
    subprocess.run(command, cwd=build, check=True)
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build/webgpu-pipeline-lifetime/Release")
    parser.add_argument("--output", type=Path, default=ROOT / "build/webgpu-small-m-simt")
    args = parser.parse_args()
    build, output = args.build_dir.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = output / "diagnostic_matmul_nbits.cc"
    source.write_text(instrument_source(SOURCE.read_text()))
    commands = subprocess.check_output(
        ["ninja", "-C", str(build), "-t", "commands", "onnxruntime_provider_test"], text=True
    ).splitlines()
    obj = output / "diagnostic_matmul_nbits.o"
    compile_command = compile_copy(commands, SOURCE, source, obj, build)
    helper_source = output / "diagnostic_shader_helper.cc"
    helper_text = SHADER_HELPER.read_text()
    # Production enables every available extension. Prove this program needs only f16.
    for feature, occurrences in (("Subgroups", 2), ("ChromiumExperimentalSubgroupMatrix", 1)):
        before = f"  if (webgpu_context_.DeviceHasFeature(wgpu::FeatureName::{feature})) {{"
        after = (
            '  if (program_.Name() != "DiagnosticSmallMSimt" && '
            f"webgpu_context_.DeviceHasFeature(wgpu::FeatureName::{feature})) {{"
        )
        if helper_text.count(before) != occurrences:
            raise ValueError(f"Unexpected ShaderHelper feature/builtin emission for {feature}")
        helper_text = helper_text.replace(before, after)
    helper_source.write_text(helper_text)
    helper_obj = output / "diagnostic_shader_helper.o"
    helper_command = compile_copy(commands, SHADER_HELPER, helper_source, helper_obj, build)
    executable = output / "diagnostic"
    original_link = shlex.split(commands[-1])
    libraries = original_link[original_link.index("-o") + 2 :]
    if libraries[-2:] == ["&&", ":"]:
        libraries = libraries[:-2]
    link_command = [
        compile_command[0],
        "-std=c++20",
        "-O3",
        "-DNDEBUG",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-I" + str(ROOT / "include/onnxruntime/core/session"),
        str(HERE / "runner.cc"),
        str(obj),
        str(helper_obj),
        "-o",
        str(executable),
        "-Wl,--start-group",
        *libraries,
        "-Wl,--end-group",
        "-ldl",
        "-lpthread",
    ]
    manifest = {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "build_dir": str(build),
        "production_matmul_sha256": sha256(SOURCE),
        "diagnostic_source_sha256": sha256(source),
        "diagnostic_header_sha256": sha256(HERE / "diagnostic.h"),
        "compile_command": compile_command,
        "shader_helper_command": helper_command,
        "production_shader_helper_sha256": sha256(SHADER_HELPER),
        "diagnostic_shader_helper_sha256": sha256(helper_source),
        "link_command": link_command,
        "isolation": (
            "Copied MatMulNBits and ShaderHelper objects only. The latter omits unused subgroup extensions "
            "only for DiagnosticSmallMSimt. Production sources/objects and other shaders are untouched."
        ),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    subprocess.run(link_command, cwd=build, check=True)
    manifest["executable_sha256"] = sha256(executable)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(executable, flush=True)


if __name__ == "__main__":
    main()
