from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess

from build_adapter_tensor_reuse import (
    FLAG, HEADER, PARENT, REPO, SDK, collect_commands, compile_original, transform)
from build_dense_double_buffer import transform_compare
from build_qmoe_node_replay import replace_once, sha


OUTPUT = Path("/workspace/qwen38-int4-adapter-tensor-reuse/build-v7")


def recover_original_n96(source):
    start = source.index("  const bool use_f32_decode_gemv =")
    end = source.index("  // A single row has no cross-row weight reuse.", start)
    source = source[:start] + source[end:]
    source = replace_once(source, "  if (use_f32_decode_gemv || (batch_size", "  if (batch_size")
    source = replace_once(source, "output_tensor->IsDataType<MLFloat16>())) {",
                          "output_tensor->IsDataType<MLFloat16>()) {")
    start = source.index("    const uint32_t workgroup_x = use_f32_decode_gemv")
    end = source.index("    MatMulGemvProgram program;", start)
    source = source[:start] + source[end:]
    source = replace_once(source, ".SetWorkgroupSize(workgroup_x, workgroup_y)",
        ".SetWorkgroupSize(MatMulGemvProgram::kWorkgroupSizeX, MatMulGemvProgram::kWorkgroupSizeY)")
    return replace_once(source, "CeilDiv(n_vec_count, workgroup_x)",
                        "CeilDiv(n_vec_count, MatMulGemvProgram::kWorkgroupSizeX)")


def command_for(command, obj, sdk, build_output=OUTPUT):
    command = command.copy()
    source = command[command.index("-c") + 1]
    sdk_origin = SDK if source.startswith("/workspace/onnxruntime-webgpu-fp8-checkpoint/") else (
        REPO / "include/onnxruntime")
    if "-include" in command:
        index = command.index("-include")
        del command[index:index + 2]
    command[1:1] = [
        "-I" + str(sdk), "-I" + str(sdk / "core/session"),
        "-include", str(sdk / "ep/adapters.h"),
        f"-fmacro-prefix-map={build_output}/recovered/={REPO}/",
        f"-fmacro-prefix-map={sdk}/={sdk_origin}/"]
    command[command.index("-o") + 1] = str(obj)
    for flag in ("-MF", "-MT"):
        if flag in command:
            command[command.index(flag) + 1] = str(obj) + ".d" if flag == "-MF" else str(obj)
    return command


def compile_batch(entries, output, sdk, parent, baseline, build_output=OUTPUT):
    commands = []
    futures = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for index, (original, command) in enumerate(entries):
            obj = output / f"object-{index}.o"
            command = command_for(command, obj, sdk, build_output)
            commands.append(command)
            if baseline:
                future = executor.submit(compile_original, original, command, obj, parent)
            else:
                future = executor.submit(subprocess.run, command, cwd=parent["working_directory"], check=True)
            futures.append(future)
        completed = False
        try:
            for index, future in enumerate(futures):
                future.result()
                print(f'{"Original byte-identical" if baseline else "Candidate compiled"} '
                      f"object {index + 1}/{len(entries)}", flush=True)
            completed = True
        finally:
            if not completed:
                for future in futures:
                    future.cancel()
    return commands


def main(output=OUTPUT, header_transform=transform, flag=FLAG, object_layout_changes=False):
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    commands = collect_commands(parent)
    assert len(commands) == 152
    output.mkdir(parents=True, exist_ok=False)
    original_source = REPO / "onnxruntime/core/providers/webgpu/math/matmul.cc"
    recovered = output / "recovered" / original_source.relative_to(REPO)
    recovered.parent.mkdir(parents=True)
    recovered.write_text(recover_original_n96(original_source.read_text()))
    n96_manifest = json.loads((REPO / "build-int4-matmul-gemv-n96/build-manifest.json").read_text())
    assert sha(recovered) == n96_manifest["source_sha256"], "Recovered N96 source is not exact"
    for command in commands.values():
        if command[command.index("-c") + 1] == str(original_source):
            command[command.index("-c") + 1] = str(recovered)
    subgroup_relative = Path("onnxruntime/contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits.cc")
    assert sha(REPO / subgroup_relative) == "56225b7293a00bc3bd31bbd7ab23adbf0d0239be513abd56782d1e1c94a7de72"
    for command in commands.values():
        if command[command.index("-c") + 1] == str(REPO / subgroup_relative):
            command.insert(1, "-I" + str(REPO / "build-int4-uniform-arena"))
    (output / "recovered-commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    baseline_sdk = output / "sdk-baseline"
    shutil.copytree(SDK, baseline_sdk)
    profiler = REPO / "include/onnxruntime/ep/adapter/ep.h"
    original_ep = (SDK / "ep/adapter/ep.h").read_text()
    expected_ep = original_ep.replace(" public:\n  inline IExecutionProvider* EpImpl()",
        "  profiling::EpProfiler* ProfilerImpl() const noexcept {\n"
        "    return profiler_.get();\n"
        "  }\n\n public:\n  inline IExecutionProvider* EpImpl()", 1)
    assert expected_ep == profiler.read_text(), "SDK profiler difference is not the single known accessor"
    shutil.copy2(profiler, baseline_sdk / "ep/adapter/ep.h")
    library = output / "libonnxruntime_providers_webgpu.so"
    link = parent["link_command"].copy()
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=parent["working_directory"], check=True)
    assert sha(library) == parent["provider_sha256"], "Original provider relink mismatch"
    library.unlink()
    baseline = output / "baseline"
    baseline.mkdir()
    entries = list(commands.items())
    baseline_commands = compile_batch(entries, baseline, baseline_sdk, parent, True, output)
    (output / "baseline.qualification.json").write_text(json.dumps(dict(
        all_152_original_objects_byte_identical=True, compile_commands=baseline_commands,
        sdk_sha256={str(p.relative_to(output)): sha(p) for p in baseline_sdk.rglob("*") if p.is_file()},
        original_profiler_header_sha256=sha(profiler)), indent=2) + "\n")
    candidate_sdk = output / "sdk-candidate"
    shutil.copytree(baseline_sdk, candidate_sdk)
    header = candidate_sdk / "ep/adapter/op_kernel.h"
    header.write_text(header_transform(header.read_text()))
    candidate_commands = compile_batch(entries, output, candidate_sdk, parent, False, output)
    replacements = {original: str(output / f"object-{i}.o") for i, (original, _) in enumerate(entries)}
    link = [replacements.get(part, part) for part in link]
    subprocess.run(link, cwd=parent["working_directory"], check=True)
    assert flag.encode() in library.read_bytes(), "The changed adapter header was not used"
    comparator_source = Path("/workspace/qwen38-int4-dense-block32/build-v1/native_compare.cpp")
    comparator = output / "native_compare.cpp"
    comparator.write_text(transform_compare(comparator_source.read_text(), flag="ORT_WEBGPU_DENSE_BLOCK32"))
    comparator_command = ["c++", "-O3", "-std=c++20", str(comparator),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(output / "native_compare")]
    subprocess.run(comparator_command, check=True)
    sources = [HEADER, profiler, comparator, comparator_source,
               REPO / "build-int4-uniform-arena/wgsl_template_gen/index.h",
               *[Path(command[command.index("-c") + 1]) for command in candidate_commands],
               *[p for sdk in (baseline_sdk, candidate_sdk) for p in sdk.rglob("*") if p.is_file()]]
    manifest = dict(status="built_not_qualified", provider_sha256=sha(library),
        baseline_provider_sha256=parent["provider_sha256"], baseline_byte_identical=True,
        all_152_original_source_objects_byte_identical=True,
        comparator_sha256=sha(output / "native_compare"), compile_commands=candidate_commands,
        link_command=link, working_directory=parent["working_directory"],
        source_sha256={os.path.relpath(path, output): sha(path) for path in sources},
        process_static_selector=True, default_enabled=False, shader_and_arithmetic_unchanged=True,
        no_object_layout_changes=not object_layout_changes, complete_sdk_copy_for_relative_include_wiring=True,
        original_profiler_accessor_retained=True, quality_qualified=False, headline_eligible=False)
    (output / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ("status", "provider_sha256")}))


if __name__ == "__main__":
    main()
