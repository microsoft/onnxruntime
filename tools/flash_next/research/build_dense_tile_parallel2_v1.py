import json
import os
from pathlib import Path
import subprocess

from build_qmoe_node_replay import replace_once, sha
from dense_tile_parallel2_transform import FLAG, NAME, SHARED_BYTES, transform


ROOT = Path(__file__).parent
OUTPUT = Path("/workspace/qwen38-int4-dense-tile-parallel2/build-v1")
PARENT = Path("/workspace/qwen38-int4-fc2-private-a/build-v1")
DENSE = Path("/workspace/qwen38-int4-dense-columns8-double-buffer/build-v1/build-manifest.json")
PROOF = Path("/workspace/qwen38-int4-adapter-inline-tensors/build-v3")


def native_source(source):
    source = replace_once(source, "#include <array>", "#include <array>\n#include <sys/resource.h>")
    source = replace_once(source, "constexpr size_t rounds = 16;", "constexpr size_t rounds = 36;")
    source = replace_once(source, '{"maxNumPendingDispatches", "1"}', '{"maxNumPendingDispatches", "64"}')
    source = replace_once(source, "    size_t mismatches = 0;",
                          "    size_t mismatches = 0;\n    struct rusage measured_start{}, measured_end{};")
    source = replace_once(
        source, "      for (int repeat = 0; repeat < 4; ++repeat) {",
        "      for (int repeat = 0; repeat < 4; ++repeat) {\n"
        "        if (round * 4 + repeat == 16 && getrusage(RUSAGE_SELF, &measured_start) != 0)\n"
        '          throw std::runtime_error("getrusage start failed");')
    source = replace_once(
        source, "    if (capture) {\n      std::ofstream file",
        "    if (getrusage(RUSAGE_SELF, &measured_end) != 0)\n"
        '      throw std::runtime_error("getrusage end failed");\n'
        "    if (measured_end.ru_majflt != measured_start.ru_majflt)\n"
        '      throw std::runtime_error("Measured major faults");\n'
        '    std::cout << "SCOPE warmups=16 measured=128 major_faults=0" << std::endl;\n'
        "    if (capture) {\n      std::ofstream file")
    return source


def main():
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    dense = json.loads(DENSE.read_text())
    proof = json.loads((PROOF / "baseline.qualification.json").read_text())
    if (parent["provider_sha256"] != "3a283ecfb892613b19a6f49d15b726e928b41027c8b3d838bd8cd1dfed9939d3"
            or sha(PARENT / "libonnxruntime_providers_webgpu.so") != parent["provider_sha256"]
            or not proof["all_152_original_objects_byte_identical"]):
        raise RuntimeError("Not the supported carrier and exact SDK proof")
    for name, digest in proof["sdk_sha256"].items():
        if sha(PROOF / name) != digest:
            raise RuntimeError(f"Changed frozen SDK input: {name}")
    command = list(dense["compile_command"])
    original = Path(command[command.index("-c") + 1])
    original_object = Path(command[command.index("-o") + 1])
    link = list(parent["link_command"])
    if link.count(str(original_object)) != 1:
        raise RuntimeError("Dense original object is not unique in supported carrier")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    scratch = OUTPUT / "scratch"
    scratch.mkdir()
    environment = dict(os.environ, TMPDIR=str(scratch))
    baseline = OUTPUT / "original-matmul_nbits.cc.o"
    command[command.index("-o") + 1] = str(baseline)
    for option in ("-MF", "-MT"):
        if option in command:
            command[command.index(option) + 1] = str(baseline) + ".d" if option == "-MF" else str(baseline)
    subprocess.run(command, cwd=parent["working_directory"], env=environment, check=True)
    if sha(baseline) != sha(original_object):
        raise RuntimeError("Original dense object does not reproduce")
    library = OUTPUT / "libonnxruntime_providers_webgpu.so"
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    if sha(library) != parent["provider_sha256"]:
        raise RuntimeError("Supported original provider does not reproduce")
    source = OUTPUT / original.name
    source.write_text(f'#line 1 "{original}"\n' + transform(original.read_text()))
    obj = OUTPUT / "matmul_nbits.cc.o"
    command.insert(1, "-I" + str(original.parent))
    command[command.index("-c") + 1] = str(source)
    command[command.index("-o") + 1] = str(obj)
    for option in ("-MF", "-MT"):
        if option in command:
            command[command.index(option) + 1] = str(obj) + ".d" if option == "-MF" else str(obj)
    subprocess.run(command, cwd=parent["working_directory"], env=environment, check=True)
    link[link.index(str(original_object))] = str(obj)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    cpp = OUTPUT / "dense_compare.cpp"
    cpp.write_text(native_source((ROOT / "dense_paired_matrix_compare.cpp").read_text()))
    binary = OUTPUT / "dense_compare"
    subprocess.run([
        "c++", "-O3", "-std=c++20", "-Wall", "-Wextra", "-Werror", str(cpp),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib", "-lonnxruntime", "-o", str(binary),
    ], check=True)
    inputs = [
        Path(__file__), ROOT / "dense_tile_parallel3_transform.py",
        ROOT / "build_dense_tile_parallel3.py",
        ROOT / "prepare_dense_tile_parallel2.py", ROOT / "dense_tile_parallel2_transform.py",
        ROOT / "build_dense_columns8_double_buffer.py", ROOT / "dense_paired_matrix_compare.cpp",
        DENSE, PARENT / "build-manifest.json", PROOF / "baseline.qualification.json",
        original, original_object, baseline, source, cpp, binary,
        *[PROOF / name for name in proof["sdk_sha256"]],
    ]
    report = {
        "status": "built_not_qualified", "provider_sha256": sha(library),
        "baseline_provider_sha256": parent["provider_sha256"], "baseline_byte_identical": True,
        "original_dense_object_byte_identical": True, "original_dense_object_sha256": sha(baseline),
        "flag": FLAG, "program": NAME, "physical_threads": 256, "independent_tile_workers": 2,
        "output_columns": 8, "shared_storage_bytes": SHARED_BYTES, "packed_scratch_words": 768,
        "supported_k": [2560, 6144], "default_enabled": False, "process_static_selector": True,
        "no_object_layout_changes": True, "original_dot_tile_and_serial_fold_order_preserved": True,
        "compile_command": command, "link_command": link, "working_directory": parent["working_directory"],
        "source_sha256": {str(path): sha(path) for path in inputs},
        "performance_qualified": False, "quality_qualified": False,
    }
    with (OUTPUT / "build-manifest.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in
                      ("status", "provider_sha256", "original_dense_object_byte_identical")}), flush=True)


if __name__ == "__main__":
    main()
