import json
import os
from pathlib import Path
import subprocess

from build_dense_double_buffer import transform_compare
from build_qmoe_node_replay import replace_once, sha


FLAG = "ORT_WEBGPU_DENSE_REPLAY_SCRATCH"
PARENT = Path("/workspace/qwen38-int4-qmoe-prefill-rows4/build-v2")
HEADERS = Path("/workspace/qwen38-int4-static-node-replay/build-v8/overlay")
REPLAY = Path("/workspace/qwen38-int4-dense-bitcast-unpack/build-v1/overlay/static_node_replay.h")
COMPARATOR = Path("/workspace/qwen38-int4-dense-block32/build-v1")
OUTPUT = Path("/workspace/qwen38-int4-dense-replay-scratch/build-v1")


def transform(source):
    source = source.replace("StaticNodeReplay", "DenseScratchNodeReplay")
    signature = "int64_t row_elements = 0, uint64_t variant = 0) {"
    source = replace_once(source, signature, signature + f'''
    const int scratch = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
    ORT_RETURN_IF_NOT(scratch == 0 || scratch == 1, "{FLAG} must be 0 or 1");
''')
    source = replace_once(source, "    std::vector<uint64_t> key;\n    key.reserve(97);", '''    std::vector<uint64_t> local_key;
    static thread_local std::vector<uint64_t> scratch_key;
    auto& key = scratch == 1 ? scratch_key : local_key;
    key.clear();
    key.reserve(97);
''')
    return source


def build_variant(output, flag, replay_transform, replay_type, stats_name,
                  header_name="dense_scratch_node_replay.h", kernel_type=None,
                  source_transform=None, header_transform=None, comparator_transform=None):
    output = Path(output)
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    for name, digest in parent["source_sha256"].items():
        assert sha(PARENT / name) == digest, name
    output.mkdir(parents=True, exist_ok=False)
    overlay = output / "overlay"
    header = overlay / "contrib_ops/webgpu/quantization/matmul_nbits.h"
    header.parent.mkdir(parents=True)
    replay = overlay / header_name
    replay.write_text(replay_transform(REPLAY.read_text()))
    original_header = HEADERS / "contrib_ops/webgpu/quantization/matmul_nbits.h"
    text = replace_once(original_header.read_text(), '#include "static_node_replay.h"',
                        '#include "static_node_replay.h"\n' + f'#include "{header_name}"')
    if kernel_type is None:
        text += ("\nstatic_assert(sizeof(onnxruntime::webgpu::StaticNodeReplay) ==\n"
                 f"              sizeof(onnxruntime::webgpu::{replay_type}));\n")
    text = replace_once(text, 'StaticNodeReplay static_replay_{"MatMulNBitsExact"',
                        f'{replay_type} static_replay_{{"{stats_name}"')
    if kernel_type is not None:
        text = replace_once(text, "class MatMulNBits final", f"class {kernel_type} final")
        text = replace_once(text, "  MatMulNBits(", f"  {kernel_type}(")
    if header_transform is not None:
        text = header_transform(text)
    header.write_text(text)
    library = output / "libonnxruntime_providers_webgpu.so"
    link = parent["link_command"].copy()
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=parent["working_directory"], check=True)
    assert sha(library) == parent["provider_sha256"], "Parent relink mismatch"
    library.unlink()
    command = next(c.copy() for c in parent["compile_commands"]
                   if c[c.index("-c") + 1] == str(PARENT / "matmul_nbits.cc"))
    command.insert(1, "-I" + str(overlay))
    source = PARENT / "matmul_nbits.cc"
    if kernel_type is not None:
        source = output / "matmul_nbits.cc"
        code = (PARENT / "matmul_nbits.cc").read_text()
        if source_transform is not None:
            code = source_transform(code)
        code = replace_once(code,
                            "Status MatMulNBits::ComputeInternal",
                            f"Status {kernel_type}::ComputeInternal")
        code = replace_once(code, "    MatMulNBits);", f"    {kernel_type});")
        source.write_text(code)
        command[command.index("-c") + 1] = str(source)
    obj = output / "matmul_nbits.cc.o"
    command[command.index("-o") + 1] = str(obj)
    subprocess.run(command, cwd=parent["working_directory"], check=True)
    original_object = str(PARENT / "matmul_nbits.cc.o")
    assert link.count(original_object) == 1
    link[link.index(original_object)] = str(obj)
    subprocess.run(link, cwd=parent["working_directory"], check=True)
    comparison = output / "native_compare.cpp"
    compare = transform_compare(replace_once((COMPARATOR / "native_compare.cpp").read_text(),
        'setenv("ORT_WEBGPU_DENSE_BLOCK32", value, 1)',
        f'setenv("{flag}", value, 1)'), flag=flag)
    if comparator_transform is not None:
        compare = comparator_transform(compare)
    comparison.write_text(compare)
    comparator_command = ["c++", "-O3", "-std=c++20", str(comparison),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib", "-lonnxruntime",
        "-o", str(output / "native_compare")]
    subprocess.run(comparator_command, check=True)
    sources = [replay, header, comparison, REPLAY, original_header, source, PARENT / "matmul_nbits.cc"]
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
        baseline_provider_sha256=parent["provider_sha256"], baseline_byte_identical=True,
        comparator_sha256=sha(output / "native_compare"),
        source_sha256={os.path.relpath(p, output): sha(p) for p in sources},
        compile_commands=[command], comparator_compile_command=comparator_command,
        link_command=link, working_directory=parent["working_directory"],
        shader_and_arithmetic_unchanged=True, default_enabled=False,
        private_replay_type=True, kernel_storage_size_unchanged=kernel_type is None,
        private_kernel_type=kernel_type,
        quality_qualified=False, headline_eligible=False)
    (output / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "provider_sha256", "baseline_byte_identical")}))

def main():
    build_variant(OUTPUT, FLAG, transform, "DenseScratchNodeReplay", "MatMulNBitsScratchExact")



if __name__ == "__main__":
    main()
