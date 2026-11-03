import json
from pathlib import Path
import subprocess

from build_qmoe_node_replay import sha


ROOT = Path(__file__).parent
OUTPUT = Path("/workspace/qwen38-int4-qmoe-fc1-double-buffer/native-proof-v1")
SDK = Path("/workspace/qwen38-fp8-webgpu-runtime/sdk/lib")


if __name__ == "__main__":
    manifest = OUTPUT / "build-manifest.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    test_source = OUTPUT / "reference_io_test.cpp"
    with test_source.open("x") as stream:
        stream.write((ROOT / "hc_reference_io_test.cpp").read_text())
    binaries = []
    for name, source in (("independent_compare", OUTPUT / "independent_compare.cpp"),
                         ("reference_io_test", test_source)):
        binary = OUTPUT / name
        if binary.exists():
            raise FileExistsError(binary)
        subprocess.run(["c++", "-O3", "-std=c++20", str(source),
                        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
                        f"-L{SDK}", f"-Wl,-rpath,{SDK}", "-lonnxruntime", "-o", str(binary)], check=True)
        binaries.append(binary)
    subprocess.run([str(binaries[1]), str(OUTPUT / "reference_io_test.bin")], check=True)
    evidence = [Path(__file__), ROOT / "prepare_qmoe_fc1_double_buffer_native.py",
                ROOT / "test_qmoe_fc1_double_buffer_native.py", ROOT / "hc_reference_io_test.cpp",
                ROOT / "hc_original_reference_io.inc", OUTPUT / "hc_original_reference_io.inc",
                OUTPUT / "independent_compare.cpp", test_source, *binaries,
                Path("/workspace/qwen38-int4-fc2-compact20/build-v1/native_compare.cpp")]
    report = dict(status="independent_qmoe_fc1_comparator_built_io_tested",
                  changing_input_and_route_schedule_preserved=True,
                  same_provider_reference_controls_original_fc1_fusion=True,
                  output_serialized=True, bitwise_float16_float32=True,
                  shape_and_type_serialized=True, numerical_mismatch_exit_code=2,
                  truncated_reference_rejected=True,
                  source_sha256={str(path): sha(path) for path in evidence})
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], flush=True)
