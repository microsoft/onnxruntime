import json
from pathlib import Path
import subprocess

from attribute_cpu_allocations_v2 import sha
from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent
SOURCE = Path("/workspace/qwen38-int4-qmoe-unused-scratch/native-v2/native_compare.cpp")
OUTPUT = Path("/workspace/qwen38-int4-webgpu-literal-snapshot/native-v1")
SDK = Path("/workspace/qwen38-fp8-webgpu-runtime/sdk")


def transform(source):
    source = replace_once(
        source, "#include <cmath>", "#include <cmath>\n#include <cstdio>\n#include <fcntl.h>")
    source = replace_once(
        source, '    if (argc < 3) throw std::runtime_error("Usage: native_compare provider fixture...");',
        '    if (argc < 5) throw std::runtime_error("Usage: native_fixed provider write|compare golden fixture...");\n'
        '    const bool write_reference = std::string(argv[2]) == "write";\n'
        '    if (!write_reference && std::string(argv[2]) != "compare") throw std::runtime_error("Invalid golden mode");\n'
        '    std::FILE* golden = nullptr;\n'
        '    if (write_reference) {\n'
        '      const int fd = open(argv[3], O_WRONLY | O_CREAT | O_EXCL, 0644);\n'
        '      if (fd < 0) throw std::runtime_error("Cannot exclusively create fresh golden");\n'
        '      golden = fdopen(fd, "wb");\n'
        '    } else { golden = std::fopen(argv[3], "rb"); }\n'
        '    if (golden == nullptr) throw std::runtime_error("Cannot open golden stream");\n'
        '    static_assert(sizeof(Ort::Float16_t) == 2);')
    start = source.index('    auto set_flag = [](const char* name, const char* value) {')
    end = source.index("    size_t comparisons = 0;", start)
    source = source[:start] + source[end:]
    source = replace_once(source, "for (int file = 2;", "for (int file = 4;")
    start = source.index('        const char* gate_mode =')
    end = source.index("        const auto info =", start)
    source = source[:start] + (
        "        auto actual = session.Run(Ort::RunOptions{}, input_names, inputs.data(), 2, output_names, 1);\n"
    ) + source[end:]
    start = source.index("        const auto info =")
    end = source.index("        for (size_t i = 0;", start)
    source = source[:start] + r'''
        const auto info = actual[0].GetTensorTypeAndShapeInfo();
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
          throw std::runtime_error("Golden output is not FP16");
        auto shape = info.GetShape();
        uint64_t rank = shape.size();
        uint64_t count = info.GetElementCount();
        if (write_reference) {
          if (std::fwrite(&rank, sizeof(rank), 1, golden) != 1 ||
              std::fwrite(shape.data(), sizeof(int64_t), rank, golden) != rank ||
              std::fwrite(&count, sizeof(count), 1, golden) != 1)
            throw std::runtime_error("Golden shape write failed");
        } else {
          uint64_t expected_rank = 0, expected_count = 0;
          if (std::fread(&expected_rank, sizeof(expected_rank), 1, golden) != 1 ||
              expected_rank != rank) throw std::runtime_error("Golden rank differs");
          std::vector<int64_t> expected_shape(rank);
          if (std::fread(expected_shape.data(), sizeof(int64_t), rank, golden) != rank ||
              expected_shape != shape ||
              std::fread(&expected_count, sizeof(expected_count), 1, golden) != 1 ||
              expected_count != count) throw std::runtime_error("Golden shape/count differs");
        }
''' + source[end:]
    source = replace_once(
        source, "          const auto& expected = reference[0].GetTensorData<Ort::Float16_t>()[i];\n"
                "          const auto& value = actual[0].GetTensorData<Ort::Float16_t>()[i];",
        "          const auto& value = actual[0].GetTensorData<Ort::Float16_t>()[i];\n"
        "          Ort::Float16_t expected;\n"
        "          if (write_reference) {\n"
        "            if (std::fwrite(&value, sizeof(value), 1, golden) != 1)\n"
        '              throw std::runtime_error("Golden data write failed");\n'
        "            expected = value;\n"
        "          } else if (std::fread(&expected, sizeof(expected), 1, golden) != 1) {\n"
        '            throw std::runtime_error("Golden data read failed");\n'
        "          }")
    start = source.index('      set_flag("ORT_WEBGPU_QMOE_FC2_FINAL_MIX", "2");')
    end = source.index('      std::cout << "PASS changing_inputs_and_routes ', start)
    source = source[:start] + source[end:]
    source = replace_once(
        source, "rounds=32 invalid_flag_rejected=1 output_fingerprint=",
        "rounds=32 fixed_startup_flags=1 output_fingerprint=")
    source = replace_once(
        source, '    std::cout << "PASS total_exact_elements=" << comparisons',
        '    if (!write_reference && (std::fgetc(golden) != EOF || std::ferror(golden)))\n'
        '      throw std::runtime_error("Golden has trailing data or read failure");\n'
        '    if (std::fclose(golden) != 0) throw std::runtime_error("Golden close/flush failed");\n'
        '    std::cout << (write_reference ? "PASS total_finite_elements=" : "PASS total_exact_elements=") << comparisons')
    if "setenv(" in source or "set_flag(" in source or "reference[" in source:
        raise RuntimeError("Fixed-process native still mutates selectors or uses in-process reference")
    return source


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    source = OUTPUT / "native_fixed.cpp"
    source.write_text(transform(SOURCE.read_text()))
    binary = OUTPUT / "native_fixed"
    command = [
        "c++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
        "-I" + str(SDK / "include"), str(source),
        "-L" + str(SDK / "lib"), "-Wl,-rpath," + str(SDK / "lib"),
        "-lonnxruntime", "-o", str(binary),
    ]
    subprocess.run(command, check=True)
    evidence = [Path(__file__), SOURCE, source, binary,
                *(SDK / "include").glob("*.h"), SDK / "lib/libonnxruntime.so"]
    report = {
        "status": "fixed_startup_native_golden_comparator_built",
        "compile_command": command, "binary_sha256": sha(binary),
        "flags_never_mutated_inside_process": True,
        "exclusive_fresh_golden_writer": True,
        "golden_output_rank_shape_count_finite_and_every_byte_checked": True,
        "source_sha256": {str(path): sha(path) for path in evidence},
    }
    (OUTPUT / "generation.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
