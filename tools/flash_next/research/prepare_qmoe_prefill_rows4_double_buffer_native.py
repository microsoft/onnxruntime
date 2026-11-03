import argparse
import json
from pathlib import Path
import subprocess

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent
SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2/native_compare.cpp")
PROOF = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4-double-buffer/native-proof-v2")


def comparator_source():
    source = SOURCE.read_text()
    source = replace_once(source, "int main(int argc, char** argv) {",
                          '#include "hc_original_reference_io.inc"\n\nint main(int argc, char** argv) {')
    source = replace_once(source,
        'if (argc < 3) throw std::runtime_error("Usage: native_compare provider fixture...");',
        'if (argc < 5) throw std::runtime_error("Usage: native_compare provider capture|compare reference-dir fixture...");\n'
        '    const std::string mode = argv[2];\n'
        '    if (mode != "capture" && mode != "compare") throw std::runtime_error("Invalid reference mode");\n'
        '    const bool capture = mode == "capture";\n'
        '    if (!std::filesystem::is_directory(argv[3])) throw std::runtime_error("Missing reference directory");')
    source = replace_once(source, "for (int file = 2; file < argc; ++file) {",
        'for (int file = 4; file < argc; ++file) {\n'
        '      const auto reference_path = std::filesystem::path(argv[3]) /\n'
        '          (std::filesystem::path(argv[file]).stem().string() + ".reference.bin");\n'
        '      if (capture && std::filesystem::exists(reference_path))\n'
        '        throw std::runtime_error("Existing original reference");\n'
        '      std::fstream original(reference_path, std::ios::binary | (capture ? std::ios::out : std::ios::in));\n'
        '      if (!original) throw std::runtime_error("Cannot open original reference");')
    source = source.replace("grouped-prefill-rows4-v2-native-", "prefill-rows4-double-buffer-v2-native-")
    source = replace_once(source, "          for (size_t i = 0; i < expected.size(); ++i) {",
        '          TransferOriginalOutput(original, capture, actual[0],\n'
        '              std::string(argv[file]) + " rows=" + std::to_string(rows) + " round=" + std::to_string(round));\n'
        "          for (size_t i = 0; i < expected.size(); ++i) {")
    source = replace_once(source, '      flag("2");',
        '      if (!capture && original.peek() != std::char_traits<char>::eof())\n'
        '        throw std::runtime_error("Trailing original reference bytes");\n'
        '      original.close();\n'
        '      if (original.fail()) throw std::runtime_error("Cannot close original reference");\n'
        '      flag("2");')
    source = replace_once(source,
        'throw std::runtime_error(std::string(argv[file]) + " finite/bitwise mismatch rows="',
        'throw NumericMismatch(std::string(argv[file]) + " finite/bitwise mismatch rows="')
    return replace_once(source, "  } catch (const std::exception& error) {",
        '  } catch (const NumericMismatch& error) {\n'
        '    std::cerr << "FAIL " << error.what() << std::endl;\n'
        '    return 2;\n'
        '  } catch (const std::exception& error) {')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if not args.build:
        PROOF.mkdir(parents=True, exist_ok=False)
        (PROOF / "independent_compare.cpp").write_text(comparator_source())
        (PROOF / "hc_original_reference_io.inc").write_text((ROOT / "hc_original_reference_io.inc").read_text())
        (PROOF / "reference_io_test.cpp").write_text((ROOT / "hc_reference_io_test.cpp").read_text())
        print(PROOF)
    else:
        proof = Path(str(PROOF).replace("/home/sayanshaw/repos", "/workspace"))
        manifest = proof / "build-manifest.json"
        if manifest.exists():
            raise FileExistsError(manifest)
        sdk = Path("/workspace/qwen38-fp8-webgpu-runtime/sdk/lib")
        binaries = []
        for name in ("independent_compare", "reference_io_test"):
            binary = proof / name
            if binary.exists():
                raise FileExistsError(binary)
            subprocess.run(["c++", "-O3", "-std=c++20", str(proof / (name + ".cpp")),
                "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
                f"-L{sdk}", f"-Wl,-rpath,{sdk}", "-lonnxruntime", "-o", str(binary)], check=True)
            binaries.append(binary)
        subprocess.run([str(binaries[1]), str(proof / "io-test.bin")], check=True)
        evidence = [Path(__file__), ROOT / "hc_original_reference_io.inc",
                    ROOT / "hc_reference_io_test.cpp",
                    Path(str(SOURCE).replace("/home/sayanshaw/repos", "/workspace")),
                    proof / "independent_compare.cpp", proof / "hc_original_reference_io.inc",
                    proof / "reference_io_test.cpp", *binaries]
        manifest.write_text(json.dumps(dict(
            status="independent_prefill_comparator_built_io_tested",
            original_chunk16_reference_preserved=True, changing_input_route_schedule_preserved=True,
            bitwise_float16_float32=True, truncated_reference_rejected=True,
            source_sha256={str(path): sha(path) for path in evidence}), indent=2) + "\n")
        print("independent_prefill_comparator_built_io_tested", flush=True)
