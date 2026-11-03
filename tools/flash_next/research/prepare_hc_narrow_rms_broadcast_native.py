from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent
SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-projection/build-v2/native_compare.cpp")
OUTPUT = Path("/home/sayanshaw/repos/qwen38-int4-hc-narrow-rms-broadcast/native-proof-v1")


def comparator_source():
    source = SOURCE.read_text()
    source = replace_once(source, "struct InputStorage {",
                          '#include "hc_original_reference_io.inc"\n\nstruct InputStorage {')
    source = replace_once(
        source, 'if (argc < 3) throw std::runtime_error("Usage: native_compare provider fixture...");',
        'if (argc < 5) throw std::runtime_error("Usage: native_compare provider capture|compare reference-dir fixture...");\n'
        '    const std::string mode = argv[2];\n'
        '    if (mode != "capture" && mode != "compare") throw std::runtime_error("Invalid reference mode");\n'
        '    const bool capture = mode == "capture";\n'
        '    if (!std::filesystem::is_directory(argv[3])) throw std::runtime_error("Missing reference directory");')
    source = replace_once(
        source, "for (int file = 2; file < argc; ++file) {",
        'for (int file = 4; file < argc; ++file) {\n'
        '      const auto reference_path = std::filesystem::path(argv[3]) /\n'
        '          (std::filesystem::path(argv[file]).stem().string() + ".reference.bin");\n'
        '      if (capture && std::filesystem::exists(reference_path))\n'
        '        throw std::runtime_error("Existing original reference");\n'
        '      std::fstream original(reference_path, std::ios::binary | (capture ? std::ios::out : std::ios::in));\n'
        '      if (!original) throw std::runtime_error("Cannot open original reference");')
    source = replace_once(
        source, "            ++comparisons;\n          }\n        }\n      }\n      flag(\"2\");",
        '            ++comparisons;\n          }\n'
        '          TransferOriginalOutput(original, capture, actual[output],\n'
        '              std::string(argv[file]) + " round=" + std::to_string(round) + " output=" + std::to_string(output));\n'
        '        }\n      }\n'
        '      if (!capture && original.peek() != std::char_traits<char>::eof())\n'
        '        throw std::runtime_error("Trailing original reference bytes");\n'
        '      original.close();\n'
        '      if (original.fail()) throw std::runtime_error("Cannot close original reference");\n'
        '      flag("2");')
    source = replace_once(
        source, 'throw std::runtime_error(std::string(argv[file]) + " finite/bitwise mismatch round="',
        'throw NumericMismatch(std::string(argv[file]) + " finite/bitwise mismatch round="')
    return replace_once(
        source, "  } catch (const std::exception& error) {",
        '  } catch (const NumericMismatch& error) {\n'
        '    std::cerr << "FAIL " << error.what() << std::endl;\n'
        '    return 2;\n'
        '  } catch (const std::exception& error) {')


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "independent_compare.cpp").write_text(comparator_source())
    (OUTPUT / "hc_original_reference_io.inc").write_text(
        (ROOT / "hc_original_reference_io.inc").read_text())
    print(OUTPUT)
