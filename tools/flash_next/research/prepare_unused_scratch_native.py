import json
from pathlib import Path

from build_qmoe_node_replay import replace_once, sha


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-fc2-final-mix/build-v2/native_compare.cpp")
OUTPUT = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-unused-scratch/native-v2")


def transform(source):
    source = replace_once(source, "#include <cmath>", "#include <cmath>\n#include <cstdint>")
    source = replace_once(source, "for (int file = 2; file < argc; ++file) {",
                          "for (int file = 2; file < argc; ++file) {\n"
                          "      uint64_t fingerprint = 14695981039346656037ull;")
    source = replace_once(source, "          ++comparisons;",
        "          const auto* bytes = reinterpret_cast<const unsigned char*>(&value);\n"
        "          for (size_t byte = 0; byte < sizeof(value); ++byte) {\n"
        "            fingerprint ^= bytes[byte];\n"
        "            fingerprint *= 1099511628211ull;\n"
        "          }\n          ++comparisons;")
    return replace_once(source, '<< " rounds=32 invalid_flag_rejected=1" << std::endl;',
                        '<< " rounds=32 invalid_flag_rejected=1 output_fingerprint="\n'
                        "                << fingerprint << std::endl;")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    target = OUTPUT / SOURCE.name
    target.write_text(transform(SOURCE.read_text()))
    (OUTPUT / "generation.json").write_text(json.dumps({
        "status": "generated_not_qualified",
        "source_sha256": {str(path): sha(path) for path in (Path(__file__), SOURCE, target)},
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
