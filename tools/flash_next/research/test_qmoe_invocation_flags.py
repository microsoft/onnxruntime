import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from build_qmoe_invocation_flags import FIELDS, STRUCT, transform


REPOS = Path("/home/sayanshaw/repos")
SOURCE = REPOS / "qwen38-int4-fc2-private-a/build-v1/qmoe.cc"
HEADER = REPOS / "qwen38-int4-qmoe-prefill256/build-v1/overlay/contrib_ops/webgpu/moe/qmoe.h"


class InvocationFlagTests(unittest.TestCase):
    def test_shader_prefix_replay_keys_and_object_members_unchanged(self):
        original, header = SOURCE.read_text(), HEADER.read_text()
        source, result_header = transform(original, header)
        prefix = "Status QMoE::ComputeInternal("
        self.assertEqual(source.split(prefix)[0], original.split(prefix)[0])
        key_begin, key_end = "  key.push_back(fc1_activation);", "  auto append_tensor"
        self.assertEqual(source.split(key_begin)[1].split(key_end)[0],
                         original.split(key_begin)[1].split(key_end)[0])
        restored = result_header.replace(STRUCT, "").replace(
            "std::array<std::optional<Tensor>, 5>* scratch,\n"
            "                              const InvocationFlags* invocation_flags) const;",
            "std::array<std::optional<Tensor>, 5>* scratch) const;")
        self.assertEqual(restored, header)
        self.assertEqual(source.count(", selected_flags)"), 3)

    def test_actual_generated_reads_are_invocation_local_not_persistent(self):
        source, _ = transform(SOURCE.read_text(), HEADER.read_text())
        body = source.split("Status QMoE::ComputeWithWorkspace(")[1]
        expressions = []
        for field, flag in FIELDS:
            expression = (
                f'(invocation_flags != nullptr ? invocation_flags->{field} : '
                f'ParseEnvironmentVariableWithDefault<int>("{flag}", 0))')
            self.assertEqual(body.count(expression), 1)
            expressions.append(expression)
        cpp = (
            "#include <cassert>\n#include <cstdlib>\n#include <stdexcept>\n"
            + STRUCT
            + "template<class T> T ParseEnvironmentVariableWithDefault(const char* name, T fallback) {\n"
              "  const char* text = std::getenv(name);\n"
              "  return text == nullptr ? fallback : static_cast<T>(std::stoi(text));\n}\n"
              "int main() {\n"
              "  InvocationFlags first{0,0,0,0,0,0,0,0,0};\n"
              "  InvocationFlags second{1,1,1,1,1,1,1,1,1};\n"
        )
        for expression, (_, flag) in zip(expressions, FIELDS, strict=True):
            cpp += (
                f'  setenv("{flag}", "1", 1);\n'
                "  { const InvocationFlags* invocation_flags = &first;\n"
                f"    assert({expression} == 0); }}\n"
                "  { const InvocationFlags* invocation_flags = &second;\n"
                f"    assert({expression} == 1); }}\n"
                "  { const InvocationFlags* invocation_flags = nullptr;\n"
                f"    assert({expression} == 1);\n"
                f'    setenv("{flag}", "0", 1);\n'
                f"    assert({expression} == 0);\n"
                f'    setenv("{flag}", "invalid", 1);\n'
                "    bool rejected = false;\n"
                f"    try {{ (void){expression}; }} catch(const std::invalid_argument&) {{ rejected = true; }}\n"
                "    assert(rejected);\n  }\n"
            )
        cpp += "}\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.cc"
            path.write_text(cpp)
            executable = Path(directory) / "test"
            subprocess.run(["c++", "-std=c++20", "-Wall", "-Wextra", "-Werror",
                            str(path), "-o", str(executable)], check=True)
            subprocess.run([str(executable)], check=True, env=os.environ.copy())


if __name__ == "__main__":
    unittest.main()
