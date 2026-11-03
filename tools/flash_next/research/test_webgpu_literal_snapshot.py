from pathlib import Path
import subprocess
import tempfile
import unittest

from build_webgpu_literal_snapshot import transform


ROOT = Path(__file__).parent
STUB = r"""
#include <atomic>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#define ORT_ENFORCE(condition, message) do { if (!(condition)) throw std::runtime_error(message); } while (false)
namespace onnxruntime {
template<class T> using optional = std::optional<T>;
inline std::atomic<int> parsed_calls{0};
template<class T> optional<T> ParseEnvironmentVariable(const std::string& name) {
  ++parsed_calls;
  const char* text = std::getenv(name.c_str());
  if (text == nullptr || *text == 0) return {};
  if constexpr (std::is_same_v<T, int>) return std::stoi(text);
  else return std::stof(text);
}
template<class T> T ParseEnvironmentVariableWithDefault(const std::string& name, const T& fallback) {
  const auto parsed = ParseEnvironmentVariable<T>(name);
  return parsed.has_value() ? *parsed : fallback;
}
}
"""
MAIN = r"""
#include "snapshot.h"
#include <cassert>
#include <thread>
#include <vector>
using namespace onnxruntime;
int FromSecondTranslationUnit();
int main(int argc, char** argv) {
  assert(argc == 2);
  const int mode = std::stoi(argv[1]);
  assert(setenv("ORT_WEBGPU_LITERAL_SNAPSHOT", argv[1], 1) == 0);
  if (mode == 2) {
    bool rejected = false;
    try { (void)WebGpuLiteralSnapshotDefault<int, "ORT_WEBGPU_TEST_A">(0); }
    catch(const std::runtime_error& error) {
      rejected = std::string(error.what()).find("must be 0 or 1") != std::string::npos;
    }
    assert(rejected);
    return 0;
  }
  assert(setenv("ORT_WEBGPU_TEST_A", "1", 1) == 0);
  assert((WebGpuLiteralSnapshotDefault<int, "ORT_WEBGPU_TEST_A">(0) == 1));
  assert(setenv("ORT_WEBGPU_TEST_A", "2", 1) == 0);
  assert(FromSecondTranslationUnit() == (mode ? 1 : 2));
  assert(unsetenv("ORT_WEBGPU_TEST_B") == 0);
  assert((WebGpuLiteralSnapshotDefault<int, "ORT_WEBGPU_TEST_B">(7) == 7));
  assert((WebGpuLiteralSnapshotDefault<int, "ORT_WEBGPU_TEST_B">(9) == 9));
  assert(setenv("ORT_WEBGPU_TEST_B", "3", 1) == 0);
  assert((WebGpuLiteralSnapshotDefault<int, "ORT_WEBGPU_TEST_B">(7) == (mode ? 7 : 3)));
  assert(setenv("ORT_WEBGPU_TEST_C", "invalid", 1) == 0);
  bool rejected = false;
  try { (void)WebGpuLiteralSnapshotOptional<int, "ORT_WEBGPU_TEST_C">(); }
  catch(const std::invalid_argument&) { rejected = true; }
  assert(rejected);
  assert(setenv("ORT_WEBGPU_TEST_C", "4", 1) == 0);
  assert((WebGpuLiteralSnapshotDefault<int, "ORT_WEBGPU_TEST_C">(0) == 4));
  assert(setenv("ORT_WEBGPU_TEST_FLOAT", "1.5", 1) == 0);
  assert((WebGpuLiteralSnapshotDefault<float, "ORT_WEBGPU_TEST_FLOAT">(0.0f) == 1.5f));
  assert(setenv("ORT_WEBGPU_TEST_FLOAT", "2.5", 1) == 0);
  assert((WebGpuLiteralSnapshotDefault<float, "ORT_WEBGPU_TEST_FLOAT">(0.0f) == 2.5f));
  assert(setenv("ORT_WEBGPU_TEST_THREADS", "5", 1) == 0);
  const int before = parsed_calls;
  std::vector<std::thread> threads;
  for (int i = 0; i < 16; ++i) threads.emplace_back([] {
    assert((WebGpuLiteralSnapshotDefault<int, "ORT_WEBGPU_TEST_THREADS">(0) == 5));
  });
  for (auto& thread : threads) thread.join();
  assert(parsed_calls - before == (mode ? 1 : 16));
}
"""


class LiteralSnapshotTests(unittest.TestCase):
    def test_scoped_calls_only_and_shader_strings_comments_unchanged(self):
        source = r'''
// ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_COMMENT", 0)
const char* shader = R"wgsl(
// ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_SHADER", 0)
)wgsl";
const char* quoted = "ParseEnvironmentVariable<int>(\"ORT_WEBGPU_QUOTED\")";
auto capacity = 1'024;
auto wide = L'x';
auto a = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_A", 0);
auto b = onnxruntime::ParseEnvironmentVariable<int>("ORT_WEBGPU_B");
auto c = ParseEnvironmentVariableWithDefault<int>(flag, 0);
auto d = ParseEnvironmentVariableWithDefault<float>("ORT_WEBGPU_D", 0.0f);
auto e = ParseEnvironmentVariableWithDefault<int>("ORT_OTHER", 0);
'''
        result, rows = transform(source)
        self.assertEqual([row["flag"] for row in rows], ["ORT_WEBGPU_A", "ORT_WEBGPU_B"])
        restored = result.replace(
            'WebGpuLiteralSnapshotDefault<int, "ORT_WEBGPU_A">(',
            'ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_A",').replace(
                'WebGpuLiteralSnapshotOptional<int, "ORT_WEBGPU_B">()',
                'ParseEnvironmentVariable<int>("ORT_WEBGPU_B")')
        self.assertEqual(restored, source)

    def test_actual_header_dynamic_snapshot_cross_tu_threads_errors_and_defaults(self):
        header = (ROOT / "webgpu_literal_snapshot.h").read_text()
        self.assertEqual(header.count('#include "core/platform/env_var_utils.h"'), 1)
        header = header.replace('#include "core/platform/env_var_utils.h"', STUB)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "snapshot.h").write_text(header)
            (path / "main.cc").write_text(MAIN)
            (path / "second.cc").write_text(
                '#include "snapshot.h"\nint FromSecondTranslationUnit() {\n'
                ' return onnxruntime::WebGpuLiteralSnapshotDefault<int, "ORT_WEBGPU_TEST_A">(0);\n}\n')
            executable = path / "test"
            subprocess.run([
                "c++", "-std=c++20", "-pthread", "-Wall", "-Wextra", "-Werror",
                str(path / "main.cc"), str(path / "second.cc"), "-o", str(executable),
            ], check=True)
            for mode in ("0", "1", "2"):
                subprocess.run([str(executable), mode], check=True)


if __name__ == "__main__":
    unittest.main()
