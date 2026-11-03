import subprocess
import tempfile
from pathlib import Path
import unittest

from build_context_flag_cache import HELPERS, transform


class ContextFlagCacheTests(unittest.TestCase):
    def test_only_three_reads_are_replaced_without_changing_checks(self):
        source = (
            'int capacity = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES", 0);\n'
            'ORT_RETURN_IF_NOT(capacity >= 0 && capacity <= 8192, "range");\n'
            'int cache_capacity = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES", 0);\n'
            'ORT_ENFORCE(cache_capacity >= 0 && cache_capacity <= 8192, "range");\n'
            'int dynamic_uniforms = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_DYNAMIC_UNIFORMS", 0);\n'
            'ORT_ENFORCE(dynamic_uniforms == 0 || dynamic_uniforms == 1, "range");\n'
            'Status WebGpuContext::PrepareNodeBinding('
        )
        result = transform(source)
        self.assertIn('ORT_RETURN_IF_NOT(capacity >= 0 && capacity <= 8192, "range")', result)
        self.assertIn('ORT_ENFORCE(dynamic_uniforms == 0 || dynamic_uniforms == 1, "range")', result)
        self.assertEqual(result.count("ContextBindGroupCacheCapacity()"), 3)
        self.assertEqual(result.count("ContextDynamicUniforms()"), 2)

    def test_compiled_actual_helpers_dynamic_frozen_concurrent_and_retry(self):
        prelude = r'''
#include <atomic>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#define ORT_ENFORCE(condition, message) do { if (!(condition)) throw std::runtime_error(message); } while (false)
std::atomic<int> reads{0};
template <typename T> T ParseEnvironmentVariableWithDefault(const std::string& name, T fallback) {
  ++reads;
  const char* text = std::getenv(name.c_str());
  if (!text || !*text) return fallback;
  std::string value(text);
  size_t end = 0;
  int parsed = std::stoi(value, &end);
  if (end != value.size()) throw std::runtime_error("invalid integer");
  return static_cast<T>(parsed);
}
'''
        driver = r'''
int main(int argc, char** argv) {
  if (argc != 2) return 1;
  setenv("ORT_WEBGPU_STATIC_CONTEXT_FLAGS", argv[1], 1);
  int mode = std::stoi(argv[1]);
  if (mode == 2) {
    try { ContextDynamicUniforms(); } catch (const std::runtime_error&) { return 0; }
    return 2;
  }
  setenv("ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES", "invalid", 1);
  try { ContextBindGroupCacheCapacity(); return 3; } catch (const std::exception&) {}
  setenv("ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES", "8192", 1);
  if (ContextBindGroupCacheCapacity() != 8192) return 4;
  int before = reads;
  std::vector<std::thread> threads;
  for (int i = 0; i < 8; ++i) threads.emplace_back([] {
    for (int j = 0; j < 1024; ++j) if (ContextBindGroupCacheCapacity() != 8192) std::abort();
  });
  for (auto& thread : threads) thread.join();
  if (reads - before != (mode ? 0 : 8192)) return 5;
  setenv("ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES", "0", 1);
  if (ContextBindGroupCacheCapacity() != (mode ? 8192 : 0)) return 6;
  setenv("ORT_WEBGPU_DYNAMIC_UNIFORMS", "1", 1);
  if (ContextDynamicUniforms() != 1) return 7;
  setenv("ORT_WEBGPU_DYNAMIC_UNIFORMS", "0", 1);
  if (ContextDynamicUniforms() != (mode ? 1 : 0)) return 8;
  return 0;
}
'''
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "helpers.cc"
            source.write_text(prelude + HELPERS + driver)
            executable = Path(directory) / "helpers"
            subprocess.run(["c++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                            "-pthread", str(source), "-o", str(executable)], check=True)
            for mode in ("0", "1", "2"):
                subprocess.run([str(executable), mode], check=True)


if __name__ == "__main__":
    unittest.main()
