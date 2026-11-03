from pathlib import Path
import subprocess
import tempfile
import unittest

from build_dense_flag_cache import HELPER, KERNEL, PATTERN, compute_parts, transform_comparator, transform_source
from prepare_dense_flag_cache_proofs import ROOT as EVIDENCE, base, model_source, native_source


ROOT = Path("/home/sayanshaw/repos")
SOURCE = ROOT / "qwen38-int4-qmoe-prefill-rows4/build-v2/matmul_nbits.cc"
HEADER = ROOT / "qwen38-int4-static-node-replay/build-v8/overlay/contrib_ops/webgpu/quantization/matmul_nbits.h"


class DenseFlagCacheTests(unittest.TestCase):
    def test_private_compute_changes_only_flag_reads(self):
        original = SOURCE.read_text()
        changed = transform_source(original, HEADER.read_text())
        _, _, original_compute = compute_parts(original)
        self.assertIn(original_compute, changed)
        start = changed.index(f"Status {KERNEL}::ComputeInternal(")
        end = changed.index("\n}\n", start) + 2
        cached = changed[start:end].replace(f"{KERNEL}::ComputeInternal", "MatMulNBits::ComputeInternal")
        flags = list(dict.fromkeys(PATTERN.findall(original_compute)))
        for index, (name, default) in enumerate(flags):
            cached = cached.replace(f'flag_cache_.Read({index}, "{name}", {default})',
                                    f'ParseEnvironmentVariableWithDefault<int>("{name}", {default})')
        self.assertEqual(PATTERN.sub(lambda m: f'ParseEnvironmentVariableWithDefault<int>("{m[1]}", {m[2]})',
                                    original_compute), cached)
        self.assertIn("    MatMulNBits,\n", changed)
        self.assertIn(f"    {KERNEL});", changed)
        self.assertIn('static_replay_{"MatMulNBitsExact"', changed)

    def test_comparator_uses_separate_sessions_and_fresh_invalid_flags(self):
        changed = transform_comparator((ROOT / "qwen38-int4-adapter-inline-tensors/build-v3/native_compare.cpp").read_text())
        self.assertIn("auto actual = candidate_session.Run(", changed)
        self.assertIn("Ort::Session invalid_session", changed)

    def test_native_and_model_gates_preserve_original_controls(self):
        native = native_source((EVIDENCE / "run_adapter_inline_native.py").read_text())
        model = model_source((EVIDENCE / "run_adapter_inline_model.py").read_text())
        screen = base((EVIDENCE / "run_adapter_inline_screen.py").read_text())
        self.assertIn('row["requests"] == 128', native)
        self.assertIn('row["entries"] == row["captures"] == 1', native)
        self.assertIn('assert build["private_kernel_layout_only"]', model)
        self.assertNotIn("no_object_layout_changes", model)
        self.assertIn("unset ORT_WEBGPU_DENSE_FLAG_CACHE_STATS_FILE;", model)
        self.assertIn('if phase == "coverage" and mode == "cache":', model)
        self.assertIn("237408768", model)
        self.assertIn("previous = 2048 if args.context == 8192 else 8192", model)
        self.assertIn('providers["control"] != providers["cache"]', screen)
        self.assertIn("0955e1d6cc30cac13044450f692100ffa5f33361aaef229e25ff0efa20a8c8d4", screen)

    def test_lazy_thread_safe_reads_and_dynamic_disabled_mode(self):
        code = r'''
#include <array>
#include <atomic>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>
static std::atomic<int> calls{0};
static std::atomic<int> value{17};
#define LOGS_DEFAULT(severity) std::cerr
template <typename T> T ParseEnvironmentVariableWithDefault(const char*, T fallback) {
  ++calls;
  int current = value.load();
  if (current == -99) throw std::runtime_error("invalid integer");
  return current == -1 ? fallback : current;
}
''' + HELPER + r'''
int main() {
  unsetenv("ORT_WEBGPU_DENSE_FLAG_CACHE_STATS_FILE");
  DenseFrozenFlags dynamic;
  for (int round = 0; round < 1000; ++round) {
    value = round;
    assert(dynamic.Read(0, "fixture", 3) == round);
  }
  assert(calls == 1000);
  DenseFrozenFlags cached;
  cached.Enable(true);
  calls = 0; value = 29;
  std::vector<std::thread> threads;
  for (int worker = 0; worker < 32; ++worker) threads.emplace_back([&]() {
    for (int round = 0; round < 256; ++round)
      for (int slot = 0; slot < 8; ++slot) assert(cached.Read(slot, "fixture", 3) == 29);
  });
  for (auto& thread : threads) thread.join();
  assert(calls == 8);
  value = 73;
  assert(cached.Read(0, "fixture", 3) == 29);
  DenseFrozenFlags fresh;
  fresh.Enable(true);
  assert(fresh.Read(0, "fixture", 3) == 73);
  DenseFrozenFlags retry;
  retry.Enable(true); value = -99;
  try { retry.Read(0, "fixture", 3); assert(false); }
  catch (const std::runtime_error& error) { assert(std::string(error.what()) == "invalid integer"); }
  value = -1;
  assert(retry.Read(0, "fixture", 3) == 3);
  std::cout << "PASS cached reads=65536 parses=8; dynamic=1000; exception retry\n";
}
'''
        with tempfile.TemporaryDirectory(prefix="dense-flags-") as directory:
            cpp, binary = Path(directory) / "probe.cpp", Path(directory) / "probe"
            cpp.write_text(code)
            subprocess.run(["c++", "-O2", "-std=c++20", "-pthread", str(cpp), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main()
