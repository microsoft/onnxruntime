import unittest
from pathlib import Path
import subprocess
import tempfile

from build_dense_replay_scratch import FLAG, transform
from run_dense_replay_scratch_model import model_command, verify_coverage


class ScratchSourceTests(unittest.TestCase):
    def test_default_and_validation(self):
        source = self.source()
        self.assertIn(f'"{FLAG}", 0', source)
        self.assertIn("scratch == 0 || scratch == 1", source)
        self.assertLess(source.index("const int scratch"), source.index("const int enabled"))

    def test_key_lifetime_and_fallback(self):
        source = self.source()
        self.assertIn("auto& key = scratch == 1 ? scratch_key : local_key;", source)
        self.assertLess(source.index("key.clear()"), source.index("key.reserve(97)"))
        self.assertLess(source.index("entry.key = std::move(key)"), source.index("compute());"))
        self.assertIn("cpu && input->SizeInBytes() > 64", source)

    def test_no_storage_or_command_changes(self):
        original = self.original()
        source = transform(original)
        self.assertEqual(original[original.index(" private:"):],
                         source[source.index(" private:"):])
        self.assertEqual(original.count("context.AppendNodeCommands("),
                         source.count("context.AppendNodeCommands("))
        self.assertNotIn("class StaticNodeReplay", source)

    def test_model_statistics_are_not_overridden(self):
        from run_qmoe_prefill_rows4_standard import TEMPLATE, OLD_PROVIDER
        script = (f"{OLD_PROVIDER} {TEMPLATE}.json "
                  f"export ORT_WEBGPU_QMOE_COMMAND_REPLAY_STATS_FILE={TEMPLATE}-stats.jsonl; "
                  f"export ORT_WEBGPU_STATIC_NODE_REPLAY_STATS_FILE={TEMPLATE}-static-stats.jsonl; "
                  "--contexts 2048 --tokens 256 --repetitions 3 --chunk 1024"
                  " --webgpu-validation-mode full")
        command = model_command(["bash", "-lc", script], Path("/provider.so"),
                                Path("/scratch"), 2048, True, "scratch", "coverage")
        self.assertEqual(command[-1].count("ORT_WEBGPU_STATIC_NODE_REPLAY_STATS_FILE="), 1)
        self.assertIn("scratch-static-stats.jsonl", command[-1])
        self.assertIn("--tokens 36 --repetitions 1", command[-1])
        rows = [dict(operator="MatMulNBitsScratchExact", hits=69, binding_hits=68, rebinds=1)] * 171
        self.assertEqual(verify_coverage(rows)["actual_private_dense_replay_instances"], 171)
        with self.assertRaises(AssertionError):
            verify_coverage(rows[:-1])
        with self.assertRaises(AssertionError):
            verify_coverage([dict(kernel_identity=1, hits=69)] * 171)

    def test_actual_key_fragment_reuses_capacity(self):
        source = self.source()
        fragment = source[source.index("    std::vector<uint64_t> local_key;"):
                          source.index("    auto bindings =")]
        program = r'''
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <cassert>
#include <initializer_list>
static size_t allocations = 0;
static bool counting = false;
void* operator new(size_t n) {
  if (counting) ++allocations;
  if (void* p = std::malloc(n)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }
struct OrtDevice { enum { CPU = 0, GPU = 1 }; };
struct Device { int Type() const { return OrtDevice::GPU; } };
struct Location { Device device; };
struct Dimensions {
  std::vector<int64_t> values{1, 2560};
  size_t NumDimensions() const { return values.size(); }
  const auto& GetDims() const { return values; }
};
struct Tensor {
  Dimensions dims;
  int GetElementType() const { return 10; }
  const Dimensions& Shape() const { return dims; }
  ::Location Location() const { return {}; }
  size_t SizeInBytes() const { return 0; }
  const void* DataRaw() const { return nullptr; }
};
struct Context {
  Tensor tensors[3];
  int InputCount() const { return 3; }
  bool EnableMatmulFp32Accumulation() const { return false; }
  template<class T> const T* Input(int n) const { return &tensors[n]; }
};
std::vector<uint64_t> result;
int build(Context& context, int scratch, uint64_t variant, bool miss) {
  auto compute = [] { return -1; };
  std::initializer_list<Tensor*> outputs{&context.tensors[0]};
''' + fragment + r'''
  if (miss) result = std::move(key);
  else result.assign(key.begin(), key.end());
  return 0;
}
int main() {
  Context context;
  result.reserve(256);
  assert(build(context, 1, 0, true) == 0);
  assert(build(context, 1, 0, false) == 0);
  for (int flag : {0, 1}) {
    allocations = 0;
    counting = true;
    for (int i = 0; i < 1000; ++i) {
      context.tensors[0].dims.values[1] = 2560 + i;
      assert(build(context, flag, i, false) == 0);
      assert(result[0] == static_cast<uint64_t>(i));
      assert(result[7] == static_cast<uint64_t>(2560 + i));
      assert(result.size() == 28);
    }
    counting = false;
    assert(allocations == (flag == 0 ? 1000u : 0u));
  }
}
'''
        with tempfile.TemporaryDirectory(prefix="dense-key-test-") as directory:
            path = Path(directory)
            (path / "test.cc").write_text(program)
            subprocess.run(["c++", "-std=c++20", "-O2", str(path / "test.cc"),
                            "-o", str(path / "test")], check=True)
            subprocess.run([str(path / "test")], check=True)

    @staticmethod
    def original():
        return Path("/home/sayanshaw/repos/qwen38-int4-dense-bitcast-unpack/"
                    "build-v1/overlay/static_node_replay.h").read_text()

    def source(self):
        return transform(self.original())


if __name__ == "__main__":
    unittest.main()
