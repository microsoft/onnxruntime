from pathlib import Path
import json
import os
import subprocess
import tempfile
import unittest

from build_adapter_tensor_reuse import FLAG, HEADER, HELPERS, relocated, transform


HOST_HEADER = Path(str(HEADER).replace("/workspace", "/home/sayanshaw/repos", 1))


class AdapterTensorReuseTests(unittest.TestCase):
    def test_tensor_getters_and_storage_layout_are_unchanged(self):
        original = HOST_HEADER.read_text()
        changed = transform(original)
        start = "  template <typename T,\n            typename ="
        a = original.index(start, original.index("struct OpKernelContext {"))
        b = changed.index(start, changed.index("struct OpKernelContext {"))
        self.assertEqual(original[a:], changed[b:])

    def test_relocation_preserves_source_and_does_not_overwrite_dependencies(self):
        command = ["c++", "-include", "old-pch.hxx", "-MT", "old.o", "-MF", "old.o.d",
                   "-c", "original.cc", "-o", "old.o"]
        actual = relocated(command, Path("/tmp/adapter-test.o"), Path("/tmp/adapter-overlay"))
        self.assertEqual(command[-1], "old.o")
        self.assertNotIn("old-pch.hxx", actual)
        self.assertTrue(actual[actual.index("-include") + 1].endswith("/ep/adapters.h"))
        self.assertEqual(actual[actual.index("-c") + 1], "original.cc")
        self.assertEqual(actual[actual.index("-MF") + 1], "/tmp/adapter-test.o.d")
        self.assertEqual(actual[1], "-I/tmp/adapter-overlay")

    def test_actual_constructor_destructor_clear_slots_and_reuse_capacity(self):
        original = HOST_HEADER.read_text()
        changed = transform(original)
        start = original.index("  explicit OpKernelContext(")
        end = original.index("  template <typename T,", start)
        old_ctor = original[start:end].replace("OpKernelContext(", "OriginalContext(")
        start = changed.index("  explicit OpKernelContext(")
        end = changed.index("  template <typename T,", start)
        constructors = changed[start:end]
        source = r'''
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>
static bool counting;
static size_t allocations;
void* operator new(size_t n) {
  if (counting) ++allocations;
  if (void* p = std::malloc(n ? n : 1)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }
#define ORT_ENFORCE(condition, message) do { if (!(condition)) throw std::runtime_error(message); } while (0)
#define LOGS_DEFAULT(severity) std::cerr
template <typename T>
T ParseEnvironmentVariableWithDefault(const char* name, T fallback) {
  const char* value = std::getenv(name);
  return value ? static_cast<T>(std::stoi(value)) : fallback;
}
struct Tensor { int value = 0; };
struct OrtKernelContext { size_t inputs; size_t outputs; };
namespace Ort {
struct KernelContext {
  OrtKernelContext* context;
  size_t GetInputCount() const { return context->inputs; }
  size_t GetOutputCount() const { return context->outputs; }
};
}
struct KernelInfo {
  std::vector<Tensor> constants;
  const std::vector<Tensor>& GetConstantInputTensors() const { return constants; }
};
struct OpKernel {
  KernelInfo info;
  const KernelInfo& Info() const { return info; }
};
''' + HELPERS
        fields = r'''
  Ort::KernelContext context_;
  const OpKernel& op_kernel_;
  const std::vector<Tensor>& constant_input_tensors_;
  std::vector<Tensor> input_tensors_;
  std::vector<Tensor> output_tensors_;
'''
        source += "struct OriginalContext {\n" + old_ctor + fields + "};\n"
        source += "struct OpKernelContext {\n" + constructors + fields + "};\n"
        source += r'''
int main() {
  static_assert(sizeof(OriginalContext) == sizeof(OpKernelContext));
  OpKernel kernel;
  OrtKernelContext native{16, 8};
  const char* setting = std::getenv("ORT_WEBGPU_ADAPTER_TENSOR_VECTOR_REUSE");
  const std::string mode = setting ? setting : "0";
  if (mode == "2") {
    try { OpKernelContext rejected{&native, kernel}; }
    catch (const std::runtime_error& error) {
      assert(std::string(error.what()) == "ORT_WEBGPU_ADAPTER_TENSOR_VECTOR_REUSE must be 0 or 1");
      return 0;
    }
    return 1;
  }
  { OpKernelContext warmup{&native, kernel}; }
  counting = true;
  for (int i = 0; i < 1000; ++i) {
    native = {static_cast<size_t>(1 + i % 16), static_cast<size_t>(1 + i % 8)};
    OpKernelContext context{&native, kernel};
    assert(context.input_tensors_.size() == native.inputs);
    assert(context.output_tensors_.size() == native.outputs);
    for (auto& tensor : context.input_tensors_) {
      assert(tensor.value == 0);
      tensor.value = 17;
    }
    for (auto& tensor : context.output_tensors_) {
      assert(tensor.value == 0);
      tensor.value = 91;
    }
  }
  counting = false;
  assert(allocations == (mode == "1" ? 0 : 2000));
  native = {3, 1};
  {
    OpKernelContext outer{&native, kernel};
    outer.input_tensors_[0].value = 73;
    const Tensor* pointer = &outer.input_tensors_[0];
    {
      OrtKernelContext inner_native{8, 2};
      OpKernelContext inner{&inner_native, kernel};
      assert(&inner.input_tensors_[0] != pointer);
      inner.input_tensors_[0].value = 29;
    }
    assert(pointer->value == 73);
  }
  try {
    OpKernelContext failing{&native, kernel};
    failing.input_tensors_[0].value = 55;
    throw std::runtime_error("expected compute failure");
  } catch (const std::runtime_error& error) {
    assert(std::string(error.what()) == "expected compute failure");
  }
  OpKernelContext after_error{&native, kernel};
  assert(after_error.input_tensors_[0].value == 0);
  std::cout << "PASS mode=" << mode << " allocations=" << allocations << "\n";
}
'''
        with tempfile.TemporaryDirectory(prefix="adapter-vector-test-") as directory:
            root = Path(directory)
            cpp = root / "test.cpp"
            cpp.write_text(source)
            executable = root / "test"
            subprocess.run(["c++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                            str(cpp), "-o", str(executable)], check=True)
            for mode in ("0", "1", "2"):
                stats = root / f"mode-{mode}.jsonl"
                environment = {**os.environ, FLAG: mode,
                               "ORT_WEBGPU_ADAPTER_TENSOR_VECTOR_STATS_FILE": str(stats)}
                subprocess.run([str(executable)], env=environment, check=True)
                if mode == "1":
                    rows = [json.loads(line) for line in stats.read_text().splitlines()]
                    self.assertEqual(len(rows), 1)
                    self.assertGreater(rows[0]["requests"], 1000)
                    self.assertEqual(rows[0]["requests"], rows[0]["returns"])
                    self.assertGreater(rows[0]["nonempty_capacity_reuses"], 1000)
                    self.assertEqual(rows[0]["retained_input_elements"], 0)
                    self.assertEqual(rows[0]["retained_output_elements"], 0)
                else:
                    self.assertFalse(stats.exists())
            environment = os.environ.copy()
            environment.pop(FLAG, None)
            environment.pop("ORT_WEBGPU_ADAPTER_TENSOR_VECTOR_STATS_FILE", None)
            subprocess.run([str(executable)], env=environment, check=True)


if __name__ == "__main__":
    unittest.main()
