import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from build_adapter_inline_tensors import FLAG, transform
from test_adapter_tensor_reuse import HOST_HEADER


BASE = Path("/home/sayanshaw/repos/onnxruntime-webgpu-fp8-checkpoint/build-webgpu-fp8/Release/_deps")


class AdapterInlineTensorTests(unittest.TestCase):
    header_transform = staticmethod(transform)
    input_capacity = 8
    output_capacity = 5

    def test_getters_preserve_constant_inputs_and_original_guard_behavior(self):
        original = HOST_HEADER.read_text()
        changed = self.header_transform(original)
        start = original.index("  template <typename T,", original.index("struct OpKernelContext {"))
        end = original.index("  [[nodiscard]] Status GetTempSpaceCPUAllocator", start)
        expected = original[start:end]
        actual_start = changed.index("  template <typename T,", changed.index("struct OpKernelContext {"))
        actual_end = changed.index("  [[nodiscard]] Status GetTempSpaceCPUAllocator", actual_start)
        actual = changed[actual_start:actual_end]
        actual = actual.replace("InputTensorAt(index)", "input_tensors_[index]")
        actual = actual.replace("OutputTensorAt(index)", "output_tensors_[index]")
        actual = actual.replace("InputTensorCount()", "input_tensors_.size()")
        actual = actual.replace("OutputTensorCount()", "output_tensors_.size()")
        self.assertEqual(actual, expected)
        self.assertIn("constant_input_tensors_[index]", changed)
        self.assertNotIn("thread_local", changed)
        self.assertIn('ParseEnvironmentVariableWithDefault<int>("' + FLAG + '", 0)', changed)

    def test_actual_constructor_uses_no_heap_within_inline_capacity(self):
        changed = self.header_transform(HOST_HEADER.read_text())
        start = changed.index("  explicit OpKernelContext(")
        end = changed.index("  template <typename T,", start)
        constructor = changed[start:end]
        helpers = changed[changed.index("struct AdapterInlineTensorSettings"):
                          changed.index("struct OpKernelContext {")]
        source = r'''
#include <atomic>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>
#include "core/common/inlined_containers_fwd.h"
using onnxruntime::InlinedVector;
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
template <typename T> T ParseEnvironmentVariableWithDefault(const char* name, T fallback) {
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
''' + helpers + "struct OpKernelContext {\n" + constructor + r'''
  Ort::KernelContext context_;
  const OpKernel& op_kernel_;
  const std::vector<Tensor>& constant_input_tensors_;
  mutable std::vector<Tensor> input_tensors_;
  std::vector<Tensor> output_tensors_;
''' + changed[changed.index("  bool inline_enabled_ = false;"):
              changed.index("\n};", changed.index("struct OpKernelContext {"))] + r'''
};
int main() {
  OpKernel kernel;
  OrtKernelContext native{8, 5};
  const char* value = std::getenv("ORT_WEBGPU_ADAPTER_INLINE_TENSORS");
  const int mode = value ? std::stoi(value) : 0;
  if (mode == 2) {
    try { OpKernelContext invalid{&native, kernel}; }
    catch (const std::runtime_error& error) {
      assert(std::string(error.what()) == "ORT_WEBGPU_ADAPTER_INLINE_TENSORS must be 0 or 1");
      return 0;
    }
    return 1;
  }
#if defined(DISABLE_ABSEIL)
  if (mode == 1) {
    try { OpKernelContext invalid{&native, kernel}; }
    catch (const std::runtime_error& error) {
      assert(std::string(error.what()) ==
             "ORT_WEBGPU_ADAPTER_INLINE_TENSORS requires Abseil inline-container support");
      return 0;
    }
    return 1;
  }
#endif
  { OpKernelContext warmup{&native, kernel}; }
  counting = true;
  for (int round = 0; round < 1000; ++round) {
    native = {static_cast<size_t>(1 + round % 8), static_cast<size_t>(1 + round % 5)};
    OpKernelContext context{&native, kernel};
    assert(context.InputTensorCount() == native.inputs);
    assert(context.OutputTensorCount() == native.outputs);
    for (size_t index = 0; index < native.inputs; ++index) {
      assert(context.InputTensorAt(index).value == 0);
      context.InputTensorAt(index).value = 17;
    }
    for (size_t index = 0; index < native.outputs; ++index) {
      assert(context.OutputTensorAt(index).value == 0);
      context.OutputTensorAt(index).value = 91;
    }
  }
  counting = false;
  assert(allocations == (mode == 1 ? 0 : 2000));
  std::cout << "PASS mode=" << mode << " allocations=" << allocations << "\n";
  native = {9, 6};
  allocations = 0;
  counting = true;
  { OpKernelContext fallback{&native, kernel};
    assert(fallback.InputTensorCount() == 9 && fallback.OutputTensorCount() == 6);
    fallback.InputTensorAt(8).value = 101;
    fallback.OutputTensorAt(5).value = 103;
  }
  counting = false;
  assert(allocations == 2);
  native = {3, 1};
  {
    OpKernelContext outer{&native, kernel};
    outer.InputTensorAt(0).value = 73;
    Tensor* pointer = &outer.InputTensorAt(0);
    {
      OrtKernelContext inner_native{8, 5};
      OpKernelContext inner{&inner_native, kernel};
      assert(&inner.InputTensorAt(0) != pointer);
      inner.InputTensorAt(0).value = 29;
    }
    assert(pointer->value == 73);
  }
  try {
    OpKernelContext failed{&native, kernel};
    failed.OutputTensorAt(0).value = 47;
    throw std::runtime_error("compute error");
  } catch (const std::runtime_error& error) {
    assert(std::string(error.what()) == "compute error");
  }
  OpKernelContext after_error{&native, kernel};
  assert(after_error.OutputTensorAt(0).value == 0);
  native = {0, 0};
  OpKernelContext empty{&native, kernel};
  assert(empty.InputTensorCount() == 0 && empty.OutputTensorCount() == 0);
}
'''
        for old, new in (
            ("OrtKernelContext native{8, 5};",
             f"OrtKernelContext native{{{self.input_capacity}, {self.output_capacity}}};"),
            ("round % 8", f"round % {self.input_capacity}"),
            ("round % 5", f"round % {self.output_capacity}"),
            ("native = {9, 6};", f"native = {{{self.input_capacity + 1}, {self.output_capacity + 1}}};"),
            ("fallback.InputTensorCount() == 9", f"fallback.InputTensorCount() == {self.input_capacity + 1}"),
            ("fallback.OutputTensorCount() == 6", f"fallback.OutputTensorCount() == {self.output_capacity + 1}"),
            ("fallback.InputTensorAt(8)", f"fallback.InputTensorAt({self.input_capacity})"),
            ("fallback.OutputTensorAt(5)", f"fallback.OutputTensorAt({self.output_capacity})"),
            ("OrtKernelContext inner_native{8, 5};",
             f"OrtKernelContext inner_native{{{self.input_capacity}, {self.output_capacity}}};"),
        ):
            self.assertEqual(source.count(old), 1, old)
            source = source.replace(old, new)
        with tempfile.TemporaryDirectory(prefix="adapter-inline-tensors-") as directory:
            path = Path(directory)
            cpp, binary = path / "probe.cpp", path / "probe"
            cpp.write_text(source)
            command = ["c++", "-O2", "-std=c++20", str(cpp),
                "-I/home/sayanshaw/repos/onnxruntime-webgpu-h200-perf/include/onnxruntime",
                "-I" + str(BASE / "abseil_cpp-src"),
                str(BASE / "abseil_cpp-build/absl/base/libabsl_throw_delegate.a"),
                str(BASE / "abseil_cpp-build/absl/base/libabsl_raw_logging_internal.a"),
                "-o", str(binary)]
            for disabled in (False, True):
                subprocess.run(command + (["-DDISABLE_ABSEIL"] if disabled else []), check=True)
                for mode in (None, "0", "1", "2"):
                    env = dict(os.environ)
                    env.pop(FLAG, None)
                    env.pop("ORT_WEBGPU_ADAPTER_INLINE_STATS_FILE", None)
                    if mode is not None:
                        env[FLAG] = mode
                    subprocess.run([str(binary)], env=env, check=True)


if __name__ == "__main__":
    unittest.main()
