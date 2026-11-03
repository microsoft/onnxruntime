from pathlib import Path
import subprocess
import tempfile
import unittest

from build_gpu_greedy_argmax import LAZY_SEARCH, interface_transform, model_transform


MOCK = r'''
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>
const char* device_label = "WebGPU";
struct DeviceBuffer {
  virtual ~DeviceBuffer() = default;
  virtual const char* GetType() const = 0;
  virtual void AllocateCpu() = 0;
  virtual void CopyDeviceToCpu() = 0;
  virtual void CopyCpuToDevice() = 0;
  virtual void CopyFrom(size_t, DeviceBuffer&, size_t, size_t) = 0;
  virtual void Zero() = 0;
  uint8_t* p_device_{};
  uint8_t* p_cpu_{};
  size_t size_in_bytes_{};
};
void CopyThroughCpu(DeviceBuffer& dest, size_t d, DeviceBuffer& source, size_t s, size_t n) {
  source.CopyDeviceToCpu();
  dest.AllocateCpu();
  std::memcpy(dest.p_cpu_ + d, source.p_cpu_ + s, n);
  dest.CopyCpuToDevice();
}
struct State {
  std::vector<float> gpu{1, 2};
  std::vector<float> cpu{0, 0};
  int downloads{}, uploads{};
};
template <typename T> struct DeviceSpan {
  std::shared_ptr<State> state;
  std::span<float> Span() { return state->gpu; }
  size_t size() { return state->gpu.size(); }
  std::span<float> CopyDeviceToCpu() { ++state->downloads; state->cpu = state->gpu; return state->cpu; }
  void CopyCpuToDevice() { ++state->uploads; state->gpu = state->cpu; }
  void Zero() { std::fill(state->gpu.begin(), state->gpu.end(), 0); }
};
'''
BODY = r'''
int main() {
  auto s = std::make_shared<State>();
  DeviceSpan<float> data{s};
  LazyWebGpuLogits lazy(data);
  if (lazy.CpuAccessed() || s->downloads) return 1;
  lazy.AllocateCpu();
  lazy.AllocateCpu();
  if (!lazy.CpuAccessed() || s->downloads != 1) return 2;
  reinterpret_cast<float*>(lazy.p_cpu_)[0] = -10;
  lazy.CopyDeviceToCpu();
  if (reinterpret_cast<float*>(lazy.p_cpu_)[0] != -10 || s->downloads != 1 || s->uploads) return 3;
  lazy.CopyCpuToDevice();
  if (s->gpu[0] != -10 || s->uploads != 1) return 4;
  lazy.Zero();
  if (lazy.CpuAccessed()) return 5;
  lazy.AllocateCpu();
  if (s->downloads != 2 || s->cpu[0] != 0) return 6;
  s->gpu[0] = 42;
  LazyWebGpuLogits next(data);
  next.AllocateCpu();
  if (s->downloads != 3 || s->cpu[0] != 42 || s->uploads != 1) return 7;
  return 0;
}
'''


class GpuGreedyTests(unittest.TestCase):
    def test_lazy_download_and_cpu_authority(self):
        with tempfile.TemporaryDirectory(prefix="webgpu-greedy-") as directory:
            source, binary = Path(directory) / "test.cpp", Path(directory) / "test"
            source.write_text(MOCK + LAZY_SEARCH[:LAZY_SEARCH.index("struct GreedySearch_WebGpu")] + BODY)
            subprocess.run(["c++", "-O3", "-std=c++20", str(source), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_model_default_input_placement_and_gpu_scoring_gate(self):
        source = Path("/home/sayanshaw/repos/onnxruntime-genai-webgpu-fp8-checkpoint/src/models/model.cpp").read_text()
        result = model_transform(source)
        original_inputs = source[source.index("  DeviceInterface* p_host_accessible_inputs"):
                                 source.index("  // Logits are read back")]
        self.assertIn(original_inputs, result)
        self.assertIn("p_device_logits_ = p_device_;", result)
        self.assertIn("p_device_scoring_ = p_device_;", result)
        self.assertIn("requires ORT_WEBGPU_ARGMAX_PARALLEL_BITWISE=1", result)

    def test_factory_default_and_all_cpu_access_surfaces_share_proxy(self):
        source = Path("/home/sayanshaw/repos/onnxruntime-genai-webgpu-fp8-checkpoint/src/ep/webgpu/interface.cpp").read_text()
        result = interface_transform(source)
        self.assertIn("return std::make_unique<GreedySearch_Cpu>(params);", result)
        self.assertIn("lazy_->CpuAccessed()", result)
        self.assertIn("CopyThroughCpu(*this", result)
        self.assertNotIn("virtual DeviceSpan", result)


if __name__ == "__main__":
    unittest.main()
