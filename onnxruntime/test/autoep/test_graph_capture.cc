// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Test graph capture/replay for the WebGPU plugin EP using only public C/C++ APIs.
// Requires a build with USE_WEBGPU and ORT_USE_EP_API_ADAPTERS (plugin EP path).

#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "core/session/onnxruntime_cxx_api.h"

#include "core/graph/constants.h"
#include "test/util/include/api_asserts.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

#if defined(USE_WEBGPU) && defined(ORT_USE_EP_API_ADAPTERS)

// Find the WebGPU EP device from the environment. Returns nullptr if not found.
static Ort::ConstEpDevice FindWebGpuEpDevice() {
  auto ep_devices = ort_env->GetEpDevices();
  for (const auto& device : ep_devices) {
    if (std::string_view(device.EpName()) == onnxruntime::kWebGpuExecutionProvider) {
      return device;
    }
  }
  return Ort::ConstEpDevice(nullptr);
}

// Test that the graph capture/replay path works end-to-end for the WebGPU plugin EP.
//
// Uses mul_1.onnx (Y = X * [1,2,3,4,5,6]) with IO Binding so that GPU memory addresses
// are stable across runs. The first Run() triggers warm-up + capture; the second Run()
// exercises the replay path with different input values.
TEST(PluginEpGraphCapture, WebGpuGraphCaptureAndReplay) {
  auto webgpu_device = FindWebGpuEpDevice();
  ASSERT_NE(webgpu_device, nullptr) << "Expected a valid OrtEpDevice for the webgpu plugin EP";

  // Create session with graph capture enabled
  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> ep_options;
  ep_options["enableGraphCapture"] = "1";
  session_options.AppendExecutionProvider_V2(*ort_env, {webgpu_device}, ep_options);

  Ort::Session session(*ort_env, ORT_TSTR("testdata/mul_1.onnx"), session_options);

  // Get GPU allocator from the session using the EP device's memory info
  Ort::ConstMemoryInfo gpu_mem_info = webgpu_device.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
  ASSERT_NE(gpu_mem_info, nullptr) << "Expected webgpu plugin EP's OrtEpDevice to return a valid OrtMemoryInfo";

  auto gpu_allocator = ort_env->GetSharedAllocator(gpu_mem_info);
  ASSERT_NE(gpu_allocator, nullptr) << "Expected webgpu plugin EP to have a shared allocator";

  // Allocate GPU tensors for input and output
  std::vector<int64_t> shape = {3, 2};
  Ort::Value gpu_input = Ort::Value::CreateTensor<float>(gpu_allocator, shape.data(), shape.size());
  Ort::Value gpu_output = Ort::Value::CreateTensor<float>(gpu_allocator, shape.data(), shape.size());

  // Helper to copy a single tensor using the C API (the C++ wrapper takes vector<Value>
  // which is inconvenient since Value is move-only).
  auto copy_tensor = [](const Ort::Value& src, Ort::Value& dst) -> Ort::Status {
    const OrtValue* src_ptr = src;
    OrtValue* dst_ptr = dst;
    return ort_env->CopyTensors(&src_ptr, &dst_ptr, nullptr, 1);
  };

  Ort::MemoryInfo cpu_mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

  // --- First run: warm-up + capture ---
  // Copy input data (CPU → GPU).
  // Input X = [1, 2, 3, 4, 5, 6]. mul_1.onnx computes Y = X * [1, 2, 3, 4, 5, 6].
  // Expected output: [1, 4, 9, 16, 25, 36].
  {
    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Ort::Value cpu_input = Ort::Value::CreateTensor<float>(
        cpu_mem_info, input_data.data(), input_data.size(), shape.data(), shape.size());
    ASSERT_ORTSTATUS_OK(copy_tensor(cpu_input, gpu_input));
  }

  // Bind inputs and outputs
  Ort::IoBinding io_binding(session);
  io_binding.BindInput("X", gpu_input);
  io_binding.BindOutput("Y", gpu_output);
  io_binding.SynchronizeInputs();

  Ort::RunOptions run_options;
  session.Run(run_options, io_binding);
  io_binding.SynchronizeOutputs();

  // Copy GPU output to CPU and verify
  {
    Ort::AllocatorWithDefaultOptions cpu_allocator;
    Ort::Value cpu_output = Ort::Value::CreateTensor<float>(cpu_allocator, shape.data(), shape.size());
    ASSERT_ORTSTATUS_OK(copy_tensor(gpu_output, cpu_output));

    const float* output_data = cpu_output.GetTensorData<float>();
    std::vector<float> expected = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_FLOAT_EQ(output_data[i], expected[i]) << "First run mismatch at index " << i;
    }
  }

  // --- Second run: replay with different input values ---
  // Copy new input data into the same GPU buffer (CPU → GPU).
  // Input X = [2, 3, 4, 5, 6, 7]. Expected output: [4, 9, 16, 25, 36, 49].
  {
    std::vector<float> input_data_2 = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    Ort::Value cpu_input_2 = Ort::Value::CreateTensor<float>(
        cpu_mem_info, input_data_2.data(), input_data_2.size(), shape.data(), shape.size());
    ASSERT_ORTSTATUS_OK(copy_tensor(cpu_input_2, gpu_input));
  }

  session.Run(run_options, io_binding);
  io_binding.SynchronizeOutputs();

  // Copy GPU output to CPU and verify
  {
    Ort::AllocatorWithDefaultOptions cpu_allocator;
    Ort::Value cpu_output = Ort::Value::CreateTensor<float>(cpu_allocator, shape.data(), shape.size());
    ASSERT_ORTSTATUS_OK(copy_tensor(gpu_output, cpu_output));

    const float* output_data = cpu_output.GetTensorData<float>();
    std::vector<float> expected = {4.0f, 9.0f, 16.0f, 25.0f, 36.0f, 49.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_FLOAT_EQ(output_data[i], expected[i]) << "Replay run mismatch at index " << i;
    }
  }
}
#endif  // defined(USE_WEBGPU) && defined(ORT_USE_EP_API_ADAPTERS)

}  // namespace test
}  // namespace onnxruntime
