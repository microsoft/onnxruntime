// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <atomic>
#include <barrier>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include "core/graph/constants.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/autoep/test_autoep_utils.h"
#include "test/util/include/file_util.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

#if defined(USE_WEBGPU) && defined(ORT_USE_EP_API_ADAPTERS)

namespace {

class FirstError {
 public:
  void Set(std::string message) {
    bool expected = false;
    if (failed_.compare_exchange_strong(expected, true)) {
      std::lock_guard<std::mutex> lock{mutex_};
      message_ = std::move(message);
    }
  }

  bool Failed() const { return failed_.load(); }

  std::string Message() const {
    std::lock_guard<std::mutex> lock{mutex_};
    return message_;
  }

 private:
  std::atomic<bool> failed_{false};
  mutable std::mutex mutex_;
  std::string message_;
};

void ThrowOnError(OrtStatus* status_ptr) {
  Ort::Status status{status_ptr};
  if (!status.IsOK()) {
    throw std::runtime_error(status.GetErrorMessage());
  }
}

}  // namespace

TEST(PluginEpWebGpuConcurrency, MultiSessionRunAndSharedServices) {
  const Utils::ExamplePluginInfo webgpu_ep_info(
      GetSharedLibraryFileName(ORT_TSTR("onnxruntime_providers_webgpu")),
      "webgpu_ep_concurrency_library",
      kWebGpuExecutionProvider);
  RegisteredEpDeviceUniquePtr webgpu_ep_device_holder;
  ASSERT_NO_FATAL_FAILURE(
      Utils::RegisterAndGetExampleEp(*ort_env, webgpu_ep_info, webgpu_ep_device_holder));
  Ort::ConstEpDevice webgpu_ep_device{webgpu_ep_device_holder.get()};
  Ort::ConstMemoryInfo gpu_memory_info = webgpu_ep_device.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
  ASSERT_NE(gpu_memory_info, nullptr);
  auto gpu_allocator = ort_env->CreateSharedAllocator(
      webgpu_ep_device_holder.get(), OrtDeviceMemoryType_DEFAULT, OrtDeviceAllocator, nullptr);
  ASSERT_NE(gpu_allocator, nullptr);

  constexpr int kThreads = 4;
  constexpr int kIterations = 20;
  constexpr size_t kElements = 6;
  const std::array<int64_t, 2> shape{3, 2};
  FirstError error;
  std::barrier start{kThreads};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);

  for (int thread_id = 0; thread_id < kThreads; ++thread_id) {
    threads.emplace_back([&, thread_id]() {
      try {
        Ort::SessionOptions session_options;
        session_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
        std::unordered_map<std::string, std::string> ep_options;
        session_options.AppendExecutionProvider_V2(*ort_env, {webgpu_ep_device}, ep_options);
        Ort::Session session(*ort_env, ORT_TSTR("testdata/mul_1.onnx"), session_options);

        Ort::MemoryInfo cpu_memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        start.arrive_and_wait();

        for (int iteration = 0; iteration < kIterations && !error.Failed(); ++iteration) {
          const float value = static_cast<float>(thread_id * kIterations + iteration + 1);
          std::array<float, kElements> input_data{};
          input_data.fill(value);
          Ort::Value cpu_input = Ort::Value::CreateTensor<float>(
              cpu_memory_info, input_data.data(), input_data.size(), shape.data(), shape.size());

            Ort::Value gpu_input = Ort::Value::CreateTensor<float>(gpu_allocator, shape.data(), shape.size());
            Ort::Value gpu_output = Ort::Value::CreateTensor<float>(gpu_allocator, shape.data(), shape.size());
            ThrowOnError(ort_env->CopyTensor(cpu_input, gpu_input, nullptr));

            Ort::IoBinding io_binding(session);
            io_binding.BindInput("X", gpu_input);
            io_binding.BindOutput("Y", gpu_output);
            io_binding.SynchronizeInputs();
            session.Run(Ort::RunOptions{nullptr}, io_binding);
            io_binding.SynchronizeOutputs();

            std::array<float, kElements> output_data{};
            Ort::Value cpu_output = Ort::Value::CreateTensor<float>(
              cpu_memory_info, output_data.data(), output_data.size(), shape.data(), shape.size());
            ThrowOnError(ort_env->CopyTensor(gpu_output, cpu_output, nullptr));
          for (size_t index = 0; index < kElements; ++index) {
            const float expected = value * static_cast<float>(index + 1);
            if (output_data[index] != expected) {
              throw std::runtime_error(
                  "Concurrent WebGPU plugin Session::Run returned incorrect data at iteration " +
                  std::to_string(iteration) + ", index " + std::to_string(index) + ": actual=" +
                  std::to_string(output_data[index]) + ", expected=" + std::to_string(expected));
            }
          }
        }
      } catch (const std::exception& ex) {
        error.Set("thread " + std::to_string(thread_id) + ": " + ex.what());
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  ASSERT_FALSE(error.Failed()) << error.Message();
}

#endif  // defined(USE_WEBGPU) && defined(ORT_USE_EP_API_ADAPTERS)

}  // namespace test
}  // namespace onnxruntime