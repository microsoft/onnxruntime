// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include "core/graph/constants.h"
#include "core/graph/onnx_protobuf.h"
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

void ORT_API_CALL CountGraphReplays(void* param, OrtLoggingLevel severity, const char*,
                                    const char*, const char*, const char* message) noexcept {
  if (severity == ORT_LOGGING_LEVEL_INFO && message != nullptr) {
    const std::string_view text{message};
    if (text.starts_with("Replaying the captured ") && text.find(kWebGpuExecutionProvider) != std::string_view::npos) {
      static_cast<std::atomic<int>*>(param)->fetch_add(1, std::memory_order_relaxed);
    }
  }
}

constexpr int kThreads = 4;
constexpr int kIterations = 20;
constexpr size_t kElements = 6;
constexpr std::array<int64_t, 2> kShape{3, 2};

void VerifyOutput(const std::array<float, kElements>& output_data, float value) {
  for (size_t index = 0; index < kElements; ++index) {
    const float expected = value * static_cast<float>(index + 1);
    if (output_data[index] != expected) {
      throw std::runtime_error("Incorrect output at index " + std::to_string(index) +
                               ": actual=" + std::to_string(output_data[index]) +
                               ", expected=" + std::to_string(expected));
    }
  }
}

template <typename Work>
void RunWorkers(FirstError& error, Work work, int num_threads = kThreads) {
  std::barrier start{num_threads};
  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
    threads.emplace_back([&, thread_id]() {
      try {
        start.arrive_and_wait();
        work(thread_id);
      } catch (const std::exception& ex) {
        error.Set("thread " + std::to_string(thread_id) + ": " + ex.what());
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

template <typename Allocator>
void CopyTensorRoundTrip(Allocator& allocator, float value) {
  const auto cpu_memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::array<float, kElements> input_data{};
  input_data.fill(value);
  std::array<float, kElements> output_data{};
  auto cpu_input = Ort::Value::CreateTensor<float>(
      cpu_memory_info, input_data.data(), input_data.size(), kShape.data(), kShape.size());
  auto gpu_tensor = Ort::Value::CreateTensor<float>(allocator, kShape.data(), kShape.size());
  auto gpu_copy = Ort::Value::CreateTensor<float>(allocator, kShape.data(), kShape.size());
  auto cpu_output = Ort::Value::CreateTensor<float>(
      cpu_memory_info, output_data.data(), output_data.size(), kShape.data(), kShape.size());

  ThrowOnError(ort_env->CopyTensor(cpu_input, gpu_tensor, nullptr));
  ThrowOnError(ort_env->CopyTensor(gpu_tensor, gpu_copy, nullptr));
  ThrowOnError(ort_env->CopyTensor(gpu_copy, cpu_output, nullptr));
  if (output_data != input_data) {
    throw std::runtime_error("CopyTensor round trip returned incorrect data");
  }
}

}  // namespace

class PluginEpWebGpuConcurrency : public ::testing::Test {
 protected:
  void SetUp() override {
    webgpu_ep_info_ = std::make_unique<Utils::ExamplePluginInfo>(
        GetSharedLibraryFileName(ORT_TSTR("onnxruntime_providers_webgpu")),
        "webgpu_ep_concurrency_library",
        kWebGpuExecutionProvider);
    ASSERT_NO_FATAL_FAILURE(
        Utils::RegisterAndGetExampleEp(*ort_env, *webgpu_ep_info_, webgpu_ep_device_holder_));
    ASSERT_NE(Device().GetMemoryInfo(OrtDeviceMemoryType_DEFAULT), nullptr);
  }

  Ort::ConstEpDevice Device() const {
    return Ort::ConstEpDevice{webgpu_ep_device_holder_.get()};
  }

  Ort::UnownedAllocator CreateSharedAllocator() const {
    return ort_env->CreateSharedAllocator(
        webgpu_ep_device_holder_.get(), OrtDeviceMemoryType_DEFAULT, OrtDeviceAllocator, nullptr);
  }

  std::unique_ptr<Ort::Session> CreateSession() const {
    Ort::SessionOptions session_options;
    session_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
    std::unordered_map<std::string, std::string> ep_options;
    session_options.AppendExecutionProvider_V2(*ort_env, {Device()}, ep_options);
    return std::make_unique<Ort::Session>(
        *ort_env, ORT_TSTR("testdata/mul_1.onnx"), session_options);
  }

  void RunAndVerify(Ort::Session& session, float value) const {
    const auto cpu_memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    std::array<float, kElements> input_data{};
    input_data.fill(value);
    std::array<float, kElements> output_data{};
    auto cpu_input = Ort::Value::CreateTensor<float>(
        cpu_memory_info, input_data.data(), input_data.size(), kShape.data(), kShape.size());
    auto cpu_output = Ort::Value::CreateTensor<float>(
        cpu_memory_info, output_data.data(), output_data.size(), kShape.data(), kShape.size());

    Ort::Allocator allocator(session, Device().GetMemoryInfo(OrtDeviceMemoryType_DEFAULT));
    auto gpu_input = Ort::Value::CreateTensor<float>(allocator, kShape.data(), kShape.size());
    auto gpu_output = Ort::Value::CreateTensor<float>(allocator, kShape.data(), kShape.size());
    ThrowOnError(ort_env->CopyTensor(cpu_input, gpu_input, nullptr));

    Ort::IoBinding io_binding(session);
    io_binding.BindInput("X", gpu_input);
    io_binding.BindOutput("Y", gpu_output);
    io_binding.SynchronizeInputs();
    session.Run(Ort::RunOptions{nullptr}, io_binding);
    io_binding.SynchronizeOutputs();

    ThrowOnError(ort_env->CopyTensor(gpu_output, cpu_output, nullptr));
    VerifyOutput(output_data, value);
  }

  void RunWithCpuInputAndOutput(Ort::Session& session, float value) const {
    const auto cpu_memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    std::array<float, kElements> input_data{};
    input_data.fill(value);
    auto cpu_input = Ort::Value::CreateTensor<float>(
        cpu_memory_info, input_data.data(), input_data.size(), kShape.data(), kShape.size());
    const std::array<const char*, 1> input_names{"X"};
    const std::array<const char*, 1> output_names{"Y"};
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &cpu_input, 1,
                               output_names.data(), output_names.size());
    const float* output = outputs.front().GetTensorData<float>();
    std::array<float, kElements> output_data{};
    std::copy_n(output, output_data.size(), output_data.begin());
    VerifyOutput(output_data, value);
  }

  void RunWithCpuInputAndGpuOutput(Ort::Session& session, float value) const {
    const auto cpu_memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    std::array<float, kElements> input_data{};
    input_data.fill(value);
    auto cpu_input = Ort::Value::CreateTensor<float>(
        cpu_memory_info, input_data.data(), input_data.size(), kShape.data(), kShape.size());

    Ort::Allocator allocator(session, Device().GetMemoryInfo(OrtDeviceMemoryType_DEFAULT));
    auto gpu_output = Ort::Value::CreateTensor<float>(allocator, kShape.data(), kShape.size());
    Ort::IoBinding io_binding(session);
    io_binding.BindInput("X", cpu_input);
    io_binding.BindOutput("Y", gpu_output);
    session.Run(Ort::RunOptions{nullptr}, io_binding);

    std::array<float, kElements> output_data{};
    auto cpu_output = Ort::Value::CreateTensor<float>(
        cpu_memory_info, output_data.data(), output_data.size(), kShape.data(), kShape.size());
    ThrowOnError(ort_env->CopyTensor(gpu_output, cpu_output, nullptr));
    VerifyOutput(output_data, value);
  }

  void RunWithGpuInputAndCpuOutput(Ort::Session& session, float value) const {
    const auto cpu_memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    std::array<float, kElements> input_data{};
    input_data.fill(value);
    auto cpu_input = Ort::Value::CreateTensor<float>(
        cpu_memory_info, input_data.data(), input_data.size(), kShape.data(), kShape.size());
    Ort::Allocator allocator(session, Device().GetMemoryInfo(OrtDeviceMemoryType_DEFAULT));
    auto gpu_input = Ort::Value::CreateTensor<float>(allocator, kShape.data(), kShape.size());
    ThrowOnError(ort_env->CopyTensor(cpu_input, gpu_input, nullptr));

    const std::array<const char*, 1> input_names{"X"};
    const std::array<const char*, 1> output_names{"Y"};
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &gpu_input, 1,
                               output_names.data(), output_names.size());
    const float* output = outputs.front().GetTensorData<float>();
    std::array<float, kElements> output_data{};
    std::copy_n(output, output_data.size(), output_data.begin());
    VerifyOutput(output_data, value);
  }

  void RunMixedSessionOperations(bool include_session_allocator) const {
    constexpr int kMixedIterations = 10;
    std::array<std::unique_ptr<Ort::Session>, kThreads> run_sessions;
    for (auto& session : run_sessions) {
      session = CreateSession();
    }

    const auto gpu_memory_info = Device().GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
    auto shared_allocator = CreateSharedAllocator();
    ASSERT_NE(shared_allocator, nullptr);

    FirstError error;
    const int num_threads = (include_session_allocator ? 4 : 3) * kThreads;
    const auto work = [&](int thread_id) {
      for (int iteration = 0; iteration < kMixedIterations && !error.Failed(); ++iteration) {
        const float value = static_cast<float>(thread_id * kMixedIterations + iteration + 1);
        if (thread_id < kThreads) {
          auto session = CreateSession();
        } else if (thread_id < 2 * kThreads) {
          RunWithCpuInputAndOutput(*run_sessions[thread_id - kThreads], value);
        } else if (thread_id < 3 * kThreads) {
          CopyTensorRoundTrip(shared_allocator, value);
        } else {
          Ort::Allocator allocator(*run_sessions[thread_id - 3 * kThreads], gpu_memory_info);
          CopyTensorRoundTrip(allocator, value);
        }
      }
    };
    RunWorkers(error, work, num_threads);

    ASSERT_FALSE(error.Failed()) << error.Message();
  }

 private:
  std::unique_ptr<Utils::ExamplePluginInfo> webgpu_ep_info_;
  RegisteredEpDeviceUniquePtr webgpu_ep_device_holder_;
};

TEST_F(PluginEpWebGpuConcurrency, EnvironmentCopiesBeforeAndAfterSerialSessionCreation) {
  auto allocator = CreateSharedAllocator();
  CopyTensorRoundTrip(allocator, 1.0f);
  std::array<std::unique_ptr<Ort::Session>, kThreads> sessions;
  for (auto& session : sessions) {
    session = CreateSession();
    CopyTensorRoundTrip(allocator, 2.0f);
    RunWithCpuInputAndOutput(*session, 3.0f);
  }
  for (auto& session : sessions) {
    session.reset();
  }
  CopyTensorRoundTrip(allocator, 4.0f);
  auto session = CreateSession();
  RunWithCpuInputAndOutput(*session, 5.0f);
}

TEST_F(PluginEpWebGpuConcurrency, DifferentSessionsCreateConcurrently) {
  std::array<std::unique_ptr<Ort::Session>, kThreads> sessions;
  FirstError error;
  RunWorkers(error, [&](int thread_id) {
    sessions[thread_id] = CreateSession();
  });

  ASSERT_FALSE(error.Failed()) << error.Message();
  for (int session_index = 0; session_index < kThreads; ++session_index) {
    ASSERT_NE(sessions[session_index], nullptr);
    RunWithCpuInputAndOutput(*sessions[session_index], static_cast<float>(session_index + 1));
  }

  RunWorkers(error, [&](int thread_id) {
    auto& session = *sessions[(thread_id + 1) % kThreads];
    for (int iteration = 0; iteration < kIterations && !error.Failed(); ++iteration) {
      RunWithCpuInputAndOutput(session, static_cast<float>(thread_id * kIterations + iteration + 1));
    }
  });
  ASSERT_FALSE(error.Failed()) << error.Message();
}

TEST_F(PluginEpWebGpuConcurrency, DifferentSessionsCreateAndRunConcurrently) {
  std::array<std::unique_ptr<Ort::Session>, kThreads> run_sessions;
  for (auto& session : run_sessions) {
    session = CreateSession();
  }

  FirstError error;
  std::barrier iteration_start{2 * kThreads};
  const auto work = [&](int thread_id) {
    try {
      for (int iteration = 0; iteration < kIterations; ++iteration) {
        iteration_start.arrive_and_wait();
        const float value = static_cast<float>(thread_id * kIterations + iteration + 1);
        if (thread_id < kThreads) {
          auto session = CreateSession();
          RunWithCpuInputAndOutput(*session, value);
        } else {
          RunWithCpuInputAndOutput(*run_sessions[thread_id - kThreads], value);
        }
      }
    } catch (...) {
      iteration_start.arrive_and_drop();
      throw;
    }
  };
  RunWorkers(error, work, 2 * kThreads);

  ASSERT_FALSE(error.Failed()) << error.Message();
}

// Future support: workers also create GPU tensors and call Env copies concurrently.
TEST_F(PluginEpWebGpuConcurrency, DISABLED_DifferentSessionsRunConcurrently) {
  std::array<std::unique_ptr<Ort::Session>, kThreads> sessions;
  for (auto& session : sessions) {
    session = CreateSession();
  }

  FirstError error;
  RunWorkers(error, [&](int thread_id) {
    for (int iteration = 0; iteration < kIterations && !error.Failed(); ++iteration) {
      const float value = static_cast<float>(thread_id * kIterations + iteration + 1);
      RunAndVerify(*sessions[thread_id], value);
    }
  });

  ASSERT_FALSE(error.Failed()) << error.Message();
}

// Future support: GPU tensors are allocated serially, but Env uploads and downloads overlap.
TEST_F(PluginEpWebGpuConcurrency, DISABLED_DifferentSessionsGraphCaptureAndReplayConcurrently) {
  std::array<std::atomic<int>, kThreads> replay_counts{};
  std::array<std::unique_ptr<Ort::Session>, kThreads> sessions;
  auto allocator = CreateSharedAllocator();
  ASSERT_NE(allocator, nullptr);
  std::vector<Ort::Value> gpu_inputs;
  std::vector<Ort::Value> gpu_outputs;
  std::vector<Ort::IoBinding> bindings;
  gpu_inputs.reserve(kThreads);
  gpu_outputs.reserve(kThreads);
  bindings.reserve(kThreads);

  for (int session_index = 0; session_index < kThreads; ++session_index) {
    Ort::SessionOptions options;
    options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
    options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_INFO);
    ThrowOnError(Ort::GetApi().SetUserLoggingFunction(options, CountGraphReplays, &replay_counts[session_index]));
    const std::unordered_map<std::string, std::string> ep_options{{"enableGraphCapture", "1"}};
    options.AppendExecutionProvider_V2(*ort_env, {Device()}, ep_options);
    sessions[session_index] = std::make_unique<Ort::Session>(
        *ort_env, ORT_TSTR("testdata/mul_1.onnx"), options);
    gpu_inputs.push_back(Ort::Value::CreateTensor<float>(allocator, kShape.data(), kShape.size()));
    gpu_outputs.push_back(Ort::Value::CreateTensor<float>(allocator, kShape.data(), kShape.size()));
    bindings.emplace_back(*sessions[session_index]);
    bindings.back().BindInput("X", gpu_inputs.back());
    bindings.back().BindOutput("Y", gpu_outputs.back());
  }

  FirstError error;
  std::barrier run_start{kThreads};
  RunWorkers(error, [&](int thread_id) {
    try {
      const auto cpu_memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
      std::array<float, kElements> input_data{};
      std::array<float, kElements> output_data{};
      auto cpu_input = Ort::Value::CreateTensor<float>(
          cpu_memory, input_data.data(), input_data.size(), kShape.data(), kShape.size());
      auto cpu_output = Ort::Value::CreateTensor<float>(
          cpu_memory, output_data.data(), output_data.size(), kShape.data(), kShape.size());
      Ort::RunOptions run_options;
      auto& binding = bindings[thread_id];
      for (int iteration = 0; iteration < kIterations; ++iteration) {
        const float value = static_cast<float>(thread_id * kIterations + iteration + 1);
        input_data.fill(value);
        output_data.fill(-1.0f);
        ThrowOnError(ort_env->CopyTensor(cpu_input, gpu_inputs[thread_id], nullptr));
        binding.SynchronizeInputs();
        run_start.arrive_and_wait();
        sessions[thread_id]->Run(run_options, binding);
        binding.SynchronizeOutputs();
        ThrowOnError(ort_env->CopyTensor(gpu_outputs[thread_id], cpu_output, nullptr));
        VerifyOutput(output_data, value);
      }
    } catch (...) {
      run_start.arrive_and_drop();
      throw;
    }
  });

  ASSERT_FALSE(error.Failed()) << error.Message();
  for (int session_index = 0; session_index < kThreads; ++session_index) {
    EXPECT_GE(replay_counts[session_index].load(), kIterations - 1)
        << "Session " << session_index << " did not execute the graph replay path";
  }
}

TEST_F(PluginEpWebGpuConcurrency, CpuInputAndOutputRun) {
  auto session = CreateSession();
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    RunWithCpuInputAndOutput(*session, static_cast<float>(iteration + 1));
  }
}

TEST_F(PluginEpWebGpuConcurrency, CpuPartitionBetweenGpuKernels) {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  model.add_opset_import()->set_version(18);
  auto* graph = model.mutable_graph();
  graph->set_name("webgpu_session_partition_copy");
  for (auto* value_info : {graph->add_input(), graph->add_output()}) {
    value_info->set_name(value_info == &graph->input(0) ? "X" : "Y");
    auto* tensor_type = value_info->mutable_type()->mutable_tensor_type();
    tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    for (auto dimension : kShape) {
      tensor_type->mutable_shape()->add_dim()->set_dim_value(dimension);
    }
  }
  const std::array<const char*, 4> values{"X", "gpu_value", "cpu_value", "Y"};
  const std::array<const char*, 3> nodes{"gpu_first", "cpu_middle", "gpu_last"};
  for (size_t index = 0; index < nodes.size(); ++index) {
    auto* node = graph->add_node();
    node->set_name(nodes[index]);
    node->set_op_type("Neg");
    node->add_input(values[index]);
    node->add_output(values[index + 1]);
  }
  auto* cpu_output_info = graph->add_output();
  cpu_output_info->CopyFrom(graph->output(0));
  cpu_output_info->set_name("cpu_value");
  const auto model_bytes = model.SerializeAsString();
  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
  const std::unordered_map<std::string, std::string> ep_options{{"forceCpuNodeNames", "cpu_middle"}};
  options.AppendExecutionProvider_V2(*ort_env, {Device()}, ep_options);
  Ort::Session session(*ort_env, model_bytes.data(), model_bytes.size(), options);
  const auto input_devices = session.GetEpDeviceForInputs();
  const auto output_devices = session.GetEpDeviceForOutputs();
  ASSERT_EQ(input_devices.size(), 1u);
  ASSERT_EQ(output_devices.size(), 2u);
  ASSERT_NE(input_devices.front(), nullptr);
  ASSERT_NE(output_devices.front(), nullptr);
  ASSERT_STREQ(input_devices.front().EpName(), kWebGpuExecutionProvider);
  ASSERT_STREQ(output_devices.front().EpName(), kWebGpuExecutionProvider);
  ASSERT_NE(output_devices[1], nullptr);
  ASSERT_STREQ(output_devices[1].EpName(), kCpuExecutionProvider);
  const auto cpu_memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  const std::array<const char*, 1> inputs{"X"};
  const std::array<const char*, 2> outputs{"Y", "cpu_value"};
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    SCOPED_TRACE(iteration);
    std::array<float, kElements> data{};
    for (size_t index = 0; index < data.size(); ++index) {
      data[index] = static_cast<float>((iteration + 1) * (index + 1));
    }
    auto input = Ort::Value::CreateTensor<float>(cpu_memory, data.data(), data.size(), kShape.data(), kShape.size());
    auto result = session.Run(Ort::RunOptions{nullptr}, inputs.data(), &input, 1, outputs.data(), outputs.size());
    ASSERT_EQ(result.size(), 2u);
    const auto* actual = result.front().GetTensorData<float>();
    const auto* cpu_actual = result[1].GetTensorData<float>();
    for (size_t index = 0; index < data.size(); ++index) {
      EXPECT_EQ(cpu_actual[index], data[index]) << "CPU intermediate at index " << index;
      EXPECT_EQ(actual[index], -data[index]) << "GPU output at index " << index;
    }
  }
}

TEST_F(PluginEpWebGpuConcurrency, CpuInputAndGpuOutputRun) {
  auto session = CreateSession();
  RunWithCpuInputAndGpuOutput(*session, 1.0f);
}

TEST_F(PluginEpWebGpuConcurrency, GpuInputAndCpuOutputRun) {
  auto session = CreateSession();
  RunWithGpuInputAndCpuOutput(*session, 1.0f);
}

TEST_F(PluginEpWebGpuConcurrency, DifferentSessionsWithCpuInputAndOutputRunConcurrently) {
  std::array<std::unique_ptr<Ort::Session>, kThreads> sessions;
  for (auto& session : sessions) {
    session = CreateSession();
  }

  FirstError error;
  RunWorkers(error, [&](int thread_id) {
    for (int iteration = 0; iteration < kIterations && !error.Failed(); ++iteration) {
      const float value = static_cast<float>(thread_id * kIterations + iteration + 1);
      RunWithCpuInputAndOutput(*sessions[thread_id], value);
    }
  });

  ASSERT_FALSE(error.Failed()) << error.Message();
}

// Future support: external Session allocator operations and Env copies are concurrent.
TEST_F(PluginEpWebGpuConcurrency, DISABLED_SessionAllocatorsCreateAndCopyConcurrently) {
  std::array<std::unique_ptr<Ort::Session>, kThreads> sessions;
  for (auto& session : sessions) {
    session = CreateSession();
  }
  const auto gpu_memory_info = Device().GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);

  FirstError error;
  RunWorkers(error, [&](int thread_id) {
    Ort::Allocator allocator(*sessions[thread_id], gpu_memory_info);
    for (int iteration = 0; iteration < kIterations && !error.Failed(); ++iteration) {
      const float value = static_cast<float>(thread_id * kIterations + iteration + 1);
      CopyTensorRoundTrip(allocator, value);
    }
  });

  ASSERT_FALSE(error.Failed()) << error.Message();
}

TEST_F(PluginEpWebGpuConcurrency, DISABLED_SharedAllocatorCreatesAndCopiesConcurrently) {
  auto allocator = CreateSharedAllocator();
  ASSERT_NE(allocator, nullptr);

  FirstError error;
  RunWorkers(error, [&](int thread_id) {
    for (int iteration = 0; iteration < kIterations && !error.Failed(); ++iteration) {
      const float value = static_cast<float>(thread_id * kIterations + iteration + 1);
      CopyTensorRoundTrip(allocator, value);
    }
  });

  ASSERT_FALSE(error.Failed()) << error.Message();
}

TEST_F(PluginEpWebGpuConcurrency, EnvironmentCopiesBatchedEmptyAndUnalignedTensors) {
  constexpr std::array<int64_t, 7> byte_counts{0, 1, 3, 4, 5, 16, 17};
  constexpr size_t kCapacity = 20;
  constexpr uint8_t kSentinel = 0xcc;
  auto allocator = CreateSharedAllocator();
  ASSERT_NE(allocator, nullptr);
  const auto cpu_memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::array<std::array<uint8_t, kCapacity>, byte_counts.size()> input_data{};
  std::array<std::array<uint8_t, kCapacity>, byte_counts.size()> output_data{};
  std::vector<Ort::Value> cpu_inputs;
  std::vector<Ort::Value> gpu_inputs;
  std::vector<Ort::Value> gpu_outputs;
  std::vector<Ort::Value> cpu_outputs;
  cpu_inputs.reserve(byte_counts.size());
  gpu_inputs.reserve(byte_counts.size());
  gpu_outputs.reserve(byte_counts.size());
  cpu_outputs.reserve(byte_counts.size());
  for (size_t tensor_index = 0; tensor_index < byte_counts.size(); ++tensor_index) {
    const auto byte_count = static_cast<size_t>(byte_counts[tensor_index]);
    for (size_t byte_index = 0; byte_index < kCapacity; ++byte_index) {
      input_data[tensor_index][byte_index] = static_cast<uint8_t>(tensor_index * kCapacity + byte_index + 1);
    }
    output_data[tensor_index].fill(kSentinel);
    cpu_inputs.push_back(Ort::Value::CreateTensor<uint8_t>(
        cpu_memory, input_data[tensor_index].data(), byte_count, &byte_counts[tensor_index], 1));
    gpu_inputs.push_back(Ort::Value::CreateTensor<uint8_t>(allocator, &byte_counts[tensor_index], 1));
    gpu_outputs.push_back(Ort::Value::CreateTensor<uint8_t>(allocator, &byte_counts[tensor_index], 1));
    cpu_outputs.push_back(Ort::Value::CreateTensor<uint8_t>(
        cpu_memory, output_data[tensor_index].data(), byte_count, &byte_counts[tensor_index], 1));
  }

  ThrowOnError(ort_env->CopyTensors(cpu_inputs, gpu_inputs, nullptr));
  ThrowOnError(ort_env->CopyTensors(gpu_inputs, gpu_outputs, nullptr));
  ThrowOnError(ort_env->CopyTensors(gpu_outputs, cpu_outputs, nullptr));
  for (size_t tensor_index = 0; tensor_index < byte_counts.size(); ++tensor_index) {
    SCOPED_TRACE("byte count " + std::to_string(byte_counts[tensor_index]));
    for (size_t byte_index = 0; byte_index < kCapacity; ++byte_index) {
      const auto expected = byte_index < static_cast<size_t>(byte_counts[tensor_index])
                                ? input_data[tensor_index][byte_index]
                                : kSentinel;
      EXPECT_EQ(output_data[tensor_index][byte_index], expected) << "byte " << byte_index;
    }
  }
}

TEST_F(PluginEpWebGpuConcurrency, SharedGpuCopyCompletesBeforeSessionRun) {
  auto session = CreateSession();
  auto allocator = CreateSharedAllocator();
  const auto cpu_memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  const std::array<const char*, 1> input_names{"X"};
  const std::array<const char*, 1> output_names{"Y"};
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    const float value = static_cast<float>(iteration + 1);
    std::array<float, kElements> input_data{};
    input_data.fill(value);
    auto input = Ort::Value::CreateTensor<float>(cpu_memory, input_data.data(), input_data.size(),
                                                 kShape.data(), kShape.size());
    auto source = Ort::Value::CreateTensor<float>(allocator, kShape.data(), kShape.size());
    auto destination = Ort::Value::CreateTensor<float>(allocator, kShape.data(), kShape.size());
    ThrowOnError(ort_env->CopyTensor(input, source, nullptr));
    ThrowOnError(ort_env->CopyTensor(source, destination, nullptr));
    auto result = session->Run(Ort::RunOptions{nullptr}, input_names.data(), &destination, 1,
                               output_names.data(), output_names.size());
    std::array<float, kElements> output{};
    std::copy_n(result.front().GetTensorData<float>(), output.size(), output.begin());
    VerifyOutput(output, value);
  }
}

TEST_F(PluginEpWebGpuConcurrency, DISABLED_DifferentSessionsAndEnvironmentCopiesRunConcurrently) {
  std::array<std::unique_ptr<Ort::Session>, kThreads> run_sessions;
  for (auto& session : run_sessions) {
    session = CreateSession();
  }

  auto shared_allocator = CreateSharedAllocator();
  ASSERT_NE(shared_allocator, nullptr);

  FirstError error;
  RunWorkers(error, [&](int thread_id) {
               for (int iteration = 0; iteration < kIterations && !error.Failed(); ++iteration) {
                 const float value = static_cast<float>(thread_id * kIterations + iteration + 1);
                 if (thread_id < kThreads) {
                   RunWithCpuInputAndOutput(*run_sessions[thread_id], value);
                 } else {
                   CopyTensorRoundTrip(shared_allocator, value);
                 }
               } }, 2 * kThreads);

  ASSERT_FALSE(error.Failed()) << error.Message();
}

// Next-phase target: shared Env allocator operations and copies concurrent with Session creation and Run.
// Keep disabled until shared-allocator concurrency is supported.
TEST_F(PluginEpWebGpuConcurrency, DISABLED_MixedSessionAndEnvironmentOperationsConcurrently12Threads) {
  RunMixedSessionOperations(false);
}

// Next-phase target: also exercise a Session's allocator concurrently with that same Session's Run.
// Keep disabled until both Session and shared-allocator concurrency are supported.
TEST_F(PluginEpWebGpuConcurrency, DISABLED_MixedSessionAndAllocatorOperationsConcurrently16Threads) {
  RunMixedSessionOperations(true);
}

TEST_F(PluginEpWebGpuConcurrency, SessionAllocatorClearsReusedBufferOutsideRun) {
  auto session = CreateSession();
  Ort::Allocator allocator(*session, Device().GetMemoryInfo(OrtDeviceMemoryType_DEFAULT));
  const auto cpu_memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::array<float, kElements> input_data{};
  input_data.fill(7.0f);
  auto cpu_input = Ort::Value::CreateTensor<float>(
      cpu_memory, input_data.data(), input_data.size(), kShape.data(), kShape.size());
  const void* released_buffer = nullptr;
  {
    auto gpu_tensor = Ort::Value::CreateTensor<float>(allocator, kShape.data(), kShape.size());
    released_buffer = gpu_tensor.GetTensorRawData();
    ThrowOnError(ort_env->CopyTensor(cpu_input, gpu_tensor, nullptr));
  }

  auto reused_tensor = Ort::Value::CreateTensor<float>(allocator, kShape.data(), kShape.size());
  ASSERT_EQ(reused_tensor.GetTensorRawData(), released_buffer);
  std::array<float, kElements> output_data{};
  output_data.fill(-1.0f);
  auto cpu_output = Ort::Value::CreateTensor<float>(
      cpu_memory, output_data.data(), output_data.size(), kShape.data(), kShape.size());
  ThrowOnError(ort_env->CopyTensor(reused_tensor, cpu_output, nullptr));
  const std::array<float, kElements> zeros{};
  EXPECT_EQ(output_data, zeros);

  ThrowOnError(ort_env->CopyTensor(cpu_input, reused_tensor, nullptr));
  ThrowOnError(ort_env->CopyTensor(reused_tensor, cpu_output, nullptr));
  EXPECT_EQ(output_data, input_data);
}

#endif  // defined(USE_WEBGPU) && defined(ORT_USE_EP_API_ADAPTERS)

}  // namespace test
}  // namespace onnxruntime
