// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "core/common/inlined_containers.h"
#include "core/graph/constants.h"
#include "core/graph/onnx_protobuf.h"
#include "core/platform/env.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_run_options_config_keys.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/webgpu/plugin/webgpu_plugin_test_utils.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

#if defined(ORT_UNIT_TEST_HAS_WEBGPU_PLUGIN_EP)

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
void RunWorkers(FirstError& error, Work work) {
  std::barrier start{kThreads};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int thread_id = 0; thread_id < kThreads; ++thread_id) {
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
void CopyTensorRoundTrip(Allocator& allocator, float value, bool verify_zero_initialization = false) {
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

  if (verify_zero_initialization) {
    output_data.fill(-1.0f);
    Ort::ThrowOnError(ort_env->CopyTensor(gpu_tensor, cpu_output, nullptr));
    if (output_data != std::array<float, kElements>{}) {
      throw std::runtime_error("Streamless allocation was not zero initialized");
    }
  }

  Ort::ThrowOnError(ort_env->CopyTensor(cpu_input, gpu_tensor, nullptr));
  Ort::ThrowOnError(ort_env->CopyTensor(gpu_tensor, gpu_copy, nullptr));
  Ort::ThrowOnError(ort_env->CopyTensor(gpu_copy, cpu_output, nullptr));
  if (output_data != input_data) {
    throw std::runtime_error("CopyTensor round trip returned incorrect data");
  }
}

template <typename Allocator>
Ort::Value CreateReusedNonzeroGpuTensor(Allocator& allocator, float value) {
  const auto cpu_memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::array<float, kElements> dirty_data{};
  dirty_data.fill(value);
  std::array<float, kElements> readback_data{};
  auto dirty_source = Ort::Value::CreateTensor<float>(
      cpu_memory, dirty_data.data(), dirty_data.size(), kShape.data(), kShape.size());
  auto readback = Ort::Value::CreateTensor<float>(
      cpu_memory, readback_data.data(), readback_data.size(), kShape.data(), kShape.size());
  void* dirty_buffer = nullptr;
  {
    auto dirty_tensor = Ort::Value::CreateTensor<float>(allocator, kShape.data(), kShape.size());
    dirty_buffer = dirty_tensor.GetTensorMutableRawData();
    Ort::ThrowOnError(ort_env->CopyTensor(dirty_source, dirty_tensor, nullptr));
    Ort::ThrowOnError(ort_env->CopyTensor(dirty_tensor, readback, nullptr));
    if (value == 0.0f || readback_data != dirty_data) {
      throw std::runtime_error("Expected a nonzero GPU buffer before releasing it for reuse");
    }
  }

  auto reused_tensor = Ort::Value::CreateTensor<float>(allocator, kShape.data(), kShape.size());
  if (reused_tensor.GetTensorMutableRawData() != dirty_buffer) {
    throw std::runtime_error("Expected the allocator to reuse the released nonzero GPU buffer");
  }
  return reused_tensor;
}

}  // namespace

class PluginEpWebGpuConcurrency : public ::testing::Test {
 protected:
  void SetUp() override {
    registration_.emplace(*ort_env, "webgpu_ep_concurrency_library");
    const auto devices = registration_->GetEpDevices();
    ASSERT_FALSE(devices.empty());
    webgpu_ep_device_ = devices.front();
    ASSERT_NE(Device().GetMemoryInfo(OrtDeviceMemoryType_DEFAULT), nullptr);
  }

  Ort::ConstEpDevice Device() const {
    return webgpu_ep_device_;
  }

  Ort::UnownedAllocator CreateSharedAllocator() const {
    return ort_env->CreateSharedAllocator(
        Device(), OrtDeviceMemoryType_DEFAULT, OrtDeviceAllocator, nullptr);
  }

  std::unique_ptr<Ort::Session> CreateSession(bool enable_graph_capture = false) const {
    Ort::SessionOptions session_options;
    session_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
    std::unordered_map<std::string, std::string> ep_options;
    if (enable_graph_capture) {
      ep_options["enableGraphCapture"] = "1";
    }
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
    Ort::ThrowOnError(ort_env->CopyTensor(cpu_input, gpu_input, nullptr));

    Ort::IoBinding io_binding(session);
    io_binding.BindInput("X", gpu_input);
    io_binding.BindOutput("Y", gpu_output);
    io_binding.SynchronizeInputs();
    session.Run(Ort::RunOptions{nullptr}, io_binding);
    io_binding.SynchronizeOutputs();

    Ort::ThrowOnError(ort_env->CopyTensor(gpu_output, cpu_output, nullptr));
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
    Ort::ThrowOnError(ort_env->CopyTensor(gpu_output, cpu_output, nullptr));
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
    Ort::ThrowOnError(ort_env->CopyTensor(cpu_input, gpu_input, nullptr));

    const std::array<const char*, 1> input_names{"X"};
    const std::array<const char*, 1> output_names{"Y"};
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &gpu_input, 1,
                               output_names.data(), output_names.size());
    const float* output = outputs.front().GetTensorData<float>();
    std::array<float, kElements> output_data{};
    std::copy_n(output, output_data.size(), output_data.begin());
    VerifyOutput(output_data, value);
  }

  void RunWithCpuBoundInputAndDirtyGpuReuse(Ort::Session& session, float value) const {
    const auto cpu_memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    std::array<float, kElements> input_data{};
    input_data.fill(value);
    std::array<float, kElements> dirty_data{};
    dirty_data.fill(-value);
    std::array<float, kElements> output_data{};
    auto cpu_input = Ort::Value::CreateTensor<float>(
        cpu_memory, input_data.data(), input_data.size(), kShape.data(), kShape.size());
    auto dirty_source = Ort::Value::CreateTensor<float>(
        cpu_memory, dirty_data.data(), dirty_data.size(), kShape.data(), kShape.size());
    auto cpu_output = Ort::Value::CreateTensor<float>(
        cpu_memory, output_data.data(), output_data.size(), kShape.data(), kShape.size());
    Ort::Allocator allocator(session, Device().GetMemoryInfo(OrtDeviceMemoryType_DEFAULT));
    auto gpu_output = Ort::Value::CreateTensor<float>(allocator, kShape.data(), kShape.size());
    Ort::IoBinding binding(session);
    binding.BindOutput("Y", gpu_output);

    // Release a dirty same-size buffer so BindInput's internal allocation can reuse it.
    // The user input remains on CPU; the GPU scratch tensor only seeds the allocator cache.
    {
      auto scratch = Ort::Value::CreateTensor<float>(allocator, kShape.data(), kShape.size());
      Ort::ThrowOnError(ort_env->CopyTensor(dirty_source, scratch, nullptr));
    }

    binding.BindInput("X", cpu_input);
    // No intervening allocation, copy, or synchronization may flush the deferred clear.
    session.Run(Ort::RunOptions{nullptr}, binding);
    Ort::ThrowOnError(ort_env->CopyTensor(gpu_output, cpu_output, nullptr));
    VerifyOutput(output_data, value);
  }

  enum class OutputBinding { None,
                             CpuToGpu,
                             CpuOnlyGraphToGpu };

  void RunWithCpuPartitionFeeds(bool use_gpu_feed, bool reverse_feeds = false,
                                OutputBinding output_binding = OutputBinding::None,
                                bool preallocate_gpu_input = false) const {
    const bool bind_cpu_output_to_gpu = output_binding != OutputBinding::None;
    const bool cpu_only_graph = output_binding == OutputBinding::CpuOnlyGraphToGpu;
    ONNX_NAMESPACE::ModelProto model;
    model.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
    model.add_opset_import()->set_version(18);
    auto* graph = model.mutable_graph();
    graph->set_name("webgpu_mixed_feed_copies");
    const std::array<const char*, 2> input_names{"gpu_node_input", "cpu_node_input"};
    const std::array<const char*, 2> output_names{"gpu_node_output", "cpu_node_output"};
    const std::array<const char*, 2> node_names{"gpu_neg", "cpu_neg"};
    for (size_t index = 0; index < input_names.size(); ++index) {
      auto* input = graph->add_input();
      input->set_name(input_names[index]);
      auto* output = graph->add_output();
      output->set_name(output_names[index]);
      for (auto* value_info : {input, output}) {
        auto* tensor_type = value_info->mutable_type()->mutable_tensor_type();
        tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
        for (auto dimension : kShape) {
          tensor_type->mutable_shape()->add_dim()->set_dim_value(dimension);
        }
      }
      auto* node = graph->add_node();
      node->set_name(node_names[index]);
      node->set_op_type("Neg");
      node->add_input(input_names[index]);
      node->add_output(output_names[index]);
    }

    const auto model_bytes = model.SerializeAsString();
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    const std::unordered_map<std::string, std::string> ep_options{
        {"forceCpuNodeNames", cpu_only_graph ? "gpu_neg\ncpu_neg" : node_names[1]}};
    options.AppendExecutionProvider_V2(*ort_env, {Device()}, ep_options);
    Ort::Session session(*ort_env, model_bytes.data(), model_bytes.size(), options);
    const auto input_memory = session.GetMemoryInfoForInputs();
    const auto output_memory = session.GetMemoryInfoForOutputs();
    ASSERT_EQ(input_memory.size(), 2u);
    ASSERT_EQ(output_memory.size(), 2u);
    const auto first_node_device = cpu_only_graph ? OrtMemoryInfoDeviceType_CPU : OrtMemoryInfoDeviceType_GPU;
    ASSERT_EQ(input_memory[0].GetDeviceType(), first_node_device);
    ASSERT_EQ(input_memory[1].GetDeviceType(), OrtMemoryInfoDeviceType_CPU);
    ASSERT_EQ(output_memory[0].GetDeviceType(), first_node_device);
    ASSERT_EQ(output_memory[1].GetDeviceType(), OrtMemoryInfoDeviceType_CPU);
    const auto input_devices = session.GetEpDeviceForInputs();
    ASSERT_EQ(input_devices.size(), 2u);
    if (!cpu_only_graph) {
      ASSERT_NE(input_devices[0], nullptr);
      ASSERT_STREQ(input_devices[0].EpName(), kWebGpuExecutionProvider);
    }

    const auto cpu_memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Allocator allocator(session, Device().GetMemoryInfo(OrtDeviceMemoryType_DEFAULT));
    std::array<float, kElements> gpu_node_data{};
    std::array<float, kElements> cpu_node_data{};
    auto gpu_node_source = Ort::Value::CreateTensor<float>(
        cpu_memory, gpu_node_data.data(), gpu_node_data.size(), kShape.data(), kShape.size());
    auto gpu_node_feed = preallocate_gpu_input
                             ? Ort::Value::CreateTensor<float>(allocator, kShape.data(), kShape.size())
                             : Ort::Value::CreateTensor<float>(
                                   cpu_memory, gpu_node_data.data(), gpu_node_data.size(), kShape.data(), kShape.size());
    auto cpu_node_source = Ort::Value::CreateTensor<float>(
        cpu_memory, cpu_node_data.data(), cpu_node_data.size(), kShape.data(), kShape.size());
    auto cpu_node_feed = use_gpu_feed
                             ? Ort::Value::CreateTensor<float>(allocator, kShape.data(), kShape.size())
                             : Ort::Value::CreateTensor<float>(
                                   cpu_memory, cpu_node_data.data(), cpu_node_data.size(), kShape.data(), kShape.size());
    std::array<Ort::Value, 2> feeds{std::move(gpu_node_feed), std::move(cpu_node_feed)};
    auto feed_names = input_names;
    if (reverse_feeds) {
      std::swap(feeds[0], feeds[1]);
      std::swap(feed_names[0], feed_names[1]);
    }
    Ort::IoBinding binding(session);

    for (int iteration = 0; iteration < kIterations; ++iteration) {
      SCOPED_TRACE(iteration);
      for (size_t index = 0; index < kElements; ++index) {
        gpu_node_data[index] = static_cast<float>((iteration + 1) * (index + 1));
        cpu_node_data[index] = static_cast<float>(100 + iteration * kElements + index);
      }
      if (use_gpu_feed) {
        Ort::ThrowOnError(ort_env->CopyTensor(cpu_node_source, feeds[reverse_feeds ? 0 : 1], nullptr));
      }
      if (preallocate_gpu_input) {
        Ort::ThrowOnError(ort_env->CopyTensor(gpu_node_source, feeds[reverse_feeds ? 1 : 0], nullptr));
      }
      if (bind_cpu_output_to_gpu) {
        for (size_t index = 0; index < feeds.size(); ++index) {
          binding.BindInput(feed_names[index], feeds[index]);
        }
        for (size_t index = 0; index < output_names.size(); ++index) {
          const size_t output_index = reverse_feeds ? output_names.size() - 1 - index : index;
          binding.BindOutput(output_names[output_index],
                             output_index == 0 ? Ort::ConstMemoryInfo{cpu_memory}
                                               : Device().GetMemoryInfo(OrtDeviceMemoryType_DEFAULT));
        }
      }

      // Seed feed/output allocations before Run, without overlapping external allocation and Run.
      {
        std::array<Ort::Value, 2> cached_tensors{
            Ort::Value::CreateTensor<float>(allocator, kShape.data(), kShape.size()),
            Ort::Value::CreateTensor<float>(allocator, kShape.data(), kShape.size())};
        for (auto& cached_tensor : cached_tensors) {
          Ort::ThrowOnError(ort_env->CopyTensor(feeds[reverse_feeds ? 1 : 0], cached_tensor, nullptr));
        }
      }

      std::vector<Ort::Value> outputs;
      std::array<float, kElements> cpu_output_data{};
      if (bind_cpu_output_to_gpu) {
        session.Run(Ort::RunOptions{nullptr}, binding);
        outputs = binding.GetOutputValues();
        ASSERT_EQ(outputs.size(), 2u);
        if (reverse_feeds) {
          std::swap(outputs[0], outputs[1]);
        }
        ASSERT_EQ(outputs[1].GetTensorMemoryInfo().GetDeviceType(), OrtMemoryInfoDeviceType_GPU);
        auto cpu_output = Ort::Value::CreateTensor<float>(
            cpu_memory, cpu_output_data.data(), cpu_output_data.size(), kShape.data(), kShape.size());
        Ort::ThrowOnError(ort_env->CopyTensor(outputs[1], cpu_output, nullptr));
        outputs[1] = std::move(cpu_output);
        binding.ClearBoundOutputs();
      } else {
        // Both feed copies must be prepared in the same Run, rather than separately by BindInput.
        outputs = session.Run(Ort::RunOptions{nullptr}, feed_names.data(), feeds.data(), feeds.size(),
                              output_names.data(), output_names.size());
      }
      ASSERT_EQ(outputs.size(), 2u);
      for (size_t index = 0; index < kElements; ++index) {
        ASSERT_EQ(outputs[0].GetTensorData<float>()[index], -gpu_node_data[index]) << "GPU branch, element " << index;
        ASSERT_EQ(outputs[1].GetTensorData<float>()[index], -cpu_node_data[index]) << "CPU branch, element " << index;
      }
    }
  }

  void RunDeferredProducerThenCopy(const char* copy_op) const {
    ONNX_NAMESPACE::ModelProto model;
    model.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
    model.add_opset_import()->set_version(18);
    auto* graph = model.mutable_graph();
    graph->set_name("webgpu_deferred_producer_copy");
    const std::array<int64_t, 1> shape{6};
    const auto add_value = [&](const char* name, bool input) {
      auto* value = input ? graph->add_input() : graph->add_output();
      value->set_name(name);
      auto* type = value->mutable_type()->mutable_tensor_type();
      type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
      type->mutable_shape()->add_dim()->set_dim_value(shape[0]);
    };
    add_value("X", true);
    add_value("P", false);
    add_value("Y", false);
    auto* producer = graph->add_node();
    producer->set_name("producer");
    producer->set_op_type("Neg");
    producer->add_input("X");
    producer->add_output("P");
    auto* copy = graph->add_node();
    copy->set_name("copy");
    copy->set_op_type(copy_op);
    copy->add_input("P");
    copy->add_output("Y");
    if (std::string_view{copy_op} == "Reshape" || std::string_view{copy_op} == "ReduceSum") {
      auto* parameter = graph->add_initializer();
      parameter->set_name("parameter");
      parameter->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
      parameter->add_dims(std::string_view{copy_op} == "Reshape" ? 1 : 0);
      if (std::string_view{copy_op} == "Reshape") {
        parameter->add_int64_data(shape[0]);
      } else {
        auto* attr = copy->add_attribute();
        attr->set_name("noop_with_empty_axes");
        attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
        attr->set_i(1);
      }
      copy->add_input("parameter");
    }
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
    options.EnableProfiling(ORT_TSTR("webgpu_copy_regression"));
    Ort::KeyValuePairs ep_options;
    ep_options.Add("maxNumPendingDispatches", "4096");
    options.AppendExecutionProvider_V2(*ort_env, {Device()}, ep_options);
    const auto bytes = model.SerializeAsString();
    Ort::Session session(*ort_env, bytes.data(), bytes.size(), options);
    for (const auto& device : session.GetEpDeviceForOutputs()) {
      ASSERT_NE(device, nullptr);
      ASSERT_STREQ(device.EpName(), kWebGpuExecutionProvider);
    }

    const auto cpu_memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    std::array<float, kElements> input_data{1, -2, 3, -4, 5, -6};
    auto input = Ort::Value::CreateTensor<float>(cpu_memory, input_data.data(),
                                                 input_data.size(), shape.data(), shape.size());
    Ort::Allocator allocator(session, Device().GetMemoryInfo(OrtDeviceMemoryType_DEFAULT));
    auto gpu_input = Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
    auto gpu_producer = Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
    auto gpu_copy = Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
    ASSERT_NE(gpu_producer.GetTensorMutableData<float>(), gpu_copy.GetTensorMutableData<float>());
    Ort::ThrowOnError(ort_env->CopyTensor(input, gpu_input, nullptr));
    Ort::IoBinding binding(session);
    binding.BindInput("X", gpu_input);
    // Preallocated, distinct graph outputs force the no-op kernel's actual copy branch,
    // rather than its Alias(0, 0) shortcut.
    binding.BindOutput("P", gpu_producer);
    binding.BindOutput("Y", gpu_copy);
    session.Run(Ort::RunOptions{nullptr}, binding);
    auto bound_outputs = binding.GetOutputValues();
    ASSERT_EQ(bound_outputs.size(), 2u);
    EXPECT_EQ(bound_outputs[0].GetTensorMutableData<float>(), gpu_producer.GetTensorMutableData<float>());
    EXPECT_EQ(bound_outputs[1].GetTensorMutableData<float>(), gpu_copy.GetTensorMutableData<float>());
    std::array<float, kElements> output_data{};
    auto output = Ort::Value::CreateTensor<float>(cpu_memory, output_data.data(), output_data.size(),
                                                  shape.data(), shape.size());
    Ort::ThrowOnError(ort_env->CopyTensor(gpu_copy, output, nullptr));
    for (size_t i = 0; i < input_data.size(); ++i) {
      EXPECT_EQ(output_data[i], -input_data[i]);
    }

    Ort::AllocatorWithDefaultOptions cpu_allocator;
    const auto profile_path = session.EndProfilingAllocated(cpu_allocator);
    nlohmann::json profile;
    {
      std::ifstream file(profile_path.get());
      ASSERT_TRUE(file.is_open());
      file >> profile;
    }
    std::filesystem::remove(profile_path.get());
    size_t copy_kernels = 0;
    size_t producer_kernels = 0;
    for (const auto& event : profile) {
      if (event.value("cat", "") != "Node" || !event.contains("args")) {
        continue;
      }
      const auto& args = event["args"];
      if (args.value("op_name", "") == copy_op || args.value("op_name", "") == "Neg") {
        EXPECT_EQ(args.value("provider", ""), kWebGpuExecutionProvider);
        copy_kernels += args.value("op_name", "") == copy_op;
        producer_kernels += args.value("op_name", "") == "Neg";
      }
    }
    EXPECT_EQ(producer_kernels, 1u);
    EXPECT_EQ(copy_kernels, 1u);
  }

  void RunGraphsConcurrently(bool separate_sessions, bool capture_before_workers) const {
    InlinedVector<std::unique_ptr<Ort::Session>> sessions;
    const int session_count = separate_sessions ? kThreads : 1;
    for (int i = 0; i < session_count; ++i) {
      sessions.emplace_back(CreateSession(true));
    }
    const auto cpu_memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    InlinedVector<Ort::Allocator> allocators;
    for (const auto& session : sessions) {
      allocators.emplace_back(*session, Device().GetMemoryInfo(OrtDeviceMemoryType_DEFAULT));
    }
    InlinedVector<Ort::Value> inputs;
    InlinedVector<Ort::Value> outputs;
    InlinedVector<Ort::IoBinding> bindings;
    InlinedVector<Ort::RunOptions> options;
    for (int thread_id = 0; thread_id < kThreads; ++thread_id) {
      auto& session = *sessions[separate_sessions ? thread_id : 0];
      auto& allocator = allocators[separate_sessions ? thread_id : 0];
      inputs.emplace_back(Ort::Value::CreateTensor<float>(allocator, kShape.data(), kShape.size()));
      outputs.emplace_back(Ort::Value::CreateTensor<float>(allocator, kShape.data(), kShape.size()));
      std::array<float, kElements> input_data{};
      input_data.fill(static_cast<float>(thread_id + 1));
      auto cpu_input = Ort::Value::CreateTensor<float>(
          cpu_memory, input_data.data(), input_data.size(), kShape.data(), kShape.size());
      Ort::ThrowOnError(ort_env->CopyTensor(cpu_input, inputs.back(), nullptr));
      bindings.emplace_back(session);
      bindings.back().BindInput("X", inputs.back());
      bindings.back().BindOutput("Y", outputs.back());
      options.emplace_back();
      options.back().AddConfigEntry("gpu_graph_id", std::to_string(thread_id).c_str());
      if (capture_before_workers) {
        session.Run(options.back(), bindings.back());
      }
    }

    FirstError error;
    RunWorkers(error, [&](int thread_id) {
      auto& session = *sessions[separate_sessions ? thread_id : 0];
      for (int iteration = 0; iteration < kIterations && !error.Failed(); ++iteration) {
        // Without pre-capture, the first call includes warm-up and capture retries.
        // Later calls replay using the same bound buffers and graph annotation ID.
        session.Run(options[thread_id], bindings[thread_id]);
        std::array<float, kElements> output_data{};
        auto cpu_output = Ort::Value::CreateTensor<float>(
            cpu_memory, output_data.data(), output_data.size(), kShape.data(), kShape.size());
        Ort::ThrowOnError(ort_env->CopyTensor(outputs[thread_id], cpu_output, nullptr));
        VerifyOutput(output_data, static_cast<float>(thread_id + 1));
      }
    });
    ASSERT_FALSE(error.Failed()) << error.Message();
  }

 private:
  std::optional<ScopedWebGpuPluginRegistration> registration_;
  Ort::ConstEpDevice webgpu_ep_device_{nullptr};
};

TEST_F(PluginEpWebGpuConcurrency, DeferredProducerThenIdentityCopy) {
  RunDeferredProducerThenCopy("Identity");
}

TEST_F(PluginEpWebGpuConcurrency, DeferredProducerThenReshapeCopy) {
  RunDeferredProducerThenCopy("Reshape");
}

TEST_F(PluginEpWebGpuConcurrency, DeferredProducerThenOneDimensionalTransposeCopy) {
  RunDeferredProducerThenCopy("Transpose");
}

TEST_F(PluginEpWebGpuConcurrency, DeferredProducerThenNoOpReductionCopy) {
  RunDeferredProducerThenCopy("ReduceSum");
}

TEST_F(PluginEpWebGpuConcurrency, DifferentSessionsCreateConcurrently) {
  std::array<std::unique_ptr<Ort::Session>, kThreads> sessions;
  FirstError error;
  RunWorkers(error, [&](int thread_id) {
    sessions[thread_id] = CreateSession();
  });

  ASSERT_FALSE(error.Failed()) << error.Message();
  for (const auto& session : sessions) {
    ASSERT_NE(session, nullptr);
  }
}

TEST_F(PluginEpWebGpuConcurrency, DifferentSessionsRunConcurrently) {
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

TEST_F(PluginEpWebGpuConcurrency, CpuInputAndOutputRun) {
  auto session = CreateSession();
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    RunWithCpuInputAndOutput(*session, static_cast<float>(iteration + 1));
  }
}

TEST_F(PluginEpWebGpuConcurrency, MixedCpuAndGpuFeeds) {
  RunWithCpuPartitionFeeds(true);
}

TEST_F(PluginEpWebGpuConcurrency, MixedCpuAndGpuFeedsReversed) {
  RunWithCpuPartitionFeeds(true, true);
}

TEST_F(PluginEpWebGpuConcurrency, CpuFeedsWithCpuPartition) {
  RunWithCpuPartitionFeeds(false);
}

TEST_F(PluginEpWebGpuConcurrency, CpuOutputBoundToGpu) {
  RunWithCpuPartitionFeeds(false, false, OutputBinding::CpuToGpu);
}

TEST_F(PluginEpWebGpuConcurrency, CpuOutputBoundToGpuFirst) {
  RunWithCpuPartitionFeeds(false, true, OutputBinding::CpuToGpu);
}

TEST_F(PluginEpWebGpuConcurrency, PreallocatedGpuInputWithMixedOutputBindings) {
  // Bypass BindInput's CPU-to-GPU copy to isolate output-copy ordering.
  RunWithCpuPartitionFeeds(false, false, OutputBinding::CpuToGpu, true);
}

TEST_F(PluginEpWebGpuConcurrency, PreallocatedGpuInputWithMixedOutputBindingsReversed) {
  RunWithCpuPartitionFeeds(false, true, OutputBinding::CpuToGpu, true);
}

TEST_F(PluginEpWebGpuConcurrency, CpuOnlyGraphOutputBoundToGpu) {
  RunWithCpuPartitionFeeds(false, true, OutputBinding::CpuOnlyGraphToGpu);
}

TEST_F(PluginEpWebGpuConcurrency, RepeatedKernelScratchBufferReuse) {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  model.add_opset_import()->set_version(18);
  auto* graph = model.mutable_graph();
  graph->set_name("webgpu_scratch_buffer_reuse");
  auto* input = graph->add_input();
  input->set_name("X");
  auto* output = graph->add_output();
  output->set_name("Y");
  for (auto* value_info : {input, output}) {
    auto* type = value_info->mutable_type()->mutable_tensor_type();
    type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    for (auto dimension : kShape) {
      type->mutable_shape()->add_dim()->set_dim_value(dimension);
    }
  }
  // Four-input Min allocates intermediate GPU tensors through CreateGPUTensor.
  for (int index = 0; index < 2; ++index) {
    auto* node = graph->add_node();
    node->set_name(index == 0 ? "first_min" : "second_min");
    node->set_op_type("Min");
    node->add_input(index == 0 ? "X" : "intermediate");
    for (int input_index = 0; input_index < 3; ++input_index) {
      node->add_input("X");
    }
    node->add_output(index == 0 ? "intermediate" : "Y");
  }
  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
  options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
  options.AppendExecutionProvider_V2(*ort_env, {Device()}, std::unordered_map<std::string, std::string>{});
  const auto model_bytes = model.SerializeAsString();
  Ort::Session session(*ort_env, model_bytes.data(), model_bytes.size(), options);
  const auto cpu_memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  const std::array<const char*, 1> input_names{"X"};
  const std::array<const char*, 1> output_names{"Y"};
  std::array<float, kElements> input_data{};
  auto input_tensor = Ort::Value::CreateTensor<float>(
      cpu_memory, input_data.data(), input_data.size(), kShape.data(), kShape.size());
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    SCOPED_TRACE(iteration);
    input_data.fill(static_cast<float>(iteration + 1));
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1,
                               output_names.data(), output_names.size());
    ASSERT_EQ(outputs.size(), 1u);
    for (size_t index = 0; index < input_data.size(); ++index) {
      ASSERT_EQ(outputs[0].GetTensorData<float>()[index], input_data[index]);
    }
  }
}

TEST_F(PluginEpWebGpuConcurrency, CpuPartitionBetweenGpuKernels) {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  model.add_opset_import()->set_version(18);
  auto* graph = model.mutable_graph();
  graph->set_name("webgpu_stream_partition_copy");
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
  const auto model_bytes = model.SerializeAsString();
  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
  const std::unordered_map<std::string, std::string> ep_options{{"forceCpuNodeNames", "cpu_middle"}};
  options.AppendExecutionProvider_V2(*ort_env, {Device()}, ep_options);
  Ort::Session session(*ort_env, model_bytes.data(), model_bytes.size(), options);
  const auto input_devices = session.GetEpDeviceForInputs();
  const auto output_devices = session.GetEpDeviceForOutputs();
  ASSERT_EQ(input_devices.size(), 1u);
  ASSERT_EQ(output_devices.size(), 1u);
  ASSERT_NE(input_devices.front(), nullptr);
  ASSERT_NE(output_devices.front(), nullptr);
  EXPECT_STREQ(input_devices.front().EpName(), kWebGpuExecutionProvider);
  EXPECT_STREQ(output_devices.front().EpName(), kWebGpuExecutionProvider);
  const auto cpu_memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  const std::array<const char*, 1> inputs{"X"};
  const std::array<const char*, 1> outputs{"Y"};
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    std::array<float, kElements> data{};
    data.fill(static_cast<float>(iteration + 1));
    auto input = Ort::Value::CreateTensor<float>(cpu_memory, data.data(), data.size(), kShape.data(), kShape.size());
    auto result = session.Run(Ort::RunOptions{nullptr}, inputs.data(), &input, 1, outputs.data(), outputs.size());
    const auto* actual = result.front().GetTensorData<float>();
    for (size_t index = 0; index < data.size(); ++index) {
      EXPECT_EQ(actual[index], -data[index]);
    }
  }
}

TEST_F(PluginEpWebGpuConcurrency, CpuInputAndGpuOutputRun) {
  auto session = CreateSession();
  RunWithCpuInputAndGpuOutput(*session, 1.0f);
}

TEST_F(PluginEpWebGpuConcurrency, CpuBindInputReusesDirtyGpuBuffer) {
  auto session = CreateSession();
  const auto input_memory = session->GetMemoryInfoForInputs();
  const auto output_memory = session->GetMemoryInfoForOutputs();
  ASSERT_EQ(input_memory.size(), 1u);
  ASSERT_EQ(output_memory.size(), 1u);
  ASSERT_EQ(input_memory.front().GetDeviceType(), OrtMemoryInfoDeviceType_GPU);
  ASSERT_EQ(output_memory.front().GetDeviceType(), OrtMemoryInfoDeviceType_GPU);
  const auto input_devices = session->GetEpDeviceForInputs();
  ASSERT_EQ(input_devices.size(), 1u);
  ASSERT_NE(input_devices.front(), nullptr);
  ASSERT_STREQ(input_devices.front().EpName(), kWebGpuExecutionProvider);

  for (int iteration = 0; iteration < kIterations; ++iteration) {
    SCOPED_TRACE(iteration);
    RunWithCpuBoundInputAndDirtyGpuReuse(*session, static_cast<float>(iteration + 1));
  }
}

TEST_F(PluginEpWebGpuConcurrency, SerialInterleavedCpuBindInputsAcrossSessions) {
  std::array<std::unique_ptr<Ort::Session>, 2> sessions{CreateSession(), CreateSession()};
  const auto gpu_memory = Device().GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
  const auto cpu_memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::array<Ort::Allocator, 2> allocators{
      Ort::Allocator(*sessions[0], gpu_memory), Ort::Allocator(*sessions[1], gpu_memory)};
  std::array<Ort::Value, 2> gpu_outputs{
      CreateReusedNonzeroGpuTensor(allocators[0], -101.0f),
      CreateReusedNonzeroGpuTensor(allocators[1], -202.0f)};
  std::array<std::array<float, kElements>, 2> input_data{};
  std::array<Ort::Value, 2> cpu_inputs{
      Ort::Value::CreateTensor<float>(
          cpu_memory, input_data[0].data(), kElements, kShape.data(), kShape.size()),
      Ort::Value::CreateTensor<float>(
          cpu_memory, input_data[1].data(), kElements, kShape.data(), kShape.size())};
  std::array<Ort::IoBinding, 2> bindings{Ort::IoBinding(*sessions[0]), Ort::IoBinding(*sessions[1])};
  for (size_t index = 0; index < sessions.size(); ++index) {
    bindings[index].BindOutput("Y", gpu_outputs[index]);
  }
  std::array<float, kElements> output_data{};
  auto cpu_output = Ort::Value::CreateTensor<float>(
      cpu_memory, output_data.data(), output_data.size(), kShape.data(), kShape.size());

  for (int iteration = 0; iteration < kIterations; ++iteration) {
    SCOPED_TRACE(iteration);
    const std::array<float, 2> values{static_cast<float>(iteration + 1),
                                      -static_cast<float>(100 + iteration)};
    for (size_t index = 0; index < sessions.size(); ++index) {
      bindings[index].ClearBoundInputs();
      input_data[index].fill(-values[index]);
      {
        auto scratch = Ort::Value::CreateTensor<float>(allocators[index], kShape.data(), kShape.size());
        Ort::ThrowOnError(ort_env->CopyTensor(cpu_inputs[index], scratch, nullptr));
      }
      input_data[index].fill(values[index]);
    }

    // Both CPU uploads precede either Run, with no intervening copy or synchronization.
    bindings[0].BindInput("X", cpu_inputs[0]);
    bindings[1].BindInput("X", cpu_inputs[1]);
    const size_t first = static_cast<size_t>(iteration % 2);
    sessions[first]->Run(Ort::RunOptions{nullptr}, bindings[first]);
    sessions[1 - first]->Run(Ort::RunOptions{nullptr}, bindings[1 - first]);

    for (size_t index = 0; index < sessions.size(); ++index) {
      SCOPED_TRACE(index);
      auto outputs = bindings[index].GetOutputValues();
      ASSERT_EQ(outputs.size(), 1u);
      EXPECT_EQ(outputs[0].GetTensorMutableRawData(), gpu_outputs[index].GetTensorMutableRawData());
      Ort::ThrowOnError(ort_env->CopyTensor(gpu_outputs[index], cpu_output, nullptr));
      VerifyOutput(output_data, values[index]);
    }
  }
}

TEST_F(PluginEpWebGpuConcurrency, SerialReusedNonzeroGpuInputAndPreallocatedOutput) {
  auto session = CreateSession();
  Ort::Allocator allocator(*session, Device().GetMemoryInfo(OrtDeviceMemoryType_DEFAULT));
  const auto cpu_memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::array<float, kElements> input_data{};
  std::array<float, kElements> output_data{};
  auto cpu_input = Ort::Value::CreateTensor<float>(
      cpu_memory, input_data.data(), input_data.size(), kShape.data(), kShape.size());
  auto cpu_output = Ort::Value::CreateTensor<float>(
      cpu_memory, output_data.data(), output_data.size(), kShape.data(), kShape.size());

  for (int iteration = 0; iteration < kIterations; ++iteration) {
    SCOPED_TRACE(iteration);
    const float value = static_cast<float>(iteration + 1);
    input_data.fill(value);
    auto gpu_output = CreateReusedNonzeroGpuTensor(allocator, -value);
    auto gpu_input = CreateReusedNonzeroGpuTensor(allocator, -100.0f - value);
    Ort::ThrowOnError(ort_env->CopyTensor(cpu_input, gpu_input, nullptr));
    Ort::IoBinding binding(*session);
    binding.BindInput("X", gpu_input);
    binding.BindOutput("Y", gpu_output);
    // A late reused-buffer clear must not overwrite the new input upload.
    session->Run(Ort::RunOptions{nullptr}, binding);
    auto outputs = binding.GetOutputValues();
    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_EQ(outputs[0].GetTensorMutableRawData(), gpu_output.GetTensorMutableRawData());
    Ort::ThrowOnError(ort_env->CopyTensor(gpu_output, cpu_output, nullptr));
    VerifyOutput(output_data, value);
  }
}

TEST_F(PluginEpWebGpuConcurrency, SerialSessionDestructionPreservesBoundSessionAndEnvTensors) {
  auto shared_allocator = CreateSharedAllocator();
  ASSERT_NE(shared_allocator, nullptr);
  auto env_source = Ort::Value::CreateTensor<float>(shared_allocator, kShape.data(), kShape.size());
  auto env_destination = Ort::Value::CreateTensor<float>(shared_allocator, kShape.data(), kShape.size());
  const auto cpu_memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::array<float, kElements> input_data{};
  std::array<float, kElements> env_data{};
  std::array<float, kElements> output_data{};
  auto cpu_input = Ort::Value::CreateTensor<float>(
      cpu_memory, input_data.data(), input_data.size(), kShape.data(), kShape.size());
  auto env_input = Ort::Value::CreateTensor<float>(
      cpu_memory, env_data.data(), env_data.size(), kShape.data(), kShape.size());
  auto cpu_output = Ort::Value::CreateTensor<float>(
      cpu_memory, output_data.data(), output_data.size(), kShape.data(), kShape.size());
  {
    auto surviving_session = CreateSession();
    Ort::Allocator allocator(*surviving_session, Device().GetMemoryInfo(OrtDeviceMemoryType_DEFAULT));
    auto gpu_output = Ort::Value::CreateTensor<float>(allocator, kShape.data(), kShape.size());
    Ort::IoBinding binding(*surviving_session);
    binding.BindOutput("Y", gpu_output);

    for (int iteration = 0; iteration < kIterations; ++iteration) {
      SCOPED_TRACE(iteration);
      auto retiring_session = CreateSession();
      RunWithCpuInputAndOutput(*retiring_session, static_cast<float>(100 + iteration));
      const float value = static_cast<float>(iteration + 1);
      input_data.fill(value);
      env_data.fill(-value);
      binding.ClearBoundInputs();
      binding.BindInput("X", cpu_input);
      Ort::ThrowOnError(ort_env->CopyTensor(env_input, env_source, nullptr));
      Ort::ThrowOnError(ort_env->CopyTensor(env_source, env_destination, nullptr));

      // Destroy only the idle session while another session's binding and Env copies remain live.
      retiring_session.reset();
      surviving_session->Run(Ort::RunOptions{nullptr}, binding);
      Ort::ThrowOnError(ort_env->CopyTensor(gpu_output, cpu_output, nullptr));
      VerifyOutput(output_data, value);
      Ort::ThrowOnError(ort_env->CopyTensor(env_destination, cpu_output, nullptr));
      EXPECT_EQ(output_data, env_data);
    }
  }

  // The same Env allocations must remain usable after the last session is destroyed.
  output_data.fill(0.0f);
  Ort::ThrowOnError(ort_env->CopyTensor(env_destination, cpu_output, nullptr));
  EXPECT_EQ(output_data, env_data);
  env_data.fill(321.0f);
  Ort::ThrowOnError(ort_env->CopyTensor(env_input, env_source, nullptr));
  Ort::ThrowOnError(ort_env->CopyTensor(env_source, env_destination, nullptr));
  Ort::ThrowOnError(ort_env->CopyTensor(env_destination, cpu_output, nullptr));
  EXPECT_EQ(output_data, env_data);
  CopyTensorRoundTrip(shared_allocator, -456.0f);
}

TEST_F(PluginEpWebGpuConcurrency, SerialGraphCaptureWithCpuFeedsAndFetches) {
  auto session = CreateSession(true);
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    SCOPED_TRACE(iteration);
    RunWithCpuInputAndOutput(*session, static_cast<float>(iteration + 1));
    // Framework feed/fetch copies run while the graph manager is active. Recapture each time:
    // those copies are not replayed, and the next Run supplies new CPU inputs and outputs.
    session->ReleaseCapturedGraph(0);
  }
}

TEST_F(PluginEpWebGpuConcurrency, SerialSingleSessionMultipleGraphCaptureIds) {
  auto session = CreateSession(true);
  Ort::Allocator allocator(*session, Device().GetMemoryInfo(OrtDeviceMemoryType_DEFAULT));
  const auto cpu_memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  constexpr std::array<int, 2> graph_ids{1, 2};
  std::array<Ort::RunOptions, 2> run_options;
  std::array<Ort::Value, 2> gpu_inputs{
      Ort::Value::CreateTensor<float>(allocator, kShape.data(), kShape.size()),
      Ort::Value::CreateTensor<float>(allocator, kShape.data(), kShape.size())};
  std::array<Ort::Value, 2> gpu_outputs{
      Ort::Value::CreateTensor<float>(allocator, kShape.data(), kShape.size()),
      Ort::Value::CreateTensor<float>(allocator, kShape.data(), kShape.size())};
  std::array<Ort::IoBinding, 2> bindings{Ort::IoBinding(*session), Ort::IoBinding(*session)};
  for (size_t index = 0; index < graph_ids.size(); ++index) {
    run_options[index].AddConfigEntry(kOrtRunOptionsConfigCudaGraphAnnotation,
                                      std::to_string(graph_ids[index]).c_str());
    bindings[index].BindInput("X", gpu_inputs[index]);
    bindings[index].BindOutput("Y", gpu_outputs[index]);
  }
  std::array<float, kElements> input_data{};
  std::array<float, kElements> output_data{};
  auto cpu_input = Ort::Value::CreateTensor<float>(
      cpu_memory, input_data.data(), input_data.size(), kShape.data(), kShape.size());
  auto cpu_output = Ort::Value::CreateTensor<float>(
      cpu_memory, output_data.data(), output_data.size(), kShape.data(), kShape.size());

  for (int iteration = 0; iteration < kIterations; ++iteration) {
    SCOPED_TRACE(iteration);
    if (iteration == kIterations / 2) {
      session->ReleaseCapturedGraph(graph_ids[0]);
    }
    const std::array<float, 2> values{static_cast<float>(iteration + 1),
                                      -static_cast<float>(100 + iteration)};
    for (size_t index = 0; index < graph_ids.size(); ++index) {
      input_data.fill(values[index]);
      Ort::ThrowOnError(ort_env->CopyTensor(cpu_input, gpu_inputs[index], nullptr));
    }
    // Distinct stable bindings expose replay of the wrong ID; alternate replay order as well.
    const size_t first = static_cast<size_t>(iteration % 2);
    session->Run(run_options[first], bindings[first]);
    session->Run(run_options[1 - first], bindings[1 - first]);
    for (size_t index = 0; index < graph_ids.size(); ++index) {
      SCOPED_TRACE(graph_ids[index]);
      Ort::ThrowOnError(ort_env->CopyTensor(gpu_outputs[index], cpu_output, nullptr));
      VerifyOutput(output_data, values[index]);
    }
  }
  for (int graph_id : graph_ids) {
    session->ReleaseCapturedGraph(graph_id);
  }
}

TEST_F(PluginEpWebGpuConcurrency, LegacyOnlyConcurrentRunsAreRejected) {
  if (Env::Default().GetEnvironmentVar("ORT_WEBGPU_EP_FORCE_LEGACY") != "1") {
    GTEST_SKIP() << "Requires ORT_WEBGPU_EP_FORCE_LEGACY=1 before loading the plugin.";
  }

  struct LogGate {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered{false};
    bool released{false};
    bool timed_out{false};
    bool finished{false};

    static void ORT_API_CALL Log(void* param, OrtLoggingLevel, const char*, const char*, const char*,
                                 const char* message) {
      if (std::string_view{message}.find("Starting program") == std::string_view::npos) {
        return;
      }
      auto& gate = *static_cast<LogGate*>(param);
      std::unique_lock lock{gate.mutex};
      gate.entered = true;
      gate.changed.notify_all();
      // Bound the hold so a missing overlap guard cannot leave the test waiting indefinitely.
      if (!gate.changed.wait_for(lock, std::chrono::seconds{10}, [&] { return gate.released; })) {
        gate.timed_out = true;
      }
    }
  } gate;

  Ort::SessionOptions options;
  options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
  options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_INFO);
  Ort::ThrowOnError(Ort::GetApi().SetUserLoggingFunction(options, LogGate::Log, &gate));
  options.AppendExecutionProvider_V2(
      *ort_env, {Device()}, std::unordered_map<std::string, std::string>{});
  Ort::Session first_session(*ort_env, ORT_TSTR("testdata/mul_1.onnx"), options);
  auto second_session = CreateSession();
  const auto gpu_memory = Device().GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
  Ort::Allocator first_allocator(first_session, gpu_memory);
  Ort::Allocator second_allocator(*second_session, gpu_memory);
  auto first_input = Ort::Value::CreateTensor<float>(first_allocator, kShape.data(), kShape.size());
  auto first_output = Ort::Value::CreateTensor<float>(first_allocator, kShape.data(), kShape.size());
  auto second_input = Ort::Value::CreateTensor<float>(second_allocator, kShape.data(), kShape.size());
  auto second_output = Ort::Value::CreateTensor<float>(second_allocator, kShape.data(), kShape.size());
  const auto cpu_memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::array<float, kElements> input_data{};
  input_data.fill(2.0f);
  auto cpu_input = Ort::Value::CreateTensor<float>(
      cpu_memory, input_data.data(), input_data.size(), kShape.data(), kShape.size());
  Ort::ThrowOnError(ort_env->CopyTensor(cpu_input, first_input, nullptr));
  Ort::ThrowOnError(ort_env->CopyTensor(cpu_input, second_input, nullptr));
  Ort::IoBinding first_binding(first_session);
  first_binding.BindInput("X", first_input);
  first_binding.BindOutput("Y", first_output);
  Ort::IoBinding second_binding(*second_session);
  second_binding.BindInput("X", second_input);
  second_binding.BindOutput("Y", second_output);

  std::string first_error;
  std::thread worker([&] {
    try {
      first_session.Run(Ort::RunOptions{nullptr}, first_binding);
    } catch (const std::exception& ex) {
      first_error = ex.what();
    } catch (...) {
      first_error = "Unexpected non-standard exception";
    }
    std::lock_guard lock{gate.mutex};
    gate.finished = true;
    gate.changed.notify_all();
  });
  bool first_started = false;
  {
    std::unique_lock lock{gate.mutex};
    gate.changed.wait_for(lock, std::chrono::seconds{30}, [&] { return gate.entered || gate.finished; });
    first_started = gate.entered;
  }

  std::string rejection;
  if (first_started) {
    // Only Run overlaps: all allocation, upload, and binding happened before starting the worker.
    try {
      second_session->Run(Ort::RunOptions{nullptr}, second_binding);
    } catch (const std::exception& ex) {
      rejection = ex.what();
    } catch (...) {
      rejection = "Unexpected non-standard exception";
    }
  }
  {
    std::lock_guard lock{gate.mutex};
    gate.released = true;
    gate.changed.notify_all();
  }
  worker.join();

  // No assertion can leave a worker waiting in the callback.
  ASSERT_TRUE(first_started) << first_error;
  ASSERT_FALSE(gate.timed_out) << "Overlapping Run did not return while the first Run was held";
  ASSERT_TRUE(first_error.empty()) << first_error;
  ASSERT_NE(rejection.find("Sessions on the same device to run sequentially"), std::string::npos) << rejection;
  ASSERT_NE(rejection.find("upgrade to the latest ONNX Runtime"), std::string::npos) << rejection;
  std::array<float, kElements> output_data{};
  auto cpu_output = Ort::Value::CreateTensor<float>(
      cpu_memory, output_data.data(), output_data.size(), kShape.data(), kShape.size());
  Ort::ThrowOnError(ort_env->CopyTensor(first_output, cpu_output, nullptr));
  VerifyOutput(output_data, 2.0f);
  RunAndVerify(*second_session, 3.0f);
}

TEST_F(PluginEpWebGpuConcurrency, DifferentSessionsCpuBindInputReusesDirtyGpuBuffersConcurrently) {
  std::array<std::unique_ptr<Ort::Session>, kThreads> sessions;
  for (auto& session : sessions) {
    session = CreateSession();
  }

  FirstError error;
  RunWorkers(error, [&](int thread_id) {
    for (int iteration = 0; iteration < kIterations && !error.Failed(); ++iteration) {
      const float value = static_cast<float>(thread_id * kIterations + iteration + 1);
      RunWithCpuBoundInputAndDirtyGpuReuse(*sessions[thread_id], value);
    }
  });

  ASSERT_FALSE(error.Failed()) << error.Message();
}

TEST_F(PluginEpWebGpuConcurrency, GraphCaptureReplayInterleavedWithIdleSessionCpuBindInput) {
  auto captured_session = CreateSession(true);
  auto idle_session = CreateSession();
  const auto cpu_memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  const auto gpu_memory = Device().GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
  Ort::Allocator captured_allocator(*captured_session, gpu_memory);
  Ort::Allocator idle_allocator(*idle_session, gpu_memory);
  auto captured_input = Ort::Value::CreateTensor<float>(captured_allocator, kShape.data(), kShape.size());
  auto captured_output = Ort::Value::CreateTensor<float>(captured_allocator, kShape.data(), kShape.size());
  auto idle_output = Ort::Value::CreateTensor<float>(idle_allocator, kShape.data(), kShape.size());
  std::array<float, kElements> captured_data{};
  std::array<float, kElements> idle_data{};
  std::array<float, kElements> dirty_data{};
  std::array<float, kElements> output_data{};
  auto captured_source = Ort::Value::CreateTensor<float>(
      cpu_memory, captured_data.data(), captured_data.size(), kShape.data(), kShape.size());
  auto idle_input = Ort::Value::CreateTensor<float>(
      cpu_memory, idle_data.data(), idle_data.size(), kShape.data(), kShape.size());
  auto dirty_source = Ort::Value::CreateTensor<float>(
      cpu_memory, dirty_data.data(), dirty_data.size(), kShape.data(), kShape.size());
  auto cpu_output = Ort::Value::CreateTensor<float>(
      cpu_memory, output_data.data(), output_data.size(), kShape.data(), kShape.size());
  Ort::IoBinding captured_binding(*captured_session);
  captured_binding.BindInput("X", captured_input);
  captured_binding.BindOutput("Y", captured_output);
  Ort::IoBinding idle_binding(*idle_session);
  idle_binding.BindOutput("Y", idle_output);

  for (int iteration = 0; iteration < kIterations; ++iteration) {
    SCOPED_TRACE(iteration);
    const float captured_value = static_cast<float>(iteration + 1);
    const float idle_value = static_cast<float>(100 + iteration);
    captured_data.fill(captured_value);
    idle_data.fill(idle_value);
    dirty_data.fill(-idle_value);
    idle_binding.ClearBoundInputs();
    {
      auto scratch = Ort::Value::CreateTensor<float>(idle_allocator, kShape.data(), kShape.size());
      Ort::ThrowOnError(ort_env->CopyTensor(dirty_source, scratch, nullptr));
    }
    // BindInput must flush the idle session's reused-buffer clear without changing
    // the captured session's buffer-manager routing or retained graph resources.
    idle_binding.BindInput("X", idle_input);
    Ort::ThrowOnError(ort_env->CopyTensor(captured_source, captured_input, nullptr));
    captured_session->Run(Ort::RunOptions{nullptr}, captured_binding);
    Ort::ThrowOnError(ort_env->CopyTensor(captured_output, cpu_output, nullptr));
    VerifyOutput(output_data, captured_value);

    idle_session->Run(Ort::RunOptions{nullptr}, idle_binding);
    Ort::ThrowOnError(ort_env->CopyTensor(idle_output, cpu_output, nullptr));
    VerifyOutput(output_data, idle_value);
    if (iteration == kIterations / 2) {
      captured_session->ReleaseCapturedGraph(0);
    }
  }
  captured_session->ReleaseCapturedGraph(0);
}

TEST_F(PluginEpWebGpuConcurrency, SessionCreationAndDestructionDuringStreamlessCopies) {
  std::array<std::unique_ptr<Ort::Session>, kThreads - 1> sessions;
  for (auto& session : sessions) {
    session = CreateSession();
  }
  auto shared_allocator = CreateSharedAllocator();
  ASSERT_NE(shared_allocator, nullptr);
  std::atomic<bool> lifecycle_done{false};

  FirstError error;
  RunWorkers(error, [&](int thread_id) {
    if (thread_id == 0) {
      for (int iteration = 0; iteration < kIterations && !error.Failed(); ++iteration) {
        auto session = CreateSession();
        RunWithCpuBoundInputAndDirtyGpuReuse(*session, static_cast<float>(iteration + 1));
      }
      lifecycle_done.store(true);
      return;
    }

    // Keep copying through the final session destruction, not just its slower initial creation.
    for (int iteration = 0; (iteration < kIterations || !lifecycle_done.load()) && !error.Failed(); ++iteration) {
      const float value = static_cast<float>(thread_id * kIterations + iteration + 1);
      RunWithCpuBoundInputAndDirtyGpuReuse(*sessions[thread_id - 1], value);
      CopyTensorRoundTrip(shared_allocator, -value);
    }
  });

  ASSERT_FALSE(error.Failed()) << error.Message();
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

TEST_F(PluginEpWebGpuConcurrency, SessionAllocatorsCreateAndCopyConcurrently) {
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

TEST_F(PluginEpWebGpuConcurrency, SharedAllocatorCreatesAndCopiesConcurrently) {
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

TEST_F(PluginEpWebGpuConcurrency, SameSessionRunsAreSerialized) {
  auto session = CreateSession();
  FirstError error;
  RunWorkers(error, [&](int thread_id) {
    for (int iteration = 0; iteration < kIterations && !error.Failed(); ++iteration) {
      RunWithCpuInputAndOutput(*session, static_cast<float>(thread_id * kIterations + iteration + 1));
    }
  });
  ASSERT_FALSE(error.Failed()) << error.Message();
}

TEST_F(PluginEpWebGpuConcurrency, SameSessionAllocatorCreatesAndCopiesConcurrently) {
  auto allocator_session = CreateSession();
  Ort::Allocator allocator(*allocator_session, Device().GetMemoryInfo(OrtDeviceMemoryType_DEFAULT));
  FirstError error;
  RunWorkers(error, [&](int thread_id) {
    for (int iteration = 0; iteration < kIterations && !error.Failed(); ++iteration) {
      CopyTensorRoundTrip(allocator, static_cast<float>(thread_id * kIterations + iteration + 1));
    }
  });
  ASSERT_FALSE(error.Failed()) << error.Message();
}

TEST_F(PluginEpWebGpuConcurrency, SameSessionAllocatorCreatesAndCopiesDuringGraphReplay) {
  auto session = CreateSession(true);
  Ort::Allocator allocator(*session, Device().GetMemoryInfo(OrtDeviceMemoryType_DEFAULT));
  auto gpu_input = Ort::Value::CreateTensor<float>(allocator, kShape.data(), kShape.size());
  auto gpu_output = Ort::Value::CreateTensor<float>(allocator, kShape.data(), kShape.size());
  const auto cpu_memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::array<float, kElements> input_data{};
  std::array<float, kElements> output_data{};
  auto cpu_input = Ort::Value::CreateTensor<float>(
      cpu_memory, input_data.data(), input_data.size(), kShape.data(), kShape.size());
  auto cpu_output = Ort::Value::CreateTensor<float>(
      cpu_memory, output_data.data(), output_data.size(), kShape.data(), kShape.size());
  Ort::IoBinding binding(*session);
  binding.BindInput("X", gpu_input);
  binding.BindOutput("Y", gpu_output);
  const auto run_and_verify = [&](float value) {
    input_data.fill(value);
    Ort::ThrowOnError(ort_env->CopyTensor(cpu_input, gpu_input, nullptr));
    session->Run(Ort::RunOptions{nullptr}, binding);
    Ort::ThrowOnError(ort_env->CopyTensor(gpu_output, cpu_output, nullptr));
    VerifyOutput(output_data, value);
  };

  // Warm up and capture before starting allocator workers. Only replay is concurrent;
  // bound tensors and the binding itself stay on the Run thread.
  run_and_verify(1.0f);
  CopyTensorRoundTrip(allocator, 1.0f, true);
  FirstError error;
  RunWorkers(error, [&](int thread_id) {
    for (int iteration = 0; iteration < kIterations && !error.Failed(); ++iteration) {
      const float value = static_cast<float>(thread_id * kIterations + iteration + 2);
      if (thread_id == 0) {
        run_and_verify(value);
      } else {
        CopyTensorRoundTrip(allocator, value, true);
      }
    }
  });
  ASSERT_FALSE(error.Failed()) << error.Message();
}

TEST_F(PluginEpWebGpuConcurrency, SessionAllocatorReusesBuffersFreedDuringGraphReplay) {
  auto session = CreateSession(true);
  Ort::Allocator allocator(*session, Device().GetMemoryInfo(OrtDeviceMemoryType_DEFAULT));
  auto gpu_input = Ort::Value::CreateTensor<float>(allocator, kShape.data(), kShape.size());
  auto gpu_output = Ort::Value::CreateTensor<float>(allocator, kShape.data(), kShape.size());
  Ort::IoBinding binding(*session);
  binding.BindInput("X", gpu_input);
  binding.BindOutput("Y", gpu_output);
  session->Run(Ort::RunOptions{nullptr}, binding);

  // Hold a full bucket so all available buffers of this size belong to this test.
  // The model's tensors use the separate 64-byte bucket.
  constexpr size_t kBufferSize = 4096;
  constexpr size_t kBucketCapacity = 200;
  InlinedVector<std::optional<Ort::MemoryAllocation>> buffers;
  InlinedHashSet<void*> handles;
  for (size_t i = 0; i < kBucketCapacity; ++i) {
    buffers.emplace_back(allocator.GetAllocation(kBufferSize));
    ASSERT_NE(buffers.back()->get(), nullptr);
    handles.insert(buffers.back()->get());
  }

  FirstError error;
  RunWorkers(error, [&](int thread_id) {
    if (thread_id == 0) {
      for (int iteration = 0; iteration < kIterations && !error.Failed(); ++iteration) {
        session->Run(Ort::RunOptions{nullptr}, binding);
      }
    } else {
      for (size_t i = static_cast<size_t>(thread_id - 1); i < buffers.size(); i += kThreads - 1) {
        buffers[i].reset();
        std::this_thread::yield();
      }
    }
  });
  ASSERT_FALSE(error.Failed()) << error.Message();
  // Submit once after every Free has completed, including frees at the end of the replay loop.
  session->Run(Ort::RunOptions{nullptr}, binding);

  InlinedVector<Ort::MemoryAllocation> reacquired;
  for (size_t i = 0; i < kBucketCapacity; ++i) {
    reacquired.emplace_back(allocator.GetAllocation(kBufferSize));
    EXPECT_EQ(handles.erase(reacquired.back().get()), 1u);
  }
  EXPECT_TRUE(handles.empty());
}

TEST_F(PluginEpWebGpuConcurrency, SameSessionCapturedGraphsReplayConcurrently) {
  RunGraphsConcurrently(/*separate_sessions=*/false, /*capture_before_workers=*/true);
}

TEST_F(PluginEpWebGpuConcurrency, SameSessionGraphsCaptureAndReplayConcurrently) {
  RunGraphsConcurrently(/*separate_sessions=*/false, /*capture_before_workers=*/false);
}

TEST_F(PluginEpWebGpuConcurrency, DifferentSessionsGraphsCaptureAndReplayConcurrently) {
  RunGraphsConcurrently(/*separate_sessions=*/true, /*capture_before_workers=*/false);
}

TEST_F(PluginEpWebGpuConcurrency, DedicatedSessionAllocatorFeedsConcurrentSessions) {
  auto allocator_session = CreateSession();
  Ort::Allocator allocator(*allocator_session, Device().GetMemoryInfo(OrtDeviceMemoryType_DEFAULT));
  std::array<std::unique_ptr<Ort::Session>, kThreads> sessions;
  for (auto& session : sessions) {
    session = CreateSession();
  }
  const auto cpu_memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  const std::array<const char*, 1> input_names{"X"};
  const std::array<const char*, 1> output_names{"Y"};
  FirstError error;
  RunWorkers(error, [&](int thread_id) {
    for (int iteration = 0; iteration < kIterations && !error.Failed(); ++iteration) {
      const float value = static_cast<float>(thread_id * kIterations + iteration + 1);
      std::array<float, kElements> input_data{};
      input_data.fill(value);
      auto input = Ort::Value::CreateTensor<float>(
          cpu_memory, input_data.data(), input_data.size(), kShape.data(), kShape.size());
      auto gpu_input = Ort::Value::CreateTensor<float>(allocator, kShape.data(), kShape.size());
      Ort::ThrowOnError(ort_env->CopyTensor(input, gpu_input, nullptr));
      auto outputs = sessions[thread_id]->Run(Ort::RunOptions{nullptr}, input_names.data(), &gpu_input, 1,
                                              output_names.data(), output_names.size());
      std::array<float, kElements> output_data{};
      std::copy_n(outputs.front().GetTensorData<float>(), output_data.size(), output_data.begin());
      VerifyOutput(output_data, value);
    }
  });
  ASSERT_FALSE(error.Failed()) << error.Message();
}

TEST_F(PluginEpWebGpuConcurrency, SharedGpuCopyIsSubmittedBeforeSessionRun) {
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
    Ort::ThrowOnError(ort_env->CopyTensor(input, source, nullptr));
    Ort::ThrowOnError(ort_env->CopyTensor(source, destination, nullptr));
    auto result = session->Run(Ort::RunOptions{nullptr}, input_names.data(), &destination, 1,
                               output_names.data(), output_names.size());
    std::array<float, kElements> output{};
    std::copy_n(result.front().GetTensorData<float>(), output.size(), output.begin());
    VerifyOutput(output, value);
  }
}

TEST_F(PluginEpWebGpuConcurrency, MixedSessionAndAllocatorOperationsConcurrently) {
  constexpr int kOperationGroups = 4;
  constexpr int kTotalThreads = kOperationGroups * kThreads;
  constexpr int kMixedIterations = 10;

  std::array<std::unique_ptr<Ort::Session>, kThreads> run_sessions;
  for (int thread_id = 0; thread_id < kThreads; ++thread_id) {
    run_sessions[thread_id] = CreateSession();
  }

  const auto gpu_memory_info = Device().GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
  auto shared_allocator = CreateSharedAllocator();
  ASSERT_NE(shared_allocator, nullptr);

  FirstError error;
  std::barrier start{kTotalThreads};
  std::vector<std::thread> threads;
  threads.reserve(kTotalThreads);

  const auto add_workers = [&](std::string group_name, auto work) {
    for (int thread_id = 0; thread_id < kThreads; ++thread_id) {
      threads.emplace_back([&, group_name, work, thread_id]() {
        try {
          start.arrive_and_wait();
          work(thread_id);
        } catch (const std::exception& ex) {
          error.Set(group_name + " thread " + std::to_string(thread_id) + ": " + ex.what());
        }
      });
    }
  };

  add_workers("create session", [&](int /*thread_id*/) {
    for (int iteration = 0; iteration < kMixedIterations && !error.Failed(); ++iteration) {
      auto session = CreateSession();
    }
  });

  add_workers("run session", [&](int thread_id) {
    for (int iteration = 0; iteration < kMixedIterations && !error.Failed(); ++iteration) {
      const float value = static_cast<float>(thread_id * kMixedIterations + iteration + 1);
      RunWithCpuInputAndOutput(*run_sessions[thread_id], value);
    }
  });

  add_workers("session allocator", [&](int thread_id) {
    Ort::Allocator allocator(*run_sessions[thread_id], gpu_memory_info);
    for (int iteration = 0; iteration < kMixedIterations && !error.Failed(); ++iteration) {
      const float value = static_cast<float>(100 + thread_id * kMixedIterations + iteration);
      CopyTensorRoundTrip(allocator, value);
    }
  });

  add_workers("shared allocator", [&](int thread_id) {
    for (int iteration = 0; iteration < kMixedIterations && !error.Failed(); ++iteration) {
      const float value = static_cast<float>(200 + thread_id * kMixedIterations + iteration);
      CopyTensorRoundTrip(shared_allocator, value);
    }
  });

  for (auto& thread : threads) {
    thread.join();
  }

  ASSERT_FALSE(error.Failed()) << error.Message();
}

#endif  // defined(ORT_UNIT_TEST_HAS_WEBGPU_PLUGIN_EP)

}  // namespace test
}  // namespace onnxruntime
