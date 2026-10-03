// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <set>
#include <thread>
#include <vector>

#include <gsl/gsl>
#include <gtest/gtest.h>

#include "core/framework/allocator.h"
#include "core/graph/constants.h"
#include "core/graph/onnx_protobuf.h"
#include "core/platform/env_var_utils.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_ep_device_ep_metadata_keys.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

#include "test/providers/webgpu/plugin/webgpu_plugin_test_utils.h"
#include "test/util/include/api_asserts.h"

#if defined(ORT_UNIT_TEST_HAS_WEBGPU_PLUGIN_EP) && !defined(USE_EXTERNAL_DAWN) && !defined(BUILD_DAWN_SHARED_LIBRARY)
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif
#include "dawn/dawn_proc.h"
#include "dawn/native/DawnNative.h"
#include "dawn/webgpu_cpp.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#endif

extern std::unique_ptr<Ort::Env> ort_env;
extern "C" void ortenv_setup();
extern "C" void ortenv_teardown();

namespace onnxruntime {
namespace test {

#if defined(ORT_UNIT_TEST_HAS_WEBGPU_PLUGIN_EP)

namespace {

struct UserWebGpuAllocator : OrtAllocator {
  explicit UserWebGpuAllocator(const OrtMemoryInfo* memory_info) : memory_info_{memory_info} {
    version = ORT_API_VERSION;
    Alloc = AllocImpl;
    Free = FreeImpl;
    Info = InfoImpl;
    Reserve = AllocImpl;
    GetStats = nullptr;
    AllocOnStream = nullptr;
    Shrink = nullptr;
  }

  size_t NumAllocations() const {
    return num_allocations_;
  }

  size_t NumFrees() const {
    return num_frees_;
  }

 private:
  static void* ORT_API_CALL AllocImpl(OrtAllocator* this_, size_t size) {
    auto& allocator = *static_cast<UserWebGpuAllocator*>(this_);
    void* allocation = std::malloc(size);
    if (allocation != nullptr) {
      ++allocator.num_allocations_;
    }

    return allocation;
  }

  static void ORT_API_CALL FreeImpl(OrtAllocator* this_, void* allocation) {
    auto& allocator = *static_cast<UserWebGpuAllocator*>(this_);
    if (allocation != nullptr) {
      ++allocator.num_frees_;
    }

    std::free(allocation);
  }

  static const OrtMemoryInfo* ORT_API_CALL InfoImpl(const OrtAllocator* this_) {
    return static_cast<const UserWebGpuAllocator*>(this_)->memory_info_;
  }

  const OrtMemoryInfo* memory_info_;
  size_t num_allocations_{0};
  size_t num_frees_{0};
};

// The existing test models finish too quickly to make GPU utilization observable. This configurable workload supports
// the disabled manual diagnostic below, which verifies that the WebGPU device selected by the developer is the device
// that actually executes the workload in a multi-GPU setup.
std::string BuildMatMulLoadModelBytes(int64_t dimension, size_t depth) {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  auto* opset = model.add_opset_import();
  opset->set_domain(onnxruntime::kOnnxDomain);
  opset->set_version(18);

  auto* graph = model.mutable_graph();
  graph->set_name("webgpu_multi_device_load");

  const auto add_value_info = [graph, dimension](std::string_view name, bool is_input) {
    auto* value_info = is_input ? graph->add_input() : graph->add_output();
    value_info->set_name(std::string{name});
    auto* tensor_type = value_info->mutable_type()->mutable_tensor_type();
    tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    tensor_type->mutable_shape()->add_dim()->set_dim_value(dimension);
    tensor_type->mutable_shape()->add_dim()->set_dim_value(dimension);
  };

  add_value_info("A", true);
  add_value_info("B", true);
  add_value_info("Y", false);

  std::string current_input = "A";
  for (size_t i = 0; i < depth; ++i) {
    const std::string matmul_output = "matmul_" + std::to_string(i);
    const std::string relu_output = i + 1 == depth ? "Y" : "relu_" + std::to_string(i);

    auto* matmul = graph->add_node();
    matmul->set_name("MatMul_" + std::to_string(i));
    matmul->set_op_type("MatMul");
    matmul->add_input(current_input);
    matmul->add_input("B");
    matmul->add_output(matmul_output);

    auto* relu = graph->add_node();
    relu->set_name("Relu_" + std::to_string(i));
    relu->set_op_type("Relu");
    relu->add_input(matmul_output);
    relu->add_output(relu_output);

    current_input = relu_output;
  }

  std::string model_bytes;
  ORT_ENFORCE(model.SerializeToString(&model_bytes), "Failed to serialize WebGPU load model.");
  return model_bytes;
}

class WebGpuPluginSharedAllocatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    registration_.emplace(*ort_env, "webgpu_allocator_test");

    const auto is_virtual_device = [](const Ort::ConstEpDevice& device) {
      const auto metadata = device.Device().Metadata().GetKeyValuePairs();
      const auto is_virtual = metadata.find(kOrtHardwareDevice_MetadataKey_IsVirtual);
      return is_virtual != metadata.end() && is_virtual->second == "1";
    };

    for (const auto& device : registration_->GetEpDevices()) {
      if (!is_virtual_device(device)) {
        ep_devices_.push_back(device);
      }
    }
    ASSERT_FALSE(ep_devices_.empty());
  }

  Ort::ConstEpDevice EpDevice() const {
    return Ort::ConstEpDevice{ep_devices_.front()};
  }

  std::vector<Ort::ConstEpDevice> EpDevices() const {
    std::vector<Ort::ConstEpDevice> devices;
    devices.reserve(ep_devices_.size());
    for (const auto* device : ep_devices_) {
      devices.emplace_back(device);
    }
    return devices;
  }

  Ort::Env& Env() {
    return *ort_env;
  }

 private:
  std::optional<ScopedWebGpuPluginRegistration> registration_;
  std::vector<const OrtEpDevice*> ep_devices_;
};

#if !defined(USE_EXTERNAL_DAWN)
#if !defined(BUILD_DAWN_SHARED_LIBRARY)
namespace {
WGPUBuffer free_failure_buffer = nullptr;
size_t free_failure_probes = 0;
std::atomic<size_t> nonempty_queue_submissions{0};
std::atomic<size_t> storage_buffer_clears{0};
WGPUDevice public_allocation_device = nullptr;

WGPUBuffer CapturePublicAllocationDevice(WGPUDevice device, const WGPUBufferDescriptor* descriptor) {
  public_allocation_device = device;
  return dawn::native::GetProcs().deviceCreateBuffer(device, descriptor);
}

std::array<float, 8> ReadPublicBufferWithDawn(WGPUBuffer buffer) {
  ORT_ENFORCE(public_allocation_device != nullptr, "The allocation hook did not capture a WebGPU device.");
  wgpu::Device device{public_allocation_device};
  constexpr size_t bytes = sizeof(std::array<float, 8>);
  wgpu::BufferDescriptor descriptor{};
  descriptor.size = bytes;
  descriptor.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
  auto staging = device.CreateBuffer(&descriptor);
  auto encoder = device.CreateCommandEncoder();
  encoder.CopyBufferToBuffer(buffer, 0, staging, 0, bytes);
  auto commands = encoder.Finish();
  device.GetQueue().Submit(1, &commands);

  wgpu::MapAsyncStatus map_status{};
  auto future = staging.MapAsync(
      wgpu::MapMode::Read, 0, bytes, wgpu::CallbackMode::WaitAnyOnly,
      [](wgpu::MapAsyncStatus status, wgpu::StringView, wgpu::MapAsyncStatus* result) noexcept {
        *result = status;
      },
      &map_status);
  ORT_ENFORCE(device.GetAdapter().GetInstance().WaitAny(future, UINT64_MAX) == wgpu::WaitStatus::Success,
              "Failed to wait for direct Dawn readback.");
  ORT_ENFORCE(map_status == wgpu::MapAsyncStatus::Success,
              "Failed to map direct Dawn readback: ", static_cast<uint32_t>(map_status));

  const auto* mapped = static_cast<const float*>(staging.GetConstMappedRange());
  ORT_ENFORCE(mapped != nullptr, "Direct Dawn readback returned a null mapped range.");
  std::array<float, 8> result;
  std::copy_n(mapped, result.size(), result.begin());
  staging.Unmap();
  return result;
}

void CountNonemptyQueueSubmissions(WGPUQueue queue, size_t command_count, const WGPUCommandBuffer* commands) {
  if (command_count != 0) {
    nonempty_queue_submissions.fetch_add(1, std::memory_order_relaxed);
  }
  dawn::native::GetProcs().queueSubmit(queue, command_count, commands);
}

void CountStorageBufferClears(WGPUCommandEncoder encoder, WGPUBuffer buffer, uint64_t offset, uint64_t size) {
  const auto& procs = dawn::native::GetProcs();
  if ((procs.bufferGetUsage(buffer) & WGPUBufferUsage_Storage) != 0) {
    storage_buffer_clears.fetch_add(1, std::memory_order_relaxed);
  }
  procs.commandEncoderClearBuffer(encoder, buffer, offset, size);
}

int VerifyPublicAllocSubmitsReusedBufferClear(Ort::Env& env, Ort::ConstEpDevice ep_device,
                                              bool use_session_allocator, bool cancel_run = false) {
  auto procs = dawn::native::GetProcs();
  procs.queueSubmit = CountNonemptyQueueSubmissions;
  procs.deviceCreateBuffer = CapturePublicAllocationDevice;
  dawnProcSetProcs(&procs);
  Ort::SessionOptions options;
  options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
  options.AppendExecutionProvider_V2(
      env, {ep_device}, {{"dawnProcTable", std::to_string(reinterpret_cast<uintptr_t>(&procs))}});
  std::optional<Ort::Session> session;
  session.emplace(env, ORT_TSTR("testdata/mul_1.onnx"), options);
  const auto gpu_memory = ep_device.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
  std::optional<Ort::Allocator> session_allocator;
  OrtAllocator* allocator = nullptr;
  if (use_session_allocator) {
    session_allocator.emplace(*session, gpu_memory);
    allocator = *session_allocator;
  } else {
    // Configure the plugin's proc table, then exercise the Env allocator without a live Session.
    session.reset();
    allocator = env.GetSharedAllocator(gpu_memory);
  }
  if (allocator == nullptr) {
    return 2;
  }

  if (cancel_run) {
    ORT_ENFORCE(session.has_value(), "Run cancellation requires a Session allocator.");
    constexpr std::array<int64_t, 2> run_shape{3, 2};
    std::array<float, 6> run_data;
    run_data.fill(1.0f);
    auto cpu_memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    auto input = Ort::Value::CreateTensor<float>(
        cpu_memory, run_data.data(), run_data.size(), run_shape.data(), run_shape.size());
    Ort::IoBinding binding(*session);
    binding.BindInput("X", input);
    binding.BindOutput("Y", gpu_memory);
    Ort::RunOptions run_options;
    run_options.SetTerminate();
    Ort::Status status{Ort::GetApi().RunWithBinding(*session, run_options, binding)};
    const auto message = status.GetErrorMessage();
    // This error comes from the executor after OnRunStart, not input validation.
    if (status.IsOK() || message.find("Exiting due to terminate flag") == std::string::npos) {
      std::fprintf(stderr, "Expected an executor cancellation, got: %s\n", message.c_str());
      return 9;
    }
  }

  constexpr std::array<int64_t, 1> shape{8};
  std::array<float, 8> nonzero_data;
  nonzero_data.fill(7.0f);
  const auto cpu_memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  auto cpu_input = Ort::Value::CreateTensor<float>(
      cpu_memory, nonzero_data.data(), nonzero_data.size(), shape.data(), shape.size());
  void* buffer = allocator->Alloc(allocator, sizeof(nonzero_data));
  auto free_buffer = gsl::finally([&] {
    if (buffer != nullptr) {
      allocator->Free(allocator, buffer);
    }
  });
  if (buffer == nullptr) {
    return 3;
  }
  {
    auto gpu_tensor = Ort::Value::CreateTensor<float>(
        gpu_memory, static_cast<float*>(buffer), nonzero_data.size(), shape.data(), shape.size());
    Ort::ThrowOnError(env.CopyTensor(cpu_input, gpu_tensor, nullptr));
    if (ReadPublicBufferWithDawn(static_cast<WGPUBuffer>(buffer)) != nonzero_data) {
      std::fputs("Failed to seed a nonzero cached GPU buffer\n", stderr);
      return 4;
    }
  }
  void* dirty_buffer = buffer;
  allocator->Free(allocator, buffer);
  buffer = nullptr;

  const size_t before = nonempty_queue_submissions.load(std::memory_order_relaxed);
  if (before == 0) {
    std::fputs("Queue submission hook did not observe the seed readback\n", stderr);
    return 8;
  }
  buffer = allocator->Alloc(allocator, sizeof(nonzero_data));
  const size_t after = nonempty_queue_submissions.load(std::memory_order_relaxed);
  if (buffer == nullptr || buffer != dirty_buffer) {
    std::fputs("Ordinary Alloc did not reuse the dirty cached GPU buffer\n", stderr);
    return 5;
  }
  // This encoder only reads the raw buffer. It cannot submit clears left in ORT's recording.
  const auto readback_data = ReadPublicBufferWithDawn(static_cast<WGPUBuffer>(buffer));
  for (size_t index = 0; index < readback_data.size(); ++index) {
    if (readback_data[index] != 0.0f) {
      std::fprintf(stderr, "Direct Dawn readback found old data at element %zu: %g (expected 0)\n",
                   index, static_cast<double>(readback_data[index]));
      return 7;
    }
  }
  // These counts were captured around Alloc, excluding the independent Dawn readback above.
  if (after <= before) {
    std::fputs("Ordinary Alloc did not submit the reused-buffer clear before readback\n", stderr);
    return 6;
  }

  std::fputs("Ordinary Alloc submitted the reused-buffer clear before readback\n", stderr);
  return 0;
}

int VerifyInstanceNormScratchAllocationsStayBatched(Ort::Env& env, Ort::ConstEpDevice ep_device) {
  constexpr std::array<int64_t, 4> shape{1, 2, 2, 3};
  constexpr std::array<int64_t, 1> parameter_shape{2};
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  model.add_opset_import()->set_version(18);
  auto* graph = model.mutable_graph();
  graph->set_name("webgpu_batched_instance_norm_scratch");
  auto* input = graph->add_input();
  input->set_name("X");
  auto* output = graph->add_output();
  output->set_name("Y");
  for (auto* value_info : {input, output}) {
    auto* type = value_info->mutable_type()->mutable_tensor_type();
    type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    for (int64_t dimension : shape) {
      type->mutable_shape()->add_dim()->set_dim_value(dimension);
    }
  }
  for (const char* name : {"scale", "bias"}) {
    auto* parameter = graph->add_input();
    parameter->set_name(name);
    auto* type = parameter->mutable_type()->mutable_tensor_type();
    type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    type->mutable_shape()->add_dim()->set_dim_value(parameter_shape[0]);
  }
  auto* node = graph->add_node();
  node->set_name("instance_norm");
  node->set_op_type("InstanceNormalization");
  node->add_input("X");
  node->add_input("scale");
  node->add_input("bias");
  node->add_output("Y");

  struct SubmissionProbe {
    bool armed{false};
    bool submitted_before_program{false};
    size_t baseline{0};
    size_t programs{0};

    static void ORT_API_CALL Log(void* param, OrtLoggingLevel, const char*, const char*, const char*,
                                 const char* message) {
      auto& probe = *static_cast<SubmissionProbe*>(param);
      if (probe.armed && std::string_view{message}.find("Starting program") != std::string_view::npos) {
        ++probe.programs;
        probe.submitted_before_program |=
            nonempty_queue_submissions.load(std::memory_order_relaxed) != probe.baseline;
      }
    }
  } probe;
  auto procs = dawn::native::GetProcs();
  procs.queueSubmit = CountNonemptyQueueSubmissions;
  procs.commandEncoderClearBuffer = CountStorageBufferClears;
  dawnProcSetProcs(&procs);
  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
  options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
  options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_INFO);
  Ort::ThrowOnError(Ort::GetApi().SetUserLoggingFunction(options, SubmissionProbe::Log, &probe));
  options.AppendExecutionProvider_V2(
      env, {ep_device}, {{"maxNumPendingDispatches", "4096"}, {"dawnProcTable", std::to_string(reinterpret_cast<uintptr_t>(&procs))}});
  const auto model_bytes = model.SerializeAsString();
  Ort::Session session(env, model_bytes.data(), model_bytes.size(), options);
  Ort::Allocator allocator(session, ep_device.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT));
  auto gpu_input = Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
  auto gpu_output = Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
  auto gpu_scale = Ort::Value::CreateTensor<float>(allocator, parameter_shape.data(), parameter_shape.size());
  auto gpu_bias = Ort::Value::CreateTensor<float>(allocator, parameter_shape.data(), parameter_shape.size());
  std::array<float, 12> input_data{-5.0f, -3.0f, -1.0f, 1.0f, 3.0f, 5.0f,
                                   -10.0f, -6.0f, -2.0f, 2.0f, 6.0f, 10.0f};
  std::array<float, 12> output_data{};
  std::array<float, 2> scale_data{1.5f, 0.5f};
  std::array<float, 2> bias_data{0.25f, -0.75f};
  const auto cpu_memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  auto cpu_input = Ort::Value::CreateTensor<float>(
      cpu_memory, input_data.data(), input_data.size(), shape.data(), shape.size());
  auto cpu_output = Ort::Value::CreateTensor<float>(
      cpu_memory, output_data.data(), output_data.size(), shape.data(), shape.size());
  auto cpu_scale = Ort::Value::CreateTensor<float>(
      cpu_memory, scale_data.data(), scale_data.size(), parameter_shape.data(), parameter_shape.size());
  auto cpu_bias = Ort::Value::CreateTensor<float>(
      cpu_memory, bias_data.data(), bias_data.size(), parameter_shape.data(), parameter_shape.size());
  Ort::ThrowOnError(env.CopyTensor(cpu_input, gpu_input, nullptr));
  Ort::ThrowOnError(env.CopyTensor(cpu_scale, gpu_scale, nullptr));
  Ort::ThrowOnError(env.CopyTensor(cpu_bias, gpu_bias, nullptr));
  Ort::IoBinding binding(session);
  binding.BindInput("X", gpu_input);
  binding.BindInput("scale", gpu_scale);
  binding.BindInput("bias", gpu_bias);
  binding.BindOutput("Y", gpu_output);

  auto expected = input_data;
  for (size_t channel = 0; channel < scale_data.size(); ++channel) {
    float mean = 0.0f;
    for (size_t spatial = 0; spatial < 6; ++spatial) {
      mean += expected[channel * 6 + spatial] / 6.0f;
    }
    float variance = 0.0f;
    for (size_t spatial = 0; spatial < 6; ++spatial) {
      const float centered = expected[channel * 6 + spatial] - mean;
      variance += centered * centered / 6.0f;
    }
    for (size_t spatial = 0; spatial < 6; ++spatial) {
      auto& value = expected[channel * 6 + spatial];
      value = scale_data[channel] * (value - mean) / std::sqrt(variance + 1e-5f) + bias_data[channel];
    }
  }

  // Warm pipelines and the scratch-buffer cache before measuring allocation-triggered submits.
  for (int iteration = 0; iteration < 3; ++iteration) {
    session.Run(Ort::RunOptions{nullptr}, binding);
  }
  for (int iteration = 0; iteration < 3; ++iteration) {
    probe.baseline = nonempty_queue_submissions.load(std::memory_order_relaxed);
    const size_t clears_before = storage_buffer_clears.load(std::memory_order_relaxed);
    probe.programs = 0;
    probe.submitted_before_program = false;
    probe.armed = true;
    session.Run(Ort::RunOptions{nullptr}, binding);
    probe.armed = false;
    const size_t submissions = nonempty_queue_submissions.load(std::memory_order_relaxed) - probe.baseline;
    const size_t clears = storage_buffer_clears.load(std::memory_order_relaxed) - clears_before;
    // InstanceNormalization computes scale/shift into CreateGPUTensor scratch, then applies it.
    // Stable graph bindings leave only internal scratch storage eligible for these cache clears.
    if (probe.programs < 2 || probe.submitted_before_program || submissions != 1 || clears == 0) {
      std::fprintf(stderr, "Scratch batching failed: programs=%zu, premature_submit=%d, submissions=%zu, clears=%zu\n",
                   probe.programs, static_cast<int>(probe.submitted_before_program), submissions, clears);
      return 9;
    }
    Ort::ThrowOnError(env.CopyTensor(gpu_output, cpu_output, nullptr));
    for (size_t index = 0; index < expected.size(); ++index) {
      if (!(std::abs(output_data[index] - expected[index]) <= 1e-4f)) {
        std::fputs("Batched InstanceNormalization returned incorrect output\n", stderr);
        return 10;
      }
    }
  }
  std::fputs("Warmed InstanceNormalization scratch allocations retained a single Run submission\n", stderr);
  return 0;
}

WGPUBufferMapState InjectMappedBufferOnFree(WGPUBuffer buffer) {
  if (buffer == free_failure_buffer) {
    ++free_failure_probes;
    std::fputs("Injecting mapped state in actual Session allocator Free\n", stderr);
    std::fflush(stderr);
    return WGPUBufferMapState_Mapped;
  }
  return dawn::native::GetProcs().bufferGetMapState(buffer);
}
}  // namespace
#endif

class WebGpuSessionAllocatorDeathTest : public WebGpuPluginSharedAllocatorTest {
 protected:
  void SetUp() override {
    // Re-exec on POSIX rather than inheriting initialized Dawn/driver state through fork.
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    WebGpuPluginSharedAllocatorTest::SetUp();
  }
};

TEST_F(WebGpuSessionAllocatorDeathTest, PublicApiReportsFactoryStreamsUnsupported) {
  ASSERT_EXIT(
      {
        OrtSyncStream* stream = nullptr;
        std::fputs("Calling public CreateSyncStreamForEpDevice for WebGPU\n", stderr);
        std::fflush(stderr);
        Ort::Status status{Ort::GetApi().CreateSyncStreamForEpDevice(EpDevice(), nullptr, &stream)};
        if (status.IsOK() || status.GetErrorCode() != ORT_NOT_IMPLEMENTED ||
            status.GetErrorMessage() != "WebGPU supports Session-owned streams only; factory streams are not supported.") {
          std::_Exit(72);
        }
        if (stream != nullptr) {
          std::_Exit(73);
        }
        std::fputs(status.GetErrorMessage().c_str(), stderr);
        std::fflush(stderr);
        std::_Exit(0);
      },
      ::testing::ExitedWithCode(0), "WebGPU supports Session-owned streams only; factory streams are not supported.");
}

TEST_F(WebGpuSessionAllocatorDeathTest, FreeContainsExceptionAndPreservesUnreleasedBuffer) {
#if defined(BUILD_DAWN_SHARED_LIBRARY)
  GTEST_SKIP() << "Shared Dawn calls bypass the replaceable proc table required for map-state fault injection.";
#else
  ASSERT_EXIT(
      {
        auto procs = dawn::native::GetProcs();
        procs.bufferGetMapState = InjectMappedBufferOnFree;
        dawnProcSetProcs(&procs);
        Ort::SessionOptions options;
        options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
        options.AppendExecutionProvider_V2(
            Env(), {EpDevice()},
            {{"validationMode", "full"},
             {"dawnProcTable", std::to_string(reinterpret_cast<uintptr_t>(&procs))}});
        Ort::Session session(Env(), ORT_TSTR("testdata/mul_1.onnx"), options);
        Ort::Allocator allocator(session, EpDevice().GetMemoryInfo(OrtDeviceMemoryType_DEFAULT));
        void* buffer = allocator.Alloc(64);
        if (buffer == nullptr) {
          std::_Exit(2);
        }
        // Inject a mapped state only for this allocation. The actual Session wrapper and
        // GpuBufferAllocator::Free must contain EnforceBufferUnmapped's exception.
        free_failure_buffer = static_cast<WGPUBuffer>(buffer);
        allocator.Free(buffer);
        if (free_failure_probes != 1) {
          std::_Exit(73);
        }
        free_failure_buffer = nullptr;
        if (procs.bufferGetSize(static_cast<WGPUBuffer>(buffer)) != 64) {
          std::_Exit(4);
        }
        allocator.Free(buffer);
        std::_Exit(0);
      },
      ::testing::ExitedWithCode(0), "Buffer is still mapped");
#endif
}

TEST_F(WebGpuSessionAllocatorDeathTest, PublicSessionAllocSubmitsReusedBufferClearBeforeReadback) {
#if defined(BUILD_DAWN_SHARED_LIBRARY)
  GTEST_SKIP() << "Shared Dawn calls bypass the replaceable proc table required for submission counting.";
#else
  ASSERT_EXIT(
      std::_Exit(VerifyPublicAllocSubmitsReusedBufferClear(Env(), EpDevice(), true)),
      ::testing::ExitedWithCode(0), "Ordinary Alloc submitted the reused-buffer clear before readback");
#endif
}

TEST_F(WebGpuSessionAllocatorDeathTest, PublicEnvAllocSubmitsReusedBufferClearBeforeReadback) {
#if defined(BUILD_DAWN_SHARED_LIBRARY)
  GTEST_SKIP() << "Shared Dawn calls bypass the replaceable proc table required for submission counting.";
#else
  ASSERT_EXIT(
      std::_Exit(VerifyPublicAllocSubmitsReusedBufferClear(Env(), EpDevice(), false)),
      ::testing::ExitedWithCode(0), "Ordinary Alloc submitted the reused-buffer clear before readback");
#endif
}

TEST_F(WebGpuSessionAllocatorDeathTest, PublicSessionAllocSubmitsClearAfterCancelledRun) {
#if defined(BUILD_DAWN_SHARED_LIBRARY)
  GTEST_SKIP() << "Shared Dawn calls bypass the replaceable proc table required for submission counting.";
#else
  ASSERT_EXIT(
      std::_Exit(VerifyPublicAllocSubmitsReusedBufferClear(Env(), EpDevice(), true, true)),
      ::testing::ExitedWithCode(0), "Ordinary Alloc submitted the reused-buffer clear before readback");
#endif
}

TEST_F(WebGpuSessionAllocatorDeathTest, WarmedInstanceNormalizationScratchAllocationsStayBatched) {
#if defined(BUILD_DAWN_SHARED_LIBRARY)
  GTEST_SKIP() << "Shared Dawn calls bypass the replaceable proc table required for submission counting.";
#else
  ASSERT_EXIT(
      std::_Exit(VerifyInstanceNormScratchAllocationsStayBatched(Env(), EpDevice())),
      ::testing::ExitedWithCode(0), "Warmed InstanceNormalization scratch allocations retained a single Run submission");
#endif
}
#endif

}  // namespace

TEST(WebGpuPluginSharedAllocatorRegistrationTest, UserAllocatorRegisteredBeforePluginRemainsAvailable) {
  // provider_test pre-registers the plugin; this case requires an environment without it.
  ortenv_teardown();
  auto restore_env = gsl::finally([] { ortenv_setup(); });

  Ort::MemoryInfo user_memory_info{WEBGPU_BUFFER, OrtMemoryInfoDeviceType_GPU,
                                   0, 0, OrtDeviceMemoryType_DEFAULT, 0, OrtDeviceAllocator};
  UserWebGpuAllocator user_allocator{user_memory_info};
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "webgpu_allocator_registration");

  env.RegisterAllocator(&user_allocator);

  auto allocator_before_plugin_registration = env.GetSharedAllocator(user_memory_info);
  ASSERT_EQ(static_cast<OrtAllocator*>(allocator_before_plugin_registration),
            static_cast<OrtAllocator*>(&user_allocator));

  ScopedWebGpuPluginRegistration registration(env, "webgpu_user_allocator_test");
  ASSERT_FALSE(registration.GetEpDevices().empty());

  auto allocator_after_plugin_registration = env.GetSharedAllocator(user_memory_info);
  ASSERT_NE(allocator_after_plugin_registration, nullptr);
  EXPECT_EQ(static_cast<OrtAllocator*>(allocator_after_plugin_registration),
            static_cast<OrtAllocator*>(&user_allocator));

  {
    auto allocation = allocator_after_plugin_registration.GetAllocation(256);
    ASSERT_NE(allocation.get(), nullptr);
  }

  EXPECT_EQ(user_allocator.NumAllocations(), 1u);
  EXPECT_EQ(user_allocator.NumFrees(), 1u);
}

TEST_F(WebGpuPluginSharedAllocatorTest, SharedAllocatorCreatedOnPluginRegistration) {
  auto device_memory_info = EpDevice().GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
  ASSERT_NE(device_memory_info, nullptr);

  auto allocator = Env().GetSharedAllocator(device_memory_info);
  ASSERT_NE(allocator, nullptr);
  EXPECT_TRUE(allocator.GetInfo() == device_memory_info);
  EXPECT_EQ(static_cast<OrtAllocator*>(allocator)->AllocOnStream, nullptr);
}

TEST_F(WebGpuPluginSharedAllocatorTest, SharedAllocatorAllocAndFree) {
  auto device_memory_info = EpDevice().GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
  ASSERT_NE(device_memory_info, nullptr);

  auto allocator = Env().GetSharedAllocator(device_memory_info);
  ASSERT_NE(allocator, nullptr);

  EXPECT_EQ(allocator.Alloc(0), nullptr);
  allocator.Free(nullptr);

  {
    auto allocation = allocator.GetAllocation(256);
    ASSERT_NE(allocation.get(), nullptr);
  }
}

TEST_F(WebGpuPluginSharedAllocatorTest, SharedAllocatorAndTensorOutliveSession) {
  constexpr std::array<int64_t, 1> shape{8};
  std::array<float, 8> input_data{1.0f, -2.0f, 3.5f, 4.0f, -5.25f, 6.0f, 7.75f, -8.0f};
  auto cpu_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto cpu_input = Ort::Value::CreateTensor<float>(
      cpu_memory_info, input_data.data(), input_data.size(), shape.data(), shape.size());
  auto shared_allocator = Env().GetSharedAllocator(EpDevice().GetMemoryInfo(OrtDeviceMemoryType_DEFAULT));
  ASSERT_NE(shared_allocator, nullptr);

  // Allocate before a Session exists and keep the tensor across Session destruction.
  auto device_tensor = Ort::Value::CreateTensor<float>(shared_allocator, shape.data(), shape.size());
  {
    Ort::SessionOptions options;
    options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
    options.AppendExecutionProvider_V2(Env(), {EpDevice()}, Ort::KeyValuePairs{});
    Ort::Session session(Env(), ORT_TSTR("testdata/mul_1.onnx"), options);
  }

  ASSERT_ORTSTATUS_OK(Env().CopyTensor(cpu_input, device_tensor, nullptr));
  std::array<float, 8> output_data{};
  auto cpu_output = Ort::Value::CreateTensor<float>(
      cpu_memory_info, output_data.data(), output_data.size(), shape.data(), shape.size());
  ASSERT_ORTSTATUS_OK(Env().CopyTensor(device_tensor, cpu_output, nullptr));
  EXPECT_EQ(output_data, input_data);

  auto new_device_tensor = Ort::Value::CreateTensor<float>(shared_allocator, shape.data(), shape.size());
  ASSERT_ORTSTATUS_OK(Env().CopyTensor(cpu_input, new_device_tensor, nullptr));
  output_data.fill(0.0f);
  ASSERT_ORTSTATUS_OK(Env().CopyTensor(new_device_tensor, cpu_output, nullptr));
  EXPECT_EQ(output_data, input_data);
  EXPECT_EQ(static_cast<OrtAllocator*>(shared_allocator)->AllocOnStream, nullptr);
}

TEST_F(WebGpuPluginSharedAllocatorTest, SharedAllocatorSubmitsReusedBufferClearWithoutSession) {
  constexpr std::array<int64_t, 1> shape{8};
  std::array<float, 8> nonzero_data;
  nonzero_data.fill(7.0f);
  auto cpu_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto cpu_input = Ort::Value::CreateTensor<float>(
      cpu_memory_info, nonzero_data.data(), nonzero_data.size(), shape.data(), shape.size());
  auto allocator = Env().GetSharedAllocator(EpDevice().GetMemoryInfo(OrtDeviceMemoryType_DEFAULT));
  ASSERT_NE(allocator, nullptr);

  void* dirty_buffer = nullptr;
  {
    auto tensor = Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
    dirty_buffer = tensor.GetTensorMutableRawData();
    ASSERT_ORTSTATUS_OK(Env().CopyTensor(cpu_input, tensor, nullptr));
  }

  auto reused_tensor = Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
  ASSERT_EQ(reused_tensor.GetTensorMutableRawData(), dirty_buffer);
  std::array<float, 8> output_data;
  output_data.fill(-1.0f);
  auto cpu_output = Ort::Value::CreateTensor<float>(
      cpu_memory_info, output_data.data(), output_data.size(), shape.data(), shape.size());
  // Env copies use a different recording: readback must not be what submits the allocator's clear.
  ASSERT_ORTSTATUS_OK(Env().CopyTensor(reused_tensor, cpu_output, nullptr));
  const std::array<float, 8> expected{};
  EXPECT_EQ(output_data, expected);
}

// TODO: Re-enable when WebGPU EP supports distinct allocators for multiple OrtEpDevices. Currently every device uses
// the same allocator memory info with device_id 0, so the environment intentionally resolves them to one allocator.
TEST_F(WebGpuPluginSharedAllocatorTest, DISABLED_SharedAllocatorsAreDistinctPerDevice) {
  std::set<int> device_ids;
  std::set<OrtAllocator*> allocators;

  for (const auto& ep_device : EpDevices()) {
    auto memory_info = ep_device.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
    ASSERT_NE(memory_info, nullptr);
    EXPECT_TRUE(device_ids.insert(memory_info.GetDeviceId()).second);

    auto allocator = Env().GetSharedAllocator(memory_info);
    ASSERT_NE(allocator, nullptr);
    EXPECT_TRUE(allocators.insert(static_cast<OrtAllocator*>(allocator)).second);

    auto allocation = allocator.GetAllocation(256);
    ASSERT_NE(allocation.get(), nullptr);
  }
}

TEST_F(WebGpuPluginSharedAllocatorTest, CreateSharedAllocatorIsReturnedByGet) {
  auto device_memory_info = EpDevice().GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
  ASSERT_NE(device_memory_info, nullptr);

  Ort::KeyValuePairs allocator_options;
  auto created_allocator = Env().CreateSharedAllocator(
      EpDevice(), OrtDeviceMemoryType_DEFAULT, OrtDeviceAllocator, allocator_options);
  ASSERT_NE(created_allocator, nullptr);

  auto fetched_allocator = Env().GetSharedAllocator(device_memory_info);
  ASSERT_NE(fetched_allocator, nullptr);
  EXPECT_EQ(static_cast<OrtAllocator*>(created_allocator), static_cast<OrtAllocator*>(fetched_allocator));
}

TEST_F(WebGpuPluginSharedAllocatorTest, DeviceTensorDataRoundTripsWithSharedAndSessionAllocators) {
  constexpr std::array<int64_t, 1> shape{8};
  std::array<float, 8> input_data{1.0f, -2.0f, 3.5f, 4.0f, -5.25f, 6.0f, 7.75f, -8.0f};
  auto cpu_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  const auto round_trip = [&](auto& allocator, const char* allocator_name, int device_id) {
    SCOPED_TRACE(::testing::Message() << allocator_name << " device " << device_id);
    std::array<float, 8> output_data{};
    auto cpu_input = Ort::Value::CreateTensor<float>(
        cpu_memory_info, input_data.data(), input_data.size(), shape.data(), shape.size());
    auto device_tensor = Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
    auto cpu_output = Ort::Value::CreateTensor<float>(
        cpu_memory_info, output_data.data(), output_data.size(), shape.data(), shape.size());

    ASSERT_ORTSTATUS_OK(Env().CopyTensor(cpu_input, device_tensor, nullptr));
    ASSERT_ORTSTATUS_OK(Env().CopyTensor(device_tensor, cpu_output, nullptr));
    EXPECT_EQ(output_data, input_data);
  };

  for (const auto& ep_device : EpDevices()) {
    auto device_memory_info = ep_device.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
    ASSERT_NE(device_memory_info, nullptr);
    const int device_id = device_memory_info.GetDeviceId();

    Ort::KeyValuePairs allocator_options;
    auto shared_allocator = Env().CreateSharedAllocator(
        ep_device, OrtDeviceMemoryType_DEFAULT, OrtDeviceAllocator, allocator_options);
    ASSERT_NE(shared_allocator, nullptr);

    Ort::SessionOptions session_options;
    Ort::KeyValuePairs ep_options;
    session_options.AppendExecutionProvider_V2(Env(), {ep_device}, ep_options);
    Ort::Session session(Env(), ORT_TSTR("testdata/mul_1.onnx"), session_options);
    Ort::Allocator session_allocator(session, device_memory_info);

    round_trip(shared_allocator, "shared allocator", device_id);
    round_trip(session_allocator, "session allocator", device_id);
  }
}

// BufferManager::MemCpy rejects a self-copy with ORT_ENFORCE, which throws rather than returning a
// Status. CopyTensorsImpl is a noexcept C ABI callback, so before the throw was converted to an
// OrtStatus this ended the process instead of reporting the misuse.
TEST_F(WebGpuPluginSharedAllocatorTest, DeviceTensorSelfCopyIsReportedInsteadOfTerminating) {
  constexpr std::array<int64_t, 1> shape{8};
  std::array<float, 8> input_data{1.0f, -2.0f, 3.5f, 4.0f, -5.25f, 6.0f, 7.75f, -8.0f};
  auto cpu_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  for (const auto& ep_device : EpDevices()) {
    auto device_memory_info = ep_device.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
    ASSERT_NE(device_memory_info, nullptr);
    SCOPED_TRACE(::testing::Message() << "device " << device_memory_info.GetDeviceId());

    Ort::KeyValuePairs allocator_options;
    auto shared_allocator = Env().CreateSharedAllocator(
        ep_device, OrtDeviceMemoryType_DEFAULT, OrtDeviceAllocator, allocator_options);
    ASSERT_NE(shared_allocator, nullptr);

    Ort::SessionOptions session_options;
    Ort::KeyValuePairs ep_options;
    session_options.AppendExecutionProvider_V2(Env(), {ep_device}, ep_options);
    Ort::Session session(Env(), ORT_TSTR("testdata/mul_1.onnx"), session_options);

    auto device_tensor = Ort::Value::CreateTensor<float>(shared_allocator, shape.data(), shape.size());

    // Source and destination resolve to the same WGPUBuffer, which MemCpy refuses.
    Ort::Status self_copy = Env().CopyTensor(device_tensor, device_tensor, nullptr);
    ASSERT_FALSE(self_copy.IsOK());
    EXPECT_THAT(self_copy.GetErrorMessage(), ::testing::HasSubstr("must be different"));

    // The misuse was reported, not fatal: the same tensor still round trips.
    std::array<float, 8> output_data{};
    auto cpu_input = Ort::Value::CreateTensor<float>(
        cpu_memory_info, input_data.data(), input_data.size(), shape.data(), shape.size());
    auto cpu_output = Ort::Value::CreateTensor<float>(
        cpu_memory_info, output_data.data(), output_data.size(), shape.data(), shape.size());

    ASSERT_ORTSTATUS_OK(Env().CopyTensor(cpu_input, device_tensor, nullptr));
    ASSERT_ORTSTATUS_OK(Env().CopyTensor(device_tensor, cpu_output, nullptr));
    EXPECT_EQ(output_data, input_data);
  }
}

// Manual Task Manager test. This intentionally reserves substantial GPU memory and runs for an extended period.
// Enable explicitly with --gtest_also_run_disabled_tests and this test's full name.
TEST_F(WebGpuPluginSharedAllocatorTest, DISABLED_ManualPerDeviceMemoryAndComputeLoad) {
  const auto ep_devices = EpDevices();
  if (ep_devices.size() < 2) {
    GTEST_SKIP() << "This manual test requires at least two WebGPU devices.";
  }

  const size_t allocation_mb = ParseEnvironmentVariableWithDefault<size_t>(
      "ORT_WEBGPU_MANUAL_ALLOCATION_MB", 512);
  const int64_t matrix_dimension = ParseEnvironmentVariableWithDefault<int64_t>(
      "ORT_WEBGPU_MANUAL_MATMUL_DIM", 2048);
  const size_t matmul_depth = ParseEnvironmentVariableWithDefault<size_t>(
      "ORT_WEBGPU_MANUAL_MATMUL_DEPTH", 8);
  const int run_seconds = ParseEnvironmentVariableWithDefault<int>(
      "ORT_WEBGPU_MANUAL_RUN_SECONDS", 15);
  const int pause_seconds = ParseEnvironmentVariableWithDefault<int>(
      "ORT_WEBGPU_MANUAL_PAUSE_SECONDS", 3);

  ASSERT_GT(allocation_mb, 0u);
  ASSERT_LE(allocation_mb, 4096u);
  ASSERT_GT(matrix_dimension, 0);
  ASSERT_LE(matrix_dimension, 4096);
  ASSERT_GT(matmul_depth, 0u);
  ASSERT_LE(matmul_depth, 32u);
  ASSERT_GT(run_seconds, 0);
  ASSERT_LE(run_seconds, 300);
  ASSERT_GE(pause_seconds, 0);
  ASSERT_LE(pause_seconds, 60);

  const size_t allocation_size = allocation_mb * 1024 * 1024;
  std::vector<Ort::MemoryAllocation> held_allocations;
  held_allocations.reserve(2);

  std::cout << "Reserving " << allocation_mb << " MiB on each WebGPU device before session creation.\n";
  for (size_t i = 0; i < 2; ++i) {
    const auto memory_info = ep_devices[i].GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
    ASSERT_NE(memory_info, nullptr);
    auto allocator = Env().GetSharedAllocator(memory_info);
    ASSERT_NE(allocator, nullptr);

    held_allocations.emplace_back(allocator.GetAllocation(allocation_size));
    ASSERT_NE(held_allocations.back().get(), nullptr);

    const auto hardware_device = ep_devices[i].Device();
    std::cout << "Reserved device " << memory_info.GetDeviceId()
              << " vendor=" << hardware_device.Vendor()
              << " vendor_id=0x" << std::hex << hardware_device.VendorId() << std::dec
              << " allocation=" << allocation_mb << " MiB\n";
  }

  std::cout << "Both allocations are live. Waiting " << pause_seconds
            << " seconds before creating sessions.\n";
  std::this_thread::sleep_for(std::chrono::seconds{pause_seconds});

  const std::string model_bytes = BuildMatMulLoadModelBytes(matrix_dimension, matmul_depth);
  const size_t element_count = static_cast<size_t>(matrix_dimension * matrix_dimension);
  std::vector<float> input_data(element_count, 1.0f / static_cast<float>(matrix_dimension));
  const std::array<int64_t, 2> input_shape{matrix_dimension, matrix_dimension};
  auto cpu_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  const std::array<const char*, 2> input_names{"A", "B"};
  const std::array<const char*, 1> output_names{"Y"};

  for (size_t i = 0; i < 2; ++i) {
    const auto memory_info = ep_devices[i].GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
    ASSERT_NE(memory_info, nullptr);
    const auto hardware_device = ep_devices[i].Device();

    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    session_options.AppendExecutionProvider_V2(Env(), {ep_devices[i]}, Ort::KeyValuePairs{});
    Ort::Session session(Env(), model_bytes.data(), model_bytes.size(), session_options);

    std::array<Ort::Value, 2> inputs{
        Ort::Value::CreateTensor<float>(cpu_memory_info, input_data.data(), input_data.size(),
                                        input_shape.data(), input_shape.size()),
        Ort::Value::CreateTensor<float>(cpu_memory_info, input_data.data(), input_data.size(),
                                        input_shape.data(), input_shape.size())};

    std::cout << "Running WebGPU device " << memory_info.GetDeviceId()
              << " vendor=" << hardware_device.Vendor()
              << " for at least " << run_seconds << " seconds. Observe this GPU now.\n";

    size_t iterations = 0;
    const auto end_time = std::chrono::steady_clock::now() + std::chrono::seconds{run_seconds};
    do {
      auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(), inputs.size(),
                                 output_names.data(), output_names.size());
      ASSERT_EQ(outputs.size(), 1u);
      ++iterations;
    } while (std::chrono::steady_clock::now() < end_time);

    std::cout << "Completed device " << memory_info.GetDeviceId() << " after " << iterations
              << " iterations. Waiting " << pause_seconds << " seconds before the next device.\n";
    std::this_thread::sleep_for(std::chrono::seconds{pause_seconds});
  }

  std::cout << "Manual multi-GPU load completed; both " << allocation_mb
            << " MiB allocations remained live for the full test.\n";
}

#endif  // defined(ORT_UNIT_TEST_HAS_WEBGPU_PLUGIN_EP)

}  // namespace test
}  // namespace onnxruntime
