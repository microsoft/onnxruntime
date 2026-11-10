// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(USE_CUDA)

#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <tuple>
#include <type_traits>
#include <vector>

#include <cuda_runtime_api.h>
#include "gtest/gtest.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/util/include/stateful_execution_test_utils.h"

extern std::unique_ptr<Ort::Env> ort_env;
extern "C" int StatefulApiCDeviceConsumerTest(OrtSession*, OrtAllocator*, const OrtValue*, const OrtValue*,
                                              OrtKVCacheView**);
extern "C" int StatefulApiCGreedyDeviceConsumerTest(OrtSession*, const OrtValue*, int64_t*);

namespace onnxruntime::test {
namespace {

void Cuda(cudaError_t result) {
  ASSERT_EQ(result, cudaSuccess) << cudaGetErrorString(result);
}

OrtStatefulProviderSelection Selection(OrtStatefulProviderMode mode = ORT_STATEFUL_PROVIDER_REQUIRE) {
  return {sizeof(OrtStatefulProviderSelection), mode, "CUDAExecutionProvider", OrtMemoryInfoDeviceType_GPU, 0};
}

#if defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP)

class CudaStatefulTest : public testing::Test {
 protected:
  void SetUp() override {
    Cuda(cudaGetDevice(&previous_device_));
    Cuda(cudaSetDevice(0));
    for (auto device : ort_env->GetEpDevices()) {
      if (std::strcmp(device.EpName(), "CUDAExecutionProvider") == 0 &&
          device.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT).GetDeviceId() == 0) {
        device_ = device;
        break;
      }
    }
    ASSERT_NE(static_cast<const OrtEpDevice*>(device_), nullptr);
    session_ = Session();
    allocator_session_ = Session();
    allocator_ = Ort::Allocator(allocator_session_, device_.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT));
  }

  void TearDown() override {
    allocator_ = Ort::Allocator(nullptr);
    session_ = Ort::Session(nullptr);
    allocator_session_ = Ort::Session(nullptr);
    Cuda(cudaSetDevice(previous_device_));
  }

  Ort::Session Session(cudaStream_t stream = nullptr) {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1).SetInterOpNumThreads(1);
    options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
    if (stream) {
      options.AddConfigEntry("ep.cuda.user_compute_stream",
                             std::to_string(reinterpret_cast<uintptr_t>(stream)).c_str());
    }
    options.AppendExecutionProvider_V2(*ort_env, {device_}, Ort::KeyValuePairs{});
    auto model = stateful::MakeFeedbackModel().SerializeAsString();
    return Ort::Session(*ort_env, model.data(), model.size(), options);
  }

  Ort::KVCacheOptions Options(const std::vector<int64_t>& shape, int axis,
                              ONNXTensorElementDataType type, int64_t capacity = 0) {
    const OrtStatefulKVCacheSpec spec{"key", type, shape.data(), shape.size(), axis, capacity};
    const OrtStatefulKVCacheConfig config{sizeof(config), &spec, 1};
    return Ort::KVCacheOptions(config);
  }

  Ort::Value Value(const std::vector<int64_t>& shape, ONNXTensorElementDataType type,
                   bool gpu, uint8_t seed = 1) {
    Ort::AllocatorWithDefaultOptions cpu;
    auto value = Ort::Value::CreateTensor(gpu ? static_cast<OrtAllocator*>(allocator_) : cpu,
                                          shape.data(), shape.size(), type);
    std::vector<uint8_t> bytes(value.GetTensorSizeInBytes());
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<uint8_t>((i + seed) % 251);
    if (!bytes.empty()) {
      if (gpu) {
        Cuda(cudaMemcpy(value.GetTensorMutableData<void>(), bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
      } else {
        std::memcpy(value.GetTensorMutableData<void>(), bytes.data(), bytes.size());
      }
    }
    return value;
  }

  static std::vector<uint8_t> Read(const Ort::StatefulValue& value) {
    std::vector<uint8_t> bytes(value.GetTensorSizeInBytes());
    if (!bytes.empty()) {
      Cuda(cudaMemcpy(bytes.data(), value.GetTensorRawData(), bytes.size(), cudaMemcpyDeviceToHost));
    }
    return bytes;
  }

  Ort::Value Scores(const std::vector<int64_t>& shape, const std::vector<float>& scores,
                    ONNXTensorElementDataType type, bool gpu) {
    Ort::AllocatorWithDefaultOptions cpu;
    auto host = Ort::Value::CreateTensor(cpu, shape.data(), shape.size(), type);
    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      std::copy(scores.begin(), scores.end(), host.GetTensorMutableData<float>());
    } else {
      std::transform(scores.begin(), scores.end(), host.GetTensorMutableData<Ort::Float16_t>(),
                     [](float value) { return Ort::Float16_t(value); });
    }
    if (!gpu) return host;
    auto value = Ort::Value::CreateTensor(allocator_, shape.data(), shape.size(), type);
    if (!scores.empty()) {
      Cuda(cudaMemcpy(value.GetTensorMutableData<void>(), host.GetTensorRawData(),
                      host.GetTensorSizeInBytes(), cudaMemcpyHostToDevice));
    }
    return value;
  }

  void CheckGreedy(const std::vector<int64_t>& shape, const std::vector<float>& scores,
                   ONNXTensorElementDataType type, std::optional<int64_t> position = std::nullopt) {
    const OrtStatefulSamplerConfig config{sizeof(config), ORT_STATEFUL_GREEDY,
                                          position.has_value(), position.value_or(0)};
    Ort::SamplerOptions gpu(config), cpu(config);
    gpu.SetProvider(Selection(), session_);
    auto input = Scores(shape, scores, type, true), reference = Scores(shape, scores, type, false);
    if (std::any_of(scores.begin(), scores.end(), [](float value) { return std::isnan(value); })) {
      const auto& api = Ort::GetStatefulApi();
      int64_t untouched = 912;
      Ort::Status status(api.SamplerSample(gpu, input, &untouched));
      EXPECT_FALSE(status.IsOK());
      EXPECT_EQ(untouched, 912);
      EXPECT_THROW(cpu.Sample(reference), Ort::Exception);
      EXPECT_EQ(gpu.GetDispatchInfo().failures, 1u);
    } else {
      EXPECT_EQ(gpu.Sample(input), cpu.Sample(reference));
    }
    std::vector<uint8_t> unchanged(input.GetTensorSizeInBytes());
    Cuda(cudaMemcpy(unchanged.data(), input.GetTensorRawData(), unchanged.size(), cudaMemcpyDeviceToHost));
    EXPECT_EQ(std::memcmp(unchanged.data(), reference.GetTensorRawData(), unchanged.size()), 0);
    const auto info = gpu.GetDispatchInfo();
    EXPECT_EQ(info.provider_calls, 1u);
    EXPECT_EQ(info.cpu_calls, 0u);
    EXPECT_EQ(info.cpu_delegations, 0u);
  }

  static void Check(const Ort::KVCacheManager& gpu, const Ort::KVCacheManager& cpu,
                    cudaStream_t stream = nullptr) {
    // Check completion before any readback can implicitly synchronize.
    Cuda(cudaStreamQuery(stream));
    const auto actual = gpu.GetView("key").GetInfo();
    const auto expected = cpu.GetView("key").GetInfo();
    EXPECT_EQ(actual.length, expected.length);
    EXPECT_EQ(actual.capacity, expected.capacity);
    EXPECT_EQ(actual.value.GetTensorMemoryInfo().GetDeviceType(), OrtMemoryInfoDeviceType_GPU);
    EXPECT_EQ(actual.value.GetTensorMemoryInfo().GetDeviceId(), 0);
    const auto bytes = Read(actual.value);
    EXPECT_EQ(bytes.size(), expected.value.GetTensorSizeInBytes());
    if (!bytes.empty()) {
      EXPECT_EQ(std::memcmp(bytes.data(), expected.value.GetTensorRawData(), bytes.size()), 0);
      cudaPointerAttributes attributes{};
      Cuda(cudaPointerGetAttributes(&attributes, actual.value.GetTensorRawData()));
      EXPECT_EQ(attributes.type, cudaMemoryTypeDevice);
      EXPECT_EQ(attributes.device, 0);
    }
    const auto a = gpu.GetStats(), b = cpu.GetStats();
    EXPECT_EQ(a.allocations, b.allocations);
    EXPECT_EQ(a.bytes_copied, b.bytes_copied);
    EXPECT_EQ(a.shared_updates, b.shared_updates);
  }

  Ort::ConstEpDevice device_{nullptr};
  Ort::Session allocator_session_{nullptr};
  Ort::Session session_{nullptr};
  Ort::Allocator allocator_{nullptr};
  int previous_device_ = 0;
};

class CudaStatefulLayoutTest : public CudaStatefulTest,
                               public testing::WithParamInterface<std::tuple<int, ONNXTensorElementDataType>> {};

TEST_P(CudaStatefulLayoutTest, CompactOperationsAndImmutableSnapshots) {
  const auto [axis, type] = GetParam();
  std::vector<int64_t> shape{2, 3, 2, 4};
  shape[axis] = 0;
  auto options = Options(shape, axis, type, axis % 2 ? 3 : 0);
  Ort::AllocatorWithDefaultOptions cpu_allocator;
  Ort::KVCacheManager cpu(options, cpu_allocator);
  options.SetProvider(Selection(), session_);
  Ort::KVCacheManager gpu(options, allocator_);
  Check(gpu, cpu);
  const auto caps = Ort::QueryStatefulProviderCapabilities(session_, "CUDAExecutionProvider");
  EXPECT_EQ(caps.operations, 127u);
  EXPECT_EQ(caps.sampler_element_types, (uint64_t{1} << 1) | (uint64_t{1} << 10));
  EXPECT_EQ(caps.cache_element_types, (uint64_t{1} << 1) | (uint64_t{1} << 10) | (uint64_t{1} << 16));
  EXPECT_EQ(caps.memory_types, uint32_t{1} << OrtMemoryInfoDeviceType_GPU);
  EXPECT_EQ(caps.device_id, 0);

  auto update = [&](int64_t length, OrtKVCacheUpdateKind kind) {
    shape[axis] = length;
    auto source = Value(shape, type, true);
    auto reference = Value(shape, type, false);
    gpu.Update("key", source, kind);
    cpu.Update("key", reference, kind);
    if (kind == ORT_KV_CACHE_SHARE) {
      EXPECT_EQ(gpu.GetView("key").GetInfo().value.GetTensorRawData(), source.GetTensorRawData());
    }
    Check(gpu, cpu);
  };
  update(0, ORT_KV_CACHE_COPY);
  gpu.Reserve("key", 4);
  cpu.Reserve("key", 4);
  Check(gpu, cpu);
  update(1, ORT_KV_CACHE_SHARE);
  auto snapshot = gpu.GetView("key").GetInfo();
  const auto saved = Read(snapshot.value);
  update(2, ORT_KV_CACHE_COPY);
  update(1, ORT_KV_CACHE_APPEND);
  gpu.Reserve("key", 8);
  cpu.Reserve("key", 8);
  Check(gpu, cpu);
  const auto calls = gpu.GetDispatchInfo(ORT_STATEFUL_CACHE_APPEND).provider_calls;
  update(0, ORT_KV_CACHE_APPEND);
  EXPECT_EQ(gpu.GetDispatchInfo(ORT_STATEFUL_CACHE_APPEND).provider_calls, calls);
  gpu.Reserve("key", 3);
  cpu.Reserve("key", 3);
  gpu.Reset();
  cpu.Reset();
  gpu.Reset();
  cpu.Reset();
  Check(gpu, cpu);
  update(1, ORT_KV_CACHE_APPEND);
  update(2, ORT_KV_CACHE_COPY);
  EXPECT_EQ(Read(snapshot.value), saved);
  for (int operation = 0; operation < 6; ++operation) {
    const auto dispatch = gpu.GetDispatchInfo(static_cast<OrtStatefulProviderOperation>(operation));
    EXPECT_GT(dispatch.provider_calls, 0u);
    EXPECT_EQ(dispatch.origin, ORT_STATEFUL_DISPATCH_PROVIDER);
    EXPECT_EQ(dispatch.cpu_calls, 0u);
    EXPECT_EQ(dispatch.cpu_delegations, 0u);
  }
}

INSTANTIATE_TEST_SUITE_P(AllLayouts, CudaStatefulLayoutTest,
                         testing::Combine(testing::Range(0, 4),
                                          testing::Values(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
                                                          ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16)));

struct TestAllocator : OrtAllocator {
  OrtAllocator* backing;
  bool fail = false;
  void* reserved = nullptr;
  size_t reserved_bytes = 0;
  bool reserved_used = false;
  size_t allocation_calls = 0;
  std::promise<void>* allocated = nullptr;
  explicit TestAllocator(OrtAllocator* value) : OrtAllocator{}, backing(value) {
    version = ORT_API_VERSION;
    Alloc = [](OrtAllocator* self, size_t bytes) -> void* {
      auto& allocator = *static_cast<TestAllocator*>(self);
      ++allocator.allocation_calls;
      if (allocator.fail) return nullptr;
      if (allocator.reserved && bytes != 0) {
        if (allocator.reserved_used || bytes > allocator.reserved_bytes) return nullptr;
        allocator.reserved_used = true;
        allocator.allocated->set_value();
        return allocator.reserved;
      }
      return allocator.backing->Alloc(allocator.backing, bytes);
    };
    Free = [](OrtAllocator* self, void* pointer) {
      auto& allocator = *static_cast<TestAllocator*>(self);
      if (pointer != allocator.reserved) allocator.backing->Free(allocator.backing, pointer);
    };
    Info = [](const OrtAllocator* self) {
      auto* backing = static_cast<const TestAllocator*>(self)->backing;
      return backing->Info(backing);
    };
  }
};

TEST_F(CudaStatefulTest, ExplicitPoliciesErrorsFailureRecoveryAndDeviceRestoration) {
  const auto type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  auto options = Options({2, 0, 2}, 1, type);
  EXPECT_THROW(Ort::KVCacheManager(options, allocator_), Ort::Exception);
  options.SetProvider(Selection(ORT_STATEFUL_PROVIDER_PREFER), session_);
  TestAllocator allocator(allocator_);
  Ort::KVCacheManager manager(options, &allocator);
  auto source = Value({2, 1, 2}, type, true);
  auto host = Value({2, 1, 2}, type, false);
  EXPECT_THROW(manager.Update("key", host, ORT_KV_CACHE_COPY), Ort::Exception);
  auto wrong_type = Value({2, 1, 2}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, true);
  EXPECT_THROW(manager.Update("key", wrong_type, ORT_KV_CACHE_COPY), Ort::Exception);
  const auto memory = source.GetTensorMemoryInfo();
  auto borrowed = Ort::Value::CreateTensor(memory, source.GetTensorMutableData<void>(),
                                           source.GetTensorSizeInBytes(), std::vector<int64_t>{2, 1, 2}.data(),
                                           3, type);
  EXPECT_THROW(manager.Update("key", borrowed, ORT_KV_CACHE_SHARE), Ort::Exception);
  auto bad_shape = Value({3, 1, 2}, type, true);
  EXPECT_THROW(manager.Update("key", bad_shape, ORT_KV_CACHE_COPY), Ort::Exception);
  Ort::MemoryInfo other("Cuda", OrtDeviceAllocator, 1, OrtMemTypeDefault);
  const int64_t dimensions[]{2, 1, 2};
  auto other_device = Ort::Value::CreateTensor(other, source.GetTensorMutableData<void>(),
                                               source.GetTensorSizeInBytes(), dimensions, 3, type);
  EXPECT_THROW(manager.Update("key", other_device, ORT_KV_CACHE_COPY), Ort::Exception);
  EXPECT_FALSE(manager.RequiresReset());
  int device_count = 0;
  Cuda(cudaGetDeviceCount(&device_count));
  const int restored_device = device_count > 1 ? 1 : 0;
  RecordProperty("restored_device", restored_device);
  Cuda(cudaSetDevice(restored_device));
  manager.Update("key", source, ORT_KV_CACHE_COPY);
  int current = -1;
  Cuda(cudaGetDevice(&current));
  EXPECT_EQ(current, restored_device);
  Cuda(cudaSetDevice(0));
  auto retained = manager.GetView("key").GetInfo();
  const auto bytes = Read(retained.value);
  allocator.fail = true;
  Cuda(cudaSetDevice(restored_device));
  Ort::Status allocation(Ort::GetStatefulApi().KVCacheManagerReserve(manager, "key", 4));
  EXPECT_FALSE(allocation.IsOK());
  if (!allocation.IsOK()) {
    EXPECT_NE(allocation.GetErrorMessage().find("CUDA KV cache allocation failed"), std::string::npos);
  }
  Cuda(cudaGetDevice(&current));
  EXPECT_EQ(current, restored_device);
  Cuda(cudaSetDevice(0));
  EXPECT_TRUE(manager.RequiresReset());
  EXPECT_THROW(manager.GetView("key"), Ort::Exception);
  EXPECT_THROW(manager.Update("key", source, ORT_KV_CACHE_APPEND), Ort::Exception);
  EXPECT_EQ(manager.GetDispatchInfo(ORT_STATEFUL_CACHE_RESERVE).failures, 1u);
  EXPECT_EQ(manager.GetDispatchInfo(ORT_STATEFUL_CACHE_RESERVE).cpu_calls, 0u);
  EXPECT_EQ(Read(retained.value), bytes);
  manager.Reset();
  EXPECT_FALSE(manager.RequiresReset());
  allocator.fail = false;
  manager.Update("key", source, ORT_KV_CACHE_APPEND);
  EXPECT_THROW(manager.Update("key", Value({2, 0, 2}, type, true), ORT_KV_CACHE_SHARE), Ort::Exception);
  EXPECT_THROW(Options({2, 0, 2}, 1, type, std::numeric_limits<int64_t>::max()), Ort::Exception);
  const auto allocation_calls = allocator.allocation_calls;
  Ort::Status overflow(Ort::GetStatefulApi().KVCacheManagerReserve(
      manager, "key", std::numeric_limits<int64_t>::max()));
  EXPECT_FALSE(overflow.IsOK());
  if (!overflow.IsOK()) {
    EXPECT_NE(overflow.GetErrorMessage().find("Integer overflow"), std::string::npos);
  }
  EXPECT_EQ(allocator.allocation_calls, allocation_calls);
  Cuda(cudaPeekAtLastError());
  EXPECT_TRUE(manager.RequiresReset());
  manager.Reset();
  const OrtStatefulSamplerConfig config{sizeof(config), ORT_STATEFUL_GREEDY, 0, 0};
  Ort::SamplerOptions sampler(config);
  sampler.SetProvider(Selection(), session_);
  EXPECT_THROW(sampler.Sample(host), Ort::Exception);
  auto fallback = Selection(ORT_STATEFUL_PROVIDER_PREFER);
  fallback.device_type = OrtMemoryInfoDeviceType_CPU;
  sampler.SetProvider(fallback, session_);
  auto logits = Value({1, 3}, type, false);
  EXPECT_EQ(sampler.Sample(logits), 2);
  EXPECT_EQ(sampler.GetDispatchInfo().origin, ORT_STATEFUL_DISPATCH_CPU_FALLBACK);
  EXPECT_EQ(sampler.GetDispatchInfo().provider_calls, 0u);
}

TEST_F(CudaStatefulTest, NondefaultStreamCompletionBeforeReadback) {
  cudaStream_t stream = nullptr;
  Cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  auto release = [](cudaStream_t handle) { Cuda(cudaStreamDestroy(handle)); };
  std::unique_ptr<std::remove_pointer_t<cudaStream_t>, decltype(release)> stream_owner(stream, release);
  {
    auto session = Session(stream);
    auto options = Options({2, 0, 2}, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    options.SetProvider(Selection(), session);
    auto source = Value({2, 1, 2}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, true);
    auto reserved = allocator_.GetAllocation(4096);
    std::promise<void> allocated;
    TestAllocator allocator(allocator_);
    allocator.reserved = reserved.get();
    allocator.reserved_bytes = reserved.size();
    allocator.allocated = &allocated;
    Ort::KVCacheManager manager(options, &allocator);
    struct Gate {
      std::promise<void> entered;
      std::promise<void> release;
    } gate;
    Cuda(cudaLaunchHostFunc(stream, [](void* pointer) {
      auto& gate = *static_cast<Gate*>(pointer);
      auto released = gate.release.get_future();
      gate.entered.set_value();
      released.wait(); }, &gate));
    EXPECT_EQ(gate.entered.get_future().wait_for(std::chrono::seconds(30)), std::future_status::ready);
    auto finished = std::async(std::launch::async, [&] {
      manager.Update("key", source, ORT_KV_CACHE_COPY);
      return cudaStreamQuery(stream);
    });
    // The prepared allocator cannot hide missing copy completion behind cudaMalloc.
    EXPECT_EQ(allocated.get_future().wait_for(std::chrono::seconds(30)), std::future_status::ready);
    EXPECT_EQ(finished.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);
    gate.release.set_value();
    EXPECT_EQ(finished.get(), cudaSuccess);
    Cuda(cudaStreamQuery(stream));
    const auto bytes = Read(manager.GetView("key").GetInfo().value);
    EXPECT_EQ(bytes.size(), source.GetTensorSizeInBytes());
  }
}

TEST_F(CudaStatefulTest, CConsumerAndRetainedValuesAfterSessionRelease) {
  auto delta = Value({2, 1, 2}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, true);
  auto full = Value({2, 2, 2}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, true);
  OrtKVCacheView* raw_view = nullptr;
  ASSERT_EQ(StatefulApiCDeviceConsumerTest(session_, allocator_, delta, full, &raw_view), 0);
  Ort::KVCacheView view(raw_view);
  auto value = view.GetInfo().value;
  EXPECT_EQ(value.GetTensorRawData(), full.GetTensorRawData());
  const auto bytes = Read(value);
  session_ = Ort::Session(nullptr);
  delta = Ort::Value(nullptr);
  full = Ort::Value(nullptr);
  view = Ort::KVCacheView(nullptr);
  EXPECT_EQ(Read(value), bytes);
}

TEST_F(CudaStatefulTest, CallerAllocatorAndManagerOutliveBackendSession) {
  auto options = Options({2, 0, 2}, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  options.SetProvider(Selection(), session_);
  Ort::KVCacheManager manager(options, allocator_);
  auto source = Value({2, 1, 2}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, true);
  options = Ort::KVCacheOptions(nullptr);
  session_ = Ort::Session(nullptr);
  manager.Update("key", source, ORT_KV_CACHE_COPY);
  manager.Update("key", source, ORT_KV_CACHE_SHARE);
  manager.Update("key", source, ORT_KV_CACHE_SHARE);
  auto value = manager.GetView("key").GetInfo().value;
  const auto bytes = Read(value);
  manager.Reserve("key", 4);
  manager.Reset();
  manager = Ort::KVCacheManager(nullptr);
  EXPECT_EQ(Read(value), bytes);
}

TEST_F(CudaStatefulTest, EachSelectedDeviceUsesItsOwnAllocatorAndRestoresCallerDevice) {
  int tested = 0;
  for (auto device : ort_env->GetEpDevices()) {
    if (std::strcmp(device.EpName(), "CUDAExecutionProvider") != 0) continue;
    const auto memory = device.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
    const int id = memory.GetDeviceId();
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1).SetInterOpNumThreads(1);
    session_options.AppendExecutionProvider_V2(*ort_env, {device}, Ort::KeyValuePairs{});
    const auto model = stateful::MakeFeedbackModel().SerializeAsString();
    Ort::Session session(*ort_env, model.data(), model.size(), session_options);
    Ort::Allocator allocator(session, memory);
    const auto caps = Ort::QueryStatefulProviderCapabilities(session, "CUDAExecutionProvider");
    EXPECT_EQ(caps.device_id, id);
    auto choice = Selection();
    choice.device_id = id;
    auto options = Options({2, 0, 2}, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    options.SetProvider(choice, session);
    Ort::KVCacheManager manager(options, allocator);
    const int64_t shape[]{2, 1, 2};
    Cuda(cudaSetDevice(id));
    auto source = Ort::Value::CreateTensor<float>(allocator, shape, 3);
    const std::array<float, 4> input{1, 2, 3, 4};
    Cuda(cudaMemcpy(source.GetTensorMutableData<float>(), input.data(), sizeof(input), cudaMemcpyHostToDevice));
    Cuda(cudaSetDevice(0));
    manager.Update("key", source, ORT_KV_CACHE_COPY);
    int current = -1;
    Cuda(cudaGetDevice(&current));
    EXPECT_EQ(current, 0);
    auto value = manager.GetView("key").GetInfo().value;
    EXPECT_EQ(value.GetTensorMemoryInfo().GetDeviceId(), id);
    cudaPointerAttributes attributes{};
    Cuda(cudaPointerGetAttributes(&attributes, value.GetTensorRawData()));
    EXPECT_EQ(attributes.device, id);
    Cuda(cudaSetDevice(id));
    std::array<float, 4> output{};
    Cuda(cudaMemcpy(output.data(), value.GetTensorRawData(), sizeof(output), cudaMemcpyDeviceToHost));
    EXPECT_EQ(output, input);
    const int64_t logits_shape[]{1, 3};
    auto logits = Ort::Value::CreateTensor<float>(allocator, logits_shape, 2);
    const std::array<float, 3> scores{1, 2, 9};
    Cuda(cudaMemcpy(logits.GetTensorMutableData<float>(), scores.data(), sizeof(scores), cudaMemcpyHostToDevice));
    Cuda(cudaSetDevice(0));
    const OrtStatefulSamplerConfig sampler_config{sizeof(sampler_config), ORT_STATEFUL_GREEDY, 0, 0};
    Ort::SamplerOptions sampler(sampler_config);
    sampler.SetProvider(choice, session);
    EXPECT_EQ(sampler.Sample(logits), 2);
    Cuda(cudaGetDevice(&current));
    EXPECT_EQ(current, 0);
    ++tested;
  }
  EXPECT_GT(tested, 0);
  RecordProperty("tested_cache_devices", tested);
}

TEST_F(CudaStatefulTest, GreedyMatchesCpuAcrossShapesDtypesAndReductionBoundaries) {
  for (const auto type : {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16}) {
    for (const int64_t width : {1, 31, 32, 33, 255, 256, 257, 1023, 1024, 1025, 65535, 65536, 65537, 131073}) {
      std::vector<float> scores(static_cast<size_t>(3 * width));
      for (size_t i = 0; i < scores.size(); ++i) scores[i] = static_cast<float>(i % 19) - 10.f;
      for (int64_t row = 0; row < 3; ++row) {
        const int64_t peak = row == 0 ? 0 : row == 1 ? width / 2
                                                     : width - 1;
        scores[static_cast<size_t>(row * width + peak)] = 64.f + static_cast<float>(row);
        scores[static_cast<size_t>((row + 1) * width - 1)] = 64.f + static_cast<float>(row);
      }
      CheckGreedy({1, 3, width}, scores, type);
      CheckGreedy({1, 3, width}, scores, type, 0);
      CheckGreedy({1, 3, width}, scores, type, 1);
      scores.resize(static_cast<size_t>(width));
      CheckGreedy({1, width}, scores, type);
    }
    for (const float value : {-std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::infinity(), -0.f, 0.f}) {
      CheckGreedy({1, 1025}, std::vector<float>(1025, value), type);
    }
    const float tiny = type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                           ? std::numeric_limits<float>::denorm_min()
                           : std::ldexp(1.f, -24);
    CheckGreedy({1, 4}, {0.f, tiny, -tiny, tiny}, type);
    CheckGreedy({1, 4}, {-0.f, 0.f, -0.f, 0.f}, type);
    CheckGreedy({1, 3}, {std::numeric_limits<float>::lowest(), -1.f, std::numeric_limits<float>::max()}, type);
    std::vector<float> scores(1025, -std::numeric_limits<float>::infinity());
    scores[256] = scores[1024] = std::numeric_limits<float>::infinity();
    CheckGreedy({1, 1025}, scores, type);
    for (const size_t index : {size_t{0}, size_t{256}, size_t{1024}, size_t{2049}, size_t{3074}}) {
      scores.assign(3075, 1.f);
      scores[index] = std::numeric_limits<float>::quiet_NaN();
      CheckGreedy({1, 3, 1025}, scores, type, 1);
    }
  }
}

TEST_F(CudaStatefulTest, GreedyDeviceCoverage) {
  for (const auto type : {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16}) {
    for (const int64_t width : {1, 257, 65537}) {
      std::vector<float> scores(static_cast<size_t>(width), -1.f);
      scores.back() = 4.f;
      CheckGreedy({1, width}, scores, type);
      std::fill(scores.begin(), scores.end(), -std::numeric_limits<float>::infinity());
      CheckGreedy({1, width}, scores, type);
      scores[0] = std::numeric_limits<float>::quiet_NaN();
      CheckGreedy({1, width}, scores, type);
    }
    std::vector<float> scores(771, 1.f);
    scores[260] = 7.f;
    CheckGreedy({1, 3, 257}, scores, type, 1);
    scores[0] = std::numeric_limits<float>::quiet_NaN();
    CheckGreedy({1, 3, 257}, scores, type, 1);
  }
}

TEST_F(CudaStatefulTest, GreedyCConsumerMetadataErrorsAndRetainedSessionResources) {
  const OrtStatefulSamplerConfig config{sizeof(config), ORT_STATEFUL_GREEDY, 0, 0};
  Ort::SamplerOptions sampler(config);
  sampler.SetProvider(Selection(), session_);
  auto scores = Scores({1, 3}, {-1.f, 4.f, 7.f}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, true);
  int64_t token = -1;
  ASSERT_EQ(StatefulApiCGreedyDeviceConsumerTest(session_, scores, &token), 0);
  EXPECT_EQ(token, 2);
  auto host = Scores({1, 3}, {-1.f, 4.f, 7.f}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, false);
  EXPECT_THROW(sampler.Sample(host), Ort::Exception);
  auto unsupported = Value({1, 3}, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16, true);
  EXPECT_THROW(sampler.Sample(unsupported), Ort::Exception);
  const int64_t dimensions[]{1, 3}, huge[]{1, std::numeric_limits<int64_t>::max()};
  Ort::MemoryInfo other("Cuda", OrtDeviceAllocator, 1, OrtMemTypeDefault);
  auto mismatched = Ort::Value::CreateTensor(
      other, scores.GetTensorMutableData<void>(), scores.GetTensorSizeInBytes(), dimensions, 2,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  EXPECT_THROW(sampler.Sample(mismatched), Ort::Exception);
  EXPECT_THROW(Ort::Value::CreateTensor(
                   scores.GetTensorMemoryInfo(), scores.GetTensorMutableData<void>(), scores.GetTensorSizeInBytes(),
                   huge, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
               Ort::Exception);
  for (const auto& shape : {std::vector<int64_t>{3}, std::vector<int64_t>{2, 3},
                            std::vector<int64_t>{1, 0}, std::vector<int64_t>{1, 0, 3}}) {
    const size_t count = std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>{});
    auto value = Scores(shape, std::vector<float>(count, 1.f), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, true);
    EXPECT_THROW(sampler.Sample(value), Ort::Exception);
  }
  for (const int64_t position : {-1, 1}) {
    auto invalid = config;
    invalid.has_logits_position = 1;
    invalid.logits_position = position;
    if (position < 0) {
      EXPECT_THROW(Ort::SamplerOptions{invalid}, Ort::Exception);
    } else {
      Ort::SamplerOptions selected(invalid);
      selected.SetProvider(Selection(), session_);
      EXPECT_THROW(selected.Sample(scores), Ort::Exception);
    }
  }
  int device_count = 0;
  Cuda(cudaGetDeviceCount(&device_count));
  const int restored_device = device_count > 1 ? 1 : 0;
  Cuda(cudaSetDevice(restored_device));
  EXPECT_EQ(sampler.Sample(scores), 2);
  int device = -1;
  Cuda(cudaGetDevice(&device));
  EXPECT_EQ(device, restored_device);
  Cuda(cudaSetDevice(0));
  session_ = Ort::Session(nullptr);
  EXPECT_EQ(sampler.Sample(scores), 2);
  EXPECT_EQ(sampler.GetDispatchInfo().cpu_calls, 0u);
}

TEST_F(CudaStatefulTest, GreedyNondefaultStreamCompletionAndFailureRecovery) {
  cudaStream_t stream = nullptr;
  Cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  auto release = [](cudaStream_t value) { Cuda(cudaStreamDestroy(value)); };
  std::unique_ptr<std::remove_pointer_t<cudaStream_t>, decltype(release)> owner(stream, release);
  auto session = Session(stream);
  const OrtStatefulSamplerConfig config{sizeof(config), ORT_STATEFUL_GREEDY, 0, 0};
  Ort::SamplerOptions sampler(config);
  sampler.SetProvider(Selection(), session);
  auto valid = Scores({1, 3}, {-1.f, 4.f, 7.f}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, true);
  auto invalid = Scores({1, 3}, {1.f, std::numeric_limits<float>::quiet_NaN(), 7.f},
                        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, true);
  EXPECT_THROW(sampler.Sample(invalid), Ort::Exception);
  EXPECT_EQ(sampler.Sample(valid), 2);
  std::promise<void> entered, unblock;
  auto unblocked = unblock.get_future();
  std::pair<std::promise<void>*, std::future<void>*> gate{&entered, &unblocked};
  Cuda(cudaLaunchHostFunc(stream, [](void* value) {
    auto& gate = *static_cast<std::pair<std::promise<void>*, std::future<void>*>*>(value);
    gate.first->set_value();
    gate.second->wait(); }, &gate));
  entered.get_future().wait();
  auto result = std::async(std::launch::async, [&] { return sampler.Sample(valid); });
  EXPECT_EQ(result.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);
  unblock.set_value();
  EXPECT_EQ(result.get(), 2);
  Cuda(cudaStreamQuery(stream));
}

TEST_F(CudaStatefulTest, GreedyTransferAndLatency) {
  const OrtStatefulSamplerConfig config{sizeof(config), ORT_STATEFUL_GREEDY, 0, 0};
  Ort::SamplerOptions sampler(config);
  sampler.SetProvider(Selection(), session_);
  std::vector<float> values(50257, -1.f);
  values[50255] = 8.f;
  auto logits = Scores({1, 50257}, values, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, true);
  EXPECT_EQ(sampler.Sample(logits), 50255);
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < 32; ++i) EXPECT_EQ(sampler.Sample(logits), 50255);
  const auto elapsed = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
  RecordProperty("greedy_mean_microseconds", elapsed / 32);
  RecordProperty("gpu_sample_calls", sampler.GetDispatchInfo().provider_calls);
  RecordProperty("result_copy_bytes_per_call", 16);
  EXPECT_EQ(sampler.GetDispatchInfo().provider_calls, 33u);
  Ort::SamplerOptions cpu(config);
  auto host = Scores({1, 50257}, values, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, false);
  const auto cpu_start = std::chrono::steady_clock::now();
  for (int i = 0; i < 32; ++i) EXPECT_EQ(cpu.Sample(host), 50255);
  const auto cpu_elapsed =
      std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - cpu_start).count();
  RecordProperty("cpu_greedy_mean_microseconds", cpu_elapsed / 32);
}

#else

TEST(CudaStatefulLegacyTest, RequiredCachesRemainUnsupported) {
  Cuda(cudaSetDevice(0));
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(1).SetInterOpNumThreads(1);
  options.AppendExecutionProvider_CUDA(OrtCUDAProviderOptions{});
  options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
  const auto model = stateful::MakeFeedbackModel().SerializeAsString();
  Ort::Session session(*ort_env, model.data(), model.size(), options);
  const auto capabilities = Ort::QueryStatefulProviderCapabilities(session, "CUDAExecutionProvider");
  EXPECT_EQ(capabilities.operations, 0u);
  const int64_t shape[]{2, 0, 2};
  const OrtStatefulKVCacheSpec spec{"key", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, shape, 3, 1, 0};
  const OrtStatefulKVCacheConfig config{sizeof(config), &spec, 1};
  Ort::KVCacheOptions cache(config);
  cache.SetProvider(Selection(), session);
  const auto memory = session.GetMemoryInfoForInputs();
  ASSERT_FALSE(memory.empty());
  ASSERT_EQ(memory.front().GetDeviceType(), OrtMemoryInfoDeviceType_GPU);
  Ort::Allocator allocator(session, memory.front());
  OrtKVCacheManager* manager = nullptr;
  const auto& api = *Ort::GetApi().GetStatefulApi();
  Ort::Status status(api.CreateKVCacheManager(cache, allocator, &manager));
  EXPECT_FALSE(status.IsOK());
  if (!status.IsOK()) {
    EXPECT_EQ(status.GetErrorCode(), ORT_NOT_IMPLEMENTED);
  }
  EXPECT_EQ(manager, nullptr);
  api.ReleaseKVCacheManager(manager);
}

#endif

}  // namespace
}  // namespace onnxruntime::test

#endif
