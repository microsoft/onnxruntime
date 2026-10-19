// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(USE_CUDA)

#include <array>
#include <chrono>
#include <cstring>
#include <future>
#include <limits>
#include <memory>
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
  EXPECT_EQ(caps.operations, 63u);
  EXPECT_EQ(caps.sampler_element_types, 0u);
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
  EXPECT_THROW(sampler.SetProvider(Selection(), session_), Ort::Exception);
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
    Cuda(cudaSetDevice(0));
    ++tested;
  }
  EXPECT_GT(tested, 0);
  RecordProperty("tested_cache_devices", tested);
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
