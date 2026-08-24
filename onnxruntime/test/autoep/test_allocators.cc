// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <set>
#include <thread>
#include <gsl/gsl>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "core/framework/allocator.h"
#include "core/graph/constants.h"
#include "core/graph/onnx_protobuf.h"
#include "core/platform/env_var_utils.h"
#include "core/session/abi_key_value_pairs.h"
#include "core/session/onnxruntime_cxx_api.h"

#include "test/autoep/test_autoep_utils.h"
#include "test/shared_lib/utils.h"
#include "test/util/include/api_asserts.h"
#include "test/util/include/asserts.h"
#include "test/util/include/file_util.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

namespace {

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

struct DummyAllocator : OrtAllocator {
  DummyAllocator(const OrtMemoryInfo* mem_info)
      : memory_info{mem_info} {
    Alloc = AllocImpl;
    Free = FreeImpl;
    Info = InfoImpl;
    Reserve = AllocImpl;      // no special reserve logic and most likely unnecessary unless you have your own arena
    GetStats = nullptr;       // this can be set to nullptr if not implemented
    AllocOnStream = nullptr;  // optional
    Shrink = nullptr;
  }

  static void* ORT_API_CALL AllocImpl(struct OrtAllocator* this_, size_t size) {
    auto& impl = *static_cast<DummyAllocator*>(this_);
    ++impl.stats.num_allocs;
    impl.stats.max_alloc_size = std::max<int64_t>(size, impl.stats.max_alloc_size);

    return malloc(size);
  }

  static void ORT_API_CALL FreeImpl(struct OrtAllocator* /*this_*/, void* p) {
    return free(p);
  }

  static const struct OrtMemoryInfo* ORT_API_CALL InfoImpl(const struct OrtAllocator* this_) {
    const DummyAllocator& impl = *static_cast<const DummyAllocator*>(this_);
    return impl.memory_info;
  }

 private:
  const OrtMemoryInfo* memory_info;
  AllocatorStats stats{};
};
}  // namespace

// validate CreateSharedAllocator allows adding an arena to the shared allocator
TEST(SharedAllocators, AddArenaToSharedAllocator) {
  RegisteredEpDeviceUniquePtr example_ep;
  Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_info, example_ep);

  Ort::ConstEpDevice example_ep_device{example_ep.get()};

  auto ep_memory_info = example_ep_device.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);

  // validate there is a shared allocator
  auto allocator = ort_env->GetSharedAllocator(ep_memory_info);
  ASSERT_NE(allocator, nullptr);

  // call CreateSharedAllocator to replace with arena based allocator. arena is configured with kvps
  Ort::KeyValuePairs allocator_options;
  auto initial_chunk_size = "25600";  // arena allocates in 256 byte amounts
  allocator_options.Add(OrtArenaCfg::ConfigKeyNames::InitialChunkSizeBytes, initial_chunk_size);

  allocator = ort_env->CreateSharedAllocator(example_ep.get(), OrtDeviceMemoryType_DEFAULT,
                                             // allocator is internally added by EP.
                                             // OrtArenaAllocator can only be used for the internal BFCArena
                                             OrtDeviceAllocator,
                                             allocator_options);

  // first allocation should init the arena to the initial chunk size
  void* mem = allocator.Alloc(16);
  allocator.Free(mem);

  // stats should prove the arena was used
  auto allocator_stats = allocator.GetStats();

  using ::testing::Contains;
  using ::testing::Pair;
  const auto& stats = static_cast<OrtKeyValuePairs*>(allocator_stats)->Entries();
  EXPECT_THAT(stats, Contains(Pair("NumAllocs", "1")));
  EXPECT_THAT(stats, Contains(Pair("NumArenaExtensions", "1")));
  EXPECT_THAT(stats, Contains(Pair("TotalAllocated", initial_chunk_size)));

  // optional. ORT owns the allocator but we want to test the release implementation
  ort_env->ReleaseSharedAllocator(example_ep.get(), OrtDeviceMemoryType_DEFAULT);
}

TEST(SharedAllocators, GetSharedAllocator) {
  // default CPU allocator should be available.
  // create a memory info with a different name to validate the shared allocator lookup ignores the name
  auto test_cpu_memory_info = Ort::MemoryInfo("dummy", OrtMemoryInfoDeviceType_CPU, 0, 0,
                                              OrtDeviceMemoryType_DEFAULT, 0, OrtDeviceAllocator);

  const auto get_allocator_and_check_name = [&](const std::string& expected_name) {
    auto allocator = ort_env->GetSharedAllocator(test_cpu_memory_info);
    ASSERT_NE(allocator, nullptr);

    auto ort_cpu_memory_info = allocator.GetInfo();
    auto allocator_name = ort_cpu_memory_info.GetAllocatorName();
    ASSERT_EQ(expected_name, allocator_name);  // Default ORT CPU allocator
  };

  // check we get the default ORT CPU allocator initially
  get_allocator_and_check_name(onnxruntime::CPU);

  // register custom allocator and make sure that is accessible by exact match
  DummyAllocator dummy_alloc{test_cpu_memory_info};
  ort_env->RegisterAllocator(&dummy_alloc);

  // GetSharedAllocator should now match the custom allocator
  get_allocator_and_check_name("dummy");

  // unregister custom allocator
  ort_env->UnregisterAllocator(test_cpu_memory_info);

  // there should always be a CPU allocator available
  get_allocator_and_check_name(onnxruntime::CPU);
}

#if defined(USE_WEBGPU) && defined(ORT_USE_EP_API_ADAPTERS)

namespace {

class WebGpuPluginSharedAllocatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto devices_before_registration = env_.GetEpDevices();
    const auto ep_lib_path = GetSharedLibraryFileName(ORT_TSTR("onnxruntime_providers_webgpu"));
    ep_info_ = std::make_unique<Utils::ExamplePluginInfo>(
        ep_lib_path, "webgpu_allocator_test", kWebGpuExecutionProvider);
    ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(env_, *ep_info_, ep_device_holder_));

    const auto devices_after_registration = env_.GetEpDevices();
    for (const auto& candidate : devices_after_registration) {
      const auto candidate_ptr = static_cast<const OrtEpDevice*>(candidate);
      if (candidate.EpName() == std::string{kWebGpuExecutionProvider} &&
          std::none_of(devices_before_registration.begin(), devices_before_registration.end(),
                       [&](const Ort::ConstEpDevice& existing) {
                         return static_cast<const OrtEpDevice*>(existing) == candidate_ptr;
                       })) {
        ep_devices_.push_back(candidate_ptr);
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
    return env_;
  }

 private:
  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "webgpu_allocator_test"};
  std::unique_ptr<Utils::ExamplePluginInfo> ep_info_;
  RegisteredEpDeviceUniquePtr ep_device_holder_;
  std::vector<const OrtEpDevice*> ep_devices_;
};

}  // namespace

TEST_F(WebGpuPluginSharedAllocatorTest, SharedAllocatorCreatedOnPluginRegistration) {
  auto device_memory_info = EpDevice().GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
  ASSERT_NE(device_memory_info, nullptr);

  auto allocator = Env().GetSharedAllocator(device_memory_info);
  ASSERT_NE(allocator, nullptr);
  EXPECT_TRUE(allocator.GetInfo() == device_memory_info);
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

TEST_F(WebGpuPluginSharedAllocatorTest, SharedAllocatorsAreDistinctPerDevice) {
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

#endif  // defined(USE_WEBGPU) && defined(ORT_USE_EP_API_ADAPTERS)

}  // namespace test
}  // namespace onnxruntime
