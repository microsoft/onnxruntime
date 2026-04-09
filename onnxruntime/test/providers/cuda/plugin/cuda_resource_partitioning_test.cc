// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Integration tests for resource-constrained partitioning through the CUDA plugin EP.
//
// Two test levels:
// 1. OrtResourceCount struct tests — validate the C-safe tagged union.
// 2. Partitioning verification tests — use InferenceSessionWrapper to inspect
//    per-node EP assignments after partitioning through the plugin EP.

#if defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP)

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "core/graph/model.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/inference_session.h"
#include "core/framework/error_code_helper.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/ort_env.h"
#include "core/session/utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/file_util.h"
#include "test/util/include/inference_session_wrapper.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {
namespace {

constexpr const char* kResourcePartitioningRegistrationName = "CudaPluginResourceTest";

// Resolve the CUDA plugin EP shared library path.
std::filesystem::path GetCudaPluginLibraryPath() {
  return GetSharedLibraryFileName(ORT_TSTR("onnxruntime_providers_cuda_plugin"));
}

// RAII handle that registers/unregisters the CUDA plugin EP library.
class ScopedCudaPluginRegistration {
 public:
  ScopedCudaPluginRegistration(Ort::Env& env, const char* registration_name)
      : env_(env), name_(registration_name) {
    auto lib_path = GetCudaPluginLibraryPath();
    if (!std::filesystem::exists(lib_path)) {
      available_ = false;
      return;
    }
    env_.RegisterExecutionProviderLibrary(name_.c_str(), lib_path.c_str());
    available_ = true;
  }

  ~ScopedCudaPluginRegistration() {
    if (available_) {
      try {
        env_.UnregisterExecutionProviderLibrary(name_.c_str());
      } catch (...) {
      }
    }
  }

  bool IsAvailable() const { return available_; }

  ScopedCudaPluginRegistration(const ScopedCudaPluginRegistration&) = delete;
  ScopedCudaPluginRegistration& operator=(const ScopedCudaPluginRegistration&) = delete;

 private:
  Ort::Env& env_;
  std::string name_;
  bool available_ = false;
};

// Find the CUDA plugin EP device after registration.
Ort::ConstEpDevice FindCudaPluginDevice(Ort::Env& env) {
  auto ep_devices = env.GetEpDevices();
  for (const auto& device : ep_devices) {
    if (strcmp(device.EpName(), "CudaPluginExecutionProvider") == 0) {
      return device;
    }
  }
  return Ort::ConstEpDevice{nullptr};
}

// Get the internal OrtEnv* from the C++ Ort::Env wrapper.
// Ort::Env inherits Base<OrtEnv> which has operator OrtEnv*().
OrtEnv& GetOrtEnv() {
  return *static_cast<OrtEnv*>(*ort_env);
}

}  // namespace

// ---------------------------------------------------------------------------
// OrtResourceCount struct tests
// ---------------------------------------------------------------------------

TEST(OrtResourceCountTest, None_HasKindNone) {
  OrtResourceCount rc = OrtResourceCount::None();
  EXPECT_EQ(rc.kind, OrtResourceCountKind_None);
}

TEST(OrtResourceCountTest, FromTotalBytes_RoundTrips) {
  constexpr size_t kTestValue = 42 * 1024 * 1024;  // 42 MB
  OrtResourceCount rc = OrtResourceCount::FromTotalBytes(kTestValue);
  EXPECT_EQ(rc.kind, OrtResourceCountKind_TotalBytes);
  EXPECT_EQ(rc.AsTotalBytes(), kTestValue);
}

TEST(OrtResourceCountTest, FromTotalBytes_MaxValue) {
  OrtResourceCount rc = OrtResourceCount::FromTotalBytes(std::numeric_limits<size_t>::max());
  EXPECT_EQ(rc.kind, OrtResourceCountKind_TotalBytes);
  EXPECT_EQ(rc.AsTotalBytes(), std::numeric_limits<size_t>::max());
}

TEST(OrtResourceCountTest, FromTotalBytes_Zero) {
  OrtResourceCount rc = OrtResourceCount::FromTotalBytes(0);
  EXPECT_EQ(rc.kind, OrtResourceCountKind_TotalBytes);
  EXPECT_EQ(rc.AsTotalBytes(), size_t{0});
}

TEST(OrtResourceCountTest, CopySemantics) {
  OrtResourceCount original = OrtResourceCount::FromTotalBytes(12345);
  OrtResourceCount copy = original;
  EXPECT_EQ(copy.kind, OrtResourceCountKind_TotalBytes);
  EXPECT_EQ(copy.AsTotalBytes(), size_t{12345});
  copy.value.total_bytes = 99999;
  EXPECT_EQ(original.AsTotalBytes(), size_t{12345});
}

TEST(OrtResourceCountTest, ReservedFieldIsZero) {
  OrtResourceCount rc = OrtResourceCount::FromTotalBytes(100);
  EXPECT_EQ(rc.reserved_, uint32_t{0});
}

// ---------------------------------------------------------------------------
// Lower-level partitioning tests that verify per-node EP assignments
// ---------------------------------------------------------------------------

class CudaPluginPartitioningTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "No CUDA device available.";
    }

    registration_ = std::make_unique<ScopedCudaPluginRegistration>(
        *ort_env, kResourcePartitioningRegistrationName);
    if (!registration_->IsAvailable()) {
      GTEST_SKIP() << "CUDA plugin EP library not found.";
    }

    cuda_device_ = FindCudaPluginDevice(*ort_env);
    if (!cuda_device_) {
      GTEST_SKIP() << "No CUDA plugin EP device found after registration.";
    }
  }

  void TearDown() override {
    registration_.reset();
    cudaDeviceSynchronize();
  }

  // Load a model through the CUDA plugin EP with the given resource budget,
  // then call the verifier to inspect graph node assignments.
  //
  // Uses InferenceSessionWrapper + OrtSessionOptions + InitializeSession
  // so that the plugin factory creates the EP and partitioning runs normally,
  // but we can access the graph via wrapper.GetGraph().
  void LoadAndVerifyPartitioning(const ORTCHAR_T* model_path,
                                 size_t budget_kb,
                                 const std::function<void(const Graph&)>& verifier) {
    OrtSessionOptions ort_options;

    // Create the plugin EP factory from the registered device.
    const OrtEpDevice* device_ptr = static_cast<const OrtEpDevice*>(cuda_device_);
    auto ep_devices_span = gsl::make_span(&device_ptr, 1);

    std::unique_ptr<IExecutionProviderFactory> factory;
    ASSERT_STATUS_OK(CreateIExecutionProviderFactoryForEpDevices(
        GetOrtEnv().GetEnvironment(), ep_devices_span, factory));

    ort_options.provider_factories.push_back(std::move(factory));

    // Set resource partitioning budget if requested.
    if (budget_kb > 0) {
      std::string config_value = std::to_string(budget_kb) + ",";
      ASSERT_STATUS_OK(ort_options.value.config_options.AddConfigEntry(
          "session.resource_cuda_partitioning_settings", config_value.c_str()));
    }

    // Create the session wrapper — gives us access to the graph after partitioning.
    InferenceSessionWrapper session(ort_options.value, GetOrtEnv().GetEnvironment());
    ASSERT_STATUS_OK(session.Load(model_path));

    // InitializeSession iterates provider_factories, creates the plugin EP,
    // registers it with the session, and calls session.Initialize() which
    // runs graph partitioning (invoking plugin GetCapability).
    OrtStatus* status = InitializeSession(&ort_options, session);
    ASSERT_STATUS_OK(ToStatusAndRelease(status));

    verifier(session.GetGraph());
  }

  std::unique_ptr<ScopedCudaPluginRegistration> registration_;
  Ort::ConstEpDevice cuda_device_{nullptr};
};

// With no resource budget, all CUDA-supported nodes should be assigned to the plugin EP.
TEST_F(CudaPluginPartitioningTest, NoBudget_AllNodesCudaPlugin) {
  constexpr const ORTCHAR_T* model_path = ORT_TSTR("testdata/mul_1.onnx");

  LoadAndVerifyPartitioning(model_path, /*budget_kb=*/0, [](const Graph& graph) {
    for (const auto& node : graph.Nodes()) {
      // With no budget constraint, all nodes that the CUDA plugin supports
      // should be assigned to it. The plugin EP type name may vary, so just
      // verify it's NOT assigned to CPU.
      EXPECT_NE(node.GetExecutionProviderType(), kCpuExecutionProvider)
          << "Node " << node.Name() << " (" << node.OpType()
          << ") unexpectedly assigned to CPU with no budget constraint";
    }
  });
}

// With a very large budget, all nodes should still be on the plugin EP (same as no budget).
TEST_F(CudaPluginPartitioningTest, LargeBudget_AllNodesCudaPlugin) {
  constexpr const ORTCHAR_T* model_path = ORT_TSTR("testdata/mul_1.onnx");

  // 1 GB — effectively unlimited and safe across 32-bit/64-bit builds
  LoadAndVerifyPartitioning(model_path, /*budget_kb=*/1024 * 1024, [](const Graph& graph) {
    for (const auto& node : graph.Nodes()) {
      EXPECT_NE(node.GetExecutionProviderType(), kCpuExecutionProvider)
          << "Node " << node.Name() << " (" << node.OpType()
          << ") unexpectedly assigned to CPU with large budget";
    }
  });
}

// With a tiny budget (1 byte), nodes should be offloaded to CPU because
// the resource accountant will run out of budget.
TEST_F(CudaPluginPartitioningTest, TinyBudget_NodesOffloadedToCpu) {
  // Use a model with multiple nodes so we can see some go to CPU.
  constexpr const ORTCHAR_T* model_path = ORT_TSTR("testdata/transformers/tiny_gpt2_beamsearch.onnx");

  // 1 KB budget — ad-hoc accountant will compute non-zero cost for any
  // node with initializers or known output shapes, so nodes must be offloaded.
  LoadAndVerifyPartitioning(model_path, /*budget_kb=*/1, [](const Graph& graph) {
    bool has_cpu_node = false;
    for (const auto& node : graph.Nodes()) {
      if (node.GetExecutionProviderType() == kCpuExecutionProvider) {
        has_cpu_node = true;
        break;
      }
    }
    EXPECT_TRUE(has_cpu_node)
        << "With a 1 KB budget, at least some nodes should be offloaded to CPU";
  });
}

// ---------------------------------------------------------------------------
// E2E tests (existing high-level session tests, kept for coverage)
// ---------------------------------------------------------------------------

class CudaResourcePartitioningTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "No CUDA device available.";
    }

    registration_ = std::make_unique<ScopedCudaPluginRegistration>(
        *ort_env, kResourcePartitioningRegistrationName);
    if (!registration_->IsAvailable()) {
      GTEST_SKIP() << "CUDA plugin EP library not found.";
    }

    cuda_device_ = FindCudaPluginDevice(*ort_env);
    if (!cuda_device_) {
      GTEST_SKIP() << "No CUDA plugin EP device found after registration.";
    }
  }

  void TearDown() override {
    registration_.reset();
    cudaDeviceSynchronize();
  }

  Ort::Session CreateSessionWithBudget(const ORTCHAR_T* model_path,
                                       size_t budget_kb) {
    Ort::SessionOptions so;
    so.AppendExecutionProvider_V2(*ort_env, {cuda_device_}, {});

    if (budget_kb > 0) {
      std::string config_value = std::to_string(budget_kb) + ",";
      so.AddConfigEntry("session.resource_cuda_partitioning_settings",
                        config_value.c_str());
    }

    return Ort::Session(*ort_env, model_path, so);
  }

  std::unique_ptr<ScopedCudaPluginRegistration> registration_;
  Ort::ConstEpDevice cuda_device_{nullptr};
};

TEST_F(CudaResourcePartitioningTest, NoBudget_SessionCreatesSuccessfully) {
  auto model_path = ORT_TSTR("testdata/mul_1.onnx");
  ASSERT_NO_THROW(CreateSessionWithBudget(model_path, 0));
}

TEST_F(CudaResourcePartitioningTest, BudgetConstrained_ProducesValidOutput) {
  auto model_path = ORT_TSTR("testdata/mul_1.onnx");
  Ort::Session session = CreateSessionWithBudget(model_path, 100);

  auto input_name = session.GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
  auto output_name = session.GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions());

  auto type_info = session.GetInputTypeInfo(0);
  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  auto shape = tensor_info.GetShape();
  for (auto& dim : shape) {
    if (dim < 0) dim = 1;
  }

  size_t num_elements = 1;
  for (auto dim : shape) {
    num_elements *= static_cast<size_t>(dim);
  }

  std::vector<float> input_data(num_elements, 2.0f);
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, input_data.data(), input_data.size(),
      shape.data(), shape.size());

  const char* input_names[] = {input_name.get()};
  const char* output_names[] = {output_name.get()};

  auto outputs = session.Run(Ort::RunOptions{nullptr},
                             input_names, &input_tensor, 1,
                             output_names, 1);

  ASSERT_EQ(outputs.size(), size_t{1});
  ASSERT_TRUE(outputs[0].IsTensor());

  auto* output_data = outputs[0].GetTensorData<float>();
  auto output_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
  size_t output_count = 1;
  for (auto dim : output_shape) {
    output_count *= static_cast<size_t>(dim);
  }
  for (size_t i = 0; i < output_count; ++i) {
    EXPECT_FALSE(std::isnan(output_data[i])) << "NaN at index " << i;
    EXPECT_FALSE(std::isinf(output_data[i])) << "Inf at index " << i;
  }
}

}  // namespace test
}  // namespace onnxruntime

#endif  // defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP)
