// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Integration tests for resource-constrained partitioning through the CUDA plugin EP.
//
// Two test levels:
// 1. Partitioning verification tests — use InferenceSessionWrapper to inspect
//    per-node EP assignments after partitioning through the plugin EP.
// 2. E2E session tests — validate output correctness under budget constraints.

#if defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP)

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "core/graph/constants.h"
#include "core/graph/model.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/inference_session.h"
#include "core/framework/error_code_helper.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
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
    if (strcmp(device.EpName(), kCudaPluginExecutionProvider) == 0) {
      return device;
    }
  }
  return Ort::ConstEpDevice{nullptr};
}

// Build a serialized ONNX model with a chain of Add nodes.
// Each node adds its own initializer (of `weight_elements` floats) to the
// previous node's output, producing a linear graph:
//   input -> Add(w0) -> Add(w1) -> ... -> Add(wN-1) -> output
// The initializer size directly controls what the ad-hoc resource accountant
// computes per node, giving us precise budget targeting.
std::string BuildAddChainModel(size_t num_nodes, int64_t weight_elements) {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  auto* opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(13);

  auto* graph = model.mutable_graph();
  graph->set_name("add_chain");

  // Shared shape for all tensors.
  auto set_type_shape = [weight_elements](ONNX_NAMESPACE::TypeProto* tp) {
    auto* tensor_type = tp->mutable_tensor_type();
    tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    tensor_type->mutable_shape()->add_dim()->set_dim_value(weight_elements);
  };

  // Graph input.
  auto* graph_input = graph->add_input();
  graph_input->set_name("input");
  set_type_shape(graph_input->mutable_type());

  std::string prev_output = "input";
  for (size_t i = 0; i < num_nodes; ++i) {
    std::string weight_name = "w_" + std::to_string(i);
    std::string output_name = (i + 1 < num_nodes)
                                  ? "t_" + std::to_string(i)
                                  : "output";

    // Initializer with known byte size = weight_elements * sizeof(float).
    auto* init = graph->add_initializer();
    init->set_name(weight_name);
    init->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    init->add_dims(weight_elements);
    // Use raw_data for compactness — zeros are fine.
    init->set_raw_data(std::string(weight_elements * sizeof(float), '\0'));

    // Weight input value_info (needed for valid graph).
    auto* w_input = graph->add_input();
    w_input->set_name(weight_name);
    set_type_shape(w_input->mutable_type());

    // Add node.
    auto* node = graph->add_node();
    node->set_op_type("Add");
    node->set_name("add_" + std::to_string(i));
    node->add_input(prev_output);
    node->add_input(weight_name);
    node->add_output(output_name);

    prev_output = output_name;
  }

  // Graph output.
  auto* graph_output = graph->add_output();
  graph_output->set_name("output");
  set_type_shape(graph_output->mutable_type());

  std::string serialized;
  model.SerializeToString(&serialized);
  return serialized;
}

// Get the internal OrtEnv* from the C++ Ort::Env wrapper.
// Ort::Env inherits Base<OrtEnv> which has operator OrtEnv*().
OrtEnv& GetOrtEnv() {
  return *static_cast<OrtEnv*>(*ort_env);
}

}  // namespace

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
    if (cuda_device_) {
      cudaDeviceSynchronize();
    }
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
          kOrtSessionOptionsResourceCudaPartitioningSettings, config_value.c_str()));
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

  // Overload that loads a model from serialized bytes (e.g., from BuildAddChainModel).
  void LoadAndVerifyPartitioning(const std::string& model_bytes,
                                 size_t budget_kb,
                                 const std::function<void(const Graph&)>& verifier) {
    OrtSessionOptions ort_options;

    const OrtEpDevice* device_ptr = static_cast<const OrtEpDevice*>(cuda_device_);
    auto ep_devices_span = gsl::make_span(&device_ptr, 1);

    std::unique_ptr<IExecutionProviderFactory> factory;
    ASSERT_STATUS_OK(CreateIExecutionProviderFactoryForEpDevices(
        GetOrtEnv().GetEnvironment(), ep_devices_span, factory));

    ort_options.provider_factories.push_back(std::move(factory));

    if (budget_kb > 0) {
      std::string config_value = std::to_string(budget_kb) + ",";
      ASSERT_STATUS_OK(ort_options.value.config_options.AddConfigEntry(
          kOrtSessionOptionsResourceCudaPartitioningSettings, config_value.c_str()));
    }

    InferenceSessionWrapper session(ort_options.value, GetOrtEnv().GetEnvironment());
    ASSERT_STATUS_OK(session.Load(model_bytes.data(), static_cast<int>(model_bytes.size())));

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

// With a small budget, the resource accountant should assign fewer nodes
// to the plugin EP than the no-budget baseline.
TEST_F(CudaPluginPartitioningTest, TinyBudget_NodesOffloadedToCpu) {
  // Build a chain of 6 Add nodes, each with a 256-element float initializer
  // (1 KB per weight). The ad-hoc accountant adds weight + output sizes with
  // a 1.5x multiplier, so each node costs roughly 1.5 * (1 KB + 1 KB) = 3 KB.
  // A 10 KB budget should accept ~3 nodes before halting.
  const std::string model = BuildAddChainModel(/*num_nodes=*/6, /*weight_elements=*/256);

  // Baseline: count plugin nodes with no budget.
  size_t baseline_plugin_count = 0;
  LoadAndVerifyPartitioning(model, /*budget_kb=*/0, [&](const Graph& graph) {
    for (const auto& node : graph.Nodes()) {
      if (node.GetExecutionProviderType() == kCudaPluginExecutionProvider) {
        ++baseline_plugin_count;
      }
    }
  });
  ASSERT_GT(baseline_plugin_count, size_t{1})
      << "Baseline must have multiple plugin nodes for the test to be meaningful";

  // Now run with a 10 KB budget — should accept some but not all nodes.
  size_t constrained_plugin_count = 0;
  LoadAndVerifyPartitioning(model, /*budget_kb=*/10, [&](const Graph& graph) {
    for (const auto& node : graph.Nodes()) {
      if (node.GetExecutionProviderType() == kCudaPluginExecutionProvider) {
        ++constrained_plugin_count;
      }
    }
  });

  EXPECT_GT(constrained_plugin_count, size_t{0})
      << "Budget should be large enough to accept at least one node";
  EXPECT_LT(constrained_plugin_count, baseline_plugin_count)
      << "A 10 KB budget should reduce plugin EP node count from the no-budget baseline ("
      << baseline_plugin_count << " nodes)";
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
    if (cuda_device_) {
      cudaDeviceSynchronize();
    }
  }

  Ort::Session CreateSessionWithBudget(const ORTCHAR_T* model_path,
                                       size_t budget_kb) {
    Ort::SessionOptions so;
    so.AppendExecutionProvider_V2(*ort_env, {cuda_device_},
                                  std::unordered_map<std::string, std::string>{});

    if (budget_kb > 0) {
      std::string config_value = std::to_string(budget_kb) + ",";
      so.AddConfigEntry(kOrtSessionOptionsResourceCudaPartitioningSettings,
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
