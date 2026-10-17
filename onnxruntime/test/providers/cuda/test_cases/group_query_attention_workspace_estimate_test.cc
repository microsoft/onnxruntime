// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#if !defined(USE_CUDA_MINIMAL) && !defined(DISABLE_CONTRIB_OPS) && !defined(BUILD_CUDA_EP_AS_PLUGIN)

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/framework/op_kernel.h"
#include "core/framework/config_options.h"
#include "core/framework/resource_accountant.h"
#include "core/framework/session_state.h"
#include "core/graph/graph.h"
#include "core/providers/cuda/cuda_execution_provider.h"
#include "core/providers/cuda/cuda_execution_provider_info.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "contrib_ops/cpu/bert/attention_common.h"
#include "contrib_ops/cuda/bert/group_query_attention_workspace_estimate.h"
#include "test/providers/cuda/test_cases/cuda_test_bridge.h"
#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/util/include/scoped_env_vars.h"

namespace onnxruntime {
namespace test {
namespace {

using contrib::attention::AttentionBackend;
using contrib::cuda::EstimateGroupQueryAttentionWorkspace;
using contrib::cuda::GetGQACompleteWorkspaceRecipe;
using contrib::cuda::GetGQAEffectiveWorkspaceKvLength;
using contrib::cuda::GetGQAFlashWorkspaceRecipe;
using contrib::cuda::GetGQAWorkspaceAggregateForBounds;
using contrib::cuda::GQABackend;
using contrib::cuda::GQAConcreteRoute;
using contrib::cuda::GQAExecutionPhase;
using contrib::cuda::GQAFlashConfig;
using contrib::cuda::GQAKvQuantizationType;
using contrib::cuda::GQAPreprocessMode;
using contrib::cuda::GQAReachableBackend;
using contrib::cuda::GQAWorkspaceAggregate;
using contrib::cuda::GQAWorkspaceBounds;
using contrib::cuda::GQAWorkspaceEstimateConfig;
using contrib::cuda::GQAWorkspaceProblem;
using contrib::cuda::GQAXqaHeadSinkStorage;
using contrib::cuda::HasGQAReachableBackend;
using contrib::cuda::SetGroupQueryAttentionLevel1MemoryEstimate;
using contrib::cuda::SetGroupQueryAttentionWorkspaceRequirements;

constexpr int kMath = static_cast<int>(AttentionBackend::MATH);
constexpr int kFlash = static_cast<int>(AttentionBackend::FLASH_ATTENTION);
constexpr int kCudnn = static_cast<int>(AttentionBackend::CUDNN_FLASH_ATTENTION);
constexpr int kEfficient = static_cast<int>(AttentionBackend::EFFICIENT_ATTENTION);

WorkspaceInputShape Known(std::initializer_list<int64_t> dims) {
  return WorkspaceInputShape::PresentWithShape(TensorShape{TensorShapeVector{dims}});
}

std::array<WorkspaceInputShape, 16> SeparateShapes(int64_t sequence = 4,
                                                   int64_t head = 64,
                                                   int64_t capacity = 256,
                                                   int64_t batch = 2) {
  std::array<WorkspaceInputShape, 16> shapes;
  shapes[0] = Known({batch, sequence, 8 * head});
  shapes[1] = Known({batch, sequence, 2 * head});
  shapes[2] = Known({batch, sequence, 2 * head});
  shapes[3] = Known({batch, 2, capacity, head});
  shapes[4] = Known({batch, 2, capacity, head});
  shapes[5] = Known({batch});
  shapes[6] = Known({});
  return shapes;
}

std::array<WorkspaceInputShape, 16> PackedShapes() {
  auto shapes = SeparateShapes();
  shapes[0] = Known({2, 4, 12 * 64});
  shapes[1] = {};
  shapes[2] = {};
  return shapes;
}

GQAWorkspaceEstimateConfig Config() {
  GQAWorkspaceEstimateConfig config;
  config.qkv_element_size = 2;
  config.cache_element_size = 2;
  config.num_heads = 8;
  config.kv_num_heads = 2;
  config.local_window_size = 256;
  config.sliding_window_cache = true;
  return config;
}

cudaDeviceProp Device(int major = 8, int minor = 0) {
  cudaDeviceProp device{};
  device.major = major;
  device.minor = minor;
  device.multiProcessorCount = 108;
  return device;
}

GQAWorkspaceBounds Bounds() {
  GQAWorkspaceBounds bounds;
  bounds.qkv_element_size = 2;
  bounds.cache_element_size = 2;
  bounds.batch_size_bound = 2;
  bounds.sequence_length_bound = 4;
  bounds.num_heads = 8;
  bounds.kv_num_heads = 2;
  bounds.head_size_bound = 64;
  bounds.present_kv_cache_capacity_bound = 256;
  bounds.prefill_reachable = true;
  bounds.decode_reachable = true;
  bounds.chunked_prefill_reachable = true;
  bounds.covering_backends = {GQAReachableBackend::Unfused,
                              GQAReachableBackend::Unfused,
                              GQAReachableBackend::Unfused};
  bounds.multi_processor_count = 108;
  return bounds;
}

template <typename Callback>
auto WithGroupQueryAttentionNode(Callback callback, int64_t static_head = 0,
                                 bool packed = false, bool is_bf16 = false) {
  ONNX_NAMESPACE::TypeProto query_type;
  query_type.mutable_tensor_type()->set_elem_type(
      is_bf16 ? ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16
              : ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  if (static_head > 0) {
    auto* shape = query_type.mutable_tensor_type()->mutable_shape();
    shape->add_dim()->set_dim_param("batch");
    shape->add_dim()->set_dim_param("sequence");
    shape->add_dim()->set_dim_value((packed ? 12 : 8) * static_head);
  }
  ONNX_NAMESPACE::TypeProto cache_type;
  cache_type.mutable_tensor_type()->set_elem_type(
      is_bf16 ? ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16
              : ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  NodeArg query{"query", &query_type};
  NodeArg key{"key", &query_type};
  NodeArg value{"value", &query_type};
  NodeArg past_key{"past_key", &cache_type};
  NodeArg missing{"", nullptr};

  NodeAttributes attributes;
  const auto add_int_attribute = [&attributes](const char* name, int64_t value) {
    ONNX_NAMESPACE::AttributeProto attribute;
    attribute.set_name(name);
    attribute.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
    attribute.set_i(value);
    attributes.emplace(name, std::move(attribute));
  };
  add_int_attribute("num_heads", 8);
  add_int_attribute("kv_num_heads", 2);
  add_int_attribute("causal", 1);
  add_int_attribute("local_window_size", 256);
  add_int_attribute("sliding_window_cache", 1);

  const std::vector<NodeArg*> inputs{
      &query, packed ? &missing : &key, packed ? &missing : &value, &past_key};
  const std::vector<NodeArg*> outputs;
  Node node{"gqa", "GroupQueryAttention", "", inputs, outputs, &attributes, kMSDomain};
  return callback(node);
}

std::optional<contrib::cuda::GQAWorkspaceAggregate> EstimateFromNode(
    gsl::span<const WorkspaceInputShape> input_shapes,
    const AttentionKernelOptions& options) {
  return WithGroupQueryAttentionNode([&](const Node& node) {
    return EstimateGroupQueryAttentionWorkspaceForTest(
        &node, input_shapes, Device(), options);
  });
}

void SetValueInfo(ONNX_NAMESPACE::ValueInfoProto& value_info,
                  const char* name,
                  int32_t element_type,
                  std::initializer_list<int64_t> dimensions) {
  value_info.set_name(name);
  auto* tensor_type = value_info.mutable_type()->mutable_tensor_type();
  tensor_type->set_elem_type(element_type);
  auto* shape = tensor_type->mutable_shape();
  for (int64_t dimension : dimensions) {
    shape->add_dim()->set_dim_value(dimension);
  }
}

std::string BuildGroupQueryAttentionKernelModel(bool sliding_window_cache = true,
                                                bool include_head_sink = true,
                                                int64_t sequence = 4,
                                                int64_t head = 64,
                                                int64_t capacity = 256,
                                                int64_t batch = 2,
                                                bool dynamic_sequence = false) {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  auto* onnx_opset = model.add_opset_import();
  onnx_opset->set_domain("");
  onnx_opset->set_version(17);
  auto* ms_opset = model.add_opset_import();
  ms_opset->set_domain(kMSDomain);
  ms_opset->set_version(1);

  auto* graph = model.mutable_graph();
  graph->set_name("group_query_attention_workspace_level2");
  auto* node = graph->add_node();
  node->set_domain(kMSDomain);
  node->set_name("gqa");
  node->set_op_type("GroupQueryAttention");
  for (const char* input_name :
       {"query", "key", "value", "past_key", "past_value", "seqlens_k",
        "total_sequence_length", "", "", "", ""}) {
    node->add_input(input_name);
  }
  node->add_input(include_head_sink ? "head_sink" : "");
  node->add_output("output");
  node->add_output("present_key");
  node->add_output("present_value");

  const auto add_int_attribute = [node](const char* name, int64_t value) {
    auto* attribute = node->add_attribute();
    attribute->set_name(name);
    attribute->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
    attribute->set_i(value);
  };
  add_int_attribute("num_heads", 8);
  add_int_attribute("kv_num_heads", 2);
  if (sliding_window_cache) {
    add_int_attribute("local_window_size", capacity);
    add_int_attribute("sliding_window_cache", 1);
  }

  constexpr int32_t kFloat16 = ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
  constexpr int32_t kInt32 = ONNX_NAMESPACE::TensorProto_DataType_INT32;
  SetValueInfo(*graph->add_input(), "query", kFloat16, {batch, sequence, 8 * head});
  SetValueInfo(*graph->add_input(), "key", kFloat16, {batch, sequence, 2 * head});
  SetValueInfo(*graph->add_input(), "value", kFloat16, {batch, sequence, 2 * head});
  SetValueInfo(*graph->add_input(), "past_key", kFloat16, {batch, 2, capacity, head});
  SetValueInfo(*graph->add_input(), "past_value", kFloat16, {batch, 2, capacity, head});
  SetValueInfo(*graph->add_input(), "seqlens_k", kInt32, {batch});
  SetValueInfo(*graph->add_input(), "total_sequence_length", kInt32, {});
  SetValueInfo(*graph->add_output(), "output", kFloat16, {batch, sequence, 8 * head});
  SetValueInfo(*graph->add_output(), "present_key", kFloat16, {batch, 2, capacity, head});
  SetValueInfo(*graph->add_output(), "present_value", kFloat16, {batch, 2, capacity, head});
  if (dynamic_sequence) {
    for (int i : {0, 1, 2}) {
      graph->mutable_input(i)->mutable_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(1)->set_dim_param("sequence");
    }
    graph->mutable_output(0)->mutable_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(1)->set_dim_param("sequence");
  }

  if (include_head_sink) {
    auto* head_sink = graph->add_initializer();
    head_sink->set_name("head_sink");
    head_sink->set_data_type(kFloat16);
    head_sink->add_dims(8);
    head_sink->mutable_raw_data()->assign(8 * sizeof(uint16_t), '\0');
  }

  std::string bytes;
  model.SerializeToString(&bytes);
  return bytes;
}

const Node* FindNodeByOpType(const Graph& graph, const char* op_type) {
  for (const auto& node : graph.Nodes()) {
    if (node.OpType() == op_type) {
      return &node;
    }
  }
  return nullptr;
}

bool HasCudaDevice() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

struct GQAPolicyRunConfig {
  const char* policy = "memory";
  GQAExecutionPhase phase = GQAExecutionPhase::Prefill;
  int64_t head = 512;
  int64_t capacity = 64;
  int kernel_selection = kFlash | kEfficient | kMath;
  int64_t redeclared_sequence_bound = 0;
  bool declare_envelope = true;
  bool dynamic_sequence = false;
  GQAReachableBackend required_permitted = GQAReachableBackend::None;
  GQAReachableBackend forbidden_permitted = GQAReachableBackend::None;
  const char* expected_backend = "unfused";
  bool expect_failure = false;
};

class GroupQueryAttentionDispatchPolicyTest : public testing::Test {
 protected:
  void SetUp() override {
    if (!HasCudaDevice()) {
      GTEST_SKIP() << "A CUDA device is required to exercise GQA dispatch.";
    }
    cudaDeviceProp device{};
    ASSERT_EQ(cudaGetDeviceProperties(&device, 0), cudaSuccess);
    if (device.major < 8) {
      GTEST_SKIP() << "These GQA dispatch cases require compute capability 8.0 or newer.";
    }
  }

  void RunCase(const GQAPolicyRunConfig& test_case) {
    SCOPED_TRACE(testing::Message()
                 << "policy=" << test_case.policy
                 << " phase=" << contrib::cuda::GQAExecutionPhaseName(test_case.phase));
    ScopedEnvironmentVariables scoped_env_vars{
        {{"ORT_ENABLE_XQA", "1"},
         {"ORT_DISABLE_FLASH_DECODE", "0"},
         {"ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO", "1"}}};
    SessionOptions session_options;
    session_options.graph_optimization_level = TransformerLevel::Default;
    ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
        kOrtSessionOptionsKernelDispatchPolicy, test_case.policy));
    if (test_case.declare_envelope) {
      const std::string envelope = std::to_string(test_case.capacity);
      ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
          kOrtSessionOptionsCudaGqaWorkspaceMaxTotalSequenceLength, envelope.c_str()));
    }
    ASSERT_FALSE(session_options.config_options.GetConfigEntry(
                                                   kOrtSessionOptionsResourceCudaPartitioningSettings)
                     .has_value());
    InferenceSessionWrapper session(session_options, GetEnvironment());
    CUDAExecutionProviderInfo provider_info;
    provider_info.sdpa_kernel = test_case.kernel_selection;
    auto cuda_ep = std::make_shared<CUDAExecutionProvider>(provider_info);
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(cuda_ep));

    const int64_t sequence = test_case.phase == GQAExecutionPhase::Decode ? 1 : 4;
    const std::string model = BuildGroupQueryAttentionKernelModel(
        false, false, sequence, test_case.head, test_case.capacity, 1, test_case.dynamic_sequence);
    ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
    ASSERT_STATUS_OK(session.Initialize());
    const Node* node = FindNodeByOpType(session.GetGraph(), "GroupQueryAttention");
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->GetExecutionProviderType(), kCudaExecutionProvider);
    const OpKernel* kernel = session.GetSessionState().GetKernel(node->Index());
    ASSERT_NE(kernel, nullptr);

    auto shapes = SeparateShapes(sequence, test_case.head, test_case.capacity, 1);
    if (test_case.dynamic_sequence) {
      shapes[0] = Known({1, -1, 8 * test_case.head});
    }
    if (test_case.redeclared_sequence_bound > 0) {
      shapes = SeparateShapes(test_case.redeclared_sequence_bound,
                              test_case.head, test_case.capacity, 1);
      InlinedVector<WorkspaceRequirement> requirements;
      ASSERT_STATUS_OK(kernel->DeclareWorkspaceRequirements(gsl::make_span(shapes), requirements));
      ASSERT_FALSE(requirements.empty());
    }
    auto config = Config();
    config.sliding_window_cache = false;
    config.local_window_size = -1;
    config.max_total_sequence_length = test_case.declare_envelope ? test_case.capacity : 0;
    config.static_head_size = test_case.head;
    ASSERT_TRUE(TryParseKernelDispatchPolicy(test_case.policy, config.dispatch_policy));
    const auto& options = *cuda_ep->GetAttentionKernelOptions();
    const auto estimate = EstimateGroupQueryAttentionWorkspace(
        config, gsl::make_span(shapes), cuda_ep->GetDeviceProp(), options);
    auto permitted = contrib::cuda::GetGQAReachableBackendsForPhase(
        contrib::cuda::kAllGQAReachableBackends, test_case.phase);
    if (config.dispatch_policy != KernelDispatchPolicy::Auto) {
      if (estimate.has_value()) {
        permitted = estimate->committed_backends.Get(test_case.phase);
      } else {
        ASSERT_EQ(config.dispatch_policy, KernelDispatchPolicy::Safe);
        permitted = contrib::cuda::GetGQAReachableBackendsForPhase(
            contrib::cuda::kAnalyticallyBoundedGQABackends, test_case.phase);
        InlinedVector<WorkspaceRequirement> requirements;
        ASSERT_STATUS_OK(kernel->DeclareWorkspaceRequirements(gsl::make_span(shapes), requirements));
        ASSERT_TRUE(requirements.empty());
      }
    } else if (!test_case.declare_envelope || test_case.dynamic_sequence) {
      ASSERT_FALSE(estimate.has_value());
    }
    ASSERT_EQ(contrib::cuda::IntersectGQAReachableBackends(
                  permitted, test_case.required_permitted),
              test_case.required_permitted);
    ASSERT_EQ(contrib::cuda::IntersectGQAReachableBackends(
                  permitted, test_case.forbidden_permitted),
              GQAReachableBackend::None);
    if (test_case.expect_failure) {
      ASSERT_EQ(permitted, GQAReachableBackend::None);
    } else {
      ASSERT_NE(permitted, GQAReachableBackend::None);
    }

    const auto cpu_allocator = CPUAllocator::DefaultInstance();
    NameMLValMap feeds;
    const auto add_fp16_input = [&](const char* name, TensorShapeVector dimensions, float fill) {
      const size_t count = static_cast<size_t>(TensorShape(dimensions).Size());
      std::vector<MLFloat16> data(count, MLFloat16(fill));
      OrtValue value;
      CreateMLValue<MLFloat16>(cpu_allocator, gsl::make_span(dimensions), data, &value);
      feeds.emplace(name, std::move(value));
    };
    add_fp16_input("query", {1, sequence, 8 * test_case.head}, 0.0f);
    add_fp16_input("key", {1, sequence, 2 * test_case.head}, 0.0f);
    add_fp16_input("value", {1, sequence, 2 * test_case.head}, 1.0f);
    add_fp16_input("past_key", {1, 2, test_case.capacity, test_case.head}, 0.0f);
    add_fp16_input("past_value", {1, 2, test_case.capacity, test_case.head}, 1.0f);
    const int32_t total = test_case.phase == GQAExecutionPhase::Prefill ? 4 : 8;
    OrtValue lengths;
    CreateMLValue<int32_t>(cpu_allocator, {1}, {total - 1}, &lengths);
    feeds.emplace("seqlens_k", std::move(lengths));
    OrtValue total_length;
    CreateMLValue<int32_t>(cpu_allocator, {}, {total}, &total_length);
    feeds.emplace("total_sequence_length", std::move(total_length));

    const auto gpu_allocator = session.GetSessionState().GetAllocator(
        cuda_ep->GetOrtDeviceByMemType(OrtMemTypeDefault));
    ASSERT_NE(gpu_allocator, nullptr);
    std::vector<OrtValue> fetches(3);
    Tensor::InitOrtValue(DataTypeImpl::GetType<MLFloat16>(),
                         TensorShape{1, sequence, 8 * test_case.head}, cpu_allocator, fetches[0]);
    // Explicit GPU cache outputs cannot alias the copied GPU input caches.
    for (size_t i : {size_t{1}, size_t{2}}) {
      Tensor::InitOrtValue(DataTypeImpl::GetType<MLFloat16>(),
                           TensorShape{1, 2, test_case.capacity, test_case.head}, gpu_allocator, fetches[i]);
    }

    testing::internal::CaptureStdout();
    const auto status = session.Run(feeds, {"output", "present_key", "present_value"}, &fetches);
    const std::string debug_output = testing::internal::GetCapturedStdout();
    const char* selected_backend = test_case.expect_failure ? "none" : test_case.expected_backend;
    const std::string expected_dispatch =
        std::string("GroupQueryAttentionDispatch Node=gqa Policy=") +
        KernelDispatchPolicyName(config.dispatch_policy) +
        " Phase=" + contrib::cuda::GQAExecutionPhaseName(test_case.phase) +
        " PermittedBackends=" + std::to_string(static_cast<uint32_t>(permitted)) +
        " PastPresentShareBuffer=0 SelectedBackend=" + selected_backend;
    EXPECT_NE(debug_output.find(expected_dispatch), std::string::npos) << debug_output;
    if (test_case.expect_failure) {
      ASSERT_FALSE(status.IsOK());
      EXPECT_NE(status.ErrorMessage().find("has no permitted route for the current chunked prefill workload"),
                std::string::npos)
          << status;
      return;
    }
    ASSERT_STATUS_OK(status);
    const auto output = fetches[0].Get<Tensor>().DataAsSpan<MLFloat16>();
    for (size_t i = 0; i < output.size(); ++i) {
      ASSERT_NEAR(output[i].ToFloat(), 1.0f, 1e-3f) << "output index " << i;
    }
  }
};

void CheckNonWindowedStaticCacheCapacityBound(int kernel_selection, GQABackend backend) {
  AttentionKernelOptions options;
  options.InitializeOnce(kernel_selection, true);
  auto config = Config();
  config.sliding_window_cache = false;
  config.local_window_size = -1;
  config.enable_xqa = false;
  config.max_total_sequence_length = 512;
  constexpr int64_t kPastCapacity = 1024;
  const auto estimate = EstimateGroupQueryAttentionWorkspace(
      config, SeparateShapes(/*sequence=*/1, /*head=*/64, kPastCapacity),
      Device(), options);
  ASSERT_TRUE(estimate.has_value());
  const bool use_mea = backend == GQABackend::MemoryEfficient;
  ASSERT_TRUE(HasGQAReachableBackend(
      estimate->sized_backends,
      use_mea ? GQAReachableBackend::MemoryEfficient : GQAReachableBackend::Unfused));

  GQAWorkspaceProblem problem;
  problem.qkv_element_size = 2;
  problem.cache_element_size = 2;
  problem.batch_size = 2;
  problem.sequence_length = 1;
  problem.num_heads = 8;
  problem.kv_num_heads = 2;
  problem.head_size = 64;
  problem.present_kv_cache_capacity = kPastCapacity;
  problem.requires_separate_past_buffer = true;
  problem.past_kv_cache_capacity = kPastCapacity;
  GQAConcreteRoute route;
  route.backend = backend;
  route.preparation.preprocess_mode =
      use_mea ? GQAPreprocessMode::MemoryEfficient : GQAPreprocessMode::Unfused;
  route.unfused.total_sequence_length = config.max_total_sequence_length;
  const auto runtime = GetGQACompleteWorkspaceRecipe(problem, route);
  ASSERT_TRUE(runtime.status.IsOK());
  constexpr size_t kPastBytes = 2 * 2 * kPastCapacity * 64 * 2;
  EXPECT_EQ(runtime.recipe.preparation.separate_past_bytes, kPastBytes);
  if (use_mea) {
    constexpr size_t kExpandedCacheBytes = 2 * 8 * kPastCapacity * 64 * 2;
    EXPECT_EQ(runtime.recipe.memory_efficient.expanded_key_bytes, kExpandedCacheBytes);
    EXPECT_EQ(runtime.recipe.memory_efficient.expanded_value_bytes, kExpandedCacheBytes);
  }
  EXPECT_GE(estimate->total_workspace_bytes, runtime.recipe.total_workspace_bytes);

  const auto smaller_cache = EstimateGroupQueryAttentionWorkspace(
      config, SeparateShapes(/*sequence=*/1, /*head=*/64, /*capacity=*/512),
      Device(), options);
  ASSERT_TRUE(smaller_cache.has_value());
  EXPECT_GT(estimate->total_workspace_bytes, smaller_cache->total_workspace_bytes);
}

TEST(GroupQueryAttentionWorkspaceEstimateTest, ParsesPackedAndSeparateLayouts) {
  AttentionKernelOptions options;
  options.InitializeOnce(kMath, true);
  const auto separate = EstimateGroupQueryAttentionWorkspace(
      Config(), SeparateShapes(), Device(), options);
  const auto packed = EstimateGroupQueryAttentionWorkspace(
      Config(), PackedShapes(), Device(), options);
  ASSERT_TRUE(separate.has_value());
  ASSERT_TRUE(packed.has_value());
  EXPECT_GT(separate->total_workspace_bytes, 0u);
  EXPECT_GT(packed->total_workspace_bytes, separate->total_workspace_bytes);
}

TEST(GroupQueryAttentionWorkspaceEstimateTest, MissingPastInputsDoNotProducePrefillOnlyEstimates) {
  AttentionKernelOptions options;
  options.InitializeOnce(kMath, true);
  for (bool windowed : {false, true}) {
    for (int64_t sequence : {int64_t{1}, int64_t{4}}) {
      auto config = Config();
      config.sliding_window_cache = windowed;
      config.local_window_size = windowed ? 256 : -1;
      config.max_total_sequence_length = 256;
      const auto complete_shapes = SeparateShapes(sequence);
      ASSERT_TRUE(EstimateGroupQueryAttentionWorkspace(
                      config, complete_shapes, Device(), options)
                      .has_value());
      for (int missing : {1, 2, 3}) {
        SCOPED_TRACE(testing::Message() << "windowed=" << windowed
                                        << " sequence=" << sequence << " missing=" << missing);
        auto shapes = complete_shapes;
        if ((missing & 1) != 0) shapes[3] = {};
        if ((missing & 2) != 0) shapes[4] = {};
        EXPECT_FALSE(EstimateGroupQueryAttentionWorkspace(config, shapes, Device(), options).has_value());
      }
    }
  }
}

TEST(GroupQueryAttentionWorkspaceEstimateTest, QueryBoundAdmitsEveryRuntimeClassifierOutcome) {
  AttentionKernelOptions options;
  options.InitializeOnce(kMath, true);
  for (int64_t sequence : {int64_t{1}, int64_t{4}}) {
    for (auto policy : {KernelDispatchPolicy::Auto, KernelDispatchPolicy::Latency,
                        KernelDispatchPolicy::Memory, KernelDispatchPolicy::Safe}) {
      SCOPED_TRACE(testing::Message() << "sequence=" << sequence
                                      << " policy=" << KernelDispatchPolicyName(policy));
      auto config = Config();
      config.dispatch_policy = policy;
      const auto estimate = EstimateGroupQueryAttentionWorkspace(
          config, SeparateShapes(sequence), Device(), options);
      ASSERT_TRUE(estimate.has_value());
      for (auto phase : {GQAExecutionPhase::Prefill, GQAExecutionPhase::Decode,
                         GQAExecutionPhase::ChunkedPrefill}) {
        EXPECT_EQ(estimate->committed_backends.Get(phase) != GQAReachableBackend::None,
                  phase != GQAExecutionPhase::ChunkedPrefill || sequence > 1);
      }
    }
  }
}

// Budget H512-only group ratios with rotary/sink scratch and reject malformed or inactive XQA layouts.
TEST(GroupQueryAttentionWorkspaceEstimateTest, H512EstimatorDominatesCompleteRuntimeWorkspace) {
  AttentionKernelOptions options;
  options.InitializeOnce(kMath, true);
  for (int group : {3, 33}) {
    SCOPED_TRACE(testing::Message() << "group=" << group);
    auto config = Config();
    config.num_heads = 2 * group;
    config.do_rotary = true;
    auto shapes = SeparateShapes(1, 512);
    shapes[0] = Known({2, 1, config.num_heads * 512});
    shapes[7] = Known({256, 256});
    shapes[8] = Known({256, 256});
    shapes[11] = Known({config.num_heads});
    const auto estimate = EstimateGroupQueryAttentionWorkspace(config, shapes, Device(), options);
    ASSERT_TRUE(estimate.has_value());
    EXPECT_TRUE(HasGQAReachableBackend(estimate->sized_backends, GQAReachableBackend::Xqa));

    GQAWorkspaceProblem problem;
    problem.qkv_element_size = problem.cache_element_size = 2;
    problem.batch_size = 2;
    problem.sequence_length = 1;
    problem.num_heads = config.num_heads;
    problem.kv_num_heads = config.kv_num_heads;
    problem.head_size = 512;
    problem.present_kv_cache_capacity = 256;
    problem.is_windowed_kv_cache = true;
    problem.do_rotary = true;
    GQAConcreteRoute route;
    route.backend = GQABackend::Xqa;
    route.preparation.preprocess_mode = GQAPreprocessMode::Xqa;
    route.xqa.device_major = 8;
    route.xqa.multi_processor_count = Device().multiProcessorCount;
    route.xqa.head_sink_storage = GQAXqaHeadSinkStorage::DynamicConversion;
    const auto complete = GetGQACompleteWorkspaceRecipe(problem, route);
    ASSERT_TRUE(complete.status.IsOK()) << complete.status.message;
    EXPECT_TRUE(complete.recipe.xqa.is_h512);
    EXPECT_EQ(complete.recipe.xqa.internal_scratch_bytes,
              static_cast<size_t>(2 * config.num_heads * 32 * 514 * sizeof(float)));
    EXPECT_GT(complete.recipe.xqa.dynamic_head_sink_bytes, 0U);
    EXPECT_GE(estimate->total_workspace_bytes, complete.recipe.total_workspace_bytes);

    auto invalid = complete.recipe;
    invalid.xqa.is_h512 = false;
    EXPECT_FALSE(contrib::cuda::ValidateGQACompleteWorkspaceRecipe(invalid).IsOK());
    invalid = complete.recipe;
    ++invalid.xqa.output_accumulator_bytes;
    EXPECT_FALSE(contrib::cuda::ValidateGQACompleteWorkspaceRecipe(invalid).IsOK());
    invalid = complete.recipe;
    invalid.xqa.semaphore_bytes = sizeof(int32_t);
    EXPECT_FALSE(contrib::cuda::ValidateGQACompleteWorkspaceRecipe(invalid).IsOK());

    route.backend = GQABackend::Unfused;
    route.preparation.preprocess_mode = GQAPreprocessMode::Unfused;
    route.unfused.total_sequence_length = 256;
    const auto unfused = GetGQACompleteWorkspaceRecipe(problem, route);
    ASSERT_TRUE(unfused.status.IsOK()) << unfused.status.message;
    invalid = unfused.recipe;
    invalid.xqa.is_h512 = true;
    EXPECT_FALSE(contrib::cuda::ValidateGQACompleteWorkspaceRecipe(invalid).IsOK());
  }
}

TEST(GroupQueryAttentionWorkspaceEstimateTest, NodeAdapterParsesAttributesAndTypes) {
  AttentionKernelOptions options;
  options.InitializeOnce(kMath, true);
  const auto estimate = EstimateFromNode(SeparateShapes(), options);
  ASSERT_TRUE(estimate.has_value());
  EXPECT_GT(estimate->total_workspace_bytes, 0u);
}

TEST(GroupQueryAttentionWorkspaceEstimateTest, NodeAdapterDistinguishesStaticHeadFromShapeHints) {
  WithGroupQueryAttentionNode([](const Node& node) {
    const auto config = GetGroupQueryAttentionWorkspaceEstimateConfigForTest(&node);
    ASSERT_TRUE(config.has_value());
    EXPECT_EQ(config->static_head_size, int64_t{0});
  });
  for (bool packed : {false, true}) {
    for (bool is_bf16 : {false, true}) {
      SCOPED_TRACE(testing::Message() << "packed=" << packed << " bf16=" << is_bf16);
      WithGroupQueryAttentionNode([&](const Node& node) {
        const auto config = GetGroupQueryAttentionWorkspaceEstimateConfigForTest(&node);
        ASSERT_TRUE(config.has_value());
        EXPECT_EQ(config->static_head_size, int64_t{128});
        EXPECT_EQ(config->is_bf16, is_bf16);
      },
                                  128, packed, is_bf16);
    }
  }
}

TEST(GroupQueryAttentionWorkspaceEstimateTest, NodeAdapterForwardsDispatchPolicy) {
  WithGroupQueryAttentionNode([](const Node& node) {
    const auto default_config = GetGroupQueryAttentionWorkspaceEstimateConfigForTest(&node);
    ASSERT_TRUE(default_config.has_value());
    EXPECT_EQ(default_config->dispatch_policy, KernelDispatchPolicy::Auto);

    AttentionKernelOptions options;
    options.InitializeOnce(kMath, true);
    for (const auto policy : {KernelDispatchPolicy::Auto, KernelDispatchPolicy::Latency,
                              KernelDispatchPolicy::Memory, KernelDispatchPolicy::Safe}) {
      SCOPED_TRACE(static_cast<int>(policy));
      const auto config = GetGroupQueryAttentionWorkspaceEstimateConfigForTest(
          &node, false, 4096, policy);
      ASSERT_TRUE(config.has_value());
      EXPECT_EQ(config->max_total_sequence_length, int64_t{4096});
      EXPECT_EQ(config->dispatch_policy, policy);
      const auto expected = EstimateGroupQueryAttentionWorkspace(
          *config, SeparateShapes(), Device(), options);
      ASSERT_TRUE(expected.has_value());
      const auto estimate = EstimateGroupQueryAttentionWorkspaceForTest(
          &node, SeparateShapes(), Device(), options, false, 4096, policy);
      ASSERT_TRUE(estimate.has_value());
      EXPECT_EQ(estimate->total_workspace_bytes, expected->total_workspace_bytes);
      EXPECT_EQ(estimate->sized_backends, expected->sized_backends);
      EXPECT_EQ(estimate->committed_backends, expected->committed_backends);
    }
  });
}

TEST(GroupQueryAttentionWorkspaceEstimateTest, ParsedEnvelopePreservesWindowedEstimateAndEnablesNonWindowed) {
  WithGroupQueryAttentionNode([](const Node& node) {
    AttentionKernelOptions options;
    options.InitializeOnce(kMath, true);
    const auto baseline = EstimateGroupQueryAttentionWorkspaceForTest(
        &node, SeparateShapes(), Device(), options);
    ASSERT_TRUE(baseline.has_value());

    struct Case {
      const char* value;
      int64_t expected;
    };
    for (const auto& test_case : {Case{nullptr, 0}, Case{"0", 0}, Case{"4096", 4096}}) {
      SCOPED_TRACE(test_case.value != nullptr ? test_case.value : "<unset>");
      ConfigOptions config_options;
      ASSERT_STATUS_OK(config_options.AddConfigEntry(
          kOrtSessionOptionsResourceCudaPartitioningSettings, "1000,"));
      if (test_case.value != nullptr) {
        ASSERT_STATUS_OK(config_options.AddConfigEntry(
            kOrtSessionOptionsCudaGqaWorkspaceMaxTotalSequenceLength, test_case.value));
      }
      std::optional<ResourceAccountantMap> accountants;
      ASSERT_STATUS_OK(CreateAccountants(config_options, PathString(), accountants));
      ASSERT_TRUE(accountants.has_value());
      const auto& estimator_config =
          accountants->at(kCudaExecutionProvider)->GetWorkspaceEstimatorConfig();
      const auto gqa_config = GetGroupQueryAttentionWorkspaceEstimateConfigForTest(
          &node, false, estimator_config.cuda_gqa_workspace_max_total_sequence_length);
      ASSERT_TRUE(gqa_config.has_value());
      EXPECT_EQ(gqa_config->max_total_sequence_length, test_case.expected);
      const auto estimate = EstimateGroupQueryAttentionWorkspaceForTest(
          &node, SeparateShapes(), Device(), options, false,
          estimator_config.cuda_gqa_workspace_max_total_sequence_length);
      ASSERT_TRUE(estimate.has_value());
      EXPECT_EQ(estimate->total_workspace_bytes, baseline->total_workspace_bytes);
      EXPECT_EQ(estimate->persistent_prepack_bytes, baseline->persistent_prepack_bytes);
      EXPECT_EQ(estimate->initialization_scratch_bytes, baseline->initialization_scratch_bytes);

      auto non_windowed_config = *gqa_config;
      non_windowed_config.sliding_window_cache = false;
      EXPECT_EQ(EstimateGroupQueryAttentionWorkspace(
                    non_windowed_config, SeparateShapes(), Device(), options)
                    .has_value(),
                test_case.expected > 0);
    }
  });
}

TEST(GroupQueryAttentionWorkspaceEstimateTest, GetCapabilityBudgetUsesLevel1Estimate) {
  if (!HasCudaDevice()) {
    GTEST_SKIP() << "A CUDA device is required for the budget integration test.";
  }

  ScopedEnvironmentVariables scoped_env_vars{{{"ORT_ENABLE_XQA", "1"}}};
  CUDAExecutionProviderInfo provider_info;
  provider_info.sdpa_kernel = kMath;
  const std::string model_bytes = BuildGroupQueryAttentionKernelModel();

  auto make_session_options = [](size_t memory_limit_kb) {
    SessionOptions session_options;
    session_options.graph_optimization_level = TransformerLevel::Default;
    const std::string partitioning_settings =
        std::to_string(memory_limit_kb) + ",";
    ORT_THROW_IF_ERROR(session_options.config_options.AddConfigEntry(
        kOrtSessionOptionsResourceCudaPartitioningSettings,
        partitioning_settings.c_str()));
    return session_options;
  };

  std::optional<GQAWorkspaceAggregate> estimate;
  {
    InferenceSessionWrapper session(make_session_options(1024),
                                    GetEnvironment());
    auto cuda_ep = std::make_shared<CUDAExecutionProvider>(provider_info);
    if (cuda_ep->GetDeviceProp().major < 8) {
      GTEST_SKIP() << "XQA requires compute capability 8.0 or newer.";
    }
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(cuda_ep));
    ASSERT_STATUS_OK(session.Load(model_bytes.data(), static_cast<int>(model_bytes.size())));
    ASSERT_STATUS_OK(session.Initialize());

    const Node* node = FindNodeByOpType(session.GetGraph(), "GroupQueryAttention");
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->GetExecutionProviderType(), kCudaExecutionProvider);
    auto shapes = SeparateShapes();
    shapes[11] = Known({8});
    estimate = EstimateGroupQueryAttentionWorkspaceForTest(
        node, gsl::make_span(shapes), cuda_ep->GetDeviceProp(),
        *cuda_ep->GetAttentionKernelOptions(),
        /*head_sink_is_constant_initializer=*/true);
    ASSERT_TRUE(estimate.has_value());

    auto prepacked_config = Config();
    prepacked_config.head_sink_is_prepacked = true;
    const auto prepacked = EstimateGroupQueryAttentionWorkspace(
        prepacked_config, gsl::make_span(shapes), cuda_ep->GetDeviceProp(),
        *cuda_ep->GetAttentionKernelOptions());
    ASSERT_TRUE(prepacked.has_value());
    EXPECT_GT(estimate->total_workspace_bytes, prepacked->total_workspace_bytes);
    EXPECT_EQ(estimate->persistent_prepack_bytes,
              prepacked->persistent_prepack_bytes);
    EXPECT_EQ(estimate->initialization_scratch_bytes,
              prepacked->initialization_scratch_bytes);
  }

  constexpr size_t kHeadSinkInitializerBytes = 8 * sizeof(MLFloat16);
  constexpr size_t kOutputBytes = 2 * 4 * 512 * sizeof(MLFloat16);
  constexpr size_t kPresentCacheBytes = 2 * 2 * 256 * 64 * sizeof(MLFloat16);
  constexpr size_t kAccountedTensorBytes =
      kHeadSinkInitializerBytes + kOutputBytes + 2 * kPresentCacheBytes;
  constexpr size_t kFallbackWorkspaceBytes = kAccountedTensorBytes / 2;
  ASSERT_GT(estimate->total_workspace_bytes, kFallbackWorkspaceBytes);
  EXPECT_EQ(estimate->persistent_prepack_bytes, 8 * sizeof(float));
  EXPECT_EQ(estimate->initialization_scratch_bytes,
            8 * sizeof(MLFloat16));
  const auto kilobytes_above = [](size_t bytes) {
    return bytes / 1024 + 1;
  };

  {
    const size_t accepted_limit_kb = kilobytes_above(
        kAccountedTensorBytes + estimate->total_workspace_bytes +
        estimate->persistent_prepack_bytes);
    InferenceSessionWrapper session(make_session_options(accepted_limit_kb),
                                    GetEnvironment());
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(
        std::make_shared<CUDAExecutionProvider>(provider_info)));
    ASSERT_STATUS_OK(session.Load(model_bytes.data(), static_cast<int>(model_bytes.size())));
    ASSERT_STATUS_OK(session.Initialize());

    const Node* node = FindNodeByOpType(session.GetGraph(), "GroupQueryAttention");
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->GetExecutionProviderType(), kCudaExecutionProvider);
  }

  {
    // Fallback accounting would admit this node. The larger Level-1 estimate
    // must instead make CUDA reject it.
    const size_t rejected_limit_kb = kilobytes_above(
        kAccountedTensorBytes + kFallbackWorkspaceBytes);
    ASSERT_LT(rejected_limit_kb * 1024,
              kAccountedTensorBytes + estimate->total_workspace_bytes +
                  estimate->persistent_prepack_bytes);
    InferenceSessionWrapper session(make_session_options(rejected_limit_kb),
                                    GetEnvironment());
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(
        std::make_shared<CUDAExecutionProvider>(provider_info)));
    ASSERT_STATUS_OK(session.Load(model_bytes.data(), static_cast<int>(model_bytes.size())));
    ASSERT_STATUS_OK(session.Initialize());

    const Node* node = FindNodeByOpType(session.GetGraph(), "GroupQueryAttention");
    ASSERT_NE(node, nullptr);
    EXPECT_NE(node->GetExecutionProviderType(), kCudaExecutionProvider);
  }
}

TEST(GroupQueryAttentionWorkspaceEstimateTest, CapacityAwarePartitioningRequiresProvenEstimate) {
  if (!HasCudaDevice()) {
    GTEST_SKIP() << "A CUDA device is required for the budget integration test.";
  }

  ScopedEnvironmentVariables scoped_env_vars{{{"ORT_ENABLE_XQA", "0"}}};
  const std::string model_bytes = BuildGroupQueryAttentionKernelModel(
      /*sliding_window_cache=*/false, /*include_head_sink=*/false);

  struct Case {
    int kernel_selection;
    const char* envelope;
    bool expect_cuda;
  };
  for (const auto& test_case : {
           Case{kMath, nullptr, false},
           Case{kCudnn, "512", false},
           Case{kMath, "512", true},
       }) {
    SCOPED_TRACE(testing::Message()
                 << "kernel=" << test_case.kernel_selection
                 << " envelope=" << (test_case.envelope == nullptr ? "<unset>" : test_case.envelope));
    SessionOptions session_options;
    session_options.graph_optimization_level = TransformerLevel::Default;
    ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
        kOrtSessionOptionsResourceCudaPartitioningSettings, "1048576,"));
    if (test_case.envelope != nullptr) {
      ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
          kOrtSessionOptionsCudaGqaWorkspaceMaxTotalSequenceLength, test_case.envelope));
    }

    CUDAExecutionProviderInfo provider_info;
    provider_info.sdpa_kernel = test_case.kernel_selection;
    InferenceSessionWrapper session(session_options, GetEnvironment());
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(
        std::make_shared<CUDAExecutionProvider>(provider_info)));
    ASSERT_STATUS_OK(session.Load(model_bytes.data(), static_cast<int>(model_bytes.size())));
    ASSERT_STATUS_OK(session.Initialize());

    const Node* node = FindNodeByOpType(session.GetGraph(), "GroupQueryAttention");
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->GetExecutionProviderType() == kCudaExecutionProvider,
              test_case.expect_cuda);
  }
}

TEST(GroupQueryAttentionWorkspaceEstimateTest, NonWindowedWithoutEnvelopeIsUnavailable) {
  AttentionKernelOptions options;
  options.InitializeOnce(kMath, true);
  auto config = Config();
  config.sliding_window_cache = false;
  config.local_window_size = -1;
  // total_sequence_length is a runtime scalar. Without the declared KV-length
  // envelope the non-windowed estimate cannot bound the present length, so it
  // declines -- the behavior before the envelope knob existed.
  EXPECT_FALSE(EstimateGroupQueryAttentionWorkspace(
                   config, SeparateShapes(/*sequence=*/1, /*head=*/64, /*capacity=*/128),
                   Device(), options)
                   .has_value());
}

TEST(GroupQueryAttentionWorkspaceEstimateTest, NonWindowedEnvelopeBoundsWorkspace) {
  AttentionKernelOptions options;
  options.InitializeOnce(kMath, true);
  auto config = Config();
  config.sliding_window_cache = false;
  config.local_window_size = -1;
  config.max_total_sequence_length = 512;
  const auto estimate = EstimateGroupQueryAttentionWorkspace(
      config, SeparateShapes(/*sequence=*/4, /*head=*/64, /*capacity=*/256),
      Device(), options);
  ASSERT_TRUE(estimate.has_value());
  EXPECT_GT(estimate->total_workspace_bytes, 0u);

  // The envelope is the KV length the estimate is taken over, so raising it
  // cannot lower the bound.
  auto larger = config;
  larger.max_total_sequence_length = 1024;
  const auto larger_estimate = EstimateGroupQueryAttentionWorkspace(
      larger, SeparateShapes(/*sequence=*/4, /*head=*/64, /*capacity=*/256),
      Device(), options);
  ASSERT_TRUE(larger_estimate.has_value());
  EXPECT_GE(larger_estimate->total_workspace_bytes, estimate->total_workspace_bytes);
}

TEST(GroupQueryAttentionWorkspaceEstimateTest, NonWindowedStaticCacheCapacityBoundsPreservation) {
  CheckNonWindowedStaticCacheCapacityBound(kMath, GQABackend::Unfused);
}

#if USE_MEMORY_EFFICIENT_ATTENTION
TEST(GroupQueryAttentionWorkspaceEstimateTest, NonWindowedStaticCacheCapacityBoundsMeaExpansion) {
  CheckNonWindowedStaticCacheCapacityBound(
      static_cast<int>(AttentionBackend::EFFICIENT_ATTENTION), GQABackend::MemoryEfficient);
}
#endif

TEST(GroupQueryAttentionWorkspaceEstimateTest, NonWindowedEnvelopeCannotBeSmallerThanQueryBound) {
  AttentionKernelOptions options;
  options.InitializeOnce(kMath, true);
  auto config = Config();
  config.sliding_window_cache = false;
  config.local_window_size = -1;
  config.max_total_sequence_length = 3;
  EXPECT_FALSE(EstimateGroupQueryAttentionWorkspace(
                   config, SeparateShapes(/*sequence=*/4, /*head=*/64, /*capacity=*/1024),
                   Device(), options)
                   .has_value());
}

TEST(GroupQueryAttentionWorkspaceEstimateTest, NonWindowedStaticCacheCapacityMustFitInt32) {
  AttentionKernelOptions options;
  options.InitializeOnce(kMath, true);
  auto config = Config();
  config.sliding_window_cache = false;
  config.local_window_size = -1;
  config.max_total_sequence_length = 512;
  const int64_t capacity = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
  EXPECT_FALSE(EstimateGroupQueryAttentionWorkspace(
                   config, SeparateShapes(/*sequence=*/1, /*head=*/64, capacity),
                   Device(), options)
                   .has_value());
}

TEST(GroupQueryAttentionWorkspaceBoundsTest, PartialAliasPreservationAddsFullPastCopy) {
  auto bounds = Bounds();
  bounds.reachable_backends = GQAReachableBackend::Unfused;
  const auto without = GetGQAWorkspaceAggregateForBounds(bounds);
  ASSERT_TRUE(without.status.IsOK());
  EXPECT_GT(without.total_workspace_bytes, 0u);

  // Non-windowed partial aliasing copies the full past cache, which stacks on
  // top of the selected route's workspace rather than replacing it.
  bounds.account_partial_alias_preservation = true;
  const auto with_copy = GetGQAWorkspaceAggregateForBounds(bounds);
  ASSERT_TRUE(with_copy.status.IsOK());

  const size_t expected_copy =
      static_cast<size_t>(bounds.batch_size_bound) *
      static_cast<size_t>(bounds.kv_num_heads) *
      static_cast<size_t>(bounds.present_kv_cache_capacity_bound) *
      static_cast<size_t>(bounds.head_size_bound) * bounds.cache_element_size;
  EXPECT_EQ(with_copy.total_workspace_bytes,
            without.total_workspace_bytes + expected_copy);
}

TEST(GroupQueryAttentionWorkspaceEstimateTest, SingleProvenRouteHasSameBoundAcrossPolicies) {
  AttentionKernelOptions options;
  options.InitializeOnce(kMath, true);
  for (bool windowed : {false, true}) {
    SCOPED_TRACE(windowed);
    auto config = Config();
    config.enable_xqa = false;
    EXPECT_EQ(config.dispatch_policy, KernelDispatchPolicy::Auto);
    config.sliding_window_cache = windowed;
    config.local_window_size = windowed ? 256 : -1;
    config.max_total_sequence_length = windowed ? 0 : 512;
    const auto baseline = EstimateGroupQueryAttentionWorkspace(
        config, SeparateShapes(), Device(), options);
    ASSERT_TRUE(baseline.has_value());
    for (const auto policy : {KernelDispatchPolicy::Auto, KernelDispatchPolicy::Latency,
                              KernelDispatchPolicy::Memory, KernelDispatchPolicy::Safe}) {
      config.dispatch_policy = policy;
      const auto estimate = EstimateGroupQueryAttentionWorkspace(
          config, SeparateShapes(), Device(), options);
      ASSERT_TRUE(estimate.has_value());
      EXPECT_EQ(estimate->total_workspace_bytes, baseline->total_workspace_bytes);
      EXPECT_EQ(estimate->sized_backends, baseline->sized_backends);
      EXPECT_EQ(estimate->committed_backends, baseline->committed_backends);
    }
  }
}

TEST(GroupQueryAttentionWorkspaceEstimateTest, DispatchPolicyDoesNotEnableUnspecifiedEnvelope) {
  AttentionKernelOptions options;
  options.InitializeOnce(kMath, true);
  auto config = Config();
  config.sliding_window_cache = false;
  config.local_window_size = -1;
  for (const auto policy : {KernelDispatchPolicy::Latency, KernelDispatchPolicy::Memory,
                            KernelDispatchPolicy::Safe, KernelDispatchPolicy::Auto}) {
    config.dispatch_policy = policy;
    EXPECT_FALSE(EstimateGroupQueryAttentionWorkspace(
                     config, SeparateShapes(), Device(), options)
                     .has_value());
  }
}

TEST(GroupQueryAttentionWorkspaceEstimateTest,
     SingleTokenShapeDoesNotAdmitChunkedPrefill) {
  AttentionKernelOptions options;
  options.InitializeOnce(kMath, true);
  auto config = Config();
  config.dispatch_policy = KernelDispatchPolicy::Safe;
  const auto estimate = EstimateGroupQueryAttentionWorkspace(
      config, SeparateShapes(/*sequence=*/1), Device(), options);
  ASSERT_TRUE(estimate.has_value());
  EXPECT_EQ(estimate->committed_backends.prefill,
            GQAReachableBackend::Unfused);
  EXPECT_EQ(estimate->committed_backends.decode,
            GQAReachableBackend::Unfused);
  EXPECT_EQ(estimate->committed_backends.chunked_prefill,
            GQAReachableBackend::None);
}

TEST(GroupQueryAttentionWorkspaceEstimateTest, RejectsCacheCapacityDifferentFromWindow) {
  AttentionKernelOptions options;
  options.InitializeOnce(kMath, true);
  auto config = Config();
  config.local_window_size = 128;
  EXPECT_FALSE(EstimateGroupQueryAttentionWorkspace(
                   config, SeparateShapes(), Device(), options)
                   .has_value());
  EXPECT_TRUE(EstimateGroupQueryAttentionWorkspace(
                  config, SeparateShapes(/*sequence=*/4, /*head=*/64, /*capacity=*/128),
                  Device(), options)
                  .has_value());

  auto shapes = SeparateShapes(/*sequence=*/4, /*head=*/64, /*capacity=*/128);
  shapes[4] = Known({2, 2, 256, 64});
  EXPECT_FALSE(EstimateGroupQueryAttentionWorkspace(
                   config, shapes, Device(), options)
                   .has_value());
}

TEST(GroupQueryAttentionWorkspaceEstimateTest, RejectsOptionalHolesAndMalformedGeometry) {
  AttentionKernelOptions options;
  options.InitializeOnce(kMath, true);
  auto shapes = SeparateShapes();
  shapes[2] = {};
  EXPECT_FALSE(EstimateGroupQueryAttentionWorkspace(
                   Config(), shapes, Device(), options)
                   .has_value());

  shapes = SeparateShapes();
  shapes[3] = Known({2, 2, 256});
  EXPECT_FALSE(EstimateGroupQueryAttentionWorkspace(
                   Config(), shapes, Device(), options)
                   .has_value());

  shapes = SeparateShapes();
  shapes[0] = WorkspaceInputShape::PresentWithoutShape();
  EXPECT_FALSE(EstimateGroupQueryAttentionWorkspace(
                   Config(), shapes, Device(), options)
                   .has_value());
}

TEST(GroupQueryAttentionWorkspaceEstimateTest, RejectsBiasWithoutHeadSink) {
  AttentionKernelOptions options;
  options.InitializeOnce(kMath, true);
  auto shapes = SeparateShapes();
  shapes[10] = Known({2, 8, 4});
  EXPECT_FALSE(EstimateGroupQueryAttentionWorkspace(
                   Config(), shapes, Device(), options)
                   .has_value());

  shapes[10] = Known({2, 8, 4, 256});
  EXPECT_FALSE(EstimateGroupQueryAttentionWorkspace(
                   Config(), shapes, Device(), options)
                   .has_value());
}

TEST(GroupQueryAttentionWorkspaceEstimateTest, RejectsUndersizedPositionIds) {
  AttentionKernelOptions options;
  options.InitializeOnce(kMath, true);
  auto shapes = SeparateShapes();
  shapes[9] = Known({1, 1});
  EXPECT_FALSE(EstimateGroupQueryAttentionWorkspace(
                   Config(), shapes, Device(), options)
                   .has_value());

  shapes[9] = Known({2, 4});
  EXPECT_TRUE(EstimateGroupQueryAttentionWorkspace(
                  Config(), shapes, Device(), options)
                  .has_value());
}

TEST(GroupQueryAttentionWorkspaceEstimateTest, ValidatesPartialRotaryCaches) {
  AttentionKernelOptions options;
  options.InitializeOnce(kMath, true);
  auto config = Config();
  config.do_rotary = true;
  auto shapes = SeparateShapes();
  constexpr int64_t kMaxDimension = std::numeric_limits<int64_t>::max();
  shapes[7] = Known({kMaxDimension, 8});
  shapes[8] = Known({kMaxDimension, 8});
  EXPECT_TRUE(EstimateGroupQueryAttentionWorkspace(
                  config, shapes, Device(), options)
                  .has_value());

  shapes[8] = Known({kMaxDimension, 16});
  EXPECT_FALSE(EstimateGroupQueryAttentionWorkspace(
                   config, shapes, Device(), options)
                   .has_value());

  shapes[7] = Known({kMaxDimension, 40});
  shapes[8] = Known({kMaxDimension, 40});
  EXPECT_FALSE(EstimateGroupQueryAttentionWorkspace(
                   config, shapes, Device(), options)
                   .has_value());
}

TEST(GroupQueryAttentionWorkspaceEstimateTest, RejectsNoncausalLocalWindow) {
  AttentionKernelOptions options;
  options.InitializeOnce(kMath, true);
  auto config = Config();
  config.causal = 0;
  EXPECT_FALSE(EstimateGroupQueryAttentionWorkspace(
                   config, SeparateShapes(), Device(), options)
                   .has_value());
}

TEST(GroupQueryAttentionWorkspaceEstimateTest, ValidatesQuantizedCacheMetadata) {
  AttentionKernelOptions options;
  options.InitializeOnce(kFlash | kMath, true);
  for (int bit_width : {8, 4}) {
    auto config = Config();
    config.cache_element_size = 1;
    config.kv_cache_bit_width = bit_width;
    config.k_quantization = GQAKvQuantizationType::PerTensor;
    config.v_quantization = GQAKvQuantizationType::PerChannel;
    auto shapes = SeparateShapes(/*sequence=*/4, /*head=*/128);
    const int64_t stored_head = bit_width == 4 ? 64 : 128;
    shapes[3] = Known({2, 2, 256, stored_head});
    shapes[4] = Known({2, 2, 256, stored_head});
    shapes[12] = Known({1});
    shapes[13] = Known({1, 2, 1, 128});
    EXPECT_TRUE(EstimateGroupQueryAttentionWorkspace(
                    config, shapes, Device(), options)
                    .has_value());
  }

  auto config = Config();
  config.cache_element_size = 1;
  config.kv_cache_bit_width = 8;
  config.k_quantization = GQAKvQuantizationType::PerTensor;
  config.v_quantization = GQAKvQuantizationType::PerTensor;
  auto shapes = SeparateShapes();
  shapes[12] = Known({1});
  EXPECT_FALSE(EstimateGroupQueryAttentionWorkspace(
                   config, shapes, Device(), options)
                   .has_value());
}

TEST(GroupQueryAttentionWorkspaceBoundsTest, FindsSmallerSupportedHeadRoutes) {
  auto bounds = Bounds();
  bounds.head_size_bound = 263;
  bounds.reachable_backends =
      GQAReachableBackend::Xqa | GQAReachableBackend::Flash |
      GQAReachableBackend::MemoryEfficient;
  const auto estimate = GetGQAWorkspaceAggregateForBounds(bounds);
  ASSERT_TRUE(estimate.status.IsOK()) << estimate.status.message;
  EXPECT_TRUE(HasGQAReachableBackend(estimate.sized_backends,
                                     GQAReachableBackend::Xqa));
  EXPECT_TRUE(HasGQAReachableBackend(estimate.sized_backends,
                                     GQAReachableBackend::Flash));
}

TEST(GroupQueryAttentionWorkspaceBoundsTest, PromptDecodeAndWindowRoutesAreBounded) {
  auto bounds = Bounds();
  bounds.reachable_backends = GQAReachableBackend::Unfused;
  bounds.is_windowed_kv_cache = true;
  const auto mixed = GetGQAWorkspaceAggregateForBounds(bounds);
  ASSERT_TRUE(mixed.status.IsOK());

  bounds.prefill_reachable = false;
  bounds.chunked_prefill_reachable = false;
  bounds.sequence_length_bound = 1;
  const auto decode = GetGQAWorkspaceAggregateForBounds(bounds);
  ASSERT_TRUE(decode.status.IsOK());
  EXPECT_GT(mixed.total_workspace_bytes, decode.total_workspace_bytes);

  bounds.decode_reachable = false;
  bounds.prefill_reachable = true;
  bounds.sequence_length_bound = 4;
  const auto prompt = GetGQAWorkspaceAggregateForBounds(bounds);
  EXPECT_TRUE(prompt.status.IsOK());
}

TEST(GroupQueryAttentionWorkspaceBoundsTest, WindowedBoundsUseFinalCapacityAndTransientExtent) {
  constexpr int64_t kCapacity = 256;
  constexpr int64_t kSequence = 4;
  constexpr int64_t kLargeTotalSequenceLength = 1'000'000;
  auto bounds = Bounds();
  bounds.present_kv_cache_capacity_bound = kCapacity;
  bounds.sequence_length_bound = kSequence;
  bounds.is_windowed_kv_cache = true;

  for (GQAReachableBackend backend :
       {GQAReachableBackend::Flash, GQAReachableBackend::MemoryEfficient,
        GQAReachableBackend::Unfused}) {
    bounds.reachable_backends = backend;
    const auto estimate = GetGQAWorkspaceAggregateForBounds(bounds);
    ASSERT_TRUE(estimate.status.IsOK()) << estimate.status.message;

    GQAWorkspaceProblem problem;
    problem.qkv_element_size = bounds.qkv_element_size;
    problem.cache_element_size = bounds.cache_element_size;
    problem.batch_size = bounds.batch_size_bound;
    problem.sequence_length = kSequence;
    problem.num_heads = bounds.num_heads;
    problem.kv_num_heads = bounds.kv_num_heads;
    problem.head_size = bounds.head_size_bound;
    problem.present_kv_cache_capacity = kCapacity;
    problem.is_windowed_kv_cache = true;

    GQAConcreteRoute route;
    route.backend = backend == GQAReachableBackend::Flash
                        ? GQABackend::Flash
                        : (backend == GQAReachableBackend::MemoryEfficient
                               ? GQABackend::MemoryEfficient
                               : GQABackend::Unfused);
    route.preparation.preprocess_mode =
        backend == GQAReachableBackend::Flash
            ? GQAPreprocessMode::Flash
            : (backend == GQAReachableBackend::MemoryEfficient
                   ? GQAPreprocessMode::MemoryEfficient
                   : GQAPreprocessMode::Unfused);
    const int64_t effective_kv_length = GetGQAEffectiveWorkspaceKvLength(
        kLargeTotalSequenceLength, kCapacity + kSequence, true);
    route.flash.total_sequence_length = effective_kv_length;
    route.flash.multi_processor_count = bounds.multi_processor_count;
    route.unfused.total_sequence_length = effective_kv_length;

    const auto concrete = GetGQACompleteWorkspaceRecipe(problem, route);
    ASSERT_TRUE(concrete.status.IsOK()) << concrete.status.message;
    EXPECT_EQ(concrete.recipe.preparation.effective_kv_cache_capacity,
              kCapacity + kSequence);
    EXPECT_EQ(concrete.recipe.preparation.staged_key_bytes,
              static_cast<size_t>(bounds.batch_size_bound * bounds.kv_num_heads *
                                  (kCapacity + kSequence) * bounds.head_size_bound *
                                  bounds.cache_element_size));
    if (backend == GQAReachableBackend::MemoryEfficient) {
      EXPECT_EQ(concrete.recipe.memory_efficient.effective_kv_cache_capacity,
                kCapacity + kSequence);
    } else if (backend == GQAReachableBackend::Flash) {
      EXPECT_EQ(concrete.recipe.flash.split_heuristic_kv_length,
                static_cast<size_t>(kCapacity + kSequence));
    } else {
      const size_t qk_bytes =
          static_cast<size_t>(bounds.batch_size_bound * bounds.num_heads *
                              kSequence * (kCapacity + kSequence) * sizeof(float));
      EXPECT_EQ(concrete.recipe.unfused.qk_bytes, qk_bytes);
    }
    EXPECT_GE(estimate.total_workspace_bytes,
              concrete.recipe.total_workspace_bytes);
  }

  bounds.sequence_length_bound = 1;
  bounds.prefill_reachable = false;
  bounds.chunked_prefill_reachable = false;
  bounds.reachable_backends = GQAReachableBackend::Unfused;
  const auto decode = GetGQAWorkspaceAggregateForBounds(bounds);
  ASSERT_TRUE(decode.status.IsOK());

  GQAWorkspaceProblem decode_problem;
  decode_problem.qkv_element_size = bounds.qkv_element_size;
  decode_problem.cache_element_size = bounds.cache_element_size;
  decode_problem.batch_size = bounds.batch_size_bound;
  decode_problem.sequence_length = 1;
  decode_problem.num_heads = bounds.num_heads;
  decode_problem.kv_num_heads = bounds.kv_num_heads;
  decode_problem.head_size = bounds.head_size_bound;
  decode_problem.present_kv_cache_capacity = kCapacity;
  decode_problem.is_windowed_kv_cache = true;
  GQAConcreteRoute decode_route;
  decode_route.backend = GQABackend::Unfused;
  decode_route.preparation.preprocess_mode = GQAPreprocessMode::Unfused;
  decode_route.unfused.total_sequence_length = GetGQAEffectiveWorkspaceKvLength(
      kLargeTotalSequenceLength, kCapacity, true);
  const auto concrete_decode =
      GetGQACompleteWorkspaceRecipe(decode_problem, decode_route);
  ASSERT_TRUE(concrete_decode.status.IsOK());
  EXPECT_EQ(concrete_decode.recipe.preparation.effective_kv_cache_capacity,
            kCapacity);
  EXPECT_GE(decode.total_workspace_bytes,
            concrete_decode.recipe.total_workspace_bytes);
}

TEST(GroupQueryAttentionWorkspaceBoundsTest, FlashEnvelopeDominatesDiscontinuousEndpoints) {
  auto bounds = Bounds();
  bounds.batch_size_bound = 1;
  bounds.sequence_length_bound = 1;
  bounds.num_heads = 2;
  bounds.kv_num_heads = 2;
  bounds.present_kv_cache_capacity_bound = 13830;
  bounds.prefill_reachable = false;
  bounds.chunked_prefill_reachable = false;
  bounds.reachable_backends = GQAReachableBackend::Flash;
  const auto envelope = GetGQAWorkspaceAggregateForBounds(bounds);
  ASSERT_TRUE(envelope.status.IsOK());

  for (int64_t kv_length = 13820; kv_length <= 13830; ++kv_length) {
    GQAWorkspaceProblem problem;
    problem.qkv_element_size = 2;
    problem.cache_element_size = 2;
    problem.batch_size = 1;
    problem.sequence_length = 1;
    problem.num_heads = 2;
    problem.kv_num_heads = 2;
    problem.head_size = 64;
    problem.present_kv_cache_capacity = kv_length;

    GQAFlashConfig flash;
    flash.total_sequence_length = kv_length;
    flash.multi_processor_count = 108;
    const auto flash_recipe = GetGQAFlashWorkspaceRecipe(problem, flash);
    ASSERT_TRUE(flash_recipe.status.IsOK()) << kv_length;

    GQAConcreteRoute route;
    route.backend = GQABackend::Flash;
    route.preparation.preprocess_mode = GQAPreprocessMode::Flash;
    route.flash = flash;
    const auto concrete = GetGQACompleteWorkspaceRecipe(problem, route);
    ASSERT_TRUE(concrete.status.IsOK()) << kv_length;
    EXPECT_GE(envelope.total_workspace_bytes,
              concrete.recipe.total_workspace_bytes)
        << kv_length << " selected splits "
        << flash_recipe.recipe.selected_split_count;
  }
}

TEST(GroupQueryAttentionWorkspaceBoundsTest, FlashFastDecodeEnvelopeCoversSingleTokenDecode) {
  auto bounds = Bounds();
  bounds.batch_size_bound = 1;
  bounds.sequence_length_bound = 130;
  bounds.num_heads = 8;
  bounds.kv_num_heads = 2;
  bounds.head_size_bound = 128;
  bounds.present_kv_cache_capacity_bound = 1024;
  bounds.is_windowed_kv_cache = false;
  bounds.local_window_size = -1;
  bounds.prefill_reachable = false;
  bounds.chunked_prefill_reachable = false;
  bounds.reachable_backends = GQAReachableBackend::FlashFastDecode;
  const auto envelope = GetGQAWorkspaceAggregateForBounds(bounds);
  ASSERT_TRUE(envelope.status.IsOK()) << envelope.status.message;

  GQAWorkspaceProblem problem;
  problem.qkv_element_size = bounds.qkv_element_size;
  problem.cache_element_size = bounds.cache_element_size;
  problem.batch_size = bounds.batch_size_bound;
  problem.sequence_length = 1;
  problem.num_heads = bounds.num_heads;
  problem.kv_num_heads = bounds.kv_num_heads;
  problem.head_size = bounds.head_size_bound;
  problem.present_kv_cache_capacity = bounds.present_kv_cache_capacity_bound;

  GQAConcreteRoute route;
  route.backend = GQABackend::Flash;
  route.preparation.preprocess_mode = GQAPreprocessMode::Flash;
  route.preparation.use_flash_attention_fast_decode = true;
  route.flash.total_sequence_length = bounds.present_kv_cache_capacity_bound;
  route.flash.multi_processor_count = bounds.multi_processor_count;
  route.flash.fast_decode = true;
  const auto concrete = GetGQACompleteWorkspaceRecipe(problem, route);
  ASSERT_TRUE(concrete.status.IsOK()) << concrete.status.message;
  EXPECT_GE(envelope.total_workspace_bytes,
            concrete.recipe.total_workspace_bytes);
}

TEST(GroupQueryAttentionWorkspaceBoundsTest, AggregatesExclusiveRoutesWithMax) {
  auto xqa = Bounds();
  xqa.prefill_reachable = false;
  xqa.chunked_prefill_reachable = false;
  xqa.reachable_backends = GQAReachableBackend::Xqa;
  xqa.xqa_head_sink_storage = GQAXqaHeadSinkStorage::DynamicConversion;
  xqa.head_sink_may_be_prepacked = true;
  const auto xqa_only = GetGQAWorkspaceAggregateForBounds(xqa);
  ASSERT_TRUE(xqa_only.status.IsOK());
  auto unfused = xqa;
  unfused.reachable_backends = GQAReachableBackend::Unfused;
  const auto unfused_only = GetGQAWorkspaceAggregateForBounds(unfused);
  ASSERT_TRUE(unfused_only.status.IsOK());
  ASSERT_GT(unfused_only.total_workspace_bytes,
            xqa_only.total_workspace_bytes);
  auto both = xqa;
  both.reachable_backends =
      GQAReachableBackend::Xqa | GQAReachableBackend::Unfused;
  const auto aggregate = GetGQAWorkspaceAggregateForBounds(both);
  ASSERT_TRUE(aggregate.status.IsOK());
  EXPECT_EQ(aggregate.total_workspace_bytes,
            std::max(xqa_only.total_workspace_bytes,
                     unfused_only.total_workspace_bytes));
  EXPECT_EQ(aggregate.persistent_prepack_bytes,
            static_cast<size_t>(xqa.num_heads) * sizeof(float));
  EXPECT_EQ(aggregate.initialization_scratch_bytes,
            static_cast<size_t>(xqa.num_heads) * xqa.qkv_element_size);
}

TEST(GroupQueryAttentionWorkspaceBoundsTest,
     ExactPrepackedStateImpliesPrepackLifetimes) {
  auto bounds = Bounds();
  bounds.prefill_reachable = false;
  bounds.chunked_prefill_reachable = false;
  bounds.reachable_backends = GQAReachableBackend::Xqa;
  bounds.xqa_head_sink_storage = GQAXqaHeadSinkStorage::PrepackedFp32;
  const auto aggregate = GetGQAWorkspaceAggregateForBounds(bounds);
  ASSERT_TRUE(aggregate.status.IsOK());
  EXPECT_EQ(aggregate.persistent_prepack_bytes,
            static_cast<size_t>(bounds.num_heads) * sizeof(float));
  EXPECT_EQ(aggregate.initialization_scratch_bytes,
            static_cast<size_t>(bounds.num_heads) * bounds.qkv_element_size);
}

TEST(GroupQueryAttentionWorkspaceBoundsTest, CudnnReachabilityIsUnavailable) {
  auto bounds = Bounds();
  bounds.reachable_backends =
      GQAReachableBackend::Unfused | GQAReachableBackend::Cudnn;
  EXPECT_FALSE(GetGQAWorkspaceAggregateForBounds(bounds).status.IsOK());

  const auto safe = GetGQAWorkspaceAggregateForBounds(
      bounds, KernelDispatchPolicy::Safe);
  ASSERT_TRUE(safe.status.IsOK());
  EXPECT_EQ(safe.committed_backends.prefill,
            GQAReachableBackend::Unfused);
  EXPECT_EQ(safe.committed_backends.decode,
            GQAReachableBackend::Unfused);
  EXPECT_EQ(safe.committed_backends.chunked_prefill,
            GQAReachableBackend::Unfused);
  EXPECT_EQ(safe.sized_backends, GQAReachableBackend::Unfused);

  const auto memory = GetGQAWorkspaceAggregateForBounds(
      bounds, KernelDispatchPolicy::Memory);
  ASSERT_TRUE(memory.status.IsOK());
  EXPECT_EQ(memory.committed_backends.prefill,
            GQAReachableBackend::Unfused);
  EXPECT_EQ(memory.committed_backends.decode,
            GQAReachableBackend::Unfused);
  EXPECT_EQ(memory.committed_backends.chunked_prefill,
            GQAReachableBackend::Unfused);

  bounds.reachable_backends = GQAReachableBackend::Cudnn;
  EXPECT_FALSE(GetGQAWorkspaceAggregateForBounds(
                   bounds, KernelDispatchPolicy::Safe)
                   .status.IsOK());
  EXPECT_FALSE(GetGQAWorkspaceAggregateForBounds(
                   bounds, KernelDispatchPolicy::Memory)
                   .status.IsOK());

  bounds.reachable_backends = GQAReachableBackend::Unfused;
  EXPECT_TRUE(GetGQAWorkspaceAggregateForBounds(bounds).status.IsOK());
}

TEST(GroupQueryAttentionWorkspaceBoundsTest,
     DecodeOnlyMemoryPolicyRetainsCoveringFallback) {
  auto bounds = Bounds();
  bounds.prefill_reachable = false;
  bounds.decode_reachable = true;
  bounds.chunked_prefill_reachable = false;
  bounds.reachable_backends =
      GQAReachableBackend::Xqa | GQAReachableBackend::Unfused;

  const auto latency = GetGQAWorkspaceAggregateForBounds(
      bounds, KernelDispatchPolicy::Latency);
  ASSERT_TRUE(latency.status.IsOK());
  EXPECT_EQ(latency.committed_backends.prefill,
            GQAReachableBackend::None);
  EXPECT_EQ(latency.committed_backends.decode,
            bounds.reachable_backends);
  EXPECT_EQ(latency.committed_backends.chunked_prefill,
            GQAReachableBackend::None);

  auto xqa_bounds = bounds;
  xqa_bounds.reachable_backends = GQAReachableBackend::Xqa;
  const auto xqa = GetGQAWorkspaceAggregateForBounds(xqa_bounds);
  ASSERT_TRUE(xqa.status.IsOK());
  auto unfused_bounds = bounds;
  unfused_bounds.reachable_backends = GQAReachableBackend::Unfused;
  const auto unfused = GetGQAWorkspaceAggregateForBounds(unfused_bounds);
  ASSERT_TRUE(unfused.status.IsOK());

  const auto memory = GetGQAWorkspaceAggregateForBounds(
      bounds, KernelDispatchPolicy::Memory);
  ASSERT_TRUE(memory.status.IsOK());
  const GQAReachableBackend expected =
      xqa.total_workspace_bytes <= unfused.total_workspace_bytes
          ? GQAReachableBackend::Xqa | GQAReachableBackend::Unfused
          : GQAReachableBackend::Unfused;
  EXPECT_EQ(memory.committed_backends.decode, expected);
  EXPECT_EQ(memory.sized_backends, expected);
  EXPECT_EQ(memory.total_workspace_bytes,
            unfused.total_workspace_bytes);
  EXPECT_LE(memory.total_workspace_bytes, latency.total_workspace_bytes);
}

TEST(GroupQueryAttentionWorkspaceBoundsTest,
     MemoryPolicyRetainsDecodeCoveringFallback) {
  auto bounds = Bounds();
  bounds.present_kv_cache_capacity_bound = 4096;
  bounds.reachable_backends =
      GQAReachableBackend::Xqa | GQAReachableBackend::Unfused;

  auto decode_xqa_bounds = bounds;
  decode_xqa_bounds.prefill_reachable = false;
  decode_xqa_bounds.chunked_prefill_reachable = false;
  decode_xqa_bounds.reachable_backends = GQAReachableBackend::Xqa;
  const auto decode_xqa =
      GetGQAWorkspaceAggregateForBounds(decode_xqa_bounds);
  ASSERT_TRUE(decode_xqa.status.IsOK());

  auto decode_unfused_bounds = decode_xqa_bounds;
  decode_unfused_bounds.reachable_backends =
      GQAReachableBackend::Unfused;
  const auto decode_unfused =
      GetGQAWorkspaceAggregateForBounds(decode_unfused_bounds);
  ASSERT_TRUE(decode_unfused.status.IsOK());
  ASSERT_LT(decode_xqa.total_workspace_bytes,
            decode_unfused.total_workspace_bytes);

  const auto memory = GetGQAWorkspaceAggregateForBounds(
      bounds, KernelDispatchPolicy::Memory);
  ASSERT_TRUE(memory.status.IsOK());
  EXPECT_EQ(memory.committed_backends.prefill,
            GQAReachableBackend::Unfused);
  EXPECT_EQ(memory.committed_backends.chunked_prefill,
            GQAReachableBackend::Unfused);
  EXPECT_TRUE(HasGQAReachableBackend(
      memory.committed_backends.decode, GQAReachableBackend::Xqa));
  EXPECT_TRUE(HasGQAReachableBackend(
      memory.committed_backends.decode, GQAReachableBackend::Unfused));
  EXPECT_EQ(memory.sized_backends,
            memory.committed_backends.Union());
}

TEST(GroupQueryAttentionWorkspaceBoundsTest, MemoryDecodeDoesNotInheritPrefillSelection) {
  auto bounds = Bounds();
  bounds.batch_size_bound = 1;
  bounds.sequence_length_bound = 128;
  bounds.head_size_bound = 256;
  bounds.present_kv_cache_capacity_bound = 3072;
  const auto general = GQAReachableBackend::Flash |
                       GQAReachableBackend::MemoryEfficient |
                       GQAReachableBackend::Unfused;
  bounds.reachable_backends = general | GQAReachableBackend::Xqa;
  bounds.covering_backends = {general, general, general};
  const auto memory = GetGQAWorkspaceAggregateForBounds(bounds, KernelDispatchPolicy::Memory);
  ASSERT_TRUE(memory.status.IsOK()) << memory.status.message;
  EXPECT_EQ(memory.committed_backends.prefill, GQAReachableBackend::MemoryEfficient);
  EXPECT_EQ(memory.committed_backends.chunked_prefill, GQAReachableBackend::MemoryEfficient);
  EXPECT_TRUE(HasGQAReachableBackend(memory.committed_backends.decode, GQAReachableBackend::Unfused));
  EXPECT_FALSE(HasGQAReachableBackend(memory.committed_backends.decode, GQAReachableBackend::MemoryEfficient));

  bounds.prefill_reachable = false;
  bounds.chunked_prefill_reachable = false;
  const auto decode_only = GetGQAWorkspaceAggregateForBounds(bounds, KernelDispatchPolicy::Memory);
  ASSERT_TRUE(decode_only.status.IsOK()) << decode_only.status.message;
  EXPECT_EQ(memory.committed_backends.decode, decode_only.committed_backends.decode);
  EXPECT_LT(decode_only.total_workspace_bytes, memory.total_workspace_bytes);
}

TEST(GroupQueryAttentionWorkspaceBoundsTest,
     PhaseSpecificMasksExcludeDecodeOnlyRoutesFromGeneralPhases) {
  auto bounds = Bounds();
  bounds.reachable_backends =
      GQAReachableBackend::Xqa | GQAReachableBackend::Flash |
      GQAReachableBackend::FlashFastDecode;

  const auto safe = GetGQAWorkspaceAggregateForBounds(
      bounds, KernelDispatchPolicy::Safe);
  ASSERT_TRUE(safe.status.IsOK()) << safe.status.message;
  EXPECT_EQ(safe.committed_backends.prefill,
            GQAReachableBackend::Flash);
  EXPECT_EQ(safe.committed_backends.chunked_prefill,
            GQAReachableBackend::Flash);
  EXPECT_EQ(safe.committed_backends.decode,
            bounds.reachable_backends);
}

TEST(GroupQueryAttentionWorkspaceBoundsTest,
     WorkspaceIsMaximumAcrossAdmittedPhaseRouteSets) {
  auto bounds = Bounds();
  bounds.reachable_backends =
      GQAReachableBackend::Xqa | GQAReachableBackend::Flash;

  const auto all_phases = GetGQAWorkspaceAggregateForBounds(
      bounds, KernelDispatchPolicy::Safe);
  ASSERT_TRUE(all_phases.status.IsOK()) << all_phases.status.message;

  auto prefill_bounds = bounds;
  prefill_bounds.decode_reachable = false;
  prefill_bounds.chunked_prefill_reachable = false;
  const auto prefill = GetGQAWorkspaceAggregateForBounds(
      prefill_bounds, KernelDispatchPolicy::Safe);
  ASSERT_TRUE(prefill.status.IsOK()) << prefill.status.message;

  auto decode_bounds = bounds;
  decode_bounds.prefill_reachable = false;
  decode_bounds.chunked_prefill_reachable = false;
  const auto decode = GetGQAWorkspaceAggregateForBounds(
      decode_bounds, KernelDispatchPolicy::Safe);
  ASSERT_TRUE(decode.status.IsOK()) << decode.status.message;

  auto chunked_bounds = bounds;
  chunked_bounds.prefill_reachable = false;
  chunked_bounds.decode_reachable = false;
  const auto chunked = GetGQAWorkspaceAggregateForBounds(
      chunked_bounds, KernelDispatchPolicy::Safe);
  ASSERT_TRUE(chunked.status.IsOK()) << chunked.status.message;

  EXPECT_EQ(all_phases.total_workspace_bytes,
            std::max({prefill.total_workspace_bytes,
                      decode.total_workspace_bytes,
                      chunked.total_workspace_bytes}));
}

TEST(GroupQueryAttentionWorkspaceEstimateTest, DeclaresOneAlignedSlotAndOnlyWorkspaceAtLevel1) {
  AttentionKernelOptions options;
  options.InitializeOnce(kMath, true);
  const auto estimate = EstimateGroupQueryAttentionWorkspace(
      Config(), SeparateShapes(), Device(), options);
  ASSERT_TRUE(estimate.has_value());
  InlinedVector<WorkspaceRequirement> requirements;
  SetGroupQueryAttentionWorkspaceRequirements(*estimate, requirements);
  ASSERT_EQ(requirements.size(), 1u);
  EXPECT_EQ(requirements[0].size_bytes, estimate->total_workspace_bytes);
  EXPECT_EQ(requirements[0].slot_id, 0);
  EXPECT_EQ(requirements[0].alignment_bytes, 256u);

  Level1MemoryEstimate level1;
  level1.runtime_transient_bytes = 17;
  level1.persistent_prepack_bytes = 19;
  level1.initialization_scratch_bytes = 23;
  SetGroupQueryAttentionLevel1MemoryEstimate(*estimate, level1);
  EXPECT_EQ(level1.runtime_workspace_bytes, estimate->total_workspace_bytes);
  EXPECT_EQ(level1.runtime_transient_bytes, 17u);
  EXPECT_EQ(level1.persistent_prepack_bytes, 0u);
  EXPECT_EQ(level1.initialization_scratch_bytes, 0u);

  auto lifetime_estimate = *estimate;
  lifetime_estimate.persistent_prepack_bytes = 29;
  lifetime_estimate.initialization_scratch_bytes = 31;
  SetGroupQueryAttentionLevel1MemoryEstimate(lifetime_estimate, level1);
  EXPECT_EQ(level1.persistent_prepack_bytes, 29u);
  EXPECT_EQ(level1.initialization_scratch_bytes, 31u);

  auto unavailable = *estimate;
  unavailable.status.error = contrib::cuda::GQAWorkspaceError::Unavailable;
  SetGroupQueryAttentionWorkspaceRequirements(unavailable, requirements);
  EXPECT_TRUE(requirements.empty());

  GQAWorkspaceAggregate zero;
  zero.status = {};
  Level1MemoryEstimate zero_level1;
  SetGroupQueryAttentionLevel1MemoryEstimate(zero, zero_level1);
  EXPECT_FALSE(zero_level1.runtime_workspace_bytes.has_value());
}

TEST(GroupQueryAttentionWorkspaceEstimateTest, KernelDeclaresPrepackedHeadSinkRoot) {
  if (!HasCudaDevice()) {
    GTEST_SKIP() << "A CUDA device is required to construct the CUDA kernel.";
  }

  ScopedEnvironmentVariables scoped_env_vars{{{"ORT_ENABLE_XQA", "1"}}};
  SessionOptions session_options;
  session_options.graph_optimization_level = TransformerLevel::Default;
  session_options.session_logid = "GroupQueryAttentionWorkspaceLevel2";
  InferenceSessionWrapper session(session_options, GetEnvironment());

  CUDAExecutionProviderInfo provider_info;
  provider_info.sdpa_kernel = kMath;
  auto cuda_ep = std::make_shared<CUDAExecutionProvider>(provider_info);
  if (cuda_ep->GetDeviceProp().major < 8) {
    GTEST_SKIP() << "XQA requires compute capability 8.0 or newer.";
  }
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(cuda_ep));
  const std::string model_bytes = BuildGroupQueryAttentionKernelModel();
  ASSERT_STATUS_OK(session.Load(model_bytes.data(), static_cast<int>(model_bytes.size())));
  ASSERT_STATUS_OK(session.Initialize());

  const Graph& graph = session.GetGraph();
  const Node* node = FindNodeByOpType(graph, "GroupQueryAttention");
  ASSERT_NE(node, nullptr);
  ASSERT_EQ(node->GetExecutionProviderType(), kCudaExecutionProvider);
  const OpKernel* kernel = session.GetSessionState().GetKernel(node->Index());
  ASSERT_NE(kernel, nullptr);

  auto shapes = SeparateShapes();
  shapes[11] = Known({8});
  auto expected_config = Config();
  expected_config.head_sink_is_prepacked = true;
  const auto expected = EstimateGroupQueryAttentionWorkspace(
      expected_config, gsl::make_span(shapes), cuda_ep->GetDeviceProp(),
      *cuda_ep->GetAttentionKernelOptions());
  ASSERT_TRUE(expected.has_value());

  auto dynamic_config = expected_config;
  dynamic_config.head_sink_is_prepacked = false;
  const auto dynamic = EstimateGroupQueryAttentionWorkspace(
      dynamic_config, gsl::make_span(shapes), cuda_ep->GetDeviceProp(),
      *cuda_ep->GetAttentionKernelOptions());
  ASSERT_TRUE(dynamic.has_value());
  EXPECT_LT(expected->total_workspace_bytes, dynamic->total_workspace_bytes);

  InlinedVector<WorkspaceRequirement> requirements;
  ASSERT_STATUS_OK(kernel->DeclareWorkspaceRequirements(
      gsl::make_span(shapes), requirements));
  ASSERT_EQ(requirements.size(), 1U);
  EXPECT_EQ(requirements[0].slot_id, 0);
  EXPECT_EQ(requirements[0].size_bytes, expected->total_workspace_bytes);
  EXPECT_EQ(requirements[0].alignment_bytes, 256U);

  shapes[0] = WorkspaceInputShape::PresentWithoutShape();
  ASSERT_STATUS_OK(kernel->DeclareWorkspaceRequirements(
      gsl::make_span(shapes), requirements));
  EXPECT_TRUE(requirements.empty());
}

TEST(GroupQueryAttentionWorkspaceEstimateTest,
     KernelCommitsMemoryPolicyAndFailsClosedWithoutShapes) {
  if (!HasCudaDevice()) {
    GTEST_SKIP() << "A CUDA device is required to construct the CUDA kernel.";
  }

  ScopedEnvironmentVariables scoped_env_vars{{{"ORT_ENABLE_XQA", "1"}}};
  SessionOptions session_options;
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
      kOrtSessionOptionsKernelDispatchPolicy, "memory"));
  InferenceSessionWrapper session(session_options, GetEnvironment());

  CUDAExecutionProviderInfo provider_info;
  provider_info.sdpa_kernel = kMath;
  auto cuda_ep = std::make_shared<CUDAExecutionProvider>(provider_info);
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(cuda_ep));
  const std::string model_bytes =
      BuildGroupQueryAttentionKernelModel(/*sliding_window_cache=*/true, /*include_head_sink=*/false);
  ASSERT_STATUS_OK(session.Load(model_bytes.data(), static_cast<int>(model_bytes.size())));
  ASSERT_STATUS_OK(session.Initialize());

  const Graph& graph = session.GetGraph();
  const Node* node = FindNodeByOpType(graph, "GroupQueryAttention");
  ASSERT_NE(node, nullptr);
  ASSERT_EQ(node->GetExecutionProviderType(), kCudaExecutionProvider);
  const OpKernel* kernel = session.GetSessionState().GetKernel(node->Index());
  ASSERT_NE(kernel, nullptr);

  auto shapes = SeparateShapes();
  auto memory_config = Config();
  memory_config.dispatch_policy = KernelDispatchPolicy::Memory;
  const auto expected = EstimateGroupQueryAttentionWorkspace(
      memory_config, gsl::make_span(shapes), cuda_ep->GetDeviceProp(),
      *cuda_ep->GetAttentionKernelOptions());
  ASSERT_TRUE(expected.has_value());
  const auto auto_estimate = EstimateGroupQueryAttentionWorkspace(
      Config(), gsl::make_span(shapes), cuda_ep->GetDeviceProp(),
      *cuda_ep->GetAttentionKernelOptions());
  ASSERT_TRUE(auto_estimate.has_value());
  EXPECT_LE(expected->total_workspace_bytes,
            auto_estimate->total_workspace_bytes);

  InlinedVector<WorkspaceRequirement> requirements;
  ASSERT_STATUS_OK(kernel->DeclareWorkspaceRequirements(
      gsl::make_span(shapes), requirements));
  ASSERT_EQ(requirements.size(), 1U);
  EXPECT_EQ(requirements[0].size_bytes, expected->total_workspace_bytes);

  shapes[0] = WorkspaceInputShape::PresentWithoutShape();
  const auto status = kernel->DeclareWorkspaceRequirements(
      gsl::make_span(shapes), requirements);
  EXPECT_FALSE(status.IsOK());
  EXPECT_TRUE(requirements.empty());
}

#if USE_FLASH_ATTENTION && USE_MEMORY_EFFICIENT_ATTENTION
TEST(GroupQueryAttentionWorkspaceEstimateTest, MemoryHead512Envelope64RetainsCoveringUnfusedRoute) {
  AttentionKernelOptions options;
  options.InitializeOnce(kFlash | kEfficient | kMath, true);
  auto config = Config();
  config.sliding_window_cache = false;
  config.local_window_size = -1;
  config.max_total_sequence_length = 64;
  config.enable_xqa = false;
  const auto shapes = SeparateShapes(4, 512, 64, 1);
  const auto baseline = EstimateGroupQueryAttentionWorkspace(config, shapes, Device(), options);
  ASSERT_TRUE(baseline.has_value());
  config.dispatch_policy = KernelDispatchPolicy::Memory;
  const auto memory = EstimateGroupQueryAttentionWorkspace(config, shapes, Device(), options);
  ASSERT_TRUE(memory.has_value());
  const auto covering_set = GQAReachableBackend::Flash | GQAReachableBackend::Unfused;
  EXPECT_EQ(memory->committed_backends.prefill, covering_set);
  EXPECT_EQ(memory->committed_backends.chunked_prefill, covering_set);
  EXPECT_TRUE(HasGQAReachableBackend(memory->committed_backends.decode, GQAReachableBackend::Unfused));
  EXPECT_FALSE(HasGQAReachableBackend(memory->sized_backends, GQAReachableBackend::MemoryEfficient));
  EXPECT_EQ(memory->total_workspace_bytes, 213248u);
  EXPECT_EQ(baseline->total_workspace_bytes, 1245440u);
}
#endif

#if USE_FLASH_ATTENTION
TEST(GroupQueryAttentionWorkspaceEstimateTest, MemoryCommitsStaticFlashGeometryForBothTypesAndCacheKinds) {
  AttentionKernelOptions options;
  options.InitializeOnce(kFlash, true);
  for (bool is_bf16 : {false, true}) {
    for (bool quantized : {false, true}) {
      SCOPED_TRACE(testing::Message() << "bf16=" << is_bf16 << " quantized=" << quantized);
      auto config = Config();
      config.static_head_size = 128;
      config.is_bf16 = is_bf16;
      config.enable_xqa = false;
      config.dispatch_policy = KernelDispatchPolicy::Memory;
      auto shapes = SeparateShapes(4, 128);
      if (quantized) {
        config.cache_element_size = 1;
        config.kv_cache_bit_width = 8;
        config.k_quantization = GQAKvQuantizationType::PerTensor;
        config.v_quantization = GQAKvQuantizationType::PerTensor;
        shapes[12] = Known({1});
        shapes[13] = Known({1});
      } else {
        shapes[11] = Known({8});
      }
      const auto estimate = EstimateGroupQueryAttentionWorkspace(config, shapes, Device(), options);
      ASSERT_TRUE(estimate.has_value());
      for (auto phase : {GQAExecutionPhase::Prefill, GQAExecutionPhase::Decode,
                         GQAExecutionPhase::ChunkedPrefill}) {
        EXPECT_EQ(estimate->committed_backends.Get(phase), GQAReachableBackend::Flash);
      }

      config.static_head_size = 0;
      const auto without_static_head = EstimateGroupQueryAttentionWorkspace(config, shapes, Device(), options);
#ifdef ORT_QUICK_BUILD
      EXPECT_FALSE(without_static_head.has_value());
#else
      EXPECT_TRUE(without_static_head.has_value());
#endif
      config.static_head_size = 256;
      EXPECT_FALSE(EstimateGroupQueryAttentionWorkspace(config, shapes, Device(), options).has_value());
    }
  }
}

TEST(GroupQueryAttentionWorkspaceEstimateTest, KernelCommitsStaticFlashGeometry) {
  if (!HasCudaDevice()) {
    GTEST_SKIP() << "A CUDA device is required to construct the CUDA kernel.";
  }
  ScopedEnvironmentVariables scoped_env_vars{{{"ORT_ENABLE_XQA", "0"}}};
  SessionOptions session_options;
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
      kOrtSessionOptionsKernelDispatchPolicy, "memory"));
  InferenceSessionWrapper session(session_options, GetEnvironment());
  CUDAExecutionProviderInfo provider_info;
  provider_info.sdpa_kernel = kFlash;
  auto cuda_ep = std::make_shared<CUDAExecutionProvider>(provider_info);
  if (cuda_ep->GetDeviceProp().major < 8) {
    GTEST_SKIP() << "Flash requires compute capability 8.0 or newer.";
  }
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(cuda_ep));
  const std::string model = BuildGroupQueryAttentionKernelModel(true, true, 4, 128);
  ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
  ASSERT_STATUS_OK(session.Initialize());
  const Node* node = FindNodeByOpType(session.GetGraph(), "GroupQueryAttention");
  ASSERT_NE(node, nullptr);
  ASSERT_EQ(node->GetExecutionProviderType(), kCudaExecutionProvider);
  const OpKernel* kernel = session.GetSessionState().GetKernel(node->Index());
  ASSERT_NE(kernel, nullptr);
  auto shapes = SeparateShapes(4, 128);
  shapes[11] = Known({8});
  InlinedVector<WorkspaceRequirement> requirements;
  ASSERT_STATUS_OK(kernel->DeclareWorkspaceRequirements(gsl::make_span(shapes), requirements));
  ASSERT_EQ(requirements.size(), 1U);
  EXPECT_GT(requirements[0].size_bytes, 0U);
}

TEST(GroupQueryAttentionWorkspaceEstimateTest, MemoryDeclinesPartialGeometryWithoutCoveringFallback) {
  AttentionKernelOptions options;
  options.InitializeOnce(kFlash, true);
  auto config = Config();
  config.sliding_window_cache = false;
  config.local_window_size = -1;
  config.max_total_sequence_length = 64;
  config.enable_xqa = false;
  auto shapes = SeparateShapes(4, 512, 64, 1);
  shapes[11] = Known({8});
  ASSERT_TRUE(EstimateGroupQueryAttentionWorkspace(config, shapes, Device(), options).has_value());
  config.dispatch_policy = KernelDispatchPolicy::Memory;
  EXPECT_FALSE(EstimateGroupQueryAttentionWorkspace(config, shapes, Device(), options).has_value());
}
#endif

TEST_F(GroupQueryAttentionDispatchPolicyTest, MemoryEnforcesPrefillDecodeAndChunkedPrefill) {
  for (auto phase : {GQAExecutionPhase::Prefill, GQAExecutionPhase::Decode, GQAExecutionPhase::ChunkedPrefill}) {
    GQAPolicyRunConfig config;
    config.phase = phase;
    config.required_permitted = GQAReachableBackend::Unfused;
    ASSERT_NO_FATAL_FAILURE(RunCase(config));
  }
}

TEST_F(GroupQueryAttentionDispatchPolicyTest, OtherPoliciesPreserveEligibleRoutesWithoutAccountant) {
  for (const char* policy : {"auto", "latency", "safe"}) {
    for (auto phase : {GQAExecutionPhase::Prefill, GQAExecutionPhase::Decode, GQAExecutionPhase::ChunkedPrefill}) {
      GQAPolicyRunConfig config;
      config.policy = policy;
      config.phase = phase;
#if USE_MEMORY_EFFICIENT_ATTENTION
      if (phase != GQAExecutionPhase::Decode) {
        config.expected_backend = "memory_efficient";
      }
#endif
      ASSERT_NO_FATAL_FAILURE(RunCase(config));
    }
  }
}

#if USE_FLASH_ATTENTION
TEST_F(GroupQueryAttentionDispatchPolicyTest, SafeExcludesCudnnWhileAutoUsesIt) {
  AttentionKernelOptions options;
  options.InitializeOnce(kCudnn, true, /*check_cudnn_version=*/true);
  if (!options.UseCudnnFlashAttention()) {
    GTEST_SKIP() << "The cuDNN SDPA control requires a supported device and cuDNN version.";
  }
  for (const auto policy : {KernelDispatchPolicy::Auto, KernelDispatchPolicy::Safe}) {
    for (int bound_case : {0, 1, 2}) {
      SCOPED_TRACE(bound_case);
      GQAPolicyRunConfig config;
      config.policy = KernelDispatchPolicyName(policy);
      config.phase = GQAExecutionPhase::Decode;
      config.head = 128;
      config.kernel_selection = kCudnn | kFlash | kMath;
      config.declare_envelope = bound_case != 1;
      config.dynamic_sequence = bound_case == 2;
      config.forbidden_permitted = policy == KernelDispatchPolicy::Safe
                                       ? GQAReachableBackend::Cudnn
                                       : GQAReachableBackend::None;
      config.expected_backend = policy == KernelDispatchPolicy::Auto ? "cudnn" : "flash";
      ASSERT_NO_FATAL_FAILURE(RunCase(config));
    }
  }
}
#endif

TEST_F(GroupQueryAttentionDispatchPolicyTest, SafeRunsWithoutWorkspaceBound) {
  for (bool dynamic_sequence : {false, true}) {
    for (auto phase : {GQAExecutionPhase::Prefill, GQAExecutionPhase::Decode, GQAExecutionPhase::ChunkedPrefill}) {
      GQAPolicyRunConfig config;
      config.policy = "safe";
      config.phase = phase;
      config.head = 128;
      config.declare_envelope = dynamic_sequence;
      config.dynamic_sequence = dynamic_sequence;
      config.forbidden_permitted = GQAReachableBackend::Cudnn;
#if USE_FLASH_ATTENTION
      config.expected_backend = "flash";
#elif USE_MEMORY_EFFICIENT_ATTENTION
      config.expected_backend = "memory_efficient";
#endif
      ASSERT_NO_FATAL_FAILURE(RunCase(config));
    }
  }
}

#if USE_MEMORY_EFFICIENT_ATTENTION
TEST_F(GroupQueryAttentionDispatchPolicyTest, MemoryDecodeDoesNotUsePrefillFallback) {
  GQAPolicyRunConfig config;
  config.phase = GQAExecutionPhase::Decode;
  config.head = 256;
  config.capacity = 3072;
  config.redeclared_sequence_bound = 128;
  config.required_permitted = GQAReachableBackend::Unfused;
  config.forbidden_permitted = GQAReachableBackend::MemoryEfficient;
  ASSERT_NO_FATAL_FAILURE(RunCase(config));
}
#endif

TEST_F(GroupQueryAttentionDispatchPolicyTest, DecodeFallsBackWhenCacheBuffersDoNotAlias) {
  GQAPolicyRunConfig config;
  config.phase = GQAExecutionPhase::Decode;
  config.head = 64;
  config.capacity = 16384;
  config.kernel_selection = kMath;
  config.required_permitted = GQAReachableBackend::Xqa | GQAReachableBackend::Unfused;
  ASSERT_NO_FATAL_FAILURE(RunCase(config));
}

TEST_F(GroupQueryAttentionDispatchPolicyTest, MissingPhaseCommitmentFailsWhileAutoRuns) {
  for (const auto policy : {KernelDispatchPolicy::Memory, KernelDispatchPolicy::Auto}) {
    GQAPolicyRunConfig config;
    config.policy = KernelDispatchPolicyName(policy);
    config.phase = GQAExecutionPhase::ChunkedPrefill;
    config.redeclared_sequence_bound = 1;
    config.expect_failure = policy == KernelDispatchPolicy::Memory;
#if USE_MEMORY_EFFICIENT_ATTENTION
    config.expected_backend = "memory_efficient";
#endif
    ASSERT_NO_FATAL_FAILURE(RunCase(config));
  }
}

TEST_F(GroupQueryAttentionDispatchPolicyTest, AutoRunsWithoutAccountantOrEnvelope) {
  for (auto phase : {GQAExecutionPhase::Prefill, GQAExecutionPhase::Decode, GQAExecutionPhase::ChunkedPrefill}) {
    GQAPolicyRunConfig config;
    config.policy = "auto";
    config.phase = phase;
    config.declare_envelope = false;
#if USE_MEMORY_EFFICIENT_ATTENTION
    if (phase != GQAExecutionPhase::Decode) {
      config.expected_backend = "memory_efficient";
    }
#endif
    ASSERT_NO_FATAL_FAILURE(RunCase(config));
  }
}

TEST(GroupQueryAttentionWorkspaceBoundsTest, CheckedOverflowIsUnavailable) {
  auto bounds = Bounds();
  bounds.batch_size_bound = std::numeric_limits<int32_t>::max();
  bounds.sequence_length_bound = std::numeric_limits<int32_t>::max();
  bounds.num_heads = std::numeric_limits<int32_t>::max();
  bounds.kv_num_heads = 1;
  bounds.head_size_bound = std::numeric_limits<int32_t>::max();
  bounds.present_kv_cache_capacity_bound = std::numeric_limits<int32_t>::max();
  bounds.reachable_backends = GQAReachableBackend::Unfused;
  EXPECT_FALSE(GetGQAWorkspaceAggregateForBounds(bounds).status.IsOK());

  bounds = Bounds();
  bounds.is_windowed_kv_cache = true;
  bounds.present_kv_cache_capacity_bound = std::numeric_limits<int32_t>::max();
  bounds.sequence_length_bound = 2;
  bounds.reachable_backends = GQAReachableBackend::Unfused;
  EXPECT_FALSE(GetGQAWorkspaceAggregateForBounds(bounds).status.IsOK());
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime

#endif
