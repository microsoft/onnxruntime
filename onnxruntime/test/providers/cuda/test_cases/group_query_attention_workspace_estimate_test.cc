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
#include "core/framework/session_state.h"
#include "core/graph/graph.h"
#include "core/providers/cuda/cuda_execution_provider.h"
#include "core/providers/cuda/cuda_execution_provider_info.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "contrib_ops/cpu/bert/attention_common.h"
#include "contrib_ops/cuda/bert/group_query_attention_workspace_estimate.h"
#include "test/providers/cuda/internal_testing/cuda_internal_test_helpers.h"
#include "test/test_environment.h"
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

WorkspaceInputShape Known(std::initializer_list<int64_t> dims) {
  return WorkspaceInputShape::PresentWithShape(TensorShape{TensorShapeVector{dims}});
}

std::array<WorkspaceInputShape, 16> SeparateShapes(int64_t sequence = 4,
                                                   int64_t head = 64,
                                                   int64_t capacity = 256) {
  std::array<WorkspaceInputShape, 16> shapes;
  shapes[0] = Known({2, sequence, 8 * head});
  shapes[1] = Known({2, sequence, 2 * head});
  shapes[2] = Known({2, sequence, 2 * head});
  shapes[3] = Known({2, 2, capacity, head});
  shapes[4] = Known({2, 2, capacity, head});
  shapes[5] = Known({2});
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
  bounds.prompt_reachable = true;
  bounds.decode_reachable = true;
  bounds.multi_processor_count = 108;
  return bounds;
}

std::optional<contrib::cuda::GQAWorkspaceAggregate> EstimateFromNode(
    gsl::span<const WorkspaceInputShape> input_shapes,
    const AttentionKernelOptions& options) {
  ONNX_NAMESPACE::TypeProto query_type;
  query_type.mutable_tensor_type()->set_elem_type(
      ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  ONNX_NAMESPACE::TypeProto cache_type;
  cache_type.mutable_tensor_type()->set_elem_type(
      ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  NodeArg query{"query", &query_type};
  NodeArg key{"key", &query_type};
  NodeArg value{"value", &query_type};
  NodeArg past_key{"past_key", &cache_type};

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

  const std::vector<NodeArg*> inputs{&query, &key, &value, &past_key};
  const std::vector<NodeArg*> outputs;
  Node node{"gqa", "GroupQueryAttention", "", inputs, outputs, &attributes, kMSDomain};
  return EstimateGroupQueryAttentionWorkspace(
      node, input_shapes, Device(), options);
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

std::string BuildGroupQueryAttentionKernelModel(bool sliding_window_cache = true) {
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
        "total_sequence_length", "", "", "", "", "head_sink"}) {
    node->add_input(input_name);
  }
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
    add_int_attribute("local_window_size", 256);
    add_int_attribute("sliding_window_cache", 1);
  }

  constexpr int32_t kFloat16 = ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
  constexpr int32_t kInt32 = ONNX_NAMESPACE::TensorProto_DataType_INT32;
  SetValueInfo(*graph->add_input(), "query", kFloat16, {2, 4, 512});
  SetValueInfo(*graph->add_input(), "key", kFloat16, {2, 4, 128});
  SetValueInfo(*graph->add_input(), "value", kFloat16, {2, 4, 128});
  SetValueInfo(*graph->add_input(), "past_key", kFloat16, {2, 2, 256, 64});
  SetValueInfo(*graph->add_input(), "past_value", kFloat16, {2, 2, 256, 64});
  SetValueInfo(*graph->add_input(), "seqlens_k", kInt32, {2});
  SetValueInfo(*graph->add_input(), "total_sequence_length", kInt32, {});
  SetValueInfo(*graph->add_output(), "output", kFloat16, {2, 4, 512});
  SetValueInfo(*graph->add_output(), "present_key", kFloat16, {2, 2, 256, 64});
  SetValueInfo(*graph->add_output(), "present_value", kFloat16, {2, 2, 256, 64});

  auto* head_sink = graph->add_initializer();
  head_sink->set_name("head_sink");
  head_sink->set_data_type(kFloat16);
  head_sink->add_dims(8);
  head_sink->mutable_raw_data()->assign(8 * sizeof(uint16_t), '\0');

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

TEST(GroupQueryAttentionWorkspaceEstimateTest, NodeAdapterParsesAttributesAndTypes) {
  AttentionKernelOptions options;
  options.InitializeOnce(kMath, true);
  const auto estimate = EstimateFromNode(SeparateShapes(), options);
  ASSERT_TRUE(estimate.has_value());
  EXPECT_GT(estimate->total_workspace_bytes, 0u);
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
    auto cuda_ep = CreateCudaInternalTestExecutionProvider(provider_info);
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
    estimate = EstimateGroupQueryAttentionWorkspace(
        *node, gsl::make_span(shapes), cuda_ep->GetDeviceProp(),
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
        CreateCudaInternalTestExecutionProvider(provider_info)));
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
        CreateCudaInternalTestExecutionProvider(provider_info)));
    ASSERT_STATUS_OK(session.Load(model_bytes.data(), static_cast<int>(model_bytes.size())));
    ASSERT_STATUS_OK(session.Initialize());

    const Node* node = FindNodeByOpType(session.GetGraph(), "GroupQueryAttention");
    ASSERT_NE(node, nullptr);
    EXPECT_NE(node->GetExecutionProviderType(), kCudaExecutionProvider);
  }
}

TEST(GroupQueryAttentionWorkspaceEstimateTest, NonWindowedRequiresExplicitTotalKvBound) {
  AttentionKernelOptions options;
  options.InitializeOnce(kMath, true);
  auto config = Config();
  config.sliding_window_cache = false;
  config.local_window_size = -1;
  config.enable_xqa = false;
  const auto shapes = SeparateShapes(/*sequence=*/4, /*head=*/64, /*capacity=*/128);
  EXPECT_FALSE(EstimateGroupQueryAttentionWorkspace(config, shapes, Device(), options).has_value());

  config.max_total_sequence_length = 64;
  const auto past_capacity_bound =
      EstimateGroupQueryAttentionWorkspace(config, shapes, Device(), options);
  ASSERT_TRUE(past_capacity_bound.has_value());

  config.max_total_sequence_length = 128;
  const auto matching_bound =
      EstimateGroupQueryAttentionWorkspace(config, shapes, Device(), options);
  ASSERT_TRUE(matching_bound.has_value());
  EXPECT_EQ(matching_bound->total_workspace_bytes,
            past_capacity_bound->total_workspace_bytes);

  config.max_total_sequence_length = 256;
  const auto larger_bound =
      EstimateGroupQueryAttentionWorkspace(config, shapes, Device(), options);
  ASSERT_TRUE(larger_bound.has_value());
  EXPECT_GT(larger_bound->total_workspace_bytes,
            matching_bound->total_workspace_bytes);
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

  bounds.prompt_reachable = false;
  bounds.sequence_length_bound = 1;
  const auto decode = GetGQAWorkspaceAggregateForBounds(bounds);
  ASSERT_TRUE(decode.status.IsOK());
  EXPECT_GT(mixed.total_workspace_bytes, decode.total_workspace_bytes);

  bounds.decode_reachable = false;
  bounds.prompt_reachable = true;
  bounds.sequence_length_bound = 4;
  const auto prompt = GetGQAWorkspaceAggregateForBounds(bounds);
  EXPECT_TRUE(prompt.status.IsOK());
}

TEST(GroupQueryAttentionWorkspaceBoundsTest, PartialAliasAddsPastPreservationBuffer) {
  auto bounds = Bounds();
  bounds.reachable_backends = GQAReachableBackend::Unfused;
  bounds.present_kv_cache_capacity_bound = 256;
  bounds.past_kv_cache_capacity_bound = 128;
  const auto shared_or_separate = GetGQAWorkspaceAggregateForBounds(bounds);
  ASSERT_TRUE(shared_or_separate.status.IsOK()) << shared_or_separate.status.message;

  bounds.partial_alias_reachable = true;
  const auto partial_alias = GetGQAWorkspaceAggregateForBounds(bounds);
  ASSERT_TRUE(partial_alias.status.IsOK()) << partial_alias.status.message;
  ASSERT_GE(partial_alias.total_workspace_bytes,
            shared_or_separate.total_workspace_bytes);
  const size_t past_tensor_bytes =
      static_cast<size_t>(bounds.batch_size_bound * bounds.kv_num_heads *
                          bounds.past_kv_cache_capacity_bound * bounds.head_size_bound) *
      bounds.cache_element_size;
  EXPECT_GE(partial_alias.total_workspace_bytes -
                shared_or_separate.total_workspace_bytes,
            past_tensor_bytes);

  bounds.past_kv_cache_capacity_bound = 0;
  EXPECT_FALSE(GetGQAWorkspaceAggregateForBounds(bounds).status.IsOK());
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
  bounds.prompt_reachable = false;
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
  bounds.prompt_reachable = false;
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

TEST(GroupQueryAttentionWorkspaceBoundsTest, FlashFastDecodeEnvelopeCoversEverySequenceLength) {
  auto bounds = Bounds();
  bounds.batch_size_bound = 1;
  bounds.sequence_length_bound = 130;
  bounds.num_heads = 8;
  bounds.kv_num_heads = 2;
  bounds.head_size_bound = 128;
  bounds.present_kv_cache_capacity_bound = 1024;
  bounds.is_windowed_kv_cache = false;
  bounds.local_window_size = -1;
  bounds.reachable_backends = GQAReachableBackend::FlashFastDecode;
  const auto envelope = GetGQAWorkspaceAggregateForBounds(bounds);
  ASSERT_TRUE(envelope.status.IsOK()) << envelope.status.message;

  for (int64_t sequence_length = 1;
       sequence_length <= bounds.sequence_length_bound; ++sequence_length) {
    GQAWorkspaceProblem problem;
    problem.qkv_element_size = bounds.qkv_element_size;
    problem.cache_element_size = bounds.cache_element_size;
    problem.batch_size = bounds.batch_size_bound;
    problem.sequence_length = sequence_length;
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
    ASSERT_TRUE(concrete.status.IsOK())
        << sequence_length << ": " << concrete.status.message;
    EXPECT_GE(envelope.total_workspace_bytes,
              concrete.recipe.total_workspace_bytes)
        << sequence_length << " selected splits "
        << concrete.recipe.flash.selected_split_count;
  }
}

TEST(GroupQueryAttentionWorkspaceBoundsTest, AggregatesExclusiveRoutesWithMax) {
  auto xqa = Bounds();
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
  bounds.reachable_backends = GQAReachableBackend::Unfused;
  EXPECT_TRUE(GetGQAWorkspaceAggregateForBounds(bounds).status.IsOK());
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
  auto cuda_ep = CreateCudaInternalTestExecutionProvider(provider_info);
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

TEST(GroupQueryAttentionWorkspaceEstimateTest, KernelDeclaresBoundedNonWindowedRoot) {
  if (!HasCudaDevice()) {
    GTEST_SKIP() << "A CUDA device is required to construct the CUDA kernel.";
  }

  ScopedEnvironmentVariables scoped_env_vars{{{"ORT_ENABLE_XQA", "0"}}};
  SessionOptions session_options;
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
      kOrtSessionOptionsCudaGqaWorkspaceMaxTotalSequenceLength, "512"));
  InferenceSessionWrapper session(session_options, GetEnvironment());

  CUDAExecutionProviderInfo provider_info;
  provider_info.sdpa_kernel = kMath;
  auto cuda_ep = CreateCudaInternalTestExecutionProvider(provider_info);
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(cuda_ep));
  const std::string model_bytes =
      BuildGroupQueryAttentionKernelModel(/*sliding_window_cache=*/false);
  ASSERT_STATUS_OK(session.Load(model_bytes.data(), static_cast<int>(model_bytes.size())));
  ASSERT_STATUS_OK(session.Initialize());

  const Node* node = FindNodeByOpType(session.GetGraph(), "GroupQueryAttention");
  ASSERT_NE(node, nullptr);
  ASSERT_EQ(node->GetExecutionProviderType(), kCudaExecutionProvider);
  const OpKernel* kernel = session.GetSessionState().GetKernel(node->Index());
  ASSERT_NE(kernel, nullptr);

  auto shapes = SeparateShapes();
  shapes[11] = Known({8});
  auto expected_config = Config();
  expected_config.sliding_window_cache = false;
  expected_config.local_window_size = -1;
  expected_config.max_total_sequence_length = 512;
  expected_config.enable_xqa = false;
  expected_config.head_sink_is_prepacked = true;
  const auto expected = EstimateGroupQueryAttentionWorkspace(
      expected_config, gsl::make_span(shapes), cuda_ep->GetDeviceProp(),
      *cuda_ep->GetAttentionKernelOptions());
  ASSERT_TRUE(expected.has_value());

  InlinedVector<WorkspaceRequirement> requirements;
  ASSERT_STATUS_OK(kernel->DeclareWorkspaceRequirements(
      gsl::make_span(shapes), requirements));
  ASSERT_EQ(requirements.size(), 1U);
  EXPECT_EQ(requirements[0].slot_id, 0);
  EXPECT_EQ(requirements[0].size_bytes, expected->total_workspace_bytes);
  EXPECT_EQ(requirements[0].alignment_bytes, 256U);
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
