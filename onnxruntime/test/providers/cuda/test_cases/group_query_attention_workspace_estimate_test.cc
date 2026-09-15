// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#if !defined(USE_CUDA_MINIMAL) && !defined(DISABLE_CONTRIB_OPS) && !defined(BUILD_CUDA_EP_AS_PLUGIN)

#include <array>
#include <limits>
#include <optional>
#include <vector>

#include "core/graph/graph.h"
#include "contrib_ops/cpu/bert/attention_common.h"
#include "contrib_ops/cuda/bert/group_query_attention_workspace_estimate.h"

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
  config.local_window_size = 128;
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
  add_int_attribute("local_window_size", 128);
  add_int_attribute("sliding_window_cache", 1);

  const std::vector<NodeArg*> inputs{&query, &key, &value, &past_key};
  const std::vector<NodeArg*> outputs;
  Node node{"gqa", "GroupQueryAttention", "", inputs, outputs, &attributes, kMSDomain};
  return EstimateGroupQueryAttentionWorkspace(
      node, input_shapes, Device(), options);
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

TEST(GroupQueryAttentionWorkspaceEstimateTest, NonWindowedTotalKvAndAliasingAreUnavailable) {
  AttentionKernelOptions options;
  options.InitializeOnce(kMath, true);
  auto config = Config();
  config.sliding_window_cache = false;
  config.local_window_size = -1;
  EXPECT_FALSE(EstimateGroupQueryAttentionWorkspace(
                   config, SeparateShapes(/*sequence=*/1, /*head=*/64, /*capacity=*/128),
                   Device(), options)
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

TEST(GroupQueryAttentionWorkspaceEstimateTest, RejectsNoncausalLocalWindow) {
  AttentionKernelOptions options;
  options.InitializeOnce(kMath, true);
  auto config = Config();
  config.causal = 0;
  EXPECT_FALSE(EstimateGroupQueryAttentionWorkspace(
                   config, SeparateShapes(), Device(), options)
                   .has_value());

  config = Config();
  EXPECT_FALSE(EstimateGroupQueryAttentionWorkspace(
                   config, SeparateShapes(/*sequence=*/4, /*head=*/64, /*capacity=*/64),
                   Device(), options)
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
    auto shapes = SeparateShapes();
    const int64_t stored_head = bit_width == 4 ? 32 : 64;
    shapes[3] = Known({2, 2, 256, stored_head});
    shapes[4] = Known({2, 2, 256, stored_head});
    shapes[12] = Known({1});
    shapes[13] = Known({1, 2, 1, 64});
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

TEST(GroupQueryAttentionWorkspaceBoundsTest, AggregatesExclusiveRoutesWithMax) {
  auto xqa = Bounds();
  xqa.sequence_length_bound = 1;
  xqa.prompt_reachable = false;
  xqa.reachable_backends = GQAReachableBackend::Xqa;
  const auto xqa_only = GetGQAWorkspaceAggregateForBounds(xqa);
  ASSERT_TRUE(xqa_only.status.IsOK());
  auto unfused = xqa;
  unfused.reachable_backends = GQAReachableBackend::Unfused;
  const auto unfused_only = GetGQAWorkspaceAggregateForBounds(unfused);
  ASSERT_TRUE(unfused_only.status.IsOK());
  auto both = xqa;
  both.reachable_backends =
      GQAReachableBackend::Xqa | GQAReachableBackend::Unfused;
  const auto aggregate = GetGQAWorkspaceAggregateForBounds(both);
  ASSERT_TRUE(aggregate.status.IsOK());
  EXPECT_EQ(aggregate.total_workspace_bytes,
            std::max(xqa_only.total_workspace_bytes,
                     unfused_only.total_workspace_bytes));
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
  EXPECT_EQ(level1.persistent_prepack_bytes, 19u);
  EXPECT_EQ(level1.initialization_scratch_bytes, 23u);

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
