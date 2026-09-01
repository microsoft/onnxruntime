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
#include "contrib_ops/cpu/bert/attention_common.h"
#include "contrib_ops/cuda/bert/packed_attention_workspace_estimate.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"
#include "test/util/include/inference_session_wrapper.h"

namespace onnxruntime {
namespace test {

using contrib::attention::AttentionBackend;
using contrib::cuda::BuildPackedAttentionProblem;
using contrib::cuda::BuildPackedMultiHeadAttentionProblem;
using contrib::cuda::EstimatePackedAttentionWorkspace;
using contrib::cuda::GetPackedAttentionFeasibleBackends;
using contrib::cuda::GetPackedAttentionWorkspaceAggregate;
using contrib::cuda::GetPackedAttentionWorkspaceRecipe;
using contrib::cuda::GetPackedMultiHeadAttentionFeasibleBackends;
using contrib::cuda::GetPackedMultiHeadAttentionWorkspaceAggregate;
using contrib::cuda::GetPackedMultiHeadAttentionWorkspaceRecipe;
using contrib::cuda::kPackedAttentionWorkspaceAlignment;
using contrib::cuda::PackedAttentionBackend;
using contrib::cuda::PackedAttentionBackendMask;
using contrib::cuda::PackedAttentionInputShapes;
using contrib::cuda::PackedAttentionProblem;
using contrib::cuda::PackedAttentionQkvMaterializationIndexWidth;
using contrib::cuda::PackedAttentionShape;
using contrib::cuda::PackedAttentionWorkspaceAggregate;
using contrib::cuda::PackedAttentionWorkspaceError;
using contrib::cuda::PackedAttentionWorkspaceEstimateConfig;
using contrib::cuda::PackedAttentionWorkspaceOperator;
using contrib::cuda::PackedMultiHeadAttentionInputShapes;
using contrib::cuda::PackedMultiHeadAttentionProblem;
using contrib::cuda::PackedMultiHeadAttentionQkvFormat;
using contrib::cuda::SetPackedAttentionWorkspaceRequirements;

namespace {

constexpr int kMath = static_cast<int>(AttentionBackend::MATH);
constexpr int kFlash = static_cast<int>(AttentionBackend::FLASH_ATTENTION);
constexpr int kMea = static_cast<int>(AttentionBackend::EFFICIENT_ATTENTION);
constexpr int kTrt = static_cast<int>(AttentionBackend::TRT_FUSED_ATTENTION);
constexpr int kTrtFlash = static_cast<int>(AttentionBackend::TRT_FLASH_ATTENTION);

PackedAttentionShape Shape(std::initializer_list<int64_t> dimensions) {
  PackedAttentionShape shape;
  shape.rank = dimensions.size();
  size_t index = 0;
  for (int64_t dimension : dimensions) {
    if (index < shape.dimensions.size()) {
      shape.dimensions[index] = dimension;
    }
    ++index;
  }
  return shape;
}

WorkspaceInputShape KnownShape(std::initializer_list<int64_t> dimensions) {
  return WorkspaceInputShape::PresentWithShape(
      TensorShape{TensorShapeVector{dimensions}});
}

std::array<WorkspaceInputShape, 6> PaShapes() {
  std::array<WorkspaceInputShape, 6> shapes;
  shapes[0] = KnownShape({6, 6});
  shapes[1] = KnownShape({6, 24});
  shapes[2] = KnownShape({24});
  shapes[3] = KnownShape({2, 4});
  shapes[4] = KnownShape({3});
  return shapes;
}

std::array<WorkspaceInputShape, 7> PackedPmhaShapes() {
  std::array<WorkspaceInputShape, 7> shapes;
  shapes[0] = KnownShape({6, 2, 3, 4});
  shapes[4] = KnownShape({2, 4});
  shapes[5] = KnownShape({3});
  return shapes;
}

std::array<WorkspaceInputShape, 7> SeparatePmhaShapes() {
  std::array<WorkspaceInputShape, 7> shapes;
  shapes[0] = KnownShape({6, 8});
  shapes[1] = KnownShape({6, 8});
  shapes[2] = KnownShape({6, 8});
  shapes[4] = KnownShape({2, 4});
  shapes[5] = KnownShape({3});
  return shapes;
}

std::array<WorkspaceInputShape, 7> PackedPmhaShapes(
    int64_t sequence_length, int64_t head_size = 64) {
  std::array<WorkspaceInputShape, 7> shapes;
  shapes[0] = KnownShape({sequence_length, 1, 3, head_size});
  shapes[4] = KnownShape({1, sequence_length});
  shapes[5] = KnownShape({2});
  return shapes;
}

std::array<WorkspaceInputShape, 7> SeparatePmhaShapes(
    int64_t sequence_length, bool has_attention_bias, int64_t head_size = 64) {
  std::array<WorkspaceInputShape, 7> shapes;
  shapes[0] = KnownShape({sequence_length, head_size});
  shapes[1] = KnownShape({sequence_length, head_size});
  shapes[2] = KnownShape({sequence_length, head_size});
  shapes[4] = KnownShape({1, sequence_length});
  shapes[5] = KnownShape({2});
  if (has_attention_bias) {
    shapes[6] = KnownShape({1, 1, sequence_length, sequence_length});
  }
  return shapes;
}

PackedAttentionWorkspaceEstimateConfig PaConfig() {
  PackedAttentionWorkspaceEstimateConfig config;
  config.op = PackedAttentionWorkspaceOperator::PackedAttention;
  config.element_size = 2;
  config.num_heads = 2;
  config.qkv_hidden_sizes_count = 3;
  config.qkv_hidden_sizes = {8, 8, 8};
  return config;
}

PackedAttentionWorkspaceEstimateConfig PmhaConfig() {
  PackedAttentionWorkspaceEstimateConfig config;
  config.op = PackedAttentionWorkspaceOperator::PackedMultiHeadAttention;
  config.element_size = 2;
  config.num_heads = 2;
  return config;
}

PackedAttentionWorkspaceEstimateConfig PmhaConfig(size_t element_size) {
  auto config = PmhaConfig();
  config.element_size = element_size;
  config.num_heads = 1;
  return config;
}

cudaDeviceProp Sm80Device() {
  cudaDeviceProp device_prop{};
  device_prop.major = 8;
  device_prop.minor = 0;
  return device_prop;
}

PackedAttentionProblem RoutePaProblem(int32_t head_size = 64) {
  PackedAttentionProblem problem;
  problem.element_size = 2;
  problem.token_count = 128;
  problem.batch_size = 1;
  problem.sequence_length = 128;
  problem.num_heads = 1;
  problem.input_hidden_size = head_size;
  problem.hidden_size = head_size;
  problem.v_hidden_size = head_size;
  problem.qk_head_size = head_size;
  problem.v_head_size = head_size;
  problem.qkv_materialization_index_width =
      contrib::cuda::GetPackedAttentionQkvMaterializationIndexWidth(head_size, head_size);
  return problem;
}

PackedMultiHeadAttentionProblem RoutePmhaProblem(int32_t head_size = 64) {
  PackedMultiHeadAttentionProblem problem;
  problem.element_size = 2;
  problem.token_count = 128;
  problem.batch_size = 1;
  problem.sequence_length = 128;
  problem.num_heads = 1;
  problem.hidden_size = head_size;
  problem.v_hidden_size = head_size;
  problem.qk_head_size = head_size;
  problem.v_head_size = head_size;
  problem.qkv_format = PackedMultiHeadAttentionQkvFormat::Separate;
  problem.qkv_materialization_index_width =
      contrib::cuda::GetPackedAttentionQkvMaterializationIndexWidth(head_size, head_size);
  return problem;
}

bool HasRoute(PackedAttentionBackendMask mask, PackedAttentionBackend backend) {
  return contrib::cuda::HasPackedAttentionBackend(mask, backend);
}

PackedMultiHeadAttentionProblem BuildPmhaProblem(
    int32_t sequence_length, size_t element_size, bool packed,
    bool has_attention_bias = false, int32_t head_size = 64) {
  PackedMultiHeadAttentionInputShapes inputs;
  inputs.query = packed
                     ? Shape({sequence_length, 1, 3, head_size})
                     : Shape({sequence_length, head_size});
  inputs.token_offset = Shape({1, sequence_length});
  inputs.cumulative_sequence_length = Shape({2});
  inputs.element_size = element_size;
  inputs.num_heads = 1;
  if (!packed) {
    inputs.has_key = true;
    inputs.has_value = true;
    inputs.key = Shape({sequence_length, head_size});
    inputs.value = Shape({sequence_length, head_size});
  }
  inputs.has_attention_bias = has_attention_bias;
  if (has_attention_bias) {
    inputs.attention_bias = Shape({1, 1, sequence_length, sequence_length});
  }

  const auto result = BuildPackedMultiHeadAttentionProblem(inputs);
  EXPECT_TRUE(result.status.IsOK()) << result.status.message;
  return result.problem;
}

void ExpectPmhaAggregateUsesMaximum(
    const PackedMultiHeadAttentionProblem& problem,
    PackedAttentionBackendMask routes,
    const PackedAttentionWorkspaceAggregate& aggregate) {
  size_t maximum = 0;
  size_t sum = 0;
  size_t nonzero_route_count = 0;
  for (PackedAttentionBackend backend :
       {PackedAttentionBackend::Trt, PackedAttentionBackend::Flash,
        PackedAttentionBackend::MemoryEfficient, PackedAttentionBackend::Unfused}) {
    if (!HasRoute(routes, backend)) {
      continue;
    }

    auto route_problem = problem;
    route_problem.backend = backend;
    route_problem.trt_runner_available = backend == PackedAttentionBackend::Trt;
    const auto route = GetPackedMultiHeadAttentionWorkspaceRecipe(route_problem);
    ASSERT_TRUE(route.status.IsOK()) << route.status.message;
    maximum = std::max(maximum, route.recipe.attention_workspace_bytes);
    sum += route.recipe.attention_workspace_bytes;
    nonzero_route_count += route.recipe.attention_workspace_bytes != 0 ? 1U : 0U;
  }

  EXPECT_EQ(aggregate.projection_bytes, 0U);
  EXPECT_EQ(aggregate.attention_workspace_offset_bytes, 0U);
  EXPECT_EQ(aggregate.attention_workspace_bytes, maximum);
  EXPECT_EQ(aggregate.total_workspace_bytes, maximum);
  if (nonzero_route_count > 1) {
    EXPECT_LT(aggregate.attention_workspace_bytes, sum);
  }
}

void ExpectEmptyAggregate(const PackedAttentionWorkspaceAggregate& aggregate) {
  ASSERT_TRUE(aggregate.status.IsOK()) << aggregate.status.message;
  EXPECT_EQ(aggregate.projection_bytes, 0U);
  EXPECT_EQ(aggregate.attention_workspace_offset_bytes, 0U);
  EXPECT_EQ(aggregate.attention_workspace_bytes, 0U);
  EXPECT_EQ(aggregate.total_workspace_bytes, 0U);
}

std::optional<PackedAttentionWorkspaceAggregate> EstimateFromNode(
    const char* op_type,
    int32_t element_type,
    std::optional<int64_t> num_heads,
    std::optional<std::vector<int64_t>> qkv_hidden_sizes,
    gsl::span<const WorkspaceInputShape> input_shapes,
    const AttentionKernelOptions& options) {
  ONNX_NAMESPACE::TypeProto input_type;
  input_type.mutable_tensor_type()->set_elem_type(element_type);
  NodeArg input{"input", &input_type};

  NodeAttributes attributes;
  if (num_heads.has_value()) {
    ONNX_NAMESPACE::AttributeProto attribute;
    attribute.set_name("num_heads");
    attribute.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
    attribute.set_i(*num_heads);
    attributes.emplace("num_heads", std::move(attribute));
  }
  if (qkv_hidden_sizes.has_value()) {
    ONNX_NAMESPACE::AttributeProto attribute;
    attribute.set_name("qkv_hidden_sizes");
    attribute.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INTS);
    for (int64_t value : *qkv_hidden_sizes) {
      attribute.add_ints(value);
    }
    attributes.emplace("qkv_hidden_sizes", std::move(attribute));
  }

  const std::vector<NodeArg*> inputs{&input};
  const std::vector<NodeArg*> outputs;
  Node node{"packed_attention", op_type, "", inputs, outputs, &attributes, kMSDomain};
  return EstimatePackedAttentionWorkspace(
      node, input_shapes, Sm80Device(), options);
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

std::string BuildPackedAttentionKernelModel(
    PackedAttentionWorkspaceOperator op) {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  auto* onnx_opset = model.add_opset_import();
  onnx_opset->set_domain("");
  onnx_opset->set_version(17);
  auto* ms_opset = model.add_opset_import();
  ms_opset->set_domain(kMSDomain);
  ms_opset->set_version(1);

  auto* graph = model.mutable_graph();
  graph->set_name("packed_attention_workspace_level2");
  auto* node = graph->add_node();
  node->set_domain(kMSDomain);
  node->set_name("packed_attention");

  auto* num_heads = node->add_attribute();
  num_heads->set_name("num_heads");
  num_heads->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  num_heads->set_i(2);

  constexpr int32_t kFloat16 =
      ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
  constexpr int32_t kInt32 =
      ONNX_NAMESPACE::TensorProto_DataType_INT32;
  if (op == PackedAttentionWorkspaceOperator::PackedAttention) {
    node->set_op_type("PackedAttention");
    for (const char* input_name :
         {"input", "weights", "bias", "token_offset",
          "cumulative_sequence_length"}) {
      node->add_input(input_name);
    }
    node->add_output("output");

    auto* qkv_hidden_sizes = node->add_attribute();
    qkv_hidden_sizes->set_name("qkv_hidden_sizes");
    qkv_hidden_sizes->set_type(
        ONNX_NAMESPACE::AttributeProto_AttributeType_INTS);
    qkv_hidden_sizes->add_ints(8);
    qkv_hidden_sizes->add_ints(8);
    qkv_hidden_sizes->add_ints(8);

    SetValueInfo(*graph->add_input(), "input", kFloat16, {6, 6});
    SetValueInfo(*graph->add_input(), "weights", kFloat16, {6, 24});
    SetValueInfo(*graph->add_input(), "bias", kFloat16, {24});
    SetValueInfo(*graph->add_input(), "token_offset", kInt32, {2, 4});
    SetValueInfo(*graph->add_input(), "cumulative_sequence_length",
                 kInt32, {3});
    SetValueInfo(*graph->add_output(), "output", kFloat16, {6, 8});
  } else {
    node->set_op_type("PackedMultiHeadAttention");
    node->add_input("query");
    node->add_input("");
    node->add_input("");
    node->add_input("");
    node->add_input("token_offset");
    node->add_input("cumulative_sequence_length");
    node->add_output("output");

    SetValueInfo(*graph->add_input(), "query", kFloat16, {6, 2, 3, 4});
    SetValueInfo(*graph->add_input(), "token_offset", kInt32, {2, 4});
    SetValueInfo(*graph->add_input(), "cumulative_sequence_length",
                 kInt32, {3});
    SetValueInfo(*graph->add_output(), "output", kFloat16, {6, 8});
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
  return cudaGetDeviceCount(&device_count) == cudaSuccess &&
         device_count > 0;
}

}  // namespace

TEST(PackedAttentionWorkspaceEstimateTest, AggregateUsesMaxOfMutuallyExclusiveRoutes) {
  PackedAttentionInputShapes pa_inputs;
  pa_inputs.input = Shape({6, 6});
  pa_inputs.weights = Shape({6, 24});
  pa_inputs.bias = Shape({24});
  pa_inputs.token_offset = Shape({2, 4});
  pa_inputs.cumulative_sequence_length = Shape({3});
  pa_inputs.element_size = 2;
  pa_inputs.num_heads = 2;
  pa_inputs.qkv_hidden_sizes_count = 3;
  pa_inputs.qkv_hidden_sizes = {8, 8, 8};
  auto pa_problem = BuildPackedAttentionProblem(pa_inputs);
  ASSERT_TRUE(pa_problem.status.IsOK()) << pa_problem.status.message;

  auto mea_problem = pa_problem.problem;
  mea_problem.backend = PackedAttentionBackend::MemoryEfficient;
  const auto mea = GetPackedAttentionWorkspaceRecipe(mea_problem);
  ASSERT_TRUE(mea.status.IsOK()) << mea.status.message;
  auto unfused_problem = pa_problem.problem;
  unfused_problem.backend = PackedAttentionBackend::Unfused;
  const auto unfused = GetPackedAttentionWorkspaceRecipe(unfused_problem);
  ASSERT_TRUE(unfused.status.IsOK()) << unfused.status.message;

  const auto aggregate = GetPackedAttentionWorkspaceAggregate(
      pa_problem.problem,
      PackedAttentionBackendMask::MemoryEfficient |
          PackedAttentionBackendMask::Unfused);
  ASSERT_TRUE(aggregate.status.IsOK()) << aggregate.status.message;
  EXPECT_EQ(aggregate.projection_bytes, mea.recipe.projection_bytes);
  EXPECT_EQ(aggregate.attention_workspace_bytes,
            std::max(mea.recipe.attention_workspace_bytes,
                     unfused.recipe.attention_workspace_bytes));
  size_t expected_attention_offset = 0;
  ASSERT_TRUE(contrib::cuda::CheckedPackedAttentionAlign(
                  aggregate.projection_bytes, kPackedAttentionWorkspaceAlignment,
                  expected_attention_offset)
                  .IsOK());
  EXPECT_EQ(aggregate.attention_workspace_offset_bytes,
            expected_attention_offset);
  EXPECT_EQ(aggregate.attention_workspace_offset_bytes %
                kPackedAttentionWorkspaceAlignment,
            0U);
  EXPECT_GT(aggregate.attention_workspace_offset_bytes,
            aggregate.projection_bytes);
  EXPECT_LT(aggregate.attention_workspace_offset_bytes -
                aggregate.projection_bytes,
            kPackedAttentionWorkspaceAlignment);
  EXPECT_EQ(aggregate.total_workspace_bytes,
            aggregate.attention_workspace_offset_bytes +
                aggregate.attention_workspace_bytes);
  EXPECT_NE(aggregate.attention_workspace_bytes,
            mea.recipe.attention_workspace_bytes +
                unfused.recipe.attention_workspace_bytes);
}

TEST(PackedAttentionWorkspaceEstimateTest, Level2AdaptersDeclareOneAlignedRoot) {
  AttentionKernelOptions options;
  options.InitializeOnce(kMath, true);
  const auto shapes = PaShapes();
  const auto estimate = EstimatePackedAttentionWorkspace(
      PaConfig(), gsl::make_span(shapes), Sm80Device(), options);
  ASSERT_TRUE(estimate.has_value());

  InlinedVector<WorkspaceRequirement> requirements;
  SetPackedAttentionWorkspaceRequirements(*estimate, requirements);
  ASSERT_EQ(requirements.size(), 1U);
  EXPECT_EQ(requirements[0].slot_id, 0);
  EXPECT_EQ(requirements[0].size_bytes, estimate->total_workspace_bytes);
  EXPECT_EQ(requirements[0].alignment_bytes,
            kPackedAttentionWorkspaceAlignment);
  EXPECT_EQ(estimate->attention_workspace_offset_bytes %
                kPackedAttentionWorkspaceAlignment,
            0U);
  EXPECT_EQ(estimate->total_workspace_bytes,
            estimate->attention_workspace_offset_bytes +
                estimate->attention_workspace_bytes);
  EXPECT_GT(estimate->attention_workspace_offset_bytes,
            estimate->projection_bytes);
  EXPECT_LT(estimate->attention_workspace_offset_bytes -
                estimate->projection_bytes,
            kPackedAttentionWorkspaceAlignment);

  auto graph_problem_inputs = PackedAttentionInputShapes{};
  graph_problem_inputs.input = Shape({6, 6});
  graph_problem_inputs.weights = Shape({6, 24});
  graph_problem_inputs.bias = Shape({24});
  graph_problem_inputs.token_offset = Shape({2, 4});
  graph_problem_inputs.cumulative_sequence_length = Shape({3});
  graph_problem_inputs.element_size = 2;
  graph_problem_inputs.num_heads = 2;
  graph_problem_inputs.qkv_hidden_sizes_count = 3;
  graph_problem_inputs.qkv_hidden_sizes = {8, 8, 8};
  auto problem = BuildPackedAttentionProblem(graph_problem_inputs);
  ASSERT_TRUE(problem.status.IsOK()) << problem.status.message;
  const auto graph_free = GetPackedAttentionWorkspaceAggregate(
      problem.problem, PackedAttentionBackendMask::Unfused);
  ASSERT_TRUE(graph_free.status.IsOK()) << graph_free.status.message;
  EXPECT_EQ(estimate->total_workspace_bytes, graph_free.total_workspace_bytes);

  problem.problem.backend = PackedAttentionBackend::Unfused;
  const auto runtime_recipe = GetPackedAttentionWorkspaceRecipe(problem.problem);
  ASSERT_TRUE(runtime_recipe.status.IsOK()) << runtime_recipe.status.message;
  EXPECT_EQ(estimate->projection_bytes, runtime_recipe.recipe.projection_bytes);
  EXPECT_EQ(estimate->attention_workspace_bytes,
            runtime_recipe.recipe.attention_workspace_bytes);

  const auto pmha_estimate = EstimatePackedAttentionWorkspace(
      PmhaConfig(), gsl::make_span(PackedPmhaShapes()), Sm80Device(), options);
  ASSERT_TRUE(pmha_estimate.has_value());
  EXPECT_EQ(pmha_estimate->projection_bytes, 0U);
  EXPECT_EQ(pmha_estimate->attention_workspace_offset_bytes, 0U);
  SetPackedAttentionWorkspaceRequirements(*pmha_estimate, requirements);
  ASSERT_EQ(requirements.size(), 1U);
  EXPECT_EQ(requirements[0].slot_id, 0);
  EXPECT_EQ(requirements[0].size_bytes, pmha_estimate->total_workspace_bytes);
  EXPECT_EQ(requirements[0].alignment_bytes,
            kPackedAttentionWorkspaceAlignment);
}

TEST(PackedAttentionWorkspaceEstimateTest, PackedAttentionKernelDeclaresOneAlignedRoot) {
  if (!HasCudaDevice()) {
    GTEST_SKIP() << "A CUDA device is required to construct the CUDA kernel.";
  }

  SessionOptions session_options;
  session_options.graph_optimization_level = TransformerLevel::Default;
  session_options.session_logid = "PackedAttentionWorkspaceLevel2";
  InferenceSessionWrapper session(session_options, GetEnvironment());

  CUDAExecutionProviderInfo provider_info;
  provider_info.sdpa_kernel = kMath;
  auto cuda_ep = std::make_shared<CUDAExecutionProvider>(provider_info);
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(cuda_ep));
  const std::string model_bytes = BuildPackedAttentionKernelModel(
      PackedAttentionWorkspaceOperator::PackedAttention);
  ASSERT_STATUS_OK(
      session.Load(model_bytes.data(), static_cast<int>(model_bytes.size())));
  ASSERT_STATUS_OK(session.Initialize());

  const Graph& graph = session.GetGraph();
  const Node* node = FindNodeByOpType(graph, "PackedAttention");
  ASSERT_NE(node, nullptr);
  ASSERT_EQ(node->GetExecutionProviderType(), kCudaExecutionProvider);
  const OpKernel* kernel =
      session.GetSessionState().GetKernel(node->Index());
  ASSERT_NE(kernel, nullptr);

  const auto shapes = PaShapes();
  const auto expected = EstimatePackedAttentionWorkspace(
      PaConfig(), gsl::make_span(shapes), cuda_ep->GetDeviceProp(),
      *cuda_ep->GetAttentionKernelOptions());
  ASSERT_TRUE(expected.has_value());

  InlinedVector<WorkspaceRequirement> requirements;
  ASSERT_STATUS_OK(kernel->DeclareWorkspaceRequirements(
      gsl::make_span(shapes), requirements));
  ASSERT_EQ(requirements.size(), 1U);
  EXPECT_EQ(requirements[0].slot_id, 0);
  EXPECT_EQ(requirements[0].size_bytes, expected->total_workspace_bytes);
  EXPECT_EQ(requirements[0].alignment_bytes,
            kPackedAttentionWorkspaceAlignment);

  auto zero_shapes = shapes;
  zero_shapes[0] = KnownShape({0, 6});
  ASSERT_STATUS_OK(kernel->DeclareWorkspaceRequirements(
      gsl::make_span(zero_shapes), requirements));
  EXPECT_TRUE(requirements.empty());
}

TEST(PackedAttentionWorkspaceEstimateTest, PackedMultiHeadAttentionKernelDeclaresOneAlignedRoot) {
  if (!HasCudaDevice()) {
    GTEST_SKIP() << "A CUDA device is required to construct the CUDA kernel.";
  }

  SessionOptions session_options;
  session_options.graph_optimization_level = TransformerLevel::Default;
  session_options.session_logid = "PackedMultiHeadAttentionWorkspaceLevel2";
  InferenceSessionWrapper session(session_options, GetEnvironment());

  CUDAExecutionProviderInfo provider_info;
  provider_info.sdpa_kernel = kMath;
  auto cuda_ep = std::make_shared<CUDAExecutionProvider>(provider_info);
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(cuda_ep));
  const std::string model_bytes = BuildPackedAttentionKernelModel(
      PackedAttentionWorkspaceOperator::PackedMultiHeadAttention);
  ASSERT_STATUS_OK(
      session.Load(model_bytes.data(), static_cast<int>(model_bytes.size())));
  ASSERT_STATUS_OK(session.Initialize());

  const Graph& graph = session.GetGraph();
  const Node* node = FindNodeByOpType(graph, "PackedMultiHeadAttention");
  ASSERT_NE(node, nullptr);
  ASSERT_EQ(node->GetExecutionProviderType(), kCudaExecutionProvider);
  const OpKernel* kernel =
      session.GetSessionState().GetKernel(node->Index());
  ASSERT_NE(kernel, nullptr);

  const auto shapes = PackedPmhaShapes();
  const auto expected = EstimatePackedAttentionWorkspace(
      PmhaConfig(), gsl::make_span(shapes), cuda_ep->GetDeviceProp(),
      *cuda_ep->GetAttentionKernelOptions());
  ASSERT_TRUE(expected.has_value());

  InlinedVector<WorkspaceRequirement> requirements;
  ASSERT_STATUS_OK(kernel->DeclareWorkspaceRequirements(
      gsl::make_span(shapes), requirements));
  ASSERT_EQ(requirements.size(), 1U);
  EXPECT_EQ(requirements[0].slot_id, 0);
  EXPECT_EQ(requirements[0].size_bytes, expected->total_workspace_bytes);
  EXPECT_EQ(requirements[0].alignment_bytes,
            kPackedAttentionWorkspaceAlignment);

  auto zero_shapes = shapes;
  zero_shapes[0] = KnownShape({0, 2, 3, 4});
  ASSERT_STATUS_OK(kernel->DeclareWorkspaceRequirements(
      gsl::make_span(zero_shapes), requirements));
  EXPECT_TRUE(requirements.empty());
}

TEST(PackedAttentionWorkspaceEstimateTest, PositionalShapesHandlePmhaHolesAndLayouts) {
  AttentionKernelOptions options;
  options.InitializeOnce(kMath, true);

  const auto packed = EstimatePackedAttentionWorkspace(
      PmhaConfig(), gsl::make_span(PackedPmhaShapes()), Sm80Device(), options);
  ASSERT_TRUE(packed.has_value());
  EXPECT_GT(packed->attention_workspace_bytes, 0U);

  auto separate_shapes = SeparatePmhaShapes();
  const auto separate = EstimatePackedAttentionWorkspace(
      PmhaConfig(), gsl::make_span(separate_shapes), Sm80Device(), options);
  ASSERT_TRUE(separate.has_value());
  EXPECT_GT(separate->attention_workspace_bytes, 0U);

  separate_shapes[3] = KnownShape({24});
  const auto with_bias = EstimatePackedAttentionWorkspace(
      PmhaConfig(), gsl::make_span(separate_shapes), Sm80Device(), options);
  ASSERT_TRUE(with_bias.has_value());
}

TEST(PackedAttentionWorkspaceEstimateTest, UnknownAndMissingRequiredShapesAreUnavailable) {
  AttentionKernelOptions options;
  options.InitializeOnce(kMath, true);

  auto pa_shapes = PaShapes();
  pa_shapes[1] = KnownShape({-1, 24});
  EXPECT_FALSE(EstimatePackedAttentionWorkspace(
                   PaConfig(), gsl::make_span(pa_shapes), Sm80Device(), options)
                   .has_value());

  pa_shapes = PaShapes();
  pa_shapes[3] = WorkspaceInputShape::PresentWithoutShape();
  EXPECT_FALSE(EstimatePackedAttentionWorkspace(
                   PaConfig(), gsl::make_span(pa_shapes), Sm80Device(), options)
                   .has_value());

  pa_shapes = PaShapes();
  pa_shapes[4] = WorkspaceInputShape{};
  EXPECT_FALSE(EstimatePackedAttentionWorkspace(
                   PaConfig(), gsl::make_span(pa_shapes), Sm80Device(), options)
                   .has_value());

  auto pmha_shapes = SeparatePmhaShapes();
  pmha_shapes[2] = WorkspaceInputShape::PresentWithoutShape();
  EXPECT_FALSE(EstimatePackedAttentionWorkspace(
                   PmhaConfig(), gsl::make_span(pmha_shapes), Sm80Device(), options)
                   .has_value());

  pmha_shapes = SeparatePmhaShapes();
  pmha_shapes[6] = WorkspaceInputShape::PresentWithoutShape();
  EXPECT_FALSE(EstimatePackedAttentionWorkspace(
                   PmhaConfig(), gsl::make_span(pmha_shapes), Sm80Device(), options)
                   .has_value());
}

TEST(PackedAttentionWorkspaceEstimateTest, ZeroShapeHintsAreUnavailable) {
  AttentionKernelOptions options;
  options.InitializeOnce(kMath, true);

  auto pa_shapes = PaShapes();
  pa_shapes[0] = KnownShape({0, 6});
  EXPECT_FALSE(EstimatePackedAttentionWorkspace(
                   PaConfig(), gsl::make_span(pa_shapes), Sm80Device(), options)
                   .has_value());
  PackedAttentionInputShapes pa_inputs;
  pa_inputs.input = Shape({0, 6});
  pa_inputs.weights = Shape({6, 24});
  pa_inputs.bias = Shape({24});
  pa_inputs.token_offset = Shape({2, 4});
  pa_inputs.cumulative_sequence_length = Shape({3});
  pa_inputs.element_size = 2;
  pa_inputs.num_heads = 2;
  pa_inputs.qkv_hidden_sizes_count = 3;
  pa_inputs.qkv_hidden_sizes = {8, 8, 8};
  auto pa_problem = BuildPackedAttentionProblem(pa_inputs);
  ASSERT_TRUE(pa_problem.status.IsOK()) << pa_problem.status.message;
  auto zero_aggregate = GetPackedAttentionWorkspaceAggregate(
      pa_problem.problem, PackedAttentionBackendMask::Unfused);
  ExpectEmptyAggregate(zero_aggregate);

  auto zero_v_config = PaConfig();
  zero_v_config.qkv_hidden_sizes = {8, 8, 0};
  pa_shapes = PaShapes();
  pa_shapes[1] = KnownShape({6, 16});
  pa_shapes[2] = KnownShape({16});
  EXPECT_FALSE(EstimatePackedAttentionWorkspace(
                   zero_v_config, gsl::make_span(pa_shapes), Sm80Device(), options)
                   .has_value());
  pa_inputs.input = Shape({6, 6});
  pa_inputs.weights = Shape({6, 16});
  pa_inputs.bias = Shape({16});
  pa_inputs.qkv_hidden_sizes = {8, 8, 0};
  pa_problem = BuildPackedAttentionProblem(pa_inputs);
  ASSERT_TRUE(pa_problem.status.IsOK()) << pa_problem.status.message;
  zero_aggregate = GetPackedAttentionWorkspaceAggregate(
      pa_problem.problem, PackedAttentionBackendMask::Unfused);
  ExpectEmptyAggregate(zero_aggregate);

  auto pmha_shapes = PackedPmhaShapes();
  pmha_shapes[0] = KnownShape({0, 2, 3, 4});
  EXPECT_FALSE(EstimatePackedAttentionWorkspace(
                   PmhaConfig(), gsl::make_span(pmha_shapes), Sm80Device(), options)
                   .has_value());
  PackedMultiHeadAttentionInputShapes pmha_inputs;
  pmha_inputs.query = Shape({0, 2, 3, 4});
  pmha_inputs.token_offset = Shape({2, 4});
  pmha_inputs.cumulative_sequence_length = Shape({3});
  pmha_inputs.element_size = 2;
  pmha_inputs.num_heads = 2;
  auto pmha_problem = BuildPackedMultiHeadAttentionProblem(pmha_inputs);
  ASSERT_TRUE(pmha_problem.status.IsOK()) << pmha_problem.status.message;
  zero_aggregate = GetPackedMultiHeadAttentionWorkspaceAggregate(
      pmha_problem.problem, PackedAttentionBackendMask::Unfused);
  ExpectEmptyAggregate(zero_aggregate);

  pmha_shapes = SeparatePmhaShapes();
  pmha_shapes[0] = KnownShape({0, 8});
  pmha_shapes[1] = KnownShape({0, 8});
  pmha_shapes[2] = KnownShape({0, 8});
  EXPECT_FALSE(EstimatePackedAttentionWorkspace(
                   PmhaConfig(), gsl::make_span(pmha_shapes), Sm80Device(), options)
                   .has_value());
  pmha_inputs.query = Shape({0, 8});
  pmha_inputs.key = Shape({0, 8});
  pmha_inputs.value = Shape({0, 8});
  pmha_inputs.has_key = true;
  pmha_inputs.has_value = true;
  pmha_problem = BuildPackedMultiHeadAttentionProblem(pmha_inputs);
  ASSERT_TRUE(pmha_problem.status.IsOK()) << pmha_problem.status.message;
  zero_aggregate = GetPackedMultiHeadAttentionWorkspaceAggregate(
      pmha_problem.problem, PackedAttentionBackendMask::Unfused);
  ExpectEmptyAggregate(zero_aggregate);
}

TEST(PackedAttentionWorkspaceEstimateTest, RouteSetConservativelyIncludesFallbacks) {
  const auto device = Sm80Device();

  AttentionKernelOptions all_options;
  all_options.InitializeOnce(kFlash | kMea | kTrt | kTrtFlash | kMath, true);
  const auto pa_routes =
      GetPackedAttentionFeasibleBackends(RoutePaProblem(), device, all_options);
  EXPECT_TRUE(HasRoute(pa_routes, PackedAttentionBackend::Trt));
#if USE_MEMORY_EFFICIENT_ATTENTION
  EXPECT_TRUE(HasRoute(pa_routes, PackedAttentionBackend::MemoryEfficient));
#endif
  EXPECT_TRUE(HasRoute(pa_routes, PackedAttentionBackend::Unfused));
  EXPECT_FALSE(HasRoute(pa_routes, PackedAttentionBackend::Flash));

  const auto flash_routes =
      GetPackedMultiHeadAttentionFeasibleBackends(RoutePmhaProblem(), device, all_options);
#if USE_FLASH_ATTENTION
  EXPECT_TRUE(HasRoute(flash_routes, PackedAttentionBackend::Flash));
#else
  EXPECT_FALSE(HasRoute(flash_routes, PackedAttentionBackend::Flash));
#endif
  EXPECT_TRUE(HasRoute(flash_routes, PackedAttentionBackend::Unfused));

  AttentionKernelOptions trt_mea_options;
  trt_mea_options.InitializeOnce(kMea | kTrt | kTrtFlash | kMath, true);
  const auto trt_fallback_routes =
      GetPackedMultiHeadAttentionFeasibleBackends(
          RoutePmhaProblem(), device, trt_mea_options);
  EXPECT_TRUE(HasRoute(trt_fallback_routes, PackedAttentionBackend::Trt));
#if USE_MEMORY_EFFICIENT_ATTENTION
  EXPECT_TRUE(HasRoute(trt_fallback_routes, PackedAttentionBackend::MemoryEfficient));
#endif
  EXPECT_TRUE(HasRoute(trt_fallback_routes, PackedAttentionBackend::Unfused));

  AttentionKernelOptions mea_options;
  mea_options.InitializeOnce(kMea | kMath, true);
  const auto mea_routes =
      GetPackedMultiHeadAttentionFeasibleBackends(RoutePmhaProblem(), device, mea_options);
#if USE_MEMORY_EFFICIENT_ATTENTION
  EXPECT_TRUE(HasRoute(mea_routes, PackedAttentionBackend::MemoryEfficient));
#endif
  EXPECT_TRUE(HasRoute(mea_routes, PackedAttentionBackend::Unfused));

  AttentionKernelOptions math_options;
  math_options.InitializeOnce(kMath, true);
  EXPECT_EQ(GetPackedMultiHeadAttentionFeasibleBackends(
                RoutePmhaProblem(), device, math_options),
            PackedAttentionBackendMask::Unfused);
  EXPECT_EQ(GetPackedAttentionFeasibleBackends(
                RoutePaProblem(7), device, math_options),
            PackedAttentionBackendMask::Unfused);
}

TEST(PackedAttentionWorkspaceEstimateTest, DefaultPackedPmhaCrossesFlashThreshold) {
  AttentionKernelOptions options;
  options.InitializeOnce(0, true);
  EXPECT_EQ(options.MinSeqLenForFlashAttentionPackedQkv(), 513);

  const auto maximum_problem =
      BuildPmhaProblem(/*sequence_length=*/1024, /*element_size=*/2,
                       /*packed=*/true, /*has_attention_bias=*/false,
                       /*head_size=*/128);
  const auto maximum_routes = GetPackedMultiHeadAttentionFeasibleBackends(
      maximum_problem, Sm80Device(), options);
  EXPECT_TRUE(HasRoute(maximum_routes, PackedAttentionBackend::Unfused));
#if USE_FLASH_ATTENTION
  EXPECT_TRUE(HasRoute(maximum_routes, PackedAttentionBackend::Flash));
#endif

  const auto runtime_problem =
      BuildPmhaProblem(/*sequence_length=*/512, /*element_size=*/2,
                       /*packed=*/true, /*has_attention_bias=*/false,
                       /*head_size=*/128);
  const auto runtime_bound_routes = GetPackedMultiHeadAttentionFeasibleBackends(
      runtime_problem, Sm80Device(), options);
  EXPECT_FALSE(HasRoute(runtime_bound_routes, PackedAttentionBackend::Flash));
  EXPECT_TRUE(HasRoute(runtime_bound_routes, PackedAttentionBackend::Unfused));

  const auto shapes = PackedPmhaShapes(/*sequence_length=*/1024,
                                       /*head_size=*/128);
  const auto estimate = EstimatePackedAttentionWorkspace(
      PmhaConfig(/*element_size=*/2), gsl::make_span(shapes),
      Sm80Device(), options);
  ASSERT_TRUE(estimate.has_value());
  ExpectPmhaAggregateUsesMaximum(maximum_problem, maximum_routes, *estimate);
}

TEST(PackedAttentionWorkspaceEstimateTest, DefaultFp32PmhaCrossesMeaThreshold) {
  AttentionKernelOptions options;
  options.InitializeOnce(0, true);
  EXPECT_EQ(options.MinSeqLenForEfficientAttentionFp32(), 256);

  const auto maximum_problem =
      BuildPmhaProblem(/*sequence_length=*/512, /*element_size=*/4,
                       /*packed=*/false, /*has_attention_bias=*/true);
  const auto maximum_routes = GetPackedMultiHeadAttentionFeasibleBackends(
      maximum_problem, Sm80Device(), options);
  EXPECT_TRUE(HasRoute(maximum_routes, PackedAttentionBackend::Unfused));
#if USE_MEMORY_EFFICIENT_ATTENTION
  EXPECT_TRUE(HasRoute(maximum_routes, PackedAttentionBackend::MemoryEfficient));
#endif

  const auto smaller_problem =
      BuildPmhaProblem(/*sequence_length=*/128, /*element_size=*/4,
                       /*packed=*/false, /*has_attention_bias=*/true);
  const auto smaller_routes = GetPackedMultiHeadAttentionFeasibleBackends(
      smaller_problem, Sm80Device(), options);
  EXPECT_FALSE(HasRoute(smaller_routes, PackedAttentionBackend::MemoryEfficient));
  EXPECT_TRUE(HasRoute(smaller_routes, PackedAttentionBackend::Unfused));

  const auto shapes = SeparatePmhaShapes(
      /*sequence_length=*/512, /*has_attention_bias=*/true);
  const auto estimate = EstimatePackedAttentionWorkspace(
      PmhaConfig(/*element_size=*/4), gsl::make_span(shapes),
      Sm80Device(), options);
  ASSERT_TRUE(estimate.has_value());
  ExpectPmhaAggregateUsesMaximum(maximum_problem, maximum_routes, *estimate);
}

TEST(PackedAttentionWorkspaceEstimateTest, AttentionBiasAlignmentIsPossibleBelowMaximum) {
  AttentionKernelOptions options;
  options.InitializeOnce(0, true);

  const auto too_small_problem =
      BuildPmhaProblem(/*sequence_length=*/7, /*element_size=*/2,
                       /*packed=*/false, /*has_attention_bias=*/true);
  const auto too_small_routes = GetPackedMultiHeadAttentionFeasibleBackends(
      too_small_problem, Sm80Device(), options);
  EXPECT_FALSE(HasRoute(too_small_routes, PackedAttentionBackend::MemoryEfficient));
  EXPECT_TRUE(HasRoute(too_small_routes, PackedAttentionBackend::Unfused));

  for (int32_t maximum_sequence_length : {10, 16}) {
    SCOPED_TRACE(maximum_sequence_length);
    const auto problem =
        BuildPmhaProblem(maximum_sequence_length, /*element_size=*/2,
                         /*packed=*/false, /*has_attention_bias=*/true);
    const auto routes = GetPackedMultiHeadAttentionFeasibleBackends(
        problem, Sm80Device(), options);
    EXPECT_TRUE(HasRoute(routes, PackedAttentionBackend::Unfused));
#if USE_MEMORY_EFFICIENT_ATTENTION
    EXPECT_TRUE(HasRoute(routes, PackedAttentionBackend::MemoryEfficient));
#endif

    const auto shapes = SeparatePmhaShapes(
        maximum_sequence_length, /*has_attention_bias=*/true);
    const auto estimate = EstimatePackedAttentionWorkspace(
        PmhaConfig(/*element_size=*/2), gsl::make_span(shapes),
        Sm80Device(), options);
    ASSERT_TRUE(estimate.has_value());
    ExpectPmhaAggregateUsesMaximum(problem, routes, *estimate);
  }
}

TEST(PackedAttentionWorkspaceEstimateTest, NodeOverloadParsesAttributesDtypeAndShapes) {
  AttentionKernelOptions options;
  options.InitializeOnce(kMath, true);

  const auto pa_shapes = PaShapes();
  const auto pa = EstimateFromNode(
      "PackedAttention", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
      /*num_heads=*/2, std::vector<int64_t>{8, 8, 8},
      gsl::make_span(pa_shapes), options);
  ASSERT_TRUE(pa.has_value());
  EXPECT_GT(pa->total_workspace_bytes, 0U);

  const auto pmha_shapes = PackedPmhaShapes();
  const auto pmha = EstimateFromNode(
      "PackedMultiHeadAttention", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
      /*num_heads=*/2, std::nullopt, gsl::make_span(pmha_shapes), options);
  ASSERT_TRUE(pmha.has_value());
  EXPECT_GT(pmha->total_workspace_bytes, 0U);
  EXPECT_EQ(pmha->projection_bytes, 0U);

  EXPECT_FALSE(EstimateFromNode(
                   "PackedAttention", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
                   std::nullopt, std::vector<int64_t>{8, 8, 8},
                   gsl::make_span(pa_shapes), options)
                   .has_value());
  EXPECT_FALSE(EstimateFromNode(
                   "PackedAttention", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
                   /*num_heads=*/0, std::vector<int64_t>{8, 8, 8},
                   gsl::make_span(pa_shapes), options)
                   .has_value());
  EXPECT_FALSE(EstimateFromNode(
                   "PackedAttention", ONNX_NAMESPACE::TensorProto_DataType_INT32,
                   /*num_heads=*/2, std::vector<int64_t>{8, 8, 8},
                   gsl::make_span(pa_shapes), options)
                   .has_value());
  EXPECT_FALSE(EstimateFromNode(
                   "PackedAttention", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
                   /*num_heads=*/2, std::vector<int64_t>{8, 8},
                   gsl::make_span(pa_shapes), options)
                   .has_value());
  EXPECT_FALSE(EstimateFromNode(
                   "PackedAttention", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
                   /*num_heads=*/2, std::vector<int64_t>{8, -8, 8},
                   gsl::make_span(pa_shapes), options)
                   .has_value());
  EXPECT_FALSE(EstimateFromNode(
                   "PackedAttention", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
                   /*num_heads=*/2, std::vector<int64_t>{7, 7, 8},
                   gsl::make_span(pa_shapes), options)
                   .has_value());
  EXPECT_FALSE(EstimateFromNode(
                   "PackedMultiHeadAttention", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
                   /*num_heads=*/2, std::vector<int64_t>{8, 8, 8},
                   gsl::make_span(pmha_shapes), options)
                   .has_value());
}

TEST(PackedAttentionWorkspaceEstimateTest, AttentionBiasDisablesFlashAndTrtCandidates) {
  AttentionKernelOptions options;
  options.InitializeOnce(kFlash | kMea | kTrt | kTrtFlash | kMath, true);
  auto problem = RoutePmhaProblem();
  problem.has_attention_bias = true;
  problem.broadcast_attn_bias_dim_0 = true;
  problem.broadcast_attn_bias_dim_1 = true;
  const auto routes =
      GetPackedMultiHeadAttentionFeasibleBackends(problem, Sm80Device(), options);
  EXPECT_FALSE(HasRoute(routes, PackedAttentionBackend::Flash));
  EXPECT_FALSE(HasRoute(routes, PackedAttentionBackend::Trt));
#if USE_MEMORY_EFFICIENT_ATTENTION
  EXPECT_TRUE(HasRoute(routes, PackedAttentionBackend::MemoryEfficient));
#endif
  EXPECT_TRUE(HasRoute(routes, PackedAttentionBackend::Unfused));
}

TEST(PackedAttentionWorkspaceEstimateTest, IncludedRouteOverflowMakesAggregateUnavailable) {
  PackedMultiHeadAttentionProblem problem;
  problem.element_size = 2;
  problem.token_count = 1;
  problem.batch_size = 65535;
  problem.sequence_length = 65535;
  problem.num_heads = 65535;
  problem.qk_head_size = 32768;
  problem.v_head_size = 32768;
  problem.hidden_size = 65535 * 32768;
  problem.v_hidden_size = problem.hidden_size;
  problem.qkv_format = PackedMultiHeadAttentionQkvFormat::Separate;
  problem.qkv_materialization_index_width =
      PackedAttentionQkvMaterializationIndexWidth::Vector4;

  problem.backend = PackedAttentionBackend::MemoryEfficient;
  const auto single_route = GetPackedMultiHeadAttentionWorkspaceRecipe(problem);
  EXPECT_EQ(single_route.status.error, PackedAttentionWorkspaceError::Overflow);

  const auto aggregate = GetPackedMultiHeadAttentionWorkspaceAggregate(
      problem, PackedAttentionBackendMask::MemoryEfficient);
  EXPECT_EQ(aggregate.status.error, PackedAttentionWorkspaceError::Overflow);
}

TEST(PackedAttentionWorkspaceEstimateTest, InvalidAndEmptyRouteMasksAreDistinguished) {
  auto nonempty = RoutePaProblem();
  EXPECT_EQ(GetPackedAttentionWorkspaceAggregate(
                nonempty, PackedAttentionBackendMask::None)
                .status.error,
            PackedAttentionWorkspaceError::InvalidArgument);

  nonempty.token_count = 0;
  const auto empty = GetPackedAttentionWorkspaceAggregate(
      nonempty, PackedAttentionBackendMask::None);
  ExpectEmptyAggregate(empty);

  const auto invalid_mask = static_cast<PackedAttentionBackendMask>(1U << 31);
  EXPECT_EQ(GetPackedAttentionWorkspaceAggregate(nonempty, invalid_mask).status.error,
            PackedAttentionWorkspaceError::InvalidArgument);

  auto empty_pmha = RoutePmhaProblem();
  empty_pmha.token_count = 0;
  EXPECT_EQ(GetPackedMultiHeadAttentionWorkspaceAggregate(empty_pmha, invalid_mask)
                .status.error,
            PackedAttentionWorkspaceError::InvalidArgument);
}

TEST(PackedAttentionWorkspaceEstimateTest, EmptyAggregatesStillValidateProblemGeometry) {
  auto valid_pa = RoutePaProblem();
  valid_pa.token_count = 0;
  auto aggregate =
      GetPackedAttentionWorkspaceAggregate(valid_pa, PackedAttentionBackendMask::None);
  ExpectEmptyAggregate(aggregate);

  auto malformed_pa = valid_pa;
  malformed_pa.input_hidden_size = -1;
  EXPECT_EQ(GetPackedAttentionWorkspaceAggregate(
                malformed_pa, PackedAttentionBackendMask::None)
                .status.error,
            PackedAttentionWorkspaceError::InvalidArgument);

  malformed_pa = valid_pa;
  malformed_pa.hidden_size += 1;
  EXPECT_EQ(GetPackedAttentionWorkspaceAggregate(
                malformed_pa, PackedAttentionBackendMask::None)
                .status.error,
            PackedAttentionWorkspaceError::InvalidArgument);

  auto valid_pmha = RoutePmhaProblem();
  valid_pmha.token_count = 0;
  aggregate = GetPackedMultiHeadAttentionWorkspaceAggregate(
      valid_pmha, PackedAttentionBackendMask::None);
  ExpectEmptyAggregate(aggregate);

  auto malformed_pmha = valid_pmha;
  malformed_pmha.batch_size = -1;
  EXPECT_EQ(GetPackedMultiHeadAttentionWorkspaceAggregate(
                malformed_pmha, PackedAttentionBackendMask::None)
                .status.error,
            PackedAttentionWorkspaceError::InvalidArgument);

  malformed_pmha = valid_pmha;
  malformed_pmha.hidden_size += 1;
  EXPECT_EQ(GetPackedMultiHeadAttentionWorkspaceAggregate(
                malformed_pmha, PackedAttentionBackendMask::None)
                .status.error,
            PackedAttentionWorkspaceError::InvalidArgument);

  malformed_pmha = valid_pmha;
  malformed_pmha.qkv_format =
      static_cast<PackedMultiHeadAttentionQkvFormat>(-1);
  EXPECT_EQ(GetPackedMultiHeadAttentionWorkspaceAggregate(
                malformed_pmha, PackedAttentionBackendMask::None)
                .status.error,
            PackedAttentionWorkspaceError::InvalidArgument);

  PackedMultiHeadAttentionInputShapes malformed_presence;
  malformed_presence.query = Shape({0, 64});
  malformed_presence.token_offset = Shape({1, 1});
  malformed_presence.cumulative_sequence_length = Shape({2});
  malformed_presence.element_size = 2;
  malformed_presence.num_heads = 1;
  malformed_presence.has_key = true;
  malformed_presence.key = Shape({0, 64});
  EXPECT_EQ(BuildPackedMultiHeadAttentionProblem(malformed_presence)
                .status.error,
            PackedAttentionWorkspaceError::InvalidArgument);

  malformed_presence.query = Shape({0, 1, 3, 64});
  malformed_presence.has_value = true;
  malformed_presence.value = Shape({0, 64});
  EXPECT_EQ(BuildPackedMultiHeadAttentionProblem(malformed_presence)
                .status.error,
            PackedAttentionWorkspaceError::InvalidArgument);
}

TEST(PackedAttentionWorkspaceEstimateTest, FailedAggregateEmitsNoRequirements) {
  PackedAttentionWorkspaceAggregate failed;
  failed.status.error = PackedAttentionWorkspaceError::InvalidArgument;
  failed.status.message = "deliberately failed estimate";
  failed.projection_bytes = 1024;
  failed.attention_workspace_offset_bytes = 1280;
  failed.attention_workspace_bytes = 2048;
  failed.total_workspace_bytes = 3328;

  InlinedVector<WorkspaceRequirement> requirements{
      WorkspaceRequirement{4096, /*slot_id=*/7, /*alignment_bytes=*/0}};
  SetPackedAttentionWorkspaceRequirements(failed, requirements);
  EXPECT_TRUE(requirements.empty());

  PackedAttentionWorkspaceAggregate empty;
  requirements.push_back(
      WorkspaceRequirement{4096, /*slot_id=*/7, /*alignment_bytes=*/0});
  SetPackedAttentionWorkspaceRequirements(empty, requirements);
  EXPECT_TRUE(requirements.empty());
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(DISABLE_CONTRIB_OPS) && !defined(BUILD_CUDA_EP_AS_PLUGIN)
