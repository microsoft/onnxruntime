// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Coverage for com.microsoft.PackedSparseAttentionIndexer.
//
// The "ShapeInference" suite drives Graph::Resolve() directly, so it runs in every build: it pins
// the fixed selected_indices/selected_counts shapes, the fixed (never data-dependent) state output
// shapes, and the strict policy validation. Cases that are expected to fail shape inference call
// fail_shape_inference, which aborts in ORT_NO_EXCEPTIONS builds, so they are compiled out there.
//
// The numeric suite needs the CUDA or WebGPU execution provider (the operator has no CPU kernel)
// and is skipped when neither is available. Expectations come from a float reference in this file
// that mirrors the operator contract (mean-pool/RMSNorm/RoPE for qsa, softmax-gated pooling for
// csa, deterministic top-k selection); the inputs are rounded to the tested element type first so
// the reference sees exactly what the kernel reads.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "contrib_ops/cpu/sparse/packed_sparse_attention_indexer_common.h"
#include "core/graph/constants.h"
#include "core/graph/model.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/test_environment.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

namespace psai = ::onnxruntime::contrib::packed_sparse_attention_indexer;

namespace {

constexpr int kOnnxOpsetVersion = 17;

// ---------------------------------------------------------------------------------------------
// Shape inference helpers
// ---------------------------------------------------------------------------------------------

Status BuildAndResolve(const std::function<void(ModelTestBuilder& builder)>& add_node,
                       std::unique_ptr<Model>& model) {
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = kOnnxOpsetVersion;
  domain_to_version[kMSDomain] = 1;

  model = std::unique_ptr<Model>(new Model("packed_sparse_attention_indexer", /*is_onnx_domain_only=*/false,
                                           ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
                                           domain_to_version, {}, DefaultLoggingManager().DefaultLogger()));

  ModelTestBuilder builder(model->MainGraph());
  add_node(builder);
  builder.SetGraphOutputs();
  return model->MainGraph().Resolve();
}

void ExpectShape(const Graph& graph, const std::string& name, ONNX_NAMESPACE::TensorProto_DataType elem_type,
                 const std::vector<int64_t>& expected) {
  const NodeArg* arg = graph.GetNodeArg(name);
  ASSERT_NE(arg, nullptr);
  const ONNX_NAMESPACE::TypeProto* type = arg->TypeAsProto();
  ASSERT_NE(type, nullptr);
  ASSERT_TRUE(type->has_tensor_type());
  EXPECT_EQ(type->tensor_type().elem_type(), static_cast<int32_t>(elem_type));
  const ONNX_NAMESPACE::TensorShapeProto& shape = type->tensor_type().shape();
  ASSERT_EQ(shape.dim_size(), static_cast<int>(expected.size()));
  for (int i = 0; i < shape.dim_size(); ++i) {
    ASSERT_TRUE(shape.dim(i).has_dim_value()) << "dimension " << i << " of " << name << " is not static";
    EXPECT_EQ(shape.dim(i).dim_value(), expected[static_cast<size_t>(i)]) << "dimension " << i << " of " << name;
  }
}

struct GraphOptions {
  int64_t batch_size = 2;
  int64_t total_tokens = 5;
  int64_t num_heads = 2;
  int64_t head_size = 8;
  int64_t rotary_width = 4;
  int64_t compress_ratio = 2;
  int64_t state_capacity = 6;
  int64_t input_state_capacity = -1;
  int64_t token_budget = 4;
  int64_t key_total_tokens = -1;
  int64_t kv_buffer_capacity = -1;
  int64_t gate_buffer_width = -1;
  bool add_index_topk = false;
  bool add_csa_inputs = false;
  bool add_position_ids = false;
  int output_count = psai::kFixedOutputCount;
  std::string policy_mode = psai::kPolicyModeQsa;
};

int64_t BufferCapacity(int64_t compress_ratio) { return 2 * compress_ratio - 1; }

// Builds a fixed 15-input node; csa-only slots are left empty for policy_mode "qsa", as the schema
// requires. position_ids (slot 10) is optional for "qsa" and forced on for "csa".
void AddNode(ModelTestBuilder& builder, const GraphOptions& options) {
  const bool is_csa = options.policy_mode == psai::kPolicyModeCsa;
  const int64_t width = is_csa ? 2 * options.head_size : options.head_size;
  const int64_t buffer_capacity = BufferCapacity(options.compress_ratio);
  NodeArg& empty = builder.graph_.GetOrCreateNodeArg("", nullptr);

  std::vector<NodeArg*> inputs{
      builder.MakeInput<float>(
          std::vector<int64_t>{options.total_tokens, options.num_heads, options.head_size}),
      builder.MakeInput<float>(
          std::vector<int64_t>{options.key_total_tokens >= 0 ? options.key_total_tokens : options.total_tokens, width}),
      builder.MakeInput<float>(std::vector<int64_t>{options.head_size}),
      builder.MakeInput<float>(std::vector<int64_t>{64, options.rotary_width}),
      builder.MakeInput<float>(std::vector<int64_t>{64, options.rotary_width}),
      builder.MakeInput<int32_t>(std::vector<int64_t>{options.batch_size + 1}),
      builder.MakeInput<int32_t>(std::vector<int64_t>{options.batch_size}),
  };
  if (is_csa || options.add_csa_inputs) {
    inputs.push_back(builder.MakeInput<float>(std::vector<int64_t>{options.total_tokens, width}));
    inputs.push_back(builder.MakeInput<float>(std::vector<int64_t>{options.compress_ratio, width}));
    inputs.push_back(builder.MakeInput<float>(std::vector<int64_t>{options.total_tokens, options.num_heads}));
  } else {
    inputs.push_back(&empty);
    inputs.push_back(&empty);
    inputs.push_back(&empty);
  }
  if (is_csa || options.add_position_ids) {
    inputs.push_back(builder.MakeInput<int64_t>(std::vector<int64_t>{options.total_tokens}));
  } else {
    inputs.push_back(&empty);
  }
  const int64_t input_state_capacity =
      options.input_state_capacity >= 0 ? options.input_state_capacity : options.state_capacity;
  inputs.push_back(
      builder.MakeInput<float>(std::vector<int64_t>{options.batch_size, input_state_capacity, options.head_size}));
  inputs.push_back(
      builder.MakeInput<float>(std::vector<int64_t>{
          options.batch_size, options.kv_buffer_capacity >= 0 ? options.kv_buffer_capacity : buffer_capacity, width}));
  if (is_csa) {
    inputs.push_back(builder.MakeInput<float>(
        std::vector<int64_t>{options.batch_size, buffer_capacity,
                             options.gate_buffer_width >= 0 ? options.gate_buffer_width : width}));
  } else {
    inputs.push_back(&empty);
  }
  inputs.push_back(builder.MakeInput<int32_t>(std::vector<int64_t>{options.batch_size, 2}));

  std::vector<NodeArg*> outputs;
  for (int i = 0; i < options.output_count; ++i) {
    outputs.push_back(i == psai::kPresentGateBuffer && !is_csa ? &empty : builder.MakeOutput());
  }
  Node& node = builder.AddNode("PackedSparseAttentionIndexer", inputs, outputs, kMSDomain);
  node.AddAttribute("policy_mode", options.policy_mode);
  node.AddAttribute("compress_ratio", options.compress_ratio);
  node.AddAttribute("state_capacity", options.state_capacity);
  if (is_csa || options.add_index_topk) {
    node.AddAttribute("index_topk", static_cast<int64_t>(3));
  } else {
    node.AddAttribute("token_budget", options.token_budget);
  }
}

}  // namespace

TEST(PackedSparseAttentionIndexerShapeInferenceTest, QsaInfersFixedCapacityAndState) {
  GraphOptions options;
  std::unique_ptr<Model> model;
  ASSERT_STATUS_OK(BuildAndResolve([&options](ModelTestBuilder& builder) { AddNode(builder, options); }, model));

  const Graph& graph = model->MainGraph();
  const Node& node = *graph.Nodes().begin();
  const int64_t capacity = options.token_budget + options.compress_ratio - 1;
  ExpectShape(graph, node.OutputDefs()[psai::kSelectedIndices]->Name(), ONNX_NAMESPACE::TensorProto_DataType_INT32,
              {options.total_tokens, capacity});
  ExpectShape(graph, node.OutputDefs()[psai::kSelectedCounts]->Name(), ONNX_NAMESPACE::TensorProto_DataType_INT32,
              {options.total_tokens});
  ExpectShape(graph, node.OutputDefs()[psai::kPresentKeyState]->Name(), ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
              {options.batch_size, options.state_capacity, options.head_size});
  ExpectShape(graph, node.OutputDefs()[psai::kPresentKvBuffer]->Name(), ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
              {options.batch_size, BufferCapacity(options.compress_ratio), options.head_size});
  ExpectShape(graph, node.OutputDefs()[psai::kPresentStateLengths]->Name(),
              ONNX_NAMESPACE::TensorProto_DataType_INT32, {options.batch_size, 2});
}

TEST(PackedSparseAttentionIndexerShapeInferenceTest, CsaInfersFixedCapacityAndState) {
  GraphOptions options;
  options.policy_mode = psai::kPolicyModeCsa;
  std::unique_ptr<Model> model;
  ASSERT_STATUS_OK(BuildAndResolve([&options](ModelTestBuilder& builder) { AddNode(builder, options); }, model));

  const Graph& graph = model->MainGraph();
  const Node& node = *graph.Nodes().begin();
  ExpectShape(graph, node.OutputDefs()[psai::kSelectedIndices]->Name(), ONNX_NAMESPACE::TensorProto_DataType_INT32,
              {options.total_tokens, 3});
  ExpectShape(graph, node.OutputDefs()[psai::kSelectedCounts]->Name(), ONNX_NAMESPACE::TensorProto_DataType_INT32,
              {options.total_tokens});
  ExpectShape(graph, node.OutputDefs()[psai::kPresentKeyState]->Name(), ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
              {options.batch_size, options.state_capacity, options.head_size});
  ExpectShape(graph, node.OutputDefs()[psai::kPresentKvBuffer]->Name(), ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
              {options.batch_size, BufferCapacity(options.compress_ratio), 2 * options.head_size});
  ExpectShape(graph, node.OutputDefs()[psai::kPresentGateBuffer]->Name(), ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
              {options.batch_size, BufferCapacity(options.compress_ratio), 2 * options.head_size});
  ExpectShape(graph, node.OutputDefs()[psai::kPresentStateLengths]->Name(),
              ONNX_NAMESPACE::TensorProto_DataType_INT32, {options.batch_size, 2});
}

#ifndef ORT_NO_EXCEPTIONS

namespace {

void ExpectResolveFailure(const std::function<void(ModelTestBuilder& builder)>& add_node,
                          const std::string& expected_message) {
  std::unique_ptr<Model> model;
  const Status status = BuildAndResolve(add_node, model);
  ASSERT_FALSE(status.IsOK()) << "expected shape inference to reject the node";
  EXPECT_NE(status.ErrorMessage().find(expected_message), std::string::npos)
      << "actual message: " << status.ErrorMessage();
}

}  // namespace

TEST(PackedSparseAttentionIndexerShapeInferenceTest, RejectsStateCapacityShapeMismatch) {
  GraphOptions options;
  options.input_state_capacity = options.state_capacity + 1;
  ExpectResolveFailure([&options](ModelTestBuilder& builder) { AddNode(builder, options); },
                       "past_key_state dimension 1 must equal state_capacity");
}

TEST(PackedSparseAttentionIndexerShapeInferenceTest, RejectsMismatchedKeyTokenCount) {
  GraphOptions options;
  options.key_total_tokens = options.total_tokens + 1;
  ExpectResolveFailure([&options](ModelTestBuilder& builder) { AddNode(builder, options); },
                       "key dimension 0 must equal query dimension 0");
}

TEST(PackedSparseAttentionIndexerShapeInferenceTest, RejectsWrongKvBufferCapacity) {
  GraphOptions options;
  options.kv_buffer_capacity = BufferCapacity(options.compress_ratio) + 1;
  ExpectResolveFailure([&options](ModelTestBuilder& builder) { AddNode(builder, options); },
                       "past_kv_buffer dimension 1 must equal 2 * compress_ratio - 1");
}

TEST(PackedSparseAttentionIndexerShapeInferenceTest, RejectsMismatchedCsaGateBufferWidth) {
  GraphOptions options;
  options.policy_mode = psai::kPolicyModeCsa;
  options.gate_buffer_width = 2 * options.head_size + 1;
  ExpectResolveFailure([&options](ModelTestBuilder& builder) { AddNode(builder, options); },
                       "past_gate_buffer dimension 2 must equal 2 * head_size");
}

TEST(PackedSparseAttentionIndexerShapeInferenceTest, RejectsUnknownPolicyMode) {
  GraphOptions options;
  options.policy_mode = "qsa_v2";
  ExpectResolveFailure([&options](ModelTestBuilder& builder) { AddNode(builder, options); },
                       "policy_mode must be 'qsa' or 'csa'");
}

TEST(PackedSparseAttentionIndexerShapeInferenceTest, RejectsQsaWithCsaAttribute) {
  GraphOptions options;
  options.add_index_topk = true;
  ExpectResolveFailure([&options](ModelTestBuilder& builder) { AddNode(builder, options); },
                       "index_topk and head_weight_scale must not be set");
}

TEST(PackedSparseAttentionIndexerShapeInferenceTest, RejectsZeroNumHeads) {
  GraphOptions options;
  options.num_heads = 0;
  ExpectResolveFailure([&options](ModelTestBuilder& builder) { AddNode(builder, options); },
                       "num_heads must be > 0");
}

TEST(PackedSparseAttentionIndexerShapeInferenceTest, RejectsGenericBufferCapacityOverflow) {
  GraphOptions options;
  options.policy_mode = psai::kPolicyModeCsa;
  options.compress_ratio = static_cast<int64_t>(std::numeric_limits<int>::max()) / 2 + 2;
  ExpectResolveFailure([&options](ModelTestBuilder& builder) { AddNode(builder, options); },
                       "generic buffer capacity no greater than INT_MAX");
}

TEST(PackedSparseAttentionIndexerShapeInferenceTest, RejectsQsaTokenBudgetNotDivisibleByCompressRatio) {
  GraphOptions options;
  options.token_budget = 5;
  ExpectResolveFailure([&options](ModelTestBuilder& builder) { AddNode(builder, options); },
                       "requires token_budget > 0, divisible by compress_ratio");
}

TEST(PackedSparseAttentionIndexerShapeInferenceTest, RejectsQsaWithCsaInput) {
  GraphOptions options;
  options.add_csa_inputs = true;
  ExpectResolveFailure([&options](ModelTestBuilder& builder) { AddNode(builder, options); },
                       "must be omitted when policy_mode is 'qsa'");
}

TEST(PackedSparseAttentionIndexerShapeInferenceTest, RejectsQsaWithPositionIdsAndCsaInputsMismatch) {
  // position_ids alone is allowed for qsa (optional); only the csa-only slots must be omitted.
  GraphOptions options;
  options.add_position_ids = true;
  std::unique_ptr<Model> model;
  ASSERT_STATUS_OK(BuildAndResolve([&options](ModelTestBuilder& builder) { AddNode(builder, options); }, model));
}

TEST(PackedSparseAttentionIndexerShapeInferenceTest, RejectsCsaMissingPositionIds) {
  GraphOptions options;
  options.policy_mode = psai::kPolicyModeCsa;
  options.add_position_ids = false;
  // Force the csa node to omit position_ids by rebuilding without it.
  std::unique_ptr<Model> model;
  const Status status = BuildAndResolve(
      [&options](ModelTestBuilder& builder) {
        GraphOptions local = options;
        NodeArg& empty = builder.graph_.GetOrCreateNodeArg("", nullptr);
        const int64_t width = 2 * local.head_size;
        const int64_t buffer_capacity = BufferCapacity(local.compress_ratio);
        std::vector<NodeArg*> inputs{
            builder.MakeInput<float>(std::vector<int64_t>{local.total_tokens, local.num_heads, local.head_size}),
            builder.MakeInput<float>(std::vector<int64_t>{local.total_tokens, width}),
            builder.MakeInput<float>(std::vector<int64_t>{local.head_size}),
            builder.MakeInput<float>(std::vector<int64_t>{64, local.rotary_width}),
            builder.MakeInput<float>(std::vector<int64_t>{64, local.rotary_width}),
            builder.MakeInput<int32_t>(std::vector<int64_t>{local.batch_size + 1}),
            builder.MakeInput<int32_t>(std::vector<int64_t>{local.batch_size}),
            builder.MakeInput<float>(std::vector<int64_t>{local.total_tokens, width}),
            builder.MakeInput<float>(std::vector<int64_t>{local.compress_ratio, width}),
            builder.MakeInput<float>(std::vector<int64_t>{local.total_tokens, local.num_heads}),
            &empty,  // position_ids omitted: invalid for csa
            builder.MakeInput<float>(
                std::vector<int64_t>{local.batch_size, local.state_capacity, local.head_size}),
            builder.MakeInput<float>(std::vector<int64_t>{local.batch_size, buffer_capacity, width}),
            builder.MakeInput<float>(std::vector<int64_t>{local.batch_size, buffer_capacity, width}),
            builder.MakeInput<int32_t>(std::vector<int64_t>{local.batch_size, 2}),
        };
        std::vector<NodeArg*> outputs;
        for (int i = 0; i < psai::kFixedOutputCount; ++i) {
          outputs.push_back(builder.MakeOutput());
        }
        Node& node = builder.AddNode("PackedSparseAttentionIndexer", inputs, outputs, kMSDomain);
        node.AddAttribute("policy_mode", local.policy_mode);
        node.AddAttribute("compress_ratio", local.compress_ratio);
        node.AddAttribute("state_capacity", local.state_capacity);
        node.AddAttribute("index_topk", static_cast<int64_t>(3));
      },
      model);
  ASSERT_FALSE(status.IsOK());
  EXPECT_NE(status.ErrorMessage().find("position_ids"), std::string::npos) << status.ErrorMessage();
}

TEST(PackedSparseAttentionIndexerShapeInferenceTest, RejectsWrongOutputCount) {
  GraphOptions options;
  options.output_count = 4;
  ExpectResolveFailure([&options](ModelTestBuilder& builder) { AddNode(builder, options); },
                       "output size 4 not in range [min=6, max=6]");
}

#endif  // ORT_NO_EXCEPTIONS

// ---------------------------------------------------------------------------------------------
// Numeric behaviour (CUDA / WebGPU only)
// ---------------------------------------------------------------------------------------------

namespace {

enum class ProviderKind {
  Cuda,
  WebGpu,
};

std::unique_ptr<IExecutionProvider> CreateProvider(ProviderKind provider_kind) {
  if (provider_kind == ProviderKind::Cuda) {
    return DefaultCudaExecutionProvider();
  }
#ifdef USE_WEBGPU
  return DefaultWebGpuExecutionProvider();
#else
  return nullptr;
#endif
}

void RunOnProvider(OpTester& test, std::unique_ptr<IExecutionProvider> provider) {
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(std::move(provider));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

std::vector<float> MakeWave(size_t count, float phase, float step) {
  std::vector<float> values(count);
  for (size_t i = 0; i < count; ++i) {
    values[i] = std::sin(phase + step * static_cast<float>(i));
  }
  return values;
}

template <typename T>
std::vector<T> ToElementType(const std::vector<float>& data) {
  if constexpr (std::is_same_v<T, MLFloat16>) {
    return ToFloat16(data);
  } else if constexpr (std::is_same_v<T, BFloat16>) {
    return ToBFloat16(data);
  } else {
    return data;
  }
}

// Rounds through the tested element type so the reference consumes exactly the kernel's inputs.
template <typename T>
std::vector<float> RoundTrip(const std::vector<float>& data) {
  if constexpr (std::is_same_v<T, float>) {
    return data;
  } else {
    std::vector<T> converted = ToElementType<T>(data);
    std::vector<float> result(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
      result[i] = converted[i].ToFloat();
    }
    return result;
  }
}

template <typename T>
std::vector<float> CopyTensorToFloat(const Tensor& tensor) {
  const auto values = tensor.DataAsSpan<T>();
  std::vector<float> result(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    if constexpr (std::is_same_v<T, float>) {
      result[i] = values[i];
    } else {
      result[i] = values[i].ToFloat();
    }
  }
  return result;
}

// Split-half rotary over the leading rotary_width channels.
std::vector<float> LeadingRope(const std::vector<float>& value, int rotary_width, const float* cos_row,
                               const float* sin_row) {
  const int head_size = static_cast<int>(value.size());
  const int half = rotary_width / 2;
  std::vector<float> result(value);
  for (int d = 0; d < rotary_width && d < head_size; ++d) {
    const float paired = (d < half) ? -value[static_cast<size_t>(d + half)] : value[static_cast<size_t>(d - half)];
    result[static_cast<size_t>(d)] = value[static_cast<size_t>(d)] * cos_row[d] + paired * sin_row[d];
  }
  return result;
}

// Interleaved rotary over the trailing 2 * rotary_width channels.
std::vector<float> TrailingRope(const std::vector<float>& value, int rotary_width, const float* cos_row,
                                const float* sin_row) {
  const int head_size = static_cast<int>(value.size());
  const int base = head_size - 2 * rotary_width;
  std::vector<float> result(value);
  for (int d = base; d < head_size; ++d) {
    const int offset = d - base;
    const float paired = ((offset & 1) == 0) ? -value[static_cast<size_t>(d + 1)] : value[static_cast<size_t>(d - 1)];
    result[static_cast<size_t>(d)] =
        value[static_cast<size_t>(d)] * cos_row[offset >> 1] + paired * sin_row[offset >> 1];
  }
  return result;
}

std::vector<float> RmsNormalize(const std::vector<float>& value, const std::vector<float>& weight, float epsilon) {
  float sum_squares = 0.0f;
  for (float element : value) {
    sum_squares += element * element;
  }
  const float inverse_rms = 1.0f / std::sqrt(sum_squares / static_cast<float>(value.size()) + epsilon);
  std::vector<float> result(value.size());
  for (size_t d = 0; d < value.size(); ++d) {
    result[d] = value[d] * inverse_rms * weight[d];
  }
  return result;
}

// Order used by both selection kernels: score descending, then entry index ascending.
std::vector<int> RankByScore(const std::vector<float>& scores, int count) {
  std::vector<int> order(static_cast<size_t>(count));
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&scores](int left, int right) {
    if (scores[static_cast<size_t>(left)] != scores[static_cast<size_t>(right)]) {
      return scores[static_cast<size_t>(left)] > scores[static_cast<size_t>(right)];
    }
    return left < right;
  });
  return order;
}

int CausalThreshold(int position, int compress_ratio) {
  if (position < 0) return 0;
  return position / compress_ratio + (position % compress_ratio == compress_ratio - 1 ? 1 : 0);
}

struct QsaPackedProblem {
  int batch_size = 2;
  std::vector<int32_t> cumulative_sequence_lengths;  // [batch + 1]
  std::vector<int32_t> past_sequence_lengths;        // [batch]
  std::vector<int32_t> past_state_lengths;           // [batch, 2]
  int head_size = 4;
  int num_heads = 2;
  int rotary_width = 4;
  int compress_ratio = 2;
  int token_budget = 4;
  int state_capacity = 4;
  int max_position = 64;
  float epsilon = 1.0e-6f;
  std::optional<float> scale;

  std::vector<float> query;
  std::vector<float> key;
  std::vector<float> key_norm_weight;
  std::vector<float> cos_cache;  // shared: [max_position, rotary_width]
  std::vector<float> sin_cache;
  std::vector<float> past_key_state;  // [batch, state_capacity, head_size]
  std::vector<float> past_kv_buffer;  // [batch, buffer_capacity, head_size]

  int TotalTokens() const { return cumulative_sequence_lengths.back(); }
  int BufferCapacity() const { return 2 * compress_ratio - 1; }
  int Capacity() const { return token_budget + compress_ratio - 1; }
};

struct QsaPackedResult {
  std::vector<int32_t> selected_indices;
  std::vector<int32_t> selected_counts;
  std::vector<float> present_key_state;
  std::vector<float> present_kv_buffer;
  std::vector<int32_t> present_state_lengths;
};

void QsaPackedReference(const QsaPackedProblem& p, QsaPackedResult& out) {
  const int total_tokens = p.TotalTokens();
  const int head_size = p.head_size;
  const int block_topk = p.token_budget / p.compress_ratio;
  const float scale = p.scale.value_or(1.0f / std::sqrt(static_cast<float>(head_size)));
  const int buffer_capacity = p.BufferCapacity();
  const int capacity = p.Capacity();

  out.present_key_state = p.past_key_state;
  out.present_kv_buffer = p.past_kv_buffer;
  out.present_state_lengths = p.past_state_lengths;

  std::vector<int> key_len_after(static_cast<size_t>(p.batch_size));
  std::vector<bool> overflowed(static_cast<size_t>(p.batch_size), false);
  for (int b = 0; b < p.batch_size; ++b) {
    const int req_start = p.cumulative_sequence_lengths[static_cast<size_t>(b)];
    const int req_end = p.cumulative_sequence_lengths[static_cast<size_t>(b) + 1];
    const int req_len = std::max(req_end - req_start, 0);
    const int old_key_len =
        std::clamp(p.past_state_lengths[static_cast<size_t>(b) * 2 + 0], 0, p.state_capacity);
    const int old_buf_len =
        std::clamp(p.past_state_lengths[static_cast<size_t>(b) * 2 + 1], 0, p.compress_ratio - 1);
    const int pending = old_buf_len + req_len;
    const int full_new_block_count = pending / p.compress_ratio;
    overflowed[static_cast<size_t>(b)] = full_new_block_count > std::max(p.state_capacity - old_key_len, 0);
    if (overflowed[static_cast<size_t>(b)]) {
      key_len_after[static_cast<size_t>(b)] = old_key_len;
      continue;
    }
    const int new_block_count = full_new_block_count;
    const int new_buf_len = pending % p.compress_ratio;

    auto raw_value = [&](int virtual_pos, int d) -> float {
      return virtual_pos < old_buf_len
                 ? p.past_kv_buffer[(static_cast<size_t>(b) * buffer_capacity + virtual_pos) * head_size + d]
                 : p.key[(static_cast<size_t>(req_start) + (virtual_pos - old_buf_len)) * head_size + d];
    };

    for (int k = 0; k < new_block_count; ++k) {
      std::vector<float> pooled(static_cast<size_t>(head_size), 0.0f);
      for (int t = 0; t < p.compress_ratio; ++t) {
        for (int d = 0; d < head_size; ++d) {
          pooled[static_cast<size_t>(d)] += raw_value(k * p.compress_ratio + t, d);
        }
      }
      for (float& v : pooled) v /= static_cast<float>(p.compress_ratio);
      pooled = RmsNormalize(pooled, p.key_norm_weight, p.epsilon);
      const int entry = old_key_len + k;
      const int position = std::min(entry * p.compress_ratio, p.max_position - 1);
      pooled = LeadingRope(pooled, p.rotary_width, p.cos_cache.data() + position * p.rotary_width,
                           p.sin_cache.data() + position * p.rotary_width);
      for (int d = 0; d < head_size; ++d) {
        out.present_key_state[(static_cast<size_t>(b) * p.state_capacity + entry) * head_size + d] =
            pooled[static_cast<size_t>(d)];
      }
    }
    for (int t = 0; t < new_buf_len; ++t) {
      for (int d = 0; d < head_size; ++d) {
        out.present_kv_buffer[(static_cast<size_t>(b) * buffer_capacity + t) * head_size + d] =
            raw_value(new_block_count * p.compress_ratio + t, d);
      }
    }
    out.present_state_lengths[static_cast<size_t>(b) * 2 + 0] = old_key_len + new_block_count;
    out.present_state_lengths[static_cast<size_t>(b) * 2 + 1] = new_buf_len;
    key_len_after[static_cast<size_t>(b)] = old_key_len + new_block_count;
  }

  out.selected_indices.assign(static_cast<size_t>(total_tokens) * capacity, -1);
  out.selected_counts.assign(static_cast<size_t>(total_tokens), 0);
  for (int b = 0; b < p.batch_size; ++b) {
    if (overflowed[static_cast<size_t>(b)]) {
      continue;
    }
    const int req_start = p.cumulative_sequence_lengths[static_cast<size_t>(b)];
    const int req_end = p.cumulative_sequence_lengths[static_cast<size_t>(b) + 1];
    for (int token = req_start; token < req_end; ++token) {
      const int position = p.past_sequence_lengths[static_cast<size_t>(b)] + (token - req_start);
      const int causal_count = CausalThreshold(position, p.compress_ratio);
      const int visible_block_count = std::min(key_len_after[static_cast<size_t>(b)], causal_count);
      const int selected = std::min(block_topk, visible_block_count);

      std::vector<std::vector<float>> rotated_query(static_cast<size_t>(p.num_heads));
      for (int h = 0; h < p.num_heads; ++h) {
        const size_t base = (static_cast<size_t>(token) * p.num_heads + h) * head_size;
        std::vector<float> head(p.query.begin() + base, p.query.begin() + base + head_size);
        const int clamped_position = std::min(std::max(position, 0), p.max_position - 1);
        rotated_query[static_cast<size_t>(h)] =
            LeadingRope(head, p.rotary_width, p.cos_cache.data() + clamped_position * p.rotary_width,
                        p.sin_cache.data() + clamped_position * p.rotary_width);
      }

      std::vector<float> scores(static_cast<size_t>(visible_block_count), 0.0f);
      for (int j = 0; j < visible_block_count; ++j) {
        float score = 0.0f;
        for (int h = 0; h < p.num_heads; ++h) {
          float dot = 0.0f;
          for (int d = 0; d < head_size; ++d) {
            dot += rotated_query[static_cast<size_t>(h)][static_cast<size_t>(d)] *
                   out.present_key_state[(static_cast<size_t>(b) * p.state_capacity + j) * head_size + d];
          }
          score += std::max(dot, 0.0f);
        }
        scores[static_cast<size_t>(j)] = score * scale;
      }
      const std::vector<int> order = RankByScore(scores, visible_block_count);
      const int emitted_blocks = std::min(selected, visible_block_count);
      int32_t* out_row = out.selected_indices.data() + static_cast<size_t>(token) * capacity;
      for (int rank = 0; rank < emitted_blocks; ++rank) {
        for (int t = 0; t < p.compress_ratio; ++t) {
          out_row[rank * p.compress_ratio + t] = order[static_cast<size_t>(rank)] * p.compress_ratio + t;
        }
      }
      const int block_start = visible_block_count * p.compress_ratio;
      const int natural_tail = position >= block_start ? position - block_start + 1 : 0;
      const int remaining_capacity = capacity - emitted_blocks * p.compress_ratio;
      const int tail_count = std::clamp(natural_tail, 0, remaining_capacity);
      for (int t = 0; t < tail_count; ++t) {
        out_row[emitted_blocks * p.compress_ratio + t] = block_start + t;
      }
      out.selected_counts[static_cast<size_t>(token)] = emitted_blocks * p.compress_ratio + tail_count;
    }
  }
}

QsaPackedProblem MakeQsaPackedProblem(QsaPackedProblem problem = {}) {
  if (problem.cumulative_sequence_lengths.empty()) {
    // Default: 2 requests, 2 and 3 tokens respectively (unequal lengths).
    problem.cumulative_sequence_lengths = {0, 2, 5};
    problem.past_sequence_lengths = {3, 0};
  }
  const int total_tokens = problem.TotalTokens();
  const int buffer_capacity = problem.BufferCapacity();
  if (problem.past_state_lengths.empty()) {
    problem.past_state_lengths.assign(static_cast<size_t>(problem.batch_size) * 2, 0);
    for (int b = 0; b < problem.batch_size; ++b) {
      problem.past_state_lengths[static_cast<size_t>(b) * 2 + 0] = problem.past_sequence_lengths[static_cast<size_t>(b)] / problem.compress_ratio;
      problem.past_state_lengths[static_cast<size_t>(b) * 2 + 1] = problem.past_sequence_lengths[static_cast<size_t>(b)] % problem.compress_ratio;
    }
  }

  problem.query =
      MakeWave(static_cast<size_t>(total_tokens) * problem.num_heads * problem.head_size, 0.35f, 0.41f);
  problem.key = MakeWave(static_cast<size_t>(total_tokens) * problem.head_size, 1.10f, 0.29f);
  problem.key_norm_weight = MakeWave(static_cast<size_t>(problem.head_size), 0.70f, 0.17f);
  problem.cos_cache = MakeWave(static_cast<size_t>(problem.max_position) * problem.rotary_width, 0.20f, 0.13f);
  problem.sin_cache = MakeWave(static_cast<size_t>(problem.max_position) * problem.rotary_width, 0.90f, 0.19f);
  problem.past_key_state =
      MakeWave(static_cast<size_t>(problem.batch_size) * problem.state_capacity * problem.head_size, 0.05f, 0.23f);
  problem.past_kv_buffer =
      MakeWave(static_cast<size_t>(problem.batch_size) * buffer_capacity * problem.head_size, 0.60f, 0.09f);
  return problem;
}

template <typename T>
void RunQsaPackedTest(float tolerance, QsaPackedProblem problem = MakeQsaPackedProblem(),
                      ProviderKind provider_kind = ProviderKind::Cuda, QsaPackedResult* actual = nullptr) {
  auto provider = CreateProvider(provider_kind);
  if (provider == nullptr) {
    GTEST_SKIP() << (provider_kind == ProviderKind::Cuda ? "CUDA" : "WebGPU")
                 << " execution provider is not available";
  }

  problem.query = RoundTrip<T>(problem.query);
  problem.key = RoundTrip<T>(problem.key);
  problem.key_norm_weight = RoundTrip<T>(problem.key_norm_weight);
  problem.cos_cache = RoundTrip<T>(problem.cos_cache);
  problem.sin_cache = RoundTrip<T>(problem.sin_cache);
  problem.past_key_state = RoundTrip<T>(problem.past_key_state);
  problem.past_kv_buffer = RoundTrip<T>(problem.past_kv_buffer);

  QsaPackedResult expected;
  QsaPackedReference(problem, expected);

  const int64_t total_tokens = problem.TotalTokens();
  const int64_t batch_size = problem.batch_size;
  const int64_t head_size = problem.head_size;
  const int64_t buffer_capacity = problem.BufferCapacity();

  OpTester test("PackedSparseAttentionIndexer", 1, onnxruntime::kMSDomain);
  test.AddAttribute("policy_mode", std::string(psai::kPolicyModeQsa));
  test.AddAttribute("compress_ratio", static_cast<int64_t>(problem.compress_ratio));
  test.AddAttribute("state_capacity", static_cast<int64_t>(problem.state_capacity));
  test.AddAttribute("token_budget", static_cast<int64_t>(problem.token_budget));
  if (problem.scale.has_value()) {
    test.AddAttribute("scale", *problem.scale);
  }
  test.AddInput<T>("query", {total_tokens, problem.num_heads, head_size}, ToElementType<T>(problem.query));
  test.AddInput<T>("key", {total_tokens, head_size}, ToElementType<T>(problem.key));
  test.AddInput<T>("key_norm_weight", {head_size}, ToElementType<T>(problem.key_norm_weight));
  test.AddInput<T>("cos_cache", {problem.max_position, problem.rotary_width}, ToElementType<T>(problem.cos_cache));
  test.AddInput<T>("sin_cache", {problem.max_position, problem.rotary_width}, ToElementType<T>(problem.sin_cache));
  test.AddInput<int32_t>("cumulative_sequence_lengths", {batch_size + 1}, problem.cumulative_sequence_lengths);
  test.AddInput<int32_t>("past_sequence_lengths", {batch_size}, problem.past_sequence_lengths);
  test.AddOptionalInputEdge<T>();        // gate
  test.AddOptionalInputEdge<T>();        // position_bias
  test.AddOptionalInputEdge<T>();        // head_weights
  test.AddOptionalInputEdge<int64_t>();  // position_ids
  test.AddInput<T>("past_key_state", {batch_size, problem.state_capacity, head_size},
                   ToElementType<T>(problem.past_key_state));
  test.AddInput<T>("past_kv_buffer", {batch_size, buffer_capacity, head_size},
                   ToElementType<T>(problem.past_kv_buffer));
  test.AddOptionalInputEdge<T>();  // past_gate_buffer
  test.AddInput<int32_t>("past_state_lengths", {batch_size, 2}, problem.past_state_lengths);

  test.AddOutput<int32_t>("selected_indices", {total_tokens, problem.Capacity()}, expected.selected_indices);
  test.AddOutput<int32_t>("selected_counts", {total_tokens}, expected.selected_counts);
  test.AddOutput<T>("present_key_state", {batch_size, problem.state_capacity, head_size},
                    ToElementType<T>(expected.present_key_state), false, 0.0f, tolerance);
  test.AddOutput<T>("present_kv_buffer", {batch_size, buffer_capacity, head_size},
                    ToElementType<T>(expected.present_kv_buffer), false, 0.0f, tolerance);
  test.AddOptionalOutputEdge<T>();  // present_gate_buffer
  test.AddOutput<int32_t>("present_state_lengths", {batch_size, 2}, expected.present_state_lengths);
  RunOnProvider(test, std::move(provider));
  if (actual != nullptr) {
    const auto& fetches = test.GetFetches();
    actual->present_key_state = CopyTensorToFloat<T>(fetches[2].Get<Tensor>());
    actual->present_kv_buffer = CopyTensorToFloat<T>(fetches[3].Get<Tensor>());
    actual->present_state_lengths.assign(fetches[4].Get<Tensor>().DataAsSpan<int32_t>().begin(),
                                         fetches[4].Get<Tensor>().DataAsSpan<int32_t>().end());
  }
}

}  // namespace

TEST(PackedSparseAttentionIndexerTest, QsaFloat) { RunQsaPackedTest<float>(1.0e-5f); }

TEST(PackedSparseAttentionIndexerTest, QsaFloat16) { RunQsaPackedTest<MLFloat16>(2.0e-3f); }

TEST(PackedSparseAttentionIndexerTest, QsaBFloat16) { RunQsaPackedTest<BFloat16>(2.0e-2f); }

TEST(PackedSparseAttentionIndexerTest, QsaPrefillThenDecodeIndependentState) {
  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA execution provider is not available";
  }

  QsaPackedProblem prefill;
  prefill.cumulative_sequence_lengths = {0, 3, 5};
  prefill.past_sequence_lengths = {0, 0};
  QsaPackedResult prefill_outputs;
  RunQsaPackedTest<float>(1.0e-5f, MakeQsaPackedProblem(std::move(prefill)), ProviderKind::Cuda,
                          &prefill_outputs);

  QsaPackedProblem decode;
  decode.cumulative_sequence_lengths = {0, 1, 3};
  decode.past_sequence_lengths = {3, 2};
  decode = MakeQsaPackedProblem(std::move(decode));
  decode.past_key_state = std::move(prefill_outputs.present_key_state);
  decode.past_kv_buffer = std::move(prefill_outputs.present_kv_buffer);
  decode.past_state_lengths = std::move(prefill_outputs.present_state_lengths);
  RunQsaPackedTest<float>(1.0e-5f, std::move(decode));
}

// A zero-token request row (repeated cumulative offset) must not affect the other request and
// must not read out of bounds.
TEST(PackedSparseAttentionIndexerTest, QsaZeroTokenRequestRow) {
  QsaPackedProblem problem;
  problem.batch_size = 3;
  problem.cumulative_sequence_lengths = {0, 2, 2, 5};
  problem.past_sequence_lengths = {3, 7, 0};
  RunQsaPackedTest<float>(1.0e-5f, MakeQsaPackedProblem(std::move(problem)));
}

// state_capacity smaller than what the incoming tokens would naturally produce: the update must
// reject the request's step without partially changing its state.
TEST(PackedSparseAttentionIndexerTest, QsaStateCapacityOverflowIsSafe) {
  QsaPackedProblem problem;
  problem.cumulative_sequence_lengths = {0, 6};
  problem.past_sequence_lengths = {4};
  problem.batch_size = 1;
  problem.state_capacity = 2;  // only room for 2 - 2(already used) = 0 new complete blocks
  problem.past_state_lengths = {2, 0};
  RunQsaPackedTest<float>(1.0e-5f, MakeQsaPackedProblem(std::move(problem)));
}

#ifdef USE_WEBGPU
TEST(PackedSparseAttentionIndexerWebGpuTest, QsaFloat) {
  RunQsaPackedTest<float>(1.0e-5f, MakeQsaPackedProblem(), ProviderKind::WebGpu);
}

TEST(PackedSparseAttentionIndexerWebGpuTest, QsaFloat16) {
  RunQsaPackedTest<MLFloat16>(2.0e-3f, MakeQsaPackedProblem(), ProviderKind::WebGpu);
}

TEST(PackedSparseAttentionIndexerWebGpuTest, QsaStateCapacityOverflowIsRejected) {
  QsaPackedProblem problem;
  problem.batch_size = 1;
  problem.cumulative_sequence_lengths = {0, 6};
  problem.past_sequence_lengths = {4};
  problem.state_capacity = 2;
  problem.past_state_lengths = {2, 0};
  RunQsaPackedTest<float>(1.0e-5f, MakeQsaPackedProblem(std::move(problem)), ProviderKind::WebGpu);
}
#endif

namespace {

struct CsaPackedProblem {
  int batch_size = 2;
  std::vector<int32_t> cumulative_sequence_lengths;
  std::vector<int32_t> past_sequence_lengths;  // unused by csa math, kept for symmetry with qsa
  std::vector<int32_t> past_state_lengths;     // [batch, 2]: compressed count, buffer length
  int head_size = 4;
  int num_heads = 2;
  int rotary_width = 2;
  int compress_ratio = 2;
  int index_topk = 3;
  int state_capacity = 4;
  int max_position = 64;
  float epsilon = 1.0e-6f;
  std::optional<float> scale;
  std::optional<float> head_weight_scale;

  std::vector<float> query;
  std::vector<float> key;
  std::vector<float> key_norm_weight;
  std::vector<float> cos_cache;
  std::vector<float> sin_cache;
  std::vector<float> gate;
  std::vector<float> position_bias;
  std::vector<float> head_weights;
  std::vector<int64_t> position_ids;
  std::vector<float> past_key_state;
  std::vector<float> past_kv_buffer;
  std::vector<float> past_gate_buffer;

  int Width() const { return 2 * head_size; }
  int TotalTokens() const { return cumulative_sequence_lengths.back(); }
  int BufferCapacity() const { return 2 * compress_ratio - 1; }
};

struct CsaPackedResult {
  std::vector<int32_t> selected_indices;
  std::vector<int32_t> selected_counts;
  std::vector<float> present_key_state;
  std::vector<float> present_kv_buffer;
  std::vector<float> present_gate_buffer;
  std::vector<int32_t> present_state_lengths;
};

void CsaPackedReference(const CsaPackedProblem& p, CsaPackedResult& out) {
  const int total_tokens = p.TotalTokens();
  const int head_size = p.head_size;
  const int width = p.Width();
  const int buffer_capacity = p.BufferCapacity();
  const int capacity = p.index_topk;
  const float scale = p.scale.value_or(1.0f / std::sqrt(static_cast<float>(head_size)));
  const float head_weight_scale = p.head_weight_scale.value_or(1.0f / std::sqrt(static_cast<float>(p.num_heads)));

  out.present_key_state = p.past_key_state;
  out.present_kv_buffer = p.past_kv_buffer;
  out.present_gate_buffer = p.past_gate_buffer;
  out.present_state_lengths = p.past_state_lengths;
  std::vector<int> key_len_after(static_cast<size_t>(p.batch_size));
  std::vector<bool> overflowed(static_cast<size_t>(p.batch_size), false);

  for (int b = 0; b < p.batch_size; ++b) {
    const int req_start = p.cumulative_sequence_lengths[static_cast<size_t>(b)];
    const int req_end = p.cumulative_sequence_lengths[static_cast<size_t>(b) + 1];
    const int req_len = std::max(req_end - req_start, 0);
    const int old_key_len =
        std::clamp(p.past_state_lengths[static_cast<size_t>(b) * 2 + 0], 0, p.state_capacity);
    const int old_buf_len =
        std::clamp(p.past_state_lengths[static_cast<size_t>(b) * 2 + 1], 0, buffer_capacity);
    const int overlap_length = old_buf_len >= p.compress_ratio ? p.compress_ratio : 0;
    const int leftover_length = old_buf_len - overlap_length;
    const int pending = leftover_length + req_len;
    const int full_new_window_count = pending / p.compress_ratio;
    overflowed[static_cast<size_t>(b)] = full_new_window_count > std::max(p.state_capacity - old_key_len, 0);
    if (overflowed[static_cast<size_t>(b)]) {
      key_len_after[static_cast<size_t>(b)] = old_key_len;
      continue;
    }
    const int new_window_count = full_new_window_count;
    int present_buffer_length = 0;
    int present_buffer_start = 0;
    if (new_window_count > 0) {
      present_buffer_length = p.compress_ratio + pending % p.compress_ratio;
      present_buffer_start = overlap_length + (new_window_count - 1) * p.compress_ratio;
    } else {
      present_buffer_length = old_buf_len + req_len;
      present_buffer_start = 0;
    }
    present_buffer_length = std::min(present_buffer_length, buffer_capacity);

    auto extended_key = [&](int virtual_pos, int channel) -> float {
      return virtual_pos < old_buf_len
                 ? p.past_kv_buffer[(static_cast<size_t>(b) * buffer_capacity + virtual_pos) * width + channel]
                 : p.key[(static_cast<size_t>(req_start) + (virtual_pos - old_buf_len)) * width + channel];
    };
    auto extended_gate = [&](int virtual_pos, int channel) -> float {
      return virtual_pos < old_buf_len
                 ? p.past_gate_buffer[(static_cast<size_t>(b) * buffer_capacity + virtual_pos) * width + channel]
                 : p.gate[(static_cast<size_t>(req_start) + (virtual_pos - old_buf_len)) * width + channel];
    };

    for (int k = 0; k < new_window_count; ++k) {
      const bool has_previous = k >= 1 || overlap_length >= p.compress_ratio;
      const int previous_base = overlap_length + (k - 1) * p.compress_ratio;
      const int current_base = overlap_length + k * p.compress_ratio;
      std::vector<float> pooled(static_cast<size_t>(head_size), 0.0f);
      for (int d = 0; d < head_size; ++d) {
        float max_gate = -std::numeric_limits<float>::infinity();
        if (has_previous) {
          for (int slot = 0; slot < p.compress_ratio; ++slot) {
            max_gate = std::max(max_gate, extended_gate(previous_base + slot, d) + p.position_bias[static_cast<size_t>(slot) * width + d]);
          }
        }
        for (int slot = 0; slot < p.compress_ratio; ++slot) {
          max_gate = std::max(max_gate, extended_gate(current_base + slot, head_size + d) +
                                            p.position_bias[static_cast<size_t>(slot) * width + head_size + d]);
        }
        float denom = 0.0f, acc = 0.0f;
        if (has_previous) {
          for (int slot = 0; slot < p.compress_ratio; ++slot) {
            const float logit = extended_gate(previous_base + slot, d) + p.position_bias[static_cast<size_t>(slot) * width + d];
            const float w = std::exp(logit - max_gate);
            denom += w;
            acc += w * extended_key(previous_base + slot, d);
          }
        }
        for (int slot = 0; slot < p.compress_ratio; ++slot) {
          const float logit = extended_gate(current_base + slot, head_size + d) +
                              p.position_bias[static_cast<size_t>(slot) * width + head_size + d];
          const float w = std::exp(logit - max_gate);
          denom += w;
          acc += w * extended_key(current_base + slot, head_size + d);
        }
        pooled[static_cast<size_t>(d)] = denom > 0.0f ? acc / denom : 0.0f;
      }
      pooled = RmsNormalize(pooled, p.key_norm_weight, p.epsilon);
      const int entry = old_key_len + k;
      const int position = std::min(entry * p.compress_ratio, p.max_position - 1);
      pooled = TrailingRope(pooled, p.rotary_width, p.cos_cache.data() + position * p.rotary_width,
                            p.sin_cache.data() + position * p.rotary_width);
      for (int d = 0; d < head_size; ++d) {
        out.present_key_state[(static_cast<size_t>(b) * p.state_capacity + entry) * head_size + d] =
            pooled[static_cast<size_t>(d)];
      }
    }
    for (int t = 0; t < present_buffer_length; ++t) {
      const int virtual_pos = present_buffer_start + t;
      for (int c = 0; c < width; ++c) {
        out.present_kv_buffer[(static_cast<size_t>(b) * buffer_capacity + t) * width + c] = extended_key(virtual_pos, c);
        out.present_gate_buffer[(static_cast<size_t>(b) * buffer_capacity + t) * width + c] = extended_gate(virtual_pos, c);
      }
    }
    out.present_state_lengths[static_cast<size_t>(b) * 2 + 0] = old_key_len + new_window_count;
    out.present_state_lengths[static_cast<size_t>(b) * 2 + 1] = present_buffer_length;
    key_len_after[static_cast<size_t>(b)] = old_key_len + new_window_count;
  }

  out.selected_indices.assign(static_cast<size_t>(total_tokens) * capacity, -1);
  out.selected_counts.assign(static_cast<size_t>(total_tokens), 0);
  for (int b = 0; b < p.batch_size; ++b) {
    if (overflowed[static_cast<size_t>(b)]) {
      continue;
    }
    const int req_start = p.cumulative_sequence_lengths[static_cast<size_t>(b)];
    const int req_end = p.cumulative_sequence_lengths[static_cast<size_t>(b) + 1];
    for (int token = req_start; token < req_end; ++token) {
      const int position = static_cast<int>(p.position_ids[static_cast<size_t>(token)]);
      const int threshold = std::min(CausalThreshold(position, p.compress_ratio), key_len_after[static_cast<size_t>(b)]);
      const int selected = std::min(p.index_topk, threshold);

      std::vector<std::vector<float>> rotated_query(static_cast<size_t>(p.num_heads));
      for (int h = 0; h < p.num_heads; ++h) {
        const size_t base = (static_cast<size_t>(token) * p.num_heads + h) * head_size;
        std::vector<float> head(p.query.begin() + base, p.query.begin() + base + head_size);
        const int clamped_position = std::min(std::max(position, 0), p.max_position - 1);
        rotated_query[static_cast<size_t>(h)] =
            TrailingRope(head, p.rotary_width, p.cos_cache.data() + clamped_position * p.rotary_width,
                         p.sin_cache.data() + clamped_position * p.rotary_width);
      }

      std::vector<float> scores(static_cast<size_t>(threshold), 0.0f);
      for (int e = 0; e < threshold; ++e) {
        float score = 0.0f;
        for (int h = 0; h < p.num_heads; ++h) {
          float dot = 0.0f;
          for (int d = 0; d < head_size; ++d) {
            dot += rotated_query[static_cast<size_t>(h)][static_cast<size_t>(d)] *
                   out.present_key_state[(static_cast<size_t>(b) * p.state_capacity + e) * head_size + d];
          }
          score += std::max(dot, 0.0f) * p.head_weights[static_cast<size_t>(token) * p.num_heads + h];
        }
        scores[static_cast<size_t>(e)] = score * scale * head_weight_scale;
      }
      const std::vector<int> order = RankByScore(scores, threshold);
      int32_t* out_row = out.selected_indices.data() + static_cast<size_t>(token) * capacity;
      for (int rank = 0; rank < selected; ++rank) {
        out_row[rank] = order[static_cast<size_t>(rank)];
      }
      out.selected_counts[static_cast<size_t>(token)] = selected;
    }
  }
}

CsaPackedProblem MakeCsaPackedProblem(CsaPackedProblem problem = {}) {
  if (problem.cumulative_sequence_lengths.empty()) {
    problem.cumulative_sequence_lengths = {0, 2, 5};
  }
  if (problem.past_state_lengths.empty()) {
    problem.past_state_lengths.assign(static_cast<size_t>(problem.batch_size) * 2, 0);
  }
  if (problem.past_sequence_lengths.empty()) {
    problem.past_sequence_lengths.assign(static_cast<size_t>(problem.batch_size), 0);
  }
  const int total_tokens = problem.TotalTokens();
  const int width = problem.Width();
  const int buffer_capacity = problem.BufferCapacity();

  problem.query = MakeWave(static_cast<size_t>(total_tokens) * problem.num_heads * problem.head_size, 0.25f, 0.37f);
  problem.key = MakeWave(static_cast<size_t>(total_tokens) * width, 0.60f, 0.21f);
  problem.key_norm_weight = MakeWave(static_cast<size_t>(problem.head_size), 0.45f, 0.31f);
  problem.cos_cache = MakeWave(static_cast<size_t>(problem.max_position) * problem.rotary_width, 0.15f, 0.27f);
  problem.sin_cache = MakeWave(static_cast<size_t>(problem.max_position) * problem.rotary_width, 1.05f, 0.33f);
  problem.gate = MakeWave(static_cast<size_t>(total_tokens) * width, 0.80f, 0.24f);
  problem.position_bias = MakeWave(static_cast<size_t>(problem.compress_ratio) * width, 0.33f, 0.11f);
  problem.head_weights = MakeWave(static_cast<size_t>(total_tokens) * problem.num_heads, 1.30f, 0.47f);
  problem.past_key_state =
      MakeWave(static_cast<size_t>(problem.batch_size) * problem.state_capacity * problem.head_size, 0.50f, 0.39f);
  problem.past_kv_buffer = MakeWave(static_cast<size_t>(problem.batch_size) * buffer_capacity * width, 0.95f, 0.18f);
  problem.past_gate_buffer = MakeWave(static_cast<size_t>(problem.batch_size) * buffer_capacity * width, 1.45f, 0.22f);

  if (problem.position_ids.empty()) {
    problem.position_ids.assign(static_cast<size_t>(total_tokens), 0);
    for (int b = 0; b < problem.batch_size; ++b) {
      const int req_start = problem.cumulative_sequence_lengths[static_cast<size_t>(b)];
      const int req_end = problem.cumulative_sequence_lengths[static_cast<size_t>(b) + 1];
      for (int token = req_start; token < req_end; ++token) {
        problem.position_ids[static_cast<size_t>(token)] = 2 + (token - req_start);
      }
    }
  }
  return problem;
}

template <typename T>
void RunCsaPackedTest(const CsaPackedProblem& base, float tolerance,
                      ProviderKind provider_kind = ProviderKind::Cuda, CsaPackedResult* actual = nullptr) {
  auto provider = CreateProvider(provider_kind);
  if (provider == nullptr) {
    GTEST_SKIP() << (provider_kind == ProviderKind::Cuda ? "CUDA" : "WebGPU")
                 << " execution provider is not available";
  }

  CsaPackedProblem problem = base;
  problem.query = RoundTrip<T>(problem.query);
  problem.key = RoundTrip<T>(problem.key);
  problem.key_norm_weight = RoundTrip<T>(problem.key_norm_weight);
  problem.cos_cache = RoundTrip<T>(problem.cos_cache);
  problem.sin_cache = RoundTrip<T>(problem.sin_cache);
  problem.gate = RoundTrip<T>(problem.gate);
  problem.position_bias = RoundTrip<T>(problem.position_bias);
  problem.head_weights = RoundTrip<T>(problem.head_weights);
  problem.past_key_state = RoundTrip<T>(problem.past_key_state);
  problem.past_kv_buffer = RoundTrip<T>(problem.past_kv_buffer);
  problem.past_gate_buffer = RoundTrip<T>(problem.past_gate_buffer);

  CsaPackedResult expected;
  CsaPackedReference(problem, expected);

  const int64_t total_tokens = problem.TotalTokens();
  const int64_t batch_size = problem.batch_size;
  const int64_t head_size = problem.head_size;
  const int64_t width = problem.Width();
  const int64_t buffer_capacity = problem.BufferCapacity();

  OpTester test("PackedSparseAttentionIndexer", 1, onnxruntime::kMSDomain);
  test.AddAttribute("policy_mode", std::string(psai::kPolicyModeCsa));
  test.AddAttribute("compress_ratio", static_cast<int64_t>(problem.compress_ratio));
  test.AddAttribute("state_capacity", static_cast<int64_t>(problem.state_capacity));
  test.AddAttribute("index_topk", static_cast<int64_t>(problem.index_topk));
  if (problem.scale.has_value()) test.AddAttribute("scale", *problem.scale);
  if (problem.head_weight_scale.has_value()) test.AddAttribute("head_weight_scale", *problem.head_weight_scale);

  test.AddInput<T>("query", {total_tokens, problem.num_heads, head_size}, ToElementType<T>(problem.query));
  test.AddInput<T>("key", {total_tokens, width}, ToElementType<T>(problem.key));
  test.AddInput<T>("key_norm_weight", {head_size}, ToElementType<T>(problem.key_norm_weight));
  test.AddInput<T>("cos_cache", {problem.max_position, problem.rotary_width}, ToElementType<T>(problem.cos_cache));
  test.AddInput<T>("sin_cache", {problem.max_position, problem.rotary_width}, ToElementType<T>(problem.sin_cache));
  test.AddInput<int32_t>("cumulative_sequence_lengths", {batch_size + 1}, problem.cumulative_sequence_lengths);
  test.AddInput<int32_t>("past_sequence_lengths", {batch_size}, problem.past_sequence_lengths);
  test.AddInput<T>("gate", {total_tokens, width}, ToElementType<T>(problem.gate));
  test.AddInput<T>("position_bias", {problem.compress_ratio, width}, ToElementType<T>(problem.position_bias));
  test.AddInput<T>("head_weights", {total_tokens, problem.num_heads}, ToElementType<T>(problem.head_weights));
  test.AddInput<int64_t>("position_ids", {total_tokens}, problem.position_ids);
  test.AddInput<T>("past_key_state", {batch_size, problem.state_capacity, head_size},
                   ToElementType<T>(problem.past_key_state));
  test.AddInput<T>("past_kv_buffer", {batch_size, buffer_capacity, width}, ToElementType<T>(problem.past_kv_buffer));
  test.AddInput<T>("past_gate_buffer", {batch_size, buffer_capacity, width},
                   ToElementType<T>(problem.past_gate_buffer));
  test.AddInput<int32_t>("past_state_lengths", {batch_size, 2}, problem.past_state_lengths);

  test.AddOutput<int32_t>("selected_indices", {total_tokens, problem.index_topk}, expected.selected_indices);
  test.AddOutput<int32_t>("selected_counts", {total_tokens}, expected.selected_counts);
  test.AddOutput<T>("present_key_state", {batch_size, problem.state_capacity, head_size},
                    ToElementType<T>(expected.present_key_state), false, 0.0f, tolerance);
  test.AddOutput<T>("present_kv_buffer", {batch_size, buffer_capacity, width},
                    ToElementType<T>(expected.present_kv_buffer), false, 0.0f, tolerance);
  test.AddOutput<T>("present_gate_buffer", {batch_size, buffer_capacity, width},
                    ToElementType<T>(expected.present_gate_buffer), false, 0.0f, tolerance);
  test.AddOutput<int32_t>("present_state_lengths", {batch_size, 2}, expected.present_state_lengths);
  RunOnProvider(test, std::move(provider));
  if (actual != nullptr) {
    const auto& fetches = test.GetFetches();
    actual->present_key_state = CopyTensorToFloat<T>(fetches[2].Get<Tensor>());
    actual->present_kv_buffer = CopyTensorToFloat<T>(fetches[3].Get<Tensor>());
    actual->present_gate_buffer = CopyTensorToFloat<T>(fetches[4].Get<Tensor>());
    actual->present_state_lengths.assign(fetches[5].Get<Tensor>().DataAsSpan<int32_t>().begin(),
                                         fetches[5].Get<Tensor>().DataAsSpan<int32_t>().end());
  }
}

}  // namespace

TEST(PackedSparseAttentionIndexerTest, CsaFloat) { RunCsaPackedTest<float>(MakeCsaPackedProblem(), 1.0e-5f); }

TEST(PackedSparseAttentionIndexerTest, CsaFloat16) { RunCsaPackedTest<MLFloat16>(MakeCsaPackedProblem(), 4.0e-3f); }

TEST(PackedSparseAttentionIndexerTest, CsaBFloat16) { RunCsaPackedTest<BFloat16>(MakeCsaPackedProblem(), 3.0e-2f); }

TEST(PackedSparseAttentionIndexerTest, CsaPrefillThenDecodeIndependentState) {
  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA execution provider is not available";
  }

  CsaPackedProblem prefill;
  prefill.cumulative_sequence_lengths = {0, 3, 5};
  prefill.past_sequence_lengths = {0, 0};
  CsaPackedResult prefill_outputs;
  RunCsaPackedTest<float>(MakeCsaPackedProblem(std::move(prefill)), 1.0e-5f, ProviderKind::Cuda,
                          &prefill_outputs);

  CsaPackedProblem decode;
  decode.cumulative_sequence_lengths = {0, 1, 3};
  decode.past_sequence_lengths = {3, 2};
  decode.position_ids = {3, 2, 3};
  decode = MakeCsaPackedProblem(std::move(decode));
  decode.past_key_state = std::move(prefill_outputs.present_key_state);
  decode.past_kv_buffer = std::move(prefill_outputs.present_kv_buffer);
  decode.past_gate_buffer = std::move(prefill_outputs.present_gate_buffer);
  decode.past_state_lengths = std::move(prefill_outputs.present_state_lengths);
  RunCsaPackedTest<float>(decode, 1.0e-5f);
}

TEST(PackedSparseAttentionIndexerTest, CsaStateCapacityOverflowIsRejected) {
  CsaPackedProblem problem;
  problem.batch_size = 1;
  problem.cumulative_sequence_lengths = {0, 4};
  problem.state_capacity = 1;
  problem.past_state_lengths = {1, 0};
  RunCsaPackedTest<float>(MakeCsaPackedProblem(std::move(problem)), 1.0e-5f);
}

#ifdef USE_WEBGPU
TEST(PackedSparseAttentionIndexerWebGpuTest, CsaFloat) {
  RunCsaPackedTest<float>(MakeCsaPackedProblem(), 1.0e-5f, ProviderKind::WebGpu);
}

TEST(PackedSparseAttentionIndexerWebGpuTest, CsaFloat16) {
  RunCsaPackedTest<MLFloat16>(MakeCsaPackedProblem(), 4.0e-3f, ProviderKind::WebGpu);
}

TEST(PackedSparseAttentionIndexerWebGpuTest, CsaStateCapacityOverflowIsRejected) {
  CsaPackedProblem problem;
  problem.batch_size = 1;
  problem.cumulative_sequence_lengths = {0, 4};
  problem.state_capacity = 1;
  problem.past_state_lengths = {1, 0};
  RunCsaPackedTest<float>(MakeCsaPackedProblem(std::move(problem)), 1.0e-5f, ProviderKind::WebGpu);
}
#endif

}  // namespace test
}  // namespace onnxruntime
