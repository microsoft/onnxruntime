// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Coverage for com.microsoft.SparseAttentionIndexer.
//
// The "ShapeInference" suite drives Graph::Resolve() directly, so it runs in every build: it pins
// the fixed selected_indices capacity, the policy-specific state outputs and the strict policy
// validation. Cases that are expected to fail shape inference call fail_shape_inference, which
// aborts in ORT_NO_EXCEPTIONS builds, so they are compiled out there.
//
// The numeric suite needs the CUDA execution provider (the operator has no CPU kernel) and is
// skipped when it is unavailable. Expectations come from a float reference in this file that
// mirrors the operator contract; the inputs are rounded to the tested element type first so the
// reference sees exactly what the kernel reads.

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

#include "contrib_ops/cpu/sparse/sparse_attention_indexer_common.h"
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

namespace sai = ::onnxruntime::contrib::sparse_attention_indexer;

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

  model = std::unique_ptr<Model>(new Model("sparse_attention_indexer", /*is_onnx_domain_only=*/false, ModelMetaData(),
                                           PathString(), IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                                           DefaultLoggingManager().DefaultLogger()));

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

struct QsaGraphOptions {
  int64_t batch_size = 2;
  int64_t sequence_length = 3;
  int64_t num_heads = 2;
  int64_t head_size = 8;
  int64_t past_sequence_length = 4;
  int64_t key_cache_capacity = 0;
  int64_t rotary_width = 8;
  int64_t compress_ratio = 2;
  int64_t token_budget = 4;
  bool add_index_topk = false;
  bool add_csa_inputs = false;
  bool share_cache = false;
  bool shared_rotary_cache = false;
  bool packed_qk = false;
  bool add_mask = true;
  std::string policy_mode = sai::kPolicyModeQsa;
};

// Builds a "qsa" node whose csa-only input slots are left empty, as the schema requires.
void AddQsaNode(ModelTestBuilder& builder, const QsaGraphOptions& options) {
  const int64_t total = options.past_sequence_length + options.sequence_length;
  const int64_t cache_capacity = options.share_cache ? options.key_cache_capacity : options.past_sequence_length;
  NodeArg& empty = builder.graph_.GetOrCreateNodeArg("", nullptr);
  std::vector<NodeArg*> inputs{
      builder.MakeInput<float>(
          std::vector<int64_t>{options.batch_size, options.sequence_length,
                               (options.num_heads + (options.packed_qk ? 1 : 0)) * options.head_size}),
      options.packed_qk
          ? &empty
          : builder.MakeInput<float>(
                std::vector<int64_t>{options.batch_size, options.sequence_length, options.head_size}),
      builder.MakeInput<float>(std::vector<int64_t>{options.head_size}),
      builder.MakeInput<float>(std::vector<int64_t>{options.head_size}),
      builder.MakeInput<float>(options.shared_rotary_cache
                                   ? std::vector<int64_t>{total, options.rotary_width}
                                   : std::vector<int64_t>{options.batch_size, total, options.rotary_width}),
      builder.MakeInput<float>(options.shared_rotary_cache
                                   ? std::vector<int64_t>{total, options.rotary_width}
                                   : std::vector<int64_t>{options.batch_size, total, options.rotary_width}),
      options.add_mask
          ? builder.MakeInput<int64_t>(std::vector<int64_t>{options.batch_size, total})
          : &empty,
      builder.MakeInput<float>(std::vector<int64_t>{options.batch_size, cache_capacity, options.head_size}),
  };
  if (options.add_csa_inputs) {
    inputs.push_back(builder.MakeInput<float>(
        std::vector<int64_t>{options.batch_size, options.sequence_length, 2 * options.head_size}));
  } else {
    inputs.push_back(&empty);
  }
  for (int slot = sai::kPositionBias; slot < sai::kInputCount; ++slot) {
    inputs.push_back(&empty);
  }
  if (options.share_cache) {
    inputs[sai::kPastSequenceLength] = builder.MakeInput<int32_t>(std::vector<int64_t>{1});
  }

  std::vector<NodeArg*> outputs{builder.MakeOutput(), builder.MakeOutput()};
  Node& node = builder.AddNode("SparseAttentionIndexer", inputs, outputs, kMSDomain);
  node.AddAttribute("policy_mode", options.policy_mode);
  node.AddAttribute("compress_ratio", options.compress_ratio);
  node.AddAttribute("token_budget", options.token_budget);
  if (options.add_index_topk) {
    node.AddAttribute("index_topk", static_cast<int64_t>(4));
  }
}

struct CsaGraphOptions {
  int64_t batch_size = 2;
  int64_t sequence_length = 5;
  int64_t num_heads = 2;
  int64_t head_size = 8;
  int64_t rotary_width = 4;
  int64_t compress_ratio = 4;
  int64_t index_topk = 3;
  int64_t past_compressed_length = 6;
  int64_t compressed_cache_capacity = 0;
  int64_t past_buffer_length = 5;
  int64_t output_count = sai::kCsaOutputCount;
  bool add_token_budget = false;
  bool share_cache = false;
  bool shared_rotary_cache = false;
};

void AddCsaNode(ModelTestBuilder& builder, const CsaGraphOptions& options) {
  const int64_t width = 2 * options.head_size;
  NodeArg& empty = builder.graph_.GetOrCreateNodeArg("", nullptr);
  std::vector<NodeArg*> inputs{
      builder.MakeInput<float>(
          std::vector<int64_t>{options.batch_size, options.sequence_length, options.num_heads * options.head_size}),
      builder.MakeInput<float>(std::vector<int64_t>{options.batch_size, options.sequence_length, width}),
      builder.MakeInput<float>(std::vector<int64_t>{options.head_size}),
      builder.MakeInput<float>(std::vector<int64_t>{options.head_size}),
      builder.MakeInput<float>(options.shared_rotary_cache
                                   ? std::vector<int64_t>{64, options.rotary_width}
                                   : std::vector<int64_t>{options.batch_size, 64, options.rotary_width}),
      builder.MakeInput<float>(options.shared_rotary_cache
                                   ? std::vector<int64_t>{64, options.rotary_width}
                                   : std::vector<int64_t>{options.batch_size, 64, options.rotary_width}),
      &empty,
      builder.MakeInput<float>(
          std::vector<int64_t>{options.batch_size,
                               options.share_cache ? options.compressed_cache_capacity
                                                   : options.past_compressed_length,
                               options.head_size}),
      builder.MakeInput<float>(std::vector<int64_t>{options.batch_size, options.sequence_length, width}),
      builder.MakeInput<float>(std::vector<int64_t>{options.compress_ratio, width}),
      builder.MakeInput<float>(std::vector<int64_t>{options.batch_size, options.sequence_length, options.num_heads}),
      builder.MakeInput<int64_t>(std::vector<int64_t>{options.batch_size, options.sequence_length}),
      options.share_cache ? builder.MakeInput<int32_t>(std::vector<int64_t>{1}) : &empty,
      builder.MakeInput<float>(std::vector<int64_t>{2, options.batch_size, options.past_buffer_length, width}),
  };

  std::vector<NodeArg*> outputs;
  for (int64_t slot = 0; slot < options.output_count; ++slot) {
    outputs.push_back(builder.MakeOutput());
  }

  Node& node = builder.AddNode("SparseAttentionIndexer", inputs, outputs, kMSDomain);
  node.AddAttribute("policy_mode", std::string(sai::kPolicyModeCsa));
  node.AddAttribute("compress_ratio", options.compress_ratio);
  node.AddAttribute("index_topk", options.index_topk);
  if (options.add_token_budget) {
    node.AddAttribute("token_budget", static_cast<int64_t>(8));
  }
}

// ---------------------------------------------------------------------------------------------
// Numeric reference
// ---------------------------------------------------------------------------------------------

// Deterministic values in [-1, 1]; distinct phases keep the per-block scores well separated so the
// selection order does not depend on rounding.
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

struct QsaProblem {
  int batch_size = 2;
  int sequence_length = 2;
  int num_heads = 2;
  int head_size = 4;
  int past_sequence_length = 3;
  int key_cache_capacity = 0;
  int rotary_width = 4;
  int compress_ratio = 2;
  int token_budget = 4;
  float epsilon = 1.0e-6f;
  std::optional<float> scale;
  bool shared_rotary_cache = false;

  std::vector<float> query;
  std::vector<float> key;
  std::vector<float> query_norm_weight;
  std::vector<float> key_norm_weight;
  std::vector<float> cos_cache;
  std::vector<float> sin_cache;
  std::vector<int64_t> mask;
  std::vector<float> past_key;

  int TotalSequenceLength() const { return past_sequence_length + sequence_length; }
  int PastKeyCapacity() const { return key_cache_capacity > 0 ? key_cache_capacity : past_sequence_length; }
  int PresentKeyCapacity() const { return key_cache_capacity > 0 ? key_cache_capacity : TotalSequenceLength(); }
  int MaxRotaryLength() const { return TotalSequenceLength(); }
  int Capacity() const { return token_budget + compress_ratio - 1; }
};

void QsaReference(const QsaProblem& problem, std::vector<int32_t>& selected, std::vector<float>& present_key) {
  const int total = problem.TotalSequenceLength();
  const int head_size = problem.head_size;
  const int capacity = problem.Capacity();
  const int block_topk = problem.token_budget / problem.compress_ratio;
  const float scale = problem.scale.value_or(1.0f / std::sqrt(static_cast<float>(head_size)));

  const int past_capacity = problem.PastKeyCapacity();
  const int present_capacity = problem.PresentKeyCapacity();
  present_key.assign(static_cast<size_t>(problem.batch_size) * present_capacity * head_size, 0.0f);
  for (int b = 0; b < problem.batch_size; ++b) {
    for (int t = 0; t < present_capacity; ++t) {
      for (int d = 0; d < head_size; ++d) {
        const size_t output_index = (static_cast<size_t>(b) * present_capacity + t) * head_size + d;
        if (t >= problem.past_sequence_length && t < total) {
          present_key[output_index] =
              problem.key[(static_cast<size_t>(b) * problem.sequence_length + t - problem.past_sequence_length) *
                              head_size +
                          d];
        } else if (t < past_capacity) {
          present_key[output_index] =
              problem.past_key[(static_cast<size_t>(b) * past_capacity + t) * head_size + d];
        }
      }
    }
  }

  selected.assign(static_cast<size_t>(problem.batch_size) * problem.sequence_length * capacity, -1);
  for (int b = 0; b < problem.batch_size; ++b) {
    const size_t cache_batch = problem.shared_rotary_cache ? 0 : static_cast<size_t>(b);
    const float* cos_base =
        problem.cos_cache.data() + cache_batch * problem.MaxRotaryLength() * problem.rotary_width;
    const float* sin_base =
        problem.sin_cache.data() + cache_batch * problem.MaxRotaryLength() * problem.rotary_width;
    for (int s = 0; s < problem.sequence_length; ++s) {
      const size_t row = static_cast<size_t>(b) * problem.sequence_length + s;

      std::vector<std::vector<float>> rotated_query(static_cast<size_t>(problem.num_heads));
      const int query_position = problem.past_sequence_length + s;
      for (int h = 0; h < problem.num_heads; ++h) {
        const size_t base = (row * problem.num_heads + h) * head_size;
        std::vector<float> head(problem.query.begin() + base, problem.query.begin() + base + head_size);
        head = RmsNormalize(head, problem.query_norm_weight, problem.epsilon);
        rotated_query[static_cast<size_t>(h)] =
            LeadingRope(head, problem.rotary_width, cos_base + query_position * problem.rotary_width,
                        sin_base + query_position * problem.rotary_width);
      }

      std::vector<int> visible;
      for (int t = 0; t < total; ++t) {
        if (t <= problem.past_sequence_length + s && problem.mask[static_cast<size_t>(b) * total + t] != 0) {
          visible.push_back(t);
        }
      }
      const int block_count = static_cast<int>(visible.size()) / problem.compress_ratio;

      std::vector<float> scores(static_cast<size_t>(block_count), 0.0f);
      for (int block = 0; block < block_count; ++block) {
        std::vector<float> pooled(static_cast<size_t>(head_size), 0.0f);
        for (int t = 0; t < problem.compress_ratio; ++t) {
          const int token = visible[static_cast<size_t>(block * problem.compress_ratio + t)];
          for (int d = 0; d < head_size; ++d) {
            pooled[static_cast<size_t>(d)] +=
                present_key[(static_cast<size_t>(b) * present_capacity + token) * head_size + d];
          }
        }
        for (float& element : pooled) {
          element /= static_cast<float>(problem.compress_ratio);
        }
        pooled = RmsNormalize(pooled, problem.key_norm_weight, problem.epsilon);
        const int key_position = visible[static_cast<size_t>(block * problem.compress_ratio)];
        pooled = LeadingRope(pooled, problem.rotary_width, cos_base + key_position * problem.rotary_width,
                             sin_base + key_position * problem.rotary_width);

        float score = 0.0f;
        for (int h = 0; h < problem.num_heads; ++h) {
          float dot = 0.0f;
          for (int d = 0; d < head_size; ++d) {
            dot += rotated_query[static_cast<size_t>(h)][static_cast<size_t>(d)] * pooled[static_cast<size_t>(d)];
          }
          score += std::max(dot, 0.0f);
        }
        scores[static_cast<size_t>(block)] = score * scale;
      }

      const std::vector<int> order = RankByScore(scores, block_count);
      const int emitted = std::min(block_topk, block_count);
      int32_t* out_row = selected.data() + row * capacity;
      for (int rank = 0; rank < emitted; ++rank) {
        for (int t = 0; t < problem.compress_ratio; ++t) {
          out_row[rank * problem.compress_ratio + t] =
              visible[static_cast<size_t>(order[static_cast<size_t>(rank)] * problem.compress_ratio + t)];
        }
      }
      const int tail_start = block_count * problem.compress_ratio;
      for (size_t t = static_cast<size_t>(tail_start); t < visible.size(); ++t) {
        out_row[emitted * problem.compress_ratio + static_cast<int>(t) - tail_start] = visible[t];
      }
    }
  }
}

struct CsaProblem {
  int batch_size = 2;
  int sequence_length = 3;
  int num_heads = 2;
  int head_size = 4;
  int rotary_width = 2;
  int compress_ratio = 2;
  int index_topk = 2;
  int past_compressed_length = 1;
  int compressed_cache_capacity = 0;
  int past_buffer_length = 3;
  int max_rotary_length = 5;
  float epsilon = 1.0e-6f;
  std::optional<float> scale;
  std::optional<float> head_weight_scale;
  bool shared_rotary_cache = false;

  std::vector<float> query;
  std::vector<float> key;
  std::vector<float> query_norm_weight;
  std::vector<float> key_norm_weight;
  std::vector<float> cos_cache;
  std::vector<float> sin_cache;
  std::vector<float> gate;
  std::vector<float> position_bias;
  std::vector<float> head_weights;
  std::vector<int64_t> position_ids;
  std::vector<float> past_compressed_key;
  std::vector<float> past_kv_buffer;
  std::vector<float> past_gate_buffer;

  int Width() const { return 2 * head_size; }
  int PastCompressedCapacity() const {
    return compressed_cache_capacity > 0 ? compressed_cache_capacity : past_compressed_length;
  }
};

// Value of channel `channel` of token `position` of the virtual sequence [past buffer | new tokens].
float ExtendedValue(const CsaProblem& problem, const std::vector<float>& past, const std::vector<float>& current,
                    int batch, int position, int channel) {
  const int width = problem.Width();
  if (position < problem.past_buffer_length) {
    return past[(static_cast<size_t>(batch) * problem.past_buffer_length + position) * width + channel];
  }
  return current[(static_cast<size_t>(batch) * problem.sequence_length + position - problem.past_buffer_length) *
                     width +
                 channel];
}

void CsaReference(const CsaProblem& problem, std::vector<int32_t>& selected,
                  std::vector<float>& present_compressed_key, std::vector<float>& present_kv_buffer,
                  std::vector<float>& present_gate_buffer) {
  sai::CsaWindowPlan plan;
  ASSERT_TRUE(sai::TryComputeCsaWindowPlan(problem.past_buffer_length, problem.sequence_length,
                                           problem.compress_ratio, plan));

  const int head_size = problem.head_size;
  const int width = problem.Width();
  const int present_compressed_length =
      problem.past_compressed_length + static_cast<int>(plan.new_window_count);
  const int compressed_capacity =
      problem.compressed_cache_capacity > 0 ? problem.compressed_cache_capacity : present_compressed_length;
  const int past_compressed_capacity = problem.PastCompressedCapacity();
  const int present_buffer_length = static_cast<int>(plan.present_buffer_length);
  const float scale = problem.scale.value_or(1.0f / std::sqrt(static_cast<float>(head_size)));
  const float head_weight_scale =
      problem.head_weight_scale.value_or(1.0f / std::sqrt(static_cast<float>(problem.num_heads)));

  present_compressed_key.assign(
      static_cast<size_t>(problem.batch_size) * compressed_capacity * head_size, 0.0f);
  present_kv_buffer.assign(static_cast<size_t>(problem.batch_size) * present_buffer_length * width, 0.0f);
  present_gate_buffer.assign(present_kv_buffer.size(), 0.0f);
  selected.assign(static_cast<size_t>(problem.batch_size) * problem.sequence_length * problem.index_topk, -1);

  for (int b = 0; b < problem.batch_size; ++b) {
    for (int entry = 0; entry < past_compressed_capacity; ++entry) {
      for (int d = 0; d < head_size; ++d) {
        present_compressed_key[(static_cast<size_t>(b) * compressed_capacity + entry) * head_size + d] =
            problem.past_compressed_key[(static_cast<size_t>(b) * past_compressed_capacity + entry) * head_size + d];
      }
    }

    const size_t cache_batch = problem.shared_rotary_cache ? 0 : static_cast<size_t>(b);
    const float* cos_base =
        problem.cos_cache.data() + cache_batch * problem.max_rotary_length * problem.rotary_width;
    const float* sin_base =
        problem.sin_cache.data() + cache_batch * problem.max_rotary_length * problem.rotary_width;

    for (int window = 0; window < plan.new_window_count; ++window) {
      const bool has_previous = window >= 1 || plan.overlap_length >= problem.compress_ratio;
      const int previous_base = static_cast<int>(plan.overlap_length) + (window - 1) * problem.compress_ratio;
      const int current_base = static_cast<int>(plan.overlap_length) + window * problem.compress_ratio;

      std::vector<float> pooled(static_cast<size_t>(head_size), 0.0f);
      for (int d = 0; d < head_size; ++d) {
        std::vector<float> logits;
        std::vector<float> values;
        if (has_previous) {
          for (int slot = 0; slot < problem.compress_ratio; ++slot) {
            logits.push_back(
                ExtendedValue(problem, problem.past_gate_buffer, problem.gate, b, previous_base + slot, d) +
                problem.position_bias[static_cast<size_t>(slot) * width + d]);
            values.push_back(
                ExtendedValue(problem, problem.past_kv_buffer, problem.key, b, previous_base + slot, d));
          }
        }
        for (int slot = 0; slot < problem.compress_ratio; ++slot) {
          logits.push_back(
              ExtendedValue(problem, problem.past_gate_buffer, problem.gate, b, current_base + slot, head_size + d) +
              problem.position_bias[static_cast<size_t>(slot) * width + head_size + d]);
          values.push_back(
              ExtendedValue(problem, problem.past_kv_buffer, problem.key, b, current_base + slot, head_size + d));
        }

        const float max_logit = *std::max_element(logits.begin(), logits.end());
        float denominator = 0.0f;
        float accumulator = 0.0f;
        for (size_t slot = 0; slot < logits.size(); ++slot) {
          const float weight = std::exp(logits[slot] - max_logit);
          denominator += weight;
          accumulator += weight * values[slot];
        }
        pooled[static_cast<size_t>(d)] = accumulator / denominator;
      }

      pooled = RmsNormalize(pooled, problem.key_norm_weight, problem.epsilon);
      const int entry = problem.past_compressed_length + window;
      const int position = std::min(entry * problem.compress_ratio, problem.max_rotary_length - 1);
      pooled = TrailingRope(pooled, problem.rotary_width, cos_base + position * problem.rotary_width,
                            sin_base + position * problem.rotary_width);
      for (int d = 0; d < head_size; ++d) {
        present_compressed_key[(static_cast<size_t>(b) * compressed_capacity + entry) * head_size + d] =
            pooled[static_cast<size_t>(d)];
      }
    }

    for (int token = 0; token < present_buffer_length; ++token) {
      const int source = static_cast<int>(plan.present_buffer_start) + token;
      for (int channel = 0; channel < width; ++channel) {
        const size_t index = (static_cast<size_t>(b) * present_buffer_length + token) * width + channel;
        present_kv_buffer[index] = ExtendedValue(problem, problem.past_kv_buffer, problem.key, b, source, channel);
        present_gate_buffer[index] = ExtendedValue(problem, problem.past_gate_buffer, problem.gate, b, source, channel);
      }
    }

    for (int s = 0; s < problem.sequence_length; ++s) {
      const size_t row = static_cast<size_t>(b) * problem.sequence_length + s;
      const int64_t position = problem.position_ids[row];
      const int query_position = static_cast<int>(
          std::min<int64_t>(std::max<int64_t>(position, 0), problem.max_rotary_length - 1));

      std::vector<std::vector<float>> rotated_query(static_cast<size_t>(problem.num_heads));
      for (int h = 0; h < problem.num_heads; ++h) {
        const size_t base = (row * problem.num_heads + h) * head_size;
        std::vector<float> head(problem.query.begin() + base, problem.query.begin() + base + head_size);
        head = RmsNormalize(head, problem.query_norm_weight, problem.epsilon);
        rotated_query[static_cast<size_t>(h)] =
            TrailingRope(head, problem.rotary_width, cos_base + query_position * problem.rotary_width,
                         sin_base + query_position * problem.rotary_width);
      }

      const int64_t threshold =
          position < 0 ? 0
                       : position / problem.compress_ratio +
                             (position % problem.compress_ratio == problem.compress_ratio - 1);
      std::vector<float> scores(static_cast<size_t>(present_compressed_length), 0.0f);
      for (int entry = 0; entry < present_compressed_length; ++entry) {
        if (static_cast<int64_t>(entry) >= threshold) {
          scores[static_cast<size_t>(entry)] = -std::numeric_limits<float>::infinity();
          continue;
        }
        float total_score = 0.0f;
        for (int h = 0; h < problem.num_heads; ++h) {
          float dot = 0.0f;
          for (int d = 0; d < head_size; ++d) {
            dot += rotated_query[static_cast<size_t>(h)][static_cast<size_t>(d)] *
                   present_compressed_key[(static_cast<size_t>(b) * compressed_capacity + entry) * head_size + d];
          }
          total_score += std::max(dot, 0.0f) * problem.head_weights[row * problem.num_heads + h];
        }
        scores[static_cast<size_t>(entry)] = total_score * scale * head_weight_scale;
      }

      const std::vector<int> order = RankByScore(scores, present_compressed_length);
      const int emitted = std::min(problem.index_topk, present_compressed_length);
      int32_t* out_row = selected.data() + row * problem.index_topk;
      for (int rank = 0; rank < emitted; ++rank) {
        const int entry = order[static_cast<size_t>(rank)];
        out_row[rank] = static_cast<int64_t>(entry) < threshold ? entry : -1;
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Numeric runners
// ---------------------------------------------------------------------------------------------

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

QsaProblem MakeQsaProblem(QsaProblem problem = {}) {
  const int total = problem.TotalSequenceLength();
  problem.query = MakeWave(static_cast<size_t>(problem.batch_size) * problem.sequence_length * problem.num_heads *
                               problem.head_size,
                           0.35f, 0.41f);
  problem.key = MakeWave(static_cast<size_t>(problem.batch_size) * problem.sequence_length * problem.head_size,
                         1.10f, 0.29f);
  problem.query_norm_weight = MakeWave(static_cast<size_t>(problem.head_size), 1.20f, 0.23f);
  problem.key_norm_weight = MakeWave(static_cast<size_t>(problem.head_size), 0.70f, 0.17f);
  const size_t cache_batches = problem.shared_rotary_cache ? 1 : static_cast<size_t>(problem.batch_size);
  problem.cos_cache = MakeWave(cache_batches * total * problem.rotary_width, 0.20f, 0.13f);
  problem.sin_cache = MakeWave(cache_batches * total * problem.rotary_width, 0.90f, 0.19f);
  problem.past_key = MakeWave(
      static_cast<size_t>(problem.batch_size) * problem.PastKeyCapacity() * problem.head_size, 0.05f, 0.23f);

  problem.mask.assign(static_cast<size_t>(problem.batch_size) * total, 1);
  for (int b = 0; b < problem.batch_size; ++b) {
    for (int t = 0; t <= b && t < problem.past_sequence_length; ++t) {
      problem.mask[static_cast<size_t>(b) * total + t] = 0;
    }
  }
  return problem;
}

template <typename T>
void RunQsaTest(float tolerance, QsaProblem problem = MakeQsaProblem(),
                ProviderKind provider_kind = ProviderKind::Cuda, bool packed_qk = false,
                bool omit_mask = false) {
  auto provider = CreateProvider(provider_kind);
  if (provider == nullptr) {
    GTEST_SKIP() << (provider_kind == ProviderKind::Cuda ? "CUDA" : "WebGPU")
                 << " execution provider is not available";
  }

  problem.query = RoundTrip<T>(problem.query);
  problem.key = RoundTrip<T>(problem.key);
  problem.query_norm_weight = RoundTrip<T>(problem.query_norm_weight);
  problem.key_norm_weight = RoundTrip<T>(problem.key_norm_weight);
  problem.cos_cache = RoundTrip<T>(problem.cos_cache);
  problem.sin_cache = RoundTrip<T>(problem.sin_cache);
  problem.past_key = RoundTrip<T>(problem.past_key);

  std::vector<int32_t> selected;
  std::vector<float> present_key;
  QsaReference(problem, selected, present_key);

  const int64_t batch_size = problem.batch_size;
  const int64_t sequence_length = problem.sequence_length;
  const int64_t total = problem.TotalSequenceLength();
  const int64_t head_size = problem.head_size;

  OpTester test("SparseAttentionIndexer", 1, onnxruntime::kMSDomain);
  test.AddAttribute("policy_mode", std::string(sai::kPolicyModeQsa));
  test.AddAttribute("compress_ratio", static_cast<int64_t>(problem.compress_ratio));
  test.AddAttribute("token_budget", static_cast<int64_t>(problem.token_budget));
  if (problem.scale.has_value()) {
    test.AddAttribute("scale", *problem.scale);
  }
  if (packed_qk) {
    std::vector<float> packed;
    const size_t query_row_size = static_cast<size_t>(problem.num_heads * problem.head_size);
    const size_t key_row_size = static_cast<size_t>(problem.head_size);
    packed.reserve(static_cast<size_t>(batch_size * sequence_length) * (query_row_size + key_row_size));
    for (int64_t row = 0; row < batch_size * sequence_length; ++row) {
      packed.insert(packed.end(), problem.query.begin() + row * query_row_size,
                    problem.query.begin() + (row + 1) * query_row_size);
      packed.insert(packed.end(), problem.key.begin() + row * key_row_size,
                    problem.key.begin() + (row + 1) * key_row_size);
    }
    test.AddInput<T>("query", {batch_size, sequence_length, (problem.num_heads + 1) * head_size},
                     ToElementType<T>(packed));
    test.AddOptionalInputEdge<T>();
  } else {
    test.AddInput<T>("query", {batch_size, sequence_length, problem.num_heads * head_size},
                     ToElementType<T>(problem.query));
    test.AddInput<T>("key", {batch_size, sequence_length, head_size}, ToElementType<T>(problem.key));
  }
  test.AddInput<T>("query_norm_weight", {head_size}, ToElementType<T>(problem.query_norm_weight));
  test.AddInput<T>("key_norm_weight", {head_size}, ToElementType<T>(problem.key_norm_weight));
  const std::vector<int64_t> rotary_cache_shape = problem.shared_rotary_cache
                                                      ? std::vector<int64_t>{total, problem.rotary_width}
                                                      : std::vector<int64_t>{batch_size, total, problem.rotary_width};
  test.AddInput<T>("cos_cache", rotary_cache_shape, ToElementType<T>(problem.cos_cache));
  test.AddInput<T>("sin_cache", rotary_cache_shape, ToElementType<T>(problem.sin_cache));
  if (omit_mask) {
    test.AddOptionalInputEdge<int64_t>();
  } else {
    test.AddInput<int64_t>("mask", {batch_size, total}, problem.mask);
  }
  test.AddInput<T>("past_key", {batch_size, problem.PastKeyCapacity(), head_size},
                   ToElementType<T>(problem.past_key));
  for (int slot = sai::kGate; slot < sai::kPastSequenceLength; ++slot) {
    test.AddOptionalInputEdge<T>();
  }
  if (problem.key_cache_capacity > 0) {
    test.AddInput<int32_t>("past_sequence_length", {1}, {problem.past_sequence_length});
  }
  test.AddOutput<int32_t>("selected_indices", {batch_size, sequence_length, problem.Capacity()}, selected);
  test.AddOutput<T>("present_key", {batch_size, problem.PresentKeyCapacity(), head_size},
                    ToElementType<T>(present_key), false, 0.0f, tolerance);
  RunOnProvider(test, std::move(provider));
}

CsaProblem MakeCsaProblem(CsaProblem problem = {}) {
  const int width = problem.Width();
  problem.query = MakeWave(static_cast<size_t>(problem.batch_size) * problem.sequence_length * problem.num_heads *
                               problem.head_size,
                           0.25f, 0.37f);
  problem.key = MakeWave(static_cast<size_t>(problem.batch_size) * problem.sequence_length * width, 0.60f, 0.21f);
  problem.query_norm_weight = MakeWave(static_cast<size_t>(problem.head_size), 1.10f, 0.19f);
  problem.key_norm_weight = MakeWave(static_cast<size_t>(problem.head_size), 0.45f, 0.31f);
  const size_t cache_batches = problem.shared_rotary_cache ? 1 : static_cast<size_t>(problem.batch_size);
  problem.cos_cache =
      MakeWave(cache_batches * problem.max_rotary_length * problem.rotary_width, 0.15f, 0.27f);
  problem.sin_cache =
      MakeWave(cache_batches * problem.max_rotary_length * problem.rotary_width, 1.05f, 0.33f);
  problem.gate = MakeWave(static_cast<size_t>(problem.batch_size) * problem.sequence_length * width, 0.80f, 0.24f);
  problem.position_bias = MakeWave(static_cast<size_t>(problem.compress_ratio) * width, 0.33f, 0.11f);
  problem.head_weights = MakeWave(
      static_cast<size_t>(problem.batch_size) * problem.sequence_length * problem.num_heads, 1.30f, 0.47f);
  problem.past_compressed_key = MakeWave(
      static_cast<size_t>(problem.batch_size) * problem.PastCompressedCapacity() * problem.head_size, 0.50f, 0.39f);
  problem.past_kv_buffer =
      MakeWave(static_cast<size_t>(problem.batch_size) * problem.past_buffer_length * width, 0.95f, 0.18f);
  problem.past_gate_buffer =
      MakeWave(static_cast<size_t>(problem.batch_size) * problem.past_buffer_length * width, 1.45f, 0.22f);

  problem.position_ids.assign(static_cast<size_t>(problem.batch_size) * problem.sequence_length, 0);
  for (int b = 0; b < problem.batch_size; ++b) {
    for (int s = 0; s < problem.sequence_length; ++s) {
      problem.position_ids[static_cast<size_t>(b) * problem.sequence_length + s] = 2 + s;
    }
  }
  return problem;
}

template <typename T>
void RunCsaTest(const CsaProblem& base, float tolerance,
                ProviderKind provider_kind = ProviderKind::Cuda) {
  auto provider = CreateProvider(provider_kind);
  if (provider == nullptr) {
    GTEST_SKIP() << (provider_kind == ProviderKind::Cuda ? "CUDA" : "WebGPU")
                 << " execution provider is not available";
  }

  CsaProblem problem = base;
  problem.query = RoundTrip<T>(problem.query);
  problem.key = RoundTrip<T>(problem.key);
  problem.query_norm_weight = RoundTrip<T>(problem.query_norm_weight);
  problem.key_norm_weight = RoundTrip<T>(problem.key_norm_weight);
  problem.cos_cache = RoundTrip<T>(problem.cos_cache);
  problem.sin_cache = RoundTrip<T>(problem.sin_cache);
  problem.gate = RoundTrip<T>(problem.gate);
  problem.position_bias = RoundTrip<T>(problem.position_bias);
  problem.head_weights = RoundTrip<T>(problem.head_weights);
  problem.past_compressed_key = RoundTrip<T>(problem.past_compressed_key);
  problem.past_kv_buffer = RoundTrip<T>(problem.past_kv_buffer);
  problem.past_gate_buffer = RoundTrip<T>(problem.past_gate_buffer);

  std::vector<int32_t> selected;
  std::vector<float> present_compressed_key;
  std::vector<float> present_kv_buffer;
  std::vector<float> present_gate_buffer;
  CsaReference(problem, selected, present_compressed_key, present_kv_buffer, present_gate_buffer);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());

  sai::CsaWindowPlan plan;
  ASSERT_TRUE(sai::TryComputeCsaWindowPlan(problem.past_buffer_length, problem.sequence_length,
                                           problem.compress_ratio, plan));
  const int64_t batch_size = problem.batch_size;
  const int64_t sequence_length = problem.sequence_length;
  const int64_t head_size = problem.head_size;
  const int64_t width = problem.Width();
  const int64_t present_compressed_length = problem.past_compressed_length + plan.new_window_count;

  OpTester test("SparseAttentionIndexer", 1, onnxruntime::kMSDomain);
  test.AddAttribute("policy_mode", std::string(sai::kPolicyModeCsa));
  test.AddAttribute("compress_ratio", static_cast<int64_t>(problem.compress_ratio));
  test.AddAttribute("index_topk", static_cast<int64_t>(problem.index_topk));
  if (problem.scale.has_value()) {
    test.AddAttribute("scale", *problem.scale);
  }
  if (problem.head_weight_scale.has_value()) {
    test.AddAttribute("head_weight_scale", *problem.head_weight_scale);
  }
  test.AddInput<T>("query", {batch_size, sequence_length, problem.num_heads * head_size},
                   ToElementType<T>(problem.query));
  test.AddInput<T>("key", {batch_size, sequence_length, width}, ToElementType<T>(problem.key));
  test.AddInput<T>("query_norm_weight", {head_size}, ToElementType<T>(problem.query_norm_weight));
  test.AddInput<T>("key_norm_weight", {head_size}, ToElementType<T>(problem.key_norm_weight));
  const std::vector<int64_t> rotary_cache_shape =
      problem.shared_rotary_cache
          ? std::vector<int64_t>{problem.max_rotary_length, problem.rotary_width}
          : std::vector<int64_t>{batch_size, problem.max_rotary_length, problem.rotary_width};
  test.AddInput<T>("cos_cache", rotary_cache_shape, ToElementType<T>(problem.cos_cache));
  test.AddInput<T>("sin_cache", rotary_cache_shape, ToElementType<T>(problem.sin_cache));
  test.AddOptionalInputEdge<int64_t>();
  test.AddInput<T>("past_key", {batch_size, problem.PastCompressedCapacity(), head_size},
                   ToElementType<T>(problem.past_compressed_key));
  test.AddInput<T>("gate", {batch_size, sequence_length, width}, ToElementType<T>(problem.gate));
  test.AddInput<T>("position_bias", {problem.compress_ratio, width}, ToElementType<T>(problem.position_bias));
  test.AddInput<T>("head_weights", {batch_size, sequence_length, problem.num_heads},
                   ToElementType<T>(problem.head_weights));
  test.AddInput<int64_t>("position_ids", {batch_size, sequence_length}, problem.position_ids);
  if (problem.compressed_cache_capacity > 0) {
    test.AddInput<int32_t>("past_sequence_length", {1}, {problem.past_compressed_length});
  } else {
    test.AddOptionalInputEdge<int32_t>();
  }
  std::vector<float> past_proj_buffer = problem.past_kv_buffer;
  past_proj_buffer.insert(past_proj_buffer.end(), problem.past_gate_buffer.begin(), problem.past_gate_buffer.end());
  test.AddInput<T>("past_proj_buffer", {2, batch_size, problem.past_buffer_length, width},
                   ToElementType<T>(past_proj_buffer));

  test.AddOutput<int32_t>("selected_indices", {batch_size, sequence_length, problem.index_topk}, selected);
  const int64_t compressed_capacity =
      problem.compressed_cache_capacity > 0 ? problem.compressed_cache_capacity : present_compressed_length;
  test.AddOutput<T>("present_key", {batch_size, compressed_capacity, head_size},
                    ToElementType<T>(present_compressed_key), false, 0.0f, tolerance);
  present_kv_buffer.insert(present_kv_buffer.end(), present_gate_buffer.begin(), present_gate_buffer.end());
  test.AddOutput<T>("present_proj_buffer", {2, batch_size, plan.present_buffer_length, width},
                    ToElementType<T>(present_kv_buffer), false, 0.0f, tolerance);
  RunOnProvider(test, std::move(provider));
}

// A call whose tokens do not close a window: the buffer only grows and the compressed state is
// unchanged, so the queries score against the entries produced by earlier calls.
CsaProblem MakeCsaBufferOnlyProblem() {
  CsaProblem problem;
  problem.batch_size = 1;
  problem.sequence_length = 1;
  problem.num_heads = 1;
  problem.compress_ratio = 4;
  problem.past_compressed_length = 2;
  problem.past_buffer_length = 2;
  problem.max_rotary_length = 9;
  const int width = problem.Width();

  problem.query = MakeWave(static_cast<size_t>(problem.num_heads) * problem.head_size, 0.31f, 0.43f);
  problem.key = MakeWave(static_cast<size_t>(problem.sequence_length) * width, 0.66f, 0.25f);
  problem.query_norm_weight = MakeWave(static_cast<size_t>(problem.head_size), 1.15f, 0.21f);
  problem.key_norm_weight = MakeWave(static_cast<size_t>(problem.head_size), 0.41f, 0.35f);
  problem.cos_cache = MakeWave(static_cast<size_t>(problem.max_rotary_length) * problem.rotary_width, 0.12f, 0.29f);
  problem.sin_cache = MakeWave(static_cast<size_t>(problem.max_rotary_length) * problem.rotary_width, 1.02f, 0.36f);
  problem.gate = MakeWave(static_cast<size_t>(problem.sequence_length) * width, 0.84f, 0.26f);
  problem.position_bias = MakeWave(static_cast<size_t>(problem.compress_ratio) * width, 0.37f, 0.13f);
  problem.head_weights = MakeWave(static_cast<size_t>(problem.num_heads), 1.20f, 0.51f);
  problem.past_compressed_key =
      MakeWave(static_cast<size_t>(problem.past_compressed_length) * problem.head_size, 0.52f, 0.41f);
  problem.past_kv_buffer = MakeWave(static_cast<size_t>(problem.past_buffer_length) * width, 0.97f, 0.20f);
  problem.past_gate_buffer = MakeWave(static_cast<size_t>(problem.past_buffer_length) * width, 1.48f, 0.24f);
  problem.position_ids = {8};
  return problem;
}

CsaProblem MakeCsaNoCompressedEntryProblem() {
  CsaProblem problem = MakeCsaBufferOnlyProblem();
  problem.past_compressed_length = 0;
  problem.past_compressed_key.clear();
  problem.position_ids = {0};
  return problem;
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Shape inference
// ---------------------------------------------------------------------------------------------

TEST(SparseAttentionIndexerShapeInferenceTest, QsaInfersFixedCapacityAndPresentKey) {
  QsaGraphOptions options;
  std::unique_ptr<Model> model;
  ASSERT_STATUS_OK(BuildAndResolve([&options](ModelTestBuilder& builder) { AddQsaNode(builder, options); }, model));

  const Graph& graph = model->MainGraph();
  const Node& node = *graph.Nodes().begin();
  const int64_t capacity = options.token_budget + options.compress_ratio - 1;
  ExpectShape(graph, node.OutputDefs()[sai::kSelectedIndices]->Name(), ONNX_NAMESPACE::TensorProto_DataType_INT32,
              {options.batch_size, options.sequence_length, capacity});
  ExpectShape(graph, node.OutputDefs()[sai::kPresentKey]->Name(), ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
              {options.batch_size, options.past_sequence_length + options.sequence_length, options.head_size});
}

TEST(SparseAttentionIndexerShapeInferenceTest, QsaAcceptsPackedQk) {
  QsaGraphOptions options;
  options.packed_qk = true;
  std::unique_ptr<Model> model;
  ASSERT_STATUS_OK(BuildAndResolve([&options](ModelTestBuilder& builder) { AddQsaNode(builder, options); }, model));
  ExpectShape(model->MainGraph(), model->MainGraph().GetOutputs()[0]->Name(),
              ONNX_NAMESPACE::TensorProto_DataType_INT32,
              {options.batch_size, options.sequence_length, options.token_budget + options.compress_ratio - 1});
}

TEST(SparseAttentionIndexerShapeInferenceTest, QsaAcceptsOmittedMask) {
  QsaGraphOptions options;
  options.add_mask = false;
  std::unique_ptr<Model> model;
  ASSERT_STATUS_OK(BuildAndResolve([&](ModelTestBuilder& builder) { AddQsaNode(builder, options); }, model));
}

TEST(SparseAttentionIndexerShapeInferenceTest, QsaSharedCacheKeepsCapacity) {
  QsaGraphOptions options;
  options.share_cache = true;
  options.key_cache_capacity = 32;
  std::unique_ptr<Model> model;
  ASSERT_STATUS_OK(BuildAndResolve([&options](ModelTestBuilder& builder) { AddQsaNode(builder, options); }, model));

  const Graph& graph = model->MainGraph();
  const Node& node = *graph.Nodes().begin();
  ExpectShape(graph, node.OutputDefs()[sai::kPresentKey]->Name(), ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
              {options.batch_size, options.key_cache_capacity, options.head_size});
}

TEST(SparseAttentionIndexerShapeInferenceTest, QsaAcceptsSharedRotaryCache) {
  QsaGraphOptions options;
  options.shared_rotary_cache = true;
  std::unique_ptr<Model> model;
  ASSERT_STATUS_OK(BuildAndResolve([&options](ModelTestBuilder& builder) { AddQsaNode(builder, options); }, model));
}

TEST(SparseAttentionIndexerShapeInferenceTest, CsaInfersCompressedStateShapes) {
  CsaGraphOptions options;
  std::unique_ptr<Model> model;
  ASSERT_STATUS_OK(BuildAndResolve([&options](ModelTestBuilder& builder) { AddCsaNode(builder, options); }, model));

  // buffer_length 5 with compress_ratio 4 means one complete window is buffered and one token is
  // pending, so the five new tokens close exactly one window and leave two pending.
  sai::CsaWindowPlan plan;
  ASSERT_TRUE(sai::TryComputeCsaWindowPlan(options.past_buffer_length, options.sequence_length,
                                           options.compress_ratio, plan));
  ASSERT_EQ(plan.new_window_count, 1);
  ASSERT_EQ(plan.present_buffer_length, 6);

  const Graph& graph = model->MainGraph();
  const Node& node = *graph.Nodes().begin();
  ExpectShape(graph, node.OutputDefs()[sai::kSelectedIndices]->Name(), ONNX_NAMESPACE::TensorProto_DataType_INT32,
              {options.batch_size, options.sequence_length, options.index_topk});
  ExpectShape(graph, node.OutputDefs()[sai::kPresentKey]->Name(),
              ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
              {options.batch_size, options.past_compressed_length + plan.new_window_count, options.head_size});
  ExpectShape(graph, node.OutputDefs()[sai::kPresentProjBuffer]->Name(), ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
              {2, options.batch_size, plan.present_buffer_length, 2 * options.head_size});
}

TEST(SparseAttentionIndexerShapeInferenceTest, CsaSharedCacheKeepsCapacity) {
  CsaGraphOptions options;
  options.share_cache = true;
  options.compressed_cache_capacity = 32;
  std::unique_ptr<Model> model;
  ASSERT_STATUS_OK(BuildAndResolve([&options](ModelTestBuilder& builder) { AddCsaNode(builder, options); }, model));

  const Graph& graph = model->MainGraph();
  const Node& node = *graph.Nodes().begin();
  ExpectShape(graph, node.OutputDefs()[sai::kPresentKey]->Name(),
              ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
              {options.batch_size, options.compressed_cache_capacity, options.head_size});
}

TEST(SparseAttentionIndexerShapeInferenceTest, CsaAcceptsSharedRotaryCache) {
  CsaGraphOptions options;
  options.shared_rotary_cache = true;
  std::unique_ptr<Model> model;
  ASSERT_STATUS_OK(BuildAndResolve([&options](ModelTestBuilder& builder) { AddCsaNode(builder, options); }, model));
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

TEST(SparseAttentionIndexerShapeInferenceTest, RejectsUnknownPolicyMode) {
  QsaGraphOptions options;
  options.policy_mode = "qsa_v2";
  ExpectResolveFailure([&options](ModelTestBuilder& builder) { AddQsaNode(builder, options); },
                       "policy_mode must be 'qsa' or 'csa'");
}

TEST(SparseAttentionIndexerShapeInferenceTest, RejectsQsaWithCsaAttribute) {
  QsaGraphOptions options;
  options.add_index_topk = true;
  ExpectResolveFailure([&options](ModelTestBuilder& builder) { AddQsaNode(builder, options); },
                       "index_topk and head_weight_scale must not be set");
}

TEST(SparseAttentionIndexerShapeInferenceTest, RejectsZeroNumHeads) {
  QsaGraphOptions options;
  options.num_heads = 0;
  ExpectResolveFailure([&options](ModelTestBuilder& builder) { AddQsaNode(builder, options); },
                       "query width must be > 0");
}

TEST(SparseAttentionIndexerShapeInferenceTest, RejectsPackedQkWithoutQueryHead) {
  QsaGraphOptions options;
  options.packed_qk = true;
  options.num_heads = 0;
  ExpectResolveFailure([&options](ModelTestBuilder& builder) { AddQsaNode(builder, options); },
                       "packed QK input must contain at least one query head and one key");
}

TEST(SparseAttentionIndexerShapeInferenceTest, RejectsQsaTokenBudgetNotDivisibleByCompressRatio) {
  QsaGraphOptions options;
  options.token_budget = 5;
  ExpectResolveFailure([&options](ModelTestBuilder& builder) { AddQsaNode(builder, options); },
                       "requires token_budget > 0, divisible by compress_ratio");
}

TEST(SparseAttentionIndexerShapeInferenceTest, RejectsOversizedQsaCapacity) {
  QsaGraphOptions options;
  options.compress_ratio = 2;
  options.token_budget = static_cast<int64_t>(std::numeric_limits<int>::max()) + 1;
  ExpectResolveFailure([&options](ModelTestBuilder& builder) { AddQsaNode(builder, options); },
                       "selected capacity no greater than INT_MAX");
}

TEST(SparseAttentionIndexerShapeInferenceTest, RejectsQsaWithCsaInput) {
  QsaGraphOptions options;
  options.add_csa_inputs = true;
  ExpectResolveFailure([&options](ModelTestBuilder& builder) { AddQsaNode(builder, options); },
                       "must be omitted when policy_mode is 'qsa'");
}

TEST(SparseAttentionIndexerShapeInferenceTest, RejectsCsaWithQsaAttribute) {
  CsaGraphOptions options;
  options.add_token_budget = true;
  ExpectResolveFailure([&options](ModelTestBuilder& builder) { AddCsaNode(builder, options); },
                       "token_budget must not be set when policy_mode is 'csa'");
}

// A "csa" node must declare every state output. Rejecting the node before any output is written
// keeps inference from touching an output index the node does not have.
TEST(SparseAttentionIndexerShapeInferenceTest, RejectsCsaWithMissingStateOutputs) {
  CsaGraphOptions options;
  options.output_count = 2;
  ExpectResolveFailure([&options](ModelTestBuilder& builder) { AddCsaNode(builder, options); },
                       "requires exactly 3 declared outputs");
}

TEST(SparseAttentionIndexerShapeInferenceTest, RejectsOversizedCsaBuffer) {
  CsaGraphOptions options;
  options.past_buffer_length = 2 * options.compress_ratio;
  ExpectResolveFailure([&options](ModelTestBuilder& builder) { AddCsaNode(builder, options); },
                       "past_proj_buffer sequence length must be in [0, 2 * compress_ratio)");
}

#endif  // ORT_NO_EXCEPTIONS

// ---------------------------------------------------------------------------------------------
// Numeric behaviour (CUDA only)
// ---------------------------------------------------------------------------------------------

TEST(SparseAttentionIndexerTest, QsaFloat) { RunQsaTest<float>(1.0e-5f); }

TEST(SparseAttentionIndexerTest, QsaPackedQkFloat) {
  RunQsaTest<float>(1.0e-5f, MakeQsaProblem(), ProviderKind::Cuda, true);
}

TEST(SparseAttentionIndexerTest, QsaMasklessPrefixCausal) {
  QsaProblem problem = MakeQsaProblem();
  std::fill(problem.mask.begin(), problem.mask.end(), 1);
  RunQsaTest<float>(1.0e-5f, std::move(problem), ProviderKind::Cuda, false, true);
}

TEST(SparseAttentionIndexerTest, QsaFloat16) { RunQsaTest<MLFloat16>(2.0e-3f); }

TEST(SparseAttentionIndexerTest, QsaBFloat16) { RunQsaTest<BFloat16>(2.0e-2f); }

TEST(SparseAttentionIndexerTest, QsaMultiTileAndStridedChannels) {
  QsaProblem problem;
  problem.batch_size = 1;
  problem.head_size = 192;
  problem.past_sequence_length = 200;
  problem.rotary_width = 4;
  RunQsaTest<float>(1.0e-5f, MakeQsaProblem(std::move(problem)));
}

TEST(SparseAttentionIndexerTest, QsaNonPrefixMask) {
  QsaProblem problem = MakeQsaProblem();
  const int total = problem.TotalSequenceLength();
  problem.mask[1] = 0;
  problem.mask[total - 1] = 1;
  RunQsaTest<float>(1.0e-5f, std::move(problem));
}

TEST(SparseAttentionIndexerTest, QsaMultiTileNonPrefixMask) {
  QsaProblem problem;
  problem.batch_size = 1;
  problem.sequence_length = 1;
  problem.past_sequence_length = 299;
  problem.token_budget = 64;
  problem = MakeQsaProblem(std::move(problem));
  std::fill(problem.mask.begin(), problem.mask.end(), 0);
  std::fill_n(problem.mask.begin(), 127, 1);
  problem.mask[128] = 1;
  problem.mask[200] = 1;
  RunQsaTest<float>(1.0e-5f, std::move(problem));
}

TEST(SparseAttentionIndexerTest, QsaSinglePassTopKLimit) {
  QsaProblem problem;
  problem.batch_size = 1;
  problem.sequence_length = 1;
  problem.past_sequence_length = 300;
  problem.token_budget = 64;
  RunQsaTest<float>(1.0e-5f, MakeQsaProblem(std::move(problem)));
}

TEST(SparseAttentionIndexerTest, QsaRepeatedScanTopKFallback) {
  QsaProblem problem;
  problem.batch_size = 1;
  problem.sequence_length = 1;
  problem.past_sequence_length = 300;
  problem.token_budget = 66;
  RunQsaTest<float>(1.0e-5f, MakeQsaProblem(std::move(problem)));
}

TEST(SparseAttentionIndexerTest, QsaLongContextBoundedTopKParity) {
  for (const int context_length : {8192, 32768, 65536, 131072, 262144}) {
    SCOPED_TRACE("context_length=" + std::to_string(context_length));
    QsaProblem problem;
    problem.batch_size = 1;
    problem.sequence_length = 1;
    problem.past_sequence_length = context_length - 1;
    problem.compress_ratio = 4;
    problem.token_budget = 2048;
    problem = MakeQsaProblem(std::move(problem));

    std::fill(problem.query.begin(), problem.query.end(), 0.0f);
    for (int head = 0; head < problem.num_heads; ++head) {
      problem.query[static_cast<size_t>(head) * problem.head_size] = 1.0f;
    }
    std::fill(problem.key_norm_weight.begin(), problem.key_norm_weight.end(), 1.0f);
    std::fill(problem.cos_cache.begin(), problem.cos_cache.end(), 1.0f);
    std::fill(problem.sin_cache.begin(), problem.sin_cache.end(), 0.0f);
    std::fill(problem.past_key.begin(), problem.past_key.end(), 0.0f);
    std::fill(problem.key.begin(), problem.key.end(), 0.0f);

    const int block_count = context_length / problem.compress_ratio;
    for (int position = 0; position < context_length; ++position) {
      const float rank = static_cast<float>(position / problem.compress_ratio + 1) /
                         static_cast<float>(block_count);
      std::vector<float>& cache = position < problem.past_sequence_length ? problem.past_key : problem.key;
      const int cache_position = position < problem.past_sequence_length ? position : 0;
      const size_t offset = static_cast<size_t>(cache_position) * problem.head_size;
      cache[offset] = rank;
      cache[offset + 1] = 1.0f;
    }

    RunQsaTest<float>(1.0e-5f, std::move(problem));
  }
}

TEST(SparseAttentionIndexerTest, QsaLongContextBoundedTopKTies) {
  QsaProblem problem;
  problem.batch_size = 1;
  problem.sequence_length = 1;
  problem.past_sequence_length = 65536;
  problem.compress_ratio = 16;
  problem.token_budget = 8192;
  RunQsaTest<float>(1.0e-5f, MakeQsaProblem(std::move(problem)));
}

TEST(SparseAttentionIndexerTest, QsaExplicitZeroScale) {
  QsaProblem problem = MakeQsaProblem();
  problem.scale = 0.0f;
  RunQsaTest<float>(1.0e-5f, std::move(problem));
}

TEST(SparseAttentionIndexerTest, QsaSharedCacheCapacity) {
  QsaProblem problem;
  problem.key_cache_capacity = 16;
  RunQsaTest<float>(1.0e-5f, MakeQsaProblem(std::move(problem)));
}

TEST(SparseAttentionIndexerTest, QsaSharedRotaryCache) {
  QsaProblem problem;
  problem.shared_rotary_cache = true;
  RunQsaTest<float>(1.0e-5f, MakeQsaProblem(std::move(problem)));
}

TEST(SparseAttentionIndexerTest, CsaFloat) { RunCsaTest<float>(MakeCsaProblem(), 1.0e-5f); }

TEST(SparseAttentionIndexerTest, CsaFloat16) { RunCsaTest<MLFloat16>(MakeCsaProblem(), 4.0e-3f); }

TEST(SparseAttentionIndexerTest, CsaBFloat16) { RunCsaTest<BFloat16>(MakeCsaProblem(), 3.0e-2f); }

TEST(SparseAttentionIndexerTest, CsaBufferOnlyStep) { RunCsaTest<float>(MakeCsaBufferOnlyProblem(), 1.0e-5f); }

TEST(SparseAttentionIndexerTest, CsaNoCompressedEntry) {
  RunCsaTest<float>(MakeCsaNoCompressedEntryProblem(), 1.0e-5f);
}

TEST(SparseAttentionIndexerTest, CsaSharedCacheCapacity) {
  CsaProblem problem;
  problem.compressed_cache_capacity = 16;
  RunCsaTest<float>(MakeCsaProblem(std::move(problem)), 1.0e-5f);
}

TEST(SparseAttentionIndexerTest, CsaSharedRotaryCache) {
  CsaProblem problem;
  problem.shared_rotary_cache = true;
  RunCsaTest<float>(MakeCsaProblem(std::move(problem)), 1.0e-5f);
}

TEST(SparseAttentionIndexerTest, CsaExplicitZeroScales) {
  CsaProblem problem = MakeCsaProblem();
  problem.scale = 0.0f;
  problem.head_weight_scale = 0.0f;
  RunCsaTest<float>(problem, 1.0e-5f);
}

TEST(SparseAttentionIndexerTest, CsaInt64MaxPosition) {
  CsaProblem problem = MakeCsaProblem();
  problem.position_ids[0] = std::numeric_limits<int64_t>::max();
  RunCsaTest<float>(problem, 1.0e-5f);
}

TEST(SparseAttentionIndexerTest, CsaEmptyBatch) {
  CsaProblem problem;
  problem.batch_size = 0;
  RunCsaTest<float>(MakeCsaProblem(std::move(problem)), 1.0e-5f);
}

#ifdef USE_WEBGPU
TEST(SparseAttentionIndexerWebGpuTest, QsaFloat) {
  RunQsaTest<float>(1.0e-5f, MakeQsaProblem(), ProviderKind::WebGpu);
}

TEST(SparseAttentionIndexerWebGpuTest, QsaPackedQkFloat) {
  RunQsaTest<float>(1.0e-5f, MakeQsaProblem(), ProviderKind::WebGpu, true);
}

TEST(SparseAttentionIndexerWebGpuTest, QsaFloat16) {
  RunQsaTest<MLFloat16>(4.0e-3f, MakeQsaProblem(), ProviderKind::WebGpu);
}

TEST(SparseAttentionIndexerWebGpuTest, QsaExplicitZeroScale) {
  QsaProblem problem = MakeQsaProblem();
  problem.scale = 0.0f;
  RunQsaTest<float>(1.0e-5f, std::move(problem), ProviderKind::WebGpu);
}

TEST(SparseAttentionIndexerWebGpuTest, QsaSharedRotaryCache) {
  QsaProblem problem;
  problem.shared_rotary_cache = true;
  RunQsaTest<float>(1.0e-5f, MakeQsaProblem(std::move(problem)), ProviderKind::WebGpu);
}

TEST(SparseAttentionIndexerWebGpuTest, CsaFloat) {
  RunCsaTest<float>(MakeCsaProblem(), 1.0e-5f, ProviderKind::WebGpu);
}

TEST(SparseAttentionIndexerWebGpuTest, CsaFloat16) {
  RunCsaTest<MLFloat16>(MakeCsaProblem(), 6.0e-3f, ProviderKind::WebGpu);
}

TEST(SparseAttentionIndexerWebGpuTest, CsaSharedRotaryCache) {
  CsaProblem problem;
  problem.shared_rotary_cache = true;
  RunCsaTest<float>(MakeCsaProblem(std::move(problem)), 1.0e-5f, ProviderKind::WebGpu);
}

TEST(SparseAttentionIndexerWebGpuTest, CsaBufferOnlyStep) {
  RunCsaTest<float>(MakeCsaBufferOnlyProblem(), 1.0e-5f, ProviderKind::WebGpu);
}

TEST(SparseAttentionIndexerWebGpuTest, CsaNoCompressedEntry) {
  RunCsaTest<float>(MakeCsaNoCompressedEntryProblem(), 1.0e-5f, ProviderKind::WebGpu);
}

TEST(SparseAttentionIndexerWebGpuTest, CsaExplicitZeroScales) {
  CsaProblem problem = MakeCsaProblem();
  problem.scale = 0.0f;
  problem.head_weight_scale = 0.0f;
  RunCsaTest<float>(problem, 1.0e-5f, ProviderKind::WebGpu);
}

TEST(SparseAttentionIndexerWebGpuTest, CsaInt64MaxPosition) {
  CsaProblem problem = MakeCsaProblem();
  problem.position_ids[0] = std::numeric_limits<int64_t>::max();
  RunCsaTest<float>(problem, 1.0e-5f, ProviderKind::WebGpu);
}
#endif
}  // namespace test
}  // namespace onnxruntime
