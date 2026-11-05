// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <optional>
#include <random>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

#include "gtest/gtest.h"

#include "core/framework/session_options.h"
#include "core/graph/constants.h"
#include "core/graph/model.h"
#include "test/providers/provider_test_utils.h"
#include "test/test_environment.h"
#include "test/unittest_util/conversion.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {
namespace {

// One MultiHeadLatentAttention call. Rows of past_kv beyond past_rows[b] hold noise, which the op must not read.
struct MlaTestCase {
  int batch_size = 1;
  int sequence_length = 1;
  int num_heads = 4;
  int head_size = 40;
  int v_head_size = 24;
  int v_offset = 8;
  bool has_past = true;
  int past_length = 0;  // length of the past_kv buffer
  std::vector<int32_t> seqlens_k;
  int total_sequence_length = 0;
  bool has_present = true;
  int threads = 0;  // intra-op threads, 0 for the default
};

struct MlaTestData {
  std::vector<float> query;
  std::vector<float> key;
  std::vector<float> past_kv;
  std::vector<double> output;         // (B, S, H * Dv); NaN where unspecified (padding tokens)
  std::vector<float> present_kv;      // (B, 1, L', D)
  std::vector<bool> present_written;  // rows of present_kv the op defines
  int present_length = 0;
  float scale = 0.0f;
};

template <typename T>
float RoundTo(float x) {
  if constexpr (std::is_same_v<T, MLFloat16>) {
    return MLFloat16(x).ToFloat();
  } else {
    return x;
  }
}

// Random inputs and, when `compute_reference` is set, the float64 reference of the op with GroupQueryAttention's
// cache rules.
template <typename T>
MlaTestData MakeTestData(const MlaTestCase& c, bool compute_reference = true) {
  const int B = c.batch_size, S = c.sequence_length, H = c.num_heads, D = c.head_size, Dv = c.v_head_size;
  const int L = c.has_past ? c.past_length : 0;
  MlaTestData d;
  d.scale = 1.0f / std::sqrt(static_cast<float>(D));
  std::mt19937 gen(1234 + B * 7 + S * 13 + H * 17 + D);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  auto fill = [&](std::vector<float>& v, size_t n, float stddev) {
    v.resize(n);
    for (auto& x : v) x = RoundTo<T>(stddev * normal(gen));
  };
  fill(d.query, static_cast<size_t>(B) * S * H * D, 0.5f);
  fill(d.key, static_cast<size_t>(B) * S * D, 1.0f);
  fill(d.past_kv, static_cast<size_t>(B) * L * D, 1.0f);

  const bool first_prompt = S == c.total_sequence_length;
  d.present_length = std::max(c.total_sequence_length, L);
  const size_t Lp = static_cast<size_t>(d.present_length);
  d.present_kv.assign(static_cast<size_t>(B) * Lp * D, 0.0f);
  d.present_written.assign(static_cast<size_t>(B) * Lp, false);
  d.output.assign(static_cast<size_t>(B) * S * H * Dv, std::nan(""));

  for (int b = 0; compute_reference && b < B; ++b) {
    const int total = c.seqlens_k[b] + 1;
    const int past = first_prompt ? 0 : total - S;
    float* present = d.present_kv.data() + b * Lp * D;
    for (int j = 0; j < past; ++j) {
      std::copy_n(d.past_kv.data() + (static_cast<size_t>(b) * L + j) * D, D, present + static_cast<size_t>(j) * D);
      d.present_written[b * Lp + j] = true;
    }
    for (int s = 0; s < S; ++s) {
      std::copy_n(d.key.data() + (static_cast<size_t>(b) * S + s) * D, D, present + static_cast<size_t>(past + s) * D);
      d.present_written[b * Lp + past + s] = true;
    }
    for (int s = 0; s < S; ++s) {
      if (first_prompt && s >= total) {
        continue;  // padding token
      }
      const int limit = std::min(past + s + 1, total);
      for (int h = 0; h < H; ++h) {
        const float* q = d.query.data() + ((static_cast<size_t>(b) * S + s) * H + h) * D;
        std::vector<double> p(limit);
        double max_score = -INFINITY;
        for (int j = 0; j < limit; ++j) {
          double dot = 0.0;
          for (int k = 0; k < D; ++k) dot += static_cast<double>(q[k]) * present[static_cast<size_t>(j) * D + k];
          p[j] = dot * d.scale;
          max_score = std::max(max_score, p[j]);
        }
        double sum = 0.0;
        for (auto& x : p) {
          x = std::exp(x - max_score);
          sum += x;
        }
        double* out = d.output.data() + ((static_cast<size_t>(b) * S + s) * H + h) * Dv;
        for (int v = 0; v < Dv; ++v) {
          double acc = 0.0;
          for (int j = 0; j < limit; ++j) acc += p[j] * present[static_cast<size_t>(j) * D + c.v_offset + v];
          out[v] = acc / sum;
        }
      }
    }
  }
  return d;
}

template <typename T>
std::vector<T> ToTyped(const std::vector<float>& v) {
  if constexpr (std::is_same_v<T, MLFloat16>) {
    return FloatsToMLFloat16s(v);
  } else {
    return v;
  }
}

template <typename T>
float ToFloat(T v) {
  if constexpr (std::is_same_v<T, MLFloat16>) {
    return v.ToFloat();
  } else {
    return v;
  }
}

template <typename T>
void AddMlaInputs(OpTester& tester, const MlaTestCase& c, const MlaTestData& d) {
  const int64_t B = c.batch_size, S = c.sequence_length, H = c.num_heads, D = c.head_size;
  tester.AddAttribute<int64_t>("num_heads", c.num_heads);
  tester.AddAttribute<int64_t>("v_head_size", c.v_head_size);
  tester.AddAttribute<int64_t>("v_offset", c.v_offset);
  tester.AddAttribute<float>("scale", d.scale);
  tester.AddInput<T>("query", {B, S, H * D}, ToTyped<T>(d.query));
  tester.AddInput<T>("key", {B, S, D}, ToTyped<T>(d.key));
  if (c.has_past) {
    tester.AddInput<T>("past_kv", {B, 1, c.past_length, D}, ToTyped<T>(d.past_kv));
  } else {
    tester.AddOptionalInputEdge<T>();
  }
  tester.AddInput<int32_t>("seqlens_k", {B}, c.seqlens_k);
  tester.AddInput<int32_t>("total_sequence_length", {1}, {c.total_sequence_length});
}

template <typename T>
void RunMlaTest(const MlaTestCase& c) {
  const MlaTestData d = MakeTestData<T>(c);
  const int64_t B = c.batch_size, S = c.sequence_length, H = c.num_heads, D = c.head_size, Dv = c.v_head_size;
  OpTester tester("MultiHeadLatentAttention", 1, kMSDomain);
  AddMlaInputs<T>(tester, c, d);
  // Expected values are checked by the custom verifier below: padding outputs and unwritten cache rows are
  // unspecified.
  tester.AddOutput<T>("output", {B, S, H * Dv}, std::vector<T>(B * S * H * Dv));
  if (c.has_present) {
    tester.AddOutput<T>("present_kv", {B, 1, d.present_length, D}, std::vector<T>(B * d.present_length * D));
  }

  const float tolerance = std::is_same_v<T, MLFloat16> ? 2e-3f : 2e-5f;
  tester.SetCustomOutputVerifier([&](const std::vector<OrtValue>& fetches, const std::string&) {
    ASSERT_EQ(fetches.size(), c.has_present ? 2u : 1u);
    const Tensor& output = fetches[0].Get<Tensor>();
    ASSERT_EQ(output.Shape(), TensorShape({B, S, H * Dv}));
    double ref_max = 0.0;
    for (double x : d.output) {
      if (!std::isnan(x)) ref_max = std::max(ref_max, std::abs(x));
    }
    double max_err = 0.0;
    const T* out = output.Data<T>();
    for (size_t i = 0; i < d.output.size(); ++i) {
      if (!std::isnan(d.output[i])) {
        const double actual = ToFloat(out[i]);
        ASSERT_TRUE(std::isfinite(actual)) << "output[" << i << "] = " << actual;
        max_err = std::max(max_err, std::abs(actual - d.output[i]));
      }
    }
    EXPECT_LE(max_err, tolerance * std::max(ref_max, 1e-3)) << "max abs error " << max_err << ", max |ref| " << ref_max;

    if (c.has_present) {
      const Tensor& present = fetches[1].Get<Tensor>();
      ASSERT_EQ(present.Shape(), TensorShape({B, 1, d.present_length, D}));
      const T* p = present.Data<T>();
      for (size_t row = 0; row < d.present_written.size(); ++row) {
        if (!d.present_written[row]) continue;
        for (int64_t k = 0; k < D; ++k) {
          ASSERT_EQ(ToFloat(p[row * D + k]), d.present_kv[row * D + k]) << "present_kv row " << row;
        }
      }
    }
  });

  SessionOptions so;
  if (c.threads > 0) {
    so.intra_op_param.thread_pool_size = c.threads;
  }
  tester.Config(so).ConfigEp(DefaultCpuExecutionProvider()).RunWithConfig();
}

void RunMlaTestBothTypes(const MlaTestCase& c) {
  RunMlaTest<float>(c);
  RunMlaTest<MLFloat16>(c);
}

void RunMlaFailureTest(const MlaTestCase& c, const std::string& expected_message) {
  const MlaTestData d = MakeTestData<float>(c, /*compute_reference=*/false);
  const int64_t B = c.batch_size, S = c.sequence_length, H = c.num_heads, D = c.head_size, Dv = c.v_head_size;
  OpTester tester("MultiHeadLatentAttention", 1, kMSDomain);
  AddMlaInputs<float>(tester, c, d);
  tester.AddOutput<float>("output", {B, S, H * Dv}, std::vector<float>(B * S * H * Dv));
  if (c.has_present) {
    tester.AddOutput<float>("present_kv", {B, 1, d.present_length, D}, std::vector<float>(B * d.present_length * D));
  }
  tester.Config(OpTester::ExpectResult::kExpectFailure, expected_message)
      .ConfigEp(DefaultCpuExecutionProvider())
      .RunWithConfig();
}

// An OpTester that leaves input shapes out of the model, as in an export with symbolic shapes.
class ShapelessOpTester : public OpTester {
 public:
  using BaseTester::SetAddShapeToTensorData;
  using OpTester::OpTester;
};

// Runs one decode step with inputs of the given shapes but no shapes in the model, so the kernel's own input checks
// run rather than shape inference.
void RunMlaKernelCheckFailure(int64_t v_offset, const std::vector<int64_t>& query_dims,
                              const std::vector<int64_t>& key_dims, const std::vector<int64_t>& past_dims,
                              const std::string& expected_message) {
  constexpr int64_t kNumHeads = 4, kValueSize = 24;
  auto zeros = [](const std::vector<int64_t>& dims) {
    return std::vector<float>(static_cast<size_t>(TensorShape(dims).Size()), 0.0f);
  };
  const int64_t B = query_dims[0], S = query_dims[1];
  ShapelessOpTester tester("MultiHeadLatentAttention", 1, kMSDomain);
  tester.SetAddShapeToTensorData(false);
  tester.AddAttribute<int64_t>("num_heads", kNumHeads);
  tester.AddAttribute<int64_t>("v_head_size", kValueSize);
  tester.AddAttribute<int64_t>("v_offset", v_offset);
  tester.AddAttribute<float>("scale", 0.125f);
  tester.AddInput<float>("query", query_dims, zeros(query_dims));
  tester.AddInput<float>("key", key_dims, zeros(key_dims));
  tester.AddInput<float>("past_kv", past_dims, zeros(past_dims));
  tester.AddInput<int32_t>("seqlens_k", {B}, std::vector<int32_t>(static_cast<size_t>(B), 3));
  tester.AddInput<int32_t>("total_sequence_length", {1}, {8});
  const std::vector<int64_t> output_dims{B, S, kNumHeads * kValueSize};
  tester.AddOutput<float>("output", output_dims, zeros(output_dims));
  tester.AddOutput<float>("present_kv", past_dims, zeros(past_dims));
  tester.Config(OpTester::ExpectResult::kExpectFailure, expected_message)
      .ConfigEp(DefaultCpuExecutionProvider())
      .RunWithConfig();
}

MlaTestCase Case(int batch_size, int sequence_length, bool has_past, int past_length,
                 std::vector<int32_t> seqlens_k, int total_sequence_length) {
  MlaTestCase c;
  c.batch_size = batch_size;
  c.sequence_length = sequence_length;
  c.has_past = has_past;
  c.past_length = past_length;
  c.seqlens_k = std::move(seqlens_k);
  c.total_sequence_length = total_sequence_length;
  return c;
}

}  // namespace

TEST(MultiHeadLatentAttentionTest, FirstPromptWithoutPast) {
  RunMlaTestBothTypes(Case(2, 6, false, 0, {5, 5}, 6));
}

TEST(MultiHeadLatentAttentionTest, FirstPromptWithoutPastOrPresent) {
  MlaTestCase c = Case(1, 5, false, 0, {4}, 5);
  c.has_present = false;
  RunMlaTestBothTypes(c);
}

TEST(MultiHeadLatentAttentionTest, FirstPromptEmptyPast) {
  RunMlaTestBothTypes(Case(1, 4, true, 0, {3}, 4));
}

// A first prompt writes from row 0 of a preallocated cache, and right-padded entries only attend to their real rows.
TEST(MultiHeadLatentAttentionTest, FirstPromptRightPadded) {
  RunMlaTestBothTypes(Case(3, 7, true, 16, {6, 3, 0}, 7));
}

TEST(MultiHeadLatentAttentionTest, DecodePreallocatedRagged) {
  RunMlaTestBothTypes(Case(3, 1, true, 64, {40, 7, 63}, 64));
}

TEST(MultiHeadLatentAttentionTest, DecodeGrowingCacheRagged) {
  RunMlaTestBothTypes(Case(2, 1, true, 20, {20, 12}, 21));
}

TEST(MultiHeadLatentAttentionTest, ChunkGrowingCache) {
  RunMlaTestBothTypes(Case(1, 9, true, 37, {45}, 46));
}

TEST(MultiHeadLatentAttentionTest, ChunkPreallocatedCache) {
  RunMlaTestBothTypes(Case(1, 9, true, 128, {60}, 128));
}

TEST(MultiHeadLatentAttentionTest, DecodeWithoutPresent) {
  MlaTestCase c = Case(2, 1, true, 32, {20, 31}, 32);
  c.has_present = false;
  RunMlaTestBothTypes(c);
}

TEST(MultiHeadLatentAttentionTest, ValueIsWholeRow) {
  MlaTestCase c = Case(1, 3, true, 24, {12}, 24);
  c.v_offset = 0;
  c.v_head_size = c.head_size;
  RunMlaTestBothTypes(c);
}

TEST(MultiHeadLatentAttentionTest, OddSizes) {
  MlaTestCase c = Case(2, 1, true, 50, {33, 49}, 50);
  c.num_heads = 3;
  c.head_size = 13;
  c.v_head_size = 7;
  c.v_offset = 5;
  RunMlaTestBothTypes(c);
}

// Decode with a long cache splits the cache across threads and merges the partial softmax states.
TEST(MultiHeadLatentAttentionTest, LongDecodeSplitsCache) {
  MlaTestCase c = Case(2, 1, true, 3000, {2999, 1800}, 3000);
  c.num_heads = 16;
  c.head_size = 72;
  c.v_head_size = 64;
  c.v_offset = 0;
  c.threads = 4;
  RunMlaTestBothTypes(c);
}

// A chunk with few row blocks also splits the cache, with causal masking inside the splits.
TEST(MultiHeadLatentAttentionTest, LongChunkSplitsCache) {
  MlaTestCase c = Case(1, 40, true, 700, {659}, 700);
  c.num_heads = 8;
  c.threads = 4;
  RunMlaTestBothTypes(c);
}

TEST(MultiHeadLatentAttentionTest, ManyHeads) {
  MlaTestCase c = Case(1, 2, true, 300, {200}, 300);
  c.num_heads = 144;
  c.head_size = 24;
  c.v_head_size = 16;
  c.v_offset = 4;
  RunMlaTestBothTypes(c);
}

TEST(MultiHeadLatentAttentionTest, RejectsFullCache) {
  RunMlaFailureTest(Case(1, 1, true, 8, {8}, 8), "seqlens_k[0] = 8 is out of range [0, 8)");
}

TEST(MultiHeadLatentAttentionTest, RejectsSeqlensKTooSmall) {
  RunMlaFailureTest(Case(1, 4, true, 16, {2}, 16), "seqlens_k[0] = 2 is too small for sequence_length 4");
}

TEST(MultiHeadLatentAttentionTest, RejectsBadTotalSequenceLength) {
  RunMlaFailureTest(Case(1, 1, true, 8, {8}, 12),
                    "total_sequence_length must equal past_sequence_length + sequence_length");
}

TEST(MultiHeadLatentAttentionTest, RejectsBatchedChunk) {
  RunMlaFailureTest(Case(2, 4, true, 16, {9, 9}, 16), "batch_size must be 1 when sequence_length > 1");
}

// These tests exercise shape inference, which uses fail_shape_inference: in no-exception builds that calls abort().
#ifndef ORT_NO_EXCEPTIONS
TEST(MultiHeadLatentAttentionTest, RejectsValueOutsideRow) {
  MlaTestCase c = Case(1, 1, true, 8, {4}, 8);
  c.v_offset = 20;
  RunMlaFailureTest(c, "v_offset + v_head_size must not exceed head_size");
}
#endif  // ORT_NO_EXCEPTIONS

// The same checks in the kernel, for models whose shapes are only known at run time. head_size = 160 / 4 = 40.
TEST(MultiHeadLatentAttentionTest, KernelRejectsValueOutsideRow) {
  RunMlaKernelCheckFailure(20, {1, 1, 160}, {1, 1, 40}, {1, 1, 8, 40}, "must not exceed head_size (40)");
}

TEST(MultiHeadLatentAttentionTest, KernelRejectsIndivisibleQuery) {
  RunMlaKernelCheckFailure(0, {1, 1, 162}, {1, 1, 40}, {1, 1, 8, 40}, "must be divisible by num_heads");
}

TEST(MultiHeadLatentAttentionTest, KernelRejectsKeyShape) {
  RunMlaKernelCheckFailure(0, {1, 1, 160}, {1, 1, 39}, {1, 1, 8, 40}, "Input 'key' is expected to have shape");
}

TEST(MultiHeadLatentAttentionTest, KernelRejectsPastShape) {
  RunMlaKernelCheckFailure(0, {1, 1, 160}, {1, 1, 40}, {1, 2, 8, 40}, "Input 'past_kv' is expected to have shape");
}

namespace {

using Dim = std::variant<int64_t, std::string>;

// Resolves one MultiHeadLatentAttention node and returns present_kv's inferred shape.
Status InferPresentShape(const std::vector<Dim>& query_shape, const std::optional<std::vector<Dim>>& past_shape,
                         std::optional<int32_t> constant_total_sequence_length,
                         std::optional<ONNX_NAMESPACE::TensorShapeProto>& present_shape, int64_t v_offset = 0) {
  const std::unordered_map<std::string, int> domain_to_version{{kOnnxDomain, 21}, {kMSDomain, 1}};
  Model model("mla_shape_inference", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              domain_to_version, {}, DefaultLoggingManager().DefaultLogger());
  Graph& graph = model.MainGraph();
  ModelTestBuilder builder(graph);
  constexpr int64_t kHeads = 16, kHeadSize = 576, kValueSize = 512;

  NodeArg& empty = graph.GetOrCreateNodeArg("", nullptr);
  NodeArg* query = builder.MakeSymbolicInput<float>(query_shape);
  NodeArg* key = builder.MakeSymbolicInput<float>({query_shape[0], query_shape[1], kHeadSize});
  NodeArg* past = past_shape ? builder.MakeSymbolicInput<float>(*past_shape) : &empty;
  NodeArg* seqlens_k = builder.MakeSymbolicInput<int32_t>({query_shape[0]});
  NodeArg* total = constant_total_sequence_length ? builder.MakeScalarInitializer<int32_t>(*constant_total_sequence_length)
                                                  : builder.MakeSymbolicInput<int32_t>({});
  NodeArg* output = builder.MakeOutput();
  NodeArg* present = builder.MakeOutput();
  Node& node = builder.AddNode("MultiHeadLatentAttention", {query, key, past, seqlens_k, total}, {output, present},
                               kMSDomain);
  node.AddAttribute("num_heads", kHeads);
  node.AddAttribute("v_head_size", kValueSize - v_offset);
  node.AddAttribute("v_offset", v_offset);
  node.AddAttribute("scale", 0.0721f);
  builder.SetGraphOutputs();
  ORT_RETURN_IF_ERROR(graph.Resolve());

  const auto* output_shape = output->Shape();
  ORT_RETURN_IF_NOT(output_shape != nullptr && output_shape->dim_size() == 3 &&
                        output_shape->dim(2).dim_value() == kHeads * (kValueSize - v_offset),
                    "unexpected output shape");
  present_shape.reset();
  if (present->Shape() != nullptr) {
    present_shape = *present->Shape();
  }
  return Status::OK();
}

void ExpectPresentShape(const std::optional<ONNX_NAMESPACE::TensorShapeProto>& shape,
                        const std::vector<std::optional<int64_t>>& expected) {
  ASSERT_TRUE(shape.has_value());
  ASSERT_EQ(shape->dim_size(), static_cast<int>(expected.size()));
  for (size_t i = 0; i < expected.size(); ++i) {
    if (expected[i]) {
      EXPECT_EQ(shape->dim(static_cast<int>(i)).dim_value(), *expected[i]) << "dim " << i;
    } else {
      EXPECT_FALSE(shape->dim(static_cast<int>(i)).has_dim_value()) << "dim " << i;
    }
  }
}

}  // namespace

TEST(MultiHeadLatentAttentionShapeInferenceTest, PresentLength) {
  constexpr int64_t D = 576;
  const std::vector<Dim> decode{int64_t{1}, int64_t{1}, int64_t{16 * D}};
  std::optional<ONNX_NAMESPACE::TensorShapeProto> present;

  // Preallocated cache: present keeps the buffer's length.
  ASSERT_STATUS_OK(InferPresentShape(decode, std::vector<Dim>{int64_t{1}, int64_t{1}, int64_t{4096}, D}, 101, present));
  ExpectPresentShape(present, {1, 1, 4096, D});

  // Growing cache: present is past + new rows.
  ASSERT_STATUS_OK(InferPresentShape(decode, std::vector<Dim>{int64_t{1}, int64_t{1}, int64_t{100}, D}, 101, present));
  ExpectPresentShape(present, {1, 1, 101, D});

  // total_sequence_length known only at run time.
  ASSERT_STATUS_OK(InferPresentShape(decode, std::vector<Dim>{int64_t{1}, int64_t{1}, int64_t{4096}, D}, std::nullopt,
                                     present));
  ExpectPresentShape(present, {1, 1, std::nullopt, D});

  // No past: a first prompt, present holds the new rows.
  ASSERT_STATUS_OK(InferPresentShape({int64_t{2}, int64_t{8}, int64_t{16 * D}}, std::nullopt, 8, present));
  ExpectPresentShape(present, {2, 1, 8, D});
}

#ifndef ORT_NO_EXCEPTIONS
TEST(MultiHeadLatentAttentionShapeInferenceTest, RejectsInvalidLengths) {
  constexpr int64_t D = 576;
  std::optional<ONNX_NAMESPACE::TensorShapeProto> present;
  // A growing cache must grow by exactly sequence_length.
  auto status = InferPresentShape({int64_t{1}, int64_t{1}, int64_t{16 * D}},
                                  std::vector<Dim>{int64_t{1}, int64_t{1}, int64_t{100}, D}, 105, present);
  ASSERT_FALSE(status.IsOK());
  EXPECT_NE(status.ErrorMessage().find("total_sequence_length must equal past_sequence_length + sequence_length"),
            std::string::npos)
      << status.ErrorMessage();

  // past_kv has one head: the cache shared by all query heads.
  status = InferPresentShape({int64_t{1}, int64_t{1}, int64_t{16 * D}},
                             std::vector<Dim>{int64_t{1}, int64_t{2}, int64_t{100}, D}, 101, present);
  ASSERT_FALSE(status.IsOK());
  EXPECT_NE(status.ErrorMessage().find("past_kv dimension 1 must be 1"), std::string::npos) << status.ErrorMessage();
}
#endif  // ORT_NO_EXCEPTIONS

}  // namespace test
}  // namespace onnxruntime
