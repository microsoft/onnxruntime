// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Unit tests for the DQMatMulNBitsFusion graph transformer.
// Tests Pattern 1: DQ(3D,axis=2)->Reshape->Transpose([1,0])->[Cast]->MatMul/Gemm -> MatMulNBits
// Tests Pattern 2: DQ(2D,axis=0)->MatMul/Gemm -> MatMulNBits
#include <unordered_map>

#include "core/common/span_utils.h"
#include "core/framework/int4.h"
#include "core/framework/prepacked_weights_container.h"
#include "core/graph/constants.h"
#include "core/graph/model.h"
#include "core/graph/node_attr_utils.h"
#include "core/optimizer/dq_matmulnbits_fusion.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/optimizer/graph_transform_test_fixture.h"
#include "test/util/include/asserts.h"
#include "test/util/include/inference_session_wrapper.h"

#include "gtest/gtest.h"

#if !defined(DISABLE_CONTRIB_OPS)

namespace onnxruntime {
namespace test {

static std::vector<UInt4x2> MakePackedUint4(const std::vector<uint8_t>& values) {
  const size_t num_pairs = UInt4x2::CalcNumInt4Pairs(values.size());
  std::vector<UInt4x2> packed(num_pairs);
  for (size_t i = 0; i < values.size(); i += 2) {
    uint8_t lo = values[i] & 0x0F;
    uint8_t hi = (i + 1 < values.size()) ? (values[i + 1] & 0x0F) : 0;
    packed[i / 2] = UInt4x2(lo, hi);
  }
  return packed;
}

static void BuildPattern1Graph(ModelTestBuilder& builder,
                               int64_t M, int64_t N, int64_t K,
                               int64_t block_size,
                               bool with_zp,
                               bool with_cast,
                               bool use_gemm,
                               const std::vector<uint8_t>* weight_values = nullptr,
                               const std::vector<float>* scale_values = nullptr,
                               const std::vector<uint8_t>* zp_values = nullptr) {
  const int64_t num_blocks = K / block_size;

  auto* input_a = builder.MakeInput<float>({M, K}, -1.0f, 1.0f);
  auto* output = builder.MakeOutput();

  const int64_t weight_elems = N * num_blocks * block_size;
  std::vector<uint8_t> w_vals;
  if (weight_values) {
    w_vals = *weight_values;
  } else {
    w_vals.resize(static_cast<size_t>(weight_elems));
    for (size_t i = 0; i < w_vals.size(); ++i) {
      w_vals[i] = static_cast<uint8_t>(i % 16);
    }
  }
  auto w_packed = MakePackedUint4(w_vals);
  auto* weight_arg = builder.MakeInitializer<UInt4x2>(
      {N, num_blocks, block_size}, w_packed);

  std::vector<float> s_vals;
  if (scale_values) {
    s_vals = *scale_values;
  } else {
    s_vals.resize(static_cast<size_t>(N * num_blocks));
    for (size_t i = 0; i < s_vals.size(); ++i) {
      s_vals[i] = 0.1f + 0.01f * static_cast<float>(i % 10);
    }
  }
  auto* scale_arg = builder.MakeInitializer<float>({N, num_blocks, 1}, s_vals);

  NodeAttributes dq_attrs;
  utils::SetNodeAttribute(utils::MakeAttribute("axis", static_cast<int64_t>(2)), dq_attrs);
  utils::SetNodeAttribute(utils::MakeAttribute("block_size", block_size), dq_attrs);

  auto* dq_output = builder.MakeIntermediate();
  if (with_zp) {
    std::vector<uint8_t> z_vals;
    if (zp_values) {
      z_vals = *zp_values;
    } else {
      z_vals.resize(static_cast<size_t>(N * num_blocks));
      for (size_t i = 0; i < z_vals.size(); ++i) {
        z_vals[i] = 8;
      }
    }
    auto zp_packed = MakePackedUint4(z_vals);
    auto* zp_arg = builder.MakeInitializer<UInt4x2>({N, num_blocks, 1}, zp_packed);
    builder.AddNode("DequantizeLinear", {weight_arg, scale_arg, zp_arg}, {dq_output}, "", &dq_attrs);
  } else {
    builder.AddNode("DequantizeLinear", {weight_arg, scale_arg}, {dq_output}, "", &dq_attrs);
  }

  auto* reshape_shape = builder.MakeInitializer<int64_t>({2}, {N, K});
  auto* reshape_output = builder.MakeIntermediate();
  builder.AddNode("Reshape", {dq_output, reshape_shape}, {reshape_output});

  NodeAttributes tp_attrs;
  utils::SetNodeAttribute(utils::MakeAttribute("perm", std::vector<int64_t>{1, 0}), tp_attrs);
  auto* transpose_output = builder.MakeIntermediate();
  builder.AddNode("Transpose", {reshape_output}, {transpose_output}, "", &tp_attrs);

  NodeArg* matmul_b = transpose_output;

  if (with_cast) {
    auto* cast_output = builder.MakeIntermediate();
    NodeAttributes cast_attrs;
    utils::SetNodeAttribute(utils::MakeAttribute("to", static_cast<int64_t>(1)), cast_attrs);
    builder.AddNode("Cast", {transpose_output}, {cast_output}, "", &cast_attrs);
    matmul_b = cast_output;
  }

  if (use_gemm) {
    builder.AddNode("Gemm", {input_a, matmul_b}, {output});
  } else {
    builder.AddNode("MatMul", {input_a, matmul_b}, {output});
  }
}

static void BuildPattern1GemmBiasGraph(ModelTestBuilder& builder,
                                       int64_t M, int64_t N, int64_t K,
                                       int64_t block_size,
                                       bool with_zp) {
  const int64_t num_blocks = K / block_size;

  auto* input_a = builder.MakeInput<float>({M, K}, -1.0f, 1.0f);
  auto* output = builder.MakeOutput();

  const int64_t weight_elems = N * num_blocks * block_size;
  std::vector<uint8_t> w_vals(static_cast<size_t>(weight_elems));
  for (size_t i = 0; i < w_vals.size(); ++i) w_vals[i] = static_cast<uint8_t>(i % 16);
  auto w_packed = MakePackedUint4(w_vals);
  auto* weight_arg = builder.MakeInitializer<UInt4x2>({N, num_blocks, block_size}, w_packed);

  std::vector<float> s_vals(static_cast<size_t>(N * num_blocks));
  for (size_t i = 0; i < s_vals.size(); ++i) s_vals[i] = 0.1f;
  auto* scale_arg = builder.MakeInitializer<float>({N, num_blocks, 1}, s_vals);

  NodeAttributes dq_attrs;
  utils::SetNodeAttribute(utils::MakeAttribute("axis", static_cast<int64_t>(2)), dq_attrs);
  utils::SetNodeAttribute(utils::MakeAttribute("block_size", block_size), dq_attrs);
  auto* dq_output = builder.MakeIntermediate();

  if (with_zp) {
    std::vector<uint8_t> z_vals(static_cast<size_t>(N * num_blocks), 8);
    auto zp_packed = MakePackedUint4(z_vals);
    auto* zp_arg = builder.MakeInitializer<UInt4x2>({N, num_blocks, 1}, zp_packed);
    builder.AddNode("DequantizeLinear", {weight_arg, scale_arg, zp_arg}, {dq_output}, "", &dq_attrs);
  } else {
    builder.AddNode("DequantizeLinear", {weight_arg, scale_arg}, {dq_output}, "", &dq_attrs);
  }

  auto* reshape_shape = builder.MakeInitializer<int64_t>({2}, {N, K});
  auto* reshape_output = builder.MakeIntermediate();
  builder.AddNode("Reshape", {dq_output, reshape_shape}, {reshape_output});

  NodeAttributes tp_attrs;
  utils::SetNodeAttribute(utils::MakeAttribute("perm", std::vector<int64_t>{1, 0}), tp_attrs);
  auto* transpose_output = builder.MakeIntermediate();
  builder.AddNode("Transpose", {reshape_output}, {transpose_output}, "", &tp_attrs);

  auto* bias_arg = builder.MakeInitializer<float>({N}, std::vector<float>(static_cast<size_t>(N), 0.5f));
  builder.AddNode("Gemm", {input_a, transpose_output, bias_arg}, {output});
}

static void BuildPattern2Graph(ModelTestBuilder& builder,
                               int64_t M, int64_t N, int64_t K,
                               int64_t block_size,
                               bool with_zp,
                               bool use_gemm) {
  const int64_t k_blocks = K / block_size;

  auto* input_a = builder.MakeInput<float>({M, K}, -1.0f, 1.0f);
  auto* output = builder.MakeOutput();

  std::vector<uint8_t> w_vals(static_cast<size_t>(K * N));
  for (size_t i = 0; i < w_vals.size(); ++i) w_vals[i] = static_cast<uint8_t>(i % 16);
  auto w_packed = MakePackedUint4(w_vals);
  auto* weight_arg = builder.MakeInitializer<UInt4x2>({K, N}, w_packed);

  std::vector<float> s_vals(static_cast<size_t>(k_blocks * N));
  for (size_t i = 0; i < s_vals.size(); ++i) s_vals[i] = 0.1f + 0.01f * static_cast<float>(i % 10);
  auto* scale_arg = builder.MakeInitializer<float>({k_blocks, N}, s_vals);

  NodeAttributes dq_attrs;
  utils::SetNodeAttribute(utils::MakeAttribute("axis", static_cast<int64_t>(0)), dq_attrs);
  utils::SetNodeAttribute(utils::MakeAttribute("block_size", block_size), dq_attrs);
  auto* dq_output = builder.MakeIntermediate();

  if (with_zp) {
    std::vector<uint8_t> z_vals(static_cast<size_t>(k_blocks * N), 8);
    auto zp_packed = MakePackedUint4(z_vals);
    auto* zp_arg = builder.MakeInitializer<UInt4x2>({k_blocks, N}, zp_packed);
    builder.AddNode("DequantizeLinear", {weight_arg, scale_arg, zp_arg}, {dq_output}, "", &dq_attrs);
  } else {
    builder.AddNode("DequantizeLinear", {weight_arg, scale_arg}, {dq_output}, "", &dq_attrs);
  }

  if (use_gemm) {
    builder.AddNode("Gemm", {input_a, dq_output}, {output});
  } else {
    builder.AddNode("MatMul", {input_a, dq_output}, {output});
  }
}

static void BuildPattern2GemmBiasGraph(ModelTestBuilder& builder,
                                       int64_t M, int64_t N, int64_t K,
                                       int64_t block_size,
                                       bool with_zp) {
  const int64_t k_blocks = K / block_size;

  auto* input_a = builder.MakeInput<float>({M, K}, -1.0f, 1.0f);
  auto* output = builder.MakeOutput();

  std::vector<uint8_t> w_vals(static_cast<size_t>(K * N));
  for (size_t i = 0; i < w_vals.size(); ++i) w_vals[i] = static_cast<uint8_t>(i % 16);
  auto w_packed = MakePackedUint4(w_vals);
  auto* weight_arg = builder.MakeInitializer<UInt4x2>({K, N}, w_packed);

  std::vector<float> s_vals(static_cast<size_t>(k_blocks * N));
  for (size_t i = 0; i < s_vals.size(); ++i) s_vals[i] = 0.1f;
  auto* scale_arg = builder.MakeInitializer<float>({k_blocks, N}, s_vals);

  NodeAttributes dq_attrs;
  utils::SetNodeAttribute(utils::MakeAttribute("axis", static_cast<int64_t>(0)), dq_attrs);
  utils::SetNodeAttribute(utils::MakeAttribute("block_size", block_size), dq_attrs);
  auto* dq_output = builder.MakeIntermediate();

  if (with_zp) {
    std::vector<uint8_t> z_vals(static_cast<size_t>(k_blocks * N), 8);
    auto zp_packed = MakePackedUint4(z_vals);
    auto* zp_arg = builder.MakeInitializer<UInt4x2>({k_blocks, N}, zp_packed);
    builder.AddNode("DequantizeLinear", {weight_arg, scale_arg, zp_arg}, {dq_output}, "", &dq_attrs);
  } else {
    builder.AddNode("DequantizeLinear", {weight_arg, scale_arg}, {dq_output}, "", &dq_attrs);
  }

  auto* bias_arg = builder.MakeInitializer<float>({N}, std::vector<float>(static_cast<size_t>(N), 0.5f));
  builder.AddNode("Gemm", {input_a, dq_output, bias_arg}, {output});
}

static void BuildPattern1WrongAxis(ModelTestBuilder& builder,
                                   int64_t M, int64_t N, int64_t K,
                                   int64_t block_size) {
  const int64_t num_blocks = K / block_size;
  auto* input_a = builder.MakeInput<float>({M, K}, -1.0f, 1.0f);
  auto* output = builder.MakeOutput();

  std::vector<uint8_t> w_vals(static_cast<size_t>(N * num_blocks * block_size));
  for (size_t i = 0; i < w_vals.size(); ++i) w_vals[i] = static_cast<uint8_t>(i % 16);
  auto w_packed = MakePackedUint4(w_vals);
  auto* weight_arg = builder.MakeInitializer<UInt4x2>({N, num_blocks, block_size}, w_packed);

  std::vector<float> s_vals(static_cast<size_t>(N * num_blocks), 0.1f);
  auto* scale_arg = builder.MakeInitializer<float>({N, num_blocks, 1}, s_vals);

  NodeAttributes dq_attrs;
  utils::SetNodeAttribute(utils::MakeAttribute("axis", static_cast<int64_t>(0)), dq_attrs);
  utils::SetNodeAttribute(utils::MakeAttribute("block_size", block_size), dq_attrs);
  auto* dq_output = builder.MakeIntermediate();
  builder.AddNode("DequantizeLinear", {weight_arg, scale_arg}, {dq_output}, "", &dq_attrs);

  auto* reshape_shape = builder.MakeInitializer<int64_t>({2}, {N, K});
  auto* reshape_output = builder.MakeIntermediate();
  builder.AddNode("Reshape", {dq_output, reshape_shape}, {reshape_output});

  NodeAttributes tp_attrs;
  utils::SetNodeAttribute(utils::MakeAttribute("perm", std::vector<int64_t>{1, 0}), tp_attrs);
  auto* transpose_output = builder.MakeIntermediate();
  builder.AddNode("Transpose", {reshape_output}, {transpose_output}, "", &tp_attrs);

  builder.AddNode("MatMul", {input_a, transpose_output}, {output});
}

static void BuildPattern2NonConstWeight(ModelTestBuilder& builder,
                                        int64_t M, int64_t N, int64_t K,
                                        int64_t block_size) {
  const int64_t k_blocks = K / block_size;
  auto* input_a = builder.MakeInput<float>({M, K}, -1.0f, 1.0f);
  auto* output = builder.MakeOutput();

  auto* weight_arg = builder.MakeInput<UInt4x2>({K, N},
                                                UInt4x2(UInt4x2::min_val, 0),
                                                UInt4x2(UInt4x2::max_val, 0));

  std::vector<float> s_vals(static_cast<size_t>(k_blocks * N), 0.1f);
  auto* scale_arg = builder.MakeInitializer<float>({k_blocks, N}, s_vals);

  NodeAttributes dq_attrs;
  utils::SetNodeAttribute(utils::MakeAttribute("axis", static_cast<int64_t>(0)), dq_attrs);
  utils::SetNodeAttribute(utils::MakeAttribute("block_size", block_size), dq_attrs);
  auto* dq_output = builder.MakeIntermediate();
  builder.AddNode("DequantizeLinear", {weight_arg, scale_arg}, {dq_output}, "", &dq_attrs);

  builder.AddNode("MatMul", {input_a, dq_output}, {output});
}

static std::map<std::string, int> CountOpsInGraphByDomain(const Graph& graph) {
  std::map<std::string, int> op_counts;
  for (const auto& node : graph.Nodes()) {
    std::string key = node.OpType();
    if (!node.Domain().empty() && node.Domain() != kOnnxDomain) {
      key = node.Domain() + "." + key;
    }
    op_counts[key]++;
  }
  return op_counts;
}

class DQMatMulNBitsFusionTest : public GraphTransformationTests {};

TEST_F(DQMatMulNBitsFusionTest, Pattern1_MatMul_NoZP) {
  constexpr int64_t M = 4, N = 8, K = 32, block_size = 16;
  auto build = [&](ModelTestBuilder& builder) {
    BuildPattern1Graph(builder, M, N, K, block_size, false, false, false);
  };

  auto pre_check = [&](Graph& graph) -> Status {
    auto ops = CountOpsInGraphByDomain(graph);
    EXPECT_EQ(ops["DequantizeLinear"], 1);
    EXPECT_EQ(ops["Reshape"], 1);
    EXPECT_EQ(ops["Transpose"], 1);
    EXPECT_EQ(ops["MatMul"], 1);
    return Status::OK();
  };

  auto post_check = [&](Graph& graph) -> Status {
    auto ops = CountOpsInGraphByDomain(graph);
    EXPECT_EQ(ops.count("DequantizeLinear"), 0);
    EXPECT_EQ(ops.count("Reshape"), 0);
    EXPECT_EQ(ops.count("Transpose"), 0);
    EXPECT_EQ(ops.count("MatMul"), 0);
    EXPECT_EQ(ops["com.microsoft.MatMulNBits"], 1);

    for (const auto& node : graph.Nodes()) {
      if (node.OpType() == "MatMulNBits") {
        const auto& attrs = node.GetAttributes();
        EXPECT_EQ(attrs.at("K").i(), K);
        EXPECT_EQ(attrs.at("N").i(), N);
        EXPECT_EQ(attrs.at("bits").i(), 4);
        EXPECT_EQ(attrs.at("block_size").i(), block_size);
        EXPECT_EQ(node.InputDefs().size(), static_cast<size_t>(4));
      }
    }
    return Status::OK();
  };

  auto transformer = std::make_unique<DQMatMulNBitsFusion>(4);
  ASSERT_STATUS_OK(TestGraphTransformer(build, 21, *logger_, std::move(transformer),
                                        TransformerLevel::Level1, 1, pre_check, post_check));
}

// Validates the cross-session-sharing tag the fusion attaches to the generated B weight. The tag is a
// stable, content-derived enrollment identity: identical source quantization groups yield the SAME
// identity, while a semantic difference -- here, different zero points -- yields a DIFFERENT identity.
// (The tag only enrolls B into the shared container; the actual sharing is keyed by the packed-bytes
// hash, so a stable, content-distinct tag just keeps enrollment deterministic across sessions.)
TEST_F(DQMatMulNBitsFusionTest, TagsGeneratedWeightWithStableContentIdentity) {
  constexpr int64_t M = 4, N = 8, K = 32, block_size = 16;
  const int64_t num_blocks = K / block_size;

  std::vector<uint8_t> weight(static_cast<size_t>(N * num_blocks * block_size));
  for (size_t i = 0; i < weight.size(); ++i) {
    weight[i] = static_cast<uint8_t>(i % 16);
  }
  std::vector<float> scale(static_cast<size_t>(N * num_blocks));
  for (size_t i = 0; i < scale.size(); ++i) {
    scale[i] = 0.1f + 0.01f * static_cast<float>(i % 10);
  }
  // Non-default (non-8) zero points so the fusion keeps them (it elides uniform-8 zero points).
  std::vector<uint8_t> zp_a(static_cast<size_t>(N * num_blocks), 3);
  std::vector<uint8_t> zp_b(zp_a.size(), 5);

  // Runs the fusion on a Pattern-1 model built from the given zero points and returns the sharing
  // identity tagged onto the generated MatMulNBits B weight.
  auto tag_for = [&](const std::vector<uint8_t>& zp) -> std::string {
    std::string captured;
    auto build = [&](ModelTestBuilder& builder) {
      BuildPattern1Graph(builder, M, N, K, block_size, /*with_zp*/ true, /*with_cast*/ false,
                         /*use_gemm*/ false, &weight, &scale, &zp);
    };
    auto pre_check = [](Graph&) -> Status { return Status::OK(); };
    auto post_check = [&](Graph& graph) -> Status {
      int matmulnbits = 0;
      for (const auto& node : graph.Nodes()) {
        if (node.OpType() == "MatMulNBits") {
          ++matmulnbits;
          const std::string& b_name = node.InputDefs()[1]->Name();  // input 1 == quantized B
          const std::string* id = graph.GetSharedPrepackInitializerId(b_name);
          EXPECT_NE(id, nullptr) << "generated B weight was not tagged for cross-session sharing";
          if (id != nullptr) {
            captured = *id;
          }
        }
      }
      EXPECT_EQ(matmulnbits, 1);
      return Status::OK();
    };
    auto transformer = std::make_unique<DQMatMulNBitsFusion>(4);
    EXPECT_TRUE(TestGraphTransformer(build, 21, *logger_, std::move(transformer),
                                     TransformerLevel::Level1, 1, pre_check, post_check)
                    .IsOK());
    return captured;
  };

  const std::string id_a1 = tag_for(zp_a);
  const std::string id_a2 = tag_for(zp_a);
  const std::string id_b = tag_for(zp_b);

  ASSERT_FALSE(id_a1.empty());
  EXPECT_EQ(id_a1, id_a2);  // stable: identical source quantization group -> identical identity
  EXPECT_NE(id_a1, id_b);   // collision-safe: different zero points -> different identity
}

// Builds and serializes a Pattern-1 DQ->Reshape->Transpose->MatMul model (UINT4 constant weight). When
// loaded into a session with the DQ->MatMulNBits fusion enabled, it becomes a MatMulNBits whose B is
// tagged for cross-session sharing.
static void SerializeDQMatMulModel(int64_t M, int64_t N, int64_t K, int64_t block_size,
                                   const std::vector<uint8_t>& weight, const std::vector<float>& scale,
                                   const std::vector<uint8_t>& zp, std::string& model_bytes) {
  const std::unordered_map<std::string, int> domain_to_version{{"", 21}, {kMSDomain, 1}};
  Model model("dq_matmulnbits_share", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), domain_to_version,
              std::vector<ONNX_NAMESPACE::FunctionProto>(), DefaultLoggingManager().DefaultLogger());
  ModelTestBuilder builder(model.MainGraph());
  BuildPattern1Graph(builder, M, N, K, block_size, /*with_zp*/ true, /*with_cast*/ false,
                     /*use_gemm*/ false, &weight, &scale, &zp);
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(model.MainGraph().Resolve());
  ASSERT_TRUE(model.ToProto().SerializeToString(&model_bytes));
}

// Loads the serialized model on the CPU EP with the DQ->MatMulNBits fusion enabled and the supplied
// shared container. Reports whether the fusion produced a MatMulNBits and how many pre-packed weights
// this session served from the container.
static void RunSharedFusionSession(const std::string& model_bytes, PrepackedWeightsContainer& container,
                                   bool& produced_matmulnbits, size_t& used_shared_count) {
  SessionOptions so;
  // This test exercises prepack-weight sharing, not parallel execution. Cap the intra-op thread pool
  // to a single thread so we don't spin up one worker per core: under AddressSanitizer each thread adds
  // fake-stack and thread-local allocator overhead, which on a high-core CI runner multiplies across the
  // sessions every test creates (the sibling SessionStatePrepackingTest caps it for the same reason).
  so.intra_op_param.thread_pool_size = 1;
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsEnableDQMatMulNBitsFusion, "1"));

  InferenceSessionWrapper session{so, GetEnvironment()};
  ASSERT_STATUS_OK(session.AddPrePackedWeightsContainer(&container));
  ASSERT_STATUS_OK(session.Load(model_bytes.data(), static_cast<int>(model_bytes.size())));
  ASSERT_STATUS_OK(session.Initialize());

  produced_matmulnbits = false;
  for (const auto& node : session.GetGraph().Nodes()) {
    if (node.OpType() == "MatMulNBits") {
      produced_matmulnbits = true;
      break;
    }
  }
  used_shared_count = session.GetSessionState().GetUsedSharedPrePackedWeightCounter();
}

// End-to-end: two sessions optimizing the same DQ+MatMul model share the fused MatMulNBits B weight
// through a common container WITHOUT any session option -- the fusion tags it to enroll it, and
// SessionState keys the sharing by the packed-bytes hash. A model whose quantized weight differs packs
// to different bytes, so it gets a different key and must NOT share. (A zero-point-only difference is
// intentionally NOT used: on the CompFp32 path the zero points are not folded into the packed B, so two
// such models pack identically and would correctly share a byte-identical buffer.)
TEST_F(DQMatMulNBitsFusionTest, SharesFusedWeightAcrossSessionsViaTag) {
  constexpr int64_t M = 4, N = 8, K = 32, block_size = 16;
  const int64_t num_blocks = K / block_size;

  std::vector<uint8_t> weight(static_cast<size_t>(N * num_blocks * block_size));
  for (size_t i = 0; i < weight.size(); ++i) {
    weight[i] = static_cast<uint8_t>(i % 16);
  }
  // A different quantized weight -> different packed B on every compute type (unlike a zp-only change).
  std::vector<uint8_t> weight_other(weight.size());
  for (size_t i = 0; i < weight_other.size(); ++i) {
    weight_other[i] = static_cast<uint8_t>((i + 7) % 16);
  }
  std::vector<float> scale(static_cast<size_t>(N * num_blocks));
  for (size_t i = 0; i < scale.size(); ++i) {
    scale[i] = 0.1f + 0.01f * static_cast<float>(i % 10);
  }
  std::vector<uint8_t> zp(static_cast<size_t>(N * num_blocks), 3);

  std::string model_a, model_other;
  SerializeDQMatMulModel(M, N, K, block_size, weight, scale, zp, model_a);
  SerializeDQMatMulModel(M, N, K, block_size, weight_other, scale, zp, model_other);

  PrepackedWeightsContainer container;
  bool fused1 = false, fused2 = false, fused_other = false;
  size_t used1 = 0, used2 = 0, used_other = 0;

  RunSharedFusionSession(model_a, container, fused1, used1);
  ASSERT_TRUE(fused1) << "DQ -> MatMulNBits fusion did not run";
  if (container.GetNumberOfElements() == 0) {
    GTEST_SKIP() << "MatMulNBits B was not pre-packed on this platform";
  }
  EXPECT_EQ(used1, static_cast<size_t>(0));  // first session: nothing to share yet

  // Second session over the SAME model shares the tagged B from the container.
  RunSharedFusionSession(model_a, container, fused2, used2);
  ASSERT_TRUE(fused2);
  EXPECT_GT(used2, static_cast<size_t>(0));

  // A model with a different quantized weight packs to different bytes -> different key, so it must NOT
  // reuse the buffer (on any compute type).
  RunSharedFusionSession(model_other, container, fused_other, used_other);
  ASSERT_TRUE(fused_other);
  EXPECT_EQ(used_other, static_cast<size_t>(0));
}

TEST_F(DQMatMulNBitsFusionTest, Pattern1_MatMul_WithDefaultZP8) {
  constexpr int64_t M = 4, N = 8, K = 32, block_size = 16;

  auto build = [&](ModelTestBuilder& builder) {
    BuildPattern1Graph(builder, M, N, K, block_size, true, false, false);
  };

  auto post_check = [&](Graph& graph) -> Status {
    auto ops = CountOpsInGraphByDomain(graph);
    EXPECT_EQ(ops["com.microsoft.MatMulNBits"], 1);
    EXPECT_EQ(ops.count("DequantizeLinear"), 0);

    for (const auto& node : graph.Nodes()) {
      if (node.OpType() == "MatMulNBits") {
        EXPECT_EQ(node.InputDefs().size(), static_cast<size_t>(3));
      }
    }
    return Status::OK();
  };

  auto transformer = std::make_unique<DQMatMulNBitsFusion>(4);
  ASSERT_STATUS_OK(TestGraphTransformer(build, 21, *logger_, std::move(transformer),
                                        TransformerLevel::Level1, 1, nullptr, post_check));
}

TEST_F(DQMatMulNBitsFusionTest, Pattern1_MatMul_WithNonDefaultZP) {
  constexpr int64_t M = 4, N = 8, K = 32, block_size = 16;
  const int64_t num_blocks = K / block_size;

  std::vector<uint8_t> zp_vals(static_cast<size_t>(N * num_blocks), 3);

  auto build = [&](ModelTestBuilder& builder) {
    BuildPattern1Graph(builder, M, N, K, block_size, true, false, false,
                       nullptr, nullptr, &zp_vals);
  };

  auto post_check = [&](Graph& graph) -> Status {
    auto ops = CountOpsInGraphByDomain(graph);
    EXPECT_EQ(ops["com.microsoft.MatMulNBits"], 1);

    for (const auto& node : graph.Nodes()) {
      if (node.OpType() == "MatMulNBits") {
        EXPECT_EQ(node.InputDefs().size(), static_cast<size_t>(4));
      }
    }
    return Status::OK();
  };

  auto transformer = std::make_unique<DQMatMulNBitsFusion>(4);
  ASSERT_STATUS_OK(TestGraphTransformer(build, 21, *logger_, std::move(transformer),
                                        TransformerLevel::Level1, 1, nullptr, post_check));
}

TEST_F(DQMatMulNBitsFusionTest, Pattern1_MatMul_WithCast) {
  constexpr int64_t M = 4, N = 8, K = 32, block_size = 16;

  auto build = [&](ModelTestBuilder& builder) {
    BuildPattern1Graph(builder, M, N, K, block_size, false, true, false);
  };

  auto post_check = [&](Graph& graph) -> Status {
    auto ops = CountOpsInGraphByDomain(graph);
    EXPECT_EQ(ops["com.microsoft.MatMulNBits"], 1);
    EXPECT_EQ(ops.count("Cast"), 0);
    EXPECT_EQ(ops.count("MatMul"), 0);
    return Status::OK();
  };

  auto transformer = std::make_unique<DQMatMulNBitsFusion>(4);
  ASSERT_STATUS_OK(TestGraphTransformer(build, 21, *logger_, std::move(transformer),
                                        TransformerLevel::Level1, 1, nullptr, post_check));
}

TEST_F(DQMatMulNBitsFusionTest, Pattern1_Gemm_WithBias) {
  constexpr int64_t M = 4, N = 8, K = 32, block_size = 16;

  auto build = [&](ModelTestBuilder& builder) {
    BuildPattern1GemmBiasGraph(builder, M, N, K, block_size, true);
  };

  auto post_check = [&](Graph& graph) -> Status {
    auto ops = CountOpsInGraphByDomain(graph);
    EXPECT_EQ(ops["com.microsoft.MatMulNBits"], 1);
    EXPECT_EQ(ops.count("Gemm"), 0);

    for (const auto& node : graph.Nodes()) {
      if (node.OpType() == "MatMulNBits") {
        EXPECT_GE(node.InputDefs().size(), static_cast<size_t>(6));
        EXPECT_TRUE(node.InputDefs()[5] != nullptr);
        EXPECT_TRUE(node.InputDefs()[5]->Exists());
      }
    }
    return Status::OK();
  };

  auto transformer = std::make_unique<DQMatMulNBitsFusion>(4);
  ASSERT_STATUS_OK(TestGraphTransformer(build, 21, *logger_, std::move(transformer),
                                        TransformerLevel::Level1, 1, nullptr, post_check));
}

TEST_F(DQMatMulNBitsFusionTest, Pattern1_Gemm_NoZP) {
  constexpr int64_t M = 4, N = 8, K = 32, block_size = 16;

  auto build = [&](ModelTestBuilder& builder) {
    BuildPattern1Graph(builder, M, N, K, block_size, false, false, true);
  };

  auto post_check = [&](Graph& graph) -> Status {
    auto ops = CountOpsInGraphByDomain(graph);
    EXPECT_EQ(ops["com.microsoft.MatMulNBits"], 1);
    EXPECT_EQ(ops.count("Gemm"), 0);
    return Status::OK();
  };

  auto transformer = std::make_unique<DQMatMulNBitsFusion>(4);
  ASSERT_STATUS_OK(TestGraphTransformer(build, 21, *logger_, std::move(transformer),
                                        TransformerLevel::Level1, 1, nullptr, post_check));
}

TEST_F(DQMatMulNBitsFusionTest, Pattern2_MatMul_NoZP) {
  constexpr int64_t M = 4, N = 8, K = 32, block_size = 16;

  auto build = [&](ModelTestBuilder& builder) {
    BuildPattern2Graph(builder, M, N, K, block_size, false, false);
  };

  auto post_check = [&](Graph& graph) -> Status {
    auto ops = CountOpsInGraphByDomain(graph);
    EXPECT_EQ(ops.count("DequantizeLinear"), 0);
    EXPECT_EQ(ops.count("MatMul"), 0);
    EXPECT_EQ(ops["com.microsoft.MatMulNBits"], 1);

    for (const auto& node : graph.Nodes()) {
      if (node.OpType() == "MatMulNBits") {
        EXPECT_EQ(node.GetAttributes().at("K").i(), K);
        EXPECT_EQ(node.GetAttributes().at("N").i(), N);
        EXPECT_EQ(node.InputDefs().size(), static_cast<size_t>(4));
      }
    }
    return Status::OK();
  };

  auto transformer = std::make_unique<DQMatMulNBitsFusion>(4);
  ASSERT_STATUS_OK(TestGraphTransformer(build, 21, *logger_, std::move(transformer),
                                        TransformerLevel::Level1, 1, nullptr, post_check));
}

TEST_F(DQMatMulNBitsFusionTest, Pattern2_MatMul_WithDefaultZP8) {
  constexpr int64_t M = 4, N = 8, K = 32, block_size = 16;

  auto build = [&](ModelTestBuilder& builder) {
    BuildPattern2Graph(builder, M, N, K, block_size, true, false);
  };

  auto post_check = [&](Graph& graph) -> Status {
    auto ops = CountOpsInGraphByDomain(graph);
    EXPECT_EQ(ops["com.microsoft.MatMulNBits"], 1);

    for (const auto& node : graph.Nodes()) {
      if (node.OpType() == "MatMulNBits") {
        EXPECT_EQ(node.InputDefs().size(), static_cast<size_t>(3));
      }
    }
    return Status::OK();
  };

  auto transformer = std::make_unique<DQMatMulNBitsFusion>(4);
  ASSERT_STATUS_OK(TestGraphTransformer(build, 21, *logger_, std::move(transformer),
                                        TransformerLevel::Level1, 1, nullptr, post_check));
}

TEST_F(DQMatMulNBitsFusionTest, Pattern2_Gemm_WithBias) {
  constexpr int64_t M = 4, N = 8, K = 32, block_size = 16;

  auto build = [&](ModelTestBuilder& builder) {
    BuildPattern2GemmBiasGraph(builder, M, N, K, block_size, false);
  };

  auto post_check = [&](Graph& graph) -> Status {
    auto ops = CountOpsInGraphByDomain(graph);
    EXPECT_EQ(ops["com.microsoft.MatMulNBits"], 1);
    EXPECT_EQ(ops.count("Gemm"), 0);

    for (const auto& node : graph.Nodes()) {
      if (node.OpType() == "MatMulNBits") {
        EXPECT_GE(node.InputDefs().size(), static_cast<size_t>(6));
      }
    }
    return Status::OK();
  };

  auto transformer = std::make_unique<DQMatMulNBitsFusion>(4);
  ASSERT_STATUS_OK(TestGraphTransformer(build, 21, *logger_, std::move(transformer),
                                        TransformerLevel::Level1, 1, nullptr, post_check));
}

TEST_F(DQMatMulNBitsFusionTest, Negative_Pattern1_WrongAxis) {
  constexpr int64_t M = 4, N = 8, K = 32, block_size = 16;

  auto build = [&](ModelTestBuilder& builder) {
    BuildPattern1WrongAxis(builder, M, N, K, block_size);
  };

  auto post_check = [&](Graph& graph) -> Status {
    auto ops = CountOpsInGraphByDomain(graph);
    EXPECT_EQ(ops.count("com.microsoft.MatMulNBits"), 0);
    EXPECT_EQ(ops["MatMul"], 1);
    return Status::OK();
  };

  auto transformer = std::make_unique<DQMatMulNBitsFusion>(4);
  ASSERT_STATUS_OK(TestGraphTransformer(build, 21, *logger_, std::move(transformer),
                                        TransformerLevel::Level1, 1, nullptr, post_check));
}

TEST_F(DQMatMulNBitsFusionTest, Negative_Pattern2_NonConstWeight) {
  constexpr int64_t M = 4, N = 8, K = 32, block_size = 16;

  auto build = [&](ModelTestBuilder& builder) {
    BuildPattern2NonConstWeight(builder, M, N, K, block_size);
  };

  auto post_check = [&](Graph& graph) -> Status {
    auto ops = CountOpsInGraphByDomain(graph);
    EXPECT_EQ(ops.count("com.microsoft.MatMulNBits"), 0);
    EXPECT_EQ(ops["DequantizeLinear"], 1);
    EXPECT_EQ(ops["MatMul"], 1);
    return Status::OK();
  };

  auto transformer = std::make_unique<DQMatMulNBitsFusion>(4);
  ASSERT_STATUS_OK(TestGraphTransformer(build, 21, *logger_, std::move(transformer),
                                        TransformerLevel::Level1, 1, nullptr, post_check));
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(DISABLE_CONTRIB_OPS)
