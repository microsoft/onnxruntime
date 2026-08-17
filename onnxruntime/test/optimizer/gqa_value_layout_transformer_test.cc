// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <string>
#include <vector>

#include "core/graph/model.h"
#include "core/optimizer/gqa_value_layout_transformer.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

#include "test/util/include/asserts.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/optimizer/graph_transform_test_fixture.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

#if !defined(DISABLE_CONTRIB_OPS)

namespace {

// Geometry kept small, but with max_sequence_length != head_size so that a transpose which fails to
// swap the last two dimensions is caught by the shape assertions rather than passing silently.
constexpr int64_t kBatch = 1;
constexpr int64_t kSeq = 1;
constexpr int64_t kNumHeads = 2;
constexpr int64_t kKvNumHeads = 1;
constexpr int64_t kHeadSize = 16;
constexpr int64_t kMaxSeq = 8;
constexpr int64_t kQHidden = kNumHeads * kHeadSize;
constexpr int64_t kKvHidden = kKvNumHeads * kHeadSize;

struct BuildOptions {
  // Feed past_value through an Identity so it is no longer a graph input.
  bool past_value_behind_identity = false;
  // Route present_value through an Identity so it is no longer a graph output.
  bool present_value_behind_identity = false;
  // Omit the past cache inputs entirely (prefill-only model). GQA type inference requires past_key
  // and past_value to be present or absent together, so both are dropped.
  bool no_past_kv = false;
  // Omit the present_value output entirely.
  bool no_present_value = false;
  // Configure a 4-bit quantized Value cache, which cannot be transposed byte-wise.
  bool four_bit_value_cache = false;

  // Length of the present cache. With a past cache the model shares one max_sequence_length buffer;
  // without one, GQA infers a present cache holding just the new tokens.
  int64_t PresentCacheLength() const { return no_past_kv ? kSeq : kMaxSeq; }
};

void BuildGqaModel(ModelTestBuilder& builder, const BuildOptions& opts) {
  NodeArg& empty_arg = builder.graph_.GetOrCreateNodeArg("", nullptr);

  NodeArg* query = builder.MakeInput<MLFloat16>(
      std::vector<int64_t>{kBatch, kSeq, kQHidden}, MLFloat16(-1.0f), MLFloat16(1.0f));
  NodeArg* key = builder.MakeInput<MLFloat16>(
      std::vector<int64_t>{kBatch, kSeq, kKvHidden}, MLFloat16(-1.0f), MLFloat16(1.0f));
  NodeArg* value = builder.MakeInput<MLFloat16>(
      std::vector<int64_t>{kBatch, kSeq, kKvHidden}, MLFloat16(-1.0f), MLFloat16(1.0f));

  NodeArg* past_key = &empty_arg;
  NodeArg* past_value = &empty_arg;
  if (!opts.no_past_kv) {
    past_key = builder.MakeInput<MLFloat16>(
        std::vector<int64_t>{kBatch, kKvNumHeads, kMaxSeq, kHeadSize}, MLFloat16(0.0f), MLFloat16(0.0f));
    past_value = builder.MakeInput<MLFloat16>(
        std::vector<int64_t>{kBatch, kKvNumHeads, kMaxSeq, kHeadSize}, MLFloat16(0.0f), MLFloat16(0.0f));

    if (opts.past_value_behind_identity) {
      NodeArg* forwarded = builder.MakeIntermediate<MLFloat16>(
          std::vector<int64_t>{kBatch, kKvNumHeads, kMaxSeq, kHeadSize});
      builder.AddNode("Identity", {past_value}, {forwarded});
      past_value = forwarded;
    }
  }

  NodeArg* seqlens_k = builder.MakeInput<int32_t>(std::vector<int64_t>{kBatch}, std::vector<int32_t>{0});
  NodeArg* total_seq_len = builder.MakeInput<int32_t>(std::vector<int64_t>{1}, std::vector<int32_t>{1});

  const std::vector<int64_t> present_shape{kBatch, kKvNumHeads, opts.PresentCacheLength(), kHeadSize};

  NodeArg* gqa_out = builder.MakeOutput<MLFloat16>(std::vector<int64_t>{kBatch, kSeq, kQHidden});
  NodeArg* present_key = builder.MakeOutput<MLFloat16>(present_shape);

  // present_value is either the graph output directly, or an intermediate that an Identity forwards
  // to the graph output.
  NodeArg* present_value = &empty_arg;
  NodeArg* identity_target = nullptr;
  if (!opts.no_present_value) {
    if (opts.present_value_behind_identity) {
      present_value = builder.MakeIntermediate<MLFloat16>(present_shape);
      identity_target = builder.MakeOutput<MLFloat16>(present_shape);
    } else {
      present_value = builder.MakeOutput<MLFloat16>(present_shape);
    }
  }

  std::vector<NodeArg*> gqa_inputs{query, key, value, past_key, past_value, seqlens_k, total_seq_len};

  Node& gqa = builder.AddNode("GroupQueryAttention",
                              gqa_inputs,
                              {gqa_out, present_key, present_value},
                              kMSDomain);
  gqa.AddAttribute("num_heads", static_cast<int64_t>(kNumHeads));
  gqa.AddAttribute("kv_num_heads", static_cast<int64_t>(kKvNumHeads));

  if (opts.four_bit_value_cache) {
    gqa.AddAttribute("v_quant_type", std::string("PER_CHANNEL"));
    gqa.AddAttribute("kv_cache_bit_width", static_cast<int64_t>(4));
  }

  if (identity_target != nullptr) {
    builder.AddNode("Identity", {present_value}, {identity_target});
  }
}

std::unique_ptr<GraphTransformer> MakeTransformer() {
  return std::make_unique<GqaValueLayoutTransformer>();
}

const std::vector<int64_t> kBnsh{kBatch, kKvNumHeads, kMaxSeq, kHeadSize};
const std::vector<int64_t> kBnhs{kBatch, kKvNumHeads, kHeadSize, kMaxSeq};

// ModelTestBuilder generates positional names ("input_3", "output_2"), so the checkers navigate the
// graph structurally instead of by name.
const Node* FindGqa(const Graph& graph) {
  for (const auto& node : graph.Nodes()) {
    if (node.OpType() == "GroupQueryAttention" && node.Domain() == kMSDomain) {
      return &node;
    }
  }
  return nullptr;
}

bool IsValueLayoutTranspose(const Node& node) {
  if (node.OpType() != "Transpose" || node.Domain() != kOnnxDomain) {
    return false;
  }
  const auto& attrs = node.GetAttributes();
  auto perm = attrs.find("perm");
  if (perm == attrs.end() || perm->second.ints_size() != 4) {
    return false;
  }
  return perm->second.ints(0) == 0 && perm->second.ints(1) == 1 &&
         perm->second.ints(2) == 3 && perm->second.ints(3) == 2;
}

Status ExpectShape(const NodeArg* arg, const std::vector<int64_t>& expected, const std::string& what) {
  ORT_RETURN_IF(arg == nullptr, what, " not found.");

  const auto* shape = arg->Shape();
  ORT_RETURN_IF(shape == nullptr, what, " ('", arg->Name(), "') has no shape.");
  ORT_RETURN_IF_NOT(static_cast<size_t>(shape->dim_size()) == expected.size(),
                    what, " ('", arg->Name(), "') has rank ", shape->dim_size(), ", expected ", expected.size(), ".");

  for (size_t i = 0; i < expected.size(); ++i) {
    const auto& dim = shape->dim(static_cast<int>(i));
    ORT_RETURN_IF_NOT(dim.has_dim_value() && dim.dim_value() == expected[i],
                      what, " ('", arg->Name(), "') dimension ", i, " is ",
                      dim.has_dim_value() ? std::to_string(dim.dim_value()) : dim.dim_param(),
                      ", expected ", expected[i], ".");
  }

  return Status::OK();
}

Status ExpectTransposeCount(const Graph& graph, int expected) {
  const auto op_to_count = CountOpsInGraph(graph);
  const int actual = OpCount(op_to_count, "Transpose");
  ORT_RETURN_IF_NOT(actual == expected, "Expected ", expected, " Transpose nodes, found ", actual, ".");
  ORT_RETURN_IF_NOT(OpCount(op_to_count, "com.microsoft.GroupQueryAttention") == 1,
                    "Expected the GroupQueryAttention node to be preserved.");
  return Status::OK();
}

Status ExpectNoTransposes(const Graph& graph) {
  return ExpectTransposeCount(graph, 0);
}

// Walks GQA input 4 back through the inserted Transpose to the graph input, asserting the operand
// stayed BNSH and the boundary became BNHS.
Status ExpectBnhsPastValue(const Graph& graph, const Node& gqa) {
  const NodeArg* operand = gqa.InputDefs()[4];
  ORT_RETURN_IF_ERROR(ExpectShape(operand, kBnsh, "GQA past_value operand"));

  const Node* transpose = graph.GetProducerNode(operand->Name());
  ORT_RETURN_IF(transpose == nullptr || !IsValueLayoutTranspose(*transpose),
                "GQA past_value is not produced by a Transpose(perm=[0,1,3,2]).");

  const NodeArg* boundary = transpose->InputDefs()[0];
  ORT_RETURN_IF_NOT(graph.IsInputsIncludingInitializers(boundary),
                    "past_value ('", boundary->Name(), "') must remain a graph input.");
  return ExpectShape(boundary, kBnhs, "past_value graph input");
}

// Mirror of the above for GQA output 2. cache_len differs from kMaxSeq for a prefill-only model,
// where GQA infers a present cache holding just the new tokens.
Status ExpectBnhsPresentValue(const Graph& graph, const Node& gqa, int64_t cache_len = kMaxSeq) {
  const std::vector<int64_t> bnsh{kBatch, kKvNumHeads, cache_len, kHeadSize};
  const std::vector<int64_t> bnhs{kBatch, kKvNumHeads, kHeadSize, cache_len};

  const NodeArg* operand = gqa.OutputDefs()[2];
  ORT_RETURN_IF_ERROR(ExpectShape(operand, bnsh, "GQA present_value operand"));

  const auto consumers = graph.GetConsumerNodes(operand->Name());
  ORT_RETURN_IF(consumers.size() != 1 || consumers[0] == nullptr || !IsValueLayoutTranspose(*consumers[0]),
                "GQA present_value is not consumed by exactly one Transpose(perm=[0,1,3,2]).");

  const NodeArg* boundary = consumers[0]->OutputDefs()[0];
  ORT_RETURN_IF_NOT(graph.IsOutput(boundary),
                    "present_value ('", boundary->Name(), "') must remain a graph output.");
  return ExpectShape(boundary, bnhs, "present_value graph output");
}

Status ExpectBnhsBoundary(Graph& graph) {
  ORT_RETURN_IF_ERROR(ExpectTransposeCount(graph, 2));

  const Node* gqa = FindGqa(graph);
  ORT_RETURN_IF(gqa == nullptr, "GroupQueryAttention node is missing.");

  ORT_RETURN_IF_ERROR(ExpectBnhsPastValue(graph, *gqa));
  ORT_RETURN_IF_ERROR(ExpectBnhsPresentValue(graph, *gqa));

  // The Key cache must be untouched: still wired straight to the graph boundary, still BNSH.
  ORT_RETURN_IF_NOT(graph.IsInputsIncludingInitializers(gqa->InputDefs()[3]),
                    "past_key must remain wired directly to the graph input.");
  ORT_RETURN_IF_ERROR(ExpectShape(gqa->InputDefs()[3], kBnsh, "past_key graph input"));
  ORT_RETURN_IF_NOT(graph.IsOutput(gqa->OutputDefs()[1]),
                    "present_key must remain wired directly to the graph output.");
  ORT_RETURN_IF_ERROR(ExpectShape(gqa->OutputDefs()[1], kBnsh, "present_key graph output"));

  return Status::OK();
}

// Serializes the default GQA model so an InferenceSession can load it, which is the only way to
// exercise the session option plumbing and the optimization-level behaviour.
Status BuildSerializedGqaModel(const logging::Logger& logger, std::string& model_bytes) {
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = 21;
  domain_to_version[kMSDomain] = 1;

  Model model("GqaValueLayoutTest", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {}, logger);
  Graph& graph = model.MainGraph();

  ModelTestBuilder helper(graph);
  BuildGqaModel(helper, BuildOptions{});
  helper.SetGraphOutputs();
  ORT_RETURN_IF_ERROR(graph.Resolve());

  ORT_RETURN_IF_NOT(model.ToProto().SerializeToString(&model_bytes), "Failed to serialize the test model.");
  return Status::OK();
}

}  // namespace

class GqaValueLayoutTransformerTest : public GraphTransformationTests {};

TEST_F(GqaValueLayoutTransformerTest, InsertsTransposesAndSwapsBoundaryShapes) {
  auto build = [](ModelTestBuilder& builder) { BuildGqaModel(builder, BuildOptions{}); };

  ASSERT_STATUS_OK(TestGraphTransformer(
      build, /*opset_version=*/21, *logger_, MakeTransformer(),
      TransformerLevel::Level1, /*steps=*/1,
      [](Graph& graph) { return ExpectNoTransposes(graph); },
      [](Graph& graph) { return ExpectBnhsBoundary(graph); }));
}

TEST_F(GqaValueLayoutTransformerTest, IsIdempotent) {
  auto build = [](ModelTestBuilder& builder) { BuildGqaModel(builder, BuildOptions{}); };

  // steps=2 runs the transformer twice. A second insertion would produce four Transposes and swap
  // the boundary shapes back to BNSH, so ExpectBnhsBoundary catches a missing idempotency guard.
  ASSERT_STATUS_OK(TestGraphTransformer(
      build, /*opset_version=*/21, *logger_, MakeTransformer(),
      TransformerLevel::Level1, /*steps=*/2,
      [](Graph& graph) { return ExpectNoTransposes(graph); },
      [](Graph& graph) { return ExpectBnhsBoundary(graph); }));
}

TEST_F(GqaValueLayoutTransformerTest, OutputSideOnlyWhenPastValueIsAbsent) {
  BuildOptions opts;
  opts.no_past_kv = true;
  auto build = [opts](ModelTestBuilder& builder) { BuildGqaModel(builder, opts); };

  ASSERT_STATUS_OK(TestGraphTransformer(
      build, /*opset_version=*/21, *logger_, MakeTransformer(),
      TransformerLevel::Level1, /*steps=*/1,
      [](Graph& graph) { return ExpectNoTransposes(graph); },
      [](Graph& graph) {
        ORT_RETURN_IF_ERROR(ExpectTransposeCount(graph, 1));
        const Node* gqa = FindGqa(graph);
        ORT_RETURN_IF(gqa == nullptr, "GroupQueryAttention node is missing.");
        return ExpectBnhsPresentValue(graph, *gqa, /*cache_len=*/kSeq);
      }));
}

TEST_F(GqaValueLayoutTransformerTest, InputSideOnlyWhenPresentValueIsAbsent) {
  BuildOptions opts;
  opts.no_present_value = true;
  auto build = [opts](ModelTestBuilder& builder) { BuildGqaModel(builder, opts); };

  ASSERT_STATUS_OK(TestGraphTransformer(
      build, /*opset_version=*/21, *logger_, MakeTransformer(),
      TransformerLevel::Level1, /*steps=*/1,
      [](Graph& graph) { return ExpectNoTransposes(graph); },
      [](Graph& graph) {
        ORT_RETURN_IF_ERROR(ExpectTransposeCount(graph, 1));
        const Node* gqa = FindGqa(graph);
        ORT_RETURN_IF(gqa == nullptr, "GroupQueryAttention node is missing.");
        return ExpectBnhsPastValue(graph, *gqa);
      }));
}

TEST_F(GqaValueLayoutTransformerTest, SkipsWhenPastValueIsNotAGraphInput) {
  BuildOptions opts;
  opts.past_value_behind_identity = true;
  auto build = [opts](ModelTestBuilder& builder) { BuildGqaModel(builder, opts); };

  ASSERT_STATUS_OK(TestGraphTransformer(
      build, /*opset_version=*/21, *logger_, MakeTransformer(),
      TransformerLevel::Level1, /*steps=*/1,
      [](Graph& graph) { return ExpectNoTransposes(graph); },
      [](Graph& graph) { return ExpectNoTransposes(graph); }));
}

TEST_F(GqaValueLayoutTransformerTest, SkipsWhenPresentValueIsNotAGraphOutput) {
  BuildOptions opts;
  opts.present_value_behind_identity = true;
  auto build = [opts](ModelTestBuilder& builder) { BuildGqaModel(builder, opts); };

  ASSERT_STATUS_OK(TestGraphTransformer(
      build, /*opset_version=*/21, *logger_, MakeTransformer(),
      TransformerLevel::Level1, /*steps=*/1,
      [](Graph& graph) { return ExpectNoTransposes(graph); },
      [](Graph& graph) { return ExpectNoTransposes(graph); }));
}

TEST_F(GqaValueLayoutTransformerTest, RejectsFourBitValueCache) {
  BuildOptions opts;
  opts.four_bit_value_cache = true;
  auto build = [opts](ModelTestBuilder& builder) { BuildGqaModel(builder, opts); };

  // Two 4-bit values are packed per byte along head_size, so a byte-wise Transpose cannot express
  // the layout change. Failing loudly beats silently producing wrong results on a non-fusing EP.
  ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(
      TestGraphTransformer(build, /*opset_version=*/21, *logger_, MakeTransformer(),
                           TransformerLevel::Level1, /*steps=*/1, nullptr, nullptr),
      "4-bit quantized Value cache");
}

// The transform changes the layout the session expects at its own inputs and outputs, so it is
// applied directly by TransformGraph rather than registered as a level 1 optimizer. This test pins
// that down: registered optimizers are skipped entirely at ORT_DISABLE_ALL.
TEST_F(GqaValueLayoutTransformerTest, AppliedWhenOptimizationsAreDisabled) {
  std::string model_bytes;
  ASSERT_STATUS_OK(BuildSerializedGqaModel(*logger_, model_bytes));

  SessionOptions session_options;
  session_options.graph_optimization_level = TransformerLevel::Default;  // ORT_DISABLE_ALL
  session_options.session_logid = "GqaValueLayoutTransformerTest";
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsGqaValueLayout, "BNHS"));

  InferenceSessionWrapper session{session_options, GetEnvironment()};
  ASSERT_STATUS_OK(session.Load(model_bytes.data(), static_cast<int>(model_bytes.size())));
  ASSERT_STATUS_OK(session.Initialize());

  ASSERT_STATUS_OK(ExpectBnhsBoundary(session.GetMutableGraph()));
}

TEST_F(GqaValueLayoutTransformerTest, NotAppliedForTheDefaultLayout) {
  std::string model_bytes;
  ASSERT_STATUS_OK(BuildSerializedGqaModel(*logger_, model_bytes));

  SessionOptions session_options;
  session_options.session_logid = "GqaValueLayoutTransformerTest";
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsGqaValueLayout, "BNSH"));

  InferenceSessionWrapper session{session_options, GetEnvironment()};
  ASSERT_STATUS_OK(session.Load(model_bytes.data(), static_cast<int>(model_bytes.size())));
  ASSERT_STATUS_OK(session.Initialize());

  ASSERT_STATUS_OK(ExpectNoTransposes(session.GetGraph()));
}

TEST_F(GqaValueLayoutTransformerTest, RejectsAnInvalidLayoutValue) {
  std::string model_bytes;
  ASSERT_STATUS_OK(BuildSerializedGqaModel(*logger_, model_bytes));

  SessionOptions session_options;
  session_options.session_logid = "GqaValueLayoutTransformerTest";
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsGqaValueLayout, "NHWC"));

  InferenceSessionWrapper session{session_options, GetEnvironment()};
  ASSERT_STATUS_OK(session.Load(model_bytes.data(), static_cast<int>(model_bytes.size())));
  ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(session.Initialize(), "Invalid value for session option");
}

#endif  // !defined(DISABLE_CONTRIB_OPS)

}  // namespace test
}  // namespace onnxruntime
