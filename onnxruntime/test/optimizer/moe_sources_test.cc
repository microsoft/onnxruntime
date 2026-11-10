// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS)
#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <sstream>

#include "gtest/gtest.h"
#include "core/framework/op_kernel.h"
#include "contrib_ops/cuda/fusion/moe_sources.h"
#include "contrib_ops/cuda/fusion/cuda_source_replacements.h"
#include "core/graph/contrib_ops/source_replacement_legacy.h"
#include "core/graph/graph_utils.h"
#include "core/graph/model.h"
#include "core/optimizer/cuda_runtime_fusion_internal.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"
#include "test/util/include/test_environment.h"

namespace onnxruntime::test {
namespace {
using namespace rewrite;
using contrib::cuda::fusion::internal::MoEStructuralCandidate;
using Clock = std::chrono::steady_clock;

std::string Chain(size_t nodes, size_t outputs = 1) {
  std::ostringstream text;
  text << "(x)=>{";
  for (size_t n = 0; n < nodes; ++n) {
    for (size_t slot = 1; slot < outputs; ++slot) text << "unused" << n << "_" << slot << ",";
    text << "v" << n << "=Identity(" << (n ? "v" + std::to_string(n - 1) : "x") << ");";
  }
  text << "return v" << nodes - 1 << "}";
  return text.str();
}

TEST(MoEPatternLimitsTest, DefaultsBoundsExpansionAndCopies) {
  EXPECT_NO_THROW((void)Pattern(Chain(64)));
  EXPECT_THROW((void)Pattern(Chain(65)), OnnxRuntimeException);
  EXPECT_THROW((void)Pattern(Chain(64, 4)), OnnxRuntimeException);  // 257 values, only 64 nodes.
  EXPECT_NO_THROW((void)Pattern(Chain(64, 4), PatternLimits{64, 257}));
  EXPECT_NO_THROW((void)Pattern(Chain(2048), PatternLimits{2048, 4096}));
  EXPECT_THROW((void)Pattern(Chain(2049), PatternLimits{2048, 4096}), OnnxRuntimeException);
  EXPECT_NO_THROW((void)Pattern(Chain(1365, 3), PatternLimits{2048, 4096}));
  EXPECT_THROW((void)Pattern(Chain(1024, 4), PatternLimits{2048, 4096}), OnnxRuntimeException);
  EXPECT_NO_THROW((void)Pattern(Chain(1), PatternLimits{1, 2}));
  EXPECT_THROW((void)Pattern(Chain(1), PatternLimits{1, 1}), OnnxRuntimeException);
  for (auto limits : {PatternLimits{0, 256}, PatternLimits{2049, 256},
                      PatternLimits{64, 0}, PatternLimits{64, 4097}}) {
    EXPECT_THROW((void)Pattern("(x)=>Identity(x)", limits), OnnxRuntimeException);
    EXPECT_THROW((void)PatternLibrary("pattern A(x){return Identity(x)}", limits), OnnxRuntimeException);
  }
  const std::string library = "pattern A(x)" + Chain(40).substr(5) +
                              " pattern B(x){a=A(x);b=A(a);return b}";
  EXPECT_THROW((void)PatternLibrary(library).GetPattern("B"), OnnxRuntimeException);
  auto copy = PatternLibrary(library, {80, 256}).GetPattern("B");
  Model model("expanded-limits", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              {{kOnnxDomain, 18}}, {}, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  ModelTestBuilder builder(graph);
  auto* value = builder.MakeInput<float>({2, 16}, 0.0f, 1.0f);
  NodeIndex root = 0;
  for (int n = 0; n < 80; ++n) {
    auto* next = builder.MakeIntermediate();
    root = builder.AddNode("Identity", {value}, {next}).Index();
    value = next;
  }
  graph.SetOutputs({value});
  ASSERT_STATUS_OK(graph.Resolve());
  std::optional<Match> match;
  ASSERT_STATUS_OK(Pattern(copy).MatchAt(graph, root, match));
  ASSERT_TRUE(match);
  EXPECT_EQ(match->nodes.size(), 80u);
}

TEST(MoEPatternLimitsTest, EnlargedMultiSinkRejectedAtParseAndDirectEntry) {
  std::string text = Chain(65);
  text.replace(text.find("return v64"), 10, "side=Identity(v0);return v64,side");
  EXPECT_THROW((void)Pattern(text, PatternLimits{128, 512}), OnnxRuntimeException);
  EXPECT_THROW((void)PatternLibrary("pattern A(x)" + text.substr(5), {128, 512}), OnnxRuntimeException);
  auto value_only = Chain(64, 4);
  value_only.replace(value_only.find("Identity(v62)"), 13, "Identity(v0)");
  try {
    (void)Pattern(value_only, {64, 512});
    FAIL() << "Enlarged value-only multi-sink pattern was accepted.";
  } catch (const OnnxRuntimeException& error) {
    EXPECT_NE(std::string(error.what()).find("producer-rooted"), std::string::npos);
  }
  // Deliberately corrupt a successfully parsed plan to exercise the independent
  // public matching-entry safeguard rather than relying solely on the parser.
  Pattern pattern(Chain(65), {128, 512});
  auto& nodes = const_cast<InlinedVector<Pattern::NodePattern, 1>&>(pattern.Nodes());
  nodes.back().inputs[0] = pattern.NamedValue("v0");
  Model model("limits", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              {{kOnnxDomain, 18}}, {}, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  ModelTestBuilder builder(graph);
  auto* x = builder.MakeInput<float>({2, 16}, 0.0f, 1.0f);
  auto* y = builder.MakeOutput();
  auto& root = builder.AddNode("Identity", {x}, {y});
  graph.SetOutputs({y});
  ASSERT_STATUS_OK(graph.Resolve());
  size_t visits = 0;
  EXPECT_THROW((void)pattern.VisitMatches(graph, root.Index(), [&](Match&) { ++visits; return true; }),
               OnnxRuntimeException);
  EXPECT_EQ(visits, 0u);
}

// Independently builds ONNX nodes, not a graph instantiated from the Rule's DSL.
struct MoEGraph {
  explicit MoEGraph(int experts, int k, std::string_view miss = {}, int64_t hidden = 16, int distractors = 0,
                    int64_t tokens = 2)
      : model("moe-structural", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              {{kOnnxDomain, 18}, {kMSDomain, 1}}, {}, DefaultLoggingManager().DefaultLogger()),
        graph(model.MainGraph()),
        builder(graph) {
    // Large token-count admission tests need metadata, never tensor allocations.
    x = tokens == 2 ? builder.MakeInput<float>({2, hidden}, 0.0f, 1.0f)
                    : builder.MakeInput<float>(std::vector<int64_t>{tokens, hidden});
    logits = tokens == 2 ? builder.MakeInput<float>({2, experts}, 0.0f, 1.0f)
                         : builder.MakeInput<float>(std::vector<int64_t>{tokens, experts});
    w1 = builder.MakeInitializer<float>({experts, 32, hidden}, 0.1f, 0.2f);
    w2 = builder.MakeInitializer<float>({experts, hidden, 16}, 0.1f, 0.2f);
    if (miss == "mutable") {
      graph.RemoveInitializedTensor(w1->Name());
      graph.SetInputs({x, logits, w1});
    } else {
      graph.SetInputs({x, logits});
    }
    auto* route = logits;
    if (miss == "routing_order") route = Op("Softmax", {logits}, {{"axis", int64_t{1}}})[0];
    auto selected = Op("TopK", {route, Ints({k})},
                       {{"axis", int64_t{1}}, {"largest", int64_t{1}}, {"sorted", int64_t{miss == "sorted" ? 0 : 1}}}, 2);
    auto* prob = miss == "routing_order" ? selected[0]
                                         : Op("Softmax", {selected[0]}, {{"axis", int64_t{1}}})[0];
    NodeArg* total = nullptr;
    NodeArg* exposed = nullptr;
    for (int e = 0; e < experts; ++e) {
      auto project = [&](NodeArg* bank) {
        auto* gathered = Op("Gather", {bank, Ints({e})}, {{"axis", int64_t{0}}})[0];
        auto* squeezed = Op("Squeeze", {gathered, Ints({0})})[0];
        return Op("Transpose", {squeezed}, {{"perm", std::vector<int64_t>{1, 0}}})[0];
      };
      auto* fc = Op("MatMul", {x, project(w1)})[0];
      auto halves = Op("Split", {fc}, {{"axis", int64_t{1}}, {"num_outputs", int64_t{2}}}, 2);
      // Reversing gate/up changes the activation even though every shape agrees.
      auto* gate = halves[miss == "gate_layout" && e == 0 ? 1 : 0];
      auto* sigmoid = Op(miss == "activation" && e == 0 ? "Relu" : "Sigmoid", {gate})[0];
      auto* swish = Op("Mul", {gate, sigmoid})[0];
      auto* act = Op("Mul", {swish, miss == "gate_layout" && e == 0 ? halves[0] : halves[1]})[0];
      auto* down = Op("MatMul", {act, project(w2)})[0];
      auto* eq = Op("Equal", {selected[1], Scalar(miss == "constant" && e == 0 ? 1 : e)})[0];
      auto* mask = Op("CastLike", {eq, prob})[0];
      auto* weighted = Op("Mul", {prob, mask})[0];
      auto* weight = Op("ReduceSum", {weighted, Ints({1})}, {{"keepdims", int64_t{1}}})[0];
      auto* out = Op("Mul", {down, weight})[0];
      if (e == 0) exposed = act;
      total = total ? Op("Add", miss == "accumulation" ? std::vector<NodeArg*>{out, total}
                                                       : std::vector<NodeArg*>{total, out})[0]
                    : out;
    }
    y = total;
    anchor = last;
    graph.SetOutputs({y});
    if (miss == "consumer") {
      auto* extra = Op("Identity", {exposed})[0];
      graph.SetOutputs({y, extra});
    }
    for (int d = 0; d < distractors; ++d) {
      auto* extra = Op("Add", {x, x})[0];
      auto outputs = graph.GetOutputs();
      outputs.push_back(extra);
      graph.SetOutputs(outputs);
    }
  }

  NodeArg* Ints(std::vector<int64_t> data) {
    return builder.MakeInitializer<int64_t>({static_cast<int64_t>(data.size())}, data);
  }
  NodeArg* Scalar(int64_t value) { return builder.MakeInitializer<int64_t>({}, {value}); }
  void DuplicateRegion() {
    // Independent computed values, shared boundary inputs and immutable constants.
    InlinedHashMap<const NodeArg*, NodeArg*> copied;
    InlinedVector<NodeIndex> originals;
    for (const auto& node : graph.Nodes()) originals.push_back(node.Index());
    for (auto index : originals) {
      const auto* node = graph.GetNode(index);
      std::vector<NodeArg*> inputs, outputs;
      for (const auto* input : node->InputDefs()) {
        auto found = copied.find(input);
        inputs.push_back(found == copied.end() ? graph.GetNodeArg(input->Name()) : found->second);
      }
      for (const auto* output : node->OutputDefs()) {
        auto* copy = &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("duplicate"), output->TypeAsProto());
        copied[output] = copy;
        outputs.push_back(copy);
      }
      auto& copy = builder.AddNode(node->OpType(), inputs, outputs);
      for (const auto& [name, attr] : node->GetAttributes()) copy.AddAttributeProto(attr);
    }
    graph.SetOutputs({y, copied.at(y)});
  }
  using Attribute = std::variant<int64_t, std::vector<int64_t>>;
  std::vector<NodeArg*> Op(const std::string& op, std::vector<NodeArg*> inputs,
                           std::initializer_list<std::pair<std::string, Attribute>> attributes = {}, size_t count = 1) {
    std::vector<NodeArg*> outputs;
    for (size_t i = 0; i < count; ++i) outputs.push_back(builder.MakeIntermediate());
    auto& node = builder.AddNode(op, inputs, outputs);
    for (const auto& [name, value] : attributes)
      std::visit([&](const auto& v) { node.AddAttribute(name, v); }, value);
    last = node.Index();
    return outputs;
  }
  Model model;
  Graph& graph;
  ModelTestBuilder builder;
  NodeArg *x, *logits, *w1, *w2, *y;
  NodeIndex anchor = 0;
  NodeIndex last = 0;
};

TEST(MoESourcesTest, RealRulesMatchRewriteAndScale) {
  for (int experts : {8, 16, 64}) {
    for (int k : {1, 2, experts}) {
      const auto begin = Clock::now();
      MoEStructuralCandidate candidate(experts, k);
      const auto constructed = Clock::now();
      MoEGraph source(experts, k, {}, 16, 128);
      ASSERT_STATUS_OK(source.graph.Resolve());
      std::optional<Match> match;
      const auto matching = Clock::now();
      ASSERT_STATUS_OK(candidate.Rule().pattern.MatchAt(source.graph, source.anchor, match));
      ASSERT_TRUE(match);
      ASSERT_TRUE(candidate.Rule().check_and_bind(source.graph, *match)) << match->RejectionReason();
      const auto matched = Clock::now();
      InlinedHashSet<NodeIndex> region;
      for (const auto* node : match->nodes) region.insert(node->Index());
      PatternRewriter rewriter({candidate.Rule()});
      Statistics statistics;
      ASSERT_STATUS_OK(rewriter.RunOnce(source.graph, statistics, region, source.anchor));
      const auto rewritten = Clock::now();
      ASSERT_EQ(statistics.rewrites, 1u);
      ASSERT_EQ(region.size(), 1u);
      const auto* target = source.graph.GetNode(*region.begin());
      ASSERT_NE(target, nullptr);
      EXPECT_EQ(target->OpType(), "MoE");
      EXPECT_EQ(target->Domain(), kMSDomain);
      ASSERT_EQ(target->InputDefs().size(), 5u);
      EXPECT_EQ(target->InputDefs()[0], source.x);
      EXPECT_EQ(target->InputDefs()[1], source.logits);
      EXPECT_EQ(target->InputDefs()[2], source.w1);
      EXPECT_FALSE(target->InputDefs()[3]->Exists());
      EXPECT_EQ(target->InputDefs()[4], source.w2);
      EXPECT_EQ(target->OutputDefs()[0], source.y);
      const auto& attrs = target->GetAttributes();
      ASSERT_EQ(attrs.size(), 7u);
      EXPECT_EQ(attrs.at("k").i(), k);
      EXPECT_EQ(attrs.at("activation_type").s(), "swiglu");
      EXPECT_EQ(attrs.at("swiglu_fusion").i(), 2);
      EXPECT_EQ(attrs.at("activation_alpha").f(), 1.0f);
      EXPECT_EQ(attrs.at("activation_beta").f(), 0.0f);
      EXPECT_EQ(attrs.at("normalize_routing_weights").i(), 1);
      EXPECT_EQ(attrs.at("use_sparse_mixer").i(), 0);
      EXPECT_EQ(attrs.count("swiglu_limit"), 0u);
      ASSERT_STATUS_OK(source.graph.Resolve());
      size_t targets = 0;
      for (const auto& node : source.graph.Nodes()) targets += node.OpType() == "MoE";
      EXPECT_EQ(targets, 1u);
      auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
      std::cout << "MOE_SCALING E=" << experts << " k=" << k << " nodes=" << candidate.Rule().pattern.Nodes().size()
                << " values=" << candidate.Rule().pattern.ValueCount() << " distractors=128 construction_ms="
                << ms(begin, constructed) << " match_ms=" << ms(matching, matched)
                << " rewrite_ms=" << ms(matched, rewritten) << '\n';
    }
  }
}

TEST(MoESourcesTest, StructuralAndMetadataNearMisses) {
  for (int experts : {8, 16, 64}) {
    MoEStructuralCandidate candidate(experts, 2);
    for (auto miss : {"routing_order", "sorted", "activation", "constant", "accumulation", "consumer",
                      "mutable", "alignment", "gate_layout"}) {
      SCOPED_TRACE(std::to_string(experts) + ":" + miss);
      MoEGraph source(experts, 2, miss, std::string_view(miss) == "alignment" ? 17 : 16);
      ASSERT_STATUS_OK(source.graph.Resolve());
      Statistics statistics;
      ASSERT_STATUS_OK(PatternRewriter({candidate.Rule()}).Run(source.graph, statistics));
      EXPECT_EQ(statistics.rewrites, 0u);
    }
    MoEGraph source(experts, 1);
    ASSERT_STATUS_OK(source.graph.Resolve());
    Statistics statistics;
    ASSERT_STATUS_OK(PatternRewriter({candidate.Rule()}).Run(source.graph, statistics));
    EXPECT_EQ(statistics.rewrites, 0u);  // Same E, wrong k literal.
  }
}

TEST(MoESourcesTest, ConstructionDomainAndProductionRegistration) {
  for (const auto& [e, k] : {std::pair{1, 1}, {65, 1}, {8, 0}, {8, 9}})
    EXPECT_THROW((void)MoEStructuralCandidate(e, k), OnnxRuntimeException);
  MoEStructuralCandidate lower(2, 2);
  MoEGraph source(2, 2);
  ASSERT_STATUS_OK(source.graph.Resolve());
  std::optional<Match> match;
  ASSERT_STATUS_OK(lower.Rule().pattern.MatchAt(source.graph, source.anchor, match));
  ASSERT_TRUE(match);
  EXPECT_TRUE(lower.Rule().check_and_bind(source.graph, *match)) << match->RejectionReason();
  InlinedVector<contrib::ExecutableSource> executables;
  for (const auto* production : contrib::cuda::fusion::CudaSourceReplacements()) {
    executables.push_back(production->Metadata());
  }
  EXPECT_EQ(contrib::ResolveSourceStatus({kMSDomain, "MoE", 1}, kCudaExecutionProvider, "moe_fp32_dense_swiglu_v1",
                                         contrib::GetSourcePolicies(), executables, contrib::LegacySourcePolicyKeys()),
            contrib::SourceStatus::Available);
}

class MoELookup final : public IExecutionProvider::IKernelLookup {
 public:
  mutable size_t calls = 0;
  bool supported = true;
  KernelCreateInfo kernel;
  const KernelCreateInfo* LookUpKernel(const Node& node) const override {
    ++calls;
    EXPECT_EQ(node.OpType(), "MoE");
    EXPECT_EQ(node.Domain(), kMSDomain);
    EXPECT_EQ(node.SinceVersion(), 1);
    EXPECT_EQ(node.InputDefs().size(), 5u);
    if (node.InputDefs().size() == 5) {
      EXPECT_FALSE(node.InputDefs()[3]->Exists());
      for (size_t slot : {0u, 1u, 2u, 4u})
        EXPECT_EQ(node.InputDefs()[slot]->TypeAsProto()->tensor_type().elem_type(),
                  ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    }
    EXPECT_EQ(node.OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type(),
              ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    EXPECT_EQ(node.GetAttributes().size(), 7u);
    EXPECT_EQ(node.GetAttributes().count("swiglu_limit"), 0u);
    EXPECT_EQ(node.GetAttributes().at("activation_type").s(), "swiglu");
    EXPECT_EQ(node.GetAttributes().at("swiglu_fusion").i(), 2);
    EXPECT_EQ(node.GetAttributes().at("activation_alpha").f(), 1.0f);
    EXPECT_EQ(node.GetAttributes().at("activation_beta").f(), 0.0f);
    EXPECT_EQ(node.GetAttributes().at("normalize_routing_weights").i(), 1);
    EXPECT_EQ(node.GetAttributes().at("use_sparse_mixer").i(), 0);
    EXPECT_GE(node.GetAttributes().at("k").i(), 1);
    EXPECT_LE(node.GetAttributes().at("k").i(), node.InputDefs()[1]->Shape()->dim(1).dim_value());
    return supported ? &kernel : nullptr;
  }
};

class MoEHostTest : public ::testing::Test {
 protected:
  SessionOptions options;
  CPUExecutionProvider cpu{CPUExecutionProviderInfo{}};
  MoELookup lookup;
  const logging::Logger& logger = DefaultLoggingManager().DefaultLogger();
  Status Select(const GraphViewer& viewer, std::vector<std::unique_ptr<ComputeCapability>>& offers,
                bool consumer = true) {
    GraphOptimizerRegistry registry(&options, &cpu, &logger, consumer);
    return cuda_runtime_fusion::Select(viewer, lookup, registry, "moe_fp32_dense_swiglu_v1", offers);
  }
  void Enable() { ASSERT_STATUS_OK(options.config_options.AddConfigEntry("ep.cuda.runtime_fusion.moe", "1")); }
  Status Accept(MoEGraph& f, ComputeCapability& offer) {
    GraphOptimizerRegistry registry(&options, &cpu, &logger);
    auto& selected = *offer.nodes_to_optimize[0];
    return selected.optimization_func(f.graph, selected, offer, registry);
  }
};

TEST_F(MoEHostTest, DefaultOffAndExplicitZeroDoNotLookup) {
  MoEGraph f(2, 1);
  ASSERT_STATUS_OK(f.graph.Resolve());
  const auto before = f.graph.ToGraphProto().SerializeAsString();
  std::vector<std::unique_ptr<ComputeCapability>> offers;
  ASSERT_STATUS_OK(Select(GraphViewer(f.graph), offers));
  EXPECT_TRUE(offers.empty());
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry("ep.cuda.runtime_fusion.moe", "0"));
  ASSERT_STATUS_OK(Select(GraphViewer(f.graph), offers));
  EXPECT_TRUE(offers.empty());
  EXPECT_EQ(lookup.calls, 0u);
  EXPECT_EQ(f.graph.ToGraphProto().SerializeAsString(), before);
}

TEST_F(MoEHostTest, RoutingGridTokenBoundaryUsesMetadataOnly) {
  Enable();
  constexpr int64_t max_tokens = int64_t{65535} * 1024;
  MoEStructuralCandidate candidate(2, 1);
  for (int64_t tokens : {max_tokens, max_tokens + 1}) {
    SCOPED_TRACE(tokens);
    MoEGraph f(2, 1, {}, 16, 0, tokens);
    ASSERT_STATUS_OK(f.graph.Resolve());
    const auto before = f.graph.ToGraphProto().SerializeAsString();
    std::optional<Match> match;
    ASSERT_STATUS_OK(candidate.Rule().pattern.MatchAt(f.graph, f.anchor, match));
    ASSERT_TRUE(match);
    const bool eligible = tokens == max_tokens;
    EXPECT_EQ(candidate.Rule().check_and_bind(f.graph, *match), eligible);
    if (!eligible)
      EXPECT_NE(match->RejectionReason().find("grid-Y"), std::string::npos);
    std::vector<std::unique_ptr<ComputeCapability>> offers;
    const auto previous_lookups = lookup.calls;
    ASSERT_STATUS_OK(Select(GraphViewer(f.graph), offers));
    EXPECT_EQ(offers.size(), eligible ? 1u : 0u);
    EXPECT_EQ(lookup.calls - previous_lookups, eligible ? 1u : 0u);
    EXPECT_EQ(f.graph.ToGraphProto().SerializeAsString(), before);
  }
}

size_t moe_factory_calls = 0;
gsl::span<const contrib::cuda::fusion::SourceReplacement> CountMoEFactory() {
  ++moe_factory_calls;
  return contrib::cuda::fusion::MoESources();
}

TEST_F(MoEHostTest, FactoryMaterializesOnlySelectedRuleAfterAdmission) {
  using namespace contrib::cuda::fusion;
  auto observed = MoEFactory();
  observed.materialize = CountMoEFactory;  // Instrument, then delegate the REAL factory.
  InlinedVector<const SourceReplacementFactory*> factories;
  for (const auto* factory : CudaSourceReplacementFactories())
    factories.push_back(factory->rule_id == observed.rule_id ? &observed : factory);
  moe_factory_calls = 0;
  MoEGraph f(2, 1);
  ASSERT_STATUS_OK(f.graph.Resolve());
  std::vector<std::unique_ptr<ComputeCapability>> offers;
  auto select = [&](const GraphViewer& viewer, bool consumer = true,
                    const std::string& rule = cuda_runtime_fusion::kMoERule) {
    GraphOptimizerRegistry registry(&options, &cpu, &logger, consumer);
    return cuda_runtime_fusion::detail::SelectFromFactories(factories, viewer, lookup, registry, rule, offers);
  };
  ASSERT_STATUS_OK(select(GraphViewer(f.graph)));  // Default off.
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(cuda_runtime_fusion::kMoEConfig, "0"));
  ASSERT_STATUS_OK(select(GraphViewer(f.graph)));
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(cuda_runtime_fusion::kMoEConfig, "invalid"));
  EXPECT_FALSE(select(GraphViewer(f.graph), false).IsOK());
  EXPECT_FALSE(select(GraphViewer(f.graph), true, "unknown").IsOK());
  Enable();
  ASSERT_STATUS_OK(select(GraphViewer(f.graph), false));
  options.graph_optimization_level = TransformerLevel::Level1;
  ASSERT_STATUS_OK(select(GraphViewer(f.graph)));
  options.graph_optimization_level = TransformerLevel::Level2;
  IndexedSubGraph filter;
  filter.nodes.push_back(f.anchor);
  auto meta = std::make_unique<IndexedSubGraph::MetaDef>();
  meta->name = "factory_filter";
  meta->domain = "test";
  meta->since_version = 1;
  meta->outputs = {f.y->Name()};
  filter.SetMetaDef(std::move(meta));
  ASSERT_STATUS_OK(select(GraphViewer(f.graph, filter)));
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(cuda_runtime_fusion::kUnitSwishConfig, "1"));
  ASSERT_STATUS_OK(select(GraphViewer(f.graph), true, cuda_runtime_fusion::kUnitSwishRule));
  EXPECT_TRUE(offers.empty());
  EXPECT_EQ(moe_factory_calls, 0u);
  EXPECT_EQ(lookup.calls, 0u);
  ASSERT_STATUS_OK(select(GraphViewer(f.graph)));
  EXPECT_EQ(moe_factory_calls, 1u);
  ASSERT_EQ(offers.size(), 1u);
  ASSERT_STATUS_OK(Accept(f, *offers[0]));
}

TEST_F(MoEHostTest, BatchedRootsPrepareOnceForAddHeavyGraph) {
  for (const char* miss : {"", "activation"}) {
    SCOPED_TRACE(miss);
    MoEGraph f(2, 1, miss);
    auto* tail = f.x;
    for (int i = 0; i < 128; ++i) tail = f.Op("Add", {tail, f.x})[0];
    f.graph.SetOutputs({f.y, tail});
    ASSERT_STATUS_OK(f.graph.Resolve());
    const auto before = f.graph.ToGraphProto().SerializeAsString();
    const GraphViewer viewer(f.graph);
    const auto roots = contrib::cuda::fusion::internal::CollectMoERoots(viewer);
    ASSERT_EQ(roots.size(), 1u);
    EXPECT_EQ(roots[0], f.anchor);
    MatchScanStatistics statistics;
    size_t matches = 0;
    const auto& pattern = contrib::cuda::fusion::MoESources()[0].body.rule.pattern;
    ASSERT_STATUS_OK(pattern.VisitMatches(f.graph, roots, [&](NodeIndex root, Match&) {
    EXPECT_EQ(root, f.anchor);
    ++matches;
    return true; }, &statistics));
    EXPECT_EQ(matches, std::string_view(miss).empty() ? 1u : 0u);
    EXPECT_EQ(statistics.compiled_plans, 63u);
    EXPECT_EQ(statistics.producer_index_builds, 1u);
    EXPECT_EQ(statistics.indexed_nodes, f.graph.NumberOfNodes());
    Enable();
    std::vector<std::unique_ptr<ComputeCapability>> offers;
    const auto previous_lookups = lookup.calls;
    ASSERT_STATUS_OK(Select(viewer, offers));
    EXPECT_EQ(offers.size(), matches);
    EXPECT_EQ(lookup.calls - previous_lookups, matches);
    EXPECT_EQ(f.graph.ToGraphProto().SerializeAsString(), before);
  }
}

TEST_F(MoEHostTest, UnrelatedAddsSkipMatcherPreparation) {
  Model model("unrelated-adds", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              {{kOnnxDomain, 18}, {kMSDomain, 1}}, {}, logger);
  auto& graph = model.MainGraph();
  ModelTestBuilder builder(graph);
  auto* x = builder.MakeInput<float>(std::vector<int64_t>{2, 16});
  auto* tail = x;
  for (int i = 0; i < 10000; ++i) {
    auto* next = builder.MakeIntermediate();
    builder.AddNode("Add", i % 2 ? std::vector<NodeArg*>{x, tail} : std::vector<NodeArg*>{tail, x}, {next});
    tail = next;
  }
  graph.SetOutputs({tail});
  ASSERT_STATUS_OK(graph.Resolve());
  const auto before = graph.ToGraphProto().SerializeAsString();
  const GraphViewer viewer(graph);
  const auto roots = contrib::cuda::fusion::internal::CollectMoERoots(viewer);
  EXPECT_TRUE(roots.empty());  // Right inputs have either no producer or an Add producer.
  MatchScanStatistics statistics;
  size_t visits = 0;
  const auto& pattern = contrib::cuda::fusion::MoESources()[0].body.rule.pattern;
  ASSERT_STATUS_OK(pattern.VisitMatches(graph, roots, [&](NodeIndex, Match&) {
    ++visits;
    return true; }, &statistics));
  EXPECT_EQ(visits, 0u);
  EXPECT_EQ(statistics.compiled_plans, 0u);
  EXPECT_EQ(statistics.producer_index_builds, 0u);
  EXPECT_EQ(statistics.indexed_nodes, 0u);
  Enable();
  std::vector<std::unique_ptr<ComputeCapability>> offers;
  ASSERT_STATUS_OK(Select(viewer, offers));
  EXPECT_TRUE(offers.empty());
  EXPECT_EQ(lookup.calls, 0u);
  EXPECT_EQ(graph.ToGraphProto().SerializeAsString(), before);
}

TEST_F(MoEHostTest, E2MatchOwnsOnlyConcretePlanAndAcceptsExactRegion) {
  MoEGraph f(2, 1);
  ASSERT_STATUS_OK(f.graph.Resolve());
  std::optional<Match> selected;
  {
    auto pattern = contrib::cuda::fusion::MoESources()[0].body.rule.pattern;
    ASSERT_STATUS_OK(pattern.MatchAt(f.graph, f.anchor, selected));
  }
  ASSERT_TRUE(selected);
  EXPECT_EQ(selected->bindings().alternative(), "e2");
  MatchScanStatistics statistics;
  size_t matches = 0;
  ASSERT_STATUS_OK(selected->pattern->VisitMatches(
      f.graph, gsl::span<const NodeIndex>(&f.anchor, 1), [&](NodeIndex, Match& match) {
        EXPECT_EQ(match.bindings().alternative(), "e2");
        ++matches;
        return true;
      },
      &statistics));
  EXPECT_EQ(matches, 1u);
  EXPECT_EQ(statistics.compiled_plans, 1u);
  Enable();
  std::vector<std::unique_ptr<ComputeCapability>> offers;
  ASSERT_STATUS_OK(Select(GraphViewer(f.graph), offers));
  ASSERT_EQ(offers.size(), 1u);
  EXPECT_EQ(offers[0]->sub_graph->nodes.size(), selected->nodes.size());
  ASSERT_STATUS_OK(Accept(f, *offers[0]));
  EXPECT_EQ(f.graph.NumberOfNodes(), 1);
}

TEST_F(MoEHostTest, BatchedScanHasLexicalLifetimeAndLegacyErrors) {
  MoEGraph f(2, 1);
  ASSERT_STATUS_OK(f.graph.Resolve());
  const Pattern add("(a,b)=>{ y=Add(a,b); return y }");
  InlinedVector<NodeIndex> roots{f.anchor, f.anchor};
  size_t visits = 0;
  ASSERT_STATUS_OK(add.VisitMatches(f.graph, roots, [&](NodeIndex root, Match&) {
    EXPECT_EQ(root, f.anchor);
    ++visits;
    return false;
  }));
  EXPECT_EQ(visits, 1u);
  EXPECT_FALSE(add.VisitMatches(f.graph, f.anchor, {}).IsOK());
  EXPECT_FALSE(add.VisitMatches(f.graph, f.graph.MaxNodeIndex(), [](Match&) { return true; }).IsOK());
  roots = {f.anchor, static_cast<NodeIndex>(f.graph.MaxNodeIndex())};
  EXPECT_FALSE(add.VisitMatches(f.graph, roots, [&](NodeIndex, Match&) { ++visits; return true; }).IsOK());
  EXPECT_EQ(visits, 1u);  // Validate all roots before invoking the visitor.
  MatchScanStatistics empty_statistics;
  roots.clear();
  ASSERT_STATUS_OK(add.VisitMatches(f.graph, roots, [](NodeIndex, Match&) { return true; }, &empty_statistics));
  EXPECT_EQ(empty_statistics.compiled_plans, 0u);
  EXPECT_EQ(empty_statistics.producer_index_builds, 0u);
  // A later call prepares afresh after mutation and Resolve; no index outlives a scan.
  const auto next = f.Op("Add", {f.y, f.x})[0];
  f.graph.SetOutputs({next});
  EXPECT_FALSE(add.VisitMatches(f.graph, f.last, [](Match&) { return true; }).IsOK());
  ASSERT_STATUS_OK(f.graph.Resolve());
  roots = {f.last};
  MatchScanStatistics statistics;
  ASSERT_STATUS_OK(add.VisitMatches(f.graph, roots, [&](NodeIndex root, Match&) {
    EXPECT_EQ(root, f.last);
    ++visits;
    return true; }, &statistics));
  EXPECT_EQ(visits, 2u);
  EXPECT_EQ(statistics.producer_index_builds, 1u);
  EXPECT_EQ(statistics.indexed_nodes, f.graph.NumberOfNodes());
}

TEST_F(MoEHostTest, MalformedLiteralKeepsThrowingDecoderBoundary) {
  Enable();
  MoEGraph f(2, 1);
  ASSERT_STATUS_OK(f.graph.Resolve());
  for (const auto& node : f.graph.Nodes()) {
    if (node.OpType() != "Squeeze") continue;
    auto* tensor = const_cast<ONNX_NAMESPACE::TensorProto*>(
        f.graph.GetConstantInitializer(node.InputDefs()[1]->Name(), false));
    tensor->clear_int64_data();
    tensor->set_raw_data("bad");
    break;
  }
  const auto before = f.graph.ToGraphProto().SerializeAsString();
  std::vector<std::unique_ptr<ComputeCapability>> offers;
  EXPECT_THROW((void)Select(GraphViewer(f.graph), offers), OnnxRuntimeException);
  EXPECT_TRUE(offers.empty());
  EXPECT_EQ(lookup.calls, 0u);
  EXPECT_EQ(f.graph.ToGraphProto().SerializeAsString(), before);
}

TEST_F(MoEHostTest, DeclarationGuardDrivesDiscoveryAndOwnedAcceptance) {
  MoEGraph f(2, 1);
  ASSERT_STATUS_OK(f.graph.Resolve());
  const auto catalog = contrib::cuda::fusion::CudaSourceReplacements();
  const auto found = std::find_if(catalog.begin(), catalog.end(),
                                  [](const auto* source) { return source->target.name == "MoE"; });
  ASSERT_NE(found, catalog.end());
  auto source = **found;
  const auto original_guard = source.body.rule.check_and_bind;
  source.body.rule.check_and_bind = [](const Graph&, Match&) { return false; };
  std::vector<std::unique_ptr<ComputeCapability>> offers;
  ASSERT_STATUS_OK(source.discover(source, GraphViewer(f.graph), lookup, logger, offers));
  EXPECT_TRUE(offers.empty());
  EXPECT_EQ(lookup.calls, 0u);
  size_t calls = 0;
  source.body.rule.check_and_bind = [original_guard, &calls](const Graph& graph, Match& match) {
    ++calls;
    return original_guard(graph, match);
  };
  ASSERT_STATUS_OK(source.discover(source, GraphViewer(f.graph), lookup, logger, offers));
  ASSERT_EQ(offers.size(), 1u);
  const auto discovery_calls = calls;
  source.body.rule.check_and_bind = [](const Graph&, Match&) { return false; };
  ASSERT_STATUS_OK(Accept(f, *offers[0]));
  EXPECT_GT(calls, discovery_calls);
}

TEST_F(MoEHostTest, EnabledProductionOffersAndAccepts) {
  Enable();
  for (int experts : {2, 8, 16, 64}) {
    for (int k : {1, 2, experts}) {
      MoEGraph f(experts, k);
      ASSERT_STATUS_OK(f.graph.Resolve());
      const auto before = f.graph.ToGraphProto().SerializeAsString();
      std::vector<std::unique_ptr<ComputeCapability>> offers;
      ASSERT_STATUS_OK(Select(GraphViewer(f.graph), offers));
      ASSERT_EQ(offers.size(), 1u);
      EXPECT_EQ(f.graph.ToGraphProto().SerializeAsString(), before);
      ASSERT_STATUS_OK(Accept(f, *offers[0]));
      ASSERT_EQ(f.graph.NumberOfNodes(), 1);
      const auto* target = f.graph.GetNode(offers[0]->sub_graph->nodes[0]);
      ASSERT_NE(target, nullptr);
      EXPECT_EQ(target->InputDefs()[0], f.x);
      EXPECT_EQ(target->InputDefs()[1], f.logits);
      EXPECT_EQ(target->InputDefs()[2], f.w1);
      EXPECT_EQ(target->InputDefs()[4], f.w2);
      EXPECT_EQ(target->OutputDefs()[0], f.y);
      EXPECT_EQ(target->GetAttributes().at("k").i(), k);
    }
  }
}

TEST_F(MoEHostTest, AdmissionAndMissingKernelLeaveGraphUnchanged) {
  MoEGraph f(8, 2);
  ASSERT_STATUS_OK(f.graph.Resolve());
  const auto before = f.graph.ToGraphProto().SerializeAsString();
  std::vector<std::unique_ptr<ComputeCapability>> offers;
  for (const char* value : {"", "true", "01", " 1", "2", "-1"}) {
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry("ep.cuda.runtime_fusion.moe", value));
    EXPECT_FALSE(Select(GraphViewer(f.graph), offers, false).IsOK());
    EXPECT_TRUE(offers.empty());
  }
  Enable();
  ASSERT_STATUS_OK(Select(GraphViewer(f.graph), offers, false));
  EXPECT_TRUE(offers.empty());
  options.graph_optimization_level = TransformerLevel::Level1;
  ASSERT_STATUS_OK(Select(GraphViewer(f.graph), offers));
  EXPECT_TRUE(offers.empty());
  options.graph_optimization_level = TransformerLevel::Level2;
  IndexedSubGraph filter;
  filter.nodes.push_back(f.anchor);
  auto meta = std::make_unique<IndexedSubGraph::MetaDef>();
  meta->name = "filtered_moe";
  meta->domain = "test";
  meta->since_version = 1;
  meta->outputs = {f.y->Name()};
  filter.SetMetaDef(std::move(meta));
  ASSERT_STATUS_OK(Select(GraphViewer(f.graph, filter), offers));
  EXPECT_TRUE(offers.empty());
  EXPECT_EQ(lookup.calls, 0u);
  lookup.supported = false;
  ASSERT_STATUS_OK(Select(GraphViewer(f.graph), offers));
  EXPECT_TRUE(offers.empty());
  EXPECT_EQ(lookup.calls, 1u);
  EXPECT_EQ(f.graph.ToGraphProto().SerializeAsString(), before);
}

TEST_F(MoEHostTest, ProductionNearMissesDeclineWithoutLookup) {
  Enable();
  for (const char* miss : {"routing_order", "sorted", "activation", "constant", "accumulation",
                           "consumer", "mutable", "alignment", "gate_layout"}) {
    SCOPED_TRACE(miss);
    MoEGraph f(8, 2, miss, std::string_view(miss) == "alignment" ? 17 : 16);
    ASSERT_STATUS_OK(f.graph.Resolve());
    const auto before = f.graph.ToGraphProto().SerializeAsString();
    std::vector<std::unique_ptr<ComputeCapability>> offers;
    ASSERT_STATUS_OK(Select(GraphViewer(f.graph), offers));
    EXPECT_TRUE(offers.empty());
    EXPECT_EQ(f.graph.ToGraphProto().SerializeAsString(), before);
  }
  EXPECT_EQ(lookup.calls, 0u);
}

TEST_F(MoEHostTest, SubgraphDeclinesWithoutLookup) {
  Enable();
  MoEGraph f(2, 1);
  ASSERT_STATUS_OK(f.graph.Resolve());
  auto proto = f.graph.ToGraphProto();
  proto.clear_input();  // If branches capture x/logits from the outer scope.
  Model outer("moe-subgraph", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              {{kOnnxDomain, 18}, {kMSDomain, 1}}, {}, logger);
  auto& graph = outer.MainGraph();
  ModelTestBuilder builder(graph);
  auto& x = graph.GetOrCreateNodeArg(f.x->Name(), f.x->TypeAsProto());
  auto& logits = graph.GetOrCreateNodeArg(f.logits->Name(), f.logits->TypeAsProto());
  auto* condition = builder.MakeInput<bool>(std::vector<int64_t>{}, std::vector<bool>{true});
  auto* output = builder.MakeOutput<float>(std::vector<int64_t>{2, 16});
  auto& branch = builder.AddNode("If", {condition}, {output});
  for (const char* name : {"then_branch", "else_branch"}) branch.AddAttribute(name, proto);
  graph.SetInputs({condition, &x, &logits});
  graph.SetOutputs({output});
  ASSERT_STATUS_OK(graph.Resolve());
  GraphViewer viewer(*branch.GetAttributeNameToMutableSubgraphMap().at("then_branch"));
  ASSERT_TRUE(viewer.IsSubgraph());
  std::vector<std::unique_ptr<ComputeCapability>> offers;
  ASSERT_STATUS_OK(Select(viewer, offers));
  EXPECT_TRUE(offers.empty());
  EXPECT_EQ(lookup.calls, 0u);
  auto observed = contrib::cuda::fusion::MoEFactory();
  observed.materialize = CountMoEFactory;
  const contrib::cuda::fusion::SourceReplacementFactory* factories[]{&observed};
  moe_factory_calls = 0;
  GraphOptimizerRegistry registry(&options, &cpu, &logger);
  ASSERT_STATUS_OK(cuda_runtime_fusion::detail::SelectFromFactories(
      factories, viewer, lookup, registry, cuda_runtime_fusion::kMoERule, offers));
  EXPECT_EQ(moe_factory_calls, 0u);
}

const NodeArg* RouteK(const MoEGraph& f) {
  for (const auto& node : f.graph.Nodes())
    if (node.OpType() == "TopK") return node.InputDefs()[1];
  return nullptr;
}

void SetInt64Control(ONNX_NAMESPACE::TensorProto& tensor, int64_t value) {
  tensor.clear_raw_data();
  tensor.clear_int64_data();
  tensor.add_int64_data(value);
}

TEST_F(MoEHostTest, InvalidDynamicAndMalformedRouteK) {
  Enable();
  for (int mutation = 0; mutation < 6; ++mutation) {
    SCOPED_TRACE(mutation);
    MoEGraph f(8, 2);
    ASSERT_STATUS_OK(f.graph.Resolve());
    const auto name = RouteK(f)->Name();
    auto* tensor = const_cast<ONNX_NAMESPACE::TensorProto*>(f.graph.GetConstantInitializer(name, false));
    if (mutation == 0) SetInt64Control(*tensor, 0);
    if (mutation == 1) SetInt64Control(*tensor, 9);
    if (mutation == 2) {
      f.graph.RemoveInitializedTensor(name);
      f.graph.SetInputs({f.x, f.logits, f.graph.GetNodeArg(name)});
      ASSERT_STATUS_OK(f.graph.Resolve());
    }
    if (mutation == 3) tensor->set_dims(0, 2);
    if (mutation == 4) {
      tensor->clear_int64_data();
      tensor->set_raw_data("bad");  // Exact header, malformed INT64 payload.
    }
    if (mutation == 5) tensor->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);
    const auto before = f.graph.ToGraphProto().SerializeAsString();
    std::vector<std::unique_ptr<ComputeCapability>> offers;
    if (mutation == 4) {
      InlinedVector<int64_t> data;
      bool available = false;
      const auto expected = ReadConstantTensor(f.graph, RouteK(f), data, available);
      ASSERT_FALSE(expected.IsOK());
      const auto actual = Select(GraphViewer(f.graph), offers);
      EXPECT_EQ(actual.Code(), expected.Code());
      EXPECT_EQ(actual.ErrorMessage(), expected.ErrorMessage());
    } else {
      ASSERT_STATUS_OK(Select(GraphViewer(f.graph), offers));
    }
    EXPECT_TRUE(offers.empty());
    EXPECT_EQ(f.graph.ToGraphProto().SerializeAsString(), before);
  }
  EXPECT_EQ(lookup.calls, 0u);
}

TEST_F(MoEHostTest, DiscoveryPublishesOnlyAfterAllRegionsSucceed) {
  Enable();
  MoEGraph f(2, 1);
  ASSERT_STATUS_OK(f.graph.Resolve());
  f.DuplicateRegion();
  auto* separate_k = f.Ints({1});
  for (auto& node : f.graph.Nodes()) {
    if (node.OpType() == "TopK" && node.Index() > f.anchor)
      graph_utils::ReplaceNodeInput(node, 1, *separate_k);
  }
  ASSERT_STATUS_OK(f.graph.Resolve());
  std::vector<std::unique_ptr<ComputeCapability>> offers;
  ASSERT_STATUS_OK(Select(GraphViewer(f.graph), offers));
  ASSERT_EQ(offers.size(), 2u);
  const NodeArg* later_k = nullptr;
  for (auto id : offers[1]->sub_graph->nodes) {
    const auto* node = f.graph.GetNode(id);
    if (node->OpType() == "TopK") later_k = node->InputDefs()[1];
  }
  ASSERT_NE(later_k, nullptr);
  auto* tensor = const_cast<ONNX_NAMESPACE::TensorProto*>(f.graph.GetConstantInitializer(later_k->Name(), false));
  tensor->clear_int64_data();
  tensor->set_raw_data("bad");
  const auto before = f.graph.ToGraphProto().SerializeAsString();
  offers.clear();
  lookup.calls = 0;
  EXPECT_FALSE(Select(GraphViewer(f.graph), offers).IsOK());
  EXPECT_TRUE(offers.empty());
  EXPECT_EQ(lookup.calls, 1u);  // The earlier region was probed but its offer was never published.
  EXPECT_TRUE(f.graph.ToGraphProto().SerializeAsString() == before);
}

TEST_F(MoEHostTest, ReadOnlyRepeatedRegionsFanoutAndSharedBanks) {
  Enable();
  MoEGraph f(8, 2);
  ASSERT_STATUS_OK(f.graph.Resolve());
  f.DuplicateRegion();
  auto* fanout = f.Op("Identity", {f.y})[0];
  auto outputs = f.graph.GetOutputs();
  outputs.push_back(fanout);
  f.graph.SetOutputs(outputs);
  ASSERT_STATUS_OK(f.graph.Resolve());
  const auto before = f.graph.ToGraphProto().SerializeAsString();
  const auto* bank1 = f.graph.GetConstantInitializer(f.w1->Name(), false);
  const auto* bank2 = f.graph.GetConstantInitializer(f.w2->Name(), false);
  std::vector<std::unique_ptr<ComputeCapability>> offers;
  ASSERT_STATUS_OK(Select(GraphViewer(f.graph), offers));
  ASSERT_EQ(offers.size(), 2u);
  EXPECT_EQ(f.graph.ToGraphProto().SerializeAsString(), before);
  for (auto& offer : offers) ASSERT_STATUS_OK(Accept(f, *offer));
  EXPECT_EQ(f.graph.NumberOfNodes(), 3);
  EXPECT_EQ(f.graph.GetConstantInitializer(f.w1->Name(), false), bank1);
  EXPECT_EQ(f.graph.GetConstantInitializer(f.w2->Name(), false), bank2);
  const auto* target = f.graph.GetProducerNode(f.y->Name());
  ASSERT_NE(target, nullptr);
  EXPECT_EQ(target->OpType(), "MoE");
  EXPECT_EQ(target->GetOutputEdgesCount(), 1u);
  EXPECT_EQ(target->InputDefs()[2], f.w1);
  EXPECT_EQ(target->InputDefs()[4], f.w2);
}

TEST_F(MoEHostTest, LearnedExternalBanksAreNotOpenedOrCopied) {
  Enable();
  MoEGraph f(2, 1);
  ASSERT_STATUS_OK(f.graph.Resolve());
  for (const auto* bank : {f.w1, f.w2}) {
    auto* tensor = const_cast<ONNX_NAMESPACE::TensorProto*>(f.graph.GetConstantInitializer(bank->Name(), false));
    tensor->clear_float_data();
    tensor->clear_raw_data();
    tensor->set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);
    auto* location = tensor->add_external_data();
    location->set_key("location");
    location->set_value("moe_learned_bank_must_not_open_missing_payload.bin");
  }
  const auto bank1 = f.graph.GetConstantInitializer(f.w1->Name(), false)->SerializeAsString();
  const auto bank2 = f.graph.GetConstantInitializer(f.w2->Name(), false)->SerializeAsString();
  std::vector<std::unique_ptr<ComputeCapability>> offers;
  ASSERT_STATUS_OK(Select(GraphViewer(f.graph), offers));
  ASSERT_EQ(offers.size(), 1u);
  ASSERT_STATUS_OK(Accept(f, *offers[0]));
  EXPECT_EQ(f.graph.GetConstantInitializer(f.w1->Name(), false)->SerializeAsString(), bank1);
  EXPECT_EQ(f.graph.GetConstantInitializer(f.w2->Name(), false)->SerializeAsString(), bank2);
}

TEST_F(MoEHostTest, StaleOffersRejectBeforeMutation) {
  Enable();
  for (int mutation = 0; mutation < 13; ++mutation) {
    SCOPED_TRACE(mutation);
    MoEGraph f(8, 2);
    ASSERT_STATUS_OK(f.graph.Resolve());
    std::vector<std::unique_ptr<ComputeCapability>> offers;
    ASSERT_STATUS_OK(Select(GraphViewer(f.graph), offers));
    ASSERT_EQ(offers.size(), 1u);
    auto& offer = *offers[0];
    auto& selected = *offer.nodes_to_optimize[0];
    if (mutation == 0) selected.sub_graph->nodes.pop_back();
    if (mutation == 1) offer.sub_graph->nodes.pop_back();
    if (mutation == 2) f.graph.GetNode(f.anchor)->SetExecutionProviderType(kCpuExecutionProvider);
    if (mutation == 3) f.x->ClearShape();
    if (mutation == 4) f.graph.GetNode(f.anchor)->SetSinceVersion(1);
    if (mutation == 5) f.graph.GetNode(f.anchor)->AddAttribute("unexpected", int64_t{1});
    if (mutation == 6) {
      auto* tensor = const_cast<ONNX_NAMESPACE::TensorProto*>(
          f.graph.GetConstantInitializer(RouteK(f)->Name(), false));
      SetInt64Control(*tensor, 1);  // Still eligible, but not the offered k.
    }
    if (mutation == 7) {
      auto* tensor = const_cast<ONNX_NAMESPACE::TensorProto*>(f.graph.GetConstantInitializer(f.w1->Name(), false));
      tensor->set_dims(0, 16);
    }
    if (mutation == 8) {
      auto type = *f.logits->TypeAsProto();
      type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16);
      ASSERT_STATUS_OK(f.logits->UpdateTypeAndShape(type, false, true, logger));
    }
    if (mutation == 9) {
      auto& imports = const_cast<std::unordered_map<std::string, int>&>(f.graph.DomainToVersionMap());
      imports[kOnnxDomain] = 19;
    }
    if (mutation == 10) {
      // Intermediate metadata remains typed/resolved, but must invalidate the offer.
      auto* value = f.graph.GetNodeArg(f.graph.GetNode(f.anchor)->InputDefs()[0]->Name());
      auto shape = *value->Shape();
      shape.mutable_dim(0)->set_dim_value(3);
      value->SetShape(shape);
    }
    if (mutation == 11) {
      for (const auto& node : f.graph.Nodes()) {
        if (node.OpType() != "Squeeze") continue;
        auto* tensor = const_cast<ONNX_NAMESPACE::TensorProto*>(
            f.graph.GetConstantInitializer(node.InputDefs()[1]->Name(), false));
        SetInt64Control(*tensor, 1);
        break;
      }
    }
    if (mutation == 12) f.graph.RemoveNode(f.anchor);
    const auto before = f.graph.ToGraphProto().SerializeAsString();
    EXPECT_FALSE(Accept(f, offer).IsOK());
    EXPECT_TRUE(f.graph.ToGraphProto().SerializeAsString() == before);
  }
}

TEST(MoESourcesTest, RepeatedRegionsShareBanksWithoutCloningThem) {
  for (int experts : {8, 16, 64}) {
    MoEGraph source(experts, 2);
    ASSERT_STATUS_OK(source.graph.Resolve());
    source.DuplicateRegion();
    ASSERT_STATUS_OK(source.graph.Resolve());
    const auto initializer_count = source.graph.GetAllInitializedTensors().size();
    MoEStructuralCandidate candidate(experts, 2);
    Statistics statistics;
    ASSERT_STATUS_OK(PatternRewriter({candidate.Rule()}).Run(source.graph, statistics));
    ASSERT_EQ(statistics.rewrites, 2u);
    ASSERT_STATUS_OK(source.graph.Resolve());
    size_t targets = 0;
    for (const auto& node : source.graph.Nodes()) {
      if (node.OpType() != "MoE") continue;
      ++targets;
      EXPECT_EQ(node.InputDefs()[2], source.w1);
      EXPECT_EQ(node.InputDefs()[4], source.w2);
    }
    EXPECT_EQ(targets, 2u);
    EXPECT_LE(source.graph.GetAllInitializedTensors().size(), initializer_count);
  }
}

TEST(MoESourcesTest, StaleMetadataRecheckedByRunOnceWithoutMutation) {
  for (int experts : {8, 16, 64}) {
    MoEStructuralCandidate candidate(experts, 2);
    PatternRewriter rewriter({candidate.Rule()});
    for (int mutation = 0; mutation < 10; ++mutation) {
      SCOPED_TRACE(std::to_string(experts) + ":" + std::to_string(mutation));
      MoEGraph source(experts, 2);
      ASSERT_STATUS_OK(source.graph.Resolve());
      std::optional<Match> match;
      ASSERT_STATUS_OK(candidate.Rule().pattern.MatchAt(source.graph, source.anchor, match));
      ASSERT_TRUE(match);
      ASSERT_TRUE(candidate.Rule().check_and_bind(source.graph, *match)) << match->RejectionReason();
      InlinedHashSet<NodeIndex> region;
      for (const auto* node : match->nodes) region.insert(node->Index());
      if (mutation == 0) source.x->ClearShape();
      if (mutation == 1) {
        auto shape = *source.w2->Shape();
        shape.mutable_dim(2)->clear_dim_value();
        shape.mutable_dim(2)->set_dim_param("unknown");
        source.w2->SetShape(shape);
      }
      if (mutation == 2 || mutation == 4 || mutation == 5 || mutation == 6) {
        auto* value = source.x;
        if (mutation == 2) value = source.w1;
        if (mutation == 4) value = source.logits;
        if (mutation == 5) value = source.y;
        auto shape = *value->Shape();
        shape.mutable_dim(0)->set_dim_value(mutation == 6 ? std::numeric_limits<int64_t>::max() : 3);
        value->SetShape(shape);
      }
      if (mutation == 3) {
        auto type = *source.x->TypeAsProto();
        type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16);
        ASSERT_STATUS_OK(source.x->UpdateTypeAndShape(type, false, true, DefaultLoggingManager().DefaultLogger()));
      }
      if (mutation == 7) source.graph.GetNodeArg(match->nodes[0]->OutputDefs()[0]->Name())->ClearShape();
      if (mutation == 8) source.graph.GetNode(match->nodes[0]->Index())->SetSinceVersion(1);
      if (mutation == 9) {
        auto& imports = const_cast<std::unordered_map<std::string, int>&>(source.graph.DomainToVersionMap());
        imports[kOnnxDomain] = 19;
      }
      EXPECT_FALSE(candidate.Rule().check_and_bind(source.graph, *match));
      const auto before = source.graph.ToGraphProto().SerializeAsString();
      const auto original_region = region;
      Statistics statistics;
      EXPECT_FALSE(rewriter.RunOnce(source.graph, statistics, region, source.anchor).IsOK());
      EXPECT_EQ(statistics.rewrites, 0u);
      EXPECT_EQ(region, original_region);
      EXPECT_EQ(source.graph.ToGraphProto().SerializeAsString(), before);
    }
  }
}
}  // namespace
}  // namespace onnxruntime::test
#endif
