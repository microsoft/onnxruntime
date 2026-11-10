// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "contrib_ops/cuda/fusion/moe_sources.h"

#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS)
#include <algorithm>
#include <array>
#include <limits>
#include <sstream>
#include <string_view>

#include "contrib_ops/cuda/fusion/cuda_source_replacements.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/optimizer/cuda_runtime_fusion_internal.h"

namespace onnxruntime::contrib::cuda::fusion::internal {
namespace {
using namespace rewrite;

std::string Topology(int experts, std::string_view route_k, bool alias_output) {
  std::ostringstream text;
  text << "{selected,indices=TopK(logits," << route_k
       << ",axis=1,largest=1,sorted=1);prob=Softmax(selected,axis=1);";
  for (int e = 0; e < experts; ++e) {
    const auto s = std::to_string(e);
    text << "a" << s << "=Gather(w1,[" << e << "],axis=0);"
         << "b" << s << "=Squeeze(a" << s << ",[0]);"
         << "c" << s << "=Transpose(b" << s << ",perm=[1,0]);"
         << "d" << s << "=MatMul(x,c" << s << ");"
         << "gate" << s << ",up" << s << "=Split(d" << s << ",axis=1,num_outputs=2);"
         << "sig" << s << "=Sigmoid(gate" << s << ");"
         << "swish" << s << "=Mul(gate" << s << ",sig" << s << ");"
         << "act" << s << "=Mul(swish" << s << ",up" << s << ");"
         << "f" << s << "=Gather(w2,[" << e << "],axis=0);"
         << "g" << s << "=Squeeze(f" << s << ",[0]);"
         << "h" << s << "=Transpose(g" << s << ",perm=[1,0]);"
         << "down" << s << "=MatMul(act" << s << ",h" << s << ");"
         << "eq" << s << "=Equal(indices," << e << ");"
         << "mask" << s << "=CastLike(eq" << s << ",prob);"
         << "weighted" << s << "=Mul(prob,mask" << s << ");"
         << "weight" << s << "=ReduceSum(weighted" << s << ",[1],keepdims=1,noop_with_empty_axes=0);"
         << "out" << s << "=Mul(down" << s << ",weight" << s << ");";
    if (e > 0) text << "sum" << s << "=Add(" << (e == 1 ? "out0" : "sum" + std::to_string(e - 1))
                    << ",out" << s << ");";
  }
  text << "return " << (alias_output ? "(y=" : "") << "sum" << experts - 1
       << (alias_output ? ")" : "") << "}";
  return text.str();
}

bool Shape(const NodeArg* value, gsl::span<const int64_t> expected) {
  if (!value || !value->Shape() || value->Shape()->dim_size() != static_cast<int>(expected.size())) return false;
  for (size_t i = 0; i < expected.size(); ++i) {
    const auto& dim = value->Shape()->dim(static_cast<int>(i));
    if (!dim.has_dim_value() || dim.dim_value() != expected[i]) return false;
  }
  return true;
}

bool Semantics(const Graph& graph, Match& match, int experts, int k) {
  const auto& imports = graph.DomainToVersionMap();
  if (!imports.count(kOnnxDomain) || imports.at(kOnnxDomain) != 18 ||
      !imports.count(kMSDomain) || imports.at(kMSDomain) != 1) return match.Reject("MoE requires ONNX-18 and MS-1 imports.");
  const auto* x = match.valueOf("x");
  const auto* w1 = match.valueOf("w1");
  const auto* w2 = match.valueOf("w2");
  if (!x->Shape() || x->Shape()->dim_size() != 2 || !w2->Shape() || w2->Shape()->dim_size() != 3)
    return match.Reject("MoE requires resolved rank-two input and rank-three banks.");
  const int64_t n = x->Shape()->dim(0).dim_value();
  const int64_t h = x->Shape()->dim(1).dim_value();
  const int64_t i = w2->Shape()->dim(2).dim_value();
  // The sort-first routing fallback launches blockExpertPrefixSum with
  // grid.y = ceil(N / 1024), bounded by CUDA's grid-Y limit.
  constexpr int64_t max_routing_tokens = int64_t{65535} * 1024;
  if (n > max_routing_tokens)
    return match.Reject("MoE token count exceeds the CUDA routing grid-Y bound.");
  constexpr int64_t bound = std::numeric_limits<int>::max();
  if (n <= 0 || h < 16 || i < 16 || h % 16 || i % 16 || h > bound || i > bound / 2 ||
      n > bound / h || n > bound / (2 * i) || n > bound / experts ||
      h > bound / (2 * i) / experts || k > experts)
    return match.Reject("MoE dimensions exceed aligned CUDA integer bounds.");
  if (!Shape(x, InlinedVector<int64_t>{n, h}) ||
      !Shape(match.valueOf("logits"), InlinedVector<int64_t>{n, experts}) ||
      !Shape(w1, InlinedVector<int64_t>{experts, 2 * i, h}) ||
      !Shape(w2, InlinedVector<int64_t>{experts, h, i}) ||
      !Shape(match.values[match.pattern->Outputs()[0].handle], InlinedVector<int64_t>{n, h}))
    return match.Reject("MoE boundary shapes are unknown or incompatible.");
  for (const auto* bank : {w1, w2}) {
#if !defined(DISABLE_SPARSE_TENSORS)
    if (graph.IsSparseInitializer(bank->Name())) return match.Reject("MoE requires dense expert banks.");
#endif
    const auto* initializer = graph.GetConstantInitializer(bank->Name(), false);
    if (!initializer || initializer->data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
        initializer->dims_size() != 3) return match.Reject("MoE requires immutable FP32 expert banks.");
    for (int d = 0; d < 3; ++d)
      if (initializer->dims(d) != bank->Shape()->dim(d).dim_value())
        return match.Reject("MoE initializer shape disagrees with metadata.");
  }
  // Constants are validated by the matcher; all computed data must have resolved
  // tensor metadata. Do not infer missing types from a neighboring value.
  for (const auto* node : match.nodes) {
    if (!node->Overload().empty() || !node->GetExecutionProviderType().empty())
      return match.Reject("MoE structural candidate requires ordinary unassigned ONNX nodes.");
    const auto& op = node->OpType();
    int version = 13;
    if (op == "TopK") version = 11;
    if (op == "CastLike") version = 15;
    if (op == "Split") version = 18;
    if (op == "Add" || op == "Mul") version = 14;
    if (node->SinceVersion() != version) return match.Reject("MoE source has an unexpected schema version.");
    size_t attributes = 0;
    if (op == "TopK")
      attributes = 3;
    else if (op == "Split" || op == "ReduceSum")
      attributes = 2;
    else if (op == "Gather" || op == "Softmax" || op == "Transpose")
      attributes = 1;
    if (node->GetAttributes().size() != attributes)
      return match.Reject("MoE source has unexpected operator attributes: " + op);
    for (size_t slot = 0; slot < node->OutputDefs().size(); ++slot) {
      const auto* value = node->OutputDefs()[slot];
      const auto* type = value->TypeAsProto();
      const int expected = op == "Equal"               ? ONNX_NAMESPACE::TensorProto_DataType_BOOL
                           : op == "TopK" && slot == 1 ? ONNX_NAMESPACE::TensorProto_DataType_INT64
                                                       : ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
      if (!type || !type->has_tensor_type() || type->tensor_type().elem_type() != expected || !value->Shape())
        return match.Reject("MoE source lacks computed tensor metadata: " + op);
    }
  }
  for (const auto* value : {x, match.valueOf("logits"), w1, w2}) {
    const auto* type = value->TypeAsProto();
    if (!type || !type->has_tensor_type() ||
        type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT)
      return match.Reject("MoE requires FP32 boundary tensors.");
  }
  return true;
}

rewrite::Rule MakeRule(int experts, int k) {
  ORT_ENFORCE(experts >= 2 && experts <= 64 && k >= 1 && k <= experts,
              "MoE structural candidate requires 2<=experts<=64 and 1<=k<=experts.");
  std::ostringstream replacement;
  replacement << "(x,logits,w1,w2)=>com.microsoft.MoE(x,logits,w1,None,w2,"
              << "activation_type=\"swiglu\",swiglu_fusion=2,activation_alpha=1.0,activation_beta=0.0,"
              << "normalize_routing_weights=1,use_sparse_mixer=0,k=" << k << ")";
  return Rule{"CUDA_MoE_structural_" + std::to_string(experts) + "_" + std::to_string(k),
              Pattern("(x,logits,w1,w2)=>" + Topology(experts, "[" + std::to_string(k) + "]", false), {2048, 4096}),
              replacement.str(),
              [experts, k](const Graph& graph, Match& match) { return Semantics(graph, match, experts, k); },
              {{kOnnxDomain, 18, 18}, {kMSDomain, 1, 1}}};
}
}  // namespace

MoEStructuralCandidate::MoEStructuralCandidate(int experts, int k) : rule_(MakeRule(experts, k)) {}

InlinedVector<NodeIndex> CollectMoERoots(const GraphViewer& viewer) {
  InlinedVector<NodeIndex> roots;
  const auto& graph = viewer.GetGraph();
  for (auto index : viewer.GetNodesInTopologicalOrder()) {
    const auto* node = graph.GetNode(index);
    if (!node || node->Domain() != kOnnxDomain || node->OpType() != "Add" ||
        !node->GetExecutionProviderType().empty()) continue;
    const auto& inputs = node->InputDefs();
    if (inputs.size() != 2 || !inputs[1] || !inputs[1]->Exists()) continue;
    // Every declared alternative ends with Add(accumulated, weighted_expert).
    // This necessary check avoids full matcher state for unrelated Add roots.
    const auto* right = graph.GetProducerNode(inputs[1]->Name());
    if (right && right->Domain() == kOnnxDomain && right->OpType() == "Mul")
      roots.push_back(index);
  }
  return roots;
}
}  // namespace onnxruntime::contrib::cuda::fusion::internal

namespace onnxruntime::contrib::cuda::fusion {
namespace {
using namespace rewrite;
using cuda_runtime_fusion::detail::MakeCapability;
using cuda_runtime_fusion::detail::OptimizationFunction;
using cuda_runtime_fusion::detail::ProbeTarget;

int Experts(const Match& match) {
  const auto tag = match.bindings().alternative();
  for (int e = 2; e <= 64; ++e)
    if (tag == "e" + std::to_string(e)) return e;
  return 0;
}

ONNX_NAMESPACE::AttributeProto KAttribute(int64_t k) {
  ONNX_NAMESPACE::AttributeProto attr;
  attr.set_name("k");
  attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  attr.set_i(k);
  return attr;
}

// Unlike the literal topology controls, route_k is a captured boundary input.
// Require an immutable initializer and exact headers before materializing its one value.
Status ReadRouteK(const Graph& graph, Match& match, int64_t& k, bool& eligible) {
  eligible = false;
  const int experts = Experts(match);
  const auto* value = match.valueOf("route_k");
  if (!experts || !internal::Shape(value, InlinedVector<int64_t>{1})) return Status::OK();
  const auto* type = value->TypeAsProto();
  if (!type || !type->has_tensor_type() ||
      type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_INT64) return Status::OK();
#if !defined(DISABLE_SPARSE_TENSORS)
  if (graph.IsSparseInitializer(value->Name())) return Status::OK();
#endif
  const auto* tensor = graph.GetConstantInitializer(value->Name(), false);
  if (!tensor || tensor->data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT64 ||
      tensor->dims_size() != 1 || tensor->dims(0) != 1) return Status::OK();
  InlinedVector<int64_t> data;
  bool available = false;
  ORT_RETURN_IF_ERROR(ReadConstantTensor(graph, value, data, available));
  if (!available || data.size() != 1 || data[0] < 1 || data[0] > experts) return Status::OK();
  k = data[0];
  eligible = true;
  return Status::OK();
}

bool MoESemantics(const Graph& graph, Match& match) {
  int64_t k = 0;
  bool eligible = false;
  ORT_THROW_IF_ERROR(ReadRouteK(graph, match, k, eligible));
  return eligible && internal::Semantics(graph, match, Experts(match), static_cast<int>(k)) &&
         match.BindAttribute("k", KAttribute(k));
}

// Saved graph pointers are identity tokens only, never dereferenced. All metadata
// and decoded controls are owned; no borrowed Match or BindingView survives discovery.
struct MoESnapshot {
  InlinedVector<NodeIndex> ids;
  InlinedVector<const Node*> nodes;
  InlinedVector<const NodeArg*> values;
  InlinedVector<std::string> metadata;
  InlinedVector<InlinedVector<int64_t>> controls;
  std::array<const NodeArg*, 5> boundary{};
  std::array<std::string, 5> names;

  bool operator==(const MoESnapshot& other) const {
    return ids == other.ids && nodes == other.nodes && values == other.values &&
           metadata == other.metadata && controls == other.controls &&
           boundary == other.boundary && names == other.names;
  }
};

Status Snapshot(const Graph& graph, const Match& match, MoESnapshot& snapshot) {
  MoESnapshot actual;
  for (const auto* node : match.nodes) {
    actual.ids.push_back(node->Index());
    actual.nodes.push_back(node);
    ONNX_NAMESPACE::NodeProto proto;
    node->ToProto(proto);
    proto.clear_attribute();
    actual.metadata.push_back(proto.SerializeAsString());
    actual.metadata.push_back(std::to_string(node->SinceVersion()));
    InlinedVector<std::pair<std::string, std::string>> attributes;
    for (const auto& [name, attr] : node->GetAttributes())
      attributes.emplace_back(name, attr.SerializeAsString());
    std::sort(attributes.begin(), attributes.end());
    for (const auto& [name, serialized] : attributes) {
      actual.metadata.push_back(name);
      actual.metadata.push_back(serialized);
    }
  }
  std::sort(actual.ids.begin(), actual.ids.end());
  for (const auto* value : match.values) {
    actual.values.push_back(value);
    actual.metadata.push_back(value ? value->Name() : "");
    actual.metadata.push_back(value && value->TypeAsProto() ? value->TypeAsProto()->SerializeAsString() : "");
    if (!value) continue;
    ONNX_NAMESPACE::TensorProto header;
    if (const auto* tensor = graph.GetConstantInitializer(value->Name(), false)) {
      // Construct a header, never copy a TensorProto and then clear its learned payload.
      header.set_name(tensor->name());
      header.set_data_type(tensor->data_type());
      for (auto dim : tensor->dims()) header.add_dims(dim);
      header.set_data_location(tensor->data_location());
      if (tensor->has_segment()) *header.mutable_segment() = tensor->segment();
      for (const auto& entry : tensor->external_data()) *header.add_external_data() = entry;
    }
    actual.metadata.push_back(header.SerializeAsString());
    InlinedVector<int64_t> control;
    bool available = false;
    const auto* type = value->TypeAsProto();
    if (type && type->has_tensor_type() &&
        type->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
      // Controls can materialize external storage. Learned FP32 banks never enter this reader.
      ORT_RETURN_IF_ERROR(ReadConstantTensor(graph, value, control, available));
    }
    actual.metadata.push_back(available ? "constant" : "computed");
    actual.controls.push_back(std::move(control));
  }
  size_t slot = 0;
  for (const char* name : {"x", "logits", "w1", "w2", "y"}) {
    const auto* value = match.valueOf(name);
    actual.boundary[slot] = value;
    actual.names[slot++] = value->Name();
  }
  actual.metadata.push_back(match.attributeOf("k").SerializeAsString());
  snapshot = std::move(actual);
  return Status::OK();
}

Status TargetAvailable(const Match& match, const Replacement& recipe,
                       const IExecutionProvider::IKernelLookup& lookup,
                       const logging::Logger& logger, bool& available) {
  Model scratch("cuda_moe_probe", false, ModelMetaData(), PathString(),
                IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 18}, {kMSDomain, 1}}, {}, logger);
  auto& graph = scratch.MainGraph();
  Replacement::TensorBindings bindings;
  InlinedVector<const NodeArg*> inputs;
  for (const char* name : {"x", "logits", "w1", "w2"}) {
    auto& input = graph.GetOrCreateNodeArg(name, match.valueOf(name)->TypeAsProto());
    bindings.emplace(name, &input);
    inputs.push_back(&input);
  }
  graph.SetInputs(inputs);
  return ProbeTarget(graph, recipe, bindings, {{"k", match.attributeOf("k")}}, 1, lookup, available);
}

std::unique_ptr<ComputeCapability> OfferMoE(const Match& match, NodeIndex anchor, MoESnapshot snapshot,
                                            const SourceBody& body) {
  // Own only the selected declaration alternative in each offer, not all 63 plans.
  const auto pattern = *match.pattern;
  const auto guard = body.rule.check_and_bind;
  auto rule = body.rule;
  rule.pattern = pattern;
  rule.check_and_bind = [guard, snapshot](const Graph& graph, Match& candidate) {
    if (!guard(graph, candidate)) return false;
    MoESnapshot actual;
    ORT_THROW_IF_ERROR(Snapshot(graph, candidate, actual));
    return actual == snapshot;
  };
  Options options;
  options.compatible_execution_providers.clear();
  auto rewriter = std::make_shared<const PatternRewriter>(Rules{std::move(rule)}, std::move(options));
  OptimizationFunction optimization =
      [rewriter, pattern, guard, anchor, snapshot](Graph& graph, const ComputeCapability& selected,
                                                   ComputeCapability& capability,
                                                   const GraphOptimizerRegistry&) -> Status {
    const InlinedHashSet<NodeIndex> expected(snapshot.ids.begin(), snapshot.ids.end());
    for (const auto* region : {selected.sub_graph.get(), capability.sub_graph.get()}) {
      ORT_RETURN_IF_NOT(region && region->nodes.size() == snapshot.ids.size() &&
                            InlinedHashSet<NodeIndex>(region->nodes.begin(), region->nodes.end()) == expected,
                        "MoE selected region changed.");
    }
    for (auto id : snapshot.ids) {
      const auto* node = graph.GetNode(id);
      ORT_RETURN_IF_NOT(node && node->GetExecutionProviderType().empty(),
                        "MoE source is missing or already assigned.");
    }
    ORT_RETURN_IF(graph.GraphResolveNeeded(), "PatternRewriter requires an already resolved graph.");
    std::optional<Match> candidate;
    ORT_RETURN_IF_ERROR(pattern.MatchAt(graph, anchor, candidate));
    ORT_RETURN_IF_NOT(candidate, "MoE offered source changed.");
    int64_t k = 0;
    bool eligible = false;
    ORT_RETURN_IF_ERROR(ReadRouteK(graph, *candidate, k, eligible));
    ORT_RETURN_IF_NOT(eligible && guard(graph, *candidate), "MoE offered source changed.");
    MoESnapshot actual;
    ORT_RETURN_IF_ERROR(Snapshot(graph, *candidate, actual));
    ORT_RETURN_IF_NOT(actual == snapshot, "MoE offered source snapshot changed.");
    auto region = expected;
    Statistics statistics;
    ORT_RETURN_IF_ERROR(rewriter->RunOnce(graph, statistics, region, anchor));
    ORT_RETURN_IF_NOT(statistics.rewrites == 1 && region.size() == 1,
                      "MoE must rewrite exactly once to one node.");
    const auto* target = graph.GetNode(*region.begin());
    ORT_RETURN_IF_NOT(target && target->Domain() == kMSDomain && target->OpType() == "MoE" &&
                          target->InputDefs().size() == 5 && !target->InputDefs()[3]->Exists() &&
                          target->InputDefs()[0] == snapshot.boundary[0] &&
                          target->InputDefs()[1] == snapshot.boundary[1] &&
                          target->InputDefs()[2] == snapshot.boundary[2] &&
                          target->InputDefs()[4] == snapshot.boundary[3] &&
                          target->OutputDefs().size() == 1 && target->OutputDefs()[0] == snapshot.boundary[4],
                      "MoE did not preserve boundary identities.");
    ORT_RETURN_IF_ERROR(graph.Resolve());
    for (size_t slot = 0; slot < snapshot.boundary.size(); ++slot) {
      ORT_RETURN_IF_NOT(graph.GetNodeArg(snapshot.names[slot]) == snapshot.boundary[slot],
                        "MoE final boundary identity changed.");
    }
    capability.sub_graph->nodes.assign(region.begin(), region.end());
    return Status::OK();
  };
  return MakeCapability(snapshot.ids, std::move(optimization));
}

Status DiscoverMoE(const SourceReplacement& source, const GraphViewer& viewer,
                   const IExecutionProvider::IKernelLookup& lookup, const logging::Logger& logger,
                   std::vector<std::unique_ptr<ComputeCapability>>& capabilities) {
  const auto& graph = viewer.GetGraph();
  const auto& imports = graph.DomainToVersionMap();
  if (!imports.count(kOnnxDomain) || imports.at(kOnnxDomain) != 18 ||
      !imports.count(kMSDomain) || imports.at(kMSDomain) != 1) return Status::OK();
  ORT_RETURN_IF(graph.GraphResolveNeeded(), "CUDA runtime fusion requires a resolved graph.");
  const auto* recipe = source.body.rule.replacement.Parsed();
  if (!recipe || !SupportsCudaReplacement(*recipe, 1)) return Status::OK();
  std::vector<std::unique_ptr<ComputeCapability>> pending;
  InlinedHashSet<NodeIndex> offered;
  const auto anchors = internal::CollectMoERoots(viewer);
  Status status;
  ORT_RETURN_IF_ERROR(source.body.rule.pattern.VisitMatches(graph, anchors, [&](NodeIndex anchor, Match& match) {
    for (const auto* member : match.nodes)
      if (offered.count(member->Index()) || !member->GetExecutionProviderType().empty()) return true;
    int64_t k = 0;
    bool eligible = false;
    // Preserve the reader's malformed-storage Status across the bool Rule guard interface.
    status = ReadRouteK(graph, match, k, eligible);
    if (!status.IsOK()) return false;
    if (!eligible || !source.body.rule.check_and_bind(graph, match)) return true;
    MoESnapshot snapshot;
    status = Snapshot(graph, match, snapshot);
    if (!status.IsOK()) return false;
    bool available = false;
    status = TargetAvailable(match, *recipe, lookup, logger, available);
    if (!status.IsOK()) return false;
    if (!available) return true;
    pending.push_back(OfferMoE(match, anchor, std::move(snapshot), source.body));
    for (const auto* member : match.nodes) offered.insert(member->Index());
    return true;
  }));
  ORT_RETURN_IF_ERROR(status);
  capabilities = std::move(pending);
  return Status::OK();
}
}  // namespace

gsl::span<const SourceReplacement> MoESources() {
  static const SourceReplacement source = [] {
    std::ostringstream library;
    library << "interface DenseMoE(x,logits,w1,w2,route_k)->(y)\n";
    for (int e = 2; e <= 64; ++e)
      library << "pattern e" << e << " implements DenseMoE " << internal::Topology(e, "route_k", true) << "\n";
    library << "pattern DenseMoEForms = e2";
    for (int e = 3; e <= 64; ++e) library << "|e" << e;
    auto pattern = PatternLibrary(library.str(), {2048, 4096}).GetPattern("DenseMoEForms");
    const Replacement replacement(
        "(x,logits,w1,w2,k)=>com.microsoft.MoE(x,logits,w1,None,w2,"
        "activation_type=\"swiglu\",swiglu_fusion=2,activation_alpha=1.0,activation_beta=0.0,"
        "normalize_routing_weights=1,use_sparse_mixer=0,k=k)");
    Rule rule(cuda_runtime_fusion::kMoERule, std::move(pattern), replacement, MoESemantics,
              {{kOnnxDomain, 18, 18}, {kMSDomain, 1, 1}});
    NumericalProfile numerical{
        "cuda_moe_fp32_mathematical_v1",
        "Bounded FP32 ONNX-18 dense raw-logit TopK/selected Softmax/chunked SwiGLU source subset. "
        "Mathematical semantics with fixture-specific tolerances; ordinary reassociation, normalization, "
        "division and accumulation rounding differences are allowed. No bitwise, universal error-bound "
        "or exceptional-value parity claim. Source equivalence is checked with TF32 disabled; "
        "fused/direct consistency is checked with TF32 both disabled and enabled. Fixture-local tolerances: "
        "onnxruntime/test/providers/cuda/cuda_runtime_fusion_test.cc, "
        "CudaRuntimeFusionTest.MoESourceFusedDirectTF32OffAndOn; docs/Cuda_Runtime_Fusion.md.",
        NumericalGuarantee::MathematicalSemanticsWithTolerance};
    SourceBody body{std::move(rule), std::move(numerical), InitializerTransformation::None, {}};
    return SourceReplacement{{kMSDomain, "MoE", 1}, cuda_runtime_fusion::kMoERule, cuda_runtime_fusion::kMoEConfig, std::move(body), DiscoverMoE};
  }();
  return {&source, 1};
}

const SourceReplacementFactory& MoEFactory() {
  static const SourceReplacementFactory factory{cuda_runtime_fusion::kMoERule, cuda_runtime_fusion::kMoEConfig, MoESources};
  return factory;
}
}  // namespace onnxruntime::contrib::cuda::fusion
#endif
