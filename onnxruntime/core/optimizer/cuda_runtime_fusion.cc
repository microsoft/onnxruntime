// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/cuda_runtime_fusion.h"

#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS)

#include <algorithm>
#include <cmath>
#include <limits>
#include "core/framework/compute_capability.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/optimizer/pattern_rewriter.h"

namespace onnxruntime::cuda_runtime_fusion {
namespace {
using namespace rewrite;
bool Type(const NodeArg* value, int expected, bool required) {
  const auto* type = value->TypeAsProto();
  if (!type || !type->has_tensor_type()) return !required && !type;
  const int element = type->tensor_type().elem_type();
  return element == expected || (!required && element == 0);
}

Status ProbeTarget(Graph& graph, const Replacement& recipe, const Replacement::TensorBindings& bindings,
                   const NodeAttributes& attributes, size_t output_count,
                   const IExecutionProvider::IKernelLookup& lookup, bool& available) {
  Replacement::Instantiation instantiated;
  ORT_RETURN_IF_ERROR(recipe.Instantiate(graph, bindings, attributes, instantiated));
  ORT_RETURN_IF_NOT(instantiated.nodes.size() == 1 && instantiated.outputs.size() == output_count,
                    "CUDA runtime fusion target recipe must emit one node with the expected outputs.");
  InlinedVector<const NodeArg*> outputs(instantiated.outputs.begin(), instantiated.outputs.end());
  graph.SetOutputs(outputs);
  ORT_RETURN_IF_ERROR(graph.Resolve());
  available = lookup.LookUpKernel(*graph.GetNode(instantiated.nodes.front())) != nullptr;
  return Status::OK();
}

constexpr char kUnitSwishPattern[] = R"(pattern UnitSwish(x) { y=Swish(x); return y })";
constexpr char kUnitSwishReplacement[] = R"((x)=>{ y=com.microsoft.ScaledSiLU(x,alpha=1.0); return y })";

// Own metadata values. Graph-owned pointers are identity tokens, never dereferenced.
// Acceptance must use the original live graph and fetch its current source by index.
struct UnitSwishSnapshot {
  const Node* node = nullptr;
  const NodeArg* input = nullptr;
  const NodeArg* output = nullptr;
  std::string input_name;
  std::string output_name;
  std::string name;
  std::string domain;
  std::string overload;
  int version = 0;
  std::string input_type;
  std::string output_type;
  InlinedVector<std::pair<std::string, std::string>> attributes;

  bool operator==(const UnitSwishSnapshot& other) const {
    return node == other.node && input == other.input && output == other.output &&
           input_name == other.input_name && output_name == other.output_name &&
           name == other.name && domain == other.domain && overload == other.overload && version == other.version &&
           input_type == other.input_type && output_type == other.output_type && attributes == other.attributes;
  }
};

// No payload inspection, including for initializer inputs. Zero extents annihilate the
// product before multiplication; unknown extents do not invent a runtime size.
bool UnitSwishShape(const ONNX_NAMESPACE::TensorShapeProto* shape) {
  if (!shape) return true;
  bool unknown = false;
  bool zero = false;
  for (const auto& dim : shape->dim()) {
    if (!dim.has_dim_value())
      unknown = true;
    else if (dim.dim_value() < 0)
      return false;
    else if (dim.dim_value() == 0)
      zero = true;
  }
  if (zero || unknown) return true;
  // Mirrors LaunchScaledSiLU in scaled_silu_impl.cu: 256 threads and an int block count.
  constexpr int64_t max_count = int64_t{256} * std::numeric_limits<int>::max();
  int64_t count = 1;
  for (const auto& dim : shape->dim()) {
    if (dim.dim_value() > max_count / count) return false;
    count *= dim.dim_value();
  }
  return true;
}

// Read live metadata only; discovery and acceptance must use the same complete check.
bool UnitSwishSemantics(const Graph& graph, const Match& match, UnitSwishSnapshot* snapshot) {
  const auto& imports = graph.DomainToVersionMap();
  if (!imports.count(kOnnxDomain) || imports.at(kOnnxDomain) != 24 ||
      !imports.count(kMSDomain) || imports.at(kMSDomain) != 1 || match.nodes.size() != 1) return false;
  const auto* node = *match.nodes.begin();
  if (node->OpType() != "Swish" || node->Domain() != kOnnxDomain || node->SinceVersion() != 24 ||
      !node->Overload().empty() || !node->GetExecutionProviderType().empty() || node->InputDefs().size() != 1 ||
      node->OutputDefs().size() != 1) return false;
  const auto* input = match.valueOf("x");
  const auto* output = match.valueOf("y");
  if (!input || !output || !input->Exists() || !output->Exists()) return false;
  const auto* input_type = input->TypeAsProto();
  if (!input_type || !input_type->has_tensor_type()) return false;
  const int type = input_type->tensor_type().elem_type();
  if (type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
      type != ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16) return false;
  if (!Type(input, type, true) || !Type(output, type, true) ||
      !UnitSwishShape(input->Shape()) || !UnitSwishShape(output->Shape())) return false;
#if !defined(DISABLE_SPARSE_TENSORS)
  if (graph.IsSparseInitializer(input->Name())) return false;
#endif
  if (const auto* tensor = graph.GetConstantInitializer(input->Name(), false)) {
    if (tensor->data_type() != type) return false;
    ONNX_NAMESPACE::TensorShapeProto initializer_shape;
    for (auto dim : tensor->dims()) initializer_shape.add_dim()->set_dim_value(dim);
    if (!UnitSwishShape(&initializer_shape)) return false;
    for (const auto* value : {input, output}) {
      if (const auto* shape = value->Shape()) {
        if (shape->dim_size() != tensor->dims_size()) return false;
        for (int i = 0; i < shape->dim_size(); ++i) {
          if (shape->dim(i).has_dim_value() && shape->dim(i).dim_value() != tensor->dims(i)) return false;
        }
      }
    }
  }
  const auto* in_shape = input->Shape();
  const auto* out_shape = output->Shape();
  if (in_shape && out_shape) {
    if (in_shape->dim_size() != out_shape->dim_size()) return false;
    for (int i = 0; i < in_shape->dim_size(); ++i) {
      const auto& a = in_shape->dim(i);
      const auto& b = out_shape->dim(i);
      if (a.has_dim_value() && b.has_dim_value() && a.dim_value() != b.dim_value()) return false;
    }
  }
  UnitSwishSnapshot actual;
  for (const auto& [name, attr] : node->GetAttributes()) {
    if (name != "alpha" || attr.type() != ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT ||
        !std::isfinite(attr.f()) || attr.f() != 1.0f) return false;
    actual.attributes.emplace_back(name, attr.SerializeAsString());
  }
  std::sort(actual.attributes.begin(), actual.attributes.end());
  actual.node = node;
  actual.input = input;
  actual.output = output;
  actual.input_name = input->Name();
  actual.output_name = output->Name();
  actual.name = node->Name();
  actual.domain = node->Domain();
  actual.overload = node->Overload();
  actual.version = node->SinceVersion();
  actual.input_type = input->TypeAsProto()->SerializeAsString();
  actual.output_type = output->TypeAsProto()->SerializeAsString();
  if (snapshot) *snapshot = std::move(actual);
  return true;
}

Status UnitSwishTargetAvailable(const Match& match, const Replacement& recipe,
                                const IExecutionProvider::IKernelLookup& lookup,
                                const logging::Logger& logger, bool& available) {
  Model scratch("cuda_unit_swish_probe", false, ModelMetaData(), PathString(),
                IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 24}, {kMSDomain, 1}}, {}, logger);
  auto& graph = scratch.MainGraph();
  auto& input = graph.GetOrCreateNodeArg("x", match.valueOf("x")->TypeAsProto());
  graph.SetInputs({&input});
  return ProbeTarget(graph, recipe, {{"x", &input}}, {}, 1, lookup, available);
}

std::unique_ptr<ComputeCapability> UnitSwishOffer(NodeIndex anchor, UnitSwishSnapshot snapshot,
                                                  const Pattern& pattern, const Replacement& recipe) {
  Options options;
  options.compatible_execution_providers.clear();
  Rules rules;
  auto predicate = [snapshot](const Graph& graph, Match& candidate) {
    UnitSwishSnapshot actual;
    return UnitSwishSemantics(graph, candidate, &actual) && actual == snapshot;
  };
  rules.emplace_back(kUnitSwishRule, pattern, recipe, std::move(predicate),
                     InlinedVector<OpsetRequirement, 1>{{kOnnxDomain, 24, 24}, {kMSDomain, 1, 1}});
  auto rewriter = std::make_shared<const PatternRewriter>(std::move(rules), std::move(options));
  auto region = std::make_unique<IndexedSubGraph>();
  region->nodes.push_back(anchor);
  auto parent = std::make_unique<ComputeCapability>(std::move(region));
  auto optimization = std::make_unique<ComputeCapability>(std::make_unique<IndexedSubGraph>());
  optimization->sub_graph->nodes.push_back(anchor);
  optimization->optimization_func =
      [rewriter, anchor, snapshot](Graph& graph, const ComputeCapability& selected, ComputeCapability& capability,
                                   const GraphOptimizerRegistry&) -> Status {
    ORT_RETURN_IF_NOT(selected.sub_graph && capability.sub_graph &&
                          selected.sub_graph->nodes.size() == 1 && selected.sub_graph->nodes[0] == anchor &&
                          capability.sub_graph->nodes.size() == 1 && capability.sub_graph->nodes[0] == anchor,
                      "Unit Swish selected region changed.");
    const auto* source = graph.GetNode(anchor);
    ORT_RETURN_IF_NOT(source && source == snapshot.node && source->GetExecutionProviderType().empty(),
                      "Unit Swish source is missing or already assigned.");
    InlinedHashSet<NodeIndex> region{anchor};
    Statistics statistics;
    ORT_RETURN_IF_ERROR(rewriter->Run(graph, statistics, region));
    ORT_RETURN_IF_NOT(statistics.rewrites == 1 && region.size() == 1,
                      "Unit Swish must rewrite exactly once to one node.");
    const auto* target = graph.GetNode(*region.begin());
    ORT_RETURN_IF_NOT(target && target->Domain() == kMSDomain && target->OpType() == "ScaledSiLU" &&
                          target->InputDefs().size() == 1 && target->InputDefs()[0] == snapshot.input &&
                          target->OutputDefs().size() == 1 && target->OutputDefs()[0] == snapshot.output,
                      "Unit Swish did not preserve input/output identities.");
    ORT_RETURN_IF_ERROR(graph.Resolve());
    ORT_RETURN_IF_NOT(target->InputDefs()[0] == snapshot.input && target->OutputDefs()[0] == snapshot.output &&
                          graph.GetNodeArg(snapshot.output_name) == snapshot.output,
                      "Unit Swish final input/output identities changed.");
    capability.sub_graph->nodes.assign(region.begin(), region.end());
    return Status::OK();
  };
  parent->nodes_to_optimize.push_back(std::move(optimization));
  return parent;
}

Status SelectUnitSwish(const GraphViewer& viewer, const IExecutionProvider::IKernelLookup& lookup,
                       const logging::Logger& logger,
                       std::vector<std::unique_ptr<ComputeCapability>>& capabilities) {
  const auto& graph = viewer.GetGraph();
  const Pattern pattern = PatternLibrary(kUnitSwishPattern).GetPattern("UnitSwish");
  const Replacement recipe(kUnitSwishReplacement);
  std::vector<std::unique_ptr<ComputeCapability>> pending;
  for (auto anchor : viewer.GetNodesInTopologicalOrder()) {
    const auto* node = graph.GetNode(anchor);
    if (!node || node->OpType() != "Swish") continue;
    Status probe_status;
    ORT_RETURN_IF_ERROR(pattern.VisitMatches(graph, anchor, [&](Match& match) {
      UnitSwishSnapshot snapshot;
      if (!UnitSwishSemantics(graph, match, &snapshot)) return true;
      bool available = false;
      probe_status = UnitSwishTargetAvailable(match, recipe, lookup, logger, available);
      if (!probe_status.IsOK()) return false;
      if (available) pending.push_back(UnitSwishOffer(anchor, std::move(snapshot), pattern, recipe));
      return true;
    }));
    ORT_RETURN_IF_ERROR(probe_status);
  }
  capabilities = std::move(pending);
  return Status::OK();
}
}  // namespace

Status Select(const GraphViewer& viewer, const IExecutionProvider::IKernelLookup& kernel_lookup,
              const GraphOptimizerRegistry& registry, const std::string& rule_id,
              std::vector<std::unique_ptr<ComputeCapability>>& capabilities) {
  ORT_RETURN_IF_NOT(capabilities.empty(), "CUDA runtime fusion requires initially empty output.");
  ORT_RETURN_IF_NOT(rule_id == kUnitSwishRule,
                    "Unknown CUDA runtime fusion rule: ", rule_id);
  const auto* config = kUnitSwishConfig;
  const auto setting = registry.GetSessionOptions().config_options.GetConfigOrDefault(config, "0");
  ORT_RETURN_IF_NOT(setting == "0" || setting == "1", config, " must be exactly 0 or 1.");
  if (!registry.SupportsCapabilityOptimizations()) return Status::OK();
  if (registry.GetSessionOptions().graph_optimization_level < TransformerLevel::Level2) return Status::OK();
  if (setting == "0" || viewer.GetFilterInfo() || viewer.IsSubgraph()) return Status::OK();
  const auto& graph = viewer.GetGraph();
  const auto& imports = graph.DomainToVersionMap();
  if (!imports.count(kOnnxDomain) || imports.at(kOnnxDomain) != 24 ||
      !imports.count(kMSDomain) || imports.at(kMSDomain) != 1) return Status::OK();
  ORT_RETURN_IF(graph.GraphResolveNeeded(), "CUDA runtime fusion requires a resolved graph.");
  return SelectUnitSwish(viewer, kernel_lookup, *registry.GetLogger(), capabilities);
}
}  // namespace onnxruntime::cuda_runtime_fusion

#endif
