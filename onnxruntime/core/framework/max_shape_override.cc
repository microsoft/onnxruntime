// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/max_shape_override.h"
#include "core/framework/max_shape_inference.h"

#include <charconv>
#include <string_view>

#include "core/common/common.h"
#include "core/common/string_utils.h"
#include "core/framework/tensor_shape.h"
#if !defined(ORT_MINIMAL_BUILD)
#include "core/graph/graph.h"
#include "core/graph/graph_proto_serializer.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/function_template.h"
#include "core/graph/model.h"
#include "core/graph/schema_registry.h"
#endif

namespace onnxruntime {

class MaxShapeInferenceBuilder {
 public:
  static void Clear(MaxShapeInferenceResult& result) {
    result.graph_shapes_.clear();
  }

#if !defined(ORT_MINIMAL_BUILD)
  static MaxShapeInferenceResult::ShapeMap& AddGraph(
      MaxShapeInferenceResult& result, const Graph& graph) {
    return result.graph_shapes_[&graph];
  }
#endif
};

namespace {

// Trim leading/trailing whitespace from a string_view
std::string_view Trim(std::string_view s) {
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.remove_suffix(1);
  return s;
}

#if !defined(ORT_MINIMAL_BUILD)
using GraphInputMap = InlinedHashMap<std::string, const NodeArg*>;
using SymbolicDimensionMap = InlinedHashMap<std::string, int64_t>;

Status ValidateMaxShapeOverrides(const Graph& graph,
                                 const MaxShapeOverrideMap& input_overrides,
                                 GraphInputMap& graph_inputs,
                                 SymbolicDimensionMap& symbolic_values) {
  graph_inputs.clear();
  graph_inputs.reserve(graph.GetInputs().size());
  for (const NodeArg* input : graph.GetInputs()) {
    graph_inputs.emplace(input->Name(), input);
  }

  symbolic_values.clear();
  for (const auto& [name, override_shape] : input_overrides) {
    const auto input_it = graph_inputs.find(name);
    ORT_RETURN_IF(input_it == graph_inputs.end(),
                  "max_shape_override: '", name, "' is not a graph input");

    const NodeArg& input = *input_it->second;
    const auto* declared_shape = input.Shape();
    ORT_RETURN_IF(declared_shape == nullptr,
                  "max_shape_override: graph input '", name, "' has no declared rank");
    ORT_RETURN_IF(static_cast<size_t>(declared_shape->dim_size()) != override_shape.NumDimensions(),
                  "max_shape_override: rank mismatch for graph input '", name, "': model rank ",
                  declared_shape->dim_size(), ", override rank ", override_shape.NumDimensions());

    for (int dim_index = 0; dim_index < declared_shape->dim_size(); ++dim_index) {
      const auto& declared_dim = declared_shape->dim(dim_index);
      const int64_t override_dim = override_shape[static_cast<size_t>(dim_index)];
      ORT_RETURN_IF(declared_dim.has_dim_value() && declared_dim.dim_value() != override_dim,
                    "max_shape_override: dimension ", dim_index, " for graph input '", name,
                    "' is static (", declared_dim.dim_value(), ") but override specifies ", override_dim);

      if (declared_dim.has_dim_param() && !declared_dim.dim_param().empty()) {
        const auto [symbol_it, inserted] = symbolic_values.emplace(declared_dim.dim_param(), override_dim);
        ORT_RETURN_IF(!inserted && symbol_it->second != override_dim,
                      "max_shape_override: symbolic dimension '", declared_dim.dim_param(),
                      "' has inconsistent values ", symbol_it->second, " and ", override_dim);
      }
    }
  }

  return Status::OK();
}

template <typename TCallable>
Status ConvertShadowInferenceExceptionsToStatus(TCallable&& callable) {
  Status status;
  ORT_TRY {
    status = callable();
  }
  ORT_CATCH(const OnnxRuntimeException& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = Status(ex.Category(), ex.Code(),
                      "max_shape_override: shadow inference failed: " + std::string{ex.what()});
    });
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "max_shape_override: shadow inference failed: ", ex.what());
    });
  }

  return status;
}

std::string GetSubgraphNodeKey(const Node& node) {
  if (!node.Name().empty()) {
    return "name:" + node.Name();
  }

  for (const NodeArg* output : node.OutputDefs()) {
    if (output != nullptr && output->Exists() && !output->Name().empty()) {
      return "output:" + output->Name();
    }
  }

  return {};
}

Status CaptureGraphShapes(const Graph& source_graph,
                          const Graph& shadow_graph,
                          MaxShapeInferenceResult& result) {
  auto& shapes = MaxShapeInferenceBuilder::AddGraph(result, source_graph);
  auto capture_shape = [&shapes](const NodeArg* node_arg) {
    if (node_arg == nullptr || !node_arg->Exists()) return;
    const auto* shape_proto = node_arg->Shape();
    if (shape_proto == nullptr) return;

    TensorShapeVector dims;
    dims.reserve(shape_proto->dim_size());
    for (const auto& dim : shape_proto->dim()) {
      if (!dim.has_dim_value() || dim.dim_value() < 0) return;
      dims.push_back(dim.dim_value());
    }
    shapes.insert_or_assign(node_arg->Name(), TensorShape{dims});
  };

  const GraphViewer shadow_viewer{shadow_graph};
  for (const NodeArg* input : shadow_viewer.GetInputsIncludingInitializers()) capture_shape(input);
  for (const NodeArg* output : shadow_viewer.GetOutputs()) capture_shape(output);
  for (const NodeArg* value_info : shadow_viewer.GetValueInfo()) capture_shape(value_info);
  for (const Node& node : shadow_viewer.Nodes()) {
    for (const NodeArg* input : node.InputDefs()) capture_shape(input);
    for (const NodeArg* output : node.OutputDefs()) capture_shape(output);
  }

  const GraphViewer source_viewer{source_graph};
  ORT_RETURN_IF(source_viewer.NumberOfNodes() != shadow_viewer.NumberOfNodes(),
                "max_shape_override: shadow graph topology does not match source graph");

  // Ordinary NodeArgs are already matched by name above. Node matching is needed only
  // to associate each source control-flow subgraph with its shadow counterpart.
  InlinedHashMap<std::string, const Node*> shadow_subgraph_nodes;
  for (const Node& shadow_node : shadow_viewer.Nodes()) {
    if (shadow_node.GetAttributeNameToSubgraphMap().empty()) continue;

    std::string key = GetSubgraphNodeKey(shadow_node);
    ORT_RETURN_IF(key.empty(),
                  "max_shape_override: cannot identify unnamed shadow node with subgraphs");
    const auto [unused, inserted] = shadow_subgraph_nodes.emplace(std::move(key), &shadow_node);
    ORT_UNUSED_PARAMETER(unused);
    ORT_RETURN_IF(!inserted,
                  "max_shape_override: duplicate shadow node identity for subgraph matching");
  }

  for (const Node& source_node : source_viewer.Nodes()) {
    const auto& source_subgraphs = source_node.GetAttributeNameToSubgraphMap();
    if (source_subgraphs.empty()) continue;

    const std::string key = GetSubgraphNodeKey(source_node);
    ORT_RETURN_IF(key.empty(),
                  "max_shape_override: cannot identify unnamed source node with subgraphs");
    const auto shadow_node_it = shadow_subgraph_nodes.find(key);
    ORT_RETURN_IF(shadow_node_it == shadow_subgraph_nodes.end(),
                  "max_shape_override: shadow graph is missing source node '", key, "'");
    const Node& shadow_node = *shadow_node_it->second;
    ORT_RETURN_IF(source_node.OpType() != shadow_node.OpType() ||
                      source_node.Domain() != shadow_node.Domain(),
                  "max_shape_override: shadow graph node '", key, "' does not match source graph");

    const auto& shadow_subgraphs = shadow_node.GetAttributeNameToSubgraphMap();
    ORT_RETURN_IF(source_subgraphs.size() != shadow_subgraphs.size(),
                  "max_shape_override: shadow graph subgraphs do not match source graph");
    for (const auto& [attribute_name, source_subgraph] : source_subgraphs) {
      const auto shadow_it = shadow_subgraphs.find(attribute_name);
      ORT_RETURN_IF(shadow_it == shadow_subgraphs.end(),
                    "max_shape_override: shadow graph is missing subgraph attribute '",
                    attribute_name, "'");
      ORT_RETURN_IF_ERROR(CaptureGraphShapes(*source_subgraph, *shadow_it->second, result));
    }

    shadow_subgraph_nodes.erase(shadow_node_it);
  }

  ORT_RETURN_IF(!shadow_subgraph_nodes.empty(),
                "max_shape_override: shadow graph contains unmatched nodes with subgraphs");
  return Status::OK();
}

void CollectGeneratedSchemas(
    const Graph& graph,
    InlinedHashMap<std::string, std::vector<ONNX_NAMESPACE::OpSchema>>& schemas_by_domain,
    InlinedHashSet<std::string>& schema_keys) {
  const auto schema_registry = graph.GetSchemaRegistry();
  for (const Node& node : graph.Nodes()) {
    const auto* node_schema = node.Op();
    if (node_schema != nullptr) {
      const auto domain_it = graph.DomainToVersionMap().find(node.Domain());
      const auto* registered_schema =
          domain_it == graph.DomainToVersionMap().end() || schema_registry == nullptr
              ? nullptr
              : schema_registry->GetSchema(node.OpType(), domain_it->second, node.Domain());
      if (registered_schema != node_schema) {
        const std::string key = node.Domain() + ":" + node.OpType() + ":" +
                                std::to_string(node_schema->SinceVersion());
        if (schema_keys.insert(key).second) {
          schemas_by_domain[node.Domain()].push_back(*node_schema);
        }
      }
    }

    for (const auto& [attribute_name, subgraph] : node.GetAttributeNameToSubgraphMap()) {
      ORT_UNUSED_PARAMETER(attribute_name);
      CollectGeneratedSchemas(*subgraph, schemas_by_domain, schema_keys);
    }
  }
}
#endif

}  // namespace

Status ParseMaxShapeOverride(std::string_view config_value, MaxShapeOverrideMap& out) {
  out.clear();

  config_value = Trim(config_value);
  if (config_value.empty()) {
    return Status::OK();
  }

  size_t pos = 0;
  while (pos < config_value.size()) {
    const size_t entry_end = config_value.find(';', pos);
    const size_t entry_length =
        (entry_end == std::string_view::npos ? config_value.size() : entry_end) - pos;
    const std::string_view entry = Trim(config_value.substr(pos, entry_length));
    ORT_RETURN_IF(entry.empty(),
                  "max_shape_override: empty entry at position ", pos);

    const size_t colon = entry.find(':');
    ORT_RETURN_IF(colon == std::string_view::npos,
                  "max_shape_override: missing ':' in entry '", entry, "'");

    const std::string_view name = Trim(entry.substr(0, colon));
    ORT_RETURN_IF(name.empty(),
                  "max_shape_override: empty name in entry '", entry, "'");

    const std::string_view shape_str = Trim(entry.substr(colon + 1));
    ORT_RETURN_IF(shape_str.size() < 2 || shape_str.front() != '[' || shape_str.back() != ']',
                  "max_shape_override: shape must be enclosed in [] for '", name, "'");

    const std::string_view dims_str = Trim(shape_str.substr(1, shape_str.size() - 2));
    TensorShapeVector dims;
    size_t dim_pos = 0;
    while (dim_pos < dims_str.size()) {
      const size_t comma = dims_str.find(',', dim_pos);
      const size_t token_length =
          (comma == std::string_view::npos ? dims_str.size() : comma) - dim_pos;
      const std::string_view dim_token = Trim(dims_str.substr(dim_pos, token_length));
      ORT_RETURN_IF(dim_token.empty(),
                    "max_shape_override: empty dimension for input '", name, "'");

      int64_t dim_value = 0;
      const auto [ptr, ec] =
          std::from_chars(dim_token.data(), dim_token.data() + dim_token.size(), dim_value);
      ORT_RETURN_IF(ec != std::errc{} || ptr != dim_token.data() + dim_token.size(),
                    "max_shape_override: invalid dimension '", dim_token,
                    "' for input '", name, "'");
      ORT_RETURN_IF(dim_value <= 0,
                    "max_shape_override: dimensions must be positive, got ", dim_value,
                    " for input '", name, "'");
      dims.push_back(dim_value);

      if (comma == std::string_view::npos) break;
      dim_pos = comma + 1;
      ORT_RETURN_IF(dim_pos == dims_str.size(),
                    "max_shape_override: empty dimension for input '", name, "'");
    }

    const auto [unused, inserted] = out.emplace(std::string{name}, TensorShape{dims});
    ORT_UNUSED_PARAMETER(unused);
    ORT_RETURN_IF(!inserted,
                  "max_shape_override: duplicate entry for '", name, "'");

    if (entry_end == std::string_view::npos) break;
    pos = entry_end + 1;
    ORT_RETURN_IF(pos == config_value.size(),
                  "max_shape_override: empty entry at position ", pos);
  }

  return Status::OK();
}

Status InferMaxShapes(const Graph& graph,
                      const MaxShapeOverrideMap& input_overrides,
                      MaxShapeInferenceResult& result) {
  MaxShapeInferenceBuilder::Clear(result);
  if (input_overrides.empty()) {
    return Status::OK();
  }

#if defined(ORT_MINIMAL_BUILD)
  ORT_UNUSED_PARAMETER(graph);
  return Status::OK();
#else
  // Reject invalid overrides before serializing and resolving a disposable model.
  GraphInputMap graph_inputs;
  SymbolicDimensionMap symbolic_values;
  ORT_RETURN_IF_ERROR(ValidateMaxShapeOverrides(graph, input_overrides, graph_inputs, symbolic_values));

  // Serialize into a disposable model so normal shape inference cannot make the
  // executable graph appear statically shaped to optimizers or runtime validation.
  const GraphViewer source_viewer{graph};
  ONNX_NAMESPACE::ModelProto model_proto;
  model_proto.set_ir_version(graph.GetModel().IrVersion());
  for (const auto& [domain, version] : graph.DomainToVersionMap()) {
    auto* opset = model_proto.add_opset_import();
    opset->set_domain(domain);
    opset->set_version(version);
  }

  GraphViewerToProto(source_viewer, *model_proto.mutable_graph(),
                     true, true, ExecutionOrder::DEFAULT, false);

  // Model-local functions and dynamically generated fused-node schemas are not fully
  // represented by GraphViewerToProto, but are required to resolve the shadow graph.
  for (const auto& [id, function_template] : graph.GetModel().GetModelLocalFunctionTemplates()) {
    ORT_UNUSED_PARAMETER(id);
    *model_proto.add_functions() = *function_template->onnx_func_proto_;
  }

  IOnnxRuntimeOpSchemaRegistryList registries;
  if (auto source_registry = graph.GetSchemaRegistry()) {
    registries.push_back(std::move(source_registry));
  }

  InlinedHashMap<std::string, std::vector<ONNX_NAMESPACE::OpSchema>> generated_schemas;
  InlinedHashSet<std::string> generated_schema_keys;
  CollectGeneratedSchemas(graph, generated_schemas, generated_schema_keys);
  if (!generated_schemas.empty()) {
    auto generated_schema_registry = std::make_shared<OnnxRuntimeOpSchemaRegistry>();
    for (auto& [domain, schemas] : generated_schemas) {
      const auto version_it = graph.DomainToVersionMap().find(domain);
      ORT_RETURN_IF(version_it == graph.DomainToVersionMap().end(),
                    "max_shape_override: no opset import for generated schema domain '", domain, "'");
      ORT_RETURN_IF_ERROR(generated_schema_registry->RegisterOpSet(
          schemas, domain, 0, version_it->second));
    }
    registries.push_back(std::move(generated_schema_registry));
  }

  ModelOptions model_options;
  model_options.strict_shape_type_inference = graph.StrictShapeTypeInference();
  std::unique_ptr<Model> shadow_model;
  ORT_RETURN_IF_ERROR(ConvertShadowInferenceExceptionsToStatus([&]() -> Status {
    shadow_model = std::make_unique<Model>(
        std::move(model_proto), graph.ModelPath().native(), &registries, graph.GetLogger(), model_options);
    return Status::OK();
  }));
  Graph& shadow_graph = shadow_model->MainGraph();

  // GraphViewerToProto omits raw initializer payloads. Share converted OrtValues
  // and copy only initializers that still live in protobuf storage.
  for (const auto& [name, initializer] : source_viewer.GetAllInitializedTensors()) {
    shadow_graph.RemoveInitializedTensor(name);
    OrtValue ort_value;
    if (source_viewer.GetOrtValueInitializer(name, ort_value)) {
      ORT_RETURN_IF_ERROR(shadow_graph.AddInitializedOrtValue(*initializer, ort_value));
    } else {
      shadow_graph.AddInitializedTensor(*initializer);
    }
  }

  // Apply direct overrides and propagate known symbolic maxima to the remaining inputs.
  for (const auto& [name, input] : graph_inputs) {
    const auto* declared_shape = input->Shape();
    if (declared_shape == nullptr) continue;

    const auto override_it = input_overrides.find(name);
    const bool has_override = override_it != input_overrides.end();
    bool changed = has_override;
    ONNX_NAMESPACE::TensorShapeProto max_shape_proto;
    for (int dim_index = 0; dim_index < declared_shape->dim_size(); ++dim_index) {
      auto* max_dim = max_shape_proto.add_dim();
      if (has_override) {
        max_dim->set_dim_value(override_it->second[static_cast<size_t>(dim_index)]);
        continue;
      }

      const auto& declared_dim = declared_shape->dim(dim_index);
      if (declared_dim.has_dim_param()) {
        const auto symbol_it = symbolic_values.find(declared_dim.dim_param());
        if (symbol_it != symbolic_values.end()) {
          max_dim->set_dim_value(symbol_it->second);
          changed = true;
          continue;
        }
      }

      *max_dim = declared_dim;
    }

    if (!changed) continue;
    NodeArg* shadow_input = shadow_graph.GetNodeArg(name);
    ORT_RETURN_IF_NOT(shadow_input != nullptr,
                      "max_shape_override: graph input '", name, "' is missing from shadow graph");
    shadow_input->SetShape(max_shape_proto);
  }

  // Resolve recursively, then associate concrete shadow shapes with their corresponding
  // source graph identities so partitioning can query main-graph and subgraph nodes.
  ORT_RETURN_IF_ERROR(
      ConvertShadowInferenceExceptionsToStatus([&]() -> Status { return shadow_graph.Resolve(); }));
  return CaptureGraphShapes(graph, shadow_graph, result);
#endif
}

}  // namespace onnxruntime
