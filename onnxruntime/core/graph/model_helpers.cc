// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include "core/graph/model_helpers.h"

#include <algorithm>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/graph/function_utils.h"
#include "core/graph/graph.h"
#include "core/graph/onnx_protobuf.h"
#include "core/graph/schema_registry.h"

namespace onnxruntime {

namespace {

using NodeRange = const google::protobuf::RepeatedPtrField<ONNX_NAMESPACE::NodeProto>*;
using PendingNodeRanges = InlinedVector<std::pair<NodeRange, size_t>>;

Status AddAttributeSubgraphs(const ONNX_NAMESPACE::AttributeProto& attr,
                             size_t subgraph_depth,
                             PendingNodeRanges& pending) {
  if ((attr.has_g() || !attr.graphs().empty()) && subgraph_depth > kMaxModelSubgraphDepth) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, NOT_IMPLEMENTED,
        "Model subgraph depth ", subgraph_depth,
        " exceeds the maximum supported depth of ", kMaxModelSubgraphDepth, ".");
  }

  if (attr.has_g()) {
    pending.push_back({&attr.g().node(), subgraph_depth});
  }
  for (const auto& graph : attr.graphs()) {
    pending.push_back({&graph.node(), subgraph_depth});
  }

  return Status::OK();
}

Status ValidateSubgraphDepth(
    const google::protobuf::RepeatedPtrField<ONNX_NAMESPACE::NodeProto>& root_nodes,
    const google::protobuf::RepeatedPtrField<ONNX_NAMESPACE::AttributeProto>* root_attributes = nullptr) {
  PendingNodeRanges pending{{&root_nodes, 0}};
  if (root_attributes != nullptr) {
    for (const auto& attr : *root_attributes) {
      ORT_RETURN_IF_ERROR(AddAttributeSubgraphs(attr, 1, pending));
    }
  }

  while (!pending.empty()) {
    const auto [nodes, depth] = pending.back();
    pending.pop_back();
    for (const auto& node : *nodes) {
      for (const auto& attr : node.attribute()) {
        ORT_RETURN_IF_ERROR(AddAttributeSubgraphs(attr, depth + 1, pending));
      }
    }
  }

  return Status::OK();
}

// Iterative collection of local function calls from a sequence of nodes,
// including nodes inside nested subgraph attributes. Avoids recursion to
// prevent stack overflow from maliciously deep subgraph nesting.
template <typename NodeRange>
void CollectLocalFunctionCalls(
    const NodeRange& nodes,
    const std::unordered_map<std::string, const ONNX_NAMESPACE::FunctionProto*>& model_local_functions,
    InlinedHashSet<std::string_view>& seen_calls,
    InlinedVector<std::string_view>& called_functions) {
  InlinedVector<const ONNX_NAMESPACE::GraphProto*> pending_graphs;

  auto process_nodes = [&](const auto& node_range) {
    for (const auto& node : node_range) {
      const auto function_id = function_utils::GetFunctionIdentifier(
          node.domain(), node.op_type(), node.overload());
      auto it = model_local_functions.find(function_id);
      if (it != model_local_functions.end()) {
        // Use string_view into the map key (stable storage).
        std::string_view key_view = it->first;
        if (seen_calls.insert(key_view).second) {
          called_functions.push_back(key_view);
        }
        continue;
      }

      for (const auto& attr : node.attribute()) {
        if (attr.has_g()) {
          pending_graphs.push_back(&attr.g());
        }
        for (const auto& sub_graph : attr.graphs()) {
          pending_graphs.push_back(&sub_graph);
        }
      }
    }
  };

  process_nodes(nodes);

  while (!pending_graphs.empty()) {
    const auto* graph = pending_graphs.back();
    pending_graphs.pop_back();
    process_nodes(graph->node());
  }
}

struct AttributeBindingContext;

struct BoundAttribute {
  const ONNX_NAMESPACE::AttributeProto* proto;
  const Graph* graph;
  std::shared_ptr<const AttributeBindingContext> context;
};

struct AttributeBinding {
  std::string_view name;
  BoundAttribute attribute;
};

using AttributeBindings = InlinedVector<AttributeBinding>;

using DomainToVersionMap = std::unordered_map<std::string, int>;

struct AttributeBindingContext {
  AttributeBindings bindings;
  DomainToVersionMap domain_to_version;
};

bool CanContainGraph(const BoundAttribute& attribute) {
  if (attribute.graph != nullptr) {
    return true;
  }

  if (attribute.proto == nullptr) {
    return false;
  }

  return attribute.proto->has_g() ||
         !attribute.proto->graphs().empty() ||
         attribute.proto->type() == ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH ||
         attribute.proto->type() == ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPHS;
}

void CollectReferencedAttributeNames(
    const ONNX_NAMESPACE::AttributeProto& root_attribute,
    InlinedHashSet<std::string_view>& referenced_attribute_names) {
  InlinedVector<const ONNX_NAMESPACE::AttributeProto*> pending_attributes{&root_attribute};

  while (!pending_attributes.empty()) {
    const auto* attribute = pending_attributes.back();
    pending_attributes.pop_back();
    if (!attribute->ref_attr_name().empty()) {
      referenced_attribute_names.insert(attribute->ref_attr_name());
    }

    const auto enqueue_graph_attributes = [&pending_attributes](const ONNX_NAMESPACE::GraphProto& graph) {
      for (const auto& node : graph.node()) {
        for (const auto& nested_attribute : node.attribute()) {
          pending_attributes.push_back(&nested_attribute);
        }
      }
    };

    if (attribute->has_g()) {
      enqueue_graph_attributes(attribute->g());
    }
    for (const auto& graph : attribute->graphs()) {
      enqueue_graph_attributes(graph);
    }
  }
}

void CollectReferencedAttributeNames(
    const ONNX_NAMESPACE::FunctionProto& function_proto,
    InlinedHashSet<std::string_view>& referenced_attribute_names) {
  InlinedVector<NodeRange> pending_node_ranges{&function_proto.node()};

  const auto process_attribute = [&](const ONNX_NAMESPACE::AttributeProto& attribute) {
    if (!attribute.ref_attr_name().empty()) {
      referenced_attribute_names.insert(attribute.ref_attr_name());
    }
    if (attribute.has_g()) {
      pending_node_ranges.push_back(&attribute.g().node());
    }
    for (const auto& graph : attribute.graphs()) {
      pending_node_ranges.push_back(&graph.node());
    }
  };

  for (const auto& attribute : function_proto.attribute_proto()) {
    process_attribute(attribute);
  }

  while (!pending_node_ranges.empty()) {
    const auto* nodes = pending_node_ranges.back();
    pending_node_ranges.pop_back();
    for (const auto& node : *nodes) {
      for (const auto& attribute : node.attribute()) {
        process_attribute(attribute);
      }
    }
  }
}

using ModelLocalFunctions =
    std::unordered_map<std::string, const ONNX_NAMESPACE::FunctionProto*>;

bool HasRegisteredSchema(const std::string& domain,
                         const std::string& op_type,
                         const DomainToVersionMap& domain_to_version,
                         const IOnnxRuntimeOpSchemaCollection& schema_registry) {
  const auto version_it = domain_to_version.find(domain);
  if (version_it == domain_to_version.end()) {
    return false;
  }

  const auto* schema = schema_registry.GetSchema(op_type, version_it->second, domain);
  return schema != nullptr && !schema->Deprecated();
}

bool HasOnnxRegisteredSchema(const std::string& domain,
                             const std::string& op_type,
                             const DomainToVersionMap& domain_to_version) {
  const auto& lookup_domain = domain == kOnnxDomainAlias ? kOnnxDomain : domain;
  const auto version_it = domain_to_version.find(lookup_domain);
  if (version_it == domain_to_version.end()) {
    return false;
  }

  const auto* schema = ONNX_NAMESPACE::OpSchemaRegistry::Instance()->GetSchema(
      op_type, version_it->second, lookup_domain);
  return schema != nullptr && !schema->Deprecated();
}

DomainToVersionMap GetFunctionDomainToVersionMap(
    const ONNX_NAMESPACE::FunctionProto& function_proto) {
  DomainToVersionMap domain_to_version;
  domain_to_version.reserve(function_proto.opset_import().size());
  for (const auto& opset : function_proto.opset_import()) {
    const auto& domain = opset.domain() == kOnnxDomainAlias ? kOnnxDomain : opset.domain();
    domain_to_version[domain] = gsl::narrow_cast<int>(opset.version());
  }

  return domain_to_version;
}

struct FunctionValidationState {
  const ONNX_NAMESPACE::FunctionProto* function_proto;
  size_t call_depth;
  size_t attribute_expansion_depth;
  AttributeBindings bindings;

  bool operator==(const FunctionValidationState& other) const {
    return function_proto == other.function_proto &&
           call_depth == other.call_depth &&
           attribute_expansion_depth == other.attribute_expansion_depth &&
           bindings.size() == other.bindings.size() &&
           std::equal(bindings.begin(), bindings.end(), other.bindings.begin(),
                      [](const AttributeBinding& lhs, const AttributeBinding& rhs) {
                        return lhs.name == rhs.name &&
                               lhs.attribute.proto == rhs.attribute.proto &&
                               lhs.attribute.graph == rhs.attribute.graph &&
                               lhs.attribute.context == rhs.attribute.context;
                      });
  }
};

struct FunctionValidationStateHash {
  size_t operator()(const FunctionValidationState& state) const {
    size_t result = std::hash<const void*>{}(state.function_proto);
    auto combine = [&result](size_t value) {
      result ^= value + 0x9e3779b9 + (result << 6) + (result >> 2);
    };

    combine(std::hash<size_t>{}(state.call_depth));
    combine(std::hash<size_t>{}(state.attribute_expansion_depth));
    for (const auto& binding : state.bindings) {
      combine(std::hash<std::string_view>{}(binding.name));
      combine(std::hash<const void*>{}(binding.attribute.proto));
      combine(std::hash<const void*>{}(binding.attribute.graph));
      combine(std::hash<const void*>{}(binding.attribute.context.get()));
    }

    return result;
  }
};

bool AttributeBindingContextsEqual(const AttributeBindingContext& lhs,
                                   const AttributeBindingContext& rhs) {
  return lhs.domain_to_version == rhs.domain_to_version &&
         lhs.bindings.size() == rhs.bindings.size() &&
         std::equal(lhs.bindings.begin(), lhs.bindings.end(), rhs.bindings.begin(),
                    [](const AttributeBinding& lhs_binding, const AttributeBinding& rhs_binding) {
                      return lhs_binding.name == rhs_binding.name &&
                             lhs_binding.attribute.proto == rhs_binding.attribute.proto &&
                             lhs_binding.attribute.graph == rhs_binding.attribute.graph &&
                             lhs_binding.attribute.context == rhs_binding.attribute.context;
                    });
}

struct AttributeBindingContextPtrHash {
  size_t operator()(const std::shared_ptr<const AttributeBindingContext>& context) const {
    size_t result = 0;
    const auto combine = [&result](size_t value) {
      result ^= value + 0x9e3779b9 + (result << 6) + (result >> 2);
    };
    for (const auto& binding : context->bindings) {
      combine(std::hash<std::string_view>{}(binding.name));
      combine(std::hash<const void*>{}(binding.attribute.proto));
      combine(std::hash<const void*>{}(binding.attribute.graph));
      combine(std::hash<const void*>{}(binding.attribute.context.get()));
    }

    size_t domain_hash = 0;
    for (const auto& [domain, version] : context->domain_to_version) {
      domain_hash ^= std::hash<std::string>{}(domain) ^
                     (std::hash<int>{}(version) + 0x9e3779b9);
    }
    combine(domain_hash);
    return result;
  }
};

struct AttributeBindingContextPtrEqual {
  bool operator()(const std::shared_ptr<const AttributeBindingContext>& lhs,
                  const std::shared_ptr<const AttributeBindingContext>& rhs) const {
    return AttributeBindingContextsEqual(*lhs, *rhs);
  }
};

struct BoundAttributeExpansionState {
  const ONNX_NAMESPACE::AttributeProto* proto;
  const Graph* graph;
  const AttributeBindingContext* context;
  size_t call_depth;
  size_t attribute_expansion_depth;
  bool use_onnx_schema_registry;

  bool operator==(const BoundAttributeExpansionState& other) const {
    return proto == other.proto && graph == other.graph && context == other.context &&
           call_depth == other.call_depth &&
           attribute_expansion_depth == other.attribute_expansion_depth &&
           use_onnx_schema_registry == other.use_onnx_schema_registry;
  }
};

struct BoundAttributeExpansionStateHash {
  size_t operator()(const BoundAttributeExpansionState& state) const {
    size_t result = std::hash<const void*>{}(state.proto);
    result ^= std::hash<const void*>{}(state.graph) + 0x9e3779b9 + (result << 6) + (result >> 2);
    result ^= std::hash<const void*>{}(state.context) + 0x9e3779b9 + (result << 6) + (result >> 2);
    result ^= std::hash<size_t>{}(state.call_depth) + 0x9e3779b9 + (result << 6) + (result >> 2);
    result ^= std::hash<size_t>{}(state.attribute_expansion_depth) + 0x9e3779b9 + (result << 6) + (result >> 2);
    result ^= std::hash<bool>{}(state.use_onnx_schema_registry) + 0x9e3779b9 + (result << 6) + (result >> 2);
    return result;
  }
};

struct BoundAttributeActiveState {
  const ONNX_NAMESPACE::AttributeProto* proto;
  const Graph* graph;
  const AttributeBindingContext* context;
  bool use_onnx_schema_registry;

  bool operator==(const BoundAttributeActiveState& other) const {
    return proto == other.proto && graph == other.graph && context == other.context &&
           use_onnx_schema_registry == other.use_onnx_schema_registry;
  }
};

struct BoundAttributeActiveStateHash {
  size_t operator()(const BoundAttributeActiveState& state) const {
    size_t result = std::hash<const void*>{}(state.proto);
    result ^= std::hash<const void*>{}(state.graph) + 0x9e3779b9 + (result << 6) + (result >> 2);
    result ^= std::hash<const void*>{}(state.context) + 0x9e3779b9 + (result << 6) + (result >> 2);
    result ^= std::hash<bool>{}(state.use_onnx_schema_registry) + 0x9e3779b9 + (result << 6) + (result >> 2);
    return result;
  }
};

struct ValidatedFunctionStates {
  std::unordered_set<FunctionValidationState, FunctionValidationStateHash> states;
  std::unordered_set<std::shared_ptr<const AttributeBindingContext>,
                     AttributeBindingContextPtrHash,
                     AttributeBindingContextPtrEqual>
      contexts;
  std::unordered_set<BoundAttributeActiveState, BoundAttributeActiveStateHash>
      active_attribute_expansions;
  std::unordered_set<BoundAttributeExpansionState, BoundAttributeExpansionStateHash>
      completed_attribute_expansions;
};

const BoundAttribute* FindAttributeBinding(const AttributeBindings& bindings,
                                           std::string_view name) {
  const auto it = std::find_if(bindings.begin(), bindings.end(),
                               [name](const AttributeBinding& binding) {
                                 return binding.name == name;
                               });
  return it == bindings.end() ? nullptr : &it->attribute;
}

BoundAttribute ResolveAttribute(const ONNX_NAMESPACE::AttributeProto& attr,
                                const AttributeBindings& bindings,
                                const Graph* graph = nullptr) {
  if (attr.ref_attr_name().empty()) {
    return {&attr, graph, nullptr};
  }

  const auto* binding = FindAttributeBinding(bindings, attr.ref_attr_name());
  if (binding == nullptr) {
    return {};
  }

  return *binding;
}

void SetAttributeBinding(AttributeBindings& bindings,
                         std::string_view name,
                         BoundAttribute attribute) {
  const auto it = std::find_if(bindings.begin(), bindings.end(),
                               [name](const AttributeBinding& binding) {
                                 return binding.name == name;
                               });
  if (it == bindings.end()) {
    bindings.push_back({name, attribute});
  } else {
    it->attribute = attribute;
  }
}

std::shared_ptr<const AttributeBindingContext> InternRelevantAttributeBindingContext(
    const ONNX_NAMESPACE::AttributeProto& attribute,
    const AttributeBindings& bindings,
    const DomainToVersionMap& domain_to_version,
    ValidatedFunctionStates& validated_states) {
  InlinedHashSet<std::string_view> referenced_attribute_names;
  CollectReferencedAttributeNames(attribute, referenced_attribute_names);

  InlinedHashSet<std::string_view> expanded_attribute_names;
  bool added_dependencies = true;
  while (added_dependencies) {
    added_dependencies = false;
    for (const auto& binding : bindings) {
      if (referenced_attribute_names.find(binding.name) == referenced_attribute_names.end() ||
          !expanded_attribute_names.insert(binding.name).second ||
          binding.attribute.proto == nullptr) {
        continue;
      }

      const size_t previous_size = referenced_attribute_names.size();
      CollectReferencedAttributeNames(*binding.attribute.proto, referenced_attribute_names);
      added_dependencies = added_dependencies || referenced_attribute_names.size() != previous_size;
    }
  }

  AttributeBindings relevant_bindings;
  for (const auto& binding : bindings) {
    if (referenced_attribute_names.find(binding.name) != referenced_attribute_names.end()) {
      relevant_bindings.push_back(binding);
    }
  }

  auto candidate = std::make_shared<AttributeBindingContext>(
      AttributeBindingContext{std::move(relevant_bindings), domain_to_version});
  const auto [context_it, inserted] = validated_states.contexts.insert(std::move(candidate));
  ORT_UNUSED_PARAMETER(inserted);
  return *context_it;
}

void CompleteFunctionAttributeBindings(
    const ONNX_NAMESPACE::FunctionProto& function_proto,
    const InlinedHashSet<std::string_view>& explicit_binding_names,
    const DomainToVersionMap& caller_domain_to_version,
    AttributeBindings& callee_bindings,
    ValidatedFunctionStates& validated_states) {
  for (const auto& attr : function_proto.attribute_proto()) {
    if (FindAttributeBinding(callee_bindings, attr.name()) == nullptr) {
      SetAttributeBinding(callee_bindings, attr.name(), {&attr, nullptr, nullptr});
    }
  }

  std::sort(callee_bindings.begin(), callee_bindings.end(),
            [](const AttributeBinding& lhs, const AttributeBinding& rhs) {
              return lhs.name < rhs.name;
            });
  const AttributeBindings complete_bindings = callee_bindings;
  const auto callee_domain_to_version = GetFunctionDomainToVersionMap(function_proto);
  for (auto& binding : callee_bindings) {
    auto& attribute = binding.attribute;
    if (attribute.context == nullptr && CanContainGraph(attribute)) {
      const auto& defining_domain_to_version =
          explicit_binding_names.find(binding.name) != explicit_binding_names.end()
              ? caller_domain_to_version
              : callee_domain_to_version;
      attribute.context = InternRelevantAttributeBindingContext(
          *attribute.proto, complete_bindings, defining_domain_to_version, validated_states);
    }
  }
}

Status ValidateFunctionCallDepth(
    const ONNX_NAMESPACE::FunctionProto& function_proto,
    AttributeBindings bindings,
    size_t call_depth,
    const ModelLocalFunctions& model_local_functions,
    const IOnnxRuntimeOpSchemaCollection& schema_registry,
    ValidatedFunctionStates& validated_states);

Status ValidateProtoNodesCallDepth(
    const google::protobuf::RepeatedPtrField<ONNX_NAMESPACE::NodeProto>& nodes,
    const AttributeBindings& bindings,
    const DomainToVersionMap& domain_to_version,
    size_t call_depth,
    const ModelLocalFunctions& model_local_functions,
    const IOnnxRuntimeOpSchemaCollection& schema_registry,
    ValidatedFunctionStates& validated_states);

Status ValidateGraphCallDepth(
    const Graph& graph,
    const AttributeBindings& bindings,
    size_t call_depth,
    bool use_onnx_schema_registry,
    const ModelLocalFunctions& model_local_functions,
    const IOnnxRuntimeOpSchemaCollection& schema_registry,
    ValidatedFunctionStates& validated_states);

Status ValidateBoundAttributeCallDepth(
    BoundAttribute attribute,
    const AttributeBindings& bindings,
    const DomainToVersionMap& domain_to_version,
    size_t call_depth,
    bool use_onnx_schema_registry,
    const ModelLocalFunctions& model_local_functions,
    const IOnnxRuntimeOpSchemaCollection& schema_registry,
    ValidatedFunctionStates& validated_states) {
  if (attribute.proto == nullptr) {
    return Status::OK();
  }
  if (!CanContainGraph(attribute)) {
    return Status::OK();
  }

  if (attribute.context == nullptr) {
    attribute.context = InternRelevantAttributeBindingContext(
        *attribute.proto, bindings, domain_to_version, validated_states);
  }

  const BoundAttributeExpansionState expansion_state{
      attribute.proto, attribute.graph, attribute.context.get(), call_depth,
      validated_states.active_attribute_expansions.size(),
      use_onnx_schema_registry};
  if (validated_states.completed_attribute_expansions.find(expansion_state) !=
      validated_states.completed_attribute_expansions.end()) {
    return Status::OK();
  }
  const BoundAttributeActiveState active_state{
      attribute.proto, attribute.graph, attribute.context.get(),
      use_onnx_schema_registry};
  if (!validated_states.active_attribute_expansions.insert(active_state).second) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, INVALID_ARGUMENT,
        "Recursive model-local function graph attribute expansion is not supported.");
  }
  auto remove_active_expansion = gsl::finally([&validated_states, active_state]() {
    validated_states.active_attribute_expansions.erase(active_state);
  });
  if (expansion_state.attribute_expansion_depth >=
      kMaxModelLocalFunctionCallDepth) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, NOT_IMPLEMENTED,
        "Model local function graph attribute expansion depth exceeds the maximum supported depth of ",
        kMaxModelLocalFunctionCallDepth, ".");
  }

  const auto& attribute_bindings =
      attribute.context == nullptr ? bindings : attribute.context->bindings;
  const auto& attribute_domain_to_version =
      attribute.context == nullptr ? domain_to_version : attribute.context->domain_to_version;

  if (attribute.graph != nullptr) {
    ORT_RETURN_IF_ERROR(ValidateGraphCallDepth(
        *attribute.graph, attribute_bindings, call_depth, use_onnx_schema_registry,
        model_local_functions,
        schema_registry, validated_states));
  } else if (attribute.proto->has_g()) {
    ORT_RETURN_IF_ERROR(ValidateProtoNodesCallDepth(
        attribute.proto->g().node(), attribute_bindings, attribute_domain_to_version,
        call_depth, model_local_functions, schema_registry, validated_states));
  }

  for (const auto& graph : attribute.proto->graphs()) {
    ORT_RETURN_IF_ERROR(ValidateProtoNodesCallDepth(
        graph.node(), attribute_bindings, attribute_domain_to_version,
        call_depth, model_local_functions, schema_registry, validated_states));
  }

  validated_states.completed_attribute_expansions.insert(expansion_state);
  return Status::OK();
}

Status ValidateFunctionCallDepth(
    const ONNX_NAMESPACE::FunctionProto& function_proto,
    AttributeBindings bindings,
    size_t call_depth,
    const ModelLocalFunctions& model_local_functions,
    const IOnnxRuntimeOpSchemaCollection& schema_registry,
    ValidatedFunctionStates& validated_states) {
  if (call_depth > kMaxModelLocalFunctionCallDepth) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, NOT_IMPLEMENTED,
        "Model local function call depth ", call_depth,
        " exceeds the maximum supported depth of ", kMaxModelLocalFunctionCallDepth, ".");
  }

  for (const auto& attr : function_proto.attribute_proto()) {
    if (FindAttributeBinding(bindings, attr.name()) == nullptr) {
      SetAttributeBinding(bindings, attr.name(), {&attr, nullptr, nullptr});
    }
  }

  std::sort(bindings.begin(), bindings.end(),
            [](const AttributeBinding& lhs, const AttributeBinding& rhs) {
              return lhs.name < rhs.name;
            });

  InlinedHashSet<std::string_view> referenced_attribute_names;
  CollectReferencedAttributeNames(function_proto, referenced_attribute_names);

  AttributeBindings graph_bindings;
  for (const auto& binding : bindings) {
    if (CanContainGraph(binding.attribute) &&
        referenced_attribute_names.find(binding.name) != referenced_attribute_names.end()) {
      graph_bindings.push_back(binding);
    }
  }

  FunctionValidationState validation_state{
      &function_proto, call_depth,
      validated_states.active_attribute_expansions.size(),
      std::move(graph_bindings)};
  if (!validated_states.states.insert(std::move(validation_state)).second) {
    return Status::OK();
  }

  const auto domain_to_version = GetFunctionDomainToVersionMap(function_proto);
  return ValidateProtoNodesCallDepth(
      function_proto.node(), bindings, domain_to_version, call_depth,
      model_local_functions, schema_registry, validated_states);
}

Status ValidateProtoNodesCallDepth(
    const google::protobuf::RepeatedPtrField<ONNX_NAMESPACE::NodeProto>& nodes,
    const AttributeBindings& bindings,
    const DomainToVersionMap& domain_to_version,
    size_t call_depth,
    const ModelLocalFunctions& model_local_functions,
    const IOnnxRuntimeOpSchemaCollection& schema_registry,
    ValidatedFunctionStates& validated_states) {
  for (const auto& node : nodes) {
    const auto function_id = function_utils::GetFunctionIdentifier(
        node.domain(), node.op_type(), node.overload());
    const auto function_it = model_local_functions.find(function_id);
    if (function_it != model_local_functions.end() &&
        !HasOnnxRegisteredSchema(node.domain(), node.op_type(), domain_to_version)) {
      AttributeBindings callee_bindings;
      InlinedHashSet<std::string_view> explicit_binding_names;
      for (const auto& attr : node.attribute()) {
        auto resolved_attr = ResolveAttribute(attr, bindings);
        if (resolved_attr.proto != nullptr) {
          SetAttributeBinding(callee_bindings, attr.name(), resolved_attr);
          explicit_binding_names.insert(attr.name());
        }
      }
      CompleteFunctionAttributeBindings(
          *function_it->second, explicit_binding_names, domain_to_version,
          callee_bindings, validated_states);
      ORT_RETURN_IF_ERROR(ValidateFunctionCallDepth(
          *function_it->second, std::move(callee_bindings), call_depth + 1,
          model_local_functions, schema_registry, validated_states));
      for (const auto& attr : node.attribute()) {
        ORT_RETURN_IF_ERROR(ValidateBoundAttributeCallDepth(
            ResolveAttribute(attr, bindings), bindings, domain_to_version,
            call_depth, /*use_onnx_schema_registry*/ true,
            model_local_functions, schema_registry, validated_states));
      }
      continue;
    }

    for (const auto& attr : node.attribute()) {
      ORT_RETURN_IF_ERROR(ValidateBoundAttributeCallDepth(
          ResolveAttribute(attr, bindings), bindings, domain_to_version,
          call_depth, /*use_onnx_schema_registry*/ true,
          model_local_functions, schema_registry, validated_states));
    }
  }

  return Status::OK();
}

Status ValidateGraphCallDepth(
    const Graph& graph,
    const AttributeBindings& bindings,
    size_t call_depth,
    bool use_onnx_schema_registry,
    const ModelLocalFunctions& model_local_functions,
    const IOnnxRuntimeOpSchemaCollection& schema_registry,
    ValidatedFunctionStates& validated_states) {
  for (const auto& node : graph.Nodes()) {
    const auto function_id = function_utils::GetFunctionIdentifier(
        node.Domain(), node.OpType(), node.Overload());
    const auto function_it = model_local_functions.find(function_id);
    if (function_it != model_local_functions.end() &&
        !(use_onnx_schema_registry
              ? HasOnnxRegisteredSchema(node.Domain(), node.OpType(), graph.DomainToVersionMap())
              : HasRegisteredSchema(node.Domain(), node.OpType(), graph.DomainToVersionMap(), schema_registry))) {
      AttributeBindings callee_bindings;
      InlinedHashSet<std::string_view> explicit_binding_names;
      for (const auto& [attr_name, attr] : node.GetAttributes()) {
        const Graph* attribute_graph = nullptr;
        if (attr.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH || attr.has_g()) {
          attribute_graph = node.GetGraphAttribute(attr_name);
        }
        auto resolved_attr = ResolveAttribute(attr, bindings, attribute_graph);
        if (resolved_attr.proto != nullptr) {
          SetAttributeBinding(callee_bindings, attr_name, resolved_attr);
          explicit_binding_names.insert(attr_name);
        }
      }
      CompleteFunctionAttributeBindings(
          *function_it->second, explicit_binding_names, graph.DomainToVersionMap(),
          callee_bindings, validated_states);
      ORT_RETURN_IF_ERROR(ValidateFunctionCallDepth(
          *function_it->second, std::move(callee_bindings), call_depth + 1,
          model_local_functions, schema_registry, validated_states));
      for (const auto& [attr_name, attr] : node.GetAttributes()) {
        const Graph* attribute_graph = nullptr;
        if (attr.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH || attr.has_g()) {
          attribute_graph = node.GetGraphAttribute(attr_name);
        }
        ORT_RETURN_IF_ERROR(ValidateBoundAttributeCallDepth(
            ResolveAttribute(attr, bindings, attribute_graph),
            bindings, graph.DomainToVersionMap(), call_depth,
            use_onnx_schema_registry, model_local_functions,
            schema_registry, validated_states));
      }
      continue;
    }

    for (const auto& [attr_name, attr] : node.GetAttributes()) {
      const Graph* attribute_graph = nullptr;
      if (attr.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH || attr.has_g()) {
        attribute_graph = node.GetGraphAttribute(attr_name);
      }
      ORT_RETURN_IF_ERROR(ValidateBoundAttributeCallDepth(
          ResolveAttribute(attr, bindings, attribute_graph),
          bindings, graph.DomainToVersionMap(), call_depth,
          use_onnx_schema_registry, model_local_functions,
          schema_registry, validated_states));
    }
  }

  return Status::OK();
}

}  // namespace

Status ValidateModelSubgraphDepth(const ONNX_NAMESPACE::ModelProto& model_proto) {
  ORT_RETURN_IF_ERROR(ValidateSubgraphDepth(model_proto.graph().node()));
  for (const auto& function : model_proto.functions()) {
    ORT_RETURN_IF_ERROR(ValidateFunctionSubgraphDepth(function));
  }

  return Status::OK();
}

Status ValidateFunctionSubgraphDepth(const ONNX_NAMESPACE::FunctionProto& function_proto) {
  return ValidateSubgraphDepth(function_proto.node(), &function_proto.attribute_proto());
}

Status BuildLocalFunctionCallGraph(
    const std::unordered_map<std::string, const ONNX_NAMESPACE::FunctionProto*>& model_local_functions,
    LocalFunctionCallGraph& call_graph) {
  call_graph.reserve(model_local_functions.size());

  for (const auto& [function_id, function_proto] : model_local_functions) {
    if (function_proto == nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Null function proto for function id: ", function_id);
    }

    InlinedHashSet<std::string_view> seen_calls;
    InlinedVector<std::string_view> callees;
    CollectLocalFunctionCalls(function_proto->node(), model_local_functions, seen_calls, callees);

    call_graph.emplace(std::string_view(function_id), std::move(callees));
  }

  return Status::OK();
}

Status ValidateCallGraphAcyclic(const LocalFunctionCallGraph& call_graph) {
  enum class VisitState { kNotVisited,
                          kVisiting,
                          kVisited };

  InlinedHashMap<std::string_view, VisitState> visit_states;
  visit_states.reserve(call_graph.size());
  for (const auto& [function_id, _] : call_graph) {
    ORT_UNUSED_PARAMETER(_);
    visit_states.emplace(function_id, VisitState::kNotVisited);
  }

  // Each frame records the function being visited and a pointer to its callees vector
  // in the call graph (no per-frame allocation).
  struct DfsFrame {
    std::string_view function_id;
    const InlinedVector<std::string_view>* callees;
    size_t next_callee_index;
  };

  std::vector<DfsFrame> dfs_stack;

  for (const auto& [root_id, root_callees] : call_graph) {
    auto root_state_it = visit_states.find(root_id);
    if (root_state_it == visit_states.end() || root_state_it->second == VisitState::kVisited) {
      continue;
    }

    root_state_it->second = VisitState::kVisiting;
    dfs_stack.push_back({root_id, &root_callees, 0});

    while (!dfs_stack.empty()) {
      auto& frame = dfs_stack.back();

      if (frame.next_callee_index >= frame.callees->size()) {
        // All callees processed — mark as fully visited and pop.
        auto it = visit_states.find(frame.function_id);
        ORT_ENFORCE(it != visit_states.end());
        it->second = VisitState::kVisited;
        dfs_stack.pop_back();
        continue;
      }

      std::string_view callee_id = (*frame.callees)[frame.next_callee_index];
      frame.next_callee_index++;

      auto callee_state_it = visit_states.find(callee_id);
      if (callee_state_it == visit_states.end()) {
        // Callee not in the graph — skip.
        continue;
      }

      if (callee_state_it->second == VisitState::kVisited) {
        continue;
      }

      if (callee_state_it->second == VisitState::kVisiting) {
        // Cycle detected. Build cycle description from the stack.
        std::string cycle;
        bool in_cycle = false;
        for (const auto& f : dfs_stack) {
          if (f.function_id == callee_id) {
            in_cycle = true;
          }
          if (in_cycle) {
            if (!cycle.empty()) {
              cycle.append(" -> ");
            }
            cycle.append(f.function_id);
          }
        }
        cycle.append(" -> ");
        cycle.append(callee_id);

        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH,
                               "Model local function definitions must not be recursive. Cycle detected: ", cycle);
      }

      // Push callee onto the DFS stack.
      auto callee_graph_it = call_graph.find(callee_id);
      if (callee_graph_it == call_graph.end()) {
        continue;
      }

      callee_state_it->second = VisitState::kVisiting;
      dfs_stack.push_back({callee_id, &callee_graph_it->second, 0});
    }
  }

  return Status::OK();
}

Status ValidateCallGraphDepth(const LocalFunctionCallGraph& call_graph,
                              gsl::span<const std::string_view> roots) {
  InlinedHashMap<std::string_view, size_t> call_depths;
  call_depths.reserve(call_graph.size());
  InlinedHashSet<std::string_view> visited;
  InlinedVector<std::string_view> postorder;

  struct DfsFrame {
    std::string_view function_id;
    size_t next_callee_index;
  };
  InlinedVector<DfsFrame> dfs_stack;

  for (const auto root_id : roots) {
    if (call_graph.find(root_id) == call_graph.end()) {
      continue;
    }
    if (!visited.insert(root_id).second) {
      continue;
    }

    dfs_stack.push_back({root_id, 0});
    while (!dfs_stack.empty()) {
      auto& frame = dfs_stack.back();
      const auto function_it = call_graph.find(frame.function_id);
      if (function_it == call_graph.end() || frame.next_callee_index >= function_it->second.size()) {
        postorder.push_back(frame.function_id);
        dfs_stack.pop_back();
        continue;
      }

      const auto callee_id = function_it->second[frame.next_callee_index++];
      if (call_graph.find(callee_id) != call_graph.end() && visited.insert(callee_id).second) {
        dfs_stack.push_back({callee_id, 0});
      }
    }
  }

  for (const auto function_id : postorder) {
    size_t call_depth = 1;
    const auto function_it = call_graph.find(function_id);
    ORT_ENFORCE(function_it != call_graph.end());
    for (const auto callee_id : function_it->second) {
      const auto callee_depth_it = call_depths.find(callee_id);
      if (callee_depth_it != call_depths.end()) {
        call_depth = std::max(call_depth, callee_depth_it->second + 1);
      }
    }

    call_depths.emplace(function_id, call_depth);
  }

  for (const auto root_id : roots) {
    const auto depth_it = call_depths.find(root_id);
    if (depth_it != call_depths.end() && depth_it->second > kMaxModelLocalFunctionCallDepth) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, NOT_IMPLEMENTED,
          "Model local function call depth ", depth_it->second,
          " exceeds the maximum supported depth of ", kMaxModelLocalFunctionCallDepth, ".");
    }
  }

  return Status::OK();
}

Status ValidateModelLocalFunctionAcyclic(
    const std::unordered_map<std::string, const ONNX_NAMESPACE::FunctionProto*>& model_local_functions) {
  LocalFunctionCallGraph call_graph;
  ORT_RETURN_IF_ERROR(BuildLocalFunctionCallGraph(model_local_functions, call_graph));
  return ValidateCallGraphAcyclic(call_graph);
}

Status ValidateModelLocalFunctionCallDepth(
    const std::unordered_map<std::string, const ONNX_NAMESPACE::FunctionProto*>& model_local_functions,
    const Graph& main_graph) {
  if (model_local_functions.empty()) {
    return Status::OK();
  }

  LocalFunctionCallGraph call_graph;
  ORT_RETURN_IF_ERROR(BuildLocalFunctionCallGraph(model_local_functions, call_graph));
  ORT_RETURN_IF_ERROR(ValidateCallGraphAcyclic(call_graph));
  ValidatedFunctionStates validated_states;
  return ValidateGraphCallDepth(
      main_graph, {}, 0, /*use_onnx_schema_registry*/ false,
      model_local_functions,
      *main_graph.GetSchemaRegistry(), validated_states);
}

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
