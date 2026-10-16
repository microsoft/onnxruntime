// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/optimizer/pattern_rewriter.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <iterator>
#include <limits>
#include <optional>
#include <sstream>
#include <type_traits>

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "onnx/checker.h"

namespace onnxruntime {
namespace rewrite {
namespace {

std::string Key(const std::string& op, const std::string& domain) { return domain + '\0' + op; }

std::string Signature(const Pattern& pattern) {
  std::ostringstream stream;
  stream << pattern.ValueCount() << ':';
  for (Value output : pattern.Outputs()) stream << 'r' << output.handle << ',';
  for (int id = 0; id < static_cast<int>(pattern.ValueCount()); ++id)
    if (pattern.IsOptional(id)) stream << '?' << id << ',';
  stream.precision(std::numeric_limits<float>::max_digits10);
  for (int id = 0; id < static_cast<int>(pattern.ValueCount()); ++id) {
    auto it = pattern.Constants().find(id);
    if (it == pattern.Constants().end()) continue;
    stream << 'c' << id << ':' << it->second.index() << ':';
    std::visit([&](const auto& value) {
      using T = std::decay_t<decltype(value)>;
      if constexpr (std::is_arithmetic_v<T>) {
        stream << value;
      } else {
        stream << value.size() << '[';
        for (auto element : value) stream << element << ',';
        stream << ']';
      }
    },
               it->second);
    stream << ';';
  }
  for (const auto& node : pattern.Nodes()) {
    stream << node.op.domain.size() << ':' << node.op.domain << node.op.name.size() << ':' << node.op.name;
    stream << '/';
    for (Value output : node.outputs) stream << output << ',';
    stream << '/';
    for (Value input : node.inputs) stream << ',' << input;
    stream << ';';
  }
  return stream.str();
}

using Producers = InlinedHashMap<const NodeArg*, NodeIndex>;

bool MatchType(const Pattern& pattern, int handle, const TypeConstraint& constraint, Match& match) {
  if (pattern.IsAttribute(handle)) {
    return match.attributeOf(constraint.value_name).type() == constraint.attribute_type;
  }
  const NodeArg* value = match.values[handle];
  if (constraint.absent) return !value;
  if (!value) return false;
  const auto* type = value->TypeAsProto();
  if (!type || !type->has_tensor_type()) return false;
  const int32_t element = type->tensor_type().elem_type();
  if (constraint.element_type && constraint.element_type != element) return false;
  if (!constraint.element_variable.empty() && !match.BindElementType(constraint.element_variable, element)) return false;
  if (constraint.any_rank) return true;
  const auto* shape = value->Shape();
  if (!shape || shape->dim_size() < static_cast<int>(constraint.dimensions.size()) ||
      (!constraint.prefix && shape->dim_size() != static_cast<int>(constraint.dimensions.size())))
    return false;
  for (size_t i = 0; i < constraint.dimensions.size(); ++i) {
    const auto& expected = constraint.dimensions[i];
    const auto& actual = shape->dim(static_cast<int>(i));
    if (const auto* fixed = std::get_if<int64_t>(&expected)) {
      if (!actual.has_dim_value() || actual.dim_value() != *fixed) return false;
    } else if (const auto* name = std::get_if<std::string>(&expected)) {
      Dimension binding;
      if (actual.has_dim_value())
        binding = actual.dim_value();
      else if (actual.has_dim_param() && !actual.dim_param().empty())
        binding = actual.dim_param();
      if (!match.BindDimension(*name, std::move(binding))) return false;
    }
  }
  return true;
}

bool MatchTypes(const Pattern& pattern, Match& match) {
  InlinedVector<int> handles;
  for (const auto& [handle, alternatives] : pattern.TypeConstraints()) {
    handles.push_back(handle);
    for (const auto& constraint : alternatives) {
      if (!constraint.element_variable.empty()) match.BindElementType(constraint.element_variable, 0);
      for (const auto& dim : constraint.dimensions)
        if (const auto* name = std::get_if<std::string>(&dim)) match.BindDimension(*name, std::monostate{});
    }
  }
  std::sort(handles.begin(), handles.end());
  std::function<bool(size_t, Match&)> search = [&](size_t index, Match& state) {
    if (index == handles.size()) return true;
    const int handle = handles[index];
    for (const auto& constraint : pattern.TypeConstraints().at(handle)) {
      Match candidate = state;
      if (MatchType(pattern, handle, constraint, candidate) && search(index + 1, candidate)) {
        state = std::move(candidate);
        return true;
      }
    }
    return false;
  };
  return search(0, match);
}

bool MatchPattern(const Pattern& pattern, const Graph& graph, const Node& root, const Producers& producers,
                  Match& match, FailureStage& stage, std::string_view& reason) {
  auto fail = [&](FailureStage failure_stage, std::string_view message) {
    stage = failure_stage;
    reason = message;
    return false;
  };
  const auto& root_op = pattern.Nodes().back().op;
  // Cheap root rejection precedes binding allocation in the reference implementation too.
  if (root.OpType() != root_op.name || root.Domain() != root_op.domain) {
    return fail(FailureStage::Structure, "Root operator/domain mismatch.");
  }
  match.values.assign(pattern.ValueCount(), nullptr);
  match.nodes.assign(pattern.Nodes().size(), nullptr);
  match.nodes.back() = &root;
  InlinedVector<bool> bound(pattern.ValueCount(), false);
  InlinedHashSet<NodeIndex> distinct;
  for (size_t i = pattern.Nodes().size(); i-- > 0;) {
    const auto& expected = pattern.Nodes()[i];
    const Node* node = match.nodes[i];
    if (!node) {
      for (Value output : expected.outputs) {
        if (!match.values[output]) continue;
        auto it = producers.find(match.values[output]);
        if (it != producers.end()) node = graph.GetNode(it->second);
        break;
      }
    }
    if (!node) return fail(FailureStage::Structure, "No producer for the expected node.");
    if (!distinct.insert(node->Index()).second)
      return fail(FailureStage::Structure, "Distinct pattern nodes mapped to one model node.");
    if (node->OpType() != expected.op.name || node->Domain() != expected.op.domain)
      return fail(FailureStage::Structure, "Producer operator/domain mismatch.");
    if (node->InputDefs().size() > expected.inputs.size() || node->OutputDefs().size() != expected.outputs.size())
      return fail(FailureStage::Structure, "Operator input/output arity mismatch.");
    if (!node->GetAttributeNameToSubgraphMap().empty() || !node->ImplicitInputDefs().empty())
      return fail(FailureStage::Structure, "Matched operators cannot own subgraphs or implicit inputs.");
    if (node->GetLayeringAnnotation() != root.GetLayeringAnnotation())
      return fail(FailureStage::Structure, "Layer annotation mismatch.");
    match.nodes[i] = node;
    for (const auto& attribute : expected.attributes) {
      auto found = node->GetAttributes().find(attribute.name);
      if (found == node->GetAttributes().end() || !attribute.Matches(found->second))
        return fail(FailureStage::Attribute, "Missing attribute or attribute constraint mismatch.");
      if (!attribute.capture.empty() && !match.BindAttribute(attribute.capture, found->second))
        return fail(FailureStage::Attribute, "Repeated attribute capture mismatch.");
    }
    for (size_t j = 0; j < expected.outputs.size(); ++j) {
      const NodeArg* output = node->OutputDefs()[j];
      const NodeArg*& binding = match.values[expected.outputs[j]];
      if (!output->Exists() || (binding && binding != output))
        return fail(FailureStage::Structure, "Output slot or repeated value mismatch.");
      binding = output;
      bound[expected.outputs[j].handle] = true;
    }
    for (size_t j = 0; j < expected.inputs.size(); ++j) {
      const NodeArg* actual = j < node->InputDefs().size() && node->InputDefs()[j]->Exists()
                                  ? node->InputDefs()[j]
                                  : nullptr;
      const Value value = expected.inputs[j];
      if (value.handle < 0) {
        if (actual) return fail(FailureStage::Structure, "Input slot must be absent.");
        continue;
      }
      const NodeArg*& binding = match.values[value];
      if ((!actual && !pattern.IsOptional(value)) || (bound[value.handle] && binding != actual))
        return fail(FailureStage::Structure, "Required input absent or repeated value mismatch.");
      binding = actual;
      bound[value.handle] = true;
    }
  }
  // Input variables must be outside the deletion set, even when aliasing another binding.
  if (!MatchTypes(pattern, match))
    return fail(FailureStage::TypeShape, "Tensor type/rank/dimension or captured attribute type mismatch.");
  for (int value = 0; value < static_cast<int>(pattern.ValueCount()); ++value) {
    if (!pattern.IsInput(value)) continue;
    auto it = producers.find(match.values[value]);
    if (it != producers.end() && distinct.count(it->second))
      return fail(FailureStage::Boundary, "Pattern input is produced inside the deletion set.");
  }
  for (const auto& [handle, expected] : pattern.Constants()) {
    const auto* tensor = graph_utils::GetConstantInitializer(graph, match.values[handle]->Name());
    const bool vector = expected.index() >= 2;
    if (!tensor || tensor->dims_size() != (vector ? 1 : 0) ||
        tensor->data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL)
      return fail(FailureStage::Constant, "Literal initializer rank, mutability or storage mismatch.");
    const bool floating = std::holds_alternative<float>(expected) ||
                          std::holds_alternative<InlinedVector<float>>(expected);
    if (tensor->data_type() !=
        (floating ? ONNX_NAMESPACE::TensorProto_DataType_FLOAT : ONNX_NAMESPACE::TensorProto_DataType_INT64))
      return fail(FailureStage::Constant, "Literal initializer dtype mismatch.");
    Initializer initializer(graph, *tensor, graph.ModelPath());
    const bool equal = std::visit([&](const auto& value) {
      using T = std::decay_t<decltype(value)>;
      if constexpr (std::is_arithmetic_v<T>) {
        return initializer.size() == 1 && *initializer.data<T>() == value;
      } else {
        using Element = typename T::value_type;
        return tensor->dims(0) == static_cast<int64_t>(value.size()) && initializer.size() == value.size() &&
               std::equal(value.begin(), value.end(), initializer.data<Element>());
      }
    },
                                  expected);
    if (!equal) return fail(FailureStage::Constant, "Literal initializer values mismatch.");
  }
  InlinedHashSet<const NodeArg*> exposed;
  for (Value value : pattern.Outputs()) exposed.insert(match.values[value]);
  for (const Node* matched : match.nodes) {
    const Node& node = *matched;
    for (const NodeArg* output : node.OutputDefs()) {
      if (!exposed.count(output) &&
          std::find(graph.GetOutputs().begin(), graph.GetOutputs().end(), output) != graph.GetOutputs().end())
        return fail(FailureStage::Boundary, "Unreturned value is a graph output.");
    }
    for (auto edge = node.OutputEdgesBegin(); edge != node.OutputEdgesEnd(); ++edge) {
      if (!distinct.count(edge->GetNode().Index()) && !exposed.count(node.OutputDefs()[edge->GetSrcArgIndex()]))
        return fail(FailureStage::Boundary, "Unreturned value has an external consumer or capture.");
    }
  }
  return true;
}

bool CanRewritePartition(const Match& match, const Options& options) {
  const auto& provider = match.nodes.back()->GetExecutionProviderType();
  if (!provider.empty() && !options.compatible_execution_providers.count(provider)) return false;
  return std::all_of(match.nodes.begin(), match.nodes.end(),
                     [&](const Node* node) { return node->GetExecutionProviderType() == provider; });
}

Status Replace(Graph& graph, const Pattern& pattern, const Match& match, const Replacement& recipe,
               Producers& producers, InlinedVector<NodeIndex>& added) {
  const auto& nodes = *recipe.Plan();
  ORT_RETURN_IF(nodes.empty(), "Replacements must contain at least one node.");
  const Node& root = *match.nodes.back();
  ORT_RETURN_IF(recipe.Outputs().size() != pattern.Outputs().size(),
                "Replacement return count must equal the pattern return count.");
  InlinedVector<InlinedVector<NodeArg*>> exposed_outputs;
  for (const auto& node : nodes) exposed_outputs.emplace_back(node.output_count, nullptr);
  for (size_t i = 0; i < recipe.Outputs().size(); ++i) {
    const auto& output = recipe.Outputs()[i];
    ORT_RETURN_IF(output.node >= nodes.size() || output.slot >= nodes[output.node].output_count,
                  "Invalid replacement return slot.");
    exposed_outputs[output.node][output.slot] = graph.GetNodeArg(match.values[pattern.Outputs()[i]]->Name());
  }
  InlinedVector<InlinedVector<NodeArg*>> external_inputs;
  for (size_t i = 0; i < nodes.size(); ++i) {
    const auto& node = nodes[i];
    for (const auto& [name, binding] : node.attribute_bindings) {
      ORT_RETURN_IF(name.empty() || node.attributes.count(name), "Invalid or duplicate replacement attribute: ", name);
      match.attributeOf(binding);
    }
    const auto& domain = node.domain == kOnnxDomainAlias ? kOnnxDomain : node.domain;
    ORT_RETURN_IF(node.op.empty() || node.inputs.size() > 64 || node.output_count == 0 || node.output_count > 64,
                  "Invalid replacement operator or arity.");
    ORT_RETURN_IF(graph.DomainToVersionMap().find(domain) == graph.DomainToVersionMap().end(),
                  "Replacement domain is not imported: ", domain);
    InlinedVector<NodeArg*> inputs;
    for (const auto& input : node.inputs) {
      if (std::holds_alternative<std::monostate>(input)) {
        inputs.push_back(&graph.GetOrCreateNodeArg("", nullptr));
        continue;
      }
      if (const auto* scalar = std::get_if<Replacement::Scalar>(&input)) {
        ORT_RETURN_IF(std::holds_alternative<float>(scalar->value) && !std::isfinite(std::get<float>(scalar->value)),
                      "Replacement float constants must be finite.");
        inputs.push_back(nullptr);
        continue;
      }
      if (const auto* output = std::get_if<Replacement::Output>(&input)) {
        ORT_RETURN_IF(output->node >= i || output->slot >= nodes[output->node].output_count,
                      "Replacement inputs must reference an earlier output slot.");
        inputs.push_back(nullptr);
        continue;
      }
      Value value;
      if (const auto* handle = std::get_if<Value>(&input)) {
        value = *handle;
      } else {
        value = pattern.NamedValue(std::get<std::string>(input));
      }
      ORT_RETURN_IF(
          value.handle < 0 || static_cast<size_t>(value.handle) >= match.values.size() || !pattern.IsInput(value) ||
              pattern.IsAttribute(value),
          "Replacement inputs must be pattern input variables.");
      const NodeArg* binding = match.values[value];
      inputs.push_back(binding ? graph.GetNodeArg(binding->Name()) : &graph.GetOrCreateNodeArg("", nullptr));
    }
    external_inputs.push_back(std::move(inputs));
  }
  // The unique sink is independent of the explicitly exposed output slots.
  InlinedHashSet<size_t> reachable{nodes.size() - 1};
  for (size_t i = nodes.size(); i-- > 0;) {
    ORT_RETURN_IF(!reachable.count(i), "Disconnected replacement node.");
    for (const auto& input : nodes[i].inputs)
      if (const auto* output = std::get_if<Replacement::Output>(&input)) reachable.insert(output->node);
  }
  InlinedVector<std::string> initializers;
  auto rollback = [&]() {
    for (auto it = added.rbegin(); it != added.rend(); ++it) graph.RemoveNode(*it);
    for (const auto& name : initializers) graph.RemoveInitializedTensor(name);
    added.clear();
  };
  // Stage and validate every schema before deleting any matched nodes or changing edges.
  for (size_t i = 0; i < nodes.size(); ++i) {
    auto& inputs = external_inputs[i];
    for (size_t j = 0; j < inputs.size(); ++j) {
      if (const auto* output = std::get_if<Replacement::Output>(&nodes[i].inputs[j]))
        inputs[j] = graph.GetNode(added[output->node])->MutableOutputDefs()[output->slot];
      if (const auto* scalar = std::get_if<Replacement::Scalar>(&nodes[i].inputs[j])) {
        ONNX_NAMESPACE::TensorProto tensor;
        tensor.set_name(graph.GenerateNodeArgName("rewrite_literal"));
        std::visit([&](const auto& value) {
          using T = std::decay_t<decltype(value)>;
          if constexpr (std::is_arithmetic_v<T>) {
            if constexpr (std::is_same_v<T, float>) {
              tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
              tensor.add_float_data(value);
            } else {
              tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
              tensor.add_int64_data(value);
            }
          } else {
            tensor.add_dims(static_cast<int64_t>(value.size()));
            if constexpr (std::is_same_v<typename T::value_type, float>) {
              tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
              for (auto element : value) tensor.add_float_data(element);
            } else {
              tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
              for (auto element : value) tensor.add_int64_data(element);
            }
          }
        },
                   scalar->value);
        ONNX_NAMESPACE::TypeProto type;
        type.mutable_tensor_type()->set_elem_type(tensor.data_type());
        auto* shape = type.mutable_tensor_type()->mutable_shape();
        if (tensor.dims_size()) shape->add_dim()->set_dim_value(tensor.dims(0));
        inputs[j] = &graph.GetOrCreateNodeArg(tensor.name(), &type);
        initializers.push_back(tensor.name());
        graph.AddInitializedTensor(tensor);
      }
    }
    InlinedVector<NodeArg*> outputs;
    for (size_t j = 0; j < nodes[i].output_count; ++j)
      outputs.push_back(exposed_outputs[i][j] ? exposed_outputs[i][j]
                                              : &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("rewrite_value"), nullptr));
    const auto& node = nodes[i];
    const auto& domain = node.domain == kOnnxDomainAlias ? kOnnxDomain : node.domain;
    NodeAttributes resolved_attributes;
    const NodeAttributes* attributes = &node.attributes;
    if (!node.attribute_bindings.empty()) {
      resolved_attributes = node.attributes;
      for (const auto& [name, binding] : node.attribute_bindings) {
        auto value = match.attributeOf(binding);
        value.set_name(name);
        resolved_attributes.emplace(name, std::move(value));
      }
      attributes = &resolved_attributes;
    }
    Node& replacement = graph.AddNode(graph.GenerateNodeName("rewrite"), node.op, "PatternRewriter", inputs, outputs,
                                      root, attributes, domain);
    added.push_back(replacement.Index());
    if (!graph.SetOpSchemaFromRegistryForNode(replacement)) {
      rollback();
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Replacement schema is unavailable: ", node.op);
    }
    ORT_TRY {
      ONNX_NAMESPACE::NodeProto proto;
      replacement.ToProto(proto);
      replacement.Op()->Verify(proto);
    }
    ORT_CATCH(const ONNX_NAMESPACE::checker::ValidationError& error) {
      rollback();
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid replacement: ", error.what());
    }
    replacement.SetExecutionProviderType(root.GetExecutionProviderType());
  }
  // External consumers of a returned intermediate may feed a pattern input.
  // Validate the prospective producer mapping before replacing their edges.
  if (std::any_of(pattern.Outputs().begin(), pattern.Outputs().end(), [&](Value value) {
        const auto& root_outputs = pattern.Nodes().back().outputs;
        return std::find(root_outputs.begin(), root_outputs.end(), value) == root_outputs.end();
      })) {
    InlinedHashMap<const NodeArg*, NodeIndex> staged_producers;
    for (NodeIndex index : added)
      for (const NodeArg* output : graph.GetNode(index)->OutputDefs()) staged_producers[output] = index;
    InlinedHashMap<NodeIndex, uint8_t> colors;
    InlinedVector<std::pair<NodeIndex, bool>> pending;
    for (NodeIndex index : added) {
      pending.emplace_back(index, false);
      while (!pending.empty()) {
        auto [current, exiting] = pending.back();
        pending.pop_back();
        if (exiting) {
          colors[current] = 2;
          continue;
        }
        if (colors[current] == 2) continue;
        if (colors[current] == 1) {
          rollback();
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Replacement output mapping introduces a cycle.");
        }
        colors[current] = 1;
        pending.emplace_back(current, true);
        const Node& node = *graph.GetNode(current);
        auto visit_input = [&](const NodeArg* input) {
          auto staged = staged_producers.find(input);
          if (staged != staged_producers.end()) {
            pending.emplace_back(staged->second, false);
          } else {
            auto existing = producers.find(input);
            if (existing != producers.end()) pending.emplace_back(existing->second, false);
          }
        };
        for (const NodeArg* input : node.InputDefs()) visit_input(input);
        for (const NodeArg* input : node.ImplicitInputDefs()) visit_input(input);
      }
    }
  }
  struct Edge {
    const NodeArg* value;
    NodeIndex destination;
    int destination_arg;
  };
  InlinedVector<Edge> outgoing;
  InlinedHashSet<NodeIndex> deletion_set;
  for (const Node* node : match.nodes) deletion_set.insert(node->Index());
  for (const Node* node : match.nodes) {
    for (auto it = node->OutputEdgesBegin(); it != node->OutputEdgesEnd(); ++it) {
      if (!deletion_set.count(it->GetNode().Index()))
        outgoing.push_back({node->OutputDefs()[it->GetSrcArgIndex()], it->GetNode().Index(), it->GetDstArgIndex()});
    }
  }
  InlinedVector<NodeIndex> removed;
  for (const Node* node : match.nodes) {
    removed.push_back(node->Index());
    for (const NodeArg* output : node->OutputDefs()) producers.erase(output);
    for (const NodeArg* input : node->InputDefs()) {
      if (!input->Exists()) continue;
      graph.RemoveConsumerNode(input->Name(), graph.GetNode(node->Index()));
    }
    graph_utils::RemoveNodeOutputEdges(graph, *graph.GetNode(node->Index()));
  }
  // Remove in reverse order so producers cannot be left with dangling edges.
  for (auto it = removed.rbegin(); it != removed.rend(); ++it) {
    ORT_ENFORCE(graph.RemoveNode(*it), "Failed to remove matched node.");
  }
  for (NodeIndex index : added) {
    Node& replacement = *graph.GetNode(index);
    for (size_t i = 0; i < replacement.InputDefs().size(); ++i) {
      const NodeArg* input = replacement.InputDefs()[i];
      if (!input->Exists()) continue;
      graph.AddConsumerNode(input->Name(), &replacement);
      auto it = producers.find(input);
      if (it == producers.end()) continue;
      const Node& producer = *graph.GetNode(it->second);
      auto output_it = std::find(producer.OutputDefs().begin(), producer.OutputDefs().end(), input);
      graph.AddEdge(producer.Index(), index, static_cast<int>(std::distance(producer.OutputDefs().begin(), output_it)),
                    static_cast<int>(i));
    }
    for (const NodeArg* output : replacement.OutputDefs()) {
      producers[output] = index;
      graph.UpdateProducerNode(output->Name(), index);
    }
  }
  for (const auto& edge : outgoing) {
    const NodeIndex source = producers.at(edge.value);
    const auto& outputs = graph.GetNode(source)->OutputDefs();
    const auto slot = std::distance(outputs.begin(), std::find(outputs.begin(), outputs.end(), edge.value));
    graph.AddEdge(source, edge.destination, static_cast<int>(slot), edge.destination_arg);
  }
  graph.NotifyNodeReplacement(removed, added.back());
  return Status::OK();
}
}  // namespace

AttributeMatcher AttributeMatcher::Equal(std::string name, int64_t value) {
  return {std::move(name), [value](const auto& attribute) {
            return attribute.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_INT && attribute.i() == value;
          }};
}

AttributeMatcher AttributeMatcher::Equal(std::string name, float value) {
  ORT_ENFORCE(std::isfinite(value), "Attribute float constants must be finite.");
  return {std::move(name), [value](const auto& attribute) {
            return attribute.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT && attribute.f() == value;
          }};
}

AttributeMatcher AttributeMatcher::Equal(std::string name, std::string value) {
  return {std::move(name), [value = std::move(value)](const auto& attribute) {
            return attribute.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_STRING && attribute.s() == value;
          }};
}

Value Pattern::AddValue(std::string_view name) {
  ORT_ENFORCE(value_count_ < 256, "Patterns are limited to 256 values.");
  if (!name.empty()) {
    ORT_ENFORCE(std::none_of(values_.begin(), values_.end(), [&](const auto& value) { return value.name == name; }),
                "Duplicate pattern value name: ", name);
  }
  Value value{std::string(name), static_cast<int>(value_count_++)};
  values_.push_back(value);
  return value;
}

Value Pattern::Input(std::string_view name) { return AddValue(name); }

Value Pattern::Constant(float constant) {
  ORT_ENFORCE(std::isfinite(constant), "Pattern float constants must be finite.");
  auto value = AddValue("");
  constants_.emplace(value.handle, constant);
  return value;
}

Value Pattern::Constant(int64_t constant) {
  auto value = AddValue("");
  constants_.emplace(value.handle, constant);
  return value;
}

Value Pattern::Constant(Literal constant) {
  auto value = AddValue("");
  constants_.emplace(value.handle, std::move(constant));
  return value;
}

Value Pattern::NamedValue(std::string_view name) const {
  auto it = std::find_if(values_.begin(), values_.end(), [&](const auto& value) { return value.name == name; });
  ORT_ENFORCE(!name.empty() && it != values_.end(), "Unknown pattern value: ", name);
  return *it;
}

void Pattern::Name(Value value, std::string_view name) {
  ORT_ENFORCE(value.handle >= 0 && static_cast<size_t>(value.handle) < value_count_ && !name.empty(),
              "Invalid pattern value or empty name.");
  ORT_ENFORCE(std::none_of(values_.begin(), values_.end(),
                           [&](const auto& other) { return other.handle != value.handle && other.name == name; }),
              "Duplicate pattern value name: ", name);
  values_[value.handle].name = name;
  for (auto& node : nodes_) {
    for (auto& output : node.outputs)
      if (output.handle == value.handle) output.name = name;
    for (auto& input : node.inputs)
      if (input.handle == value.handle) input.name = name;
  }
}

void Pattern::Attributes(InlinedVector<AttributeMatcher, 1> matchers) {
  ORT_ENFORCE(!nodes_.empty(), "Attribute matchers require a pattern node.");
  InlinedHashSet<std::string> names;
  for (const auto& attr : nodes_.back().attributes) names.insert(attr.name);
  for (const auto& attr : matchers) {
    ORT_ENFORCE(!attr.name.empty() && attr.predicate && names.insert(attr.name).second,
                "Invalid or duplicate attribute matcher: ", attr.name);
  }
  auto& target = nodes_.back().attributes;
  target.insert(target.end(), std::make_move_iterator(matchers.begin()), std::make_move_iterator(matchers.end()));
}

InlinedVector<Value> Pattern::MultiOp(Operator op, gsl::span<const Value> inputs, size_t output_count) {
  ORT_ENFORCE(!op.name.empty(), "Pattern operators need a name.");
  ORT_ENFORCE(nodes_.size() < 64, "Patterns are limited to 64 nodes.");
  ORT_ENFORCE(output_count > 0 && output_count <= 64, "Pattern operators need between 1 and 64 outputs.");
  ORT_ENFORCE(value_count_ <= 256 - output_count && inputs.size() <= 64,
              "Patterns are limited to 256 values and 64 inputs per operator.");
  if (op.domain == kOnnxDomainAlias) op.domain = kOnnxDomain;
  for (Value input : inputs)
    ORT_ENFORCE(input.handle == -1 || (input.handle >= 0 && static_cast<size_t>(input.handle) < value_count_),
                "Pattern inputs must already be defined.");
  InlinedVector<Value> outputs;
  for (size_t i = 0; i < output_count; ++i) outputs.push_back(AddValue(""));
  InlinedVector<Value> canonical_inputs;
  for (Value input : inputs) canonical_inputs.push_back(input.handle < 0 ? input : values_[input.handle]);
  nodes_.push_back({std::move(op), std::move(canonical_inputs), outputs, {}});
  return outputs;
}

bool Pattern::IsInput(Value value) const {
  if (value.handle < 0 || static_cast<size_t>(value.handle) >= value_count_) return false;
  return std::none_of(nodes_.begin(), nodes_.end(), [value](const auto& node) {
    return std::find(node.outputs.begin(), node.outputs.end(), value) != node.outputs.end();
  });
}

namespace detail {
class PatternParser {
 public:
  explicit PatternParser(std::string_view text, InlinedVector<NodeAttributes, 1>* attributes = nullptr,
                         InlinedVector<InlinedHashMap<std::string, std::string>, 1>* bindings = nullptr)
      : text_(text), replacement_attributes_(attributes), replacement_bindings_(bindings) {}

  Status Parse(Pattern& destination) {
    ORT_RETURN_IF_ERROR(Next());
    ORT_RETURN_IF_ERROR(Expect("("));
    if (token_ != ")") {
      do {
        std::string name;
        ORT_RETURN_IF_ERROR(Identifier(name));
        ORT_RETURN_IF(symbols_.count(name), Error("duplicate input name"));
        ORT_RETURN_IF(pattern_.ValueCount() >= 256, Error("too many values"));
        Value input = pattern_.Input(name);
        symbols_.emplace(name, input);
        if (token_ == ":") {
          InlinedVector<TypeConstraint, 1> alternatives;
          ORT_RETURN_IF_ERROR(Annotation(name, alternatives));
          if (std::any_of(alternatives.begin(), alternatives.end(), [](const auto& type) { return type.absent; }))
            pattern_.optional_inputs_.insert(input.handle);
          pattern_.type_constraints_.emplace(input.handle, std::move(alternatives));
        }
        if (token_ != ",") break;
        ORT_RETURN_IF_ERROR(Next());
      } while (true);
    }
    ORT_RETURN_IF_ERROR(Expect(")"));
    ORT_RETURN_IF_ERROR(Expect("=>"));
    const bool braced = token_ == "{";
    if (braced) ORT_RETURN_IF_ERROR(Next());
    bool returned = false;
    while (kind_ != Kind::End && token_ != "}") {
      if (kind_ == Kind::Symbol && token_ == ";") {
        ORT_RETURN_IF_ERROR(Next());
        continue;
      }
      InlinedVector<std::string> names;
      InlinedHashMap<std::string, InlinedVector<TypeConstraint, 1>> annotations;
      if (kind_ == Kind::Identifier && token_ == "return") {
        returned = true;
        ORT_RETURN_IF_ERROR(Next());
      }
      bool returned_values = false;
      {
        const size_t start = token_start_;
        do {
          std::string name;
          ORT_RETURN_IF_ERROR(Identifier(name, names.empty()));
          ORT_RETURN_IF(std::find(names.begin(), names.end(), name) != names.end(), Error("duplicate assignment name"));
          names.push_back(std::move(name));
          if (token_ == ":") {
            ORT_RETURN_IF(returned, Error("annotate the definition, not the return"));
            InlinedVector<TypeConstraint, 1> alternatives;
            ORT_RETURN_IF_ERROR(Annotation(names.back(), alternatives));
            ORT_RETURN_IF(std::any_of(alternatives.begin(), alternatives.end(), [](const auto& type) { return type.absent; }),
                          Error("operator outputs cannot be optional"));
            annotations.emplace(names.back(), std::move(alternatives));
          }
          if (token_ != ",") break;
          ORT_RETURN_IF_ERROR(Next());
        } while (true);
        if (names.size() == 1 && (token_ == "(" || token_ == "::")) {
          ORT_RETURN_IF(!returned && !replacement_attributes_, Error("expected assignment or return"));
          position_ = start;
          ORT_RETURN_IF_ERROR(Next());
          names.clear();
          returned = true;
        } else {
          ORT_RETURN_IF(std::any_of(names.begin(), names.end(),
                                    [](const auto& name) { return name.find('.') != std::string::npos; }),
                        Error("expected value identifier"));
          if (returned) {
            ORT_RETURN_IF(token_ == "=", Error("assign outputs before returning them"));
            ORT_RETURN_IF(pattern_.Nodes().empty(), Error("return must identify operator outputs"));
            for (size_t i = 0; i < names.size(); ++i) {
              auto found = symbols_.find(names[i]);
              ORT_RETURN_IF(found == symbols_.end(), Error("undefined return value"));
              ORT_RETURN_IF(pattern_.IsInput(found->second), Error("cannot return a leaf input"));
              pattern_.outputs_.push_back(found->second);
            }
            returned_values = true;
          } else {
            for (const auto& name : names)
              ORT_RETURN_IF(symbols_.count(name) || pattern_.attribute_names_.count(name),
                            Error("duplicate assignment or capture name"));
            ORT_RETURN_IF_ERROR(Expect("="));
          }
        }
      }
      if (!returned_values) {
        ORT_RETURN_IF_ERROR(Call(names));
        for (auto& [name, constraint] : annotations)
          pattern_.type_constraints_.emplace(symbols_.at(name).handle, std::move(constraint));
      }
      if (returned) {
        if (!returned_values) pattern_.outputs_ = pattern_.Nodes().back().outputs;
        if (kind_ == Kind::Symbol && token_ == ";") ORT_RETURN_IF_ERROR(Next());
        if (braced) ORT_RETURN_IF_ERROR(Expect("}"));
        ORT_RETURN_IF(kind_ != Kind::End, Error("unexpected text after return"));
        break;
      }
    }
    ORT_RETURN_IF(!returned, Error("missing return"));
    InlinedHashSet<std::string> dimension_names;
    InlinedHashSet<std::string> element_names;
    size_t combinations = 1;
    for (const auto& [handle, alternatives] : pattern_.type_constraints_) {
      ORT_RETURN_IF(combinations > 64 / alternatives.size(), Error("type alternatives exceed 64 combinations"));
      combinations *= alternatives.size();
      for (const auto& constraint : alternatives) {
        if (pattern_.IsAttribute(handle)) {
          ORT_RETURN_IF(alternatives.size() != 1 || !constraint.attribute_type || pattern_.IsOptional(handle),
                        Error("invalid attribute annotation or optional attribute"));
        } else {
          ORT_RETURN_IF(!constraint.tensor_type, Error("annotation requires an attribute variable"));
        }
        if (!constraint.element_variable.empty()) element_names.insert(constraint.element_variable);
        for (const auto& dim : constraint.dimensions)
          if (const auto* name = std::get_if<std::string>(&dim)) dimension_names.insert(*name);
      }
    }
    ORT_RETURN_IF(dimension_names.size() + element_names.size() > 256, Error("too many annotation variables"));
    for (const auto& name : dimension_names)
      ORT_RETURN_IF(symbols_.count(name) || pattern_.attribute_names_.count(name) || element_names.count(name),
                    Error("dimension name conflicts with another variable"));
    for (const auto& name : element_names)
      ORT_RETURN_IF(symbols_.count(name) || pattern_.attribute_names_.count(name),
                    Error("element type name conflicts with another variable"));
    for (const auto& node : pattern_.Nodes())
      for (Value input : node.inputs)
        ORT_RETURN_IF(pattern_.IsAttribute(input), Error("attribute variable used as a tensor input"));
    InlinedHashSet<int> reachable;
    for (Value output : pattern_.Nodes().back().outputs) reachable.insert(output.handle);
    for (int handle : attribute_inputs_) reachable.insert(handle);
    for (auto it = pattern_.Nodes().rbegin(); it != pattern_.Nodes().rend(); ++it) {
      ORT_RETURN_IF(std::none_of(it->outputs.begin(), it->outputs.end(),
                                 [&](Value output) { return reachable.count(output.handle); }),
                    Error("disconnected pattern node"));
      for (Value output : it->outputs) reachable.insert(output.handle);
      for (Value input : it->inputs)
        if (input.handle >= 0) reachable.insert(input.handle);
    }
    ORT_RETURN_IF(reachable.size() != pattern_.ValueCount(), Error("unused input"));
    destination = std::move(pattern_);
    return Status::OK();
  }

 private:
  Status Error(const char* message) const {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Pattern parse error at byte ", token_start_, ": ", message);
  }

  static bool Alpha(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; }
  static bool Digit(char c) { return c >= '0' && c <= '9'; }

  Status Next() {
    while (position_ < text_.size()) {
      if (std::isspace(static_cast<unsigned char>(text_[position_]))) {
        ++position_;
      } else if (text_[position_] == '#') {
        while (position_ < text_.size() && text_[position_] != '\n' && text_[position_] != '\r') ++position_;
      } else {
        break;
      }
    }
    token_start_ = position_;
    token_.clear();
    kind_ = Kind::End;
    if (position_ == text_.size()) return Status::OK();
    kind_ = Kind::Symbol;
    char c = text_[position_++];
    if (c == '.' && position_ + 1 < text_.size() && text_.substr(token_start_, 3) == "...") {
      position_ += 2;
    } else if (Alpha(c)) {
      kind_ = Kind::Identifier;
      while (position_ < text_.size() &&
             (Alpha(text_[position_]) || Digit(text_[position_]) || text_[position_] == '.'))
        ++position_;
    } else if (Digit(c) || c == '-' || c == '+' || c == '.') {
      kind_ = Kind::Number;
      while (position_ < text_.size()) {
        char d = text_[position_];
        if (!Digit(d) && d != '.' && d != 'e' && d != 'E' && d != '+' && d != '-') break;
        ++position_;
      }
    } else if (c == '\'' || c == '"') {
      kind_ = Kind::String;
      bool closed = false;
      while (position_ < text_.size()) {
        char d = text_[position_++];
        if (d == c) {
          closed = true;
          break;
        }
        if (d == '\\') {
          ORT_RETURN_IF(position_ == text_.size(), Error("unterminated string escape"));
          d = text_[position_++];
          ORT_RETURN_IF(d != '\\' && d != '\'' && d != '"', Error("unsupported string escape"));
        }
        token_.push_back(d);
      }
      ORT_RETURN_IF(!closed, Error("unterminated string"));
      return Status::OK();
    } else if ((c == '=' && position_ < text_.size() && text_[position_] == '>') ||
               (c == ':' && position_ < text_.size() && text_[position_] == ':')) {
      ++position_;
    } else {
      ORT_RETURN_IF(std::string_view("(),=;{}[]:<>|").find(c) == std::string_view::npos, Error("unexpected character"));
    }
    token_ = text_.substr(token_start_, position_ - token_start_);
    return Status::OK();
  }

  Status Expect(std::string_view token) {
    ORT_RETURN_IF(kind_ != Kind::Symbol || token_ != token, Error("unexpected token"));
    return Next();
  }
  Status Identifier(std::string& name, bool allow_domain = false) {
    ORT_RETURN_IF(
        kind_ != Kind::Identifier || token_ == "return" || token_ == "None" ||
            (!allow_domain && token_.find('.') != std::string::npos),
        Error("expected value identifier"));
    name = token_;
    return Next();
  }

  Status Annotation(const std::string& name, InlinedVector<TypeConstraint, 1>& alternatives) {
    ORT_RETURN_IF(replacement_attributes_, Error("type annotations are supported in patterns only"));
    ORT_RETURN_IF_ERROR(Expect(":"));
    do {
      TypeConstraint constraint;
      constraint.value_name = name;
      if (token_ == "None") {
        constraint.absent = true;
        ORT_RETURN_IF_ERROR(Next());
      } else {
        ORT_RETURN_IF_ERROR(AnnotationType(constraint));
      }
      auto equivalent = [&](const TypeConstraint& other) {
        return other.absent == constraint.absent && other.element_type == constraint.element_type &&
               other.element_variable == constraint.element_variable && other.dimensions == constraint.dimensions &&
               other.any_rank == constraint.any_rank && (constraint.any_rank || other.prefix == constraint.prefix) &&
               other.attribute_type == constraint.attribute_type && other.tensor_type == constraint.tensor_type;
      };
      ORT_RETURN_IF(std::any_of(alternatives.begin(), alternatives.end(), equivalent), Error("duplicate type alternative"));
      alternatives.push_back(std::move(constraint));
      ORT_RETURN_IF(alternatives.size() > 64, Error("too many type alternatives"));
      if (token_ != "|") break;
      ORT_RETURN_IF_ERROR(Next());
    } while (true);
    ORT_RETURN_IF(alternatives.size() == 1 && alternatives.front().absent, Error("None requires a tensor alternative"));
    return Status::OK();
  }

  Status AnnotationType(TypeConstraint& constraint) {
    std::string type;
    ORT_RETURN_IF_ERROR(Identifier(type));
    if (type == "int" || type == "ints" || type == "floats" || type == "strings") {
      constraint.tensor_type = false;
      if (type == "int") constraint.attribute_type = ONNX_NAMESPACE::AttributeProto_AttributeType_INT;
      if (type == "ints") constraint.attribute_type = ONNX_NAMESPACE::AttributeProto_AttributeType_INTS;
      if (type == "floats") constraint.attribute_type = ONNX_NAMESPACE::AttributeProto_AttributeType_FLOATS;
      if (type == "strings") constraint.attribute_type = ONNX_NAMESPACE::AttributeProto_AttributeType_STRINGS;
    } else if (type == "tensor") {
      constraint.attribute_type = ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR;
    } else {
      std::string upper = type;
      std::transform(upper.begin(), upper.end(), upper.begin(),
                     [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
      ONNX_NAMESPACE::TensorProto_DataType element;
      ORT_RETURN_IF(!ONNX_NAMESPACE::TensorProto_DataType_Parse(upper, &element) ||
                        element == ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED,
                    Error("unknown tensor element type"));
      constraint.element_type = element;
      if (type == "float") constraint.attribute_type = ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT;
      if (type == "string") constraint.attribute_type = ONNX_NAMESPACE::AttributeProto_AttributeType_STRING;
    }
    if (token_ == "<") {
      ORT_RETURN_IF(type != "tensor", Error("only tensor permits an element type variable"));
      ORT_RETURN_IF_ERROR(Next());
      ORT_RETURN_IF_ERROR(Identifier(constraint.element_variable));
      ORT_RETURN_IF_ERROR(Expect(">"));
      constraint.attribute_type = 0;
    }
    if (token_ == "[") {
      ORT_RETURN_IF(!constraint.tensor_type, Error("attribute list types do not accept a shape"));
      constraint.attribute_type = 0;
      ORT_RETURN_IF_ERROR(Next());
      if (token_ == "]") {
        constraint.any_rank = true;
      } else {
        while (true) {
          if (token_ == "...") {
            constraint.any_rank = constraint.dimensions.empty();
            constraint.prefix = true;
            ORT_RETURN_IF_ERROR(Next());
            break;
          }
          ORT_RETURN_IF(constraint.dimensions.size() >= 64, Error("annotations are limited to 64 dimensions"));
          if (kind_ == Kind::Identifier) {
            std::string dimension;
            ORT_RETURN_IF_ERROR(Identifier(dimension));
            if (dimension == "_")
              constraint.dimensions.emplace_back(std::monostate{});
            else
              constraint.dimensions.emplace_back(std::move(dimension));
          } else {
            std::variant<float, int64_t> number;
            ORT_RETURN_IF_ERROR(Number(number));
            const auto* fixed = std::get_if<int64_t>(&number);
            ORT_RETURN_IF(!fixed || *fixed < 0, Error("dimensions must be nonnegative integers or names"));
            constraint.dimensions.emplace_back(*fixed);
          }
          if (token_ != ",") break;
          ORT_RETURN_IF_ERROR(Next());
          ORT_RETURN_IF(token_ == "]", Error("missing dimension after comma"));
        }
      }
      ORT_RETURN_IF_ERROR(Expect("]"));
    }
    return Status::OK();
  }

  Status Number(std::variant<float, int64_t>& result) {
    ORT_RETURN_IF(kind_ != Kind::Number, Error("expected numeric literal"));
    std::string_view number = token_;
    if (!number.empty() && number.front() == '+') number.remove_prefix(1);
    if (number.find_first_of(".eE") != std::string_view::npos) {
      float value;
      auto parsed = std::from_chars(number.data(), number.data() + number.size(), value);
      ORT_RETURN_IF(parsed.ec != std::errc{} || parsed.ptr != number.data() + number.size() || !std::isfinite(value),
                    Error("invalid or nonfinite float literal"));
      result = value;
    } else {
      int64_t value;
      auto parsed = std::from_chars(number.data(), number.data() + number.size(), value);
      ORT_RETURN_IF(parsed.ec != std::errc{} || parsed.ptr != number.data() + number.size(),
                    Error("invalid integer literal"));
      result = value;
    }
    return Next();
  }

  Status List(Literal& result) {
    ORT_RETURN_IF_ERROR(Expect("["));
    InlinedVector<int64_t> integers;
    InlinedVector<float> floats;
    std::optional<bool> floating;
    while (token_ != "]") {
      std::variant<float, int64_t> number;
      ORT_RETURN_IF_ERROR(Number(number));
      const bool is_float = std::holds_alternative<float>(number);
      ORT_RETURN_IF(floating && *floating != is_float, Error("list elements must have the same numeric type"));
      floating = is_float;
      if (is_float)
        floats.push_back(std::get<float>(number));
      else
        integers.push_back(std::get<int64_t>(number));
      if (token_ != ",") break;
      ORT_RETURN_IF_ERROR(Next());
      ORT_RETURN_IF(token_ == "]", Error("expected list element after comma"));
    }
    ORT_RETURN_IF_ERROR(Expect("]"));
    if (floating.value_or(false))
      result = std::move(floats);
    else
      result = std::move(integers);
    return Status::OK();
  }

  Status Call(const InlinedVector<std::string>& names) {
    ORT_RETURN_IF(kind_ != Kind::Identifier || token_ == "return", Error("expected operator name"));
    Operator op{token_, ""};
    ORT_RETURN_IF_ERROR(Next());
    if (token_ == "::") {
      op.domain = std::move(op.name);
      ORT_RETURN_IF_ERROR(Next());
      ORT_RETURN_IF(kind_ != Kind::Identifier, Error("expected qualified operator name"));
      op.name = token_;
      ORT_RETURN_IF_ERROR(Next());
    } else {
      auto dot = op.name.rfind('.');
      if (dot != std::string::npos) {
        op.domain = op.name.substr(0, dot);
        op.name.erase(0, dot + 1);
        ORT_RETURN_IF(op.name.empty(), Error("expected qualified operator name"));
      }
    }
    ORT_RETURN_IF_ERROR(Expect("("));
    InlinedVector<Value> inputs;
    InlinedVector<AttributeMatcher, 1> attributes;
    NodeAttributes literal_attributes;
    InlinedHashMap<std::string, std::string> bindings;
    InlinedHashSet<std::string> attribute_names;
    bool keyword = false;
    if (token_ != ")") {
      do {
        if (kind_ == Kind::Identifier) {
          std::string name = token_;
          ORT_RETURN_IF_ERROR(Next());
          if (token_ == "=") {
            keyword = true;
            ORT_RETURN_IF(name == "None" || name == "return", Error("reserved attribute name"));
            ORT_RETURN_IF(!attribute_names.insert(name).second, Error("duplicate attribute"));
            ORT_RETURN_IF_ERROR(Next());
            if (kind_ == Kind::Identifier) {
              std::string capture;
              ORT_RETURN_IF_ERROR(Identifier(capture));
              if (replacement_bindings_) {
                auto found = symbols_.find(capture);
                ORT_RETURN_IF(found == symbols_.end() || !pattern_.IsInput(found->second),
                              Error("attribute binding must reference a declared replacement input"));
                bindings.emplace(name, capture);
                attribute_inputs_.insert(found->second.handle);
              } else {
                auto declared = symbols_.find(capture);
                if (declared != symbols_.end()) {
                  ORT_RETURN_IF(!pattern_.IsInput(declared->second),
                                Error("attribute capture conflicts with an operator output"));
                  pattern_.attribute_inputs_.insert(declared->second.handle);
                  attribute_inputs_.insert(declared->second.handle);
                }
                pattern_.attribute_names_.insert(capture);
                attributes.push_back({name, [](const auto&) { return true; }, capture});
              }
            } else if (kind_ == Kind::String) {
              attributes.push_back(AttributeMatcher::Equal(name, token_));
              auto& attr = literal_attributes[name];
              attr.set_name(name);
              attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRING);
              attr.set_s(token_);
              ORT_RETURN_IF_ERROR(Next());
            } else if (token_ == "[") {
              Literal list;
              ORT_RETURN_IF_ERROR(List(list));
              ONNX_NAMESPACE::AttributeProto attr;
              attr.set_name(name);
              if (const auto* values = std::get_if<InlinedVector<float>>(&list)) {
                attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOATS);
                for (auto value : *values) attr.add_floats(value);
              } else {
                attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INTS);
                for (auto value : std::get<InlinedVector<int64_t>>(list)) attr.add_ints(value);
              }
              attributes.push_back({name, [attr](const auto& actual) {
                                      if (actual.type() != attr.type()) return false;
                                      if (attr.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_FLOATS) {
                                        return actual.floats_size() == attr.floats_size() &&
                                               std::equal(attr.floats().begin(), attr.floats().end(),
                                                          actual.floats().begin());
                                      }
                                      return actual.ints_size() == attr.ints_size() &&
                                             std::equal(attr.ints().begin(), attr.ints().end(), actual.ints().begin());
                                    }});
              literal_attributes.emplace(name, std::move(attr));
            } else {
              std::variant<float, int64_t> number;
              ORT_RETURN_IF_ERROR(Number(number));
              attributes.push_back(
                  std::visit([&](auto value) { return AttributeMatcher::Equal(name, value); }, number));
              auto& attr = literal_attributes[name];
              attr.set_name(name);
              if (const auto* floating = std::get_if<float>(&number)) {
                attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
                attr.set_f(*floating);
              } else {
                attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
                attr.set_i(std::get<int64_t>(number));
              }
            }
          } else {
            ORT_RETURN_IF(keyword, Error("positional input after attribute"));
            if (name == "None") {
              inputs.push_back(Value{-1});
            } else {
              auto found = symbols_.find(name);
              ORT_RETURN_IF(found == symbols_.end(), Error("undefined input value"));
              inputs.push_back(found->second);
            }
          }
        } else {
          ORT_RETURN_IF(keyword, Error("positional input after attribute"));
          ORT_RETURN_IF(pattern_.ValueCount() >= 256, Error("too many values"));
          if (token_ == "[") {
            Literal list;
            ORT_RETURN_IF_ERROR(List(list));
            inputs.push_back(pattern_.Constant(std::move(list)));
          } else {
            std::variant<float, int64_t> number;
            ORT_RETURN_IF_ERROR(Number(number));
            inputs.push_back(std::visit([&](auto value) { return pattern_.Constant(value); }, number));
          }
        }
        ORT_RETURN_IF(inputs.size() > 64, Error("too many operator inputs"));
        if (token_ != ",") break;
        ORT_RETURN_IF_ERROR(Next());
      } while (true);
    }
    ORT_RETURN_IF_ERROR(Expect(")"));
    size_t outputs = names.empty() ? 1 : names.size();
    ORT_RETURN_IF(outputs > 64 || pattern_.Nodes().size() >= 64 || pattern_.ValueCount() + outputs > 256,
                  Error("pattern limits exceeded"));
    auto values = pattern_.MultiOp(std::move(op), gsl::span<const Value>(inputs), outputs);
    pattern_.Attributes(std::move(attributes));
    if (replacement_attributes_) replacement_attributes_->push_back(std::move(literal_attributes));
    if (replacement_bindings_) replacement_bindings_->push_back(std::move(bindings));
    for (size_t i = 0; i < names.size(); ++i) {
      ORT_RETURN_IF(pattern_.attribute_names_.count(names[i]), Error("output name conflicts with an attribute capture"));
      pattern_.Name(values[i], names[i]);
      symbols_.emplace(names[i], pattern_.NamedValue(names[i]));
    }
    return Status::OK();
  }

  enum class Kind {
    End,
    Symbol,
    Identifier,
    Number,
    String
  };
  std::string_view text_;
  size_t position_ = 0;
  size_t token_start_ = 0;
  Kind kind_ = Kind::Symbol;
  std::string token_;
  Pattern pattern_;
  InlinedHashMap<std::string, Value> symbols_;
  InlinedVector<NodeAttributes, 1>* replacement_attributes_ = nullptr;
  InlinedVector<InlinedHashMap<std::string, std::string>, 1>* replacement_bindings_ = nullptr;
  InlinedHashSet<int> attribute_inputs_;
};
}  // namespace detail

Rule::Rule(std::string name, Pattern pattern, ReplacementFunction replacement, CheckAndBind check_and_bind,
           InlinedVector<OpsetRequirement, 1> opsets)
    : name(std::move(name)),
      pattern(std::move(pattern)),
      replacement(std::move(replacement)),
      check_and_bind(std::move(check_and_bind)),
      opsets(std::move(opsets)) {}

Rule::Rule(std::string name, std::string_view pattern, ReplacementFunction replacement, CheckAndBind check_and_bind,
           InlinedVector<OpsetRequirement, 1> opsets)
    : Rule(std::move(name), Pattern(pattern), std::move(replacement), std::move(check_and_bind), std::move(opsets)) {}

Rule::Rule(std::string name, Pattern pattern, std::string_view replacement, CheckAndBind check_and_bind,
           InlinedVector<OpsetRequirement, 1> opsets)
    : Rule(std::move(name), std::move(pattern), Replacement(replacement), std::move(check_and_bind), std::move(opsets)) {}

Rule::Rule(std::string name, std::string_view pattern, std::string_view replacement, CheckAndBind check_and_bind,
           InlinedVector<OpsetRequirement, 1> opsets)
    : Rule(std::move(name), Pattern(pattern), Replacement(replacement), std::move(check_and_bind), std::move(opsets)) {}

Pattern::Pattern(std::string_view text) { ORT_THROW_IF_ERROR(detail::PatternParser(text).Parse(*this)); }

Replacement::Replacement(std::string_view text) {
  Pattern parsed;
  InlinedVector<NodeAttributes, 1> parsed_attributes;
  InlinedVector<InlinedHashMap<std::string, std::string>, 1> parsed_bindings;
  ORT_THROW_IF_ERROR(detail::PatternParser(text, &parsed_attributes, &parsed_bindings).Parse(parsed));
  InlinedHashMap<int, Output> outputs;
  InlinedVector<NodeRecipe, 1> nodes;
  InlinedHashSet<std::string> attribute_names;
  InlinedHashSet<std::string> tensor_names;
  for (size_t i = 0; i < parsed.Nodes().size(); ++i) {
    const auto& node = parsed.Nodes()[i];
    NodeRecipe recipe{node.op.name, node.op.domain, {}, std::move(parsed_attributes[i]), node.outputs.size()};
    recipe.attribute_bindings = std::move(parsed_bindings[i]);
    for (const auto& [name, binding] : recipe.attribute_bindings) attribute_names.insert(binding);
    for (const Value& input : node.inputs) {
      if (input.handle < 0) {
        recipe.inputs.push_back(std::monostate{});
        continue;
      }
      auto literal = parsed.Constants().find(input.handle);
      if (literal != parsed.Constants().end()) {
        recipe.inputs.push_back(Scalar{literal->second});
        continue;
      }
      auto found = outputs.find(input.handle);
      if (found == outputs.end()) {
        tensor_names.insert(input.name);
        recipe.inputs.push_back(input.name);
      } else
        recipe.inputs.push_back(found->second);
    }
    for (const auto& name : attribute_names)
      ORT_ENFORCE(!tensor_names.count(name), "Replacement input cannot be both a tensor and an attribute: ", name);
    for (size_t slot = 0; slot < node.outputs.size(); ++slot)
      outputs.emplace(node.outputs[slot].handle, Output{nodes.size(), slot});
    nodes.push_back(std::move(recipe));
  }
  for (Value value : parsed.Outputs()) outputs_.push_back(outputs.at(value.handle));
  plan_ = std::make_shared<const InlinedVector<NodeRecipe, 1>>(std::move(nodes));
}

const NodeArg* Match::valueOf(std::string_view name) const {
  ORT_ENFORCE(pattern, "Named lookup requires a matched pattern.");
  auto value = std::find_if(pattern->values_.begin(), pattern->values_.end(),
                            [&](const Value& candidate) { return candidate.name == name; });
  ORT_ENFORCE(!name.empty() && value != pattern->values_.end(), "Unknown pattern value: ", name);
  ORT_ENFORCE(value->handle >= 0 && static_cast<size_t>(value->handle) < values.size() &&
                  (values[value->handle] || pattern->IsOptional(*value)),
              "Pattern value is not bound: ", name);
  return values[value->handle];
}

void Match::update(std::string_view name, ONNX_NAMESPACE::AttributeProto value) {
  ORT_ENFORCE(!name.empty() && value.type() != ONNX_NAMESPACE::AttributeProto_AttributeType_UNDEFINED &&
                  !value.has_ref_attr_name(),
              "Extra bindings require a name and a concrete attribute value.");
  ORT_ENFORCE(!dimensions_.count(std::string(name)) && !element_types_.count(std::string(name)),
              "Extra attribute binding cannot overwrite a dimension or element type: ", name);
  if (pattern) {
    ORT_ENFORCE(std::none_of(pattern->values_.begin(), pattern->values_.end(),
                             [&](const Value& binding) { return binding.name == name; }),
                "Extra attribute binding cannot overwrite a pattern value: ", name);
    ORT_ENFORCE(!pattern->AttributeNames().count(std::string(name)),
                "Extra attribute binding cannot overwrite a captured attribute: ", name);
  }
  value.set_name(std::string(name));
  extra_attributes_.insert_or_assign(std::string(name), std::move(value));
}

const ONNX_NAMESPACE::AttributeProto& Match::attributeOf(std::string_view name) const {
  auto captured = captured_attributes_.find(std::string(name));
  if (captured != captured_attributes_.end()) return captured->second;
  auto found = extra_attributes_.find(std::string(name));
  ORT_ENFORCE(!name.empty() && found != extra_attributes_.end(), "Missing replacement attribute binding: ", name);
  return found->second;
}

bool Match::BindAttribute(std::string_view name, const ONNX_NAMESPACE::AttributeProto& value) {
  if (value.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_UNDEFINED || value.has_ref_attr_name()) return false;
  auto normalized = value;
  normalized.set_name(std::string(name));
  normalized.clear_doc_string();
  auto [it, inserted] = captured_attributes_.emplace(std::string(name), normalized);
  return inserted || it->second.SerializeAsString() == normalized.SerializeAsString();
}

const Dimension& Match::dimensionOf(std::string_view name) const {
  auto found = dimensions_.find(std::string(name));
  ORT_ENFORCE(found != dimensions_.end(), "Unknown pattern dimension: ", name);
  return found->second;
}

std::optional<int32_t> Match::elementTypeOf(std::string_view name) const {
  auto found = element_types_.find(std::string(name));
  ORT_ENFORCE(found != element_types_.end(), "Unknown pattern element type: ", name);
  if (!found->second) return std::nullopt;
  return found->second;
}

bool Match::BindDimension(std::string_view name, Dimension value) {
  auto [it, inserted] = dimensions_.emplace(std::string(name), value);
  if (inserted || std::holds_alternative<std::monostate>(value)) return true;
  if (std::holds_alternative<std::monostate>(it->second)) {
    it->second = std::move(value);
    return true;
  }
  return it->second == value;
}

bool Match::BindElementType(std::string_view name, int32_t value) {
  auto [it, inserted] = element_types_.emplace(std::string(name), value);
  if (inserted || !value) return true;
  if (!it->second) {
    it->second = value;
    return true;
  }
  return it->second == value;
}

PatternRewriter::PatternRewriter(Rules rules, Options options)
    : GraphTransformer("PatternRewriter"), rules_(std::move(rules)), options_(options) {
  ORT_ENFORCE(options_.max_iterations > 0, "PatternRewriter max_iterations must be positive.");
  InlinedHashMap<std::string, size_t> signatures;
  for (size_t i = 0; i < rules_.size(); ++i) {
    Rule& rule = rules_[i];
    ORT_ENFORCE(!rule.name.empty() && !rule.pattern.Nodes().empty() && rule.replacement,
                "Rules need a name, at least one node and a recipe.");
    InlinedHashSet<std::string> domains;
    for (auto& requirement : rule.opsets) {
      if (requirement.domain == kOnnxDomainAlias) requirement.domain = kOnnxDomain;
      ORT_ENFORCE(requirement.min_version > 0 && requirement.max_version >= requirement.min_version,
                  "Rule opset requirements need a positive, nonempty inclusive range.");
      ORT_ENFORCE(domains.insert(requirement.domain).second, "Duplicate rule opset domain: ", requirement.domain);
    }
    for (const auto& node : rule.pattern.Nodes()) {
      ORT_ENFORCE(domains.count(node.op.domain), "Missing rule opset requirement for domain: ", node.op.domain);
    }
    if (auto* recipe = rule.replacement.target<Replacement>(); recipe && recipe->plan_) {
      auto nodes = *recipe->plan_;
      for (auto& node : nodes) {
        if (node.domain == kOnnxDomainAlias) node.domain = kOnnxDomain;
        ORT_ENFORCE(domains.count(node.domain), "Missing rule opset requirement for replacement domain: ", node.domain);
        for (const auto& [name, binding] : node.attribute_bindings) {
          ORT_ENFORCE(std::none_of(rule.pattern.values_.begin(), rule.pattern.values_.end(),
                                   [&](const Value& value) {
                                     return value.name == binding && !rule.pattern.IsAttribute(value);
                                   }),
                      "Replacement attribute binding conflicts with a pattern value: ", binding);
        }
        for (auto& input : node.inputs) {
          if (const auto* name = std::get_if<std::string>(&input)) {
            auto value = rule.pattern.NamedValue(*name);
            ORT_ENFORCE(rule.pattern.IsInput(value) && !rule.pattern.IsAttribute(value),
                        "Replacement inputs must be pattern tensor input variables: ", *name);
            input = std::move(value);
          }
        }
      }
      recipe->plan_ = std::make_shared<const InlinedVector<Replacement::NodeRecipe, 1>>(std::move(nodes));
    }
    // Every node must reach the last node. Disconnected definitions are not patterns.
    const auto& root_outputs = rule.pattern.Nodes().back().outputs;
    InlinedHashSet<int> reachable;
    for (Value output : root_outputs) reachable.insert(output.handle);
    for (auto it = rule.pattern.Nodes().rbegin(); it != rule.pattern.Nodes().rend(); ++it) {
      ORT_ENFORCE(
          std::any_of(it->outputs.begin(), it->outputs.end(), [&](Value output) { return reachable.count(output); }),
          "Disconnected pattern node.");
      for (Value output : it->outputs) reachable.insert(output.handle);
      for (Value input : it->inputs)
        if (input.handle >= 0) reachable.insert(input.handle);
    }
    for (int handle : rule.pattern.attribute_inputs_) reachable.insert(handle);
    ORT_ENFORCE(reachable.size() == rule.pattern.ValueCount(), "Unused pattern input.");
    invalidation_depth_ = std::max(invalidation_depth_, rule.pattern.Nodes().size());
    if (!options_.indexed) continue;
    std::string signature = Signature(rule.pattern);
    // Opaque predicates cannot be compared safely; keep their plans distinct.
    if (std::any_of(rule.pattern.Nodes().begin(), rule.pattern.Nodes().end(),
                    [](const auto& node) { return !node.attributes.empty(); }) ||
        !rule.pattern.TypeConstraints().empty())
      signature += "#" + std::to_string(i);
    auto [it, inserted] = signatures.emplace(signature, groups_.size());
    if (inserted) {
      const auto& root = rule.pattern.Nodes().back().op;
      roots_[Key(root.name, root.domain)].push_back(groups_.size());
      groups_.push_back({rule.pattern, {}});
    }
    groups_[it->second].rules.push_back(i);
  }
}

Status PatternRewriter::Run(Graph& graph, Statistics& statistics) const {
  statistics = {};
  ORT_RETURN_IF(graph.GraphResolveNeeded(), "PatternRewriter requires an already resolved graph.");
  auto report = [&](size_t id, NodeIndex root, FailureStage stage, std::string_view reason) {
    if (options_.on_failure)
      options_.on_failure({rules_[id].name, root, stage, std::string(reason)});
  };
  // Import applicability is evaluated once per rule and graph, never per matched node.
  InlinedVector<char> applicable;
  InlinedVector<size_t> applicable_rules;
  for (size_t id = 0; id < rules_.size(); ++id) {
    bool supported = std::all_of(rules_[id].opsets.begin(), rules_[id].opsets.end(), [&](const auto& requirement) {
      auto imported = graph.DomainToVersionMap().find(requirement.domain);
      return imported != graph.DomainToVersionMap().end() && imported->second >= requirement.min_version &&
             imported->second <= requirement.max_version;
    });
    applicable.push_back(supported);
    if (supported)
      applicable_rules.push_back(id);
    else
      report(id, std::numeric_limits<NodeIndex>::max(), FailureStage::Opset, "Missing or unsupported domain import.");
  }
  if (applicable_rules.empty()) return Status::OK();
  InlinedVector<InlinedVector<size_t>, 1> applicable_groups;
  for (const auto& group : groups_) {
    InlinedVector<size_t> ids;
    for (size_t id : group.rules) {
      if (applicable[id]) ids.push_back(id);
    }
    applicable_groups.push_back(std::move(ids));
  }
  Producers producers;
  InlinedVector<NodeIndex> worklist;
  InlinedHashSet<NodeIndex> queued;
  GraphViewer viewer(graph);
  for (NodeIndex index : viewer.GetNodesInTopologicalOrder()) {
    const Node& node = *graph.GetNode(index);
    worklist.push_back(index);
    queued.insert(index);
    for (const NodeArg* output : node.OutputDefs()) {
      if (output->Exists()) producers[output] = index;
    }
  }
  for (size_t iteration = 0; iteration < options_.max_iterations && !worklist.empty(); ++iteration) {
    ++statistics.iterations;
    InlinedVector<NodeIndex> next_worklist;
    queued.clear();
    const size_t rewrites_before = statistics.rewrites;
    // Each batch is immutable: revisits and new nodes wait until the next iteration.
    for (size_t cursor = 0; cursor < worklist.size(); ++cursor) {
      const Node* root = graph.GetNode(worklist[cursor]);
      if (!root) continue;
      ++statistics.visited;
      size_t winner = std::numeric_limits<size_t>::max();
      Match winning_match;
      const Pattern* winning_pattern = nullptr;
      auto consider = [&](const Pattern& pattern, gsl::span<const size_t> rule_ids) {
        statistics.candidates += rule_ids.size();
        ++statistics.structural_checks;
        Match match;
        FailureStage stage = FailureStage::Structure;
        std::string_view reason;
        if (!MatchPattern(pattern, graph, *root, producers, match, stage, reason)) {
          for (size_t id : rule_ids) report(id, root->Index(), stage, reason);
          return;
        }
        if (!CanRewritePartition(match, options_)) {
          for (size_t id : rule_ids)
            report(id, root->Index(), FailureStage::ExecutionProvider, "Mixed or unsupported execution providers.");
          return;
        }
        for (size_t id : rule_ids) {
          if (!options_.match_only && id >= winner) break;
          Match rule_match = match;
          rule_match.pattern = &rules_[id].pattern;
          if (rules_[id].check_and_bind) {
            ++statistics.check_and_bind_calls;
            if (!rules_[id].check_and_bind(graph, rule_match)) {
              report(id, root->Index(), FailureStage::CheckAndBind,
                     rule_match.RejectionReason().empty() ? std::string_view("Check-and-bind rejected match.")
                                                          : rule_match.RejectionReason());
              continue;
            }
          }
          if (options_.match_only) ++statistics.matches;
          if (id < winner) {
            winner = id;
            winning_match = std::move(rule_match);
            winning_pattern = winning_match.pattern;
          }
          if (!options_.match_only) break;
        }
      };
      if (options_.indexed) {
        auto it = roots_.find(Key(root->OpType(), root->Domain()));
        if (it != roots_.end()) {
          for (size_t id : it->second) {
            const auto& group = groups_[id];
            const auto& ids = applicable_groups[id];
            if (ids.empty() || (!options_.match_only && ids.front() >= winner)) continue;
            consider(group.pattern, ids);
          }
        }
      } else {
        for (size_t id : applicable_rules) {
          consider(rules_[id].pattern, gsl::span<const size_t>(&id, 1));
          if (!options_.match_only && winner != std::numeric_limits<size_t>::max()) break;
        }
      }
      if (options_.match_only || !winning_pattern) continue;
      ++statistics.matches;
      InlinedVector<NodeIndex> affected;
      for (int value = 0; value < static_cast<int>(winning_pattern->ValueCount()); ++value) {
        if (!winning_pattern->IsInput(value)) continue;
        auto it = producers.find(winning_match.values[value]);
        if (it != producers.end()) affected.push_back(it->second);
      }
      InlinedVector<NodeIndex> added;
      auto recipe = rules_[winner].replacement(graph, winning_match);
      auto declared = [&](const std::string& node_domain) {
        const auto& domain = node_domain == kOnnxDomainAlias ? kOnnxDomain : node_domain;
        return std::any_of(rules_[winner].opsets.begin(), rules_[winner].opsets.end(),
                           [&](const auto& requirement) { return requirement.domain == domain; });
      };
      auto replace = [&]() -> Status {
        ORT_RETURN_IF(!recipe.Plan(), "Replacement has no parsed plan.");
        for (const auto& node : *recipe.Plan())
          ORT_RETURN_IF(!declared(node.domain), "Replacement domain has no rule opset requirement: ", node.domain);
        return Replace(graph, *winning_pattern, winning_match, recipe, producers, added);
      };
      auto status = replace();
      if (!status.IsOK()) {
        report(winner, root->Index(), FailureStage::Replacement, status.ErrorMessage());
        return status;
      }
      ++statistics.rewrites;
      affected.insert(affected.end(), added.begin(), added.end());
      // A changed producer or use count can affect a deeper pattern root, not just a direct consumer.
      InlinedHashSet<NodeIndex> seen;
      for (size_t depth = 0; depth <= invalidation_depth_ && !affected.empty(); ++depth) {
        std::sort(affected.begin(), affected.end());
        InlinedVector<NodeIndex> next;
        for (NodeIndex index : affected) {
          if (!seen.insert(index).second) continue;
          const Node* node = graph.GetNode(index);
          if (!node) continue;
          if (queued.insert(index).second) next_worklist.push_back(index);
          for (auto it = node->OutputEdgesBegin(); it != node->OutputEdgesEnd(); ++it) {
            next.push_back(it->GetNode().Index());
          }
        }
        affected = std::move(next);
      }
    }
    if (options_.match_only || statistics.rewrites == rewrites_before) break;
    worklist = std::move(next_worklist);
    if (iteration + 1 == options_.max_iterations && !worklist.empty())
      statistics.iteration_limit_reached = true;
  }
  return Status::OK();
}

Status PatternRewriter::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  for (auto& node : graph.Nodes()) ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));
  Statistics statistics;
  ORT_RETURN_IF_ERROR(Run(graph, statistics));
  modified |= statistics.rewrites != 0;
  return Status::OK();
}

}  // namespace rewrite
}  // namespace onnxruntime
