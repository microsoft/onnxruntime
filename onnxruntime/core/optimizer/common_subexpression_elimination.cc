// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common_subexpression_elimination.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph_utils.h"

#include <memory>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// This optimization pass will collapse expressions that always evaluate to the same value
// into one node. As an example, consider the following abstract function where x1, x2,...
// denote graph inputs, and F1, F2,... - operations.
//
// return F3(F2(F1(x1, x2), x3)) + F4(F2(F1(x1, x2), x3))
//
// Because F1 operations are given the same inputs, they can be merged into one node:
//
// y1 = F1(x1, x2)
// return F3(F2(y1, x3)) + F4(F2(y1, x3))
//
// Now we can see that F2 operations are given the same inputs, so they can be merged too:
//
// y1 = F1(x1, x2)
// y2 = F2(y1, x3)
// return F3(y2) + F4(y2)
//
// This is implemented using value numbering (https://en.wikipedia.org/wiki/Value_numbering):
// first every graph input, constant initializer and graph node output are assigned
// an equivalence class, and then nodes that have the same operation and equivalent inputs
// are collapsed.

namespace onnxruntime {

namespace {
struct DeepPointerEquality {
  template <typename Ptr>
  bool operator()(const Ptr& lhs, const Ptr& rhs) const {
    if (lhs == nullptr || rhs == nullptr) {
      return lhs == rhs;
    }
    return *lhs == *rhs;
  }
};

struct DeepPointerHash {
  template <typename Ptr>
  std::size_t operator()(const Ptr& value) const {
    if (value == nullptr) {
      return 0;
    }
    return std::hash<typename std::decay<decltype(*value)>::type>{}(*value);
  }
};

template <typename T, typename Hasher>
void UpdateHash(const T& x, Hasher hasher, std::size_t& hash) {
  constexpr std::size_t kPrime = 31013;
  hash = hash * kPrime + hasher(x);
}

template <typename T>
void UpdateHash(const T& x, std::size_t& hash) {
  UpdateHash(x, std::hash<T>{}, hash);
}

template <typename Container>
void UpdateHashWithContainer(const Container& c, std::size_t& hash) {
  for (const auto& element : c) {
    UpdateHash(element, hash);
  }
}

using OutputIndex = int;

constexpr OutputIndex kInvalidOutputIndex = -1;

const NodeArg* Normalize(const NodeArg* node_arg) {
  return node_arg == nullptr || !node_arg->Exists() ? nullptr : node_arg;
}

// Represents an equivalence class of expressions (inputs, constant initializers and node outputs)
// that will always evaluate to the same runtime value.
class EquivalenceClass {
 public:
  bool operator==(const EquivalenceClass& other) const;

  friend struct ::std::hash<EquivalenceClass>;
  friend InlinedVector<InlinedVector<const EquivalenceClass*>> Normalize(const Node& node, gsl::span<const EquivalenceClass* const> inputs);

  explicit EquivalenceClass(const NodeArg* non_op_value)
      : attributes_(nullptr),
        output_index_(kInvalidOutputIndex),
        non_op_value_(Normalize(non_op_value)),
        discriminator_(0),
        hash_(CalculateHash()) {
  }

  EquivalenceClass(const Node& node, const gsl::span<const EquivalenceClass* const>& explicit_inputs,
                   OutputIndex output_index, int discriminator)
      : op_type_(node.OpType()),
        domain_(node.Domain()),
        inputs_(Normalize(node, explicit_inputs)),
        attributes_(&node.GetAttributes()),
        output_index_(output_index),
        non_op_value_(nullptr),
        discriminator_(discriminator),
        hash_(CalculateHash()) {
  }

 private:
  std::size_t CalculateHash() const;

  // Operation and domain of the node that produces this value.
  const std::string op_type_;
  const std::string domain_;

  // Explicit inputs to the operation, sequence of inputs for each formal parameter.
  const InlinedVector<InlinedVector<const EquivalenceClass*>> inputs_;

  // Attributes of the operation.
  const NodeAttributes* attributes_;

  // Index of this value in the output list of the operation.
  const OutputIndex output_index_;

  // When the value is not an output of an operation, (i.e., a constant initializer or an input),
  // non_op_value is set to the corresponding NodeArg, and other fields are empty.
  // Currently, different inputs/initializers are always considered different values, although we
  // could merge equal initializers.
  const NodeArg* non_op_value_;

  // When an operation is not supported by the CSE optimization pass, we consider its
  // outputs unique (not equal to other values). For this purpose we assign a unique
  // discriminator for such values.
  const int discriminator_;

  const std::size_t hash_;
};

InlinedVector<InlinedVector<const EquivalenceClass*>> Normalize(const Node& node, gsl::span<const EquivalenceClass* const> inputs) {
  const auto& arg_count = node.InputArgCount();
  auto input_iter = inputs.begin();
  InlinedVector<InlinedVector<const EquivalenceClass*>> result(arg_count.size());

  for (std::size_t arg_index = 0; arg_index < arg_count.size(); ++arg_index) {
    auto& arg = result[arg_index];
    for (int i = 0; i < arg_count[arg_index]; ++i) {
      if (input_iter != inputs.end()) {
        arg.push_back(*input_iter);
        ++input_iter;
      }
    }

    // Remove missing optional inputs from the back
    while (!arg.empty() && arg.back()->output_index_ == kInvalidOutputIndex && arg.back()->non_op_value_ == nullptr) {
      arg.pop_back();
    }
  }

  return result;
}

template <typename Range>
bool AreRangesEqual(const Range& lhs, const Range& rhs) {
  return lhs.size() == rhs.size() &&
         std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

bool AreEqual(const ONNX_NAMESPACE::AttributeProto& lhs, const ONNX_NAMESPACE::AttributeProto& rhs) {
  if (&lhs == &rhs) {
    return true;
  }

  if (lhs.type() != rhs.type() || lhs.name() != rhs.name()) {
    return false;
  }

  switch (lhs.type()) {
    case onnx::AttributeProto_AttributeType_FLOAT:
      return lhs.f() == rhs.f();
    case onnx::AttributeProto_AttributeType_INT:
      return lhs.i() == rhs.i();
    case onnx::AttributeProto_AttributeType_STRING:
      return lhs.s() == rhs.s();
    case onnx::AttributeProto_AttributeType_FLOATS:
      return AreRangesEqual(lhs.floats(), rhs.floats());
    case onnx::AttributeProto_AttributeType_INTS:
      return AreRangesEqual(lhs.ints(), rhs.ints());
    case onnx::AttributeProto_AttributeType_STRINGS:
      return AreRangesEqual(lhs.strings(), rhs.strings());
    case onnx::AttributeProto_AttributeType_TENSOR:
    case onnx::AttributeProto_AttributeType_GRAPH:
    case onnx::AttributeProto_AttributeType_SPARSE_TENSOR:
    case onnx::AttributeProto_AttributeType_TYPE_PROTO:
    case onnx::AttributeProto_AttributeType_TENSORS:
    case onnx::AttributeProto_AttributeType_GRAPHS:
    case onnx::AttributeProto_AttributeType_SPARSE_TENSORS:
    case onnx::AttributeProto_AttributeType_TYPE_PROTOS:
    case onnx::AttributeProto_AttributeType_UNDEFINED:
      return false;  // Don't support these attributes for now; corresponding nodes will be considered distinct.
  }

  return false;
}

std::size_t GetAttributeHash(const ONNX_NAMESPACE::AttributeProto& attr) {
  std::size_t hash = 0;
  UpdateHash(
      static_cast<std::underlying_type<ONNX_NAMESPACE::AttributeProto_AttributeType>::type>(attr.type()),
      hash);
  UpdateHash(attr.name(), hash);
  switch (attr.type()) {
    case onnx::AttributeProto_AttributeType_FLOAT:
      UpdateHash(attr.f(), hash);
      break;
    case onnx::AttributeProto_AttributeType_INT:
      UpdateHash(attr.i(), hash);
      break;
    case onnx::AttributeProto_AttributeType_STRING:
      UpdateHash(attr.s(), hash);
      break;
    case onnx::AttributeProto_AttributeType_FLOATS:
      UpdateHashWithContainer(attr.floats(), hash);
      break;
    case onnx::AttributeProto_AttributeType_INTS:
      UpdateHashWithContainer(attr.ints(), hash);
      break;
    case onnx::AttributeProto_AttributeType_STRINGS:
      UpdateHashWithContainer(attr.strings(), hash);
      break;
    case onnx::AttributeProto_AttributeType_TENSOR:
    case onnx::AttributeProto_AttributeType_GRAPH:
    case onnx::AttributeProto_AttributeType_SPARSE_TENSOR:
    case onnx::AttributeProto_AttributeType_TYPE_PROTO:
    case onnx::AttributeProto_AttributeType_TENSORS:
    case onnx::AttributeProto_AttributeType_GRAPHS:
    case onnx::AttributeProto_AttributeType_SPARSE_TENSORS:
    case onnx::AttributeProto_AttributeType_TYPE_PROTOS:
    case onnx::AttributeProto_AttributeType_UNDEFINED:
      break;
  }

  return hash;
}

bool SameAttribute(const NodeAttributes::value_type& lhs, const NodeAttributes::value_type& rhs) {
  return lhs.first == rhs.first && AreEqual(lhs.second, rhs.second);
}

bool SameAttributes(const NodeAttributes* lhs, const NodeAttributes* rhs) {
  if (lhs == nullptr || rhs == nullptr) {
    return lhs == rhs;
  }
  return lhs->size() == rhs->size() &&
         std::equal(lhs->begin(), lhs->end(), rhs->begin(), &SameAttribute);
}

bool EquivalenceClass::operator==(const EquivalenceClass& other) const {
  if (this == &other) {
    return true;
  }

  // Below we compare inputs_ as pointers. This is valid due to how EquivalenceClass'es are constructed:
  // we'll never have two distinct but equal inputs_ here, so their addresses are effectively their value numbers.
  return hash_ == other.hash_ && output_index_ == other.output_index_ && discriminator_ == other.discriminator_ &&
         non_op_value_ == other.non_op_value_ &&
         op_type_ == other.op_type_ && domain_ == other.domain_ &&
         inputs_ == other.inputs_ &&
         SameAttributes(attributes_, other.attributes_);
}

std::size_t EquivalenceClass::CalculateHash() const {
  std::size_t hash = 0;
  UpdateHash(output_index_, hash);
  UpdateHash(discriminator_, hash);
  UpdateHash(non_op_value_, hash);
  UpdateHash(op_type_, hash);
  UpdateHash(domain_, hash);
  if (attributes_ != nullptr) {
    for (const auto& kv : *attributes_) {
      UpdateHash(kv.first, hash);
      UpdateHash(kv.second, &GetAttributeHash, hash);
    }
  }

  for (const auto& arg : inputs_) {
    for (const EquivalenceClass* input : arg) {
      UpdateHash(input, DeepPointerHash{}, hash);
    }
  }

  return hash;
}

// Representative of an equivalence class.
// node_index and output_index define the node that produced the node_arg.
// For inputs and constant initializers, output_index == kInvalidOutputIndex.
struct Representative {
  const NodeArg* node_arg;
  NodeIndex node_index;
  OutputIndex output_index;
};

struct NodeArgPtrHash {
  std::size_t operator()(const NodeArg* node_arg) const {
    return std::hash<const NodeArg*>{}(Normalize(node_arg));
  }
};

struct NodeArgPtrEquality {
  bool operator()(const NodeArg* lhs, const NodeArg* rhs) const {
    return Normalize(lhs) == Normalize(rhs);
  }
};

bool IsNodeSupported(const Node& node) {
  // skip control flow nodes, nodes that produce non-deterministic output, and DequantizeLinear (DQ) nodes.
  // the reason for skipping DQ is that the QDQ handling looks for QDQ node groups (DQ -> fp32 node -> Q node)
  // and does not allow for a DQ node to be used in multiple groups. coalescing multiple DQ nodes into one
  // would result in it having multiple consumers for its output, and it being used in multiple QDQ node groups.
  return !node.ContainsSubgraph() &&
         optimizer_utils::IsOperationDeterministic(node.Domain(), node.OpType()) &&
         !(node.Domain() == kOnnxDomain && node.OpType() == "DequantizeLinear");
}
}  // namespace

}  // namespace onnxruntime

namespace std {
template <>
struct hash<onnxruntime::EquivalenceClass> {
  std::size_t operator()(const onnxruntime::EquivalenceClass& val) const noexcept {
    return val.hash_;
  }
};
}  // namespace std

namespace onnxruntime {

Status CommonSubexpressionElimination::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  // Pool of equivalence classes; unique_ptr to guarantee stable address.
  InlinedVector<std::unique_ptr<EquivalenceClass>> unique_equivalence_classes;
  unique_equivalence_classes.reserve(graph.NumberOfNodes());

  // Maps an equivalence class of values to a representative NodeArg that belongs to this class.
  std::unordered_map<
      const EquivalenceClass*,
      Representative,
      DeepPointerHash, DeepPointerEquality>
      value_to_representative;

  // Maps every NodeArg to its equivalence class of.
  // This is the inverse of the above mapping, except that different NodeArgs can belong to the same
  // equivalence class. In that case these NodeArgs will be "merged" into one.
  std::unordered_map<const NodeArg*, const EquivalenceClass*, NodeArgPtrHash, NodeArgPtrEquality> equivalence_classes;

  int unique_discriminator = 1;

  for (NodeIndex node_index : node_topology_list) {
    Node* node = graph.GetNode(node_index);
    if (node == nullptr)
      continue;

    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));

    InlinedVector<const EquivalenceClass*> input_values;
    input_values.reserve(node->InputDefs().size());
    for (const NodeArg* input_def : node->InputDefs()) {
      auto it = equivalence_classes.find(input_def);
      if (it == equivalence_classes.end()) {
        // Because nodes are processed in topological order, this will always be
        // a non-op value (graph input or constant initializer).
        auto value = std::make_unique<EquivalenceClass>(input_def);
        const auto* raw_ptr = value.get();
        unique_equivalence_classes.push_back(std::move(value));
        value_to_representative.emplace(raw_ptr, Representative{input_def, 0, kInvalidOutputIndex});
        it = equivalence_classes.emplace_hint(it, input_def, raw_ptr);
      }

      input_values.push_back(it->second);
    }

    int discriminator = 0;
    if (!IsNodeSupported(*node)) {
      discriminator = ++unique_discriminator;
    }

    for (OutputIndex output_index = 0, end = static_cast<int>(node->OutputDefs().size());
         output_index < end; ++output_index) {
      const NodeArg* output_def = node->OutputDefs()[output_index];
      auto equivalence_class = std::make_unique<EquivalenceClass>(*node, input_values, output_index, discriminator);
      auto* raw_ptr = equivalence_class.get();

      auto it = value_to_representative.find(raw_ptr);
      if (it == value_to_representative.end()) {
        unique_equivalence_classes.push_back(std::move(equivalence_class));
        it = value_to_representative.emplace_hint(it, raw_ptr,
                                                  Representative{output_def, node_index, output_index});
      }

      equivalence_classes[output_def] = it->first;
    }
  }

  InlinedHashSet<const NodeArg*> graph_outputs;
  graph_outputs.reserve(graph_viewer.GetOutputs().size());
  graph_outputs.insert(graph_viewer.GetOutputs().begin(), graph_viewer.GetOutputs().end());

  for (NodeIndex node_index : node_topology_list) {
    Node* node = graph.GetNode(node_index);
    if (node == nullptr)
      continue;

    bool node_output_replaced = false;
    for (OutputIndex output_idx = 0, end = static_cast<int>(node->OutputDefs().size());
         output_idx < end; ++output_idx) {
      const NodeArg* output_def = node->OutputDefs()[output_idx];

      const EquivalenceClass* equivalence_class = equivalence_classes.at(output_def);
      const auto& representative = value_to_representative.at(equivalence_class);
      if (representative.node_arg == output_def) {
        // output_def is the representative of its equivalence class. All other values in the same class
        // will be replaced with output_def, but output_def itself should remain.
        continue;
      }

      ORT_ENFORCE(representative.output_index != kInvalidOutputIndex);

      if (graph_outputs.count(output_def) > 0) {
        // Currently, we don't support eliminating the graph's outputs.
        LOGS(logger, VERBOSE) << "Not eliminating output " << output_def->Name() << " of node " << node->Name() << "[" << node->OpType() << "] because it's the graph's output.";
        continue;
      }

      Node& replacement = *graph.GetNode(representative.node_index);
      OutputIndex replacement_output_idx = representative.output_index;
      graph_utils::ReplaceDownstreamNodeInput(graph, *node, output_idx, replacement, replacement_output_idx);
      node_output_replaced = true;
    }

    if (node_output_replaced) {
      modified = true;
      if (optimizer_utils::CheckOutputEdges(graph, *node, 0)) {
        graph.RemoveNode(node_index);
      }
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime
