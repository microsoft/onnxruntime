// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common_subexpression_elimination.h"
#include "core/graph/graph_utils.h"

#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace onnxruntime {

namespace {
  struct DeepPointerEquality {
    template<typename Ptr>
    bool operator()(const Ptr& lhs, const Ptr& rhs) const {
      if (lhs == nullptr || rhs == nullptr) {
        return lhs == rhs;
      }
      return *lhs == *rhs;
    }
  };

  struct DeepPointerHash {
    template<typename Ptr>
    std::size_t operator()(const Ptr& value) const {
      if (value == nullptr) {
        return 0;
      }
      return std::hash<std::decay_t<decltype(*value)>>{}(*value);
    }
  };

  template<typename T, typename Hasher>
  void UpdateHash(const T& x, Hasher hasher, std::size_t& hash) {
    constexpr std::size_t kPrime = 31013;
    hash = hash * kPrime + hasher(x);
  }

  template<typename T>
  void UpdateHash(const T& x, std::size_t& hash) {
    UpdateHash(x, std::hash<T>{}, hash);
  }

  template<typename Container>
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
    bool operator!=(const EquivalenceClass& other) const;

    friend struct ::std::hash<EquivalenceClass>;
    friend std::vector<const EquivalenceClass*> Normalize(std::vector<const EquivalenceClass*> inputs);

    explicit EquivalenceClass(const NodeArg* non_op_value)
      : attributes_(nullptr)
      , output_index_(kInvalidOutputIndex)
      , non_op_value_(Normalize(non_op_value))
      , discriminator_(0)
      , hash_(CalculateHash())
    { }

    EquivalenceClass(const Node* node, std::vector<const EquivalenceClass*> inputs,
                OutputIndex output_index, int discriminator)
      : op_type_(node->OpType())
      , domain_(node->Domain())
      , inputs_(Normalize(std::move(inputs)))
      , attributes_(&node->GetAttributes())
      , output_index_(output_index)
      , non_op_value_(nullptr)
      , discriminator_(discriminator)
      , hash_(CalculateHash())
    { }

  private:
    std::size_t CalculateHash() const;

    // Operation and domain of the node that produces this value.
    const std::string op_type_;
    const std::string domain_;

    // Inputs to the operation.
    const std::vector<const EquivalenceClass*> inputs_;

    // Attributes of the operation.
    const NodeAttributes* attributes_;

    // Index of this value in the output list of the operation.
    const OutputIndex output_index_;

    // When the value is not an output of an operation, (i.e., a constant initializer or an input),
    // non_op_value is set to the corresponding NodeArg, and other fields are empty.
    // Currently, different NodeArg's are always considered different values, although we
    // could merge equal initializers.
    const NodeArg* non_op_value_;

    // When an operation is not supported by the CSE optimization pass, we consider its
    // outputs unique (not equal to other values). For this purpose we assign a unique
    // discriminator for such values.
    const int discriminator_;

    const std::size_t hash_;
  };

  std::vector<const EquivalenceClass*> Normalize(std::vector<const EquivalenceClass*> inputs) {
    // Remove missing optional inputs from the back
    while (!inputs.empty() && inputs.back()->output_index_ == kInvalidOutputIndex && inputs.back()->non_op_value_ == nullptr) {
      inputs.pop_back();
    }

    return inputs;
  }

  template<typename Range>
  bool AreRangesEqual(const Range& lhs, const Range& rhs) {
    return std::equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
  }

  template<typename Range, typename Pred>
  bool AreRangesEqual(const Range& lhs, const Range& rhs, const Pred& pred) {
    return std::equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), pred);
  }

  bool AreEqual(const ONNX_NAMESPACE::AttributeProto& lhs, const ONNX_NAMESPACE::AttributeProto& rhs) {
    if (&lhs == &rhs) {
      return true;
    }

    if (lhs.type() != rhs.type() || lhs.name() != rhs.name()) {
      return false;
    }

    switch(lhs.type()) {
      case onnx::AttributeProto_AttributeType_FLOAT: return lhs.f() == rhs.f();
      case onnx::AttributeProto_AttributeType_INT: return lhs.i() == rhs.i();
      case onnx::AttributeProto_AttributeType_STRING: return lhs.s() == rhs.s();
      case onnx::AttributeProto_AttributeType_FLOATS: return AreRangesEqual(lhs.floats(), rhs.floats());
      case onnx::AttributeProto_AttributeType_INTS: return AreRangesEqual(lhs.ints(), rhs.ints());
      case onnx::AttributeProto_AttributeType_STRINGS: return AreRangesEqual(lhs.strings(), rhs.strings());
      case onnx::AttributeProto_AttributeType_TENSOR:
      case onnx::AttributeProto_AttributeType_GRAPH:
      case onnx::AttributeProto_AttributeType_SPARSE_TENSOR:
      case onnx::AttributeProto_AttributeType_TENSORS:
      case onnx::AttributeProto_AttributeType_GRAPHS:
      case onnx::AttributeProto_AttributeType_SPARSE_TENSORS:
      case onnx::AttributeProto_AttributeType_UNDEFINED:
        return false; // Don't support these attributes for now; corresponding nodes will be considered distinct.
    }

    return false;
  }

  std::size_t GetAttributeHash(const ONNX_NAMESPACE::AttributeProto& attr) {
    std::size_t hash = 0;
    UpdateHash(attr.type(), hash);
    UpdateHash(attr.name(), hash);
    switch (attr.type()) {
      case onnx::AttributeProto_AttributeType_FLOAT: UpdateHash(attr.f(), hash); break;
      case onnx::AttributeProto_AttributeType_INT: UpdateHash(attr.i(), hash); break;
      case onnx::AttributeProto_AttributeType_STRING: UpdateHash(attr.s(), hash); break;
      case onnx::AttributeProto_AttributeType_FLOATS: UpdateHashWithContainer(attr.floats(), hash); break;
      case onnx::AttributeProto_AttributeType_INTS: UpdateHashWithContainer(attr.ints(), hash); break;
      case onnx::AttributeProto_AttributeType_STRINGS: UpdateHashWithContainer(attr.strings(), hash); break;
      case onnx::AttributeProto_AttributeType_TENSOR:
      case onnx::AttributeProto_AttributeType_GRAPH:
      case onnx::AttributeProto_AttributeType_SPARSE_TENSOR:
      case onnx::AttributeProto_AttributeType_TENSORS:
      case onnx::AttributeProto_AttributeType_GRAPHS:
      case onnx::AttributeProto_AttributeType_SPARSE_TENSORS:
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
    return std::equal(lhs->begin(), lhs->end(), rhs->begin(), rhs->end(), &SameAttribute);
  }

  bool EquivalenceClass::operator==(const EquivalenceClass& other) const {
    // Below we compare inputs_ as pointers. This is valid due to how EquivalenceClass'es are constructed:
    // we'll never have two distinct but equal inputs_ here, so their addresses are effectively their value numbers.
    return hash_ == other.hash_ && output_index_ == other.output_index_ && discriminator_ == other.discriminator_ &&
      non_op_value_ == other.non_op_value_ &&
      op_type_ == other.op_type_ && domain_ == other.domain_ &&
      AreRangesEqual(inputs_, other.inputs_) &&
      SameAttributes(attributes_, other.attributes_);
  }

  bool EquivalenceClass::operator!=(const EquivalenceClass& other) const {
    return !operator==(other);
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

    for (const EquivalenceClass* input : inputs_) {
      UpdateHash(input, DeepPointerHash{}, hash);
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
}

}

namespace std {
  template<> struct hash<onnxruntime::EquivalenceClass> {
    std::size_t operator()(const onnxruntime::EquivalenceClass& val) const noexcept {
      return val.hash_;
    }
  };
}

namespace onnxruntime {

Status CommonSubexpressionElimination::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  // Pool of equivalence classes; unique_ptr to guarantee stable address.
  std::vector<std::unique_ptr<EquivalenceClass>> unique_equivalence_classes;

  // Maps an equivalence class of values to a representative NodeArg that belongs to this class.
  std::unordered_map<
    const EquivalenceClass*,
    Representative,
    DeepPointerHash, DeepPointerEquality> value_to_representative;

  // Reverse mapping.
  std::unordered_map<const NodeArg*, const EquivalenceClass*, NodeArgPtrHash, NodeArgPtrEquality> equivalence_classes;

  {
    std::unordered_set<const NodeArg*, NodeArgPtrHash, NodeArgPtrEquality> non_ops;

    const auto& inputs = graph.GetInputs();
    non_ops.insert(inputs.begin(), inputs.end());
    non_ops.insert(nullptr); // Signifies missing input.

    const auto& initializers = graph.GetAllInitializedTensors();
    for (const auto& kv : initializers) {
      const auto* node_arg = graph.GetNodeArg(kv.first);
      non_ops.insert(node_arg);
    }

    for (const NodeArg* non_op : non_ops) {
      auto value = onnxruntime::make_unique<EquivalenceClass>(non_op);
      const auto* raw_ptr = value.get();
      unique_equivalence_classes.emplace_back(std::move(value));
      value_to_representative.emplace(raw_ptr, Representative{non_op, 0, kInvalidOutputIndex});
      equivalence_classes[non_op] = raw_ptr;
    }
  }

  int unique_discriminator = 1;

  for (NodeIndex node_index : node_topology_list) {
    const Node* node = graph_viewer.GetNode(node_index);
    if (node == nullptr)
      continue;

    ORT_RETURN_IF_ERROR(Recurse(*graph.GetNode(node_index), modified, graph_level, logger));

    std::vector<const EquivalenceClass*> input_values;
    input_values.reserve(node->InputDefs().size());
    for (const NodeArg* input_def : node->InputDefs()) {
      input_values.push_back(equivalence_classes.at(input_def));
    }

    int discriminator = 0;
    const bool is_supported_node = node->Domain() == kOnnxDomain && !node->ContainsSubgraph();
    if (!is_supported_node) {
      discriminator = ++unique_discriminator;
    }

    for (OutputIndex output_index = 0; output_index < static_cast<int>(node->OutputDefs().size()); ++output_index) {
      const NodeArg* output_def = node->OutputDefs()[output_index];
      auto equivalence_class = onnxruntime::make_unique<EquivalenceClass>(node, input_values, output_index, discriminator);
      auto* raw_ptr = equivalence_class.get();

      auto it = value_to_representative.find(raw_ptr);
      if (it == value_to_representative.end()) {
        unique_equivalence_classes.emplace_back(std::move(equivalence_class));
        it = value_to_representative.emplace_hint(it, raw_ptr,
          Representative{output_def, node_index, output_index});
      }

      equivalence_classes[output_def] = it->first;
    }
  }

  std::unordered_set<const NodeArg*> graph_outputs;
  graph_outputs.insert(graph.GetOutputs().begin(), graph.GetOutputs().end());

  for (NodeIndex node_index : node_topology_list) {
    Node* node = graph.GetNode(node_index);
    if (node == nullptr)
      continue;

    bool node_output_replaced = false;
    for (OutputIndex output_idx = 0; output_idx < static_cast<int>(node->OutputDefs().size()); ++output_idx) {
      const NodeArg* output_def = node->OutputDefs()[output_idx];

      const EquivalenceClass* equivalence_class = equivalence_classes.at(output_def);
      const auto& representative = value_to_representative.at(equivalence_class);
      if (representative.node_arg == output_def) {
        // output_def is the representative of its equivalence class. All other values in the same class
        // will be replaced with output_def, but output_def itself should remain.
        continue;
      }

      if (graph_outputs.count(output_def) > 0) {
        // Currently, eliminating a value that is the graph's output is not supported.
        LOGS(logger, INFO) << "Not eliminating output " << output_def->Name() << " of node " << node->Name() <<
          "[" << node->OpType() << "] because it's the graph's output.";
        continue;
      }

      if (representative.output_index != kInvalidOutputIndex) {
        Node& replacement = *graph.GetNode(representative.node_index);
        OutputIndex replacement_output_idx = representative.output_index;
        graph_utils::ReplaceDownstreamNodeInput(graph, *node, output_idx, replacement, replacement_output_idx);
        node_output_replaced = true;
      }
    }

    if (node_output_replaced) {
      modified = true;
      if (node->GetOutputEdgesCount() == 0) {
        graph.RemoveNode(node_index);
      }
    }
  }

  return Status::OK();
}

}

