// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

#include "node_unit.h"
#include "core/graph/graph_viewer.h"

namespace onnxruntime {

namespace {

enum class QLinearOpType : uint8_t {
  Unknown,  // Unknown or not a linear quantized op
  DequantizeLinear,
  QuantizeLinear,
  QLinearConv,
  QLinearMatMul,
  QLinearAdd,
  QLinearSigmoid,
  QLinearAveragePool,
  QLinearMul,
  QLinearReduceMean,
  QLinearConcat,
  QLinearGlobalAveragePool,
  QLinearLeakyRelu,
};

QLinearOpType GetQLinearOpType(const onnxruntime::Node& node) {
  const auto& op_type = node.OpType();
  if (op_type == "DequantizeLinear")
    return QLinearOpType::DequantizeLinear;
  else if (op_type == "QuantizeLinear")
    return QLinearOpType::QuantizeLinear;
  else if (op_type == "QLinearConv")
    return QLinearOpType::QLinearConv;
  else if (op_type == "QLinearMatMul")
    return QLinearOpType::QLinearMatMul;
  else if (op_type == "QLinearAdd")
    return QLinearOpType::QLinearAdd;
  else if (op_type == "QLinearSigmoid")
    return QLinearOpType::QLinearSigmoid;
  else if (op_type == "QLinearAveragePool")
    return QLinearOpType::QLinearAveragePool;
  else if (op_type == "QLinearMul")
    return QLinearOpType::QLinearMul;
  else if (op_type == "QLinearReduceMean")
    return QLinearOpType::QLinearReduceMean;
  else if (op_type == "QLinearConcat")
    return QLinearOpType::QLinearConcat;
  else if (op_type == "QLinearGlobalAveragePool")
    return QLinearOpType::QLinearGlobalAveragePool;
  else if (op_type == "QLinearLeakyRelu")
    return QLinearOpType::QLinearLeakyRelu;

  return QLinearOpType::Unknown;
}

// Ops have 1 input
bool IsUnaryQLinearOp(QLinearOpType type) {
  return type == QLinearOpType::QLinearSigmoid ||
         type == QLinearOpType::QLinearAveragePool ||
         type == QLinearOpType::QLinearGlobalAveragePool ||
         type == QLinearOpType::QLinearLeakyRelu ||
         type == QLinearOpType::QLinearReduceMean;
}

// Ops have 2 inputs
bool IsBinaryQLinearOp(QLinearOpType type) {
  return type == QLinearOpType::QLinearConv ||
         type == QLinearOpType::QLinearMatMul ||
         type == QLinearOpType::QLinearAdd ||
         type == QLinearOpType::QLinearMul;
}

// Ops have 1 or more inputs
bool IsVariadicQLinearOp(QLinearOpType type) {
  return type == QLinearOpType::QLinearConcat;
}

const std::vector<const Node*> GetQDQIONodes(const GraphViewer& graph_viewer,
                                             const QDQ::NodeGroup& node_group, bool is_input) {
  std::vector<const Node*> io_nodes;
  const auto& src_nodes = is_input ? node_group.dq_nodes : node_group.q_nodes;
  io_nodes.reserve(src_nodes.size());
  for (const auto& node_idx : src_nodes) {
    io_nodes.push_back(graph_viewer.GetNode(node_idx));
  }

  return io_nodes;
}

// Get the input or output NodeUnitIODef(s) for the given QDQ NodeGroup
std::vector<NodeUnitIODef> GetQDQIODefs(const Node& target_node, const QDQ::NodeGroup& node_group, bool is_input) {
  const auto& dq_or_q_nodes = is_input ? node_group.dq_nodes : node_group.q_nodes;
  const auto target_node_io_defs = is_input ? target_node.InputDefs() : target_node.OutputDefs();
  const size_t target_node_io_defs_size = target_node_io_defs.size();

  // Find all the quantized IO defs and indices (for the input/output of the target node)
  std::unordered_map<size_t, NodeUnitIODef> quantized_io_defs;
  quantized_io_defs.reserve(target_node_io_defs_size);

  auto cur = is_input ? target_node.InputEdgesBegin() : target_node.OutputEdgesBegin();
  auto end = is_input ? target_node.InputEdgesEnd() : target_node.OutputEdgesEnd();

  for (; cur != end; ++cur) {
    const Node& node = cur->GetNode();

    // If we can find the node index in the dq or q nodes this is a quantized input/output
    if (std::find(dq_or_q_nodes.cbegin(), dq_or_q_nodes.cend(), node.Index()) != dq_or_q_nodes.cend()) {
      const auto node_inputs = node.InputDefs();
      // quantization scale and zp are always the input[1, 2]
      NodeUnitIODef::QuantParam quant_param{*node_inputs[1], node_inputs.size() == 3 ? node_inputs[2] : nullptr};

      if (is_input) {
        // DQ is input to the target node, use the DstArgIndex
        auto idx = cur->GetDstArgIndex();
        // This is a DQ node, we are using x, x_scale, x_zp (input[0, 1, 2])
        quantized_io_defs.insert({idx, NodeUnitIODef{*node_inputs[0], quant_param}});
      } else {
        // Q is output of the target node, use the SrcArgIndex
        auto idx = cur->GetSrcArgIndex();
        // This is a Q node, we are using y (output[0]), y_scale, y_zp (input[1, 2])
        const auto node_outputs = node.OutputDefs();
        quantized_io_defs.insert({idx, NodeUnitIODef{*node_outputs[0], quant_param}});
      }
    }
  }

  // Construct the IODefs for this QDQ NodeGroup
  std::vector<NodeUnitIODef> io_defs;
  io_defs.reserve(target_node_io_defs_size);
  for (size_t i = 0; i < target_node_io_defs_size; i++) {
    // If we can find the NodeUnitIODef for this index, this is a quantized input/output
    if (quantized_io_defs.find(i) != quantized_io_defs.cend()) {
      io_defs.push_back(std::move(quantized_io_defs.at(i)));
    } else {
      // This is a regular input
      io_defs.push_back({*target_node_io_defs[i], std::nullopt});
    }
  }

  return io_defs;
}

}  // namespace

Status QDQ::NodeGroup::CanCreateNodeGroup(const GraphViewer& graph_viewer,
                                          const Node& target_node,
                                          gsl::span<const Node* const> dq_nodes,
                                          gsl::span<const Node* const> q_nodes) {
  // Within a QDQ node group, a target node input is the only consumer of each DQ.
  // This should have been ensured by the EnsureUniqueDQForNodeUnit graph transformer, but other graph modifications
  // may have happened since. Verify that this is still true.
  for (const auto* dq_node : dq_nodes) {
    const bool dq_produces_graph_output = graph_viewer.NodeProducesGraphOutput(*dq_node);
    ORT_RETURN_IF(dq_produces_graph_output,
                  "QDQ node group cannot have DQ node that produces a graph output. DQ node: ", dq_node->Name(),
                  ", target node: ", target_node.Name());

    const bool dq_has_single_output_edge_to_target =
        dq_node->GetOutputEdgesCount() == 1 &&
        dq_node->OutputEdgesBegin()->GetNode().Index() == target_node.Index();
    ORT_RETURN_IF_NOT(dq_has_single_output_edge_to_target,
                      "QDQ node group cannot have DQ that doesn't have a single output edge to the target node. "
                      "DQ node: ",
                      dq_node->Name(), ", target node: ", target_node.Name());
  }

  // an output from the target node can have either Q consumers or direct consumers. it cannot have both.
  // this must be checked on a per output basis.
  // e.g. TopK produces values and indices. The indices output won't be quantized, so even if we replace the TopK QDQ
  // node group with a quantized TopK, an int64_t indices value will be produced and can provide a graph output.
  if (!q_nodes.empty()) {
    auto cur_edge = target_node.OutputEdgesBegin();
    auto end_edge = target_node.OutputEdgesEnd();
    std::vector<const Node*> output_consumers(target_node.OutputDefs().size(), nullptr);

    for (; cur_edge != end_edge; ++cur_edge) {
      auto output_idx = cur_edge->GetSrcArgIndex();
      const Node& this_consumer = cur_edge->GetNode();
      const Node* existing_consumer = output_consumers[output_idx];

      if (existing_consumer != nullptr) {
        // another edge for this output. either both are Q or both are not.
        bool valid = true;
        if (existing_consumer->OpType() == "QuantizeLinear") {
          valid = this_consumer.OpType() == "QuantizeLinear";
        } else {
          valid = this_consumer.OpType() != "QuantizeLinear";
        }

        ORT_RETURN_IF_NOT(valid,
                          "QDQ node group cannot have an output from the target node being consumed by a Q node and "
                          "a non-Q node. target node: ",
                          target_node.Name());
      } else {
        output_consumers[output_idx] = &this_consumer;
      }
    }

    const auto& graph_outputs = graph_viewer.GetOutputs();
    for (size_t idx = 0, end = output_consumers.size(); idx < end; ++idx) {
      // any output with a Q cannot be a graph output as it will disappear if the QDQ node unit is converted to
      // a quantized op.
      if (output_consumers[idx] != nullptr && output_consumers[idx]->OpType() == "QuantizeLinear") {
        const auto& output_name = target_node.OutputDefs()[idx]->Name();
        bool is_graph_output = std::any_of(graph_outputs.begin(), graph_outputs.end(),
                                           [&output_name](const NodeArg* node_arg) {
                                             return node_arg->Name() == output_name;
                                           });
        ORT_RETURN_IF(is_graph_output,
                      "QDQ node group cannot have an output from the target node that is consumed by a Q node and "
                      "a graph output. target node: ",
                      target_node.Name(), " output idx:", idx);
      }
    }
  }

  return Status::OK();
}
NodeUnit::NodeUnit(const Node& node)
    : target_node_(node),
      type_(Type::SingleNode),
      input_edge_count_(node.GetInputEdgesCount()) {
  InitForSingleNode();
}

NodeUnit::NodeUnit(const GraphViewer& graph_viewer, const QDQ::NodeGroup& node_group)
    : dq_nodes_{GetQDQIONodes(graph_viewer, node_group, true /* is_input */)},
      target_node_(*graph_viewer.GetNode(node_group.target_node)),
      q_nodes_{GetQDQIONodes(graph_viewer, node_group, false /* is_input */)},
      type_(Type::QDQGroup),
      inputs_{GetQDQIODefs(target_node_, node_group, true /* is_input */)},
      outputs_{GetQDQIODefs(target_node_, node_group, false /* is_input */)} {
  ORT_THROW_IF_ERROR(QDQ::NodeGroup::CanCreateNodeGroup(graph_viewer, target_node_, dq_nodes_, q_nodes_));

  input_edge_count_ = std::accumulate(dq_nodes_.cbegin(), dq_nodes_.cend(), size_t(0),
                                      [](size_t acc, const Node* node) { return acc + node->GetInputEdgesCount(); });

  // add edges for inputs that are not from DQ nodes. there is one edge to each DQ node.
  // other inputs could come from initializers or graph inputs (no edges) or other nodes (edge).
  input_edge_count_ += target_node_.GetInputEdgesCount() - dq_nodes_.size();

  // create output edges. each target node output either goes to Q node/s or non-Q node/s.
  // ValidateNodeGroupQDQNodes ensures this.
  auto cur_edge = target_node_.OutputEdgesBegin();
  auto end_edge = target_node_.OutputEdgesEnd();
  for (; cur_edge != end_edge; ++cur_edge) {
    const Node& node = cur_edge->GetNode();

    // if node is in q_nodes we hide the Q node.
    if (std::find(q_nodes_.cbegin(), q_nodes_.cend(), &node) != q_nodes_.cend()) {
      auto src_idx = cur_edge->GetSrcArgIndex();
      auto q_cur_edge = node.OutputEdgesBegin();
      auto q_end_edge = node.OutputEdgesEnd();
      for (; q_cur_edge != q_end_edge; ++q_cur_edge) {
        output_edges_.insert(Node::EdgeEnd{q_cur_edge->GetNode(), src_idx, q_cur_edge->GetDstArgIndex()});
      }
    } else {
      // non-Q node, or Q node that isn't in the QDQ node group (unexpected but may be possible). add as-is.
      output_edges_.insert(*cur_edge);
    }
  }
}

const std::string& NodeUnit::Domain() const noexcept { return target_node_.Domain(); }
const std::string& NodeUnit::OpType() const noexcept { return target_node_.OpType(); }
const std::string& NodeUnit::Name() const noexcept { return target_node_.Name(); }
int NodeUnit::SinceVersion() const noexcept { return target_node_.SinceVersion(); }
NodeIndex NodeUnit::Index() const noexcept { return target_node_.Index(); }
const Path& NodeUnit::ModelPath() const noexcept { return target_node_.ModelPath(); }
ProviderType NodeUnit::GetExecutionProviderType() const noexcept { return target_node_.GetExecutionProviderType(); }

void NodeUnit::InitForSingleNode() {
  const auto& input_defs = target_node_.InputDefs();
  const auto& output_defs = target_node_.OutputDefs();
  auto qlinear_type = GetQLinearOpType(target_node_);
  if (qlinear_type == QLinearOpType::Unknown || IsVariadicQLinearOp(qlinear_type)) {  // TODO, add variadic support
    // Not a Qlinear op, add all inputs / outputs
    auto add_all_io = [](std::vector<NodeUnitIODef>& defs,
                         const ConstPointerContainer<std::vector<NodeArg*>>& node_defs) {
      defs.reserve(node_defs.size());

      for (const auto def : node_defs) {
        defs.push_back(NodeUnitIODef{*def, std::nullopt});
      }
    };

    add_all_io(inputs_, input_defs);
    add_all_io(outputs_, output_defs);
  } else if (IsUnaryQLinearOp(qlinear_type)) {
    // Unary QLinear Op has 5 inputs
    // x, x_scale, x_zp, y_scale, y_zp (optional)
    inputs_.push_back(NodeUnitIODef{*input_defs[0], NodeUnitIODef::QuantParam{*input_defs[1], input_defs[2]}});
    outputs_.push_back(NodeUnitIODef{*output_defs[0],
                                     NodeUnitIODef::QuantParam{*input_defs[3],
                                                               input_defs.size() > 4 ? input_defs[4] : nullptr}});

  } else if (IsBinaryQLinearOp(qlinear_type)) {
    // Binary QLinear Op has 9 inputs
    // x1, x1_scale, x1_zp, x2/w, x2_scale, x2_zp, y_scale , y_zp, B
    inputs_.push_back(NodeUnitIODef{*input_defs[0], NodeUnitIODef::QuantParam{*input_defs[1], input_defs[2]}});
    inputs_.push_back(NodeUnitIODef{*input_defs[3], NodeUnitIODef::QuantParam{*input_defs[4], input_defs[5]}});

    if (input_defs.size() == 9) {                                      // has Bias
      inputs_.push_back(NodeUnitIODef{*input_defs[8], std::nullopt});  // for Bias the scale and zp are optional
    }

    outputs_.push_back(NodeUnitIODef{*output_defs[0], NodeUnitIODef::QuantParam{*input_defs[6], input_defs[7]}});

  } else if (qlinear_type == QLinearOpType::DequantizeLinear) {
    // DequantizeLinear has 3 inputs
    // x, x_scale, x_zp
    // output is not quantized
    inputs_.push_back(NodeUnitIODef{*input_defs[0], NodeUnitIODef::QuantParam{*input_defs[1], input_defs.size() == 3
                                                                                                  ? input_defs[2]
                                                                                                  : nullptr}});
    outputs_.push_back(NodeUnitIODef{*output_defs[0], std::nullopt});

  } else if (qlinear_type == QLinearOpType::QuantizeLinear) {
    // QuantizeLinear the input is not quantized and has 3 inputs
    // x, y_scale, y_zp (optional)
    // The output is quantized
    inputs_.push_back(NodeUnitIODef{*input_defs[0], std::nullopt});
    outputs_.push_back(NodeUnitIODef{*output_defs[0], NodeUnitIODef::QuantParam{*input_defs[1], input_defs.size() == 3
                                                                                                    ? input_defs[2]
                                                                                                    : nullptr}});
  } else {
    ORT_THROW("The QLinear op [", static_cast<uint8_t>(qlinear_type), "] is not supported");
  }
}

Node::EdgeConstIterator NodeUnit::OutputEdgesBegin() const {
  return (type_ == Type::SingleNode) ? target_node_.OutputEdgesBegin() : output_edges_.begin();
}

Node::EdgeConstIterator NodeUnit::OutputEdgesEnd() const {
  return (type_ == Type::SingleNode) ? target_node_.OutputEdgesEnd() : output_edges_.end();
}

std::vector<const Node*> NodeUnit::GetAllNodesInGroup() const noexcept {
  std::vector<const Node*> all_nodes = dq_nodes_;
  all_nodes.push_back(&target_node_);
  all_nodes.insert(all_nodes.end(), q_nodes_.begin(), q_nodes_.end());
  return all_nodes;
}

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
