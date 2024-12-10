// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <array>
#include <set>
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/openvino/qdq_transformations/qdq_stripping.h"

namespace onnxruntime {
namespace openvino_ep {

enum class SkipReason {
  Int16QDQ,
  DuplicateDQ,
  ConstInitUnsupportedDQ,
  SandwichedDQ,
  Other
};

constexpr std::string_view DuplicateDQ = "/duplicated";

constexpr ONNX_NAMESPACE::TensorProto_DataType DT_UINT16 = ONNX_NAMESPACE::TensorProto_DataType_UINT16;
constexpr ONNX_NAMESPACE::TensorProto_DataType DT_INT16 = ONNX_NAMESPACE::TensorProto_DataType_INT16;

// Return the data type of the qdq node.
// Check output type of Q and input type of DQ to determine it as zero_point is an optional input and may not exist
static ONNX_NAMESPACE::TensorProto_DataType GetQDQDataType(const Node* qdq_node) {
  if (qdq_node->OpType() == "QuantizeLinear") {
    return static_cast<ONNX_NAMESPACE::TensorProto_DataType>(
        qdq_node->OutputDefs().at(0)->TypeAsProto()->tensor_type().elem_type());
  } else if (qdq_node->OpType() == "DequantizeLinear") {
    return static_cast<ONNX_NAMESPACE::TensorProto_DataType>(
        qdq_node->InputDefs().at(0)->TypeAsProto()->tensor_type().elem_type());
  } else {
    ORT_THROW("Invalid QDQ Op Type when fetching datatype of parameter");
  }
}

// Creates a new NodeArg from an input (or output) of a QDQ node unit. If the input/output is quantized,
// this function modifies the tensor type to the specified float type.
static NodeArg& ProcessNodeUnitIO(onnxruntime::Graph& dst_graph,
                                  const onnxruntime::GraphViewer& src_graph,
                                  std::set<std::string>& initializers_to_keep,
                                  const NodeUnitIODef& io_def) {
  const std::string& name = io_def.node_arg.Name();
  const ONNX_NAMESPACE::TypeProto* orig_type_proto = io_def.node_arg.TypeAsProto();

  // Handle quantized input or output. Convert to float type.
  if (io_def.quant_param.has_value()) {
    // Copy the original quantized type proto, but update the type to be the type of scale param.
    const auto& src_initializers = src_graph.GetAllInitializedTensors();
    const std::string& scale_initializer_name = io_def.quant_param->scale.Name();
    auto tensor_proto_iter = src_initializers.find(scale_initializer_name);

    ORT_ENFORCE(tensor_proto_iter != src_initializers.end(),
                "Unable to find scale initializer ", scale_initializer_name);

    const ONNX_NAMESPACE::TensorProto* scale_tensor_proto = tensor_proto_iter->second;
    int32_t float_type = scale_tensor_proto->data_type();

    // Noe set the arg type to the float type of scale. Could be one of float/float16/bfloat16
    std::unique_ptr<ONNX_NAMESPACE::TypeProto> type_proto = ONNX_NAMESPACE::TypeProto::Create();
    type_proto->copy_from(orig_type_proto);
    type_proto->mutable_tensor_type()->set_elem_type(float_type);

    if (src_graph.GetAllInitializedTensors().count(name)) {
      initializers_to_keep.insert({name});
    }

    return dst_graph.GetOrCreateNodeArg(name, type_proto.get());
  }

  // Unquantized input or output. Just copy.
  return dst_graph.GetOrCreateNodeArg(name, orig_type_proto);
}

static void KeepInitsInDstGraph(std::set<std::string>& initializers_to_keep,
                                const onnxruntime::GraphViewer& src_graph,
                                const Node* qdq_node) {
  for (const auto& def : qdq_node->InputDefs()) {
    if (src_graph.GetAllInitializedTensors().count(def->Name())) {
      initializers_to_keep.insert({def->Name()});
    }
  }
}

static void AddNode(std::set<std::string>& initializers_to_keep,
                    const onnxruntime::GraphViewer& src_graph,
                    onnxruntime::Graph& dst_graph,
                    const Node& node) {
  dst_graph.AddNode(node);
  KeepInitsInDstGraph(initializers_to_keep, src_graph, &node);
}

static bool IsConnectedQPresent(const onnxruntime::GraphViewer& src_graph,
                                const std::vector<const Node*>& dst_nodes,
                                const Node* dq_node,
                                ConstPointerContainer<std::vector<NodeArg*>> input_defs) {
  // Check if the connected Q was already kept in the dst graph. If it's not found, don't reconnect
  if (auto it = std::find_if(dst_nodes.begin(), dst_nodes.end(),
                             [&](const Node* n) {
                               // search for the connected Q in the dst graph

                               if (src_graph.IsConstantInitializer(input_defs.at(0)->Name(), true)) {
                                 // If the the DQ's input is a constant initializer, and found in the graph then
                                 // proceed to remove the duplicate
                                 return true;
                               } else {
                                 // Otherwise, check if the DQ's Q input is already present in the dst graph
                                 // Check the OpType so we don't mistake Identity for Q
                                 return (n->Name() == dq_node->InputNodesBegin()->Name() &&
                                         n->OpType() == "QuantizeLinear");
                               }
                             });
      it != std::end(dst_nodes)) return true;
  return false;
}

static bool IsFirstComputeOpAboveSoftMax(const Node* qdq_node) {
  if (qdq_node->OpType() != "QuantizeLinear" && qdq_node->OpType() != "DequantizeLinear") {
    if (qdq_node->OpType() == "Softmax")
      return true;
    else
      return false;
  } else {
    if (qdq_node->GetInputEdgesCount())
      return IsFirstComputeOpAboveSoftMax(&*qdq_node->InputNodesBegin());
    else
      return false;
  }
}

static bool IsFirstComputeOpBelowConvMatMul(const Node* qdq_node) {
  if (qdq_node->OpType() != "QuantizeLinear" && qdq_node->OpType() != "DequantizeLinear") {
    if (qdq_node->OpType() == "Conv" || qdq_node->OpType() == "MatMul")
      return true;
    else
      return false;
  } else {
    if (qdq_node->GetOutputEdgesCount())
      return IsFirstComputeOpBelowConvMatMul(&*qdq_node->OutputNodesBegin());
    else
      return false;
  }
}

static bool IsQDQSandwichedBetweenSoftmaxAndConvMatMulOps(const Node* qdq_node) {
  return IsFirstComputeOpAboveSoftMax(qdq_node) && IsFirstComputeOpBelowConvMatMul(qdq_node);
}

static const Node* GetFirstComputeOpAboveThisDQ(const Node* dq_node) {
  if (dq_node->OpType() != "QuantizeLinear" && dq_node->OpType() != "DequantizeLinear") {
    return dq_node;
  } else {
    if (dq_node->GetInputEdgesCount())
      return GetFirstComputeOpAboveThisDQ(&*dq_node->InputNodesBegin());
    else
      return dq_node;
  }
}

static const Node* GetFirstComputeOpBelowThisQ(const Node* q_node) {
  if (q_node->OpType() != "QuantizeLinear" && q_node->OpType() != "DequantizeLinear") {
    return q_node;
  } else {
    if (q_node->GetOutputEdgesCount())
      return GetFirstComputeOpBelowThisQ(&*q_node->OutputNodesBegin());
    else
      return q_node;
  }
}

// Used to find if input 0 of the target node DQ is a constant initializer
static bool IsAnyDQAConstantInitializer(const Node* target_node, const onnxruntime::GraphViewer& src_graph) {
  bool is_const_init = false;
  for (Node::NodeConstIterator it_dq = target_node->InputNodesBegin(); it_dq != target_node->InputNodesEnd(); ++it_dq) {
    const auto& DQ = &*it_dq;
    if (DQ->OpType() != "DequantizeLinear") continue;
    is_const_init |= src_graph.IsConstantInitializer(DQ->InputDefs().at(0)->Name(), true);
  }

  return is_const_init;
}

// Used to find if input 0 of the connected Q is a constant initializer
static bool IsConnectedQAConstantInitializer(const Node* dq_node, const onnxruntime::GraphViewer& src_graph) {
  bool is_const_init = false;
  for (Node::NodeConstIterator it_q = dq_node->InputNodesBegin(); it_q != dq_node->InputNodesEnd(); ++it_q) {
    const auto& Q = &*it_q;
    if (Q->OpType() != "QuantizeLinear") continue;
    is_const_init |= src_graph.IsConstantInitializer(Q->InputDefs().at(0)->Name(), true);
  }

  return is_const_init;
}

// Check required because in some cases, when a NodeUnit cannot be formed with this standalone DQ
// we still need to check if it feeds into a supported Op
static bool DQFeedsASupportedOp(const Node* dq_node) {
  if (!dq_node->GetOutputEdgesCount()) return false;  // Only feeds the graph output, and not any node

  const auto& target_node = *dq_node->OutputNodesBegin();
  const auto& op_type = target_node.OpType();

  if (op_type == "Conv" || op_type == "MatMul") {
    // Conv and MatMul always keeps int8 DQs except if the DQ is sandwiched between Softmax and Conv/MatMul
    if (IsQDQSandwichedBetweenSoftmaxAndConvMatMulOps(dq_node)) {
      return false;
    } else {
      return true;
    }
  } else if (op_type == "Add") {
    // Add => keeps all DQs
    return true;
  }
  return false;
}

// Previous Target -> Q -> DQ -> Current Target
// Traverse back to the previous target node of DQ and check if it's an op listed in supported_ops
// If the inputs of current target node are constant initializers, then the QDQ pair is invalid
// Example: MatMul/Conv -> QDQ (uint8) -> (managed) Add/Div/Mul ==> MatMul/Conv -> (managed) Add/Div/Mul
static bool IsPreviousTargetNodeOfDQValid(const Node* DQ,
                                          const Node* current_target,
                                          const onnxruntime::GraphViewer& src_graph,
                                          bool check_const_init) {
  // Iterate over all inputs of this DQ. Typically, only one input is expected
  // We don't check for types here as it's handled in the respective ruleset functions

  const Node* prev_target = GetFirstComputeOpAboveThisDQ(DQ);

  // #1 If previous target is one of supported
  if (prev_target->OpType() == "Conv" || prev_target->OpType() == "MatMul") {
    // #2 For Mul/Div, the DQ shouldn't be a const init
    if (check_const_init && IsAnyDQAConstantInitializer(current_target, src_graph))
      return false;  // because Add/Mul/Div with const init inputs are not supported if prev target is Conv/MatMul
    else
      return true;                              // For non-Conv/non-MatMul const init doesn't matter
  } else if (prev_target->OpType() == "Add") {  // because Add is a supported Op
    return true;
  }

  return false;
}

// Current Target -> Q -> DQ -> Next Target
// Do the inverse of the function above. Check to keep the Q if the next target is valid
static bool IsNextTargetNodeOfQValid(const Node* Q,
                                     const Node* current_target,
                                     const onnxruntime::GraphViewer& src_graph,
                                     const std::vector<std::string>& supported_ops,
                                     bool check_const_init) {
  const Node* next_target = GetFirstComputeOpBelowThisQ(Q);

  if (std::find(supported_ops.begin(), supported_ops.end(), next_target->OpType()) != supported_ops.end()) {
    // Always check const inits if Add is the next target
    if ((check_const_init || next_target->OpType() == "Add") && IsAnyDQAConstantInitializer(next_target, src_graph)) {
      return false;  // because Add/Mul/Div with const init inputs are not supported
    } else if (next_target->OpType() == "Conv" || next_target->OpType() == "MatMul") {
      // If any DQ of this Conv/MatMul is sandwiched between Softmax and Conv/MatMul, then don't keep it
      bool is_valid = true;
      for (Node::NodeConstIterator it_sw = next_target->InputNodesBegin();
           it_sw != next_target->InputNodesEnd(); ++it_sw) {
        const auto& sw_dq_node = &*it_sw;
        if (sw_dq_node->OpType() != "DequantizeLinear") return true;
        if (IsQDQSandwichedBetweenSoftmaxAndConvMatMulOps(sw_dq_node))
          is_valid &= false;
      }
      return is_valid;
    } else {
      return true;  // because the next target is supported
    }
  } else if (current_target->OpType() == "Conv" || current_target->OpType() == "MatMul") {
    return true;  // Conv and MatMul can keep all Qs by default. Is there a better way to check this?
  } else {
    return false;  // because the next target is not supported
  }
}

static bool CheckDQRuleSet(const NodeUnit& node_unit,
                           const Node* dq_node,
                           const onnxruntime::GraphViewer& src_graph,
                           SkipReason& reason) {
  const auto& target_node = node_unit.GetNode();
  const auto& op_type = node_unit.OpType();

  // #1 Reverse DQ duplication
  if (dq_node->Name().find(DuplicateDQ) != std::string::npos) {
    reason = SkipReason::DuplicateDQ;
    return false;
  }

  // #2 If input 0 is a constant initializer feeding to even unsupported ops with an unsupported type, keep it
  // TODO(sspintel): check if this needs to be done only for certain supported Ops
  if (src_graph.IsConstantInitializer(dq_node->InputDefs().at(0)->Name(), true)) {
    return true;
  }

  // #3 If UInt16 DQ, don't keep it
  if (GetQDQDataType(dq_node) == DT_UINT16 || GetQDQDataType(dq_node) == DT_INT16) {
    reason = SkipReason::Int16QDQ;
    return false;
  }

  // DQs in Double QDQ cases should be kept; Use scale param's name to verify that it's converting DQ
  if (dq_node->InputDefs().at(1)->Name().find("scale_convert") != std::string::npos &&
      !IsQDQSandwichedBetweenSoftmaxAndConvMatMulOps(dq_node))
    return true;

  if (op_type == "Conv" || op_type == "MatMul") {
    // Conv and MatMul always keeps int8 DQs except if the DQ is sandwiched between Softmax and Conv/MatMul
    if (IsQDQSandwichedBetweenSoftmaxAndConvMatMulOps(dq_node)) {
      reason = SkipReason::SandwichedDQ;
      return false;
    } else {
      return true;
    }
  } else if (op_type == "Add") {
    // Add keeps all DQs except if it has const inits
    return !IsAnyDQAConstantInitializer(&target_node, src_graph);
  } else if (op_type == "Mul" || op_type == "Div") {
    // Keep DQ of Mul and Div only if the target that preceds it is a supported Op in this list and also check if
    // inputs of Mul and Div have constant initializers. If they do, then don't keep the DQ.
    return IsPreviousTargetNodeOfDQValid(dq_node, &target_node, src_graph, true);
  } else {
    // Keep DQ of an unsupported Op only if the target that preceds it is a supported Op in this list
    return IsPreviousTargetNodeOfDQValid(dq_node, &target_node, src_graph, false);
  }
}

static bool CheckQFeedsIntoQuantizedOutput(const NodeUnit& node_unit,
                                           const std::unordered_map<std::string, std::string> graph_op_data_type) {
  auto op_of_quantized_layer = node_unit.Outputs();
  for (auto& itr : op_of_quantized_layer) {
    auto it = graph_op_data_type.find(itr.node_arg.Name());
    if (it != graph_op_data_type.end() && it->second == "tensor(uint8)") {
      return true;
    }
  }
  return false;
}

static bool CheckQRuleSet(const NodeUnit& node_unit,
                          const Node* q_node,
                          const onnxruntime::GraphViewer& src_graph,
                          SkipReason& reason) {
  // If the target node of the NodeUnit following this one is one of the supported Op types, then keep this Q
  // This Q should also be uint8

  const auto& target_node = node_unit.GetNode();
  const auto& op_type = node_unit.OpType();

  auto op = src_graph.GetOutputs();
  std::unordered_map<std::string, std::string> graph_op_data_type;
  for (auto& ops : op) {
    graph_op_data_type[src_graph.GetNodeArg(ops->Name())->Name()] = ops->Type()->data();
  }

  // If UInt16 Q, don't keep it
  if (GetQDQDataType(q_node) == DT_UINT16 || GetQDQDataType(q_node) == DT_INT16) {
    reason = SkipReason::Int16QDQ;
    return false;
  }

  if (op_type == "Conv" || op_type == "MatMul") {
    // Conv and MatMul keep all Qs except if the target that succeeds it is Add/Mul/Div AND has any const init
    return IsNextTargetNodeOfQValid(q_node, &target_node, src_graph, {"Add", "Mul", "Div"}, true);
  } else if (op_type == "Add") {
    // Add keeps all Qs
    return true;
  } else if (CheckQFeedsIntoQuantizedOutput(node_unit, std::move(graph_op_data_type))) {
    return true;
  } else {
    // Keep Q of an unsupported Op only if the target that succeeds it is a supported Op in this list
    return IsNextTargetNodeOfQValid(q_node, &target_node, src_graph, {"Conv", "Add", "MatMul"}, false);
  }
}

static bool HandleDoubleQDQ(onnxruntime::Graph& dst_graph, const onnxruntime::GraphViewer& src_graph,
                            const NodeUnit& node_unit, std::set<std::string>& initializers_to_keep) {
  int node_unit_input_edge_count = node_unit.InputEdgeCount();
  int node_unit_output_edge_count = [&]() {
    int count = 0;
    for (auto it = node_unit.OutputEdgesBegin(); it != node_unit.OutputEdgesEnd(); ++it)
      count += 1;
    return count;
  }();
  bool edges_exist = node_unit_input_edge_count && node_unit_output_edge_count;

  // Detect a conversion between quantized types (e.g., int16 to int8)
  // in mixed-precision QDQ models. The pattern a standalone DQ followed by a standalone Q
  // Keep this standalone Q in the converting pair if it's not int8->int16
  if (node_unit.OpType() == "QuantizeLinear" && edges_exist) {
    const Node& q_node = node_unit.GetNode();
    const Node& i_dq_node = *q_node.InputNodesBegin();
    const Node& o_dq_node = *q_node.OutputNodesBegin();

    if (i_dq_node.OpType() == "DequantizeLinear" && o_dq_node.OpType() == "DequantizeLinear") {
      auto q_zero_point_dt = GetQDQDataType(&q_node);

      // Can ignore if this Q is uint16 as it won't be consumed by any supported node
      if (q_zero_point_dt != DT_UINT16 && q_zero_point_dt != DT_INT16 &&
          !IsQDQSandwichedBetweenSoftmaxAndConvMatMulOps(&q_node)) {
        // if it's unequal, then it's a conversion between quantized types.
        AddNode(initializers_to_keep, src_graph, dst_graph, node_unit.GetNode());
        return true;
      }
    }
  }

  // Keep this standalone DQ in the converting pair if it's not int16->int8
  if (node_unit.OpType() == "DequantizeLinear" && edges_exist) {
    const Node& dq_node = node_unit.GetNode();
    const Node& i_q_node = *dq_node.InputNodesBegin();
    const Node& o_q_node = *dq_node.OutputNodesBegin();

    if (i_q_node.OpType() == "QuantizeLinear" && o_q_node.OpType() == "QuantizeLinear") {
      auto dq_zero_point_dt = GetQDQDataType(&dq_node);

      if (dq_zero_point_dt != DT_UINT16 && dq_zero_point_dt != DT_INT16 &&
          IsConnectedQPresent(src_graph, dst_graph.Nodes(), &dq_node, dq_node.InputDefs()) &&
          !IsQDQSandwichedBetweenSoftmaxAndConvMatMulOps(&dq_node)) {
        AddNode(initializers_to_keep, src_graph, dst_graph, node_unit.GetNode());
        return true;
      }
    }
  }
  return false;
}
// Handles adding a standalone node unit (i.e., one not wrapped with DQ/Q ops) to the dst graph.
static void AddStandaloneNodeUnit(onnxruntime::Graph& dst_graph, const onnxruntime::GraphViewer& src_graph,
                                  const NodeUnit& node_unit,
                                  std::set<std::string>& initializers_to_keep,
                                  const logging::Logger& /* logger */) {
  assert(node_unit.UnitType() == NodeUnit::Type::SingleNode);

  if (HandleDoubleQDQ(dst_graph, src_graph, node_unit, initializers_to_keep)) return;

  auto add_identity_op = [&](bool duplicate_dq) {
    std::array<NodeArg*, 1> input_args, output_args;

    // Case to handle standalone duplicate DQs. Just redirect this arg to the original DQ instead and change the
    // arg type to FLOAT as we're replacing it with Identity
    if (duplicate_dq &&
        GetQDQDataType(&node_unit.GetNode()) != DT_UINT16 && GetQDQDataType(&node_unit.GetNode()) != DT_INT16) {
      std::string orig_dq_name = node_unit.Outputs()[0].node_arg.Name();  // ex: dql_output/duplicated
      std::unique_ptr<ONNX_NAMESPACE::TypeProto> type_proto = ONNX_NAMESPACE::TypeProto::Create();
      type_proto->copy_from(node_unit.Inputs()[0].node_arg.TypeAsProto());
      type_proto->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
      orig_dq_name.erase(orig_dq_name.find(DuplicateDQ), std::string::npos);  // ex: dql_output
      input_args = {&dst_graph.GetOrCreateNodeArg(orig_dq_name, type_proto.get())};
    } else {
      input_args = {&ProcessNodeUnitIO(dst_graph, src_graph, initializers_to_keep,
                                       node_unit.Inputs()[0])};
    }
    output_args = {&ProcessNodeUnitIO(dst_graph, src_graph, initializers_to_keep,
                                      node_unit.Outputs()[0])};
    dst_graph.AddNode(node_unit.Name(),
                      "Identity",
                      "",
                      input_args,
                      output_args,
                      nullptr,
                      kOnnxDomain);
  };

  if (node_unit.OpType() == "QuantizeLinear") {
    SkipReason reason;
    // keep if next target is supported
    if (CheckQRuleSet(node_unit, &node_unit.GetNode(), src_graph, reason))
      AddNode(initializers_to_keep, src_graph, dst_graph, node_unit.GetNode());
    // #2 If input 0 is a constant initializer, then don't keep the Q
    else if (src_graph.IsConstantInitializer(node_unit.GetNode().InputDefs().at(0)->Name(), true))
      return;
    else
      add_identity_op(false);
  } else if (node_unit.OpType() == "DequantizeLinear") {
    // keep if prev target is supported
    if (node_unit.GetNode().Name().find(DuplicateDQ) != std::string::npos)
      add_identity_op(true);
    else if (IsConnectedQPresent(src_graph, dst_graph.Nodes(), &node_unit.GetNode(), node_unit.GetNode().InputDefs()))
      AddNode(initializers_to_keep, src_graph, dst_graph, node_unit.GetNode());
    else if (DQFeedsASupportedOp(&node_unit.GetNode()))
      AddNode(initializers_to_keep, src_graph, dst_graph, node_unit.GetNode());
    else
      add_identity_op(false);
  } else {
    AddNode(initializers_to_keep, src_graph, dst_graph, node_unit.GetNode());
  }
}

// Handles adding a QDQ node unit (e.g., DQ -> Add -> Q) to the dst graph.
// Only adds the QDQ node unit's float-precision target node.
static void AddQDQNodeUnit(onnxruntime::Graph& dst_graph,
                           const onnxruntime::GraphViewer& src_graph,
                           const NodeUnit& node_unit,
                           std::set<std::string>& initializers_to_keep,
                           const logging::Logger& /* logger */) {
  assert(node_unit.UnitType() == NodeUnit::Type::QDQGroup);

  // Collect inputs coming into the node unit.
  const auto& node_unit_inputs = node_unit.Inputs();
  std::vector<NodeArg*> input_args;
  input_args.reserve(node_unit_inputs.size());

  // Handle DQs in the NodeUnit
  std::unordered_map<std::string, NodeArg*> dq_node_args_to_keep;  // These DQ nodes will be retained in the graph

  for (auto dq_node : node_unit.GetDQNodes()) {
    const auto& input_defs = dq_node->InputDefs();
    ORT_ENFORCE(input_defs.size() == 3);

    SkipReason reason = SkipReason::Other;
    bool keep_dq = CheckDQRuleSet(node_unit, dq_node, src_graph, reason);

    if (keep_dq) {
      AddNode(initializers_to_keep, src_graph, dst_graph, *dq_node);
      dq_node_args_to_keep.insert({input_defs.at(0)->Name(),
                                   &dst_graph.GetOrCreateNodeArg(dq_node->OutputDefs().at(0)->Name(),
                                                                 dq_node->OutputDefs().at(0)->TypeAsProto())});
    } else {
      // If it's a duplicate DQ AND the previous node unit is not of type SingleNode (i.e, input/output units)
      if (reason == SkipReason::DuplicateDQ) {
        // Skips the DQ, but keep the route to the target node
        // Add the output of the other original DQ as input arg of this target node

        if (!IsConnectedQPresent(src_graph, dst_graph.Nodes(), dq_node, input_defs)) continue;

        std::string target_arg_name = dq_node->OutputDefs().at(0)->Name();
        // erase from the first occurrence of the search string till the end of the target arg name
        target_arg_name.erase(target_arg_name.find(DuplicateDQ), std::string::npos);
        dq_node_args_to_keep.insert({input_defs.at(0)->Name(),
                                     &dst_graph.GetOrCreateNodeArg(target_arg_name,
                                                                   dq_node->OutputDefs().at(0)->TypeAsProto())});
      } else if (reason == SkipReason::SandwichedDQ) {
        dq_node_args_to_keep.clear();
        break;
      } else if (IsConnectedQAConstantInitializer(dq_node, src_graph)) {
        // Q (const input arg 0) -> DQ -> Supported Op
        // If the connected Q has a const init input, then the DQ should only have the Q as input
        ORT_ENFORCE(dq_node->GetInputEdgesCount() == 1);
        // Make the const init the input to the target node, as its Q, and DQ are not being kept
        dq_node_args_to_keep.insert(
            {input_defs.at(0)->Name(),
             &dst_graph.GetOrCreateNodeArg(dq_node->InputNodesBegin()->InputDefs().at(0)->Name(),
                                           dq_node->InputNodesBegin()->InputDefs().at(0)->TypeAsProto())});
        // Also keep the initializer in the graph
        if (src_graph.GetAllInitializedTensors().count(dq_node->InputNodesBegin()->InputDefs().at(0)->Name())) {
          initializers_to_keep.insert({dq_node->InputNodesBegin()->InputDefs().at(0)->Name()});
        }
      }
    }
  }

  // Add Node args for inputs
  for (const auto& node_unit_input : node_unit_inputs) {
    const auto& node_arg_name = node_unit_input.node_arg.Name();
    if (auto dq_node_arg = dq_node_args_to_keep.find(node_arg_name); dq_node_arg != dq_node_args_to_keep.end()) {
      // Add supported DQ as an input arg for the target node
      input_args.push_back(dq_node_arg->second);
    } else {
      // Otherwise, convert to float
      NodeArg& input_arg = ProcessNodeUnitIO(dst_graph, src_graph, initializers_to_keep, node_unit_input);
      input_args.push_back(&input_arg);
    }
  }

  const Node& target_node = node_unit.GetNode();

  // Collect outputs coming out of the node unit.
  const auto& node_unit_outputs = node_unit.Outputs();
  std::vector<NodeArg*> output_args;
  output_args.reserve(node_unit_outputs.size());

  // Handle Qs in the NodeUnit
  if (!node_unit.GetQNodes().empty()) {
    for (size_t i = 0; i < node_unit.GetQNodes().size(); i++) {
      const auto& q_node = node_unit.GetQNodes().at(i);

      SkipReason reason;

      bool keep_q = CheckQRuleSet(node_unit, q_node, src_graph, reason);

      if (keep_q) {
        AddNode(initializers_to_keep, src_graph, dst_graph, *q_node);
        // if keep_q, then output defs of the target node doesn't change
        output_args.push_back(&dst_graph.GetOrCreateNodeArg(target_node.OutputDefs().at(i)->Name(),
                                                            target_node.OutputDefs().at(i)->TypeAsProto()));
      } else {
        // convert this Q to float
        output_args.push_back(&ProcessNodeUnitIO(dst_graph, src_graph, initializers_to_keep,
                                                 node_unit_outputs.at(i)));
      }
    }
  } else {
    for (const auto& node_unit_output : node_unit_outputs) {
      // convert non-qdq outputs to float
      NodeArg& output_arg = ProcessNodeUnitIO(dst_graph, src_graph, initializers_to_keep,
                                              node_unit_output);
      output_args.push_back(&output_arg);
    }
  }

  // Add the target node in the node unit to the graph.
  dst_graph.AddNode(target_node.Name(),
                    target_node.OpType(),
                    target_node.Description(),
                    input_args,
                    output_args,
                    &target_node.GetAttributes(),
                    target_node.Domain());
  KeepInitsInDstGraph(initializers_to_keep, src_graph, &target_node);
}

// Creates a new model without the DQ/Q operators in the src graph.
Status CreateModelWithStrippedQDQNodes(const GraphViewer& src_graph,
                                       const logging::Logger& logger,
                                       /*out*/ std::unique_ptr<onnxruntime::Model>& model) {
  // NOTE: This function is a re-implementation of GraphViewerToProto() in core/graph/graph_proto_serializer.cc
  // with the following differences:
  //   - Uses onnxruntime::Graph APIs instead of onnx::GraphProto APIs.
  //   - Traverses the src graph using QDQ node units.
  //   - Filters out DQ/Q ops that wrap full-precision nodes.
  //   - Dequantizes quantized initializers.

  // Constructs model from scratch using the metadata in src_graph
  model = src_graph.CreateModel(logger);

  //
  // Initialize model/graph metadata.
  //

  auto& dst_graph = model->MainGraph();

  // Set inputs outputs explicitly to make sure the order is same as the user model.
  auto inputs = src_graph.GetInputs();
  auto outputs = src_graph.GetOutputs();

  InlinedVector<const NodeArg*> dst_graph_inputs;
  dst_graph_inputs.reserve(inputs.size());
  for (auto& input : inputs) {
    auto input_arg = src_graph.GetNodeArg(input->Name());
    auto& ep_graph_input_arg = dst_graph.GetOrCreateNodeArg(input_arg->Name(), input_arg->TypeAsProto());
    dst_graph_inputs.push_back(&ep_graph_input_arg);
  }

  InlinedVector<const NodeArg*> dst_graph_outputs;
  dst_graph_outputs.reserve(outputs.size());
  for (auto& output : outputs) {
    auto output_arg = src_graph.GetNodeArg(output->Name());
    auto& ep_graph_output_arg = dst_graph.GetOrCreateNodeArg(output_arg->Name(), output_arg->TypeAsProto());
    dst_graph_outputs.push_back(&ep_graph_output_arg);
  }

  dst_graph.SetInputs(dst_graph_inputs);
  dst_graph.SetOutputs(dst_graph_outputs);

  // TODO(sspintel): add Graph::SetName() provider api
  // dst_graph.SetName(src_graph.Name());

  // TODO(sspintel): add Graph::SetDescription() and GraphViewer::Description() to provider api
  // dst_graph.SetDescription(src_graph.Description());

  // Mark outer scope NodeArgs
  for (const auto& name : src_graph.GetOuterScopeNodeArgNames()) {
    auto* node_arg = src_graph.GetNodeArg(name);
    ORT_RETURN_IF_NOT(node_arg != nullptr, "Outer scope node arg name '" + name + "'was added but does not exist. ");
    dst_graph.AddOuterScopeNodeArg(name);
  }

  //
  // Add nodes (without their DQ/Q ops) to dst graph.
  //

  // Keep track of all the initializers we need to dequantize to float.
  std::set<std::string> initializers_to_keep{};

  // Get all the NodeUnits in the graph_viewer
  std::vector<std::unique_ptr<NodeUnit>> node_unit_holder;
  std::unordered_map<const Node*, const NodeUnit*> node_unit_map;
  std::tie(node_unit_holder, node_unit_map) = QDQ::GetAllNodeUnits(&src_graph, logger);

  std::unordered_set<const NodeUnit*> seen_node_units;
  const auto& node_indices = src_graph.GetNodesInTopologicalOrder();

  // Process node units in topological order. Filter out Q/DQ ops.
  for (size_t i = 0; i < node_indices.size(); i++) {
    gsl::not_null<const onnxruntime::Node*> node(src_graph.GetNode(node_indices[i]));

    // Get the node_unit associated with the node.
    gsl::not_null<const NodeUnit*> node_unit = node_unit_map.at(node);

    // Visiting 'nodes' in topological order does not guarantee that 'node_units' are
    // also visited in topological order. Skip this node if it is not the node_unit's target node
    // to ensure actual 'node_units' are visited in topological order.
    if (node != &node_unit->GetNode()) {
      continue;
    }

    if (seen_node_units.count(node_unit) != 0) {
      continue;  // Already handled this node unit
    }

    if (node_unit->UnitType() == NodeUnit::Type::SingleNode) {
      AddStandaloneNodeUnit(dst_graph, src_graph, *node_unit, initializers_to_keep, logger);
    } else {
      AddQDQNodeUnit(dst_graph, src_graph, *node_unit, initializers_to_keep, logger);
    }

    seen_node_units.insert(node_unit);
  }

  //
  // Copy initializers to dst graph.
  //

  std::unordered_set<std::string> current_scope_initializer_set;

  auto& initializers = src_graph.GetAllInitializedTensors();

  // Sort initializers to maintain consistency in model proto created across inference requests
  std::vector<std::string> const_inits;
  for (auto& it : initializers) {
    const_inits.push_back(it.first);
  }
  std::sort(const_inits.begin(), const_inits.end());

  for (auto& it : const_inits) {
    if (initializers_to_keep.count(it))
      dst_graph.AddInitializedTensor(*(initializers.at(it)));
    current_scope_initializer_set.insert(it);
  }

  // handle outer scope value which is a constant initializer
  for (auto& node_idx : src_graph.GetNodesInTopologicalOrder()) {
    const auto& node = src_graph.GetNode(node_idx);
    for (const auto& input : node->InputDefs()) {
      if (current_scope_initializer_set.find(input->Name()) != current_scope_initializer_set.end()) {
        continue;
      }
      if (src_graph.IsConstantInitializer(input->Name(), true)) {
        if (initializers_to_keep.count(input->Name()))
          dst_graph.AddInitializedTensor(*(src_graph.GetConstantInitializer(input->Name(), true)));
        current_scope_initializer_set.insert(input->Name());
      }
    }
  }

  // Validate graph, remove unnecessary initializers, and run type/shape inference.
  ORT_RETURN_IF_ERROR(dst_graph.Resolve());

  return Status::OK();
}
}  // namespace openvino_ep
}  // namespace onnxruntime
