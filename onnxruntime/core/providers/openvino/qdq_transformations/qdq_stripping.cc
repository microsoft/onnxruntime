// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <array>
#include <vector>
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/openvino/qdq_transformations/qdq_stripping.h"

namespace onnxruntime {
namespace openvino_ep {

enum class SkipReason {
  Uint16QDQ,
  DuplicateDQ,
  ConstInitUnsupportedDQ,
  SandwichedDQ,
  Other
};

static ONNX_NAMESPACE::TensorProto_DataType GetZeroPointDT(const Node* qdq_node) {
  return static_cast<ONNX_NAMESPACE::TensorProto_DataType>(
      qdq_node->InputDefs().at(2)->TypeAsProto()->tensor_type().elem_type());
}
// Creates a new NodeArg from an input (or output) of a QDQ node unit. If the input/output is quantized,
// this function modifies the tensor type to the specified float type.
static NodeArg& ProcessNodeUnitIO(onnxruntime::Graph& dst_graph, const onnxruntime::GraphViewer& src_graph,
                                  std::unordered_map<std::string, NodeUnitIODef>& initializers_to_dequant,
                                  const NodeUnitIODef& io_def, int float_type) {
  const std::string& name = io_def.node_arg.Name();
  const ONNX_NAMESPACE::TypeProto* orig_type_proto = io_def.node_arg.TypeAsProto();

  // Handle quantized input or output. Convert to float type.
  if (io_def.quant_param.has_value()) {
    // Copy the original quantized type proto, but update the type to float.
    std::unique_ptr<ONNX_NAMESPACE::TypeProto> type_proto = ONNX_NAMESPACE::TypeProto::Create();
    type_proto->copy_from(orig_type_proto);
    type_proto->mutable_tensor_type()->set_elem_type(float_type);

    // Handle initializer inputs.
    // By default QDQ models store quantized weights that are dequantized.
    // Ex: weight(int8) -> DequantizeLinear (to float) ->
    // Keep track of these initializers so the EP can dequantize them later.
    if (src_graph.GetAllInitializedTensors().count(name)) {
      initializers_to_dequant.insert({name, io_def});
    }

    return dst_graph.GetOrCreateNodeArg(name, type_proto.get());
  }

  // Unquantized input or output. Just copy.
  return dst_graph.GetOrCreateNodeArg(name, orig_type_proto);
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

static bool IsAnyDQBias(const Node* target_node) {
  bool is_bias = false;
  for (Node::NodeConstIterator it_dq = target_node->InputNodesBegin(); it_dq != target_node->InputNodesEnd(); ++it_dq) {
    const auto& DQ = &*it_dq;
    if (DQ->OpType() != "DequantizeLinear") continue;
    is_bias |= DQ->InputDefs().at(0)->Name().find("bias") != std::string::npos;
  }

  return is_bias;
}

static bool IsFirstComputeOpBelowConvMatMulNonBiasAdd(const Node* qdq_node) {
  if (qdq_node->OpType() != "QuantizeLinear" && qdq_node->OpType() != "DequantizeLinear") {
    if (qdq_node->OpType() == "Conv" || qdq_node->OpType() == "MatMul")
      return true;
    else if (qdq_node->OpType() == "Add")
      return !IsAnyDQBias(qdq_node);
    else
      return false;
  } else {
    if (qdq_node->GetOutputEdgesCount())
      return IsFirstComputeOpBelowConvMatMulNonBiasAdd(&*qdq_node->OutputNodesBegin());
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

// Used to find if inputs of the target node DQ's are constant initializers
static bool IsAnyDQAConstantInitializer(const Node* target_node, const onnxruntime::GraphViewer& src_graph) {
  bool is_const_init = false;
  for (Node::NodeConstIterator it_dq = target_node->InputNodesBegin(); it_dq != target_node->InputNodesEnd(); ++it_dq) {
    const auto& DQ = &*it_dq;
    if (DQ->OpType() != "DequantizeLinear") continue;
    is_const_init |= src_graph.IsConstantInitializer(DQ->InputDefs().at(0)->Name(), true);
  }

  return is_const_init;
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
    // #2 If any DQ of the previous Conv/MatMul is sandwiched between Softmax and Conv/MatMul,
    // then don't keep this DQ
    for (Node::NodeConstIterator it_sw = prev_target->InputNodesBegin();
         it_sw != prev_target->InputNodesEnd(); ++it_sw) {
      const auto& sw_dq_node = &*it_sw;
      if (sw_dq_node->OpType() != "DequantizeLinear") return true;
      if (IsQDQSandwichedBetweenSoftmaxAndConvMatMulOps(sw_dq_node))
        return false;
    }
    // #3 For Mul/Div, the DQ shouldn't be a const init
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
    if (check_const_init && IsAnyDQAConstantInitializer(next_target, src_graph)) {
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
  auto op_type = node_unit.OpType();

  // #1 Reverse DQ duplication
  if (dq_node->Name().find("/duplicated") != std::string::npos) {
    reason = SkipReason::DuplicateDQ;
    return false;
  }

  // #2 If it's a constant initializer feeding to even unsupported ops with an unsupported type, keep it
  // TODO(sspintel): check if this needs to be done only for certain supported Ops
  if (src_graph.IsConstantInitializer(dq_node->InputDefs().at(0)->Name(), true)) {
    return true;
  }

  // #3 If UInt16 DQ, don't keep it
  if (GetZeroPointDT(dq_node) == ONNX_NAMESPACE::TensorProto_DataType_UINT16) {
    reason = SkipReason::Uint16QDQ;
    return false;
  }

  // DQs in Double QDQ cases should be kept
  if (dq_node->InputDefs().at(2)->Name().find("zero_point_convert") != std::string::npos &&
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
    // Add keeps all DQs except if it is a BiasAdd
    return !IsAnyDQBias(&target_node);
  } else if (op_type == "Mul" || op_type == "Div") {
    // Keep DQ of Mul and Div only if the target that preceds it is a supported Op in this list and also check if
    // inputs of Mul and Div have constant initializers. If they do, then don't keep the DQ.
    return IsPreviousTargetNodeOfDQValid(dq_node, &target_node, src_graph, true);
  } else {
    // Keep DQ of an unsupported Op only if the target that preceds it is a supported Op in this list
    return IsPreviousTargetNodeOfDQValid(dq_node, &target_node, src_graph, false);
  }
}

static bool CheckQRuleSet(const NodeUnit& node_unit,
                          const Node* q_node,
                          const onnxruntime::GraphViewer& src_graph,
                          SkipReason& reason) {
  // If the target node of the NodeUnit following this one is one of the supported Op types, then keep this Q
  // This Q should also be uint8

  const auto& target_node = node_unit.GetNode();
  auto op_type = node_unit.OpType();

  // If UInt16 Q, don't keep it
  if (GetZeroPointDT(q_node) == ONNX_NAMESPACE::TensorProto_DataType_UINT16) {
    reason = SkipReason::Uint16QDQ;
    return false;
  }

  if (op_type == "Conv" || op_type == "MatMul") {
    // If any DQ of this Conv/MatMul is sandwiched between Softmax and Conv/MatMul, then don't keep it
    for (const auto& dq_node : node_unit.GetDQNodes()) {
      if (IsQDQSandwichedBetweenSoftmaxAndConvMatMulOps(dq_node)) {
        reason = SkipReason::SandwichedDQ;
        return false;
      }
    }
    // Conv and MatMul keep all Qs except if the target that succeeds it is Add/Mul/Div AND has any const init
    return IsNextTargetNodeOfQValid(q_node, &target_node, src_graph, {"Add", "Mul", "Div"}, true);
  } else if (op_type == "Add") {
    // Add keeps all Qs
    return true;
  } else {
    // Keep Q of an unsupported Op only if the target that succeeds it is a supported Op in this list
    return IsNextTargetNodeOfQValid(q_node, &target_node, src_graph, {"Conv", "Add", "MatMul"}, false);
  }
}

static bool HandleDoubleQDQ(onnxruntime::Graph& dst_graph, const onnxruntime::GraphViewer& src_graph,
                            const NodeUnit& node_unit) {
  int node_unit_input_edge_count = node_unit.InputEdgeCount();
  int node_unit_output_edge_count = [&]() {
    int count = 0;
    for (auto it = node_unit.OutputEdgesBegin(); it != node_unit.OutputEdgesEnd(); ++it)
      count += 1;
    return count;
  }();
  bool input_edges_exist = node_unit_input_edge_count && node_unit_output_edge_count;

  // Detect a conversion between quantized types (e.g., int16 to int8)
  // in mixed-precision QDQ models. The pattern a standalone DQ followed by a standalone Q:
  // Ex: DQ(int16 to float) -> QuantizeLinear(float to int8) ->
  if (node_unit.OpType() == "QuantizeLinear" && input_edges_exist) {
    const Node& q_node = node_unit.GetNode();
    const Node& i_dq_node = *q_node.InputNodesBegin();
    const Node& o_dq_node = *q_node.OutputNodesBegin();

    if (i_dq_node.OpType() == "DequantizeLinear" && o_dq_node.OpType() == "DequantizeLinear") {
      auto q_zero_point_dt = GetZeroPointDT(&q_node);

      // Can ignore if this Q is uint16 as it won't be consumed by any supported node
      if (q_zero_point_dt != ONNX_NAMESPACE::TensorProto_DataType_UINT16 &&
          !IsQDQSandwichedBetweenSoftmaxAndConvMatMulOps(&q_node)) {
        // if it's unequal, then it's a conversion between quantized types.
        dst_graph.AddNode(node_unit.GetNode());
        return true;
      }
    }
  }

  // Keep int8 DQ/Qs in int16 -> int8
  // Don't keep any QDQs in int8 -> int16
  // Beginning of a converting QDQ pair. Check if previous Q is int8 and the following Q has a converting zp
  if (node_unit.OpType() == "DequantizeLinear" && input_edges_exist) {
    const Node& dq_node = node_unit.GetNode();
    const Node& i_q_node = *dq_node.InputNodesBegin();
    const Node& o_q_node = *dq_node.OutputNodesBegin();

    if (i_q_node.OpType() == "QuantizeLinear" && o_q_node.OpType() == "QuantizeLinear") {
      auto dq_zero_point_dt = GetZeroPointDT(&dq_node);

      if (dq_zero_point_dt != ONNX_NAMESPACE::TensorProto_DataType_UINT16 &&
          IsConnectedQPresent(src_graph, dst_graph.Nodes(), &dq_node, dq_node.InputDefs()) &&
          !IsQDQSandwichedBetweenSoftmaxAndConvMatMulOps(&dq_node)) {
        dst_graph.AddNode(node_unit.GetNode());
        return true;
      }
    }
  }
  return false;
}
// Handles adding a standalone node unit (i.e., one not wrapped with DQ/Q ops) to the dst graph.
static void AddStandaloneNodeUnit(onnxruntime::Graph& dst_graph, const onnxruntime::GraphViewer& src_graph,
                                  const NodeUnit& node_unit, int32_t float_type,
                                  std::unordered_map<std::string, NodeUnitIODef>& initializers_to_dequant,
                                  const logging::Logger& /* logger */) {
  assert(node_unit.UnitType() == NodeUnit::Type::SingleNode);

  if (HandleDoubleQDQ(dst_graph, src_graph, node_unit)) return;

  auto add_identity_op = [&]() {
    std::array<NodeArg*, 1> input_args = {&ProcessNodeUnitIO(dst_graph, src_graph, initializers_to_dequant,
                                                             node_unit.Inputs()[0], float_type)};
    std::array<NodeArg*, 1> output_args = {&ProcessNodeUnitIO(dst_graph, src_graph, initializers_to_dequant,
                                                              node_unit.Outputs()[0], float_type)};
    dst_graph.AddNode(node_unit.Name(),
                      "Identity",
                      "",
                      input_args,
                      output_args,
                      nullptr,
                      kOnnxDomain);
  };

  if (node_unit.OpType() == "QuantizeLinear") {
    add_identity_op();
  } else if (node_unit.OpType() == "DequantizeLinear") {
    if (IsConnectedQPresent(src_graph, dst_graph.Nodes(), &node_unit.GetNode(), node_unit.GetNode().InputDefs()))
      dst_graph.AddNode(node_unit.GetNode());  // Copy standalone node unit.
    else
      add_identity_op();
  } else {
    dst_graph.AddNode(node_unit.GetNode());  // Copy standalone node unit.
  }
}

// Handles adding a QDQ node unit (e.g., DQ -> Add -> Q) to the dst graph.
// Only adds the QDQ node unit's float-precision target node.
static void AddQDQNodeUnit(onnxruntime::Graph& dst_graph,
                           const onnxruntime::GraphViewer& src_graph,
                           const NodeUnit& node_unit,
                           int32_t float_type,
                           std::unordered_map<std::string, NodeUnitIODef>& initializers_to_dequant,
                           const logging::Logger& /* logger */) {
  assert(node_unit.UnitType() == NodeUnit::Type::QDQGroup);

  // LOGS_DEFAULT(INFO) << "\nNode in QDQ Group: " << node_unit.Name() << " " << node_unit.OpType() << " ";

  // Collect inputs coming into the node unit.
  const auto& node_unit_inputs = node_unit.Inputs();
  std::vector<NodeArg*> input_args;
  input_args.reserve(node_unit_inputs.size());

  // Handle DQs in the NodeUnit
  std::unordered_map<std::string, NodeArg*> dq_node_args_to_keep;  // These DQ nodes will be retained in the graph

  for (auto dq_node : node_unit.GetDQNodes()) {
    const auto& input_defs = dq_node->InputDefs();
    ORT_ENFORCE(input_defs.size() == 3);

    SkipReason reason;
    bool keep_dq = CheckDQRuleSet(node_unit, dq_node, src_graph, reason);
    LOGS_DEFAULT(INFO) << "!!!!!!!! kept DQ " << dq_node->Name() << keep_dq;

    if (keep_dq) {
      dst_graph.AddNode(*dq_node);  // Add the node to the graph
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
        std::string duplicate_str = "/duplicated";
        // erase from the first occurence of the search string till the end of the target arg name
        target_arg_name.erase(target_arg_name.find(duplicate_str), std::string::npos);
        dq_node_args_to_keep.insert({input_defs.at(0)->Name(),
                                     &dst_graph.GetOrCreateNodeArg(target_arg_name,
                                                                   dq_node->OutputDefs().at(0)->TypeAsProto())});
      } else if (reason == SkipReason::SandwichedDQ) {
        dq_node_args_to_keep.clear();
        break;
      }
    }
  }

  // Add Node args for inputs
  for (const auto& node_unit_input : node_unit_inputs) {
    auto node_arg_name = node_unit_input.node_arg.Name();
    if (auto dq_node_arg = dq_node_args_to_keep.find(node_arg_name); dq_node_arg != dq_node_args_to_keep.end()) {
      // Add supported DQ as an input arg for the target node
      input_args.push_back(dq_node_arg->second);
    } else {
      // Otherwise, convert to float
      NodeArg& input_arg = ProcessNodeUnitIO(dst_graph, src_graph, initializers_to_dequant,
                                             node_unit_input, float_type);
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
    ORT_ENFORCE(node_unit.GetQNodes().size() == 1);
    const auto& q_node = node_unit.GetQNodes().at(0);

    SkipReason reason;

    bool keep_q = CheckQRuleSet(node_unit, q_node, src_graph, reason);

    LOGS_DEFAULT(INFO) << "!!!!!!!! kept Q " << q_node->Name() << keep_q;

    if (keep_q) {
      dst_graph.AddNode(*q_node);  // Add the node to the graph
      // if keep_q, then output defs of the target node doesn't change
      output_args.push_back(&dst_graph.GetOrCreateNodeArg(target_node.OutputDefs().at(0)->Name(),
                                                          target_node.OutputDefs().at(0)->TypeAsProto()));
    } else {
      // convert this Q to float
      output_args.push_back(&ProcessNodeUnitIO(dst_graph, src_graph, initializers_to_dequant,
                                               node_unit_outputs.at(0), float_type));
    }
  } else {
    for (const auto& node_unit_output : node_unit_outputs) {
      // convert non-qdq outputs to float
      NodeArg& output_arg = ProcessNodeUnitIO(dst_graph, src_graph, initializers_to_dequant,
                                              node_unit_output, float_type);
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
}

// Creates a new model without the DQ/Q operators in the src graph.
Status CreateModelWithStrippedQDQNodes(const GraphViewer& src_graph,
                                       const logging::Logger& logger,
                                       int32_t float_type,
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

  // TODO: add Model::SetModelVersion() to provider api
  // model->SetModelVersion(ONNX_NAMESPACE::Version::IR_VERSION);

  auto& dst_graph = model->MainGraph();

  // TODO: add Graph::SetName() provider api
  // dst_graph.SetName(src_graph.Name());

  // TODO: add Graph::SetDescription() and GraphViewer::Description() to provider api
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
  std::unordered_map<std::string, NodeUnitIODef> initializers_to_dequant;

  // Get all the NodeUnits in the graph_viewer
  std::vector<std::unique_ptr<NodeUnit>> node_unit_holder;
  std::unordered_map<const Node*, const NodeUnit*> node_unit_map;
  std::tie(node_unit_holder, node_unit_map) = QDQ::GetAllNodeUnits(&src_graph);

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
      AddStandaloneNodeUnit(dst_graph, src_graph, *node_unit, float_type, initializers_to_dequant, logger);
    } else {
      AddQDQNodeUnit(dst_graph, src_graph, *node_unit, float_type, initializers_to_dequant, logger);
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
