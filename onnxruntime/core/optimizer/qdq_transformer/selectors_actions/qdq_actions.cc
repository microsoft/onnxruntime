// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/selectors_actions/qdq_actions.h"

#include "core/optimizer/qdq_transformer/qdq_util.h"
#include "core/graph/node_attr_utils.h"
namespace onnxruntime {
namespace QDQ {

namespace {
using NTO = NodesToOptimize;

// moves for replacing a node with a single DQ input with the qlinear version
std::vector<NodeAndMoveInfo> UnaryMoves() {
  NTO::NodeLocation dq{NTO::NodeType::kInput, 0};
  NTO::NodeLocation q{NTO::NodeType::kOutput, 0};

  std::vector<NodeAndMoveInfo> moves{
      MoveAll(dq, ArgType::kInput),                           // append all inputs from dq to new node
      MoveAndAppend(q, ArgType::kInput, 1, ArgType::kInput),  // append scale (input 1) from q
      MoveAndAppend(q, ArgType::kInput, 2, ArgType::kInput),  // append zp (input 2) from q
      MoveAll(q, ArgType::kOutput)};

  return moves;
}

// moves for replacing a node with two DQ inputs with the qlinear version
std::vector<NodeAndMoveInfo> BinaryMoves() {
  NTO::NodeLocation dq1{NTO::NodeType::kInput, 0};
  NTO::NodeLocation dq2{NTO::NodeType::kInput, 1};
  NTO::NodeLocation q{NTO::NodeType::kOutput, 0};

  std::vector<NodeAndMoveInfo> moves{
      MoveAll(dq1, ArgType::kInput),                          // append all inputs from dq1 to new node
      MoveAll(dq2, ArgType::kInput),                          // append all inputs from dq2
      MoveAndAppend(q, ArgType::kInput, 1, ArgType::kInput),  // append scale (input 1) from q
      MoveAndAppend(q, ArgType::kInput, 2, ArgType::kInput),  // append zp (input 2) from q
      MoveAll(q, ArgType::kOutput)};                          // and use the outputs from q

  return moves;
}

// moves for replacing a node with a single variadic DQ input with the qlinear version
std::vector<NodeAndMoveInfo> VariadicMoves() {
  NTO::NodeLocation variadic_dq{NTO::NodeType::kInput, 0};
  NTO::NodeLocation q{NTO::NodeType::kOutput, 0};

  std::vector<NodeAndMoveInfo> moves{
      MoveAndAppend(q, ArgType::kInput, 1, ArgType::kInput),  // append scale (input 1) from q
      MoveAndAppend(q, ArgType::kInput, 2, ArgType::kInput),  // append zp (input 2) from q
      MoveAll(variadic_dq, ArgType::kInput),                  // append all inputs from all dq nodes
      MoveAll(q, ArgType::kOutput)};                          // and use the outputs from q

  return moves;
}

// moves for replacing a node with a Conv node with DQ inputs with the qlinear version
std::vector<NodeAndMoveInfo> ConvMoves() {
  NTO::NodeLocation dq_x{NTO::NodeType::kInput, 0};
  NTO::NodeLocation dq_w{NTO::NodeType::kInput, 1};
  NTO::NodeLocation dq_bias{NTO::NodeType::kInput, 2};
  NTO::NodeLocation q{NTO::NodeType::kOutput, 0};

  std::vector<NodeAndMoveInfo> moves{
      MoveAll(dq_x, ArgType::kInput),                                     // append all inputs from x
      MoveAll(dq_w, ArgType::kInput),                                     // append all inputs from w
      MoveAndAppend(q, ArgType::kInput, 1, ArgType::kInput),              // append scale (input 1) from q
      MoveAndAppend(q, ArgType::kInput, 2, ArgType::kInput),              // append zp (input 2) from q
      MoveAndAppend(dq_bias, ArgType::kInput, 0, ArgType::kInput, true),  // (optional) append bias
      MoveAll(q, ArgType::kOutput)};                                      // and use the outputs from q

  return moves;
}
std::vector<NodeAndMoveInfo> WhereMoves() {
  NTO::NodeLocation dq_x{NTO::NodeType::kInput, 0};
  NTO::NodeLocation dq_y{NTO::NodeType::kInput, 1};
  NTO::NodeLocation target{NTO::NodeType::kTarget, 0};
  NTO::NodeLocation q{NTO::NodeType::kOutput, 0};

  std::vector<NodeAndMoveInfo> moves{
      MoveAndAppend(target, ArgType::kInput, 0, ArgType::kInput),  // move the condition to the new node
      MoveAll(dq_x, ArgType::kInput),                              // append all inputs from x
      MoveAll(dq_y, ArgType::kInput),                              // append all inputs from x
      MoveAndAppend(q, ArgType::kInput, 1, ArgType::kInput),       // append scale (input 1) from q
      MoveAndAppend(q, ArgType::kInput, 2, ArgType::kInput),       // append zp (input 2) from q
      MoveAll(q, ArgType::kOutput)};
  return moves;
}
QDQReplaceWithNew SplitReplacer(bool has_split_as_input) {
  NTO::NodeLocation dq{NTO::NodeType::kInput, 0};
  NTO::NodeLocation target{NTO::NodeType::kTarget, 0};
  NTO::NodeLocation q{NTO::NodeType::kOutput, 0};
  std::vector<NodeAndMoveInfo> moves{MoveAndAppend(dq, ArgType::kInput, 0, ArgType::kInput)};

  if (has_split_as_input) {
    // Move the optional split input to the new node.
    moves.push_back(MoveAndAppend(target, ArgType::kInput, 1, ArgType::kInput, true));
  }

  moves.push_back(MoveAll(q, ArgType::kOutput));

  return QDQReplaceWithNew(kOnnxDomain, "Split", std::move(moves));
}

QDQReplaceWithNew MatMulIntToFloatReplacer() {
  NTO::NodeLocation dq1{NTO::NodeType::kInput, 0};
  NTO::NodeLocation dq2{NTO::NodeType::kInput, 1};
  NTO::NodeLocation target{NTO::NodeType::kTarget, 0};

  std::vector<NodeAndMoveInfo> moves{
      MoveAndAppend(dq1, ArgType::kInput, 0, ArgType::kInput),
      MoveAndAppend(dq2, ArgType::kInput, 0, ArgType::kInput),
      MoveAndAppend(dq1, ArgType::kInput, 1, ArgType::kInput),
      MoveAndAppend(dq2, ArgType::kInput, 1, ArgType::kInput),
      MoveAndAppend(dq1, ArgType::kInput, 2, ArgType::kInput),
      MoveAndAppend(dq2, ArgType::kInput, 2, ArgType::kInput),
      MoveAll(target, ArgType::kOutput)};

  return QDQReplaceWithNew(kMSDomain, "MatMulIntegerToFloat", std::move(moves));
}

struct SetOptionalZeroPoint {
  static void UpdateNodes(Graph&, const NodesToOptimize& selected_nodes);

 private:
  // We assume this function won't fail
  static const ONNX_NAMESPACE::TensorProto init_optional_zero_point_int8() {
    // guid as arbitrary name to provide a unique value
    const char* const name = "init_optional_zero_point_int8_b33fd0fa-cd7b-4b10-ae5a-df64cabfe1f8";
    std::array<uint8_t, 1> a{0};
    ONNX_NAMESPACE::TensorProto tensor_proto;
    tensor_proto.set_name(name);
    tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT8);
    tensor_proto.set_raw_data(a.data(), sizeof(int8_t));

    return tensor_proto;
  };

  // We assume this function won't fail
  static const ONNX_NAMESPACE::TensorProto init_optional_zero_point_uint8() {
    // guid as arbitrary name to provide a unique value
    const char* const name = "init_optional_zero_point_uint8_b33f88f7-c464-43e3-8692-97ac832bb14a";
    std::array<uint8_t, 1> a{0};
    ONNX_NAMESPACE::TensorProto tensor_proto;
    tensor_proto.set_name(name);
    tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
    tensor_proto.set_raw_data(a.data(), sizeof(uint8_t));

    return tensor_proto;
  };
  static ONNX_NAMESPACE::TensorProto GetOptionalZeroPointInt8() {
    static ONNX_NAMESPACE::TensorProto proto = init_optional_zero_point_int8();
    return proto;
  }
  static ONNX_NAMESPACE::TensorProto GetOptionalZeroPointUint8() {
    static ONNX_NAMESPACE::TensorProto proto = init_optional_zero_point_uint8();
    return proto;
  }
};

void SetOptionalZeroPoint::UpdateNodes(Graph& graph, const NodesToOptimize& selected_nodes) {
  const auto nodes = selected_nodes.AllNodes();
  for (Node* node_ptr : nodes) {
    if (node_ptr == nullptr) {
      continue;
    }

    Node& node = *node_ptr;

    bool is_dq = node.OpType() == DQOpName;
    bool is_q = node.OpType() == QOpName;
    if (!is_dq && !is_q) {
      continue;
    }

    std::vector<NodeArg*>& input_defs = node.MutableInputDefs();
    bool has_zp_input = input_defs.size() == 3;
    if (has_zp_input && input_defs[InputIndex::ZERO_POINT_ID]->Exists()) {
      continue;  // zero point was set. No need to fill.
    }

    bool is_default_zp_signed = false;
    if (is_dq) {
      auto input_type = input_defs[0]->TypeAsProto()->tensor_type().elem_type();
      is_default_zp_signed = ONNX_NAMESPACE::TensorProto_DataType_INT8 == input_type;
    }

    const ONNX_NAMESPACE::TensorProto& zp_tensor_proto = is_default_zp_signed
                                                             ? GetOptionalZeroPointInt8()
                                                             : GetOptionalZeroPointUint8();

    const ONNX_NAMESPACE::TensorProto* dummy_zp_tensor_proto;
    if (!graph.GetInitializedTensor(zp_tensor_proto.name(), dummy_zp_tensor_proto)) {
      graph.AddInitializedTensor(zp_tensor_proto);
    }

    auto& node_arg = graph.GetOrCreateNodeArg(zp_tensor_proto.name(), nullptr);
    if (!has_zp_input) {
      input_defs.push_back(&node_arg);
    } else {
      input_defs[InputIndex::ZERO_POINT_ID] = &node_arg;
    }
  }
}

}  // namespace

Status QDQReplaceWithNew::Run(Graph& graph, const NodesToOptimize& selected_nodes) const {
  SetOptionalZeroPoint::UpdateNodes(graph, selected_nodes);
  return ReplaceWithNew::Run(graph, selected_nodes);
}

#if !defined(ORT_MINIMAL_BUILD)
Status QDQReplaceWithNew::RunForSave(Graph& graph, const NodesToOptimize& selected_nodes,
                                     const SatRuntimeOptimizationSaveContext& save_context,
                                     SavedState& saved_state, bool& graph_modified) const {
  SetOptionalZeroPoint::UpdateNodes(graph, selected_nodes);
  graph_modified = true;
  return ReplaceWithNew::RunForSave(graph, selected_nodes, save_context, saved_state, graph_modified);
}
#endif  // !defined(ORT_MINIMAL_BUILD)

UnaryReplaceWithQLinear::UnaryReplaceWithQLinear(std::string domain)
    : ReplaceWithQLinear(std::move(domain), UnaryMoves()) {
}

NodeAttributes UnaryReplaceWithQLinear::ExtraAttributes(const RuntimeState& state) const {
  const auto& target = state.selected_nodes.Target();
  NodeAttributes attr;
  if (target.OpType() == "Softmax") {
    attr["opset"] = utils::MakeAttribute(std::string("opset"), int64_t(target.SinceVersion()));
  }
  return attr;
}

BinaryReplaceWithQLinear::BinaryReplaceWithQLinear(std::string domain)
    : ReplaceWithQLinear(std::move(domain), BinaryMoves()) {
}

VariadicReplaceWithQLinear::VariadicReplaceWithQLinear(std::string domain)
    : ReplaceWithQLinear(std::move(domain), VariadicMoves()) {
}

ConvReplaceWithQLinear::ConvReplaceWithQLinear()
    : ReplaceWithQLinear(kOnnxDomain, ConvMoves()) {
}
WhereReplaceWithQLinear::WhereReplaceWithQLinear()
    : ReplaceWithQLinear(kMSDomain, WhereMoves()) {
}
MatMulReplaceWithQLinear::MatMulReplaceWithQLinear()
    : matmul_int_to_float_replacer_{MatMulIntToFloatReplacer()},
      qlinear_matmul_replacer_{kOnnxDomain} {
}

Status SplitReplaceWithQuant::Run(Graph& graph, const NodesToOptimize& selected_nodes) const {
  const auto& target_node = selected_nodes.Target();
  const auto& input_defs = target_node.InputDefs();

  // The 'split' attribute became an optional input at opset 13.
  bool has_split_as_input = target_node.SinceVersion() >= 13 && input_defs.size() == 2 && input_defs[1]->Exists();
  return SplitReplacer(has_split_as_input).Run(graph, selected_nodes);
}

Status MatMulReplaceWithQLinear::Run(Graph& graph, const NodesToOptimize& selected_nodes) const {
  // if the output is empty there were no Q nodes selected, so replace with MatMulIntegerToFloat
  // otherwise replace with QLinearMatMul
  bool matmul_integer_to_float = selected_nodes.num_outputs == 0;
  if (matmul_integer_to_float) {
    return matmul_int_to_float_replacer_.Run(graph, selected_nodes);
  } else {
    return qlinear_matmul_replacer_.Run(graph, selected_nodes);
  }
}

static std::vector<NodeAndMoveInfo> GetGemmMoveInfo(bool does_q_node_exist) {
  NTO::NodeLocation dq_A{NTO::NodeType::kInput, 0};
  NTO::NodeLocation dq_B{NTO::NodeType::kInput, 1};
  NTO::NodeLocation dq_bias{NTO::NodeType::kInput, 2};
  NTO::NodeLocation target{NTO::NodeType::kTarget, 0};
  NTO::NodeLocation q{NTO::NodeType::kOutput, 0};

  std::vector<NodeAndMoveInfo> moves{
      MoveAll(dq_A, ArgType::kInput),                                            // append all inputs from DQ of A
      MoveAll(dq_B, ArgType::kInput),                                            // append all inputs from DQ of B
      MoveAndAppend(dq_bias, ArgType::kInput, 0, ArgType::kInput, true, true)};  // (optional) append bias

  if (does_q_node_exist) {
    moves.push_back(MoveAndAppend(q, ArgType::kInput, 1, ArgType::kInput));  // append scale (input 1) from Q
    moves.push_back(MoveAndAppend(q, ArgType::kInput, 2, ArgType::kInput));  // append zp (input 2) from Q
    moves.push_back(MoveAll(q, ArgType::kOutput));                           // and use the outputs from Q
  } else {
    moves.push_back(MoveAll(target, ArgType::kOutput));
  }

  return moves;
}

GemmReplaceWithQuant::GemmReplaceWithQuant()
    : qgemm_with_float_as_output_replacer_(kMSDomain, "QGemm", GetGemmMoveInfo(false)),
      qgemm_with_8bits_as_output_replacer_(kMSDomain, "QGemm", GetGemmMoveInfo(true)) {
}

Status GemmReplaceWithQuant::Run(Graph& graph, const NodesToOptimize& selected_nodes) const {
  RemoveAttrBeta(selected_nodes);
  bool is_output_float = selected_nodes.num_outputs == 0;
  if (is_output_float) {
    return qgemm_with_float_as_output_replacer_.Run(graph, selected_nodes);
  }

  return qgemm_with_8bits_as_output_replacer_.Run(graph, selected_nodes);
}

#if !defined(ORT_MINIMAL_BUILD)
Status GemmReplaceWithQuant::RunForSave(Graph& graph,
                                        const NodesToOptimize& selected_nodes,
                                        const SatRuntimeOptimizationSaveContext& save_context,
                                        SavedState& saved_state,
                                        bool& graph_modified) const {
  RemoveAttrBeta(selected_nodes);
  bool is_output_float = selected_nodes.num_outputs == 0;
  if (is_output_float) {
    return qgemm_with_float_as_output_replacer_.RunForSave(graph,
                                                           selected_nodes,
                                                           save_context,
                                                           saved_state,
                                                           graph_modified);
  }

  return qgemm_with_8bits_as_output_replacer_.RunForSave(graph,
                                                         selected_nodes,
                                                         save_context,
                                                         saved_state,
                                                         graph_modified);
}
#endif  // !defined(ORT_MINIMAL_BUILD)

}  // namespace QDQ
}  // namespace onnxruntime
