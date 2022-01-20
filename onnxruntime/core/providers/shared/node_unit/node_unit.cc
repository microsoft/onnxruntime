// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "node_unit.h"
#include "core/graph/graph.h"

namespace onnxruntime {

namespace {

// The QLinearOpType GetQLinearOpType, is very similar to the one in NNAPI
// However, the NNAPI ones are only the subset of the ones here,
// TODO, make these shared
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

}  // namespace

NodeUnit::NodeUnit(const Node& node)
    : output_nodes_{&node},
      target_node_(node),
      type_(Type::SingleNode) {
  InitForNode();
}

const std::string& NodeUnit::Domain() const noexcept { return target_node_.Domain(); }
const std::string& NodeUnit::OpType() const noexcept { return target_node_.OpType(); }
const std::string& NodeUnit::Name() const noexcept { return target_node_.Name(); }
int NodeUnit::SinceVersion() const noexcept { return target_node_.SinceVersion(); }
NodeIndex NodeUnit::Index() const noexcept { return target_node_.Index(); }
const Path& NodeUnit::ModelPath() const noexcept { return target_node_.ModelPath(); }
ProviderType NodeUnit::GetExecutionProviderType() const noexcept { return target_node_.GetExecutionProviderType(); }

void NodeUnit::InitForNode() {
  const auto& input_defs = target_node_.InputDefs();
  const auto& output_defs = target_node_.OutputDefs();
  auto qlinear_type = GetQLinearOpType(target_node_);
  if (qlinear_type == QLinearOpType::Unknown ||
      IsVariadicQLinearOp(qlinear_type)) {  // TODO, add variadic support
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
    inputs_.push_back(NodeUnitIODef{
        *input_defs[0],
        NodeUnitIODef::QuantParam{*input_defs[1], input_defs[2]}});

    outputs_.push_back(NodeUnitIODef{
        *output_defs[0],
        NodeUnitIODef::QuantParam{*input_defs[3],
                                  input_defs.size() > 4
                                      ? input_defs[4]
                                      : nullptr}});
  } else if (IsBinaryQLinearOp(qlinear_type)) {
    // Binary QLinear Op has 9 inputs
    // x1, x1_scale, x1_zp, x2/w, x2_scale, x2_zp, y_scale , y_zp, B
    inputs_.push_back(NodeUnitIODef{
        *input_defs[0],
        NodeUnitIODef::QuantParam{*input_defs[1], input_defs[2]}});
    inputs_.push_back(NodeUnitIODef{
        *input_defs[3],
        NodeUnitIODef::QuantParam{*input_defs[4], input_defs[5]}});

    if (input_defs.size() == 9) {  // has Bias
      inputs_.push_back(NodeUnitIODef{
          *input_defs[8],
          std::nullopt});  // for Bias the scale and zp are optional
    }

    outputs_.push_back(NodeUnitIODef{
        *output_defs[0],
        NodeUnitIODef::QuantParam{*input_defs[6], input_defs[7]}});
  } else if (qlinear_type == QLinearOpType::DequantizeLinear) {
    // DequantizeLinear has 3 inputs
    // x, x_scale, x_zp
    // output is not quantized
    inputs_.push_back(NodeUnitIODef{
        *input_defs[0],
        NodeUnitIODef::QuantParam{*input_defs[1],
                                  input_defs.size() == 3
                                      ? input_defs[2]
                                      : nullptr}});
    outputs_.push_back(NodeUnitIODef{*output_defs[0], std::nullopt});
  } else if (qlinear_type == QLinearOpType::QuantizeLinear) {
    // QuantizeLinear the input is not quantized and has 3 inputs
    // x, y_scale, y_zp (optional)
    // The output is quantized
    inputs_.push_back(NodeUnitIODef{*input_defs[0], std::nullopt});
    outputs_.push_back(NodeUnitIODef{
        *output_defs[0],
        NodeUnitIODef::QuantParam{*input_defs[1],
                                  input_defs.size() == 3
                                      ? input_defs[2]
                                      : nullptr}});
  } else {
    ORT_THROW("The QLinear op [", static_cast<uint8_t>(qlinear_type), "] is not supported");
  }
}

}  // namespace onnxruntime
