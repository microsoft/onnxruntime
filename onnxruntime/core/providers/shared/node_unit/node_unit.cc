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
    : nodes_{&node},
      node_(node),
      type_(Type::SingleNode) {
  InitForNode();
}

const std::string& NodeUnit::Domain() const noexcept { return node_.Domain(); }
const std::string& NodeUnit::OpType() const noexcept { return node_.OpType(); }
const std::string& NodeUnit::Name() const noexcept { return node_.Name(); }
int NodeUnit::SinceVersion() const noexcept { return node_.SinceVersion(); }
NodeIndex NodeUnit::Index() const noexcept { return node_.Index(); }
const Path& NodeUnit::ModelPath() const noexcept { return node_.ModelPath(); }
ProviderType NodeUnit::GetExecutionProviderType() const noexcept { return node_.GetExecutionProviderType(); }

void NodeUnit::InitForNode() {
  const auto& input_defs = node_.InputDefs();
  const auto& output_defs = node_.OutputDefs();
  // The 1st step is to hookup the NodeUnit with the NNAPI builder interface
  // So we are not handling quantization here now
  auto qlinear_type = GetQLinearOpType(node_);
  if (qlinear_type == QLinearOpType::Unknown) {
    //Not a Qlinear op, add all inputs / outputs
    auto add_all_io = [](std::vector<NodeUnitIODef>& defs,
                         const ConstPointerContainer<std::vector<NodeArg*>>& node_defs) {
      defs.reserve(node_defs.size());

      for (const auto def : node_defs) {
        defs.push_back(NodeUnitIODef{*def, std::nullopt});
      }
    };
    add_all_io(input_defs_, input_defs);
    add_all_io(output_defs_, output_defs);
  } else if (IsUnaryQLinearOp(qlinear_type)) {
    // Unary QLinear Op has 5 inputs
    // x, x_scale, x_zp, y_scale, y_zp (optional)
    input_defs_.push_back(NodeUnitIODef{
        *input_defs[0],
        NodeUnitIODef::QuantParam{*input_defs[1], input_defs[2]}});

    output_defs_.push_back(NodeUnitIODef{
        *output_defs[0],
        NodeUnitIODef::QuantParam{*input_defs[3],
                                  input_defs_.size() > 4
                                      ? input_defs[4]
                                      : nullptr}});
  } else if (IsBinaryQLinearOp(qlinear_type)) {
    // Binary QLinear Op has 9 inputs
    // x1, x1_scale, x1_zp, x2/w, x2_scale, x2_zp, y_scale , y_zp, B
    input_defs_.push_back(NodeUnitIODef{
        *input_defs[0],
        NodeUnitIODef::QuantParam{*input_defs[1], input_defs[2]}});
    input_defs_.push_back(NodeUnitIODef{
        *input_defs[3],
        NodeUnitIODef::QuantParam{*input_defs[4], input_defs[5]}});

    if (input_defs_.size() == 9) {  // has Bias
      input_defs_.push_back(NodeUnitIODef{
          *input_defs[8],
          std::nullopt});  // for Bias the scale and zp are optional
    }

    output_defs_.push_back(NodeUnitIODef{
        *output_defs[0],
        NodeUnitIODef::QuantParam{*input_defs[6], input_defs[7]}});
  } else if (IsVariadicQLinearOp(qlinear_type)) {
    // TODO, add variadic support
    ORT_NOT_IMPLEMENTED();
  } else if (qlinear_type == QLinearOpType::DequantizeLinear) {
    // DequantizeLinear has 3 inputs
    // x, x_scale, x_zp
    // output is not quantized
    input_defs_.push_back(NodeUnitIODef{
        *input_defs[0],
        NodeUnitIODef::QuantParam{*input_defs[1],
                                  input_defs_.size() == 3
                                      ? input_defs[2]
                                      : nullptr}});
    output_defs_.push_back(NodeUnitIODef{*output_defs[0], std::nullopt});
  } else if (qlinear_type == QLinearOpType::QuantizeLinear) {
    // QuantizeLinear the input is not quantized and has 3 inputs
    // x, y_scale, y_zp (optional)
    // The output is quantized
    input_defs_.push_back(NodeUnitIODef{*input_defs[0], std::nullopt});
    output_defs_.push_back(NodeUnitIODef{
        *output_defs[0],
        NodeUnitIODef::QuantParam{*input_defs[1],
                                  input_defs_.size() == 3
                                      ? input_defs[2]
                                      : nullptr}});
  } else {
    ORT_THROW("The QLinear op [", static_cast<uint8_t>(qlinear_type), "] is not supported");
  }
}

}  // namespace onnxruntime
