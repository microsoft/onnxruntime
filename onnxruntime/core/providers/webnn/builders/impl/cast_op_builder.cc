// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"
#include "core/providers/shared/utils/utils.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class CastOpBuilder : public BaseOpBuilder {
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
 private:
  bool HasSupportedInputsImpl(const Node& node, const emscripten::val& wnn_limits,
                              const logging::Logger& logger) const override;
};

// Add operator related.

Status CastOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                            const Node& node,
                                            const logging::Logger& logger) const {
  const auto& input_name = node.InputDefs()[0]->Name();
  emscripten::val input = model_builder.GetOperand(input_name);

  NodeAttrHelper helper(node);
  // We already checked the "to" type in IsOpSupportedImpl.
  const auto to_type = helper.Get("to", ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  std::string operand_type;
  switch (to_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      operand_type = "uint8";
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      operand_type = "int8";
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      operand_type = "float16";
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      operand_type = "float32";
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      operand_type = "int32";
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      operand_type = "int64";
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
      operand_type = "uint32";
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
      operand_type = "uint64";
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "The Cast node has unsupported 'to' type, name: ",
                             node.Name(), " type: ", to_type);
  }

  emscripten::val options = emscripten::val::object();
  options.set("label", node.Name());

  emscripten::val output =
      model_builder.GetBuilder().call<emscripten::val>("cast", input, emscripten::val(operand_type), options);

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related.
bool CastOpBuilder::HasSupportedInputsImpl(const Node& node, const emscripten::val& wnn_limits,
                                           const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& op_type = node.OpType();
  int32_t input_type;

  if (!GetType(*input_defs[0], input_type, logger))
    return false;

  if (!IsSupportedDataType(input_type, wnn_limits["cast"]["input"]["dataTypes"])) {
    LOGS(logger, VERBOSE) << "[" << op_type
                          << "] Input type: [" << input_type
                          << "] is not supported for now";
    return false;
  }

  NodeAttrHelper helper(node);
  // Check cast to type.
  const auto to_type = helper.Get("to", ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED);
  if (!IsSupportedDataType(to_type, wnn_limits["cast"]["output"]["dataTypes"])) {
    LOGS(logger, VERBOSE) << "[" << op_type
                          << "] to type: [" << to_type
                          << "] is not supported for now";
    return false;
  }

  return true;
}

void CreateCastOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<CastOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
