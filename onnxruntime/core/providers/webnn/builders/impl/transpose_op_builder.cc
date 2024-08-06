// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class TransposeOpBuilder : public BaseOpBuilder {
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;
  bool HasSupportedInputsImpl(const Node& node, const WebnnDeviceType device_type,
                              const logging::Logger& logger) const override;
};

// Add operator related.

Status TransposeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                 const Node& node,
                                                 const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  NodeAttrHelper helper(node);
  std::vector<int64_t> perm = helper.Get("perm", std::vector<int64_t>());
  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get shape");
  auto input_dims = input_shape.size();
  if (perm.empty()) {
    for (int64_t i = input_dims - 1; i >= 0; i--)
      perm.push_back(i);
  } else {
    ORT_RETURN_IF_NOT(perm.size() == input_dims, "Perm and input should have same dimension");
  }

  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  emscripten::val options = emscripten::val::object();
  options.set("label", node.Name());
  std::vector<uint32_t> permutation = GetVecUint32FromVecInt64(perm);
  options.set("permutation", emscripten::val::array(permutation));
  emscripten::val output = model_builder.GetBuilder().call<emscripten::val>("transpose", input, options);
  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

bool TransposeOpBuilder::HasSupportedInputsImpl(const Node& node, const WebnnDeviceType device_type,
                                                const logging::Logger& logger) const {
  const auto& input = *node.InputDefs()[0];
  const auto& op_type = node.OpType();
  int32_t input_type;
  if (!GetType(input, input_type, logger))
    return false;

  std::unordered_set<ONNX_NAMESPACE::TensorProto_DataType> supported_data_types = webnn_supported_data_types;
  // WebNN CPU backend doesn't support uint32, uint64 input data types for transpose.
  if (device_type == WebnnDeviceType::CPU) {
    supported_data_types.erase(ONNX_NAMESPACE::TensorProto_DataType_UINT32);
    supported_data_types.erase(ONNX_NAMESPACE::TensorProto_DataType_UINT64);
  }

  if (!IsSupportedDataType(input_type, supported_data_types)) {
    LOGS(logger, VERBOSE) << "[" << op_type
                          << "] Input type: [" << input_type
                          << "] is not supported for now";
    return false;
  }

  return true;
}

void CreateTransposeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<TransposeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
