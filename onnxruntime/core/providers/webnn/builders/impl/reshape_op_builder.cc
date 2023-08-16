// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/optimizer/initializer.h"
#include "core/providers/common.h"
#include "core/providers/cpu/tensor/reshape_helper.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class ReshapeOpBuilder : public BaseOpBuilder {
  // Add operator related.
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
};

// Add operator related.

void ReshapeOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  model_builder.AddInitializerToSkip(node.InputDefs()[1]->Name());
}

Status ReshapeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                               const Node& node,
                                               const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& initializers(model_builder.GetInitializerTensors());
  const auto& target_shape_tensor = *initializers.at(input_defs[1]->Name());
  const int64_t* raw_target_shape = target_shape_tensor.int64_data().empty()
                                        ? reinterpret_cast<const int64_t*>(target_shape_tensor.raw_data().data())
                                        : target_shape_tensor.int64_data().data();

  const auto size = target_shape_tensor.dims()[0];
  TensorShapeVector target_shape{raw_target_shape, raw_target_shape + size};
  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get shape");
  ReshapeHelper helper(TensorShape(input_shape), target_shape);
  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  std::vector<int32_t> new_shape;
  std::transform(target_shape.cbegin(), target_shape.cend(),
                 std::back_inserter(new_shape),
                 [](int64_t dim) -> uint32_t { return SafeInt<int32_t>(dim); });
  emscripten::val output = model_builder.GetBuilder().call<emscripten::val>("reshape",
                                                                            input, emscripten::val::array(new_shape));
  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related.

bool ReshapeOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& initializers,
                                         const Node& node,
                                         const WebnnDeviceType /* device_type */,
                                         const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& perm_name = input_defs[1]->Name();
  if (!Contains(initializers, perm_name)) {
    LOGS(logger, VERBOSE) << "New shape of reshape must be a constant initializer";
    return false;
  }

  const auto& perm_tensor = *initializers.at(perm_name);
  std::vector<uint8_t> unpacked_tensor;
  auto status = onnxruntime::utils::UnpackInitializerData(perm_tensor, unpacked_tensor);
  if (!status.IsOK()) {
    LOGS(logger, ERROR) << "Error while unpacking perm_tensor: " << status.ErrorMessage();
    return false;
  }

  const int64_t* raw_new_shape = reinterpret_cast<const int64_t*>(unpacked_tensor.data());
  const auto& perm_dims = perm_tensor.dims();
  if (perm_dims.empty() || perm_dims[0] == 0) {
    LOGS(logger, VERBOSE) << "New shape of reshape cannot be empty";
    return false;
  }

  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger))
    return false;

  if (input_shape.empty()) {
    LOGS(logger, VERBOSE) << "Reshape does not support empty input shape";
    return false;
  }

  // WebNN reshape does not support 0 as dimension.
  NodeAttrHelper helper(node);
  const bool allow_zero = helper.Get("allowzero ", 0) == 1;
  if (allow_zero) {
    for (int64_t i = 0; i < perm_dims[0]; i++) {
      if (raw_new_shape[i] == 0) {
        LOGS_DEFAULT(VERBOSE) << "Reshape doesn't support 0 reshape dimension when allowzero is enabled";
        return false;
      }
    }
  }

  return true;
}

void CreateReshapeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ReshapeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
