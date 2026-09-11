// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

class ConcatOpBuilder : public BaseOpBuilder {
 public:
  ConcatOpBuilder() : BaseOpBuilder("ConcatOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ConcatOpBuilder);

 protected:
  Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger,
                       std::vector<std::string>& input_names,
                       bool do_op_validation) const override ORT_MUST_USE_RESULT;

  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     const logging::Logger& logger,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

Status ConcatOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                      const NodeUnit& node_unit,
                                      const logging::Logger& logger,
                                      std::vector<std::string>& input_names,
                                      bool /*do_op_validation*/) const {
  const auto& inputs = node_unit.Inputs();

  for (const auto& input : inputs) {
    const auto& input_name = input.node_arg.Name();
    bool has_zero_dim = false;

    // Check if the tensor has a 0 dimension
    if (qnn_model_wrapper.IsConstantInput(input_name)) {
      // Process constant inputs (initializers)
      const auto* input_tensor = qnn_model_wrapper.GetConstantTensor(input_name);
      if (input_tensor != nullptr) {
        const auto& shape = input_tensor->dims();
        if (std::find(shape.begin(), shape.end(), 0) != shape.end()) {
          // Found a 0 dimension, skip this input
          LOGS(logger, VERBOSE) << "Constant input tensor " << input_name << " has a 0 dimension, excluding from Concat";
          has_zero_dim = true;
        }
      }
    } else {
      // Process non-constant inputs
      std::vector<uint32_t> shape;
      ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input.node_arg, shape), "Cannot get shape");

      if (std::find(shape.begin(), shape.end(), 0) != shape.end()) {
        // Found a 0 dimension, skip this input
        LOGS(logger, VERBOSE) << "Input tensor " << input_name << " has a 0 dimension, excluding from Concat";
        has_zero_dim = true;
      }
    }

    // Process the input if it doesn't have a 0 dimension
    if (!has_zero_dim) {
      ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, input, logger, input_names));
    }
  }

  // If all inputs have 0 dimensions, return an error as Concat requires at least one non-zero dimension input
  if (input_names.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Concat operation requires at least one input without a 0 dimension");
  }

  return Status::OK();
}

Status ConcatOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                    const NodeUnit& node_unit,
                                                    std::vector<std::string>&& input_names,
                                                    const logging::Logger& logger,
                                                    bool do_op_validation) const {
  if (input_names.size() < 1) {
    return Status::OK();
  }

  std::vector<std::string> param_tensor_names;

  // Process axis attribute
  int32_t default_axis = 0;
  Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
  ORT_RETURN_IF_ERROR(ProcessAxisAttribute(qnn_model_wrapper, node_unit, axis_qnn_scalar, default_axis));
  QnnParamWrapper axis_param(node_unit.Index(), node_unit.Name(), QNN_OP_CONCAT_PARAM_AXIS, axis_qnn_scalar);
  param_tensor_names.push_back(axis_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(axis_param));

  // Process outputs
  return ProcessOutputs(qnn_model_wrapper, node_unit,
                        std::move(input_names),
                        std::move(param_tensor_names),
                        logger, do_op_validation, GetQnnOpType(node_unit.OpType()));
}

void CreateConcatOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<ConcatOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
