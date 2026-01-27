// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn-abi/builder/op_builder_factory.h"
#include "core/providers/qnn-abi/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn-abi/builder/qnn_model_wrapper.h"
#include "core/providers/qnn-abi/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

class ConcatOpBuilder : public BaseOpBuilder {
 public:
  ConcatOpBuilder() : BaseOpBuilder("ConcatOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ConcatOpBuilder);

 protected:
  Ort::Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger,
                            std::vector<std::string>& input_names,
                            bool do_op_validation) const override ORT_MUST_USE_RESULT;

  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          const Ort::Logger& logger,
                                          bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

Ort::Status ConcatOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                           const OrtNodeUnit& node_unit,
                                           const Ort::Logger& logger,
                                           std::vector<std::string>& input_names,
                                           bool /*do_op_validation*/) const {
  const auto& inputs = node_unit.Inputs();

  for (const auto& input : inputs) {
    const auto& input_name = input.name;
    bool has_zero_dim = false;

    // Check if the tensor has a 0 dimension
    if (qnn_model_wrapper.IsConstantInput(input_name)) {
      // Process constant inputs (initializers)
      const OrtValueInfo* input_tensor = qnn_model_wrapper.GetConstantTensor(input_name);
      if (input_tensor != nullptr) {
        const auto& shape = Ort::ConstValueInfo(input_tensor).TypeInfo().GetTensorTypeAndShapeInfo().GetShape();
        if (std::find(shape.begin(), shape.end(), 0) != shape.end()) {
          // Found a 0 dimension, skip this input
          ORT_CXX_LOG(logger,
                      ORT_LOGGING_LEVEL_VERBOSE,
                      ("Constant input tensor " + input_name + " has a 0 dimension, excluding from Concat").c_str());
          has_zero_dim = true;
        }
      }
    } else {
      // Process non-constant inputs
      std::vector<uint32_t> shape;
      RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input.shape, shape), "Cannot get shape");

      if (std::find(shape.begin(), shape.end(), 0) != shape.end()) {
        // Found a 0 dimension, skip this input
        ORT_CXX_LOG(logger,
                    ORT_LOGGING_LEVEL_VERBOSE,
                    ("Constant input tensor " + input_name + " has a 0 dimension, excluding from Concat").c_str());
        has_zero_dim = true;
      }
    }

    // Process the input if it doesn't have a 0 dimension
    if (!has_zero_dim) {
      RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, input, logger, input_names));
    }
  }

  // If all inputs have 0 dimensions, return an error as Concat requires at least one non-zero dimension input
  RETURN_IF(input_names.empty(), "Concat operation requires at least one input without a 0 dimension");

  return Ort::Status();
}

Ort::Status ConcatOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                         const OrtNodeUnit& node_unit,
                                                         std::vector<std::string>&& input_names,
                                                         const Ort::Logger& logger,
                                                         bool do_op_validation) const {
  if (input_names.size() < 1) {
    return Ort::Status();
  }

  std::vector<std::string> param_tensor_names;

  // Process axis attribute
  int32_t default_axis = 0;
  Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
  RETURN_IF_ERROR(ProcessAxisAttribute(qnn_model_wrapper, node_unit, axis_qnn_scalar, default_axis));
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
