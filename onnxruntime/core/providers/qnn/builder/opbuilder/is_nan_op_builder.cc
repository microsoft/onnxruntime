// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {
class IsNanOpBuilder : public BaseOpBuilder {
 public:
  IsNanOpBuilder() : BaseOpBuilder("IsNanOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(IsNanOpBuilder);

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

Ort::Status IsNanOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          const Ort::Logger& logger,
                                          std::vector<std::string>& input_names,
                                          bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();

  if (do_op_validation) {
    TensorInfo input_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[0], input_info));
    Qnn_DataType_t target_tensor_type = input_info.qnn_data_type;

    RETURN_IF((QNN_DATATYPE_FLOAT_32 != target_tensor_type && QNN_DATATYPE_FLOAT_16 != target_tensor_type),
              "QNN IsNan Op supports only float32 and float16 input tensors.");
  }
  const auto input_count = GetInputCountQnnRequired(node_unit);
  for (size_t input_i = 0; input_i < input_count; ++input_i) {
    RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[input_i], logger, input_names));
  }

  return Ort::Status();
}

Ort::Status IsNanOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                        const OrtNodeUnit& node_unit,
                                                        std::vector<std::string>&& input_names,
                                                        const Ort::Logger& logger,
                                                        bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);
  TensorInfo output_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[0], output_info));
  if (do_op_validation) {
    RETURN_IF(QNN_DATATYPE_BOOL_8 != output_info.qnn_data_type, "QNN IsNan Op support only bool8 output tensor.");
  }
  const std::string& org_output_name = node_unit.Outputs()[0].name;
  const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(org_output_name);
  std::vector<uint32_t> output_shape = output_info.shape;
  const std::string isnan_node_name = utils::GetUniqueName(node_unit, "_IsNan");

  QnnTensorWrapper isnan_output(org_output_name,
                                is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE,
                                output_info.qnn_data_type,
                                output_info.quant_param.Copy(),
                                std::move(output_shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(isnan_output)), "Failed to add tensor.");
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(isnan_node_name,
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_IS_NAN,
                                                {input_names[0]},
                                                {org_output_name},
                                                {},
                                                false),
                "Failed to create QNN IsNan node.");

  return Ort::Status();
}

void CreateIsNanOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<IsNanOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
