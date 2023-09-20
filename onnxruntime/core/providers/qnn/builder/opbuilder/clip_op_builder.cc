// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cassert>
#include <limits>

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_utils.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace qnn {
class ClipOpBuilder : public BaseOpBuilder {
 public:
  ClipOpBuilder() : BaseOpBuilder("ClipOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ClipOpBuilder);

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

 private:
  Status ExplictOpCheck(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit) const;
};

Status ClipOpBuilder::ExplictOpCheck(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit) const {
  if (node_unit.Inputs().size() > 1) {
    const auto& min_input_name = node_unit.Inputs()[1].node_arg.Name();
    if (!min_input_name.empty() && !qnn_model_wrapper.IsInitializerInput(min_input_name)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN desn't support dynamic min/max.");
    }
  }
  if (node_unit.Inputs().size() > 2) {
    const auto& max_input_name = node_unit.Inputs()[2].node_arg.Name();
    if (!max_input_name.empty() && !qnn_model_wrapper.IsInitializerInput(max_input_name)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN desn't support dynamic min/max.");
    }
  }
  return Status::OK();
}

Status ClipOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                    const NodeUnit& node_unit,
                                    const logging::Logger& logger,
                                    std::vector<std::string>& input_names,
                                    bool do_op_validation) const {
  if (do_op_validation) {
    ORT_RETURN_IF_ERROR(ExplictOpCheck(qnn_model_wrapper, node_unit));
  }

  return ProcessInput(qnn_model_wrapper, node_unit.Inputs()[0], logger, input_names);
}

Status ClipOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                  const NodeUnit& node_unit,
                                                  std::vector<std::string>&& input_names,
                                                  const logging::Logger& logger,
                                                  bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  const size_t num_inputs = inputs.size();

  const Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;
  std::vector<std::string> param_tensor_names;

  auto get_f32_from_bytes = [](const std::vector<uint8_t>& bytes, float default_val) -> float {
    return bytes.empty() ? default_val : *reinterpret_cast<const float*>(bytes.data());
  };

  // Set the 'min' parameter.
  {
    std::vector<uint8_t> min_val_bytes;

    if (num_inputs > 1 && !inputs[1].node_arg.Name().empty()) {
      OnnxInputInfo min_input_info = {};
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetOnnxInputInfo(inputs[1], min_input_info));
      ORT_RETURN_IF_NOT(min_input_info.qnn_data_type == qnn_data_type,
                        "QNN EP: The 'min' input of the Clip operator must be of type float32.");
      assert(min_input_info.is_initializer);  // Checked by ExplicitOpCheck().
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*min_input_info.initializer_tensor, min_val_bytes));
    }

    Qnn_Scalar_t min_qnn_scalar = QNN_SCALAR_INIT;
    min_qnn_scalar.dataType = qnn_data_type;
    min_qnn_scalar.floatValue = get_f32_from_bytes(min_val_bytes, std::numeric_limits<float>::lowest());
    QnnParamWrapper min_value_param(node_unit.Index(), node_unit.Name(), QNN_OP_RELU_MIN_MAX_PARAM_MIN_VALUE,
                                    min_qnn_scalar);
    param_tensor_names.push_back(min_value_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(min_value_param));
  }

  // Set the 'max' parameter.
  {
    std::vector<uint8_t> max_val_bytes;

    if (num_inputs > 2 && !inputs[2].node_arg.Name().empty()) {
      OnnxInputInfo max_input_info = {};
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetOnnxInputInfo(inputs[2], max_input_info));
      ORT_RETURN_IF_NOT(max_input_info.qnn_data_type == qnn_data_type,
                        "QNN EP: The 'max' input of the Clip operator must of type float32.");
      assert(max_input_info.is_initializer);  // Checked by ExplicitOpCheck().
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*max_input_info.initializer_tensor, max_val_bytes));
    }

    Qnn_Scalar_t max_qnn_scalar = QNN_SCALAR_INIT;
    max_qnn_scalar.dataType = qnn_data_type;
    max_qnn_scalar.floatValue = get_f32_from_bytes(max_val_bytes, std::numeric_limits<float>::max());
    QnnParamWrapper max_value_param(node_unit.Index(), node_unit.Name(), QNN_OP_RELU_MIN_MAX_PARAM_MAX_VALUE,
                                    max_qnn_scalar);
    param_tensor_names.push_back(max_value_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(max_value_param));
  }

  ORT_RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit,
                                     std::move(input_names),
                                     std::move(param_tensor_names),
                                     logger, do_op_validation, GetQnnOpType(node_unit.OpType())));

  return Status::OK();
}

void CreateClipOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<ClipOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
