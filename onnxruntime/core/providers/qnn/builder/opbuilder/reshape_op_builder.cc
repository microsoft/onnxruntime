// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace qnn {

class ReshapeOpBuilder : public BaseOpBuilder {
 public:
  ReshapeOpBuilder() : BaseOpBuilder("ReshapeOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ReshapeOpBuilder);

 protected:
  Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger,
                       std::vector<std::string>& input_names,
                       bool do_op_validation) const override ORT_MUST_USE_RESULT;
  Status OverrideOutputQuantParam(QnnModelWrapper& qnn_model_wrapper,
                                  const NodeUnit& node_unit,
                                  const logging::Logger& logger,
                                  const std::vector<std::string>& input_names,
                                  size_t output_index,
                                  Qnn_DataType_t qnn_data_type,
                                  Qnn_QuantizeParams_t& quant_param) const override ORT_MUST_USE_RESULT;
};

Status ReshapeOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                       const NodeUnit& node_unit,
                                       const logging::Logger& logger,
                                       std::vector<std::string>& input_names,
                                       bool do_op_validation) const {
  if (do_op_validation) {
    NodeAttrHelper node_helper(node_unit);
    auto allowzero = node_helper.Get("allowzero", static_cast<int64_t>(0));
    if (0 != allowzero) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN Reshape doesn't support dynamic shape!");
    }
  }

  const auto& input_0 = node_unit.Inputs()[0];
  ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, input_0, logger, input_names));

  return Status::OK();
}

Status ReshapeOpBuilder::OverrideOutputQuantParam(QnnModelWrapper& qnn_model_wrapper,
                                                  const NodeUnit& node_unit,
                                                  const logging::Logger& logger,
                                                  const std::vector<std::string>& input_names,
                                                  size_t output_index,
                                                  Qnn_DataType_t qnn_data_type,
                                                  Qnn_QuantizeParams_t& quant_param) const {
  // Force Reshape output to use the same quantization parameters as the input if nearly equal.
  // This helps the HTP backend emply certain optimizations.
  return SetOutputQParamEqualToInputIfNearlyEqual(qnn_model_wrapper, node_unit, logger, input_names,
                                                  0 /*input_index*/, output_index, qnn_data_type, quant_param);
}

void CreateReshapeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<ReshapeOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
