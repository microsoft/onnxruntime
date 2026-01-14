// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn-abi/builder/op_builder_factory.h"
#include "core/providers/qnn-abi/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn-abi/builder/qnn_model_wrapper.h"
#include "core/providers/qnn-abi/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

class ReshapeOpBuilder : public BaseOpBuilder {
 public:
  ReshapeOpBuilder() : BaseOpBuilder("ReshapeOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ReshapeOpBuilder);

 protected:
  Ort::Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger,
                            std::vector<std::string>& input_names,
                            bool do_op_validation) const override ORT_MUST_USE_RESULT;
  Ort::Status OverrideOutputQuantParam(QnnModelWrapper& qnn_model_wrapper,
                                       const OrtNodeUnit& node_unit,
                                       const Ort::Logger& logger,
                                       const std::vector<std::string>& input_names,
                                       size_t output_index,
                                       Qnn_DataType_t qnn_data_type,
                                       QnnQuantParamsWrapper& quant_param) const override ORT_MUST_USE_RESULT;
};

Ort::Status ReshapeOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                            const OrtNodeUnit& node_unit,
                                            const Ort::Logger& logger,
                                            std::vector<std::string>& input_names,
                                            bool do_op_validation) const {
  if (do_op_validation) {
    OrtNodeAttrHelper node_helper(node_unit);
    auto allowzero = node_helper.Get("allowzero", static_cast<int64_t>(0));

    // Only reject allowzero=1 if the shape input is not constant or it actually contains zeros
    if (0 != allowzero) {
      const auto& inputs = node_unit.Inputs();
      const OrtValueInfo* shape_tensor = nullptr;
      RETURN_IF_ERROR(qnn_model_wrapper.FindInitializer(inputs[1].name, &shape_tensor));

      // Check if the constant shape contains any zeros
      std::vector<uint8_t> unpacked_tensor;
      RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(shape_tensor, unpacked_tensor));

      const int64_t* shape_data = reinterpret_cast<const int64_t*>(unpacked_tensor.data());
      size_t shape_size = unpacked_tensor.size() / sizeof(int64_t);

      for (size_t i = 0; i < shape_size; ++i) {
        RETURN_IF(shape_data[i] == 0,
                  "QNN Reshape does not support shapes with zero dimensions. "
                  "The 'allowzero' attribute is not supported by QNN.");
      }
    }
  }

  const auto& input_0 = node_unit.Inputs()[0];
  RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, input_0, logger, input_names));

  return Ort::Status();
}

Ort::Status ReshapeOpBuilder::OverrideOutputQuantParam(QnnModelWrapper& qnn_model_wrapper,
                                                       const OrtNodeUnit& node_unit,
                                                       const Ort::Logger& logger,
                                                       const std::vector<std::string>& input_names,
                                                       size_t output_index,
                                                       Qnn_DataType_t qnn_data_type,
                                                       QnnQuantParamsWrapper& quant_param) const {
  if (!quant_param.IsPerTensor()) {
    return Ort::Status();
  }

  // Force Reshape output to use the same quantization parameters as the input if nearly equal.
  // This helps the HTP backend employ certain optimizations.
  return SetOutputQParamEqualToInputIfNearlyEqual(qnn_model_wrapper, node_unit, logger, input_names,
                                                  0 /*input_index*/, output_index, qnn_data_type, quant_param);
}

void CreateReshapeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<ReshapeOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
