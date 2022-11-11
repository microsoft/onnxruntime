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
                       bool is_quantized_model,
                       std::vector<std::string>& input_names,
                       bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

Status ReshapeOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                       const NodeUnit& node_unit,
                                       const logging::Logger& logger,
                                       bool is_quantized_model,
                                       std::vector<std::string>& input_names,
                                       bool do_op_validation) const {
  if (do_op_validation) {
    NodeAttrHelper node_helper(node_unit);
    auto allowzero = node_helper.Get("allowzero", static_cast<int64_t>(0));
    if (0 != allowzero) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN Reshape doesn't support dynamic shape!");
    }
  }

  Qnn_QuantizeParams_t quantize_param = QNN_QUANTIZE_PARAMS_INIT;
  InitializeQuantizeParam(quantize_param, is_quantized_model);
  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;

  auto input_0 = node_unit.Inputs()[0];
  auto& input_name = input_0.node_arg.Name();

  if (qnn_model_wrapper.QnnContainsTensor(input_name)) {
    LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << input_name;
    input_names.push_back(input_name);
    return Status::OK();
  }

  const auto* type_proto = input_0.node_arg.TypeAsProto();
  int32_t onnx_data_type;
  ORT_RETURN_IF_ERROR(GetQnnDataType(is_quantized_model, type_proto, onnx_data_type, qnn_data_type));

  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input_0.node_arg, input_shape), "Cannot get shape");
  ORT_RETURN_IF_NOT(qnn_model_wrapper.ProcessQuantizationParameter(input_0.quant_param,
                                                                    quantize_param.scaleOffsetEncoding.scale,
                                                                    quantize_param.scaleOffsetEncoding.offset),
                    "Cannot get quantization parameter");

  std::vector<uint8_t> unpacked_tensor;
  bool is_initializer_input = qnn_model_wrapper.IsInitializerInput(input_name);
  if (is_initializer_input) {
    const auto& input_tensor = qnn_model_wrapper.GetInitializerTensors().at(input_name);
    ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(*input_tensor, unpacked_tensor));
  }

  input_names.push_back(input_name);

  Qnn_TensorType_t tensor_type = GetInputTensorType(qnn_model_wrapper, input_name);
  Qnn_TensorDataFormat_t data_format = 0;
  QnnTensorWrapper input_tensorwrapper(input_name, tensor_type, data_format, qnn_data_type, quantize_param,
                                       std::move(input_shape), std::move(unpacked_tensor));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensor(input_name, std::move(input_tensorwrapper)), "Failed to add tensor.");

  return Status::OK();
}

void CreateReshapeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<ReshapeOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
