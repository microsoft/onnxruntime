// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_utils.h"

#include "base_op_builder.h"

#include <limits>

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
                       bool is_quantized_model,
                       std::vector<std::string>& input_names,
                       bool do_op_validation) const override ORT_MUST_USE_RESULT;

  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     const logging::Logger& logger,
                                     bool is_quantized_model,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;

 private:
  Status ExplictOpCheck(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit) const;
  mutable float min_value_ = std::numeric_limits<float>::lowest();
  mutable float max_value_ = std::numeric_limits<float>::max();
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
                                    bool is_quantized_model,
                                    std::vector<std::string>& input_names,
                                    bool do_op_validation) const {
  if (do_op_validation) {
    ORT_RETURN_IF_ERROR(ExplictOpCheck(qnn_model_wrapper, node_unit));
  }
  Qnn_QuantizeParams_t quantize_param = QNN_QUANTIZE_PARAMS_INIT;
  utils::InitializeQuantizeParam(quantize_param, is_quantized_model);
  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;

  auto inputs = node_unit.Inputs();
  for (size_t input_i = 0; input_i < inputs.size(); ++input_i) {
    auto& input_name = inputs[input_i].node_arg.Name();
    if (input_name.empty()) {
      // Ignore unspecified/unused optional input
      continue;
    }
    if (qnn_model_wrapper.IsQnnTensorWrapperExist(input_name)) {
      LOGS(logger, VERBOSE) << "Tensor already added or the input is not named, skip it: " << input_name;
      input_names.push_back(input_name);
      continue;
    }

    const auto* type_proto = inputs[input_i].node_arg.TypeAsProto();
    ORT_RETURN_IF_ERROR(utils::GetQnnDataType(is_quantized_model, type_proto, qnn_data_type));

    std::vector<uint32_t> input_shape;
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[input_i].node_arg, input_shape), "Cannot get shape");

    ORT_RETURN_IF_NOT(qnn_model_wrapper.ProcessQuantizationParameter(inputs[input_i].quant_param,
                                                                     quantize_param.scaleOffsetEncoding.scale,
                                                                     quantize_param.scaleOffsetEncoding.offset),
                      "Cannot get quantization parameter");

    float* ini_data = nullptr;
    std::vector<uint8_t> unpacked_tensor;
    bool is_initializer_input = qnn_model_wrapper.IsInitializerInput(input_name);
    if (is_initializer_input) {
      const auto& input_tensor = qnn_model_wrapper.GetInitializerTensors().at(input_name);
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*input_tensor, unpacked_tensor));
      ini_data = reinterpret_cast<float*>(unpacked_tensor.data());
      if (input_i == 1) {
        min_value_ = *ini_data;
        continue;
      } else if (input_i == 2) {
        max_value_ = *ini_data;
        continue;
      }
    }
    ORT_ENFORCE(input_i == 0, "QNN ReluMinMax operator expects only one input. Min and max are expected to be parameters, ie. initializer inputs in ONNX model");

    Qnn_TensorType_t tensor_type = GetInputTensorType(qnn_model_wrapper, input_name);
    QnnTensorWrapper input_tensorwrapper(input_name, tensor_type, qnn_data_type, quantize_param,
                                         std::move(input_shape), std::move(unpacked_tensor));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
    input_names.push_back(input_name);
  }

  return Status::OK();
}

Status ClipOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                  const NodeUnit& node_unit,
                                                  std::vector<std::string>&& input_names,
                                                  const logging::Logger& logger,
                                                  bool is_quantized_model,
                                                  bool do_op_validation) const {
  std::vector<std::string> param_tensor_names;
  Qnn_Scalar_t min_qnn_scalar = QNN_SCALAR_INIT;
  min_qnn_scalar.dataType = QNN_DATATYPE_FLOAT_32;
  min_qnn_scalar.floatValue = min_value_;
  QnnParamWrapper min_value_param(node_unit.Index(), node_unit.Name(), QNN_OP_RELU_MIN_MAX_PARAM_MIN_VALUE, min_qnn_scalar);
  param_tensor_names.push_back(min_value_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(min_value_param));

  Qnn_Scalar_t max_qnn_scalar = QNN_SCALAR_INIT;
  max_qnn_scalar.dataType = QNN_DATATYPE_FLOAT_32;
  max_qnn_scalar.floatValue = max_value_;
  QnnParamWrapper max_value_param(node_unit.Index(), node_unit.Name(), QNN_OP_RELU_MIN_MAX_PARAM_MAX_VALUE, max_qnn_scalar);
  param_tensor_names.push_back(max_value_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(max_value_param));

  ORT_RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit,
                                     std::move(input_names),
                                     std::move(param_tensor_names),
                                     logger, is_quantized_model, do_op_validation, GetQnnOpType(node_unit.OpType())));

  return Status::OK();
}

void CreateClipOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<ClipOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
