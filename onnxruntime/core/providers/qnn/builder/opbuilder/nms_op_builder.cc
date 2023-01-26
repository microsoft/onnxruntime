// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/framework/utils.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
namespace onnxruntime {
namespace qnn {
const int NMS_MIN_INPUT = 2;
const int NMS_MAX_INPUT = 5;
const int NMS_IOU_THRESHOLD_INPUT_INDEX = 3;
const int NMS_SCORE_THRESHOLD_INPUT_INDEX = 4;
class NMSOpBuilder : public BaseOpBuilder {
 public:
  NMSOpBuilder() : BaseOpBuilder("NMSOpBuilder") {}

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

  Status ProcessOutputs(QnnModelWrapper& qnn_model_wrapper,
                        const NodeUnit& node_unit,
                        std::vector<std::string>&& input_names,
                        std::vector<std::string>&& param_tensor_names,
                        const logging::Logger& logger,
                        bool is_quantized_model,
                        bool do_op_validation) const ORT_MUST_USE_RESULT;

 private:
  Status ExplictOpCheck(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit) const;

  Status IsInitializerInput(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit, size_t input_index) const {
    const std::vector<NodeUnitIODef>& inputs = node_unit.Inputs();

    // The fourth input iou_threshold, if exists must be an initializer
    if (inputs.size() >= input_index) {
      const auto& iou_threshold_input_name = node_unit.Inputs()[input_index].node_arg.Name();
      if (!qnn_model_wrapper.IsInitializerInput(iou_threshold_input_name)) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "The iou_threshold must be specified as constant input.");
      }
    }
    return Status::OK();
  }

  Status GetInitializerInput(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit, size_t input_index, float& initializer) const {
    initializer = 0.0f;
    const std::vector<NodeUnitIODef>& inputs = node_unit.Inputs();
    if (inputs.size() >= input_index) {
      const auto& initializer_input_name = node_unit.Inputs()[input_index].node_arg.Name();
      if (!qnn_model_wrapper.IsInitializerInput(initializer_input_name)) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "The iou_threshold/score_threshold must be specified as constant input.");
      }
      std::vector<uint8_t> unpacked_tensor;
      const auto& initializer_input_tensor = qnn_model_wrapper.GetInitializerTensors().at(initializer_input_name);
      ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(*initializer_input_tensor, unpacked_tensor));
      const float* init_data = reinterpret_cast<const float*>(unpacked_tensor.data());
      initializer = *init_data;
    }
    return Status::OK();
  }
};

Status NMSOpBuilder::ExplictOpCheck(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit) const {
  size_t input_count = node_unit.Inputs().size();
  ORT_RETURN_IF_NOT(input_count >= NMS_MIN_INPUT && input_count <= NMS_MAX_INPUT,
                    "For ONNX NonMaxSuppression operation the expected number of inputs is 2.");
  const std::vector<NodeUnitIODef>& inputs = node_unit.Inputs();

  ORT_RETURN_IF_ERROR(IsInitializerInput(qnn_model_wrapper, node_unit, NMS_IOU_THRESHOLD_INPUT_INDEX));

  NodeAttrHelper node_helper(node_unit);
  auto center_point_box = node_helper.Get("center_point_box", 0);
  if (0 != center_point_box) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN does not support center point box specification.");
  }
  return Status::OK();
}

Status NMSOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                   const NodeUnit& node_unit,
                                   const logging::Logger& logger,
                                   bool is_quantized_model,
                                   std::vector<std::string>& input_names,
                                   bool do_op_validation) const {
  if (do_op_validation) {
    ORT_RETURN_IF_ERROR(ExplictOpCheck(qnn_model_wrapper, node_unit));
  }

  Qnn_QuantizeParams_t quantize_param = QNN_QUANTIZE_PARAMS_INIT;
  InitializeQuantizeParam(quantize_param, is_quantized_model);
  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_UNDEFINED;

  return Status::OK();
}

Status NMSOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                 const NodeUnit& node_unit,
                                                 std::vector<std::string>&& input_names,
                                                 const logging::Logger& logger,
                                                 bool is_quantized_model,
                                                 bool do_op_validation) const {
  std::vector<std::string> param_tensor_names;
  float iou_threshold = 0.0f;
  ORT_RETURN_IF_ERROR(GetInitializerInput(qnn_model_wrapper, node_unit, NMS_IOU_THRESHOLD_INPUT_INDEX, iou_threshold));
  Qnn_Scalar_t iou_threshold_scalar = QNN_SCALAR_INIT;
  iou_threshold_scalar.dataType = QNN_DATATYPE_FLOAT_32;
  iou_threshold_scalar.floatValue = iou_threshold;
  QnnParamWrapper iou_threshold_param(node_unit.Index(), node_unit.Name(), qnn_def::iou_threshold, iou_threshold_scalar);
  param_tensor_names.push_back(iou_threshold_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(iou_threshold_param));

  float score_threshold = 0.0f;
  ORT_RETURN_IF_ERROR(GetInitializerInput(qnn_model_wrapper, node_unit, NMS_SCORE_THRESHOLD_INPUT_INDEX, score_threshold));
  Qnn_Scalar_t score_threshold_scalar = QNN_SCALAR_INIT;
  score_threshold_scalar.dataType = QNN_DATATYPE_FLOAT_32;
  score_threshold_scalar.floatValue = score_threshold;
  QnnParamWrapper score_threshold_param(node_unit.Index(), node_unit.Name(), qnn_def::score_threshold, score_threshold_scalar);
  param_tensor_names.push_back(score_threshold_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(score_threshold_param));

  ORT_RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit,
                                     std::move(input_names),
                                     std::move(param_tensor_names),
                                     logger, is_quantized_model, do_op_validation));

  return Status::OK();
}

Status NMSOpBuilder::ProcessOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     std::vector<std::string>&& param_tensor_names,
                                     const logging::Logger& logger,
                                     bool is_quantized_model,
                                     bool do_op_validation) const {
  return Status::OK();
}

void CreateNMSOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<NMSOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
