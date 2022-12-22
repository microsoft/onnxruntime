// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/framework/utils.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
namespace onnxruntime {
namespace qnn {
const int TOPK_MIN_INPUT = 2;
const int TOPK_MAX_INPUT = 2;
class TopKOpBuilder : public BaseOpBuilder {
 public:
  TopKOpBuilder() : BaseOpBuilder("TopKOpBuilder") {}

 protected:
  Qnn_DataType_t GetSupportedOutputDataType(size_t index, Qnn_DataType_t qnn_data_type) const {
    if (index == 1) {
      if (qnn_data_type == QNN_DATATYPE_INT_64) {
        return QNN_DATATYPE_INT_32;
      } else if (qnn_data_type == QNN_DATATYPE_UINT_64) {
        return QNN_DATATYPE_UINT_32;
      }
    }
    return qnn_data_type;
  }

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
};

Status TopKOpBuilder::ExplictOpCheck(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit) const {
  size_t input_count = node_unit.Inputs().size();
  ORT_RETURN_IF_NOT(input_count >= TOPK_MIN_INPUT && input_count <= TOPK_MAX_INPUT,
                    "For ONNX TopK operation the expected number of inputs is 2.");
  // Skip the first input. The second input needs to be an initializer.
  const auto& input_1 = node_unit.Inputs()[1].node_arg.Name();
  if (!qnn_model_wrapper.IsInitializerInput(input_1)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "The number of top elements to retrieve must be specified as constant input.");
  }
  NodeAttrHelper node_helper(node_unit);
  auto largest = node_helper.Get("largest", 1);
  auto sorted = node_helper.Get("sorted", 1);
  if (0 == sorted) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN TopK output is always sorted");
  }
  if (0 == largest) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN TopK output is always largest values");
  }
  auto& input_0 = node_unit.Inputs()[0];
  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input_0.node_arg, input_shape), "Cannot get shape");
  auto rank = input_shape.size();
  auto axis = node_helper.Get("axis", -1);

  if (-1 == axis && axis != static_cast<int32_t>(rank - 1)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN TopK axis is always the last dimension");
  }
  return Status::OK();
}

Status TopKOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
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

  auto& input_0 = node_unit.Inputs()[0];
  auto& input_name = input_0.node_arg.Name();
  if (qnn_model_wrapper.IsQnnTensorWrapperExist(input_name)) {
    LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << input_name;
    input_names.push_back(input_name);
    return Status::OK();
  }

  const auto* type_proto = input_0.node_arg.TypeAsProto();
  ORT_RETURN_IF_ERROR(GetQnnDataType(is_quantized_model, type_proto, qnn_data_type));

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

  QnnTensorWrapper input_tensorwrapper(input_name, tensor_type, qnn_data_type, quantize_param,
                                       std::move(input_shape), std::move(unpacked_tensor));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
  return Status::OK();
}

Status TopKOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                  const NodeUnit& node_unit,
                                                  std::vector<std::string>&& input_names,
                                                  const logging::Logger& logger,
                                                  bool is_quantized_model,
                                                  bool do_op_validation) const {
  auto& input_name = node_unit.Inputs()[1].node_arg.Name();
  uint32_t k = 0;  // The number of elements to extract from the input tensor at each position.
  bool is_initializer_input = qnn_model_wrapper.IsInitializerInput(input_name);
  if (is_initializer_input) {
    std::vector<uint8_t> unpacked_tensor;
    const auto& input_tensor = qnn_model_wrapper.GetInitializerTensors().at(input_name);
    ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(*input_tensor, unpacked_tensor));
    const int64_t* tensor_data = reinterpret_cast<const int64_t*>(unpacked_tensor.data());
    k = static_cast<uint32_t>(*tensor_data);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN TopK operator requires constant input parameter k.");
  }
  Qnn_Scalar_t qnn_scalar_k = QNN_SCALAR_INIT;
  qnn_scalar_k.dataType = QNN_DATATYPE_UINT_32;
  qnn_scalar_k.uint32Value = k;
  QnnParamWrapper k_param(node_unit.Index(), node_unit.Name(), qnn_def::topk, qnn_scalar_k);
  std::string k_param_name = k_param.GetParamTensorName();
  qnn_model_wrapper.AddParamWrapper(std::move(k_param));
  std::vector<std::string> param_tensor_names{k_param_name};
  ORT_RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit, std::move(input_names), std::move(param_tensor_names),
                                     logger, is_quantized_model, do_op_validation));
  return Status::OK();
}

void CreateTopKOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<TopKOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
