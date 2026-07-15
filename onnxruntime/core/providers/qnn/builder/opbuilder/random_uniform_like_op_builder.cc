// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

class RandomUniformLikeOpBuilder : public BaseOpBuilder {
 public:
  RandomUniformLikeOpBuilder() : BaseOpBuilder("RandomUniformLikeOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(RandomUniformLikeOpBuilder);

 protected:
  Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger,
                       std::vector<std::string>& input_names,
                       bool do_op_validation) const override;

  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     const logging::Logger& logger,
                                     bool do_op_validation) const override;
};

Status RandomUniformLikeOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                                 const NodeUnit& node_unit,
                                                 const logging::Logger& logger,
                                                 std::vector<std::string>& input_names,
                                                 bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);
  ORT_UNUSED_PARAMETER(logger);
  NodeAttrHelper node_helper(node_unit);
  const auto& inputs = node_unit.Inputs();
  const auto& input_tensor = inputs[0];
  const std::string& input_tensor_name = input_tensor.node_arg.Name();

  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input_tensor.node_arg, input_shape),
                    "Failed to get shape for input tensor: ", input_tensor_name);

  const std::string shape_tensor_name = utils::GetUniqueName(input_tensor_name, "_shape");
  std::vector<uint8_t> shape_data(input_shape.size() * sizeof(uint32_t));
  memcpy(shape_data.data(), input_shape.data(), shape_data.size());
  std::vector<uint32_t> shape_tensor_shape = {static_cast<uint32_t>(input_shape.size())};

  QnnTensorWrapper shape_tensor_wrapper(shape_tensor_name,
                                        QNN_TENSOR_TYPE_STATIC,
                                        QNN_DATATYPE_UINT_32,
                                        QnnQuantParamsWrapper(),
                                        std::move(shape_tensor_shape),
                                        std::move(shape_data));

  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(shape_tensor_wrapper)),
                    "Failed to add shape tensor.");

  input_names.push_back(shape_tensor_name);

  // --- If seed attribute is present, add it as second input ---
  if (node_helper.HasAttr("seed")) {
    auto seed_value = node_helper.GetFloat("seed");

    std::vector<uint32_t> scalar_shape = {1};
    std::vector<uint8_t> seed_data(sizeof(float));
    memcpy(seed_data.data(), &seed_value, sizeof(float));

    const std::string seed_tensor_name = utils::GetUniqueName(input_tensor_name, "_ort_qnn_ep_seed");

    QnnTensorWrapper seed_tensor(seed_tensor_name, QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_32,
                                 QnnQuantParamsWrapper(), std::move(scalar_shape), std::move(seed_data));

    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(seed_tensor)),
                      "Failed to add seed tensor");

    input_names.push_back(seed_tensor_name);  // Seed is always second
  }
  return Status::OK();
}

Status RandomUniformLikeOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                               const NodeUnit& node_unit,
                                                               std::vector<std::string>&& input_names,
                                                               const logging::Logger& logger,
                                                               bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);
  NodeAttrHelper node_helper(node_unit);
  std::vector<std::string> param_names;

  // Extract 'low' attribute
  float low = node_helper.Get("low", 0.0f);
  Qnn_Scalar_t low_param = QNN_SCALAR_INIT;
  low_param.dataType = QNN_DATATYPE_FLOAT_32;
  low_param.floatValue = low;
  QnnParamWrapper low_param_wrapper(node_unit.Index(),
                                    node_unit.Name(),
                                    QNN_OP_RANDOM_UNIFORM_LIKE_PARAM_LOW,
                                    low_param);

  param_names.push_back(low_param_wrapper.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(low_param_wrapper));

  // Extract 'high' attribute
  float high = node_helper.Get("high", 1.0f);
  Qnn_Scalar_t high_param = QNN_SCALAR_INIT;
  high_param.dataType = QNN_DATATYPE_FLOAT_32;
  high_param.floatValue = high;
  QnnParamWrapper high_param_wrapper(node_unit.Index(),
                                     node_unit.Name(),
                                     QNN_OP_RANDOM_UNIFORM_LIKE_PARAM_HIGH,
                                     high_param);

  param_names.push_back(high_param_wrapper.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(high_param_wrapper));

  const auto& outputs = node_unit.Outputs();
  const std::string& output_name = outputs[0].node_arg.Name();

  bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);
  Qnn_TensorType_t tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

  // Get output info
  TensorInfo output_info{};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(outputs[0], output_info));

  // Determine if we need to use UFIXED_POINT_8 and add a dequantize node
  bool is_npu_backend = IsNpuBackend(qnn_model_wrapper.GetQnnBackendType());
  bool need_dequantize = is_npu_backend;

  if (need_dequantize) {
    // Create an intermediate tensor with UFIXED_POINT_8 data type
    const std::string intermediate_output_name = utils::GetUniqueName(output_name, "_uint8");

    // Calculate quantization parameters based on low and high values
    QnnQuantParamsWrapper quantize_param;
    float scale = 0.0f;
    int32_t zero_point = 0;
    ORT_RETURN_IF_ERROR(utils::GetQuantParams(low, high, QNN_DATATYPE_UFIXED_POINT_8, scale, zero_point));
    quantize_param = QnnQuantParamsWrapper(scale, zero_point);
    QnnTensorWrapper intermediate_output_wrapper(intermediate_output_name,
                                                 QNN_TENSOR_TYPE_NATIVE,
                                                 QNN_DATATYPE_UFIXED_POINT_8,
                                                 std::move(quantize_param),
                                                 std::vector<uint32_t>(output_info.shape));

    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(intermediate_output_wrapper)),
                      "Failed to add intermediate output tensor.");

    // Create the RandomUniformLike node with uint8 output
    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                          utils::GetUniqueName(node_unit),
                          QNN_OP_PACKAGE_NAME_QTI_AISW,
                          QNN_OP_RANDOM_UNIFORM_LIKE,
                          std::move(input_names),
                          {intermediate_output_name},
                          std::move(param_names),
                          do_op_validation),
                      "Failed to create RandomUniformLike node.");

    // Create the final output tensor with the original data type
    QnnTensorWrapper output_wrapper(output_name,
                                    tensor_type,
                                    output_info.qnn_data_type,
                                    output_info.quant_param.Copy(),
                                    std::vector<uint32_t>(output_info.shape));

    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_wrapper)),
                      "Failed to add output tensor.");

    // Create a Dequantize node to convert from uint8 to float32
    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                          utils::GetUniqueName(node_unit, "_dequantize"),
                          QNN_OP_PACKAGE_NAME_QTI_AISW,
                          QNN_OP_DEQUANTIZE,
                          {intermediate_output_name},
                          {output_name},
                          {},
                          do_op_validation),
                      "Failed to create Dequantize node.");
  } else {
    // For non-NPU backends, use the original data type directly
    QnnTensorWrapper output_wrapper(output_name,
                                    tensor_type,
                                    output_info.qnn_data_type,
                                    output_info.quant_param.Copy(),
                                    std::vector<uint32_t>(output_info.shape));

    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_wrapper)),
                      "Failed to add output tensor.");

    // Create the RandomUniformLike node with the original data type output
    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                          utils::GetUniqueName(node_unit),
                          QNN_OP_PACKAGE_NAME_QTI_AISW,
                          QNN_OP_RANDOM_UNIFORM_LIKE,
                          std::move(input_names),
                          {output_name},
                          std::move(param_names),
                          do_op_validation),
                      "Failed to create RandomUniformLike node.");
  }

  return Status::OK();
}

void CreateRandomUniformLikeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<RandomUniformLikeOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime