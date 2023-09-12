// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/common/safeint.h"
#include "core/util/qmath.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace qnn {

// Operator which only need to hanle node inputs & outputs, no attributes or no need to handle attributes
class SimpleOpBuilder : public BaseOpBuilder {
 public:
  SimpleOpBuilder() : BaseOpBuilder("SimpleOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SimpleOpBuilder);

 protected:
  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     const logging::Logger& logger,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;

 private:
  Status ExplicitOpCheck(const QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit) const;

  static constexpr std::array<std::string_view, 2> gridsample_supported_modes = {"bilinear", "nearest"};
  static constexpr std::array<std::string_view, 3> gridsample_supported_padding_modes = {"zeros", "border", "reflection"};
};

static int32_t GetDefaultAxisAttribute(const std::string& op_type, int opset_version) {
  if (op_type == "Softmax" || op_type == "LogSoftmax") {
    // Default axis changed from 1 to -1 in opset 13.
    return opset_version < 13 ? 1 : -1;
  }

  return 0;
}

Status SimpleOpBuilder::ExplicitOpCheck(const QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit) const {
  const std::string& op_type = node_unit.OpType();

  // QNN Softmax and LogSoftmax only support an axis value equal to input_rank - 1 (i.e., same as -1).
  if (op_type == "Softmax" || op_type == "LogSoftmax") {
    int32_t axis = GetDefaultAxisAttribute(op_type, node_unit.SinceVersion());
    Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
    ORT_RETURN_IF_ERROR(ProcessAxisAttribute(qnn_model_wrapper, node_unit, axis_qnn_scalar, axis));
    std::vector<uint32_t> input_shape;
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(node_unit.Inputs()[0].node_arg, input_shape),
                      "QNN EP: Cannot get shape for Softmax input");
    ORT_RETURN_IF(axis != static_cast<int32_t>(input_shape.size() - 1),
                  "QNN ", op_type.c_str(), " only supports an `axis` attribute equal to input_rank-1 (or -1)");
  }

  if (op_type == "GridSample") {
    NodeAttrHelper node_helper(node_unit);
    std::string mode = node_helper.Get("mode", "linear");
    ORT_RETURN_IF_NOT(utils::ArrayHasString(gridsample_supported_modes, mode), "GridSample does not support mode ",
                      mode.c_str());
    std::string padding_mode = node_helper.Get("padding_mode", "zeros");
    ORT_RETURN_IF_NOT(utils::ArrayHasString(gridsample_supported_padding_modes, padding_mode), "GridSample does not support padding_mode ",
                      padding_mode.c_str());
  }

  // ONNX's Min and Max operators accept a variable number of inputs (i.e., variadic).
  // However, QNN's Min and Max operators must take in exactly two inputs.
  if (op_type == "Min" || op_type == "Max") {
    ORT_RETURN_IF_NOT(node_unit.Inputs().size() == 2,
                      "QNN EP only supports Min and Max operators with exactly 2 inputs.");
  }

  return Status::OK();
}

Status ProcessAlphaAttribute(QnnModelWrapper& qnn_model_wrapper,
                             const NodeUnit& node_unit,
                             std::vector<std::string>& param_tensor_names) {
  NodeAttrHelper node_helper(node_unit);
  float alpha = node_helper.Get("alpha", 1.0f);
  Qnn_Scalar_t alpha_qnn_scalar = QNN_SCALAR_INIT;
  alpha_qnn_scalar.dataType = QNN_DATATYPE_FLOAT_32;
  alpha_qnn_scalar.floatValue = alpha;

  QnnParamWrapper alpha_param(node_unit.Index(), node_unit.Name(), QNN_OP_ELU_PARAM_ALPHA, alpha_qnn_scalar);
  param_tensor_names.push_back(alpha_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(alpha_param));

  return Status::OK();
}

Status ProcessBlockSizeAttribute(QnnModelWrapper& qnn_model_wrapper,
                                 const NodeUnit& node_unit,
                                 std::vector<std::string>& param_tensor_names) {
  NodeAttrHelper node_helper(node_unit);
  uint32_t block_size = node_helper.Get("blocksize", static_cast<uint32_t>(0));
  std::vector<uint32_t> block_size_shape{2};
  std::vector<uint32_t> block_size_data(2, block_size);
  QnnParamWrapper block_size_param(node_unit.Index(), node_unit.Name(), QNN_OP_DEPTH_TO_SPACE_PARAM_BLOCK_SIZE,
                                   std::move(block_size_shape), std::move(block_size_data));
  param_tensor_names.push_back(block_size_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(block_size_param));

  return Status::OK();
}

Status ProcessModeAttribute(QnnModelWrapper& qnn_model_wrapper,
                            const NodeUnit& node_unit,
                            std::vector<std::string>& param_tensor_names) {
  NodeAttrHelper node_helper(node_unit);
  std::string mode = node_helper.Get("mode", "DCR");
  Qnn_Scalar_t mode_qnn_scalar = QNN_SCALAR_INIT;
  mode_qnn_scalar.dataType = QNN_DATATYPE_UINT_32;
  if ("DCR" == mode) {
    mode_qnn_scalar.uint32Value = 0;
  } else if ("CRD" == mode) {
    mode_qnn_scalar.uint32Value = 1;  // CRD mode
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "DepthToSpace mode only support DCR & CRD.");
  }

  QnnParamWrapper mode_param(node_unit.Index(), node_unit.Name(), QNN_OP_DEPTH_TO_SPACE_PARAM_MODE, mode_qnn_scalar);
  param_tensor_names.push_back(mode_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(mode_param));

  return Status::OK();
}

// Process alpha attribute as input for Qnn LeakyRelu
Status ProcessAlphaAttributeAsInput(QnnModelWrapper& qnn_model_wrapper,
                                    const NodeUnit& node_unit,
                                    const std::string input_name) {
  NodeAttrHelper node_helper(node_unit);
  Qnn_QuantizeParams_t quantize_param = QNN_QUANTIZE_PARAMS_INIT;
  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;
  union {
    float alpha;
    uint8_t unpack[sizeof(float)];
  } tensor_data;
  tensor_data.alpha = node_helper.Get("alpha", 0.01f);
  std::vector<uint8_t> unpacked_data;
  // Check LeakyRelu input 0 to see if it's quantized tensor
  bool is_quantized_tensor = node_unit.Outputs()[0].quant_param.has_value();
  if (is_quantized_tensor) {
    float scale;
    uint8_t zero_point;
    int64_t num_of_elements = 1;
    concurrency::ThreadPool* thread_pool = nullptr;
    GetQuantizationParameter(&tensor_data.alpha, num_of_elements, scale, zero_point, thread_pool);
    unpacked_data.resize(1);
    ParQuantizeLinearStd(&tensor_data.alpha, unpacked_data.data(), num_of_elements, scale, zero_point, thread_pool);
    utils::InitializeQuantizeParam(quantize_param, is_quantized_tensor, scale, static_cast<int32_t>(zero_point));
    qnn_data_type = QNN_DATATYPE_UFIXED_POINT_8;
  } else {
    unpacked_data.assign(tensor_data.unpack, tensor_data.unpack + sizeof(float));
  }
  std::vector<uint32_t> input_shape{1};
  Qnn_TensorType_t tensor_type = QNN_TENSOR_TYPE_STATIC;
  QnnTensorWrapper input_tensorwrapper(input_name, tensor_type, qnn_data_type, quantize_param,
                                       std::move(input_shape), std::move(unpacked_data));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
  return Status::OK();
}

Status ProcessGridSampleAttributes(QnnModelWrapper& qnn_model_wrapper,
                                   const NodeUnit& node_unit,
                                   std::vector<std::string>& param_tensor_names) {
  NodeAttrHelper node_helper(node_unit);
  int64_t align_corners = node_helper.Get("align_corners", static_cast<int64_t>(0));
  Qnn_Scalar_t align_corners_qnn_scalar = QNN_SCALAR_INIT;
  align_corners_qnn_scalar.dataType = QNN_DATATYPE_BOOL_8;
  align_corners_qnn_scalar.bool8Value = static_cast<uint8_t>(align_corners == 0 ? 0 : 1);
  QnnParamWrapper align_corners_param(node_unit.Index(), node_unit.Name(), QNN_OP_GRID_SAMPLE_PARAM_ALIGN_CORNERS, align_corners_qnn_scalar);
  param_tensor_names.push_back(align_corners_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(align_corners_param));

  std::string mode = node_helper.Get("mode", "linear");
  Qnn_Scalar_t mode_qnn_scalar = QNN_SCALAR_INIT;
  mode_qnn_scalar.dataType = QNN_DATATYPE_UINT_32;
  if ("bilinear" == mode) {
    mode_qnn_scalar.uint32Value = QNN_OP_GRID_SAMPLE_MODE_BILINEAR;
  } else if ("nearest" == mode) {
    mode_qnn_scalar.uint32Value = QNN_OP_GRID_SAMPLE_MODE_NEAREST;
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "GridSample mode only support bilinear & nearest.");
  }
  QnnParamWrapper mode_param(node_unit.Index(), node_unit.Name(), QNN_OP_GRID_SAMPLE_PARAM_MODE, mode_qnn_scalar);
  param_tensor_names.push_back(mode_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(mode_param));

  std::string padding_mode = node_helper.Get("padding_mode", "zeros");
  Qnn_Scalar_t padding_mode_qnn_scalar = QNN_SCALAR_INIT;
  padding_mode_qnn_scalar.dataType = QNN_DATATYPE_UINT_32;
  if ("zeros" == padding_mode) {
    padding_mode_qnn_scalar.uint32Value = QNN_OP_GRID_SAMPLE_PADDING_MODE_ZEROS;
  } else if ("border" == padding_mode) {
    padding_mode_qnn_scalar.uint32Value = QNN_OP_GRID_SAMPLE_PADDING_MODE_BORDER;
  } else if ("reflection" == padding_mode) {
    padding_mode_qnn_scalar.uint32Value = QNN_OP_GRID_SAMPLE_PADDING_MODE_REFLECTION;
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "GridSample padding_mode only support zeros, border & reflection.");
  }
  QnnParamWrapper padding_mode_param(node_unit.Index(), node_unit.Name(), QNN_OP_GRID_SAMPLE_PARAM_PADDING_MODE, padding_mode_qnn_scalar);
  param_tensor_names.push_back(padding_mode_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(padding_mode_param));

  return Status::OK();
}

Status SimpleOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                    const NodeUnit& node_unit,
                                                    std::vector<std::string>&& input_names,
                                                    const logging::Logger& logger,
                                                    bool do_op_validation) const {
  if (input_names.size() < 1) {
    return Status::OK();
  }

  const std::string& op_type = node_unit.OpType();

  if (do_op_validation) {
    ORT_RETURN_IF_ERROR(ExplicitOpCheck(qnn_model_wrapper, node_unit));
    // Skip the op validation for DepthToSpace & SpaceToDepth if it's not NHWC data layout
    if (node_unit.Domain() != kMSInternalNHWCDomain && (op_type == "DepthToSpace" || op_type == "SpaceToDepth" || op_type == "GridSample")) {
      return Status::OK();
    }
  }

  std::vector<std::string> param_tensor_names;
  // Add attribute
  if (op_type == "LogSoftmax" || op_type == "Softmax" || op_type == "Concat") {
    int32_t default_axis = GetDefaultAxisAttribute(op_type, node_unit.SinceVersion());
    Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
    ORT_RETURN_IF_ERROR(ProcessAxisAttribute(qnn_model_wrapper, node_unit, axis_qnn_scalar, default_axis));
    QnnParamWrapper axis_param(node_unit.Index(), node_unit.Name(), QNN_OP_SOFTMAX_PARAM_AXIS, axis_qnn_scalar);
    param_tensor_names.push_back(axis_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(axis_param));
  }

  if (op_type == "MatMul") {
    Qnn_Scalar_t scalar_param = QNN_SCALAR_INIT;
    scalar_param.dataType = QNN_DATATYPE_BOOL_8;
    scalar_param.bool8Value = 0;
    QnnParamWrapper transpose_in0_param(node_unit.Index(), node_unit.Name(), QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0, scalar_param);
    param_tensor_names.push_back(transpose_in0_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(transpose_in0_param));

    QnnParamWrapper transpose_in1_param(node_unit.Index(), node_unit.Name(), QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1, scalar_param);
    param_tensor_names.push_back(transpose_in1_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(transpose_in1_param));
  }

  if (op_type == "LeakyRelu") {
    std::string input_name = "alpha";
    ORT_RETURN_IF_ERROR(ProcessAlphaAttributeAsInput(qnn_model_wrapper, node_unit, input_name));
    input_names.push_back(input_name);
  }

  if (op_type == "Elu") {
    ORT_RETURN_IF_ERROR(ProcessAlphaAttribute(qnn_model_wrapper, node_unit, param_tensor_names));
  }

  if (op_type == "DepthToSpace") {
    ORT_RETURN_IF_ERROR(ProcessBlockSizeAttribute(qnn_model_wrapper, node_unit, param_tensor_names));
    ORT_RETURN_IF_ERROR(ProcessModeAttribute(qnn_model_wrapper, node_unit, param_tensor_names));
  }

  if (op_type == "SpaceToDepth") {
    ORT_RETURN_IF_ERROR(ProcessBlockSizeAttribute(qnn_model_wrapper, node_unit, param_tensor_names));
  }

  if (op_type == "GridSample") {
    ORT_RETURN_IF_ERROR(ProcessGridSampleAttributes(qnn_model_wrapper, node_unit, param_tensor_names));
  }

  ORT_RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit,
                                     std::move(input_names),
                                     std::move(param_tensor_names),
                                     logger, do_op_validation, GetQnnOpType(op_type)));
  return Status::OK();
}

void CreateSimpleOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<SimpleOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
