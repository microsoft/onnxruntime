// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/common/safeint.h"
#include "onnx/defs/data_type_utils.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace qnn {

class PoolOpBuilder : public BaseOpBuilder {
 public:
  PoolOpBuilder() : BaseOpBuilder("PoolOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(PoolOpBuilder);

  Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger) const override final ORT_MUST_USE_RESULT;

 protected:
  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     const logging::Logger& logger,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;
  Status OverrideOutputQuantParam(QnnModelWrapper& qnn_model_wrapper,
                                  const NodeUnit& node_unit,
                                  const logging::Logger& logger,
                                  const std::vector<std::string>& input_names,
                                  size_t output_index,
                                  Qnn_DataType_t qnn_data_type,
                                  Qnn_QuantizeParams_t& quant_param) const override ORT_MUST_USE_RESULT;

 private:
  Status SetCommonPoolParams(const NodeAttrHelper& node_helper, std::vector<uint32_t>& filter_size,
                             std::vector<uint32_t>& pad_amount, std::vector<uint32_t>& stride,
                             int32_t& ceil_mode,
                             std::vector<uint32_t>&& input_shape,
                             std::vector<uint32_t>&& output_shape) const;
};

// Pool ops are sensitive with data layout, no special validation so far
// The nodes from 1st call of GetCapability do not get layout transformer applied, it's still NCHW
// The nodes from 2nd call of GetCapability get layout transformer applied, it's NHWC
// Need to do op validation in 1st call of GetCapability
// TODO: Check if node domain == kMSInternalNHWCDomain to determine if the layout has been transformed.
Status PoolOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                    const NodeUnit& node_unit,
                                    const logging::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);

  if (node_unit.Domain() == kMSInternalNHWCDomain) {  // Use QNN validation API if layout is NHWC.
    return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, true);
  }

  const auto& inputs = node_unit.Inputs();
  ORT_RETURN_IF_ERROR(DataTypeCheckForCpuBackend(qnn_model_wrapper, inputs[0].node_arg.Type()));

  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].node_arg, input_shape), "Cannot get shape");
  if (input_shape.size() != 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN Pool2D only support 2D!");
  }

  if (node_unit.Outputs().size() > 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN only support 1 output!");
  }

  const std::string& op_type = node_unit.OpType();
  // Onnx GlobalMaxPool doesn't have any attributes
  if (op_type == "GlobalMaxPool") {
    return Status::OK();
  }

  NodeAttrHelper node_helper(node_unit);
  auto dilation_values = node_helper.Get("dilations", std::vector<int32_t>{1, 1});
  if (dilation_values != std::vector<int32_t>{1, 1}) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN does not support Dilation attribute");
  }

  if (op_type == "MaxPool" || op_type == "AveragePool") {
    auto auto_pad = node_helper.Get("auto_pad", std::string("NOTSET"));
    ORT_RETURN_IF(auto_pad != "NOTSET" && auto_pad != "SAME_LOWER" && auto_pad != "SAME_UPPER",
                  "QNN Pool operators do not support 'auto_pad' value: ", auto_pad.c_str());
  }

  return Status::OK();
}

Status PoolOpBuilder::SetCommonPoolParams(const NodeAttrHelper& node_helper,
                                          std::vector<uint32_t>& filter_size,
                                          std::vector<uint32_t>& pad_amount, std::vector<uint32_t>& strides,
                                          int32_t& ceil_mode,
                                          std::vector<uint32_t>&& input_shape,
                                          std::vector<uint32_t>&& output_shape) const {
  filter_size = node_helper.Get("kernel_shape", std::vector<uint32_t>{1, 1});
  ORT_RETURN_IF_NOT(filter_size.size() == 2, "QNN only support kernel_shape with shape[2].");

  strides = node_helper.Get("strides", std::vector<uint32_t>{1, 1});
  ORT_RETURN_IF_NOT(strides.size() == 2, "QNN only support strides with shape[2].");

  pad_amount = node_helper.Get("pads", std::vector<uint32_t>{0, 0, 0, 0});
  auto auto_pad = node_helper.Get("auto_pad", std::string("NOTSET"));
  ORT_RETURN_IF(auto_pad != "NOTSET" && auto_pad != "SAME_LOWER" && auto_pad != "SAME_UPPER",
                "QNN Pool operators do not support 'auto_pad' value: ", auto_pad.c_str());

  if (auto_pad.compare("NOTSET") != 0) {
    std::vector<uint32_t> dilations = node_helper.Get("dilations", std::vector<uint32_t>{1, 1});

    auto total_pads_0 = (output_shape[1] - 1) * strides[0] + (filter_size[0] - 1) * dilations[0] + 1 - input_shape[1];
    auto total_pads_1 = (output_shape[2] - 1) * strides[1] + (filter_size[1] - 1) * dilations[1] + 1 - input_shape[2];
    if (auto_pad.compare("SAME_LOWER") != 0) {
      pad_amount[0] = total_pads_0 / 2;
      pad_amount[1] = total_pads_1 / 2;
      pad_amount[2] = total_pads_0 - pad_amount[0];
      pad_amount[3] = total_pads_1 - pad_amount[1];
    } else if (auto_pad.compare("SAME_UPPER") != 0) {
      pad_amount[2] = total_pads_0 / 2;
      pad_amount[3] = total_pads_1 / 2;
      pad_amount[0] = total_pads_0 - pad_amount[2];
      pad_amount[1] = total_pads_1 - pad_amount[3];
    }
  }
  ORT_RETURN_IF_NOT(pad_amount.size() == 4, "QNN only support pads with shape[2, 2].");
  ReArranagePads(pad_amount);

  ceil_mode = node_helper.Get("ceil_mode", ceil_mode);
  return Status::OK();
}  // namespace qnn

void SetPoolParam(const NodeUnit& node_unit,
                  const std::string& param_name,
                  std::vector<uint32_t>&& parm_shape,
                  std::vector<uint32_t>&& parm_data,
                  std::vector<std::string>& param_tensor_names,
                  QnnModelWrapper& qnn_model_wrapper) {
  QnnParamWrapper qnn_param(node_unit.Index(),
                            node_unit.Name(),
                            param_name,
                            std::move(parm_shape),
                            std::move(parm_data));
  param_tensor_names.push_back(qnn_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(qnn_param));
}

Status PoolOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                  const NodeUnit& node_unit,
                                                  std::vector<std::string>&& input_names,
                                                  const logging::Logger& logger,
                                                  bool do_op_validation) const {
  NodeAttrHelper node_helper(node_unit);
  // Get the NCHW from input data, use HW for the pool filter size and pool stride
  const auto& inputs = node_unit.Inputs();
  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].node_arg, input_shape), "Cannot get shape");
  ORT_RETURN_IF_NOT(input_shape.size() == 4, "Input should have 4 dimension NCHW.");
  // Default value for GlobalAveragePool
  // Pool use filter & stride with shape [filter_height, filter_width]
  // With layout transformer, the input has shape  [batch, height, width, channel],
  std::vector<uint32_t> filter_size(input_shape.begin() + 1, input_shape.begin() + 3);
  std::vector<uint32_t> stride(filter_size);
  std::vector<uint32_t> filter_size_dim{2};
  std::vector<uint32_t> stride_dim{2};
  std::vector<uint32_t> pad_amount{0, 0, 0, 0};
  std::vector<uint32_t> pad_amount_dim{2, 2};
  int32_t ceil_mode = 0;

  std::vector<std::string> param_tensor_names;
  const std::string& op_type = node_unit.OpType();
  if (op_type == "GlobalMaxPool") {
    // set default params for Qnn PoolMax2D
    SetPoolParam(node_unit, QNN_OP_POOL_MAX_2D_PARAM_FILTER_SIZE, std::move(filter_size_dim), std::move(filter_size), param_tensor_names, qnn_model_wrapper);
    SetPoolParam(node_unit, QNN_OP_POOL_MAX_2D_PARAM_PAD_AMOUNT, std::move(pad_amount_dim), std::move(pad_amount), param_tensor_names, qnn_model_wrapper);
    SetPoolParam(node_unit, QNN_OP_POOL_MAX_2D_PARAM_STRIDE, std::move(stride_dim), std::move(stride), param_tensor_names, qnn_model_wrapper);

    ORT_RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit,
                                       std::move(input_names),
                                       std::move(param_tensor_names),
                                       logger,
                                       do_op_validation,
                                       GetQnnOpType(op_type)));
    return Status::OK();
  }

  if (op_type == "MaxPool" || op_type == "AveragePool") {
    const auto& outputs = node_unit.Outputs();
    std::vector<uint32_t> output_shape;
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(outputs[0].node_arg, output_shape), "Cannot get shape");

    ORT_RETURN_IF_ERROR(SetCommonPoolParams(node_helper, filter_size, pad_amount, stride, ceil_mode,
                                            std::move(input_shape), std::move(output_shape)));
  }

  SetPoolParam(node_unit, QNN_OP_POOL_MAX_2D_PARAM_FILTER_SIZE, std::move(filter_size_dim), std::move(filter_size), param_tensor_names, qnn_model_wrapper);
  SetPoolParam(node_unit, QNN_OP_POOL_MAX_2D_PARAM_PAD_AMOUNT, std::move(pad_amount_dim), std::move(pad_amount), param_tensor_names, qnn_model_wrapper);
  SetPoolParam(node_unit, QNN_OP_POOL_MAX_2D_PARAM_STRIDE, std::move(stride_dim), std::move(stride), param_tensor_names, qnn_model_wrapper);

  if (0 != ceil_mode) {
    Qnn_Scalar_t rounding_mode_param = QNN_SCALAR_INIT;
    rounding_mode_param.dataType = QNN_DATATYPE_UINT_32;
    rounding_mode_param.int32Value = ceil_mode;
    QnnParamWrapper rounding_mode_param_wrapper(node_unit.Index(),
                                                node_unit.Name(),
                                                QNN_OP_POOL_MAX_2D_PARAM_ROUNDING_MODE,
                                                rounding_mode_param);
    param_tensor_names.push_back(rounding_mode_param_wrapper.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(rounding_mode_param_wrapper));
  }
  if (op_type == "GlobalAveragePool") {
    Qnn_Scalar_t scalar_param = QNN_SCALAR_INIT;
    scalar_param.dataType = QNN_DATATYPE_BOOL_8;
    scalar_param.bool8Value = 1;
    QnnParamWrapper count_pad_for_edges_param(node_unit.Index(),
                                              node_unit.Name(),
                                              QNN_OP_POOL_AVG_2D_PARAM_COUNT_PAD_FOR_EDGES,
                                              scalar_param);
    param_tensor_names.push_back(count_pad_for_edges_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(count_pad_for_edges_param));
  } else if (op_type == "AveragePool") {
    Qnn_Scalar_t scalar_param = QNN_SCALAR_INIT;
    scalar_param.dataType = QNN_DATATYPE_BOOL_8;
    scalar_param.bool8Value = static_cast<uint8_t>(node_helper.Get("count_include_pad", static_cast<int64_t>(0)) != 0);
    QnnParamWrapper count_pad_for_edges_param(node_unit.Index(),
                                              node_unit.Name(),
                                              QNN_OP_POOL_AVG_2D_PARAM_COUNT_PAD_FOR_EDGES,
                                              scalar_param);
    param_tensor_names.push_back(count_pad_for_edges_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(count_pad_for_edges_param));
  }

  ORT_RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit,
                                     std::move(input_names),
                                     std::move(param_tensor_names),
                                     logger,
                                     do_op_validation,
                                     GetQnnOpType(op_type)));

  return Status::OK();
}

Status PoolOpBuilder::OverrideOutputQuantParam(QnnModelWrapper& qnn_model_wrapper,
                                               const NodeUnit& node_unit,
                                               const logging::Logger& logger,
                                               const std::vector<std::string>& input_names,
                                               size_t output_index,
                                               Qnn_DataType_t qnn_data_type,
                                               Qnn_QuantizeParams_t& quant_param) const {
  // Force MaxPool outputs to use the same quantization parameters as the input.
  if (node_unit.OpType() == "MaxPool") {
    return SetOutputQParamEqualToInput(qnn_model_wrapper, node_unit, logger, input_names,
                                       0 /*input_index*/, output_index, qnn_data_type, quant_param);
  }

  return Status::OK();
}

void CreatePoolOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<PoolOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
