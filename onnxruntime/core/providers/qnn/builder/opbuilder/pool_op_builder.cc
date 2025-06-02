// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"

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
                                  QnnQuantParamsWrapper& quant_param) const override ORT_MUST_USE_RESULT;

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

  bool is1d = (input_shape.size() == 3);
  if (!is1d && input_shape.size() != 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN Pool only supports rank 3 or 4!");
  }

  NodeAttrHelper node_helper(node_unit);

  if (is1d) {
    auto kernel_shape = node_helper.Get("kernel_shape", std::vector<int32_t>{});
    ORT_RETURN_IF_NOT(kernel_shape.size() == 1, "QNN Pool1D: kernel_shape must have length 1!");

    auto pads = node_helper.Get("pads", std::vector<int32_t>{});
    ORT_RETURN_IF_NOT(pads.size() == 2, "QNN Pool1D: pads must have length 2!");

    auto strides = node_helper.Get("strides", std::vector<int32_t>{});
    ORT_RETURN_IF_NOT(strides.empty() || strides.size() == 1, "QNN Pool1D: strides must have length 1 or omitted!");

    auto dilations = node_helper.Get("dilations", std::vector<int32_t>{1});
    ORT_RETURN_IF_NOT(dilations.size() == 1, "QNN Pool1D: dilations must have length 1 or omitted!");
  } else {
    auto dilations = node_helper.Get("dilations", std::vector<int32_t>{1, 1});
    ORT_RETURN_IF_NOT(dilations.size() == 2, "QNN Pool2D: dilations must have length 2 or omitted!");
  }

  if (node_unit.Outputs().size() > 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN only support 1 output!");
  }

  const std::string& op_type = node_unit.OpType();
  // Onnx GlobalMaxPool doesn't have any attributes
  if (op_type == "GlobalMaxPool") {
    return Status::OK();
  }

  if (op_type == "MaxPool" || op_type == "AveragePool") {
    auto auto_pad = node_helper.Get("auto_pad", std::string("NOTSET"));
    ORT_RETURN_IF(auto_pad != "NOTSET" && auto_pad != "SAME_LOWER" && auto_pad != "SAME_UPPER" && auto_pad != "VALID",
                  "QNN Pool operators do not support 'auto_pad' value: ", auto_pad.c_str());
  }

  return Status::OK();
}

static std::vector<uint32_t> AmendOutputShapeForRank3Pool(
    gsl::span<const uint32_t> input_shape,   // {N, H, W, C}
    gsl::span<const uint32_t> kernel_shape,  // {k_h, k_w}
    gsl::span<const uint32_t> strides,       // {s_h, s_w}
    gsl::span<const uint32_t> pads) {
  assert(input_shape.size() == 4 &&
         kernel_shape.size() == 2 &&
         strides.size() == 2 &&
         pads.size() == 4);

  const uint32_t N = input_shape[0];
  const uint32_t H = input_shape[1];
  const uint32_t W = input_shape[2];
  const uint32_t C = input_shape[3];

  // pad the spatial dims
  uint32_t padded_H = H + pads[0] + pads[2];
  uint32_t padded_W = W + pads[1] + pads[3];

  // floor-mode on NHWC
  uint32_t out_H = (padded_H < kernel_shape[0])
                       ? 0
                       : (padded_H - kernel_shape[0]) / strides[0] + 1;
  uint32_t out_W = (padded_W < kernel_shape[1])
                       ? 0
                       : (padded_W - kernel_shape[1]) / strides[1] + 1;

  return {N, out_H, out_W, C};
}

Status PoolOpBuilder::SetCommonPoolParams(const NodeAttrHelper& node_helper,
                                          std::vector<uint32_t>& filter_size,
                                          std::vector<uint32_t>& pad_amount, std::vector<uint32_t>& strides,
                                          int32_t& ceil_mode,
                                          std::vector<uint32_t>&& input_shape,
                                          std::vector<uint32_t>&& output_shape) const {
  {
    auto raw_filter_size = node_helper.Get("kernel_shape", std::vector<uint32_t>{1, 1});
    if (raw_filter_size.size() == 1) {
      filter_size = {1, raw_filter_size[0]};
    } else {
      filter_size = raw_filter_size;
    }
  }
  ORT_RETURN_IF_NOT(filter_size.size() == 2,
                    "QNN only support kernel_shape with shape[2].");

  {
    auto raw_strides = node_helper.Get("strides", std::vector<uint32_t>{1, 1});
    if (raw_strides.size() == 1) {
      strides = {1, raw_strides[0]};
    } else {
      strides = raw_strides;
    }
  }
  ORT_RETURN_IF_NOT(strides.size() == 2,
                    "QNN only support strides with shape[2].");

  {
    auto raw_pad_amount = node_helper.Get("pads", std::vector<uint32_t>{0, 0, 0, 0});
    if (raw_pad_amount.size() == 2) {
      pad_amount = {0, raw_pad_amount[0], 0, raw_pad_amount[1]};
    } else {
      pad_amount = raw_pad_amount;
    }
  }

  auto auto_pad = node_helper.Get("auto_pad", std::string("NOTSET"));
  ORT_RETURN_IF(auto_pad != "NOTSET" && auto_pad != "SAME_LOWER" && auto_pad != "SAME_UPPER" && auto_pad != "VALID",
                "QNN Pool operators do not support 'auto_pad' value: ", auto_pad.c_str());

  if (auto_pad.compare("NOTSET") != 0) {
    std::vector<uint32_t> dilations;
    auto raw_dilations = node_helper.Get("dilations", std::vector<uint32_t>{1, 1});
    if (raw_dilations.size() == 1) {
      dilations = {1, raw_dilations[0]};
    } else {
      dilations = raw_dilations;
    }

    // Max Pool rank 3 input
    if (output_shape.size() == 3) {
      // Calculate MaxPool output for rank-4 when input is rank 3
      output_shape = AmendOutputShapeForRank3Pool(input_shape,
                                                  filter_size,
                                                  strides,
                                                  pad_amount);
    }
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

  const auto& reshape_input = node_unit.Inputs()[0];
  TensorInfo reshape_input_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(reshape_input, reshape_input_info));

  bool needs_reshape = false;
  const std::string reshape4d = input_names[0] + "_pre_reshape";
  if (input_shape.size() == 3) {
    needs_reshape = true;
    // build new_shape = {N, 1, C, L}
    std::vector<uint32_t> new_shape = {
        input_shape[0],
        1,
        input_shape[1],
        input_shape[2]};

    const std::string reshape_node_name = "pre_reshape";
    QnnTensorWrapper rw(
        reshape4d,
        QNN_TENSOR_TYPE_NATIVE,
        reshape_input_info.qnn_data_type,
        reshape_input_info.quant_param.Copy(),
        std::move(new_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(rw)),
                      "Failed to add reshape-4d tensor.");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                          reshape_node_name,
                          QNN_OP_PACKAGE_NAME_QTI_AISW,
                          "Reshape",
                          {input_names[0]},
                          {reshape4d},
                          {},
                          do_op_validation),
                      "Failed to create reshape-4d node.");
    input_names[0] = reshape4d;
    input_shape = {input_shape[0], 1, input_shape[1], input_shape[2]};
  }

  ORT_RETURN_IF_NOT(input_shape.size() == 4, "Input should have 4 dims NCHW or 3 dims for 1D pooling.");
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

  std::vector<uint32_t> onnx_in_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].node_arg, onnx_in_shape), "Cannot get shape");
  // Reshaped input rank-4 for MaxPool
  if (onnx_in_shape.size() == 3) {
    onnx_in_shape = {onnx_in_shape[0],
                     1,
                     onnx_in_shape[1],
                     onnx_in_shape[2]};
  }

  // Calculate MaxPool output for rank-4 when input is rank 3
  auto pooled_shape = AmendOutputShapeForRank3Pool(onnx_in_shape,
                                                   filter_size,
                                                   stride,
                                                   pad_amount);

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

  if (!needs_reshape) {
    ORT_RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit,
                                       std::move(input_names),
                                       std::move(param_tensor_names),
                                       logger,
                                       do_op_validation,
                                       GetQnnOpType(op_type)));

    return Status::OK();
  }
  const auto& outputs = node_unit.Outputs();
  const std::string real_out = outputs[0].node_arg.Name();
  const std::string pool_name = "poolmax2d";
  const std::string pool_out = real_out + "_post_reshape";
  const std::string post_reshape_node_name = "post_reshape";
  const std::string qnn_op = GetQnnOpType(op_type);
  TensorInfo output_info{};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[0], output_info));
  bool is_graph_output = qnn_model_wrapper.IsGraphOutput(real_out);
  Qnn_TensorType_t tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
  QnnTensorWrapper pool_tensor(
      pool_out,
      QNN_TENSOR_TYPE_NATIVE,
      reshape_input_info.qnn_data_type,
      output_info.quant_param.Copy(),
      std::move(pooled_shape));

  ORT_RETURN_IF_NOT(
      qnn_model_wrapper.AddTensorWrapper(std::move(pool_tensor)),
      "Failed to add tensor for pool_out");

  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                        pool_name,
                        QNN_OP_PACKAGE_NAME_QTI_AISW,
                        qnn_op,
                        {reshape4d},
                        {pool_out},
                        std::move(param_tensor_names),
                        do_op_validation),
                    "Failed to create QNN Pool node for rank-3 input.");

  std::vector<uint32_t> final_shape3d = output_info.shape;
  QnnTensorWrapper reshape_back_tensor(
      real_out,
      tensor_type,
      output_info.qnn_data_type,
      output_info.quant_param.Copy(),
      std::move(final_shape3d));

  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(reshape_back_tensor)), "Failed to add tensor.");
  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                        post_reshape_node_name,
                        QNN_OP_PACKAGE_NAME_QTI_AISW,
                        "Reshape",
                        {pool_out},
                        {real_out},
                        {},
                        do_op_validation),
                    "Failed to create reshape-back node.");

  return Status::OK();
}

Status PoolOpBuilder::OverrideOutputQuantParam(QnnModelWrapper& qnn_model_wrapper,
                                               const NodeUnit& node_unit,
                                               const logging::Logger& logger,
                                               const std::vector<std::string>& input_names,
                                               size_t output_index,
                                               Qnn_DataType_t qnn_data_type,
                                               QnnQuantParamsWrapper& quant_param) const {
  // Force MaxPool outputs to use the same quantization parameters as the input if they are nearly equal.
  // This helps the HTP backend employ certain optimizations.
  if (node_unit.OpType() == "MaxPool" && quant_param.IsPerTensor()) {
    return SetOutputQParamEqualToInputIfNearlyEqual(qnn_model_wrapper, node_unit, logger, input_names,
                                                    0 /*input_index*/, output_index, qnn_data_type, quant_param);
  }

  return Status::OK();
}

void CreatePoolOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<PoolOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
