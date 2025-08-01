// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/providers/qnn-abi/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn-abi/builder/qnn_utils.h"
#include "core/providers/qnn-abi/builder/qnn_model_wrapper.h"
#include "core/providers/qnn-abi/builder/op_builder_factory.h"

namespace onnxruntime {
namespace qnn {

class PoolOpBuilder : public BaseOpBuilder {
 public:
  PoolOpBuilder() : BaseOpBuilder("PoolOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(PoolOpBuilder);

  Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                       const OrtNodeUnit& node_unit,
                       const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

 protected:
  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const OrtNodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     const logging::Logger& logger,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;
  Status OverrideOutputQuantParam(QnnModelWrapper& qnn_model_wrapper,
                                  const OrtNodeUnit& node_unit,
                                  const logging::Logger& logger,
                                  const std::vector<std::string>& input_names,
                                  size_t output_index,
                                  Qnn_DataType_t qnn_data_type,
                                  QnnQuantParamsWrapper& quant_param) const override ORT_MUST_USE_RESULT;

 private:
  Status SetCommonPoolParams(const OrtNodeAttrHelper& node_helper,
                             std::vector<uint32_t>& filter_size,
                             std::vector<uint32_t>& stride,
                             std::vector<uint32_t>& pad_amount,
                             int32_t& rounding_mode,
                             std::vector<uint32_t>&& input_shape,
                             std::vector<uint32_t>&& output_shape) const;
};

// Pool ops are sensitive with data layout, no special validation so far
// The nodes from 1st call of GetCapability do not get layout transformer applied, it's still NCHW
// The nodes from 2nd call of GetCapability get layout transformer applied, it's NHWC
// Need to do op validation in 1st call of GetCapability
// TODO: Check if node domain == kMSInternalNHWCDomain to determine if the layout has been transformed.
Status PoolOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                    const OrtNodeUnit& node_unit,
                                    const logging::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);

  const auto& inputs = node_unit.Inputs();
  ORT_RETURN_IF_ERROR(DataTypeCheckForCpuBackend(qnn_model_wrapper, inputs[0].type, ""));

  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].shape, input_shape), "Cannot get shape");

  size_t rank = input_shape.size();
  ORT_RETURN_IF_NOT(rank == 3 || rank == 4 || rank == 5, "QNN Pool only supports rank 3, 4, or 5!");

  // ONNX MaxPool may have two outputs.
  ORT_RETURN_IF(node_unit.Outputs().size() > 1, "QNN Pool only supports 1 output!");

  OrtNodeAttrHelper node_helper(qnn_model_wrapper.GetOrtApi(), node_unit);
  auto dilations = node_helper.Get("dilations", std::vector<uint32_t>(rank - 2, 1));
  ORT_RETURN_IF_NOT(dilations == std::vector<uint32_t>(rank - 2, 1), "QNN Pool only supports dilations 1!");

  const std::string& op_type = node_unit.OpType();
  const bool is_npu_backend = IsNpuBackend(qnn_model_wrapper.GetQnnBackendType());

  if (rank == 5 && is_npu_backend) {
    ORT_RETURN_IF(op_type == "MaxPool" || op_type == "GlobalMaxPool", "QNN NPU does not support PoolMax3d!");
  }

  if (op_type == "MaxPool" || op_type == "AveragePool") {
    auto auto_pad = node_helper.Get("auto_pad", std::string("NOTSET"));
    ORT_RETURN_IF(auto_pad != "NOTSET" && auto_pad != "SAME_LOWER" && auto_pad != "SAME_UPPER" && auto_pad != "VALID",
                  "QNN Pool operators do not support 'auto_pad' value: ", auto_pad.c_str());
  }

  if (node_unit.Domain() == kMSInternalNHWCDomain) {  // Use QNN validation API if layout is NHWC.
    return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, true);
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

Status PoolOpBuilder::SetCommonPoolParams(const OrtNodeAttrHelper& node_helper,
                                          std::vector<uint32_t>& filter_size,
                                          std::vector<uint32_t>& stride,
                                          std::vector<uint32_t>& pad_amount,
                                          int32_t& rounding_mode,
                                          std::vector<uint32_t>&& input_shape,
                                          std::vector<uint32_t>&& output_shape) const {
  size_t rank = input_shape.size();

  // Param: filter_size.
  {
    auto raw_filter_size = node_helper.Get("kernel_shape", std::vector<uint32_t>(rank - 2, 1));
    if (raw_filter_size.size() == 1) {
      filter_size = {1, raw_filter_size[0]};
    } else {
      filter_size = raw_filter_size;
    }
  }

  // Param: stride.
  {
    auto raw_stride = node_helper.Get("strides", std::vector<uint32_t>(rank - 2, 1));
    if (raw_stride.size() == 1) {
      stride = {1, raw_stride[0]};
    } else {
      stride = raw_stride;
    }
  }

  // Param: dilations (NOT SUPPORTED by QNN).
  std::vector<uint32_t> dilations;
  {
    auto raw_dilations = node_helper.Get("dilations", std::vector<uint32_t>(rank - 2, 1));
    if (raw_dilations.size() == 1) {
      dilations = {1, raw_dilations[0]};
    } else {
      dilations = raw_dilations;
    }
  }

  // Param: pad_amount.
  {
    auto raw_pad_amount = node_helper.Get("pads", std::vector<uint32_t>((rank - 2) * 2, 0));
    if (raw_pad_amount.size() == 2) {
      pad_amount = {0, raw_pad_amount[0], 0, raw_pad_amount[1]};
    } else {
      pad_amount = raw_pad_amount;
    }
  }

  auto auto_pad = node_helper.Get("auto_pad", std::string("NOTSET"));
  if (auto_pad.compare("NOTSET") != 0) {
    if (output_shape.size() == 3) {
      // Calculate rank-4 output shape for rank-3 input.
      output_shape = AmendOutputShapeForRank3Pool(input_shape,
                                                  filter_size,
                                                  stride,
                                                  pad_amount);
    }

    for (size_t axis = 0; axis < rank - 2; ++axis) {
      uint32_t total_pads = (output_shape[axis + 1] - 1) * stride[axis] +
                            (filter_size[axis] - 1) * dilations[axis] + 1 - input_shape[axis + 1];
      if (auto_pad.compare("SAME_LOWER") == 0) {
        pad_amount[axis + rank - 2] = total_pads / 2;
        pad_amount[axis] = total_pads - pad_amount[axis + rank - 2];
      } else if (auto_pad.compare("SAME_UPPER") == 0) {
        pad_amount[axis] = total_pads / 2;
        pad_amount[axis + rank - 2] = total_pads - pad_amount[axis];
      }
    }
  }
  ReArranagePads(pad_amount);

  // Param: rounding_mode.
  rounding_mode = node_helper.Get("ceil_mode", rounding_mode);

  return Status::OK();
}

bool SetPoolParam(const OrtNodeUnit& node_unit,
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
  return qnn_model_wrapper.AddParamWrapper(std::move(qnn_param));
}

Status PoolOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                  const OrtNodeUnit& node_unit,
                                                  std::vector<std::string>&& input_names,
                                                  const logging::Logger& logger,
                                                  bool do_op_validation) const {
  OrtNodeAttrHelper node_helper(qnn_model_wrapper.GetOrtApi(), node_unit);
  // Get the NCHW from input data, use HW for the pool filter size and pool stride
  const auto& inputs = node_unit.Inputs();
  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].shape, input_shape), "Cannot get shape");

  // Reshape 3D input to 4D if necessary.
  const auto& reshape_input = node_unit.Inputs()[0];
  TensorInfo reshape_input_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(reshape_input, reshape_input_info));

  bool needs_reshape = false;
  const std::string reshape_prior_out = input_names[0] + "_prior_reshape";
  if (input_shape.size() == 3) {
    needs_reshape = true;
    // build new_shape = {N, 1, C, L}
    std::vector<uint32_t> new_shape = {
        input_shape[0],
        1,
        input_shape[1],
        input_shape[2]};

    QnnTensorWrapper reshape_prior_tensor(
        reshape_prior_out,
        QNN_TENSOR_TYPE_NATIVE,
        reshape_input_info.qnn_data_type,
        reshape_input_info.quant_param.Copy(),
        std::move(new_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(reshape_prior_tensor)),
                      "Failed to add reshape prior tensor.");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                          utils::GetNodeName(node_unit) + "_reshape_prior",
                          QNN_OP_PACKAGE_NAME_QTI_AISW,
                          QNN_OP_RESHAPE,
                          {input_names[0]},
                          {reshape_prior_out},
                          {},
                          do_op_validation),
                      "Failed to create reshape prior node for pool op.");
    input_names[0] = reshape_prior_out;
    input_shape = {input_shape[0], 1, input_shape[1], input_shape[2]};
  }

  const std::string& op_type = node_unit.OpType();
  const size_t rank = input_shape.size();

  // QNN constants for construction.
  std::string qnn_op_type;
  std::string param_filter_size;
  std::string param_stride;
  std::string param_pad_amount;
  std::string param_count_pad_for_edges;
  std::string param_rounding_mode;
  if (rank == 4) {
    if (op_type == "MaxPool" || op_type == "GlobalMaxPool") {
      qnn_op_type = QNN_OP_POOL_MAX_2D;
      param_filter_size = QNN_OP_POOL_MAX_2D_PARAM_FILTER_SIZE;
      param_stride = QNN_OP_POOL_MAX_2D_PARAM_STRIDE;
      param_pad_amount = QNN_OP_POOL_MAX_2D_PARAM_PAD_AMOUNT;
      param_rounding_mode = QNN_OP_POOL_MAX_2D_PARAM_ROUNDING_MODE;
    } else {
      qnn_op_type = QNN_OP_POOL_AVG_2D;
      param_filter_size = QNN_OP_POOL_AVG_2D_PARAM_FILTER_SIZE;
      param_stride = QNN_OP_POOL_AVG_2D_PARAM_STRIDE;
      param_pad_amount = QNN_OP_POOL_AVG_2D_PARAM_PAD_AMOUNT;
      param_count_pad_for_edges = QNN_OP_POOL_AVG_2D_PARAM_COUNT_PAD_FOR_EDGES;
      param_rounding_mode = QNN_OP_POOL_AVG_2D_PARAM_ROUNDING_MODE;
    }
  } else {
    if (op_type == "MaxPool" || op_type == "GlobalMaxPool") {
      qnn_op_type = QNN_OP_POOL_MAX_3D;
      param_filter_size = QNN_OP_POOL_MAX_3D_PARAM_FILTER_SIZE;
      param_stride = QNN_OP_POOL_MAX_3D_PARAM_STRIDE;
      param_pad_amount = QNN_OP_POOL_MAX_3D_PARAM_PAD_AMOUNT;
      param_rounding_mode = QNN_OP_POOL_MAX_3D_PARAM_ROUNDING_MODE;
    } else {
      qnn_op_type = QNN_OP_POOL_AVG_3D;
      param_filter_size = QNN_OP_POOL_AVG_3D_PARAM_FILTER_SIZE;
      param_stride = QNN_OP_POOL_AVG_3D_PARAM_STRIDE;
      param_pad_amount = QNN_OP_POOL_AVG_3D_PARAM_PAD_AMOUNT;
      param_count_pad_for_edges = QNN_OP_POOL_AVG_3D_PARAM_COUNT_PAD_FOR_EDGES;
      param_rounding_mode = QNN_OP_POOL_AVG_3D_PARAM_ROUNDING_MODE;
    }
  }

  // Default parameters for GlobalMaxPool/GlobalAveragePool with filter and stride in spatial shapes.
  // Note that input is already in spatial-first layout.
  std::vector<uint32_t> filter_size(input_shape.begin() + 1, input_shape.begin() + rank - 1);
  std::vector<uint32_t> filter_size_dim{static_cast<uint32_t>(rank - 2)};
  std::vector<uint32_t> stride(filter_size);
  std::vector<uint32_t> stride_dim{static_cast<uint32_t>(rank - 2)};
  std::vector<uint32_t> pad_amount((rank - 2) * 2, 0);
  std::vector<uint32_t> pad_amount_dim{static_cast<uint32_t>(rank - 2), 2};
  int32_t rounding_mode = 0;

  std::vector<std::string> param_tensor_names;
  if (op_type == "GlobalMaxPool") {
    ORT_RETURN_IF_NOT(SetPoolParam(node_unit,
                                   param_filter_size,
                                   std::move(filter_size_dim),
                                   std::move(filter_size),
                                   param_tensor_names,
                                   qnn_model_wrapper),
                      "Failed to add param filter_size.");
    ORT_RETURN_IF_NOT(SetPoolParam(node_unit,
                                   param_stride,
                                   std::move(stride_dim),
                                   std::move(stride),
                                   param_tensor_names,
                                   qnn_model_wrapper),
                      "Failed to add param stride.");
    ORT_RETURN_IF_NOT(SetPoolParam(node_unit,
                                   param_pad_amount,
                                   std::move(pad_amount_dim),
                                   std::move(pad_amount),
                                   param_tensor_names,
                                   qnn_model_wrapper),
                      "Failed to add param pad_amount.");

    ORT_RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit,
                                       std::move(input_names),
                                       std::move(param_tensor_names),
                                       logger,
                                       do_op_validation,
                                       qnn_op_type));
    return Status::OK();
  }

  if (op_type == "MaxPool" || op_type == "AveragePool") {
    const auto& outputs = node_unit.Outputs();
    std::vector<uint32_t> output_shape;
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(outputs[0].shape, output_shape), "Cannot get shape");

    ORT_RETURN_IF_ERROR(SetCommonPoolParams(node_helper,
                                            filter_size,
                                            stride,
                                            pad_amount,
                                            rounding_mode,
                                            std::move(input_shape),
                                            std::move(output_shape)));
  }

  // Calculate rank-4 output shape for rank-3 input.
  std::vector<uint32_t> onnx_in_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].shape, onnx_in_shape), "Cannot get shape");
  if (onnx_in_shape.size() == 3) {
    onnx_in_shape = {onnx_in_shape[0], 1, onnx_in_shape[1], onnx_in_shape[2]};
  }
  auto pooled_shape = AmendOutputShapeForRank3Pool(onnx_in_shape, filter_size, stride, pad_amount);

  // Construct param wrappers.
  ORT_RETURN_IF_NOT(SetPoolParam(node_unit,
                                 param_filter_size,
                                 std::move(filter_size_dim),
                                 std::move(filter_size),
                                 param_tensor_names,
                                 qnn_model_wrapper),
                    "Failed to add param filter_size.");
  ORT_RETURN_IF_NOT(SetPoolParam(node_unit,
                                 param_stride,
                                 std::move(stride_dim),
                                 std::move(stride),
                                 param_tensor_names,
                                 qnn_model_wrapper),
                    "Failed to add param stride.");
  ORT_RETURN_IF_NOT(SetPoolParam(node_unit,
                                 param_pad_amount,
                                 std::move(pad_amount_dim),
                                 std::move(pad_amount),
                                 param_tensor_names,
                                 qnn_model_wrapper),
                    "Failed to add param pad_amount.");

  if (0 != rounding_mode) {
    Qnn_Scalar_t scalar_param = QNN_SCALAR_INIT;
    scalar_param.dataType = QNN_DATATYPE_UINT_32;
    scalar_param.int32Value = rounding_mode;
    QnnParamWrapper rounding_mode_param_wrapper(node_unit.Index(),
                                                node_unit.Name(),
                                                param_rounding_mode,
                                                scalar_param);
    param_tensor_names.push_back(rounding_mode_param_wrapper.GetParamTensorName());
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(rounding_mode_param_wrapper)),
                      "Failed to add param rounding_mode.");
  }
  if (op_type == "GlobalAveragePool") {
    Qnn_Scalar_t scalar_param = QNN_SCALAR_INIT;
    scalar_param.dataType = QNN_DATATYPE_BOOL_8;
    scalar_param.bool8Value = 1;
    QnnParamWrapper count_pad_for_edges_param(node_unit.Index(),
                                              node_unit.Name(),
                                              param_count_pad_for_edges,
                                              scalar_param);
    param_tensor_names.push_back(count_pad_for_edges_param.GetParamTensorName());
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(count_pad_for_edges_param)),
                      "Failed to add param count_pad_for_edges.");
  } else if (op_type == "AveragePool") {
    Qnn_Scalar_t scalar_param = QNN_SCALAR_INIT;
    scalar_param.dataType = QNN_DATATYPE_BOOL_8;
    scalar_param.bool8Value = static_cast<uint8_t>(node_helper.Get("count_include_pad", static_cast<int64_t>(0)) != 0);
    QnnParamWrapper count_pad_for_edges_param(node_unit.Index(),
                                              node_unit.Name(),
                                              param_count_pad_for_edges,
                                              scalar_param);
    param_tensor_names.push_back(count_pad_for_edges_param.GetParamTensorName());
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(count_pad_for_edges_param)),
                      "Failed to add param count_include_pad.");
  }

  if (!needs_reshape) {
    ORT_RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper,
                                       node_unit,
                                       std::move(input_names),
                                       std::move(param_tensor_names),
                                       logger,
                                       do_op_validation,
                                       qnn_op_type));

    return Status::OK();
  }
  const auto& outputs = node_unit.Outputs();
  const std::string real_out = outputs[0].name;
  const std::string pool_out = real_out + "_reshape_after";
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
                        utils::GetNodeName(node_unit) + "_pool2d",
                        QNN_OP_PACKAGE_NAME_QTI_AISW,
                        qnn_op,
                        {reshape_prior_out},
                        {pool_out},
                        std::move(param_tensor_names),
                        do_op_validation),
                    "Failed to create pool node for rank-3 input.");

  std::vector<uint32_t> final_shape3d = output_info.shape;
  QnnTensorWrapper reshape_after_tensor(
      real_out,
      tensor_type,
      output_info.qnn_data_type,
      output_info.quant_param.Copy(),
      std::move(final_shape3d));

  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(reshape_after_tensor)),
                    "Failed to add reshape after tensor.");
  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                        utils::GetNodeName(node_unit) + "_reshape_after",
                        QNN_OP_PACKAGE_NAME_QTI_AISW,
                        QNN_OP_RESHAPE,
                        {pool_out},
                        {real_out},
                        {},
                        do_op_validation),
                    "Failed to create reshape after node for pool op.");

  return Status::OK();
}

Status PoolOpBuilder::OverrideOutputQuantParam(QnnModelWrapper& qnn_model_wrapper,
                                               const OrtNodeUnit& node_unit,
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
