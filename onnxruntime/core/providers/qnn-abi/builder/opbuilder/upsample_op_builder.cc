// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/providers/qnn-abi/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn-abi/builder/qnn_utils.h"
#include "core/providers/qnn-abi/builder/qnn_model_wrapper.h"
#include "core/providers/qnn-abi/builder/op_builder_factory.h"

namespace onnxruntime {
namespace qnn {

class UpsampleOpBuilder : public BaseOpBuilder {
 public:
  UpsampleOpBuilder() : BaseOpBuilder("UpsampleOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(UpsampleOpBuilder);

  Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger) const final ORT_MUST_USE_RESULT;

 protected:
  Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger,
                       std::vector<std::string>& input_names,
                       bool do_op_validation) const override ORT_MUST_USE_RESULT;

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
  const std::unordered_map<std::string, uint32_t> supported_modes = {
      {"nearest", QNN_OP_RESIZE_INTERPOLATION_MODE_NEAREST},
      {"linear", QNN_OP_RESIZE_INTERPOLATION_MODE_LINEAR},
      {"cubic", QNN_OP_RESIZE_INTERPOLATION_MODE_CUBIC}};

  // Info for Onnx Upsample attribute {<attribute_name>, <default_value>}
  const OnnxAttrInfo<std::string> onnx_mode_attr = {"mode", "nearest"};
};

Status UpsampleOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                        const NodeUnit& node_unit,
                                        const logging::Logger& logger) const {
  // Resize ops are sensitive with data layout, no special validation so far
  // The nodes from 1st call of GetCapability do not get layout transformer applied, it's still NCHW
  // The nodes from 2nd call of GetCapability get layout transformer applied, it's NHWC
  // Need to do op validation in 1st call of GetCapability
  if (node_unit.Domain() == kMSInternalNHWCDomain) {
    return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, true);
  }

  const bool is_npu_backend = IsNpuBackend(qnn_model_wrapper.GetQnnBackendType());
  NodeAttrHelper node_helper(node_unit);

  // Check mode
  const std::string interp_mode = GetOnnxAttr(node_helper, onnx_mode_attr);
  ORT_RETURN_IF_NOT(supported_modes.find(interp_mode) != supported_modes.end(),
                    "QNN EP: Resize does not support mode ", interp_mode.c_str());

  const auto& input_0 = node_unit.Inputs()[0];
  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input_0.node_arg, input_shape),
                    "QNN EP: Cannot get input shape for Onnx Upsample ", input_0.node_arg.Name().c_str());
  const size_t input_rank = input_shape.size();

  ORT_RETURN_IF(is_npu_backend && (input_rank < 3 || input_rank > 5),
                "QNN EP: The input rank for Resize must be at least 3 and no greater than 5 on the HTP.");

  const auto& output_0 = node_unit.Outputs()[0];
  std::vector<uint32_t> output_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(output_0.node_arg, output_shape),
                    "QNN EP: Cannot get output shape for Onnx Upsample ", output_0.node_arg.Name().c_str(),
                    ". Dynamic scales input is not supported in QNN EP.");

  // Check that only the spatial dimensions (width, height) are resized. The batch_size (N) and channels (C) should
  // be untouched. This code runs before layout transformation, so we know that the current layout is "channel first"
  // (e.g., N, C, S1, S2, ..., SN).
  ORT_RETURN_IF_NOT(input_shape[0] == output_shape[0] && input_shape[1] == output_shape[1],
                    "QNN EP: Resize may only change the spatial dimensions.");

  if (!is_npu_backend) {
    ONNX_NAMESPACE::DataType input_data_type = input_0.node_arg.Type();
    ORT_RETURN_IF(input_data_type != ONNX_NAMESPACE::Utils::DataTypeUtils::ToType("float"),
                  "QNN EP: Data type ", input_data_type->c_str(),
                  " is not supported for Resize operator in CPU backend.");
  }

  return Status::OK();
}

Status UpsampleOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                        const NodeUnit& node_unit,
                                        const logging::Logger& logger,
                                        std::vector<std::string>& input_names,
                                        bool do_op_validation) const {
  const int opset_version = node_unit.SinceVersion();
  const auto& inputs = node_unit.Inputs();

  if (opset_version > 7 && do_op_validation) {
    const std::string& scales_input_name = inputs[1].node_arg.Name();
    ORT_RETURN_IF_NOT(qnn_model_wrapper.IsConstantInput(scales_input_name),
                      "QNN doesn't support dynamic scales input for ONNX Upsample op ", node_unit.Name().c_str());
  }

  // Only need to consider the first input of Onnx upsample.
  ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[0], logger, input_names));

  return Status::OK();
}

Status UpsampleOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                      const NodeUnit& node_unit,
                                                      std::vector<std::string>&& input_names,
                                                      const logging::Logger& logger,
                                                      bool do_op_validation) const {
  std::vector<std::string> param_tensor_names;
  NodeAttrHelper node_helper(node_unit);
  const std::string interp_mode = GetOnnxAttr(node_helper, onnx_mode_attr);

  const auto& input_0 = node_unit.Inputs()[0];
  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input_0.node_arg, input_shape),
                    "QNN EP: Cannot get input shape for Onnx Upsample ", input_0.node_arg.Name().c_str());

  const size_t input_rank = input_shape.size();
  const bool is_npu_backend = IsNpuBackend(qnn_model_wrapper.GetQnnBackendType());
  std::string qnn_op_type = GetQnnOpType(node_unit.OpType());

  if (is_npu_backend && input_rank == 4 && interp_mode != "cubic") {
    // Translate QNN's Resize to QNN's ResizeNearestNeighbor/ResizeBilinear to achieve better performance on
    // the HTP backend. QNN's ResizeNearestNeighbor and ResizeBilinear are only supported when input rank is 4.
    qnn_op_type = (interp_mode == "nearest") ? QNN_OP_RESIZE_NEAREST_NEIGHBOR : QNN_OP_RESIZE_BILINEAR;

    // Parameter 'align_corners'
    const std::string align_corners_param_name = (qnn_op_type == QNN_OP_RESIZE_BILINEAR)
                                                     ? QNN_OP_RESIZE_BILINEAR_PARAM_ALIGN_CORNERS
                                                     : QNN_OP_RESIZE_NEAREST_NEIGHBOR_PARAM_ALIGN_CORNERS;
    ORT_RETURN_IF_ERROR(AddQnnScalar<bool>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), false, align_corners_param_name, param_tensor_names));

    // Parameter 'half_pixel_centers'
    const std::string half_pixel_centers_param_name = (qnn_op_type == QNN_OP_RESIZE_BILINEAR)
                                                          ? QNN_OP_RESIZE_BILINEAR_PARAM_HALF_PIXEL_CENTERS
                                                          : QNN_OP_RESIZE_NEAREST_NEIGHBOR_PARAM_HALF_PIXEL_CENTERS;
    ORT_RETURN_IF_ERROR(AddQnnScalar<bool>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), false, half_pixel_centers_param_name, param_tensor_names));

    if (qnn_op_type == QNN_OP_RESIZE_BILINEAR) {
      // Parameter 'antialias'
      ORT_RETURN_IF_ERROR(AddQnnScalar<bool>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), false, QNN_OP_RESIZE_BILINEAR_PARAM_ANTIALIAS, param_tensor_names));
    }
  } else {
    // Remain as QNN's Resize.
    // Parameter 'exclude_outside'
    ORT_RETURN_IF_ERROR(AddQnnScalar<bool>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), false, QNN_OP_RESIZE_PARAM_EXCLUDE_OUTSIDE, param_tensor_names));

    // Parameter 'transformation_mode'
    uint32_t transformation_mode = (supported_modes.at(interp_mode) == QNN_OP_RESIZE_INTERPOLATION_MODE_NEAREST)
                                       ? static_cast<uint32_t>(QNN_OP_RESIZE_TRANSFORMATION_MODE_HALF_PIXEL)
                                       : static_cast<uint32_t>(QNN_OP_RESIZE_TRANSFORMATION_MODE_ASYMMETRIC);
    ORT_RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), transformation_mode, QNN_OP_RESIZE_PARAM_TRANSFORMATION_MODE, param_tensor_names));

    // Parameter 'interpolation_mode'
    uint32_t qnn_interp_mode = static_cast<uint32_t>(supported_modes.at(interp_mode));
    ORT_RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), qnn_interp_mode, QNN_OP_RESIZE_PARAM_INTERPOLATION_MODE, param_tensor_names));

    // Parameter 'nearest_mode'. Process only when 'interpolation_mode' is NEAREST.
    if (qnn_interp_mode == QNN_OP_RESIZE_INTERPOLATION_MODE_NEAREST) {
      uint32_t qnn_nearest_mode = static_cast<uint32_t>(QNN_OP_RESIZE_NEAREST_MODE_ROUND_PREFER_FLOOR);
      ORT_RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), qnn_nearest_mode, QNN_OP_RESIZE_PARAM_NEAREST_MODE, param_tensor_names));
    }
  }

  ORT_RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit,
                                     std::move(input_names),
                                     std::move(param_tensor_names),
                                     logger, do_op_validation, qnn_op_type));

  return Status::OK();
}

Status UpsampleOpBuilder::OverrideOutputQuantParam(QnnModelWrapper& qnn_model_wrapper,
                                                   const NodeUnit& node_unit,
                                                   const logging::Logger& logger,
                                                   const std::vector<std::string>& input_names,
                                                   size_t output_index,
                                                   Qnn_DataType_t qnn_data_type,
                                                   QnnQuantParamsWrapper& quant_param) const {
  if (!quant_param.IsPerTensor()) {
    return Status::OK();
  }

  // Force Resize op's output to use the same quantization parameters as the input if nearly equal.
  // This helps the HTP backend employ certain optimizations.
  return SetOutputQParamEqualToInputIfNearlyEqual(qnn_model_wrapper, node_unit, logger, input_names,
                                                  0 /*input_index*/, output_index, qnn_data_type, quant_param);
}

void CreateUpsampleOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<UpsampleOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
