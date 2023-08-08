// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <string_view>

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/cpu/tensor/slice_helper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/common/safeint.h"

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"

namespace onnxruntime {
namespace qnn {

class ResizeOpBuilder : public BaseOpBuilder {
 public:
  ResizeOpBuilder() : BaseOpBuilder("ResizeOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ResizeOpBuilder);

  Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger,
                       bool is_quantized_model) const override final ORT_MUST_USE_RESULT;

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
  /**
   * Returns the QNN integer value that corresponds to the given ONNX mode (string).
   *
   * /param onnx_modes Array of ONNX modes supported by QNN. The index of each mode corresponds to the QNN value.
   * /param onnx_mode The ONNX mode for which to get the corresponding QNN value.
   * /param onnx_model_label Mode label to print out in case of error (e.g., "nearest_mode").
   * /param qnn_mode Output parameter that is set to the appropriate QNN value from the given ONNX mode.
   *
   * /returns A status indicating failure or success.
   */
  template <typename QnnValType, std::size_t N>
  Status GetQnnModeFromString(const std::array<std::string_view, N>& onnx_modes, std::string_view onnx_mode,
                              const char* onnx_mode_label, QnnValType& qnn_mode) const ORT_MUST_USE_RESULT;

  /**
   * Called by IsOpSupported to validate the op for non-quantized models.
   *
   * /param qnn_model_wrapper The QNN model wrapper instance.
   * /param node_unit The node unit containing metadata for the ONNX Resize operator.
   *
   * /returns A status indicating failure or success.
   */
  Status ValidateOp(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit) const ORT_MUST_USE_RESULT;

  /**
   * Called by IsOpSupported to validate the op for quantized models.
   *
   * /param qnn_model_wrapper The QNN model wrapper instance.
   * /param node_unit The node unit containing metadata for the ONNX Resize operator and its Q/DQ nodes.
   *
   * /returns A status indicating failure or success.
   */
  Status ValidateQDQOp(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit) const ORT_MUST_USE_RESULT;

  /**
   * Called by ProcessAttributesAndOutputs to process the op's attributes and outputs
   * for non-quantized models.
   *
   * /param qnn_model_wrapper The QNN model wrapper instance.
   * /param node_unit The node unit containing metadata for the ONNX Resize operator.
   * /param input_names The operator's input names.
   * /param logger A logger.
   * /param do_op_validation Set to true if the op should be validated using QNN's validation API.
   *
   * /returns A status indicating failure or success.
   */
  Status ProcessOpAttrsAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                  const NodeUnit& node_unit,
                                  std::vector<std::string>&& input_names,
                                  const logging::Logger& logger,
                                  bool do_op_validation) const ORT_MUST_USE_RESULT;

  /**
   * Called by ProcessAttributesAndOutputs to process the op's attributes and outputs
   * for quantized models.
   *
   * /param qnn_model_wrapper The QNN model wrapper instance.
   * /param node_unit The node unit containing metadata for the ONNX Resize operator and its Q/DQ nodes.
   * /param input_names The operator's input names.
   * /param logger A logger.
   * /param do_op_validation Set to true if the op should be validated using QNN's validation API.
   *
   * /returns A status indicating failure or success.
   */
  Status ProcessQDQOpAttrsAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     const logging::Logger& logger,
                                     bool do_op_validation) const ORT_MUST_USE_RESULT;

  // Info for each ONNX attribute of interest (attribute name + default value)
  static const OnnxAttrInfo<std::string> onnx_mode_attr;
  static const OnnxAttrInfo<std::string> onnx_coord_transf_mode_attr;
  static const OnnxAttrInfo<std::string> onnx_nearest_mode_attr;
  static const OnnxAttrInfo<int64_t> onnx_antialias_attr;
  static const OnnxAttrInfo<int64_t> onnx_exclude_outside_attr;

  // Arrays of supported QNN modes for QNN's Resize op. The index of each mode is used as the corresponding
  // QNN parameter value. Ex: The "nearest" mode is represented as the value 0 in QNN. Note, that
  // not all modes are supported by every QNN backend.

  // QNN values: NEAREST = 0, LINEAR = 1
  static constexpr std::array<std::string_view, 2> supported_modes = {"nearest", "linear"};

  // QNN values: HALF_PIXEL = 0, PYTORCH_HALF_PIXEL = 1, ALIGN_CORNERS = 2, ASYMMETRIC = 3
  static constexpr std::array<std::string_view, 4> supported_coord_transf_modes = {"half_pixel", "pytorch_half_pixel",
                                                                                   "align_corners", "asymmetric"};

  // QNN values: ROUND_PREFER_FLOOR = 0, ROUND_PREFER_CEIL = 1, FLOOR = 2, CEIL = 3
  static constexpr std::array<std::string_view, 4> supported_nearest_modes = {"round_prefer_floor", "round_prefer_ceil",
                                                                              "floor", "ceil"};
};

const OnnxAttrInfo<std::string> ResizeOpBuilder::onnx_mode_attr = {"mode", "nearest"};
const OnnxAttrInfo<std::string> ResizeOpBuilder::onnx_coord_transf_mode_attr = {"coordinate_transformation_mode",
                                                                                "half_pixel"};
const OnnxAttrInfo<std::string> ResizeOpBuilder::onnx_nearest_mode_attr = {"nearest_mode",
                                                                           "round_prefer_floor"};
const OnnxAttrInfo<int64_t> ResizeOpBuilder::onnx_antialias_attr = {"antialias", 0};
const OnnxAttrInfo<int64_t> ResizeOpBuilder::onnx_exclude_outside_attr = {"exclude_outside", 0};

template <typename QnnValType, std::size_t N>
Status ResizeOpBuilder::GetQnnModeFromString(const std::array<std::string_view, N>& onnx_modes,
                                             std::string_view onnx_mode, const char* onnx_mode_label,
                                             QnnValType& qnn_mode) const {
  for (size_t i = 0; i < onnx_modes.size(); ++i) {
    if (onnx_modes[i] == onnx_mode) {
      qnn_mode = SafeInt<QnnValType>(i);
      return Status::OK();
    }
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN EP: Resize operator does not support ", onnx_mode_label,
                         " ", std::string(onnx_mode));
}

// Utility function that checks if an array of strings contains a specific string.
// Used to validate ONNX operator attributes.
template <size_t N>
static bool ArrayHasString(const std::array<std::string_view, N>& strings, std::string_view str) {
  for (auto s : strings) {
    if (s == str) {
      return true;
    }
  }

  return false;
}

// Resize ops are sensitive with data layout, no special validation so far
// The nodes from 1st call of GetCapability do not get layout transformer applied, it's still NCHW
// The nodes from 2nd call of GetCapability get layout transformer applied, it's NHWC
// Need to do op validation in 1st call of GetCapability
Status ResizeOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                      const NodeUnit& node_unit,
                                      const logging::Logger& logger,
                                      bool is_quantized_model) const {
  if (node_unit.Domain() == kMSInternalNHWCDomain) {
    return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, is_quantized_model, true);
  }

  // QNN doesn't support anti-aliasing (added in opset 18)
  if (node_unit.SinceVersion() >= 18) {
    NodeAttrHelper node_helper(node_unit);
    const bool antialias = GetOnnxAttr(node_helper, onnx_antialias_attr) != 0;
    ORT_RETURN_IF(antialias, "QNN EP: Resize doesn't support anti-aliasing.");
  }

  // The QNN Resize op does not currently work with the QNN cpu backend, but works with the HTP backend. Therefore, we
  // currently use QNN's Resize op for quantized models and either ResizeBilinear or ResizeNearestNeighbor for
  // non-quantized models. This requires separate validation for quantized models.
  // TODO: Use only Resize once QNN's Resize op works in the QNN cpu backend.
  return is_quantized_model ? ValidateQDQOp(qnn_model_wrapper, node_unit) : ValidateOp(qnn_model_wrapper, node_unit);
}

Status ResizeOpBuilder::ValidateOp(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit) const {
  NodeAttrHelper node_helper(node_unit);
  const std::string resize_mode = GetOnnxAttr(node_helper, onnx_mode_attr);
  ORT_RETURN_IF((resize_mode != "nearest") && (resize_mode != "linear"),
                "QNN EP: Resize doesn't support mode '", resize_mode.c_str(), "'.",
                "Only 'nearest' and 'linear' are supported.");

  const std::string coordinate_mode = GetOnnxAttr(node_helper, onnx_coord_transf_mode_attr);
  ORT_RETURN_IF((coordinate_mode != "half_pixel") && (coordinate_mode != "align_corners"),
                "QNN EP: coordinate transformation mode '", coordinate_mode.c_str(), "' not supported for Resize op.",
                "Only 'align_corners' and 'half_pixel' are supported.");

  // Check for a valid "nearest_mode" if the mode is "nearest".
  if (resize_mode == "nearest") {
    // NOTE: QNN's ResizeNearestNeighbor operator does not have a way to specify rounding (i.e., "nearest_mode").
    // The output of the QNN ResizeNearestNeighbor operator is not always equivalent to ONNX's Resize
    // operator with any single specific "nearest_mode".
    //
    // For some input/output shapes, QNN's ResizeNearestNeighbor is equivalent to ONNX's Resize with "round_prefer_floor".
    // For other shapes, QNN's ResizeNearestNeighbor is equivalent to ONNX Resize with "round_prefer_ceil".
    //
    // From unit tests, I've found a relationship between input/output shapes and the equivalent ONNX "nearest_mode".
    // If the new and old spatial dimensions are evenly divisible, the "nearest_mode" is "round_prefer_floor".
    // Otherwise, the "nearest_mode" is "round_prefer_ceil".
    //
    // This relationship is probably incomplete/wrong.
    //
    // TODO: Ask Qualcomm what the correct "nearest_mode" should be,
    // OR use QNN's own Resize operator once it works on QnnCpu.
    const std::string& nearest_mode = GetOnnxAttr(node_helper, onnx_nearest_mode_attr);
    ORT_RETURN_IF_NOT("floor" == nearest_mode, "QNN Resize only supports nearest_mode: floor!");  // This is wrong!
  }

  auto& input_0 = node_unit.Inputs()[0];
  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input_0.node_arg, input_shape),
                    "QNN EP: Cannot get input shape for Resize op");

  const auto& output_0 = node_unit.Outputs()[0];
  std::vector<uint32_t> output_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(output_0.node_arg, output_shape),
                    "QNN EP: Cannot get output shape for Resize op");

  ORT_RETURN_IF(input_shape.size() != 4 || output_shape.size() != 4, "QNN Resize only supports 4D!");

  ONNX_NAMESPACE::DataType input_data_type = input_0.node_arg.Type();
  ORT_RETURN_IF(input_data_type != ONNX_NAMESPACE::Utils::DataTypeUtils::ToType("float"),
                "QNN EP: Data type ", input_data_type->c_str(),
                " is not supported for Resize operator in CPU backend.");

  return Status::OK();
}

Status ResizeOpBuilder::ValidateQDQOp(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit) const {
  NodeAttrHelper node_helper(node_unit);

  // Check mode
  const std::string interp_mode = GetOnnxAttr(node_helper, onnx_mode_attr);
  ORT_RETURN_IF_NOT(ArrayHasString(supported_modes, interp_mode), "QNN EP: Resize does not support mode ",
                    interp_mode.c_str());

  // Check coordinate transformation mode
  const std::string transformation_mode = GetOnnxAttr(node_helper, onnx_coord_transf_mode_attr);
  ORT_RETURN_IF_NOT(ArrayHasString(supported_coord_transf_modes, transformation_mode),
                    "QNN EP: Resize does not support coordinate_transformation_mode ", transformation_mode.c_str());

  // Check nearest mode
  if (interp_mode == "nearest") {
    const std::string nearest_mode = GetOnnxAttr(node_helper, onnx_nearest_mode_attr);
    ORT_RETURN_IF_NOT(ArrayHasString(supported_nearest_modes, nearest_mode),
                      "QNN EP: Resize does not support nearest_mode ", nearest_mode.c_str());

    // TODO: Support 'asymmetric' transformation mode with nearest_mode != 'floor'.
    //
    // QNN's ONNX converter tool translates 'nearest' + 'asymmetric' (regardless of rounding mode)
    // to QNN's ResizeNearestNeighbor with {align_corners: 0, half_pixel: 0}.
    // This is only accurate if the rounding mode is "floor". Need to investigate how to handle
    // other rounding modes with Qualcomm. Ideally, we would use QNN's Resize operator, but it doesn't support
    // the "asymmetric" coordinate transformation mode on HTP.
    ORT_RETURN_IF(transformation_mode == "asymmetric" && nearest_mode != "floor",
                  "QNN EP: Resize with coordinate_transformation_mode 'asymmetric' and nearest_mode '", nearest_mode,
                  "' is not currently supported on the HTP backend.");
  }

  // Check that input shape has at least a rank of 3.
  const auto& input_0 = node_unit.Inputs()[0];
  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input_0.node_arg, input_shape),
                    "QNN EP: Cannot get shape for Resize input");
  ORT_RETURN_IF(input_shape.size() < 3, "QNN EP: Resize input must have a rank >= 3.");

  const auto& output_0 = node_unit.Outputs()[0];
  std::vector<uint32_t> output_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(output_0.node_arg, output_shape),
                    "QNN EP: Cannot get shape for Resize output");
  ORT_RETURN_IF(output_shape.size() < 3, "QNN EP: Resize output must have a rank >= 3.");

  return Status::OK();
}

Status ResizeOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                      const NodeUnit& node_unit,
                                      const logging::Logger& logger,
                                      bool is_quantized_model,
                                      std::vector<std::string>& input_names,
                                      bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);

  // Only cares about the 1st input
  const auto& inputs = node_unit.Inputs();

  ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[0], logger, is_quantized_model, input_names));

  return Status::OK();
}

Status ResizeOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                    const NodeUnit& node_unit,
                                                    std::vector<std::string>&& input_names,
                                                    const logging::Logger& logger,
                                                    bool is_quantized_model,
                                                    bool do_op_validation) const {
  // The QNN Resize op does not currently work with the QNN cpu backend, but works with the HTP backend. Therefore, we
  // currently use QNN's Resize op for quantized models and either ResizeBilinear or ResizeNearestNeighbor for
  // non-quantized models. This requires separate handling for quantized models.
  // TODO: Use only Resize once QNN's Resize op works in the QNN cpu backend.
  return is_quantized_model ? ProcessQDQOpAttrsAndOutputs(qnn_model_wrapper, node_unit, std::move(input_names), logger, do_op_validation) : ProcessOpAttrsAndOutputs(qnn_model_wrapper, node_unit, std::move(input_names), logger, do_op_validation);
}

Status ResizeOpBuilder::ProcessOpAttrsAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                 const NodeUnit& node_unit,
                                                 std::vector<std::string>&& input_names,
                                                 const logging::Logger& logger,
                                                 bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);
  NodeAttrHelper node_helper(node_unit);
  const std::string resize_mode = GetOnnxAttr(node_helper, onnx_mode_attr);
  std::string qnn_node_type = "ResizeNearestNeighbor";
  if ("linear" == resize_mode) {
    qnn_node_type = "ResizeBilinear";
  }

  const std::string coordinate_mode = GetOnnxAttr(node_helper, onnx_coord_transf_mode_attr);

  Qnn_Scalar_t qnn_align_corners = QNN_SCALAR_INIT;
  qnn_align_corners.dataType = QNN_DATATYPE_BOOL_8;
  qnn_align_corners.bool8Value = static_cast<uint8_t>(0);

  Qnn_Scalar_t qnn_half_pixel = QNN_SCALAR_INIT;
  qnn_half_pixel.dataType = QNN_DATATYPE_BOOL_8;
  qnn_half_pixel.bool8Value = static_cast<uint8_t>(0);

  if ("align_corners" == coordinate_mode) {
    qnn_align_corners.bool8Value = static_cast<uint8_t>(1);
  } else if ("half_pixel" == coordinate_mode) {
    qnn_half_pixel.bool8Value = static_cast<uint8_t>(1);
  }
  QnnParamWrapper qnn_align_corners_param(node_unit.Index(), node_unit.Name(),
                                          QNN_OP_RESIZE_BILINEAR_PARAM_ALIGN_CORNERS, qnn_align_corners);
  QnnParamWrapper qnn_half_pixel_param(node_unit.Index(), node_unit.Name(),
                                       QNN_OP_RESIZE_BILINEAR_PARAM_HALF_PIXEL_CENTERS, qnn_half_pixel);

  std::vector<std::string> param_tensor_names;
  param_tensor_names.push_back(qnn_align_corners_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(qnn_align_corners_param));
  param_tensor_names.push_back(qnn_half_pixel_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(qnn_half_pixel_param));

  return ProcessOutputs(qnn_model_wrapper, node_unit, std::move(input_names), std::move(param_tensor_names),
                        logger, false, do_op_validation, qnn_node_type);
}

Status ResizeOpBuilder::ProcessQDQOpAttrsAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                    const NodeUnit& node_unit,
                                                    std::vector<std::string>&& input_names,
                                                    const logging::Logger& logger,
                                                    bool do_op_validation) const {
  std::vector<std::string> param_tensor_names;
  NodeAttrHelper node_helper(node_unit);

  const std::string interp_mode = GetOnnxAttr(node_helper, onnx_mode_attr);
  const std::string transformation_mode = GetOnnxAttr(node_helper, onnx_coord_transf_mode_attr);
  std::string qnn_op_type = "Resize";

  // Handle Resize with {mode: "nearest", coordinate_transformation_mode: "asymmetric"} uniquely.
  // QNN's ONNX converter tool translates this configuration (regardless of rounding mode)
  // to QNN's ResizeNearestNeighbor with {align_corners: 0, half_pixel: 0}.
  //
  // NOTE: This is only accurate if the rounding mode is "floor". Need to investigate how to handle
  // other rounding modes with Qualcomm. Ideally, we would use QNN's Resize operator, but it doesn't support
  // the "asymmetric" coordinate transformation mode on HTP.
  if (interp_mode == "nearest" && transformation_mode == "asymmetric") {
    qnn_op_type = "ResizeNearestNeighbor";

    // Set parameter 'align_corners' to 0
    Qnn_Scalar_t qnn_align_corners = QNN_SCALAR_INIT;
    qnn_align_corners.dataType = QNN_DATATYPE_BOOL_8;
    qnn_align_corners.bool8Value = static_cast<uint8_t>(0);
    QnnParamWrapper qnn_align_corners_param(node_unit.Index(), node_unit.Name(),
                                            QNN_OP_RESIZE_BILINEAR_PARAM_ALIGN_CORNERS, qnn_align_corners);
    param_tensor_names.push_back(qnn_align_corners_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(qnn_align_corners_param));

    // Set parameter 'half_pixel_centers' to 0
    Qnn_Scalar_t qnn_half_pixel = QNN_SCALAR_INIT;
    qnn_half_pixel.dataType = QNN_DATATYPE_BOOL_8;
    qnn_half_pixel.bool8Value = static_cast<uint8_t>(0);
    QnnParamWrapper qnn_half_pixel_param(node_unit.Index(), node_unit.Name(),
                                         QNN_OP_RESIZE_BILINEAR_PARAM_HALF_PIXEL_CENTERS, qnn_half_pixel);
    param_tensor_names.push_back(qnn_half_pixel_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(qnn_half_pixel_param));
  } else {
    // Parameter 'transformation_mode'
    Qnn_Scalar_t qnn_transformation_mode = QNN_SCALAR_INIT;
    qnn_transformation_mode.dataType = QNN_DATATYPE_UINT_32;
    ORT_RETURN_IF_ERROR(GetQnnModeFromString(supported_coord_transf_modes, transformation_mode,
                                             "coordinate_transformation_mode", qnn_transformation_mode.uint32Value));

    QnnParamWrapper qnn_transformation_mode_param(node_unit.Index(), node_unit.Name(), QNN_OP_RESIZE_PARAM_TRANSFORMATION_MODE,
                                                  qnn_transformation_mode);
    param_tensor_names.push_back(qnn_transformation_mode_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(qnn_transformation_mode_param));

    // Parameter 'exclude_outside'
    Qnn_Scalar_t qnn_exclude_outside = QNN_SCALAR_INIT;
    qnn_exclude_outside.dataType = QNN_DATATYPE_BOOL_8;
    qnn_exclude_outside.bool8Value = static_cast<uint8_t>(GetOnnxAttr(node_helper, onnx_exclude_outside_attr) != 0);

    QnnParamWrapper qnn_exclude_outside_param(node_unit.Index(), node_unit.Name(), QNN_OP_RESIZE_PARAM_EXCLUDE_OUTSIDE,
                                              qnn_exclude_outside);
    param_tensor_names.push_back(qnn_exclude_outside_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(qnn_exclude_outside_param));

    // Parameter 'interpolation_mode'
    Qnn_Scalar_t qnn_interp_mode = QNN_SCALAR_INIT;
    qnn_interp_mode.dataType = QNN_DATATYPE_UINT_32;
    ORT_RETURN_IF_ERROR(GetQnnModeFromString(supported_modes, interp_mode, "mode", qnn_interp_mode.uint32Value));

    QnnParamWrapper qnn_interp_mode_param(node_unit.Index(), node_unit.Name(), QNN_OP_RESIZE_PARAM_INTERPOLATION_MODE,
                                          qnn_interp_mode);
    param_tensor_names.push_back(qnn_interp_mode_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(qnn_interp_mode_param));

    // Parameter 'nearest_mode'. Processed only when 'interpolation_mode' is NEAREST(0).
    if (qnn_interp_mode.uint32Value == 0) {
      const std::string nearest_mode = GetOnnxAttr(node_helper, onnx_nearest_mode_attr);
      Qnn_Scalar_t qnn_nearest_mode = QNN_SCALAR_INIT;
      qnn_nearest_mode.dataType = QNN_DATATYPE_UINT_32;
      ORT_RETURN_IF_ERROR(GetQnnModeFromString(supported_nearest_modes, nearest_mode, "nearest_mode",
                                               qnn_nearest_mode.uint32Value));

      QnnParamWrapper qnn_nearest_mode_param(node_unit.Index(), node_unit.Name(), QNN_OP_RESIZE_PARAM_NEAREST_MODE,
                                             qnn_nearest_mode);
      param_tensor_names.push_back(qnn_nearest_mode_param.GetParamTensorName());
      qnn_model_wrapper.AddParamWrapper(std::move(qnn_nearest_mode_param));
    }
  }

  return ProcessOutputs(qnn_model_wrapper, node_unit, std::move(input_names), std::move(param_tensor_names),
                        logger, true, do_op_validation, qnn_op_type);
}

void CreateResizeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<ResizeOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
