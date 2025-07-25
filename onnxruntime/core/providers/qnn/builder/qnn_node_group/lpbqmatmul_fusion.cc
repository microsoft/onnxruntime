#include <gsl/gsl>
#include <algorithm>
#include <cassert>
#include <limits>
#include <optional>
#include <utility>

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"
#include "core/providers/qnn/builder/qnn_node_group/lpbqmatmul_fusion.h"

namespace onnxruntime {
namespace qnn {

static Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper,
                                    const NodeUnit& scale_dql_node_unit,
                                    const NodeUnit& w_ql_node_unit,
                                    const NodeUnit& matmul_node_unit,
                                    const logging::Logger& logger,
                                    bool validate);

std::unique_ptr<IQnnNodeGroup> LowPowerBlockQuantizedMatMulFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const NodeUnit& matmul_node_unit,
    const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
    const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const logging::Logger& logger) {
  ORT_UNUSED_PARAMETER(logger);

  // Only HTP supports LPBQ encoding format
  // Looking for a MatMul to start search for MatMul w/ LPBQ encodings pattern.
  if (!IsNpuBackend(qnn_model_wrapper.GetQnnBackendType()) || matmul_node_unit.OpType() != "MatMul") {
    return nullptr;
  }

  const GraphViewer& graph_viewer = qnn_model_wrapper.GetGraphViewer();

  // Get QuantizeLinear on Weight (input 1) of MatMul node
  const NodeUnit* p_w_ql_node_unit = GetParentOfInput(graph_viewer,
                                                      matmul_node_unit,
                                                      matmul_node_unit.Inputs()[1],
                                                      node_to_node_unit,
                                                      node_unit_to_qnn_node_group);
  if (p_w_ql_node_unit == nullptr || p_w_ql_node_unit->OpType() != "QuantizeLinear") {
    return nullptr;
  }

  // Check if input of QuantizeLinear is constant initializer
  if (!qnn_model_wrapper.IsConstantInput(p_w_ql_node_unit->Inputs()[0].node_arg.Name())) {
    return nullptr;
  }

  // Get DequantizeLinear node unit contains per-block int scales and per-channel float scales
  const std::array<std::string_view, 1> w_ql_parent_types = {"DequantizeLinear"};
  const NodeUnit* p_scale_dql_node_unit = GetParentOfType(graph_viewer,
                                                          *p_w_ql_node_unit,
                                                          w_ql_parent_types,
                                                          node_to_node_unit,
                                                          node_unit_to_qnn_node_group);
  if (p_scale_dql_node_unit == nullptr) {
    return nullptr;
  }

  TensorInfo pc_scales_tensor_info = {};
  if (Status status = qnn_model_wrapper.GetTensorInfo(p_scale_dql_node_unit->Inputs()[0], pc_scales_tensor_info);
      !status.IsOK()) {
    return nullptr;
  }
  // Check if input 0 of DequantizeLinear is constant initializer and has per-channel float scales
  if (!pc_scales_tensor_info.is_initializer || !pc_scales_tensor_info.quant_param.IsPerChannel()) {
    return nullptr;
  }

  if (Status status = CreateOrValidateOnQnn(qnn_model_wrapper,
                                            *p_scale_dql_node_unit,
                                            *p_w_ql_node_unit,
                                            matmul_node_unit,
                                            logger,
                                            true);
      !status.IsOK()) {
    return nullptr;
  }

  return std::make_unique<LowPowerBlockQuantizedMatMulFusion>(*p_scale_dql_node_unit,
                                                              *p_w_ql_node_unit,
                                                              matmul_node_unit);
}

LowPowerBlockQuantizedMatMulFusion::LowPowerBlockQuantizedMatMulFusion(const NodeUnit& Scale_DQL_node_unit,
                                                                       const NodeUnit& W_QL_node_unit,
                                                                       const NodeUnit& MatMul_node_unit)
    : node_units_{&Scale_DQL_node_unit,
                  &W_QL_node_unit,
                  &MatMul_node_unit} {
}

Status LowPowerBlockQuantizedMatMulFusion::IsSupported(QnnModelWrapper& qmw, const logging::Logger& logger) const {
  return CreateOrValidateOnQnn(qmw, *node_units_[0], *node_units_[1], *node_units_[2], logger, true);
}

Status LowPowerBlockQuantizedMatMulFusion::AddToModelBuilder(QnnModelWrapper& qmw, const logging::Logger& logger) const {
  return CreateOrValidateOnQnn(qmw, *node_units_[0], *node_units_[1], *node_units_[2], logger, false);
}

gsl::span<const NodeUnit* const> LowPowerBlockQuantizedMatMulFusion::GetNodeUnits() const {
  return node_units_;
}

const NodeUnit* LowPowerBlockQuantizedMatMulFusion::GetTargetNodeUnit() const {
  return node_units_[2];
}

namespace {
// Process input[0] for ONNX MatMul that can be translated to either a QNN MatMul.
Status ProcessInput0(QnnModelWrapper& qnn_model_wrapper,
                     const NodeUnitIODef& input_def,
                     const std::string& original_input_0_name,
                     std::vector<std::string>& input_names,
                     const logging::Logger& logger,
                     bool do_op_validation) {
  TensorInfo input_0_info{};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(input_def, input_0_info));
  bool reshape_input_0 = input_0_info.shape.size() == 1;
  std::string actual_input_0_name = original_input_0_name;

  if (reshape_input_0) {
    actual_input_0_name = original_input_0_name + "_ort_qnn_ep_reshape";
    std::vector<uint32_t> shape_2d{1, input_0_info.shape[0]};
    QnnQuantParamsWrapper quant_param_2d = input_0_info.quant_param.Copy();
    ORT_RETURN_IF_ERROR(quant_param_2d.HandleUnsqueeze<uint32_t>(input_0_info.shape, shape_2d));

    // If input_0 is initializer, unpack it and add the tensor with new quantization parameter and shape.
    // Otherwise, add a Reshape node.
    if (input_0_info.is_initializer) {
      std::vector<uint8_t> unpacked_tensor;
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*input_0_info.initializer_tensor, unpacked_tensor));
      QnnTensorWrapper input_tensorwrapper(actual_input_0_name, QNN_TENSOR_TYPE_STATIC, input_0_info.qnn_data_type,
                                           std::move(quant_param_2d), std::move(shape_2d), std::move(unpacked_tensor));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
    } else {
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(original_input_0_name, actual_input_0_name,
                                                           input_0_info.shape, shape_2d,
                                                           input_0_info.qnn_data_type, input_0_info.quant_param,
                                                           quant_param_2d, do_op_validation,
                                                           qnn_model_wrapper.IsGraphInput(original_input_0_name), false));
    }
  } else {
    if (qnn_model_wrapper.IsQnnTensorWrapperExist(actual_input_0_name)) {
      LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << actual_input_0_name;
    } else {
      QnnTensorWrapper input_0_tensor;
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(input_0_info, actual_input_0_name, input_0_tensor));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_0_tensor)), "Failed to add tensor.");
    }
  }
  input_names.emplace_back(actual_input_0_name);

  return Status::OK();
}

// Utility function to unpack weight tensor and transpose to shape [out_channels][in_channels]
Status UnpackWeightTensorData(const QnnModelWrapper& qnn_model_wrapper,
                              const onnx::TensorProto* weight_tensor_proto,
                              std::vector<uint32_t>& weight_shape,
                              int64_t& input_channel_axis,
                              std::vector<uint8_t>& unpacked_tensor) {
  ORT_RETURN_IF_NOT(weight_tensor_proto != nullptr, "Weight tensor proto is null");

  if (input_channel_axis == 0) {
    // Transpose to keep output_channel at index 0;
    // The current logic that quantizes with LPBQ encodings requires out_channels at index 0
    input_channel_axis = weight_shape.size() - 1;
    return utils::TwoDimensionTranspose(qnn_model_wrapper, weight_shape, *weight_tensor_proto, unpacked_tensor);
  } else {
    // No transpose needed, just unpack the initializer data
    return qnn_model_wrapper.UnpackInitializerData(*weight_tensor_proto, unpacked_tensor);
  }
}

// A utility function to transpose a 2D data
Status TwoDimensionTranspose(std::vector<uint8_t>& data,
                             std::vector<uint32_t>& data_shape,
                             const Qnn_DataType_t element_type) {
  ORT_RETURN_IF_NOT(data_shape.size() == 2, "Expected shape of rank 2");

  std::array<size_t, 2> perm = {1, 0};
  std::vector<uint32_t> output_shape(data_shape.size());
  ORT_RETURN_IF_ERROR((qnn::utils::PermuteShape<uint32_t, size_t>(data_shape, perm, output_shape)));

  const size_t elem_byte_size = qnn::utils::GetElementSizeByType(element_type);
  ORT_RETURN_IF_NOT(elem_byte_size != 0, "Can't get element byte size from given QNN type");

  std::vector<uint8_t> transposed_data(data.size());

  for (size_t row = 0; row < data_shape[0]; row++) {
    for (size_t col = 0; col < data_shape[1]; col++) {
      const size_t src_elem_index = (row * data_shape[1] + col);
      const size_t dst_elem_index = (col * output_shape[1] + row);
      const size_t src_byte_index = src_elem_index * elem_byte_size;
      const size_t dst_byte_index = dst_elem_index * elem_byte_size;
      assert(src_byte_index < data.size());
      assert(dst_byte_index < transposed_data.size());

      std::memcpy(&transposed_data[dst_byte_index], &data[src_byte_index], elem_byte_size);
    }
  }

  data = std::move(transposed_data);     // Update data with transposed data
  data_shape = std::move(output_shape);  // Update parameter with final transposed shape
  return Status::OK();
}

// Process LPBQWeight for ONNX MatMul that can be translated to either a QNN MatMul.
Status ProcessLPBQWeight(QnnModelWrapper& qnn_model_wrapper,
                         const NodeUnit& scale_dql_node_unit,
                         const NodeUnit& w_ql_node_unit,
                         const NodeUnit& matmul_node_unit,
                         std::vector<std::string>& input_names,
                         const logging::Logger& logger) {
  ORT_UNUSED_PARAMETER(logger);
  const NodeUnitIODef& mm_input_1_def = matmul_node_unit.Inputs()[1];
  const NodeUnitIODef& w_ql_input_1_def = w_ql_node_unit.Inputs()[0];

  // get per_channel_float_scale value from Quant param of input[0] of DequantizeLinear
  std::vector<float> per_channel_float_scale;
  const NodeUnitIODef& per_channel_float_def = scale_dql_node_unit.Inputs()[0];
  const std::optional<NodeUnitIODef::QuantParam>& scale_dql_quant_param = per_channel_float_def.quant_param;
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackScales(scale_dql_quant_param->scale.Name(), per_channel_float_scale));

  // get per_block_int_scale value from input[0] of DequantizeLinear
  std::vector<uint8_t> per_block_int_scale;
  const NodeUnitIODef& per_block_int_def = scale_dql_node_unit.Inputs()[0];
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackScales<uint8_t>(per_block_int_def.node_arg.Name(), per_block_int_scale));
  std::vector<int32_t> weight_offset(per_channel_float_scale.size(), 0);
  std::vector<uint32_t> block_scales_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(per_block_int_def.node_arg, block_scales_shape), "Failed to get block_scales shape");

  // Read axis of channels in per-block-int-scales data
  NodeAttrHelper scales_node_helper(scale_dql_node_unit.GetNode());
  auto block_scales_axis = scales_node_helper.Get("axis", static_cast<int64_t>(0));

  // Transpose per-block-int-scales to keep channels at index-0 (QNN LPBQ format requires shape [axis_size][blocks-per-axis])
  if (block_scales_axis == 1) {
    ORT_RETURN_IF_ERROR(TwoDimensionTranspose(per_block_int_scale, block_scales_shape, QNN_DATATYPE_UFIXED_POINT_8));
    block_scales_axis = 0;
  }

  // Extract weight datatype from zeropoint (aka offset) of Input1 Quant param
  const std::optional<NodeUnitIODef::QuantParam>& mm_input_1_quant_param = mm_input_1_def.quant_param;
  bool is_int4_type = false;
  if (mm_input_1_quant_param->zero_point != nullptr) {
    int32_t elem_data_type = 0;
    ORT_RETURN_IF_ERROR(utils::GetOnnxTensorElemDataType(*mm_input_1_quant_param->zero_point, elem_data_type));
    is_int4_type = (elem_data_type == ONNX_NAMESPACE::TensorProto_DataType_INT4) ||
                   (elem_data_type == ONNX_NAMESPACE::TensorProto_DataType_UINT4);
  }

  std::vector<uint32_t> weight_shape;
  std::string weight_tensor_name = w_ql_input_1_def.node_arg.Name();
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(w_ql_input_1_def.node_arg, weight_shape), "Failed to get weight shape");

  // Get attributes like weight data axis, block_size from QuantizeLinear
  NodeAttrHelper helper(w_ql_node_unit.GetNode());
  auto input_channel_axis = helper.Get("axis", static_cast<int64_t>(0));
  if (input_channel_axis < 0) {
    input_channel_axis = weight_shape.size() + input_channel_axis;  // QNN requires positive axis value
  }
  auto block_size = helper.Get("block_size", static_cast<int64_t>(0));

  std::vector<uint8_t> unpacked_tensor;
  const auto& weight_tensor_proto = qnn_model_wrapper.GetConstantTensor(weight_tensor_name);
  // if input_channel_axis = 0, UnpackWeightTensorData will transpose and keep output_channel at 0
  ORT_RETURN_IF_ERROR(UnpackWeightTensorData(qnn_model_wrapper, weight_tensor_proto, weight_shape, input_channel_axis, unpacked_tensor));

  // Quantize weight tensor
  size_t weight_elements = unpacked_tensor.size() / sizeof(float);
  auto float_data = gsl::make_span<const float>(reinterpret_cast<const float*>(unpacked_tensor.data()), weight_elements);
  std::vector<uint8_t> quant_data(weight_elements);

  // weight_data_type = 4 but store in int8 buffer
  size_t output_channel_axis = 0;  // MatMul requires axis to be rank-1
  Qnn_DataType_t weight_data_type = is_int4_type ? QNN_DATATYPE_SFIXED_POINT_4 : QNN_DATATYPE_SFIXED_POINT_8;
  ORT_RETURN_IF_ERROR(qnn::utils::LowPowerBlockQuantizeData(float_data,
                                                            weight_shape,
                                                            per_channel_float_scale,
                                                            per_block_int_scale,
                                                            weight_offset,
                                                            quant_data,
                                                            weight_data_type,
                                                            output_channel_axis,
                                                            block_scales_axis,
                                                            block_size,
                                                            block_scales_shape));

  // MatMul w/ LPBQ requies MatMul(MxK, KxN) and axis = rank-1 (out channels)
  // Transpose Weight to KxN, output_channel_axis is modified to rank-1;
  if (input_channel_axis == 1) {
    ORT_RETURN_IF_ERROR(TwoDimensionTranspose(quant_data, weight_shape, QNN_DATATYPE_SFIXED_POINT_8));
    input_channel_axis = 0;
    output_channel_axis = weight_shape.size() - 1;
  }

  // Construct Quant params for Weight
  QnnQuantParamsWrapper weight_qparams;
  weight_qparams = QnnQuantParamsWrapper(per_channel_float_scale, per_block_int_scale, weight_offset, output_channel_axis, block_size, is_int4_type);

  // Get weight tensor type from input of w_dql_tensor or output_dql_tensor
  Qnn_TensorType_t weight_tensor_type = qnn_model_wrapper.GetTensorType(weight_tensor_name);
  QnnTensorWrapper weight_tensor(weight_tensor_name, weight_tensor_type, QNN_DATATYPE_SFIXED_POINT_8,
                                 std::move(weight_qparams), std::move(weight_shape),
                                 std::move(quant_data));

  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(weight_tensor)), "Failed to add weight");
  input_names.emplace_back(weight_tensor_name);
  return Status::OK();
}
}  // namespace

Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper,
                             const NodeUnit& scale_dql_node_unit,
                             const NodeUnit& w_ql_node_unit,
                             const NodeUnit& matmul_node_unit,
                             const logging::Logger& logger,
                             bool validate) {
  assert(scale_dql_node_unit.OpType() == "DequantizeLinear" &&
         w_ql_node_unit.OpType() == "QuantizeLinear" &&
         matmul_node_unit.OpType() == "MatMul");

  const auto& node_name = utils::GetNodeName(matmul_node_unit);

  std::vector<std::string> input_names;

  // prepare input tensor
  const NodeUnitIODef& input_def = matmul_node_unit.Inputs()[0];
  const std::string& input_tensor_name = input_def.node_arg.Name();
  ORT_RETURN_IF_ERROR(ProcessInput0(qnn_model_wrapper, input_def, input_tensor_name, input_names,
                                    logger, validate));

  // Prepare LowPowerBlockQuantized(LPBQ) Weight
  ORT_RETURN_IF_ERROR(ProcessLPBQWeight(qnn_model_wrapper, scale_dql_node_unit, w_ql_node_unit,
                                        matmul_node_unit, input_names, logger));

  // Prepare Output
  const NodeUnitIODef& output_def = matmul_node_unit.Outputs()[0];
  const std::string& op_output_name = output_def.node_arg.Name();
  QnnTensorWrapper output_tensor;
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(output_def, output_tensor));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor)), "Failed to add output");

  // Create QNN Node and Validate if require.
  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name, QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_MAT_MUL,
                                                    std::move(input_names),
                                                    {op_output_name},
                                                    {},
                                                    validate),
                    "Failed to add fused Matmul node.");

  return Status();
}
}  // namespace qnn
}  // namespace onnxruntime
