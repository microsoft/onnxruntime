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
#include "core/providers/qnn/builder/qnn_node_group/lpbqgemm_fusion.h"

namespace onnxruntime {
namespace qnn {

static Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper,
                                    const NodeUnit& scale_dql_node_unit,
                                    const NodeUnit& w_ql_node_unit,
                                    const NodeUnit& w_dql_node_unit,
                                    const NodeUnit& act_dql_node_unit,
                                    const NodeUnit& gemm_node_unit,
                                    const NodeUnit& output_ql_node_unit,
                                    bool validate);

std::unique_ptr<IQnnNodeGroup> LowPowerBlockQuantizedGemmFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const NodeUnit& gemm_node_unit,
    const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
    const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const logging::Logger& logger) {
  ORT_UNUSED_PARAMETER(logger);

  // Looking for a Gemm to start search for Gemm w/ LPBQ encodings pattern.
  if (gemm_node_unit.OpType() != "Gemm") {
    return nullptr;
  }

  const GraphViewer& graph_viewer = qnn_model_wrapper.GetGraphViewer();

  // Get DequantizeLinear on Activation (input 0) of Gemm node
  const NodeUnit* p_act_dql_node_unit = GetParentOfInput(graph_viewer,
                                                         gemm_node_unit,
                                                         gemm_node_unit.Inputs()[0],
                                                         node_to_node_unit,
                                                         node_unit_to_qnn_node_group);
  if (p_act_dql_node_unit == nullptr || p_act_dql_node_unit->OpType() != "DequantizeLinear") {
    return nullptr;
  }

  // Get DequantizeLinear on Weight (input 1) of Gemm node
  const NodeUnit* p_w_dql_node_unit = GetParentOfInput(graph_viewer,
                                                       gemm_node_unit,
                                                       gemm_node_unit.Inputs()[1],
                                                       node_to_node_unit,
                                                       node_unit_to_qnn_node_group);
  if (p_w_dql_node_unit == nullptr || p_w_dql_node_unit->OpType() != "DequantizeLinear") {
    return nullptr;
  }

  // Get QuantizeLinear on Weight (input 1) of Gemm node (optional)
  const std::array<std::string_view, 1> w_dql_parent_types = {"QuantizeLinear"};
  const NodeUnit* p_w_ql_node_unit = GetParentOfType(graph_viewer,
                                                     *p_w_dql_node_unit,
                                                     w_dql_parent_types,
                                                     node_to_node_unit,
                                                     node_unit_to_qnn_node_group);
  if (p_w_ql_node_unit == nullptr) {
    return nullptr;
  }

  // Check if input of QuantizeLinear is constant initializer
  if (!qnn_model_wrapper.IsConstantInput(p_w_ql_node_unit->Inputs()[0].node_arg.Name())) {
    return nullptr;
  }

  // Get DequantizeLinear contains per-block int scales and per-channel float scales
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

  // Get QuantizeLinear NodeUnit at Gemm output
  const std::array<std::string_view, 1> gemm_child_types = {"QuantizeLinear"};
  const NodeUnit* p_output_ql_node_unit = GetOnlyChildOfType(graph_viewer,
                                                             gemm_node_unit,
                                                             gemm_child_types,
                                                             node_to_node_unit,
                                                             node_unit_to_qnn_node_group);
  if (p_output_ql_node_unit == nullptr) {
    return nullptr;
  }

  if (Status status = CreateOrValidateOnQnn(qnn_model_wrapper,
                                            *p_scale_dql_node_unit,
                                            *p_w_ql_node_unit,
                                            *p_w_dql_node_unit,
                                            *p_act_dql_node_unit,
                                            gemm_node_unit,
                                            *p_output_ql_node_unit,
                                            true);
      !status.IsOK()) {
    return nullptr;
  }

  return std::make_unique<LowPowerBlockQuantizedGemmFusion>(*p_scale_dql_node_unit,
                                                            *p_w_ql_node_unit,
                                                            *p_w_dql_node_unit,
                                                            *p_act_dql_node_unit,
                                                            gemm_node_unit,
                                                            *p_output_ql_node_unit);
}

LowPowerBlockQuantizedGemmFusion::LowPowerBlockQuantizedGemmFusion(const NodeUnit& Scale_DQL_node_unit,
                                                                   const NodeUnit& W_QL_node_unit,
                                                                   const NodeUnit& W_DQL_node_unit,
                                                                   const NodeUnit& Act_DQL_node_unit,
                                                                   const NodeUnit& Gemm_node_unit,
                                                                   const NodeUnit& Output_QL_node_unit)
    : node_units_{&Scale_DQL_node_unit,
                  &W_QL_node_unit,
                  &W_DQL_node_unit,
                  &Act_DQL_node_unit,
                  &Gemm_node_unit,
                  &Output_QL_node_unit} {
}

Status LowPowerBlockQuantizedGemmFusion::IsSupported(QnnModelWrapper& qmw, const logging::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  return CreateOrValidateOnQnn(qmw, *node_units_[0], *node_units_[1], *node_units_[2], *node_units_[3], *node_units_[4], *node_units_[5], true);
}

Status LowPowerBlockQuantizedGemmFusion::AddToModelBuilder(QnnModelWrapper& qmw, const logging::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  return CreateOrValidateOnQnn(qmw, *node_units_[0], *node_units_[1], *node_units_[2], *node_units_[3], *node_units_[4], *node_units_[5], false);
}

gsl::span<const NodeUnit* const> LowPowerBlockQuantizedGemmFusion::GetNodeUnits() const {
  return node_units_;
}

const NodeUnit* LowPowerBlockQuantizedGemmFusion::GetTargetNodeUnit() const {
  return node_units_[4];
}

Status UnpackWeightTensorData(const QnnModelWrapper& qnn_model_wrapper,
                              const onnx::TensorProto* weight_tensor_proto,
                              std::vector<uint32_t>& weight_shape,
                              int64_t input_channel_axis,
                              std::vector<uint8_t>& unpacked_tensor) {
  ORT_RETURN_IF_NOT(weight_tensor_proto != nullptr, "Weight tensor proto is null");

  if (input_channel_axis == 0) {
    // Transpose to keep output_channel at index 0;
    // This is needed for proper LPBQ encoding where output channels must be at dimension 0
    return utils::TwoDimensionTranspose(qnn_model_wrapper, weight_shape, *weight_tensor_proto, unpacked_tensor);
  } else {
    // No transpose needed, just unpack the initializer data
    return qnn_model_wrapper.UnpackInitializerData(*weight_tensor_proto, unpacked_tensor);
  }
}

Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper,
                             const NodeUnit& scale_dql_node_unit,
                             const NodeUnit& w_ql_node_unit,
                             const NodeUnit& w_dql_node_unit,
                             const NodeUnit& act_dql_node_unit,
                             const NodeUnit& gemm_node_unit,
                             const NodeUnit& output_ql_node_unit,
                             bool validate) {
  assert(scale_dql_node_unit.OpType() == "DequantizeLinear" &&
         w_ql_node_unit.OpType() == "QuantizeLinear" &&
         w_dql_node_unit.OpType() == "DequantizeLinear" &&
         act_dql_node_unit.OpType() == "DequantizeLinear" &&
         gemm_node_unit.OpType() == "Gemm" &&
         output_ql_node_unit.OpType() == "QuantizeLinear");
  const auto& node_name = utils::GetNodeName(gemm_node_unit);
  const NodeUnitIODef& act_dql_input_1_def = act_dql_node_unit.Inputs()[0];
  const NodeUnitIODef& w_dql_input_1_def = w_dql_node_unit.Inputs()[0];
  const NodeUnitIODef& w_ql_input_1_def = w_ql_node_unit.Inputs()[0];
  const NodeUnitIODef& output_def = output_ql_node_unit.Outputs()[0];

  // prepare input tensor
  std::string input_tensor_name = act_dql_input_1_def.node_arg.Name();
  QnnTensorWrapper input_tensor;
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(act_dql_input_1_def, input_tensor));
  // get input_scale value from input[1] of DequantizeLinear
  std::vector<float> input_scales;
  const std::optional<NodeUnitIODef::QuantParam>& act_dql_quant_param = act_dql_input_1_def.quant_param;
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackScales(act_dql_quant_param->scale.Name(), input_scales));
  ORT_RETURN_IF_NOT(input_scales.size() == 1,
                    "Got unexpected size of scales on Gemm Input0. Expected per-tensor scales");

  // Prepare LowPowerBlockQuantized(LPBQ) Weight
  QnnQuantParamsWrapper weight_qparams;

  // get per_channel_float_scale value from input[1] of DequantizeLinear
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
  // Get attributes like axis, block_size from QuantizeLinear
  NodeAttrHelper scales_node_helper(scale_dql_node_unit.GetNode());
  auto block_scales_axis = scales_node_helper.Get("axis", static_cast<int64_t>(0));

  const std::optional<NodeUnitIODef::QuantParam>& w_dql_quant_param = w_dql_input_1_def.quant_param;
  bool is_int4_type = false;
  if (w_dql_quant_param->zero_point != nullptr) {
    int32_t elem_data_type = 0;
    ORT_RETURN_IF_ERROR(utils::GetOnnxTensorElemDataType(*w_dql_quant_param->zero_point, elem_data_type));
    is_int4_type = (elem_data_type == ONNX_NAMESPACE::TensorProto_DataType_INT4) ||
                   (elem_data_type == ONNX_NAMESPACE::TensorProto_DataType_UINT4);
  }

  // Get attributes like axis, block_size from QuantizeLinear
  NodeAttrHelper helper(w_ql_node_unit.GetNode());
  auto input_channel_axis = helper.Get("axis", static_cast<int64_t>(0));
  auto block_size = helper.Get("block_size", static_cast<int64_t>(0));

  size_t output_channel_axis = 0;  // Current LowPowerBlockQuantize() support output_channel_axis at index=0;
  weight_qparams = QnnQuantParamsWrapper(per_channel_float_scale, per_block_int_scale, weight_offset, output_channel_axis, block_size, is_int4_type);

  std::vector<uint32_t> weight_shape;
  std::vector<uint8_t> unpacked_tensor;
  std::string weight_tensor_name = w_ql_input_1_def.node_arg.Name();
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(w_ql_input_1_def.node_arg, weight_shape), "Failed to get weight shape");
  Qnn_DataType_t weight_data_type = is_int4_type ? QNN_DATATYPE_SFIXED_POINT_4 : QNN_DATATYPE_SFIXED_POINT_8;
  const auto& weight_tensor_proto = qnn_model_wrapper.GetConstantTensor(weight_tensor_name);
  ORT_RETURN_IF_ERROR(UnpackWeightTensorData(qnn_model_wrapper, weight_tensor_proto, weight_shape, input_channel_axis, unpacked_tensor));

  // Quantize weight tensor
  size_t weight_elements = unpacked_tensor.size() / sizeof(float);
  auto float_data = gsl::make_span<const float>(reinterpret_cast<const float*>(unpacked_tensor.data()), weight_elements);
  std::vector<uint8_t> quant_data(weight_elements);

  // scale = per_channel_float_scale * per_block_int_scale
  // weight_data_type = 4 but store in int8 buffer
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

  // Get weight tensor type from input of w_dql_tensor or output_dql_tensor
  Qnn_TensorType_t weight_tensor_type = qnn_model_wrapper.GetTensorType(weight_tensor_name);
  QnnTensorWrapper weight_tensor(weight_tensor_name, weight_tensor_type, QNN_DATATYPE_SFIXED_POINT_8,
                                 std::move(weight_qparams), std::move(weight_shape),
                                 std::move(quant_data));

  // Prepare Bias tensor;
  // Bias tensor is in FP32 and need to quantize to datatype of input 0 of Gemm
  QnnTensorWrapper bias_tensor;
  const NodeUnitIODef* bias_def_ptr = nullptr;
  bool has_bias = gemm_node_unit.Inputs().size() == 3 && gemm_node_unit.Inputs()[2].node_arg.Exists();
  if (has_bias) {
    bias_def_ptr = &gemm_node_unit.Inputs()[2];
    std::vector<uint32_t> bias_shape;
    std::string bias_tensor_name = bias_def_ptr->node_arg.Name();
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(bias_def_ptr->node_arg, bias_shape), "Failed to get bias shape");
    Qnn_DataType_t bias_data_type = QNN_DATATYPE_SFIXED_POINT_32;
    Qnn_TensorType_t bias_tensor_type = qnn_model_wrapper.GetTensorType(bias_tensor_name);
    const auto& bias_tensor_proto = qnn_model_wrapper.GetConstantTensor(bias_tensor_name);

    // Prepare scale for bias as input scale * per_channel_float_scales;
    std::vector<float> bias_scales(per_channel_float_scale.size());
    std::transform(per_channel_float_scale.begin(), per_channel_float_scale.end(), bias_scales.begin(),
                   [input_scales](float x) { return x * input_scales[0]; });

    // bias_offset = vector of zero's
    std::vector<int32_t> bias_offsets(bias_scales.size(), 0);

    // Read bias tensor buffer
    std::vector<uint8_t> unpacked_bias_tensor;
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*bias_tensor_proto, unpacked_bias_tensor));

    // Quantize bias tensor
    size_t bias_elements = unpacked_bias_tensor.size() / sizeof(float);
    auto bias_float_data = gsl::make_span<const float>(reinterpret_cast<const float*>(unpacked_bias_tensor.data()), bias_elements);
    std::vector<uint8_t> bias_quant_data(bias_elements * qnn::utils::GetElementSizeByType(bias_data_type));

    ORT_RETURN_IF_ERROR(qnn::utils::QuantizeData(bias_float_data, bias_shape, bias_scales, bias_offsets, bias_quant_data, bias_data_type, /*axis*/ 0));
    QnnQuantParamsWrapper bias_qparams(bias_scales, bias_offsets, /*axis*/ 0, /*is_int4*/ 0);

    bias_tensor = QnnTensorWrapper(bias_tensor_name, bias_tensor_type, bias_data_type,
                                   std::move(bias_qparams), std::move(bias_shape),
                                   std::move(bias_quant_data));
  }

  // Prepare Output
  QnnTensorWrapper output_tensor;
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(output_def, output_tensor));

  if (validate) {
    std::vector<Qnn_Tensor_t> input_tensors = {input_tensor.GetQnnTensor(), weight_tensor.GetQnnTensor()};
    if (has_bias) {
      input_tensors.emplace_back(bias_tensor.GetQnnTensor());
    }

    ORT_RETURN_IF_ERROR(qnn_model_wrapper.ValidateQnnNode(node_name,
                                                          QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                          QNN_OP_FULLY_CONNECTED,
                                                          std::move(input_tensors),
                                                          {output_tensor.GetQnnTensor()},
                                                          {}));
  } else {
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensor)), "Failed to add input");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(weight_tensor)), "Failed to add weight");
    if (has_bias) {
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(bias_tensor)), "Failed to add bias");
    }

    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor)), "Failed to add output");
    std::vector<std::string> input_names = {input_tensor_name, weight_tensor_name};
    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name,
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_MAT_MUL,
                                                      std::move(input_names),
                                                      {output_def.node_arg.Name()},
                                                      {},
                                                      validate),
                      "Failed to Fuse Gemm w/ LPBQ Sequence into MatMul w/ LPBQ encodings.");
  }

  return Status();
}
}  // namespace qnn
}  // namespace onnxruntime
