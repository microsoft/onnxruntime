// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_utils.h"

#include <core/providers/common.h>

#include "core/providers/shared/utils/utils.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/cpu/tensor/transpose.h"
#include "core/common/safeint.h"

namespace onnxruntime {
namespace qnn {

std::string BaseOpBuilder::GetOpBuilderType() const {
  return op_builder_type_;
}

// Add operator related
Status BaseOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                    const NodeUnit& node_unit,
                                    const logging::Logger& logger) const {
  return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, true);
}

// Add operator related
Status BaseOpBuilder::AddToModelBuilder(QnnModelWrapper& qnn_model_wrapper,
                                        const NodeUnit& node_unit,
                                        const logging::Logger& logger,
                                        bool do_op_validation) const {
  LOGS(logger, VERBOSE) << "QNN node builder is trying to add node. Onnx node name: [" << node_unit.Name()
                        << "] onnx node type: [" << node_unit.OpType() << "].";

  std::vector<std::string> input_names;
  // Inputs & output handling mostly same for most of the Ops, just node attributes are different
  ORT_RETURN_IF_ERROR(ProcessInputs(qnn_model_wrapper, node_unit, logger,
                                    input_names, do_op_validation));

  ORT_RETURN_IF_ERROR(ProcessAttributesAndOutputs(qnn_model_wrapper, node_unit, std::move(input_names),
                                                  logger, do_op_validation));

  return Status::OK();
}

Status BaseOpBuilder::ProcessInput(QnnModelWrapper& qnn_model_wrapper,
                                   const NodeUnitIODef& input,
                                   const logging::Logger& logger,
                                   std::vector<std::string>& input_names) const {
  const auto& input_name = input.node_arg.Name();

  if (qnn_model_wrapper.IsQnnTensorWrapperExist(input_name)) {
    LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << input_name;
    input_names.push_back(input_name);
    return Status::OK();
  }

  QnnTensorWrapper input_tensorwrapper;
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(input, input_tensorwrapper));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
  input_names.push_back(input_name);

  return Status::OK();
}

Status BaseOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                    const NodeUnit& node_unit,
                                    const logging::Logger& logger,
                                    std::vector<std::string>& input_names,
                                    bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);

  const auto& inputs = node_unit.Inputs();
  const auto input_count = GetInputCountQnnRequired(node_unit);
  for (size_t input_i = 0; input_i < input_count; ++input_i) {
    ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[input_i], logger, input_names));
  }

  return Status::OK();
}

Status BaseOpBuilder::AddZeroBiasInput(QnnModelWrapper& qnn_model_wrapper,
                                       const QnnQuantParamsWrapper& input0_qparams,
                                       const QnnQuantParamsWrapper& input1_qparams,
                                       std::vector<uint32_t>&& bias_shape,
                                       const std::string& bias_name,
                                       const logging::Logger& logger,
                                       std::vector<std::string>& input_names) const {
  ORT_UNUSED_PARAMETER(logger);
  // For now, only handle case where input0 is per-tensor quantized and input1 is either per-tensor
  // or per-channel quantized.
  ORT_RETURN_IF_NOT(input0_qparams.IsPerTensor(/*include_bw*/ true) && input1_qparams.IsQuantized(),
                    "QNN EP currently only supports adding a dummy zero bias input for per-tensor ",
                    "input[0] and per-tensor/per-channel input[1]");

  size_t num_bias_elems = 1;
  for (size_t i = 0; i < bias_shape.size(); i++) {
    num_bias_elems *= static_cast<size_t>(bias_shape[i]);
  }

  // Bias static input should be all zeros.
  std::vector<uint8_t> bias_bytes(num_bias_elems * sizeof(int32_t), 0);

  // Bias's quantization scale(s) should be the product of the other inputs' quantization scales.
  // Input[0] is expected to have one scale (per-tensor).
  // If input[1] is per-channel (many scales), then the dummy bias also needs to be per-channel.
  std::vector<float> input0_quant_scales;
  std::vector<float> input1_quant_scales;
  ORT_RETURN_IF_ERROR(input0_qparams.GetScales(input0_quant_scales));
  ORT_RETURN_IF_ERROR(input1_qparams.GetScales(input1_quant_scales));

  const size_t num_bias_scales_offsets = input1_quant_scales.size();
  assert(input0_quant_scales.size() == 1);  // Expected for per-tensor.
  ORT_RETURN_IF_NOT(num_bias_scales_offsets >= input0_quant_scales.size(),
                    "Input[1] should have >= 1 quantization scale values");

  std::vector<float> bias_scales(num_bias_scales_offsets);
  for (size_t i = 0; i < num_bias_scales_offsets; i++) {
    bias_scales[i] = input0_quant_scales[0] * input1_quant_scales[i];
  }

  std::vector<int32_t> bias_offsets(num_bias_scales_offsets, 0);  // Bias's zero-points should be all zeros.
  QnnQuantParamsWrapper bias_qparams;

  if (input1_qparams.IsPerChannel()) {
    bias_qparams = QnnQuantParamsWrapper(bias_scales, bias_offsets, /*axis*/ 0, /*is_int4*/ false);
  } else {
    bias_qparams = QnnQuantParamsWrapper(bias_scales[0], bias_offsets[0]);
  }

  auto tensor_wrapper = QnnTensorWrapper(bias_name, QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_SFIXED_POINT_32,
                                         std::move(bias_qparams), std::move(bias_shape), std::move(bias_bytes));

  qnn_model_wrapper.AddTensorWrapper(std::move(tensor_wrapper));
  input_names.push_back(bias_name);

  return Status::OK();
}

Status BaseOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                  const NodeUnit& node_unit,
                                                  std::vector<std::string>&& input_names,
                                                  const logging::Logger& logger,
                                                  bool do_op_validation) const {
  if (input_names.size() < 1) {
    return Status::OK();
  }

  ORT_RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit, std::move(input_names), {},
                                     logger, do_op_validation, GetQnnOpType(node_unit.OpType())));
  return Status::OK();
}

Status BaseOpBuilder::ProcessOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     std::vector<std::string>&& param_tensor_names,
                                     const logging::Logger& logger,
                                     bool do_op_validation,
                                     const std::string& qnn_op_type) const {
  ORT_UNUSED_PARAMETER(logger);
  // Add output
  // Output part is common for all Ops, only difference is the Op attribute
  const auto& outputs = node_unit.Outputs();
  std::vector<std::string> output_names;
  struct CastNodeInfo {
    std::string node_name;
    std::string input_name;
    std::string output_name;
  };
  std::vector<CastNodeInfo> cast_node_info_vec;

  const auto output_count = GetOutputCountQnnRequired(node_unit);
  for (size_t output_i = 0; output_i < output_count; ++output_i) {
    const auto& output_name = outputs[output_i].node_arg.Name();

    TensorInfo output_info = {};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(outputs[output_i], output_info));

    if (output_info.quant_param.IsQuantized()) {
      ORT_RETURN_IF_ERROR(OverrideOutputQuantParam(qnn_model_wrapper, node_unit, logger, input_names,
                                                   output_i, output_info.qnn_data_type, output_info.quant_param));
    }

    Qnn_DataType_t supported_qnn_data_type = GetSupportedOutputDataType(output_i, output_info.qnn_data_type);
    bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);
    if (supported_qnn_data_type != output_info.qnn_data_type && is_graph_output && !do_op_validation) {
      std::string cast_node_name = output_name + "_ort_qnn_ep_cast";
      std::string cast_input_name = output_name + "_ort_qnn_ep_aux";
      std::vector<uint32_t> cast_output_shape = output_info.shape;
      QnnTensorWrapper cast_input_tensorwrapper(cast_input_name,
                                                QNN_TENSOR_TYPE_NATIVE,
                                                supported_qnn_data_type,
                                                output_info.quant_param.Copy(),
                                                std::move(cast_output_shape));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(cast_input_tensorwrapper)), "Failed to add tensor.");
      output_names.push_back(cast_input_name);
      cast_node_info_vec.push_back({cast_node_name, cast_input_name, output_name});
    } else {
      output_info.qnn_data_type = supported_qnn_data_type;
      output_names.push_back(output_name);
    }
    Qnn_TensorType_t tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
    QnnTensorWrapper output_tensorwrapper(output_name,
                                          tensor_type,
                                          output_info.qnn_data_type,
                                          std::move(output_info.quant_param),
                                          std::move(output_info.shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)), "Failed to add tensor.");
  }

  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetNodeName(node_unit),
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    qnn_op_type,  // Typically GetQnnOpType(), but can be overridden.
                                                    std::move(input_names),
                                                    std::move(output_names),
                                                    std::move(param_tensor_names),
                                                    do_op_validation),
                    "Failed to add node.");
  for (const auto& cast_node_info : cast_node_info_vec) {
    // Insert cast node.
    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(cast_node_info.node_name,
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      "Cast",
                                                      {cast_node_info.input_name},
                                                      {cast_node_info.output_name},
                                                      {}),
                      " Failed to add Cast node");
  }
  return Status::OK();
}

Status BaseOpBuilder::SetOutputQParamEqualToInputIfNearlyEqual(QnnModelWrapper& qnn_model_wrapper,
                                                               const NodeUnit& node_unit,
                                                               const logging::Logger& logger,
                                                               const std::vector<std::string>& input_names,
                                                               size_t input_index,
                                                               size_t output_index,
                                                               Qnn_DataType_t qnn_data_type,
                                                               QnnQuantParamsWrapper& quant_param) const {
  const QnnTensorWrapper& input_tensor_wrapper = qnn_model_wrapper.GetQnnTensorWrapper(input_names[input_index]);
  ORT_RETURN_IF_NOT(input_tensor_wrapper.GetTensorDataType() == qnn_data_type,
                    "Input and output data types do not match");
  const QnnQuantParamsWrapper& input_quant_param = input_tensor_wrapper.GetQnnQuantParams();

  float scale_diff = 0.0f;
  int32_t offset_diff = 0;
  ORT_RETURN_IF_ERROR(CompareQnnQuantParams(quant_param.Get(), input_quant_param.Get(), scale_diff, offset_diff));
  constexpr float NEARLY_EQUAL_THRESHOLD = 1e-9f;
  constexpr float WARN_THRESHOLD = 1e-6f;

  if (scale_diff != 0.0f && offset_diff == 0) {
    if (scale_diff <= NEARLY_EQUAL_THRESHOLD) {
      // Quantization params are nearly equal, so make them equal. This may allow QNN backends to employ certain graph
      // optimizations that improve inference latency.
      LOGS(logger, WARNING) << "QNN EP will override the output quantization parameters for " << node_unit.OpType()
                            << " operators to be equal to the input quantization parameters. Operator name: "
                            << node_unit.Name() << ", input_index: " << input_index << ", output index: "
                            << output_index << ".";
      quant_param = input_quant_param;  // Copy input quantization params to the output.
    } else if (scale_diff <= WARN_THRESHOLD) {
      // Quantization params are just outside of the "nearly equal" threshold, so warn user of potential latency
      // degradation.
      LOGS(logger, WARNING) << "The quantization parameters for the " << node_unit.OpType() << " operator '"
                            << node_unit.Name() << "' are not equal, which may result in latency degradation. "
                            << "input_index: " << input_index << ", output index: " << output_index << ".";
    }
  }

  return Status::OK();
}

template <bool Signed>
static Status TransposeInt4Initializer(const QnnModelWrapper& qnn_model_wrapper,
                                       const ONNX_NAMESPACE::TensorProto& tensor_proto,
                                       gsl::span<const size_t> perm,
                                       /*out*/ std::vector<uint8_t>& transposed_data) {
  auto onnx_type = tensor_proto.data_type();

  if constexpr (Signed) {
    ORT_RETURN_IF_NOT(onnx_type == ONNX_NAMESPACE::TensorProto_DataType_INT4,
                      "Expected to tranpose INT4 TensorProto");
  } else {
    ORT_RETURN_IF_NOT(onnx_type == ONNX_NAMESPACE::TensorProto_DataType_UINT4,
                      "Expected to tranpose UINT4 TensorProto");
  }

  using UnpackedType = typename Int4x2Base<Signed>::UnpackedType;

  // Step 1: Unpack TensorProto of packed 4-bit values to Tensor<int8>
  const TensorShape shape = onnxruntime::utils::GetTensorShapeFromTensorProto(tensor_proto);
  const size_t num_elems = shape.Size();
  std::vector<UnpackedType> int8_data(num_elems);

  const auto* elem_type_i8 = DataTypeImpl::GetTensorType<UnpackedType>()->AsTensorType()->GetElementType();
  Tensor tensor_i8(elem_type_i8, shape, int8_data.data(), OrtMemoryInfo{});  // Tensor does not own data.
  ORT_RETURN_IF_ERROR(onnxruntime::utils::TensorProtoInt4ToTensorInt8(
      Env::Default(), qnn_model_wrapper.GetGraphViewer().ModelPath(), tensor_proto, tensor_i8));

  // Step 2: Transpose Tensor<int8>
  const size_t rank = shape.NumDimensions();
  std::vector<int64_t> new_shape_dims;
  new_shape_dims.reserve(rank);

  for (auto p : perm) {
    new_shape_dims.push_back(shape[p]);
  }

  transposed_data.resize(num_elems * sizeof(UnpackedType));
  Tensor transposed_tensor_i8(elem_type_i8, TensorShape::FromExistingBuffer(new_shape_dims),
                              transposed_data.data(), OrtMemoryInfo{});  // Tensor does not own data.
  ORT_RETURN_IF_ERROR(Transpose::DoTranspose(perm, tensor_i8, transposed_tensor_i8));

  // Step 3: Mask off top 4 bits to workaround a QNN INT4 accuracy bug.
  // Docs explicitly state that masking off top 4 bits should not be required.
  if constexpr (Signed) {
    auto dst = gsl::make_span(reinterpret_cast<UnpackedType*>(transposed_data.data()), num_elems);
    for (size_t i = 0; i < dst.size(); i++) {
      dst[i] &= 0x0F;  // -3 (0b1111_1101) becomes 13 (0b0000_1101)
    }
  }

  return Status::OK();
}

Status BaseOpBuilder::TransposeInitializer(const QnnModelWrapper& qnn_model_wrapper,
                                           const onnx::TensorProto& initializer,
                                           const std::vector<size_t>& perm,
                                           std::vector<uint8_t>& transposed_data) const {
  int32_t onnx_type = initializer.data_type();

  // Handle transposing of INT4 initializers separately for memory efficiency.
  if (onnx_type == ONNX_NAMESPACE::TensorProto_DataType_INT4) {
    return TransposeInt4Initializer<true>(qnn_model_wrapper, initializer, perm, transposed_data);
  }

  if (onnx_type == ONNX_NAMESPACE::TensorProto_DataType_UINT4) {
    return TransposeInt4Initializer<false>(qnn_model_wrapper, initializer, perm, transposed_data);
  }

  // Copy initializer's (TensorProto) data to an ORT Tensor.
  const DataTypeImpl* tensor_dtype = DataTypeImpl::TensorTypeFromONNXEnum(initializer.data_type())->GetElementType();
  const auto tensor_shape_dims = onnxruntime::utils::GetTensorShapeFromTensorProto(initializer);
  TensorShape tensor_shape{tensor_shape_dims};
  AllocatorPtr cpu_allocator = std::make_shared<CPUAllocator>();
  Tensor in_tensor = Tensor(tensor_dtype, tensor_shape, cpu_allocator);
  ORT_RETURN_IF_ERROR(onnxruntime::utils::TensorProtoToTensor(
      Env::Default(), qnn_model_wrapper.GetGraphViewer().ModelPath(), initializer, in_tensor));

  // Determine the new transposed shape.
  auto rank = perm.size();
  std::vector<int64_t> new_tensor_shape_dims;
  new_tensor_shape_dims.reserve(rank);
  for (size_t p : perm) {
    new_tensor_shape_dims.push_back(tensor_shape_dims[p]);
  }

  // Resize the output `transposed_data` buffer to hold the result of the transpose.
  const ONNXTensorElementDataType ort_type = onnxruntime::utils::CApiElementTypeFromProtoType(onnx_type);
  const size_t elem_byte_size = utils::GetElementSizeByType(ort_type);
  const size_t num_elems = tensor_shape_dims.Size();
  transposed_data.resize(num_elems * elem_byte_size);

  // Create an output tensor that does not own the pre-allocated `transposed_data` buffer.
  // DoTranspose() will write the new transposed elements directly into the `transposed_data` buffer.
  // We do this to eliminate unnecessary weight copies.
  Tensor out_tensor(tensor_dtype, TensorShape::FromExistingBuffer(new_tensor_shape_dims),
                    transposed_data.data(), OrtMemoryInfo{});  // Tensor does not own data.
  ORT_RETURN_IF_ERROR(Transpose::DoTranspose(perm, in_tensor, out_tensor));

  return Status::OK();
}

Status BaseOpBuilder::ProcessAxisAttribute(const QnnModelWrapper& qnn_model_wrapper,
                                           const NodeUnit& node_unit,
                                           Qnn_Scalar_t& axis_qnn_scalar,
                                           int32_t& default_axis_value) const {
  const auto& inputs = node_unit.Inputs();
  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].node_arg, input_shape), "Cannot get shape");

  auto rank = static_cast<int32_t>(input_shape.size());
  NodeAttrHelper node_helper(node_unit);
  int32_t onnx_axis = node_helper.Get("axis", default_axis_value);
  if (onnx_axis < 0) {
    onnx_axis += rank;
  }
  ORT_RETURN_IF_NOT((onnx_axis >= 0 && onnx_axis < static_cast<int32_t>(input_shape.size())), "QNN requires axis range [0, rank-1].");
  default_axis_value = onnx_axis;

  bool is_gather_op = (node_unit.OpType() == "Gather");
  if (is_gather_op) {
    axis_qnn_scalar.dataType = QNN_DATATYPE_INT_32;
    axis_qnn_scalar.int32Value = onnx_axis;
  } else {
    axis_qnn_scalar.dataType = QNN_DATATYPE_UINT_32;
    axis_qnn_scalar.uint32Value = static_cast<uint32_t>(onnx_axis);
  }

  return Status::OK();
}

Status DataTypeCheckForCpuBackend(QnnModelWrapper& qnn_model_wrapper, ONNX_NAMESPACE::DataType onnx_tensor_data_type) {
  const auto float_elem_type = ONNX_NAMESPACE::Utils::DataTypeUtils::ToType("float");
  bool is_cpu_backend = (qnn_model_wrapper.GetQnnBackendType() == QnnBackendType::CPU);
  ORT_RETURN_IF(is_cpu_backend && onnx_tensor_data_type != float_elem_type, "QNN CPU backend only support float data type.");

  return Status::OK();
}

}  // namespace qnn
}  // namespace onnxruntime
