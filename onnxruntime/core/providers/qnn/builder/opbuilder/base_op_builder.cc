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

  OnnxInputInfo input_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetOnnxInputInfo(input, input_info));

  std::vector<uint8_t> unpacked_tensor;
  if (input_info.is_initializer) {
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*input_info.initializer_tensor, unpacked_tensor));
  }

  Qnn_TensorType_t tensor_type = GetInputTensorType(qnn_model_wrapper, input_name);
  QnnTensorWrapper input_tensorwrapper(input_name, tensor_type, input_info.qnn_data_type, input_info.quant_param,
                                       std::move(input_info.shape), std::move(unpacked_tensor));
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

    OnnxInputInfo output_info = {};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetOnnxInputInfo(outputs[output_i], output_info));

    if (output_info.quant_param.encodingDefinition == QNN_DEFINITION_DEFINED) {
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
                                                output_info.quant_param,
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
                                          output_info.quant_param,
                                          std::move(output_info.shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)), "Failed to add tensor.");
  }

  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(GetNodeName(node_unit),
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
                                                               Qnn_QuantizeParams_t& quant_param) const {
  const QnnTensorWrapper& input_tensor_wrapper = qnn_model_wrapper.GetQnnTensorWrapper(input_names[input_index]);
  ORT_RETURN_IF_NOT(input_tensor_wrapper.GetTensorDataType() == qnn_data_type,
                    "Input and output data types do not match");
  Qnn_QuantizeParams_t input_quant_param = GetQnnTensorQParams(input_tensor_wrapper.GetQnnTensor());

  float scale_diff = 0.0f;
  int32_t offset_diff = 0;
  ORT_RETURN_IF_ERROR(CompareQnnQuantParams(quant_param, input_quant_param, scale_diff, offset_diff));
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

Status BaseOpBuilder::TransposeInitializer(const QnnModelWrapper& qnn_model_wrapper,
                                           const onnx::TensorProto& initializer,
                                           const std::vector<size_t>& perm,
                                           std::vector<uint8_t>& transposed_data) const {
  const DataTypeImpl* tensor_dtype = DataTypeImpl::TensorTypeFromONNXEnum(initializer.data_type())->GetElementType();
  const auto tensor_shape_dims = onnxruntime::utils::GetTensorShapeFromTensorProto(initializer);
  TensorShape tensor_shape{tensor_shape_dims};
  AllocatorPtr cpu_allocator = std::make_shared<CPUAllocator>();
  Tensor in_tensor = Tensor(tensor_dtype, tensor_shape, cpu_allocator);

  auto rank = perm.size();
  std::vector<int64_t> new_tensor_shape_dims;
  std::vector<size_t> permutations;
  new_tensor_shape_dims.reserve(rank);
  permutations.reserve(rank);
  for (int64_t p : perm) {
    permutations.push_back(p);
    new_tensor_shape_dims.push_back(tensor_shape_dims[p]);
  }

  TensorShape new_tensor_shape(new_tensor_shape_dims);
  Tensor out_tensor = Tensor(tensor_dtype, new_tensor_shape, cpu_allocator);
  ORT_RETURN_IF_ERROR(onnxruntime::utils::TensorProtoToTensor(Env::Default(), nullptr, initializer, in_tensor));
  ORT_RETURN_IF_ERROR(Transpose::DoTranspose(permutations, in_tensor, out_tensor));
  onnx::TensorProto new_tensor_proto = onnxruntime::utils::TensorToTensorProto(out_tensor, "test");
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(new_tensor_proto, transposed_data));

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
  ORT_ENFORCE((onnx_axis >= 0 && onnx_axis < static_cast<int32_t>(input_shape.size())), "QNN requires axis range [0, rank-1].");
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

Qnn_TensorType_t BaseOpBuilder::GetInputTensorType(const QnnModelWrapper& qnn_model_wrapper, const std::string& input_name) const {
  if (qnn_model_wrapper.IsInitializerInput(input_name)) {
    return QNN_TENSOR_TYPE_STATIC;
  } else if (qnn_model_wrapper.IsGraphInput(input_name)) {
    return QNN_TENSOR_TYPE_APP_WRITE;
  } else if (qnn_model_wrapper.IsGraphOutput(input_name)) {
    return QNN_TENSOR_TYPE_APP_READ;
  } else {
    return QNN_TENSOR_TYPE_NATIVE;
  }
}

Status DataTypeCheckForCpuBackend(QnnModelWrapper& qnn_model_wrapper, ONNX_NAMESPACE::DataType onnx_tensor_data_type) {
  const auto float_elem_type = ONNX_NAMESPACE::Utils::DataTypeUtils::ToType("float");
  bool is_cpu_backend = (qnn_model_wrapper.GetQnnBackendType() == QnnBackendType::CPU);
  ORT_RETURN_IF(is_cpu_backend && onnx_tensor_data_type != float_elem_type, "QNN CPU backend only support float data type.");

  return Status::OK();
}

}  // namespace qnn
}  // namespace onnxruntime
