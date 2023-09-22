// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/common/safeint.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace qnn {

class GatherOpBuilder : public BaseOpBuilder {
 public:
  GatherOpBuilder() : BaseOpBuilder("GatherOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GatherOpBuilder);

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
};

class GatherElementsOpBuilder : public BaseOpBuilder {
 public:
  GatherElementsOpBuilder() : BaseOpBuilder("GatherElementsOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GatherElementsOpBuilder);

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
};

class GatherNDOpBuilder : public BaseOpBuilder {
 public:
  GatherNDOpBuilder() : BaseOpBuilder("GatherNDOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GatherNDOpBuilder);

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
};

// Converts int64 indices to another integer type (typically int32 or uint32).
// The input and output are both represented as byte arrays.
template <typename T>
static void ConvertInt64IndicesBytes(const std::vector<uint8_t>& onnx_bytes, std::vector<uint8_t>& qnn_bytes) {
  const size_t num_elems = onnx_bytes.size() / sizeof(uint64_t);
  gsl::span<const uint64_t> onnx_indices{reinterpret_cast<const uint64_t*>(onnx_bytes.data()), num_elems};

  qnn_bytes.resize(num_elems * sizeof(T));
  T* qnn_indices_ptr = reinterpret_cast<T*>(qnn_bytes.data());

  std::transform(onnx_indices.begin(), onnx_indices.end(), qnn_indices_ptr,
                 [](int64_t index) { return SafeInt<T>(index); });
}

// Processes the indices input to Gather operators.
//
// Gather ops on the QNN CPU backend require int32 indices, so this function will either add a Cast operator
// to dynamic indices or transform static indices to int32/uint32.
//
// The HTP backend does not support int64, so this function returns an error status if dynamic indices are of
// type int64. If the indices are static, then this function will convert them to int32/uint32.
static Status ProcessIndicesInput(QnnModelWrapper& qnn_model_wrapper,
                                  const NodeUnitIODef& indices_input,
                                  bool int32_type_is_signed,
                                  const logging::Logger& logger,
                                  std::vector<std::string>& input_names,
                                  bool do_op_validation) {
  Qnn_DataType_t desired_data_type = int32_type_is_signed ? QNN_DATATYPE_INT_32 : QNN_DATATYPE_UINT_32;

  const auto& input_name = indices_input.node_arg.Name();
  if (qnn_model_wrapper.IsQnnTensorWrapperExist(input_name)) {
    LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << input_name;
    input_names.push_back(input_name);
    return Status::OK();
  }

  std::string indices_input_name(input_name);
  Qnn_DataType_t qnn_data_type = desired_data_type;
  const auto* type_proto = indices_input.node_arg.TypeAsProto();
  ORT_RETURN_IF_ERROR(utils::GetQnnDataType(false, type_proto, qnn_data_type));

  std::vector<uint8_t> gather_indices;
  bool is_initializer_input = qnn_model_wrapper.IsInitializerInput(input_name);

  // Gather input 0 is quantized tensor, input 1 (indices) is int64, this is not supported by QNN
  bool is_npu_backend = IsNpuBackend(qnn_model_wrapper.GetQnnBackendType());
  ORT_RETURN_IF(is_npu_backend && qnn_data_type == QNN_DATATYPE_INT_64 && !is_initializer_input,
                "HTP backend doesn't support any int64 data type.");

  if (is_initializer_input) {
    const auto& input_tensor = qnn_model_wrapper.GetInitializerTensors().at(input_name);
    std::vector<uint8_t> unpacked_tensor;

    ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*input_tensor, unpacked_tensor));

    if (qnn_data_type == QNN_DATATYPE_INT_64) {
      if (desired_data_type == QNN_DATATYPE_INT_32) {
        ConvertInt64IndicesBytes<int32_t>(unpacked_tensor, gather_indices);
      } else {
        ConvertInt64IndicesBytes<uint32_t>(unpacked_tensor, gather_indices);
      }
    } else {
      gather_indices = std::move(unpacked_tensor);
    }
    qnn_data_type = desired_data_type;
  }

  // Even for Quantized model, Gather indices use int32 without quantization
  Qnn_QuantizeParams_t quantize_param = QNN_QUANTIZE_PARAMS_INIT;

  Qnn_TensorType_t tensor_type = GetInputTensorType(qnn_model_wrapper, input_name);
  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(indices_input.node_arg, input_shape), "Cannot get shape");
  std::vector<uint32_t> cast_output_shape(input_shape);
  QnnTensorWrapper input_tensorwrapper(input_name, tensor_type, qnn_data_type, quantize_param,
                                       std::move(input_shape), std::move(gather_indices));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");

  if (!is_initializer_input && qnn_data_type == QNN_DATATYPE_INT_64) {
    // Insert cast node int64 -> int32/uint32
    if (qnn_data_type == QNN_DATATYPE_INT_64) {
      // Add Cast node for indices
      indices_input_name = input_name + "_ort_qnn_ep_cast";
      QnnTensorWrapper cast_output(indices_input_name, QNN_TENSOR_TYPE_NATIVE, desired_data_type, quantize_param,
                                   std::move(cast_output_shape));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(cast_output)), "Failed to add tensor.");
      ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(indices_input_name,
                                                        QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                        "Cast",
                                                        {input_name},
                                                        {indices_input_name},
                                                        {},
                                                        do_op_validation),
                        "Failed to add node.");
    }
  }

  input_names.push_back(indices_input_name);

  return Status::OK();
}

Status GatherOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                      const NodeUnit& node_unit,
                                      const logging::Logger& logger,
                                      std::vector<std::string>& input_names,
                                      bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  ORT_RETURN_IF(inputs.size() != 2, "QNN EP: Gather operator must have two inputs");
  ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[0], logger, input_names));

  return ProcessIndicesInput(qnn_model_wrapper, inputs[1], true, logger, input_names, do_op_validation);
}

Status GatherElementsOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                              const NodeUnit& node_unit,
                                              const logging::Logger& logger,
                                              std::vector<std::string>& input_names,
                                              bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  ORT_RETURN_IF(inputs.size() != 2, "QNN EP: GatherElements operator must have two inputs");
  ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[0], logger, input_names));

  return ProcessIndicesInput(qnn_model_wrapper, inputs[1], false, logger, input_names, do_op_validation);
}

Status GatherNDOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                        const NodeUnit& node_unit,
                                        const logging::Logger& logger,
                                        std::vector<std::string>& input_names,
                                        bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  ORT_RETURN_IF(inputs.size() != 2, "QNN EP: GatherND operator must have two inputs");
  ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[0], logger, input_names));

  return ProcessIndicesInput(qnn_model_wrapper, inputs[1], false, logger, input_names, do_op_validation);
}

Status GatherOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                    const NodeUnit& node_unit,
                                                    std::vector<std::string>&& input_names,
                                                    const logging::Logger& logger,
                                                    bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);
  std::vector<std::string> param_tensor_names;
  int32_t axis_value = 0;
  Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
  ORT_RETURN_IF_ERROR(ProcessAxisAttribute(qnn_model_wrapper, node_unit, axis_qnn_scalar, axis_value));
  QnnParamWrapper axis_param(node_unit.Index(), node_unit.Name(), QNN_OP_GATHER_PARAM_AXIS, axis_qnn_scalar);
  param_tensor_names.push_back(axis_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(axis_param));

  // if indicies is scalar shape, then need to add Reshape node
  const auto& input_tensor_wrapper = qnn_model_wrapper.GetQnnTensorWrapper(input_names[0]);
  const auto& indices_input_tensor_wrapper = qnn_model_wrapper.GetQnnTensorWrapper(input_names[1]);

  // Calculate the output shape
  std::vector<uint32_t> qnn_output_shape;
  auto input_rank = input_tensor_wrapper.GetTensorRank();
  auto indices_rank = indices_input_tensor_wrapper.GetTensorRank();
  qnn_output_shape.reserve(static_cast<size_t>(input_rank - 1 + indices_rank));

  const auto& gather_indices = node_unit.Inputs()[1];
  std::vector<uint32_t> indices_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(gather_indices.node_arg, indices_shape),
                    "Cannot get shape");

  // replace the dimension for p.axis with the shape from the indices
  std::copy(input_tensor_wrapper.GetTensorDims().begin(), input_tensor_wrapper.GetTensorDims().begin() + axis_value,
            std::back_inserter(qnn_output_shape));

  const auto& indicies_shape = indices_input_tensor_wrapper.GetTensorDims();
  std::copy(indicies_shape.begin(), indicies_shape.end(), std::back_inserter(qnn_output_shape));

  std::copy(input_tensor_wrapper.GetTensorDims().begin() + axis_value + 1, input_tensor_wrapper.GetTensorDims().end(),
            std::back_inserter(qnn_output_shape));

  const auto& gather_output = node_unit.Outputs()[0];
  const auto& output_name = gather_output.node_arg.Name();

  Qnn_QuantizeParams_t quantize_param = QNN_QUANTIZE_PARAMS_INIT;
  bool is_quantized_tensor = gather_output.quant_param.has_value();
  utils::InitializeQuantizeParam(quantize_param, is_quantized_tensor);

  const auto* type_proto = gather_output.node_arg.TypeAsProto();
  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;
  ORT_RETURN_IF_ERROR(utils::GetQnnDataType(is_quantized_tensor, type_proto, qnn_data_type));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.ProcessQuantizationParameter(gather_output.quant_param,
                                                                   quantize_param.scaleOffsetEncoding.scale,
                                                                   quantize_param.scaleOffsetEncoding.offset),
                    "Cannot get quantization parameter");
  std::vector<uint32_t> target_output_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(gather_output.node_arg, target_output_shape),
                    "Cannot get shape");

  bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);
  bool reshape_required = (qnn_output_shape.size() != target_output_shape.size());
  std::string gather_output_name = output_name + (reshape_required ? "_ort_qnn_ep_reshape" : "");
  Qnn_TensorType_t tensor_type = (!reshape_required && is_graph_output) ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
  QnnTensorWrapper gather_output_wrapper(gather_output_name, tensor_type, qnn_data_type, quantize_param,
                                         std::move(qnn_output_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(gather_output_wrapper)), "Failed to add tensor.");

  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(GetNodeName(node_unit),
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    GetQnnOpType(node_unit.OpType()),
                                                    std::move(input_names),
                                                    {gather_output_name},
                                                    std::move(param_tensor_names),
                                                    do_op_validation),
                    "Failed to add node.");

  if (reshape_required) {
    // Add Reshape Node after Gather.
    Qnn_TensorType_t reshape_tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
    QnnTensorWrapper reshape_output(output_name, reshape_tensor_type, qnn_data_type, quantize_param,
                                    std::move(target_output_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(reshape_output)), "Failed to add tensor.");
    const static std::string qnn_node_type = "Reshape";
    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(output_name,
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      qnn_node_type,
                                                      {gather_output_name},
                                                      {output_name},
                                                      {},
                                                      do_op_validation),
                      "Failed to add node.");
  }

  return Status::OK();
}

Status GatherElementsOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                            const NodeUnit& node_unit,
                                                            std::vector<std::string>&& input_names,
                                                            const logging::Logger& logger,
                                                            bool do_op_validation) const {
  std::vector<std::string> param_tensor_names;
  int32_t axis_value = 0;
  Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
  ORT_RETURN_IF_ERROR(ProcessAxisAttribute(qnn_model_wrapper, node_unit, axis_qnn_scalar, axis_value));
  QnnParamWrapper axis_param(node_unit.Index(), node_unit.Name(), QNN_OP_GATHER_ELEMENTS_PARAM_AXIS, axis_qnn_scalar);
  param_tensor_names.push_back(axis_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(axis_param));

  return ProcessOutputs(qnn_model_wrapper, node_unit, std::move(input_names), std::move(param_tensor_names),
                        logger, do_op_validation, GetQnnOpType(node_unit.OpType()));
}

Status GatherNDOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                      const NodeUnit& node_unit,
                                                      std::vector<std::string>&& input_names,
                                                      const logging::Logger& logger,
                                                      bool do_op_validation) const {
  std::vector<std::string> param_tensor_names;
  NodeAttrHelper node_attr_helper(node_unit);
  int32_t onnx_batch_dims = node_attr_helper.Get("batch_dims", 0);

  Qnn_Scalar_t qnn_batch_dims_scalar = QNN_SCALAR_INIT;
  qnn_batch_dims_scalar.dataType = QNN_DATATYPE_UINT_32;
  qnn_batch_dims_scalar.uint32Value = SafeInt<uint32_t>(onnx_batch_dims);

  QnnParamWrapper batch_dims_param(node_unit.Index(), node_unit.Name(), QNN_OP_GATHER_ND_PARAM_BATCH_DIMS,
                                   qnn_batch_dims_scalar);

  param_tensor_names.push_back(batch_dims_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(batch_dims_param));

  return ProcessOutputs(qnn_model_wrapper, node_unit, std::move(input_names), std::move(param_tensor_names),
                        logger, do_op_validation, GetQnnOpType(node_unit.OpType()));
}

void CreateGatherOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  if (op_type == "Gather") {
    op_registrations.AddOpBuilder(op_type, std::make_unique<GatherOpBuilder>());
  } else if (op_type == "GatherElements") {
    op_registrations.AddOpBuilder(op_type, std::make_unique<GatherElementsOpBuilder>());
  } else if (op_type == "GatherND") {
    op_registrations.AddOpBuilder(op_type, std::make_unique<GatherNDOpBuilder>());
  }
}

}  // namespace qnn
}  // namespace onnxruntime
