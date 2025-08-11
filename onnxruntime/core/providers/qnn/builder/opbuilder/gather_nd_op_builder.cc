// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cassert>
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

// Handles GatherND
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

// Fixes negative indices and converts int64 to uint32 for GatherND
template <typename SrcType, typename DstType>
bool FixStaticIndicesForGatherND(const std::vector<uint8_t>& onnx_bytes,
                                 const std::vector<int64_t>& indices_shape,
                                 const std::vector<int64_t>& data_shape,
                                 int64_t batch_dims,
                                 std::vector<uint8_t>& qnn_bytes) {
  const int64_t index_tuple_size = indices_shape.back();
  const size_t num_tuples = onnx_bytes.size() / (index_tuple_size * sizeof(SrcType));

  gsl::span<const SrcType> onnx_indices{
      reinterpret_cast<const SrcType*>(onnx_bytes.data()), num_tuples * index_tuple_size};

  qnn_bytes.resize(num_tuples * index_tuple_size * sizeof(DstType));
  gsl::span<DstType> qnn_indices{
      reinterpret_cast<DstType*>(qnn_bytes.data()), num_tuples * index_tuple_size};

  for (size_t i = 0; i < num_tuples; ++i) {
    for (int64_t j = 0; j < index_tuple_size; ++j) {
      SrcType idx = onnx_indices[i * index_tuple_size + j];
      int64_t dim = data_shape[batch_dims + j];

      if (idx < 0) {
        idx += static_cast<SrcType>(dim);
      }

      if (idx < 0 || static_cast<int64_t>(idx) >= dim) {
        return false;  // Out-of-bounds index
      }

      qnn_indices[i * index_tuple_size + j] = static_cast<DstType>(idx);
    }
  }

  return true;
}

Status GatherNDOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                        const NodeUnit& node_unit,
                                        const logging::Logger& logger,
                                        std::vector<std::string>& input_names,
                                        bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();

  ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[0], logger, input_names));

  const auto& data_input = inputs[0];
  const auto& indices_input = inputs[1];
  const auto& indices_tensor_name = indices_input.node_arg.Name();

  TensorInfo indices_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(indices_input, indices_info));

  std::vector<uint8_t> qnn_indices_bytes;

  if (indices_info.is_initializer) {
    std::vector<uint8_t> onnx_indices_bytes;
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*indices_info.initializer_tensor, onnx_indices_bytes));

    std::vector<uint32_t> data_shape;
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(data_input.node_arg, data_shape),
                      "Failed to get data shape for GatherND.");

    std::vector<uint32_t> indices_shape;
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(indices_input.node_arg, indices_shape),
                      "Failed to get indices shape for GatherND.");

    if (indices_shape.empty()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Indices shape is empty for GatherND.");
    }

    // Get batch_dims for proper index processing
    NodeAttrHelper node_helper(node_unit);
    int64_t batch_dims = node_helper.Get("batch_dims", static_cast<int64_t>(0));

    if (indices_info.qnn_data_type == QNN_DATATYPE_INT_64) {
      ORT_RETURN_IF_NOT((
                            FixStaticIndicesForGatherND<int64_t, int32_t>(
                                onnx_indices_bytes,
                                std::vector<int64_t>(indices_shape.begin(), indices_shape.end()),
                                std::vector<int64_t>(data_shape.begin(), data_shape.end()),
                                batch_dims,
                                qnn_indices_bytes)),
                        "QNN does not support negative or out-of-bounds indices for GatherND.");
      indices_info.qnn_data_type = QNN_DATATYPE_INT_32;
    } else {
      qnn_indices_bytes = std::move(onnx_indices_bytes);
    }
  }

  Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(indices_tensor_name);
  std::vector<uint32_t> cast_output_shape(indices_info.shape);

  if (!qnn_model_wrapper.IsQnnTensorWrapperExist(indices_tensor_name)) {
    QnnTensorWrapper input_tensorwrapper(indices_tensor_name, tensor_type, indices_info.qnn_data_type,
                                         QnnQuantParamsWrapper(), std::move(indices_info.shape),
                                         std::move(qnn_indices_bytes));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
  }

  std::string indices_casted_name{indices_tensor_name};
  if (indices_info.qnn_data_type == QNN_DATATYPE_INT_64) {
    assert(!indices_info.is_initializer);
    indices_casted_name += "_int32";
    if (qnn_model_wrapper.IsQnnTensorWrapperExist(indices_casted_name)) {
      LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << indices_casted_name;
    } else {
      QnnTensorWrapper indices_cast_tensor(indices_casted_name,
                                           QNN_TENSOR_TYPE_NATIVE,
                                           QNN_DATATYPE_INT_32,
                                           QnnQuantParamsWrapper(),
                                           std::move(cast_output_shape));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(indices_cast_tensor)),
                        "Failed to add gather indices cast tensor.");
      ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(indices_casted_name,
                                                        QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                        QNN_OP_CAST,
                                                        {indices_tensor_name},
                                                        {indices_casted_name},
                                                        {},
                                                        do_op_validation),
                        "Failed to add GatherNd indices cast node.");
    }
  }

  input_names.push_back(indices_casted_name);

  return Status::OK();
}

Status GatherNDOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                      const NodeUnit& node_unit,
                                                      std::vector<std::string>&& input_names,
                                                      const logging::Logger& logger,
                                                      bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);
  const auto& output = node_unit.Outputs()[0];
  const std::string& output_name = output.node_arg.Name();

  QnnQuantParamsWrapper quant_params;
  ORT_RETURN_IF_ERROR(quant_params.Init(qnn_model_wrapper, output));

  const auto* type_proto = output.node_arg.TypeAsProto();
  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;
  ORT_RETURN_IF_ERROR(utils::GetQnnDataType(quant_params.IsQuantized(), type_proto, qnn_data_type));

  if (quant_params.IsPerTensor()) {
    // Make sure the output quantization parameters are equal to the input.
    ORT_RETURN_IF_ERROR(SetOutputQParamEqualToInputIfNearlyEqual(qnn_model_wrapper, node_unit, logger, input_names,
                                                                 0 /*input_index*/, 0 /*output_index*/, qnn_data_type,
                                                                 quant_params));
  }

  NodeAttrHelper node_helper(node_unit);
  int64_t batch_dims = node_helper.Get("batch_dims", static_cast<int64_t>(0));

  Qnn_Scalar_t batch_dims_scalar = QNN_SCALAR_INIT;
  batch_dims_scalar.dataType = QNN_DATATYPE_UINT_32;
  batch_dims_scalar.uint32Value = static_cast<uint32_t>(batch_dims);

  QnnParamWrapper batch_dims_param(node_unit.Index(), node_unit.Name(),
                                   QNN_OP_GATHER_ND_PARAM_BATCH_DIMS, batch_dims_scalar);
  std::vector<std::string> param_tensor_names = {batch_dims_param.GetParamTensorName()};
  qnn_model_wrapper.AddParamWrapper(std::move(batch_dims_param));

  // Get tensor wrappers for shape calculation
  const auto& data_tensor_wrapper = qnn_model_wrapper.GetQnnTensorWrapper(input_names[0]);
  const auto& indices_tensor_wrapper = qnn_model_wrapper.GetQnnTensorWrapper(input_names[1]);

  // Calculate the QNN output shape for GatherND
  std::vector<uint32_t> qnn_output_shape;
  const auto& data_dims = data_tensor_wrapper.GetTensorDims();
  const auto& indices_dims = indices_tensor_wrapper.GetTensorDims();

  // GatherND output shape calculation:
  size_t batch_dims_size = static_cast<size_t>(batch_dims);
  size_t indices_last_dim = indices_dims.back();

  // Add batch dimensions from data
  for (size_t i = 0; i < batch_dims_size && i < data_dims.size(); ++i) {
    qnn_output_shape.push_back(data_dims[i]);
  }

  // Add indices dimensions except the last one
  for (size_t i = 0; i < indices_dims.size() - 1; ++i) {
    qnn_output_shape.push_back(indices_dims[i]);
  }

  // Add remaining data dimensions after batch_dims + indices_last_dim
  size_t start_dim = batch_dims_size + indices_last_dim;
  for (size_t i = start_dim; i < data_dims.size(); ++i) {
    qnn_output_shape.push_back(data_dims[i]);
  }

  std::vector<uint32_t> target_output_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(output.node_arg, target_output_shape),
                    "Cannot get target output shape");

  bool reshape_required = (qnn_output_shape.size() != target_output_shape.size());
  bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);

  // Check if we need to add a cast node for int64
  bool needs_int64_cast = false;
  if (is_graph_output) {
    for (const auto& input_name : input_names) {
      if (input_name.find("_cast_int32") != std::string::npos) {
        needs_int64_cast = true;
        break;
      }
    }
  }
  struct CastNodeInfo {
    std::string node_name;
    std::string input_name;
    std::string output_name;
  };
  std::vector<CastNodeInfo> cast_node_info_vec;

  // Get the output info for the gather output tensor
  TensorInfo output_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(output, output_info));

  // If a cast to int64 is needed, add the cast node
  if (needs_int64_cast) {
    std::string cast_node_name = output_name + "_cast_int64";
    std::string cast_input_name = output_name + "_cast_int64_aux";
    std::string cast_output_name = output_name;

    // Create the cast input tensor wrapper - use qnn_output_shape for the intermediate tensor
    QnnTensorWrapper cast_input_tensorwrapper(cast_input_name,
                                              QNN_TENSOR_TYPE_NATIVE,
                                              output_info.qnn_data_type,
                                              output_info.quant_param.Copy(),
                                              std::vector<uint32_t>(qnn_output_shape));

    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(cast_input_tensorwrapper)), "Failed to add tensor.");
    cast_node_info_vec.push_back({cast_node_name, cast_input_name, cast_output_name});
    Qnn_TensorType_t cast_tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
    QnnTensorWrapper cast_output(output_name, cast_tensor_type, qnn_data_type, quant_params.Copy(),
                                 std::vector<uint32_t>(target_output_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(cast_output)), "Failed to add tensor.");
  }

  std::string gather_output_name = output_name;
  if (reshape_required) {
    gather_output_name += "_ort_qnn_ep_reshape";
  } else if (needs_int64_cast) {
    gather_output_name += "_cast_int64_aux";
  }

  Qnn_TensorType_t tensor_type = (!reshape_required && is_graph_output)
                                     ? QNN_TENSOR_TYPE_APP_READ
                                     : QNN_TENSOR_TYPE_NATIVE;

  QnnTensorWrapper gather_output_tensor(gather_output_name, tensor_type, qnn_data_type,
                                        quant_params.Copy(), std::move(qnn_output_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(gather_output_tensor)),
                    "Failed to add GatherND output tensor.");

  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetNodeName(node_unit),
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_GATHER_ND,
                                                    std::move(input_names),
                                                    {gather_output_name},
                                                    std::move(param_tensor_names),
                                                    do_op_validation),
                    "Failed to create GatherND node.");

  if (reshape_required) {
    Qnn_TensorType_t reshape_tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
    QnnTensorWrapper reshape_output(output_name, reshape_tensor_type, qnn_data_type,
                                    std::move(quant_params), std::move(target_output_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(reshape_output)), "Failed to add reshape output.");

    std::string node_output_name = output_name;
    if (needs_int64_cast) {
      // If needs_int64 is true, the output name should be the input name of the cast node
      node_output_name = output_name + "_cast_int64_aux";
    }

    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(output_name,
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_RESHAPE,
                                                      {gather_output_name},
                                                      {node_output_name},
                                                      {},
                                                      do_op_validation),
                      "Failed to add Reshape node.");
  }

  if (needs_int64_cast) {
    for (const auto& cast_node_info : cast_node_info_vec) {
      // Insert cast node.
      ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(cast_node_info.node_name,
                                                        QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                        QNN_OP_CAST,
                                                        {cast_node_info.input_name},
                                                        {cast_node_info.output_name},
                                                        {}),
                        "Failed to add Cast node");
    }
  }

  return Status::OK();
}

void CreateGatherNDOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<GatherNDOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime