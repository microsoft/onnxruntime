// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cassert>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

// High-level workflows:
// 1) GatherElements on HTP with BOOL input: BOOL -> Cast(U8) -> GatherElements(U8) -> Cast(BOOL).
// 2) Gather with int64 inputs: inputs gets Cast(INT32) in ProcessInt64Tensors;
//    if ONNX graph output is int64, cast back after Gather.
// 2.1) Gather with static int64 indices: negative indices get fixed and int64 gets reinterpreted as uint32
//      in ProcessAttributesAndOutputs, and Cast is added if output is int64.
// 3) Gather with rank mismatch: Gather -> Reshape -> (optional output cast).

class GatherOpBuilder : public BaseOpBuilder {
 public:
  GatherOpBuilder() : BaseOpBuilder("GatherOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GatherOpBuilder);

  Ort::Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger) const override ORT_MUST_USE_RESULT;

 protected:
  Ort::Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger,
                            std::vector<std::string>& input_names,
                            bool do_op_validation) const override ORT_MUST_USE_RESULT;

  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          const Ort::Logger& logger,
                                          bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

Ort::Status GatherOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                           const OrtNodeUnit& node_unit,
                                           const Ort::Logger& logger) const {
  // On QNN CPU backend, the QNN validator does not properly reject unsupported input shapes.
  // This causes a Qnn graph execution error. So, reject those configs here.
  // We should consider not using QNN CPU backend for onnxruntime unit tests.
  const std::string& op_type = node_unit.OpType();

  // rank of GatherElements input0 must be > 1 and <= 4 on QNN CPU backend
  if (qnn_model_wrapper.GetQnnBackendType() == QnnBackendType::CPU && op_type == "GatherElements") {
    const auto& input0 = node_unit.Inputs()[0];
    std::vector<uint32_t> input0_shape;
    RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input0.shape, input0_shape),
                  ("Cannot get input[0] shape for " + op_type + " node " + node_unit.Name()).c_str());

    const size_t input0_rank = input0_shape.size();
    RETURN_IF_NOT(input0_rank > 1 && input0_rank <= 4,
                  ("QNN CPU backend does not support " + op_type +
                   " with input[0] of rank " + std::to_string(input0_rank))
                      .c_str());
  }

  return BaseOpBuilder::IsOpSupported(qnn_model_wrapper, node_unit, logger);
}

// Makes negative indices positive and converts int64 indices to another integer type (typically int32 or uint32).
// The input and output are both represented as byte arrays.
template <typename SrcType, typename DstType>
static bool FixStaticIndices(const std::vector<uint8_t>& onnx_bytes,
                             int64_t input0_axis_dim,
                             /*out*/ std::vector<uint8_t>& qnn_bytes) {
  const size_t num_elems = onnx_bytes.size() / sizeof(SrcType);
  gsl::span<const SrcType> onnx_indices{reinterpret_cast<const SrcType*>(onnx_bytes.data()), num_elems};

  qnn_bytes.resize(num_elems * sizeof(DstType));
  gsl::span<DstType> qnn_indices{reinterpret_cast<DstType*>(qnn_bytes.data()), num_elems};

  for (size_t i = 0; i < num_elems; i++) {
    SrcType onnx_index = onnx_indices[i];

    // Try to make a negative index positive by adding rank.
    if (onnx_index < 0) {
      onnx_index += static_cast<SrcType>(input0_axis_dim);
    }

    if (onnx_index < 0 || static_cast<int64_t>(onnx_index) >= input0_axis_dim) {
      return false;  // QNN does not support out-of-bounds indices.
    }

    qnn_indices[i] = static_cast<DstType>(onnx_index);
  }

  return true;
}

// Gets the size of input0 on the axis dimension.
static Ort::Status GetInput0AxisDimValue(const QnnModelWrapper& qnn_model_wrapper,
                                         const OrtNodeUnit& node_unit,
                                         int64_t default_axis_value,
                                         /*out*/ int64_t& axis_dim_value) {
  const auto& input0 = node_unit.Inputs()[0];
  std::vector<uint32_t> input0_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input0.shape, input0_shape),
                ("Cannot get shape for " + node_unit.OpType() + " input[0] " + input0.name).c_str());

  int64_t rank = static_cast<int64_t>(input0_shape.size());
  OrtNodeAttrHelper node_helper(node_unit);
  int64_t onnx_axis = node_helper.Get("axis", default_axis_value);
  if (onnx_axis < 0) {
    onnx_axis += rank;
  }
  RETURN_IF_NOT((onnx_axis >= 0 && onnx_axis < static_cast<int64_t>(input0_shape.size())),
                ("QNN requires axis range [0, rank-1] for " + node_unit.OpType()).c_str());

  axis_dim_value = static_cast<int64_t>(input0_shape[onnx_axis]);

  return Ort::Status();
}

// Processes the indices input to Gather operators.
//
// QNN only supports int32 / uint32 as indices tensor data types.
// When indices tensor is an initializer, statically cast values int64 -> int32.
// When dynamic input, add explicit QNN Cast node for int64 -> int32 conversion.
//
// The HTP backend only supports dynamic int64 indices if they are a graph input.
static Ort::Status ProcessIndicesInput(QnnModelWrapper& qnn_model_wrapper,
                                       const OrtNodeUnitIODef& indices_input,
                                       int64_t input0_axis_dim,
                                       const Ort::Logger& logger,
                                       std::vector<std::string>& input_names,
                                       bool do_op_validation) {
  const auto& indices_tensor_name = indices_input.name;

  TensorInfo indices_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(indices_input, indices_info));

  std::vector<uint8_t> qnn_indices_bytes;

  // A. Get raw bytes for static indices.
  //    If indices are int64, convert them to int32 and update indices_info.qnn_data_type.
  if (indices_info.is_initializer) {
    std::vector<uint8_t> onnx_indices_bytes;
    RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(indices_info.initializer_tensor, onnx_indices_bytes));

    if (indices_info.qnn_data_type == QNN_DATATYPE_INT_64) {
      RETURN_IF_NOT((FixStaticIndices<int64_t, int32_t>(onnx_indices_bytes, input0_axis_dim, qnn_indices_bytes)),
                    "QNN does not support negative index values for Gather* ops");
      indices_info.qnn_data_type = QNN_DATATYPE_INT_32;
    } else if (indices_info.qnn_data_type == QNN_DATATYPE_INT_32) {
      RETURN_IF_NOT((FixStaticIndices<int32_t, int32_t>(onnx_indices_bytes, input0_axis_dim, qnn_indices_bytes)),
                    "QNN does not support negative index values for Gather* ops");
    } else {
      qnn_indices_bytes = std::move(onnx_indices_bytes);
    }
  }

  std::vector<uint32_t> cast_output_shape(indices_info.shape);
  if (qnn_model_wrapper.IsQnnTensorWrapperExist(indices_tensor_name)) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("Tensor already added, skip it: " + indices_tensor_name).c_str());
  } else {
    QnnTensorWrapper input_tensorwrapper(indices_tensor_name,
                                         qnn_model_wrapper.GetTensorType(indices_tensor_name),
                                         indices_info.qnn_data_type, QnnQuantParamsWrapper(),
                                         std::move(indices_info.shape),
                                         std::move(qnn_indices_bytes));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
  }

  // B. Insert QNN Cast op to convert dynamic indices from int64 to int32.
  auto& input_tensorwrapper = qnn_model_wrapper.GetQnnTensorWrapper(indices_tensor_name);

  std::string indices_casted_name{indices_tensor_name};
  // Check QNN Tensor data type.
  if (input_tensorwrapper.GetTensorDataType() == QNN_DATATYPE_INT_64) {
    assert(!indices_info.is_initializer);
    indices_casted_name += "_int32";
    RETURN_IF_ERROR(qnn_model_wrapper.AddCastNode(utils::GetUniqueName(indices_tensor_name, QNN_OP_CAST),
                                                  indices_tensor_name,
                                                  indices_casted_name,
                                                  QNN_TENSOR_TYPE_NATIVE,
                                                  QNN_DATATYPE_INT_32,
                                                  QnnQuantParamsWrapper(),
                                                  std::move(cast_output_shape),
                                                  do_op_validation));
  }
  input_names.push_back(indices_casted_name);
  return Ort::Status();
}

Ort::Status GatherOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                           const OrtNodeUnit& node_unit,
                                           const Ort::Logger& logger,
                                           std::vector<std::string>& input_names,
                                           bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  RETURN_IF(inputs.size() != 2, ("QNN EP: " + node_unit.OpType() + " operator must have two inputs").c_str());
  RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[0], logger, input_names));

  // Gather/GatherElements on HTP BE don't support BOOL input. Add Cast node to convert BOOL to UINT8.
  const bool needs_bool_cast = (qnn_model_wrapper.GetQnnBackendType() == QnnBackendType::HTP &&
                                inputs[0].type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
  if (needs_bool_cast) {
    const std::string& input0_name = input_names[0];
    TensorInfo input0_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], input0_info));
    const std::string cast_output_name = utils::GetUniqueName(input0_name, "_bool_to_u8");
    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(cast_output_name)) {
      RETURN_IF_ERROR(qnn_model_wrapper.AddCastNode(utils::GetUniqueName(input0_name, QNN_OP_CAST),
                                                    input0_name,
                                                    cast_output_name,
                                                    QNN_TENSOR_TYPE_NATIVE,
                                                    QNN_DATATYPE_UFIXED_POINT_8,
                                                    QnnQuantParamsWrapper(1.0f, 0),
                                                    std::vector<uint32_t>(input0_info.shape),
                                                    do_op_validation));
    }
    input_names[0] = cast_output_name;
  }

  int64_t input0_axis_dim = 0;
  RETURN_IF_ERROR(GetInput0AxisDimValue(qnn_model_wrapper, node_unit, /*default_axis_value=*/0, input0_axis_dim));
  return ProcessIndicesInput(qnn_model_wrapper, inputs[1], input0_axis_dim, logger, input_names, do_op_validation);
}

Ort::Status GatherOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                         const OrtNodeUnit& node_unit,
                                                         std::vector<std::string>&& input_names,
                                                         const Ort::Logger& logger,
                                                         bool do_op_validation) const {
  const bool is_gather_elems = node_unit.OpType() == "GatherElements";
  const bool needs_bool_cast = (qnn_model_wrapper.GetQnnBackendType() == QnnBackendType::HTP &&
                                node_unit.Inputs()[0].type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);

  // Create QNN 'axis' parameter.
  std::vector<std::string> param_tensor_names;
  int32_t axis_value = 0;
  Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
  RETURN_IF_ERROR(ProcessAxisAttribute(qnn_model_wrapper, node_unit, axis_qnn_scalar, axis_value));
  QnnParamWrapper axis_param(node_unit.Index(), node_unit.Name(),
                             (is_gather_elems ? QNN_OP_GATHER_ELEMENTS_PARAM_AXIS : QNN_OP_GATHER_PARAM_AXIS),
                             axis_qnn_scalar);
  param_tensor_names.push_back(axis_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(axis_param));

  if (is_gather_elems) {
    if (!needs_bool_cast) {
      return ProcessOutputs(qnn_model_wrapper, node_unit, std::move(input_names), std::move(param_tensor_names),
                            logger, do_op_validation, GetQnnOpType(node_unit.OpType()));
    }

    // 1. BOOL input flow for GatherElements on HTP:
    // 1.1 Input: BOOL -> Cast(U8) (done in ProcessInputs)
    // 1.2 GatherElements on U8
    // 1.3 Output: U8 -> Cast(BOOL)
    const auto& gather_output = node_unit.Outputs()[0];
    const auto& output_name = gather_output.name;
    std::vector<uint32_t> output_shape;
    RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(gather_output.shape, output_shape), "Cannot get shape");
    const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);

    const std::string gather_output_name = utils::GetUniqueName(output_name, "_u8_to_bool_in");
    QnnQuantParamsWrapper gather_quant_params(1.0f, 0);
    QnnTensorWrapper gather_output_wrapper(gather_output_name,
                                           QNN_TENSOR_TYPE_NATIVE,
                                           QNN_DATATYPE_UFIXED_POINT_8,
                                           std::move(gather_quant_params),
                                           std::vector<uint32_t>(output_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(gather_output_wrapper)), "Failed to add tensor.");

    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetUniqueName(node_unit),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  GetQnnOpType(node_unit.OpType()),
                                                  std::move(input_names),
                                                  {gather_output_name},
                                                  std::move(param_tensor_names),
                                                  do_op_validation),
                  "Failed to add node.");

    TensorInfo output_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(gather_output, output_info));
    Qnn_TensorType_t output_tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
    RETURN_IF_ERROR(qnn_model_wrapper.AddCastNode(utils::GetUniqueName(output_name, "_u8_to_bool"),
                                                  gather_output_name,
                                                  output_name,
                                                  output_tensor_type,
                                                  output_info.qnn_data_type,
                                                  output_info.quant_param.Copy(),
                                                  std::vector<uint32_t>(output_shape),
                                                  do_op_validation));
    return Ort::Status();
  }

  // if indicies is scalar shape, then need to add Reshape node
  const auto& input_tensor_wrapper = qnn_model_wrapper.GetQnnTensorWrapper(input_names[0]);
  const auto& indices_input_tensor_wrapper = qnn_model_wrapper.GetQnnTensorWrapper(input_names[1]);

  // 1. Determine Gather output shape (QNN expected shape pre-reshape)
  std::vector<uint32_t> qnn_pre_reshape_output_shape;
  auto input_rank = input_tensor_wrapper.GetTensorRank();
  auto indices_rank = indices_input_tensor_wrapper.GetTensorRank();
  qnn_pre_reshape_output_shape.reserve(static_cast<size_t>(input_rank - 1 + indices_rank));

  // replace the dimension for p.axis with the shape from the indices
  std::copy(input_tensor_wrapper.GetTensorDims().begin(), input_tensor_wrapper.GetTensorDims().begin() + axis_value,
            std::back_inserter(qnn_pre_reshape_output_shape));

  const auto& indicies_shape = indices_input_tensor_wrapper.GetTensorDims();
  std::copy(indicies_shape.begin(), indicies_shape.end(), std::back_inserter(qnn_pre_reshape_output_shape));

  std::copy(input_tensor_wrapper.GetTensorDims().begin() + axis_value + 1, input_tensor_wrapper.GetTensorDims().end(),
            std::back_inserter(qnn_pre_reshape_output_shape));

  const auto& gather_output = node_unit.Outputs()[0];
  const auto& output_name = gather_output.name;

  QnnQuantParamsWrapper quantize_param;
  RETURN_IF_ERROR(quantize_param.Init(qnn_model_wrapper, gather_output));

  ONNXTensorElementDataType gather_output_type = gather_output.type;
  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;
  RETURN_IF_ERROR(utils::GetQnnDataType(quantize_param.IsQuantized(), gather_output_type, qnn_data_type));

  if (!needs_bool_cast && quantize_param.IsPerTensor()) {
    // Make sure the output quantization parameters are equal to the input.
    RETURN_IF_ERROR(SetOutputQParamEqualToInputIfNearlyEqual(qnn_model_wrapper, node_unit, logger, input_names,
                                                             0 /*input_index*/, 0 /*output_index*/, qnn_data_type,
                                                             quantize_param));
  }

  std::vector<uint32_t> onnx_final_output_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(gather_output.shape, onnx_final_output_shape), "Cannot get shape");

  bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);
  bool reshape_required = (qnn_pre_reshape_output_shape.size() != onnx_final_output_shape.size());

  // 2. Gather output quantization info (shared by optional casts)
  TensorInfo output_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(gather_output, output_info));

  // 3. Decide optional output casts
  struct OutputCastPlan {
    bool needs_bool_cast = false;
    bool needs_int64_cast = false;
    std::string bool_cast_input_name;
    std::string int64_cast_node_name;
    std::string int64_cast_input_name;
    std::string int64_cast_tensor_name;
  };

  OutputCastPlan cast_plan;
  cast_plan.needs_bool_cast = needs_bool_cast;

  // 3.1 INT64: input-side casts may require output cast back to INT64 at graph output
  if (is_graph_output && !needs_bool_cast) {
    for (const auto& input_name : input_names) {
      if (input_name.find("_cast_int32") != std::string::npos ||
          input_name.find("_int64_to_int32") != std::string::npos) {
        cast_plan.needs_int64_cast = true;
        break;
      }
    }
  }

  // 3.2 BOOL: HTP requires BOOL -> U8 before Gather and U8 -> BOOL after
  if (cast_plan.needs_bool_cast) {
    cast_plan.bool_cast_input_name = utils::GetUniqueName(output_name, "_u8_to_bool_in");
  }

  // 3.3 Prepare int64 cast input tensor (created now, cast node added later)
  if (cast_plan.needs_int64_cast) {
    cast_plan.int64_cast_node_name = utils::GetUniqueName(node_unit, "_int32_to_int64");
    cast_plan.int64_cast_tensor_name = utils::GetUniqueName(output_name, "_int32_to_int64_in");
    cast_plan.int64_cast_input_name = cast_plan.int64_cast_tensor_name;

    QnnTensorWrapper cast_input_tensorwrapper(cast_plan.int64_cast_tensor_name,
                                              QNN_TENSOR_TYPE_NATIVE,
                                              output_info.qnn_data_type,
                                              output_info.quant_param.Copy(),
                                              std::vector<uint32_t>(qnn_pre_reshape_output_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(cast_input_tensorwrapper)), "Failed to add tensor.");
  }

  // 4. Choose gather output name based on optional reshape/cast flow
  // 4.1 BOOL flow: gather -> U8 tensor -> cast back to BOOL
  // 4.2 INT64 flow: gather -> int32 tensor -> cast back to int64
  // 4.3 Reshape flow: gather -> reshape -> (optional cast)
  std::string gather_output_name = output_name;
  if (reshape_required) {
    gather_output_name = utils::GetUniqueName(output_name, "_reshape");
  } else if (cast_plan.needs_bool_cast) {
    gather_output_name = cast_plan.bool_cast_input_name;
  } else if (cast_plan.needs_int64_cast) {
    // Use the previously stored cast input name for consistency
    gather_output_name = cast_plan.int64_cast_tensor_name;
  }

  // 5. Create Gather output tensor and node
  Qnn_DataType_t gather_qnn_data_type = qnn_data_type;
  QnnQuantParamsWrapper gather_quant_params = quantize_param.Copy();
  if (cast_plan.needs_bool_cast) {
    gather_qnn_data_type = QNN_DATATYPE_UFIXED_POINT_8;
    gather_quant_params = QnnQuantParamsWrapper(1.0f, 0);
  }

  Qnn_TensorType_t tensor_type = (!reshape_required && is_graph_output && !cast_plan.needs_bool_cast && !cast_plan.needs_int64_cast)
                                     ? QNN_TENSOR_TYPE_APP_READ
                                     : QNN_TENSOR_TYPE_NATIVE;
  QnnTensorWrapper gather_output_wrapper(gather_output_name, tensor_type, gather_qnn_data_type, gather_quant_params.Copy(),
                                         std::move(qnn_pre_reshape_output_shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(gather_output_wrapper)), "Failed to add tensor.");
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetUniqueName(node_unit),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                GetQnnOpType(node_unit.OpType()),
                                                std::move(input_names),
                                                {gather_output_name},
                                                std::move(param_tensor_names),
                                                do_op_validation),
                "Failed to add node.");

  if (reshape_required) {
    // 5.1 Add Reshape node after Gather if ONNX output rank differs from QNN rank
    const bool reshape_is_graph_output = is_graph_output && !cast_plan.needs_bool_cast && !cast_plan.needs_int64_cast;
    Qnn_TensorType_t reshape_tensor_type = reshape_is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
    std::string reshape_output_name = output_name;
    if (cast_plan.needs_bool_cast) {
      reshape_output_name = cast_plan.bool_cast_input_name;
    }
    if (cast_plan.needs_int64_cast) {
      reshape_output_name = cast_plan.int64_cast_tensor_name;
    }
    QnnTensorWrapper reshape_output(reshape_output_name, reshape_tensor_type, gather_qnn_data_type,
                                    gather_quant_params.Copy(), std::vector<uint32_t>(onnx_final_output_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(reshape_output)), "Failed to add tensor.");
    std::string node_output_name = reshape_output_name;
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetUniqueName(node_unit, QNN_OP_RESHAPE),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_RESHAPE,
                                                  {gather_output_name},
                                                  {node_output_name},
                                                  {},
                                                  do_op_validation),
                  "Failed to add node.");
  }

  // 6. Append optional output casts
  if (cast_plan.needs_bool_cast) {
    Qnn_TensorType_t output_tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
    const std::string cast_input = reshape_required ? cast_plan.bool_cast_input_name : gather_output_name;
    RETURN_IF_ERROR(qnn_model_wrapper.AddCastNode(utils::GetUniqueName(output_name, "_u8_to_bool"),
                                                  cast_input,
                                                  output_name,
                                                  output_tensor_type,
                                                  output_info.qnn_data_type,
                                                  output_info.quant_param.Copy(),
                                                  std::vector<uint32_t>(onnx_final_output_shape),
                                                  do_op_validation));
  }

  if (cast_plan.needs_int64_cast) {
    Qnn_TensorType_t cast_tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
    RETURN_IF_ERROR(qnn_model_wrapper.AddCastNode(cast_plan.int64_cast_node_name,
                                                  cast_plan.int64_cast_input_name,
                                                  output_name,
                                                  cast_tensor_type,
                                                  qnn_data_type,
                                                  quantize_param.Copy(),
                                                  std::vector<uint32_t>(onnx_final_output_shape),
                                                  do_op_validation));
  }

  return Ort::Status();
}

void CreateGatherOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<GatherOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
