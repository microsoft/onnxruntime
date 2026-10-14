// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/opbuilder/rope_op_builder.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"

namespace onnxruntime {
namespace qnn {

Status RopeOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                    const NodeUnit& node_unit,
                                    const logging::Logger& logger) const {
  LOGS(logger, VERBOSE) << "Validating RotaryEmbedding op for QNN EP";

  // Only support HTP backend for now
  if (!IsNpuBackend(qnn_model_wrapper.GetQnnBackendType())) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "QNN RotaryEmbedding is only supported on HTP backend");
  }

  const auto& inputs = node_unit.Inputs();
  const auto& outputs = node_unit.Outputs();

  // Validate input/output counts
  ORT_RETURN_IF_NOT(inputs.size() >= 3 && inputs.size() <= 4,
                    "RotaryEmbedding requires 3 or 4 inputs (input, cos_cache, sin_cache, [position_ids])");
  ORT_RETURN_IF_NOT(outputs.size() == 1, "RotaryEmbedding requires exactly 1 output");

  // Validate data types - only FP16 and FP32 supported
  TensorInfo input_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], input_info));

  ORT_RETURN_IF_NOT(input_info.qnn_data_type == QNN_DATATYPE_FLOAT_16 ||
                        input_info.qnn_data_type == QNN_DATATYPE_FLOAT_32,
                    "RotaryEmbedding only supports FP16 and FP32 data types");

  // Validate cos_cache and sin_cache have same dtype as input
  TensorInfo cos_cache_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[1], cos_cache_info));
  ORT_RETURN_IF_NOT(cos_cache_info.qnn_data_type == input_info.qnn_data_type,
                    "cos_cache must have same data type as input");

  TensorInfo sin_cache_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[2], sin_cache_info));
  ORT_RETURN_IF_NOT(sin_cache_info.qnn_data_type == input_info.qnn_data_type,
                    "sin_cache must have same data type as input");

  // Validate shapes
  ORT_RETURN_IF_ERROR(ValidateInputShapes(qnn_model_wrapper, node_unit, logger));

  // Validate attributes
  NodeAttrHelper node_helper(node_unit);
  int64_t rotary_embedding_dim = node_helper.Get("rotary_embedding_dim", static_cast<int64_t>(0));

  // Get input shape to validate rotary_embedding_dim
  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].node_arg, input_shape),
                    "Cannot get input shape");

  const size_t input_rank = input_shape.size();
  ORT_RETURN_IF_NOT(input_rank == 3 || input_rank == 4,
                    "RotaryEmbedding input must be rank 3 or 4");

  // For 3D input, num_heads attribute is required
  if (input_rank == 3) {
    int64_t num_heads = node_helper.Get("num_heads", static_cast<int64_t>(0));
    ORT_RETURN_IF_NOT(num_heads > 0,
                      "num_heads attribute is required for 3D input and must be > 0");
    const uint32_t hidden_size = input_shape[2];
    ORT_RETURN_IF_NOT(hidden_size % num_heads == 0,
                      "hidden_size must be divisible by num_heads");
  }

  // Determine head_size for validation
  uint32_t head_size = 0;
  if (input_rank == 4) {
    head_size = input_shape[3];  // [B, NH, S, HS]
  } else {
    int64_t num_heads = node_helper.Get("num_heads", static_cast<int64_t>(0));
    head_size = input_shape[2] / static_cast<uint32_t>(num_heads);  // [B, S, NH*HS]
  }

  // Validate rotary_embedding_dim
  if (rotary_embedding_dim == 0) {
    rotary_embedding_dim = head_size;
  }
  ORT_RETURN_IF_NOT(rotary_embedding_dim > 0 && rotary_embedding_dim <= static_cast<int64_t>(head_size),
                    "rotary_embedding_dim must be > 0 and <= head_size");
  ORT_RETURN_IF_NOT(rotary_embedding_dim % 2 == 0,
                    "rotary_embedding_dim must be even");

  // Validate cos/sin cache shapes
  std::vector<uint32_t> cos_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[1].node_arg, cos_shape),
                    "Cannot get cos_cache shape");
  std::vector<uint32_t> sin_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[2].node_arg, sin_shape),
                    "Cannot get sin_cache shape");

  ORT_RETURN_IF_NOT(cos_shape.size() == 2 || cos_shape.size() == 3,
                    "cos_cache must be rank 2 or 3");
  ORT_RETURN_IF_NOT(sin_shape.size() == 2 || sin_shape.size() == 3,
                    "sin_cache must be rank 2 or 3");
  ORT_RETURN_IF_NOT(cos_shape.size() == sin_shape.size(),
                    "cos_cache and sin_cache must have same rank");

  const uint32_t expected_cache_dim = static_cast<uint32_t>(rotary_embedding_dim / 2);
  ORT_RETURN_IF_NOT(cos_shape.back() == expected_cache_dim,
                    "cos_cache last dimension must equal rotary_embedding_dim/2");
  ORT_RETURN_IF_NOT(sin_shape.back() == expected_cache_dim,
                    "sin_cache last dimension must equal rotary_embedding_dim/2");

  // Validate position_ids if present
  if (inputs.size() == 4 && inputs[3].node_arg.Exists()) {
    std::vector<uint32_t> pos_ids_shape;
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[3].node_arg, pos_ids_shape),
                      "Cannot get position_ids shape");
    ORT_RETURN_IF_NOT(pos_ids_shape.size() == 2,
                      "position_ids must be rank 2 [B, S]");
  }

  LOGS(logger, VERBOSE) << "RotaryEmbedding op validation successful";
  return Status::OK();
}

Status RopeOpBuilder::ValidateInputShapes(QnnModelWrapper& qnn_model_wrapper,
                                          const NodeUnit& node_unit,
                                          const logging::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);

  const auto& inputs = node_unit.Inputs();
  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].node_arg, input_shape),
                    "Cannot get input shape");

  const size_t input_rank = input_shape.size();
  ORT_RETURN_IF_NOT(input_rank == 3 || input_rank == 4,
                    "RotaryEmbedding input must be rank 3 [B,S,NH*HS] or rank 4 [B,NH,S,HS]");

  return Status::OK();
}

Status RopeOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                  const NodeUnit& node_unit,
                                                  std::vector<std::string>&& input_names,
                                                  const logging::Logger& logger,
                                                  bool do_op_validation) const {
  LOGS(logger, VERBOSE) << "Processing RotaryEmbedding op for QNN";

  // Decompose RotaryEmbedding into QNN elementary ops
  return DecomposeRotaryEmbedding(qnn_model_wrapper, node_unit, std::move(input_names),
                                  logger, do_op_validation);
}

Status RopeOpBuilder::DecomposeRotaryEmbedding(QnnModelWrapper& qnn_model_wrapper,
                                               const NodeUnit& node_unit,
                                               std::vector<std::string>&& input_names,
                                               const logging::Logger& logger,
                                               bool do_op_validation) const {
  LOGS(logger, VERBOSE) << "Decomposing RotaryEmbedding into QNN ops";

  const auto& inputs = node_unit.Inputs();
  const auto& outputs = node_unit.Outputs();
  NodeAttrHelper node_helper(node_unit);

  // Get attributes
  bool interleaved = node_helper.Get("interleaved", false);
  int64_t rotary_embedding_dim = node_helper.Get("rotary_embedding_dim", static_cast<int64_t>(0));
  int64_t num_heads = node_helper.Get("num_heads", static_cast<int64_t>(0));

  // Get input shape and determine layout
  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].node_arg, input_shape),
                    "Cannot get input shape");

  const size_t input_rank = input_shape.size();
  const bool is_4d_input = (input_rank == 4);

  // Get tensor info for data type
  TensorInfo input_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], input_info));
  const Qnn_DataType_t qnn_data_type = input_info.qnn_data_type;

  // Determine dimensions
  uint32_t batch_size = input_shape[0];
  uint32_t seq_len = 0;
  uint32_t num_heads_val = 0;
  uint32_t head_size = 0;

  if (is_4d_input) {
    // Input is [B, NH, S, HS]
    num_heads_val = input_shape[1];
    seq_len = input_shape[2];
    head_size = input_shape[3];
  } else {
    // Input is [B, S, NH*HS]
    seq_len = input_shape[1];
    ORT_RETURN_IF_NOT(num_heads > 0, "num_heads required for 3D input");
    num_heads_val = static_cast<uint32_t>(num_heads);
    head_size = input_shape[2] / num_heads_val;
  }

  // Determine rotary_dim
  uint32_t rotary_dim = (rotary_embedding_dim == 0) ? head_size : static_cast<uint32_t>(rotary_embedding_dim);
  ORT_RETURN_IF_NOT(rotary_dim % 2 == 0 && rotary_dim <= head_size,
                    "Invalid rotary_embedding_dim");

  const uint32_t rotary_half_dim = rotary_dim / 2;

  // Input names: [input, cos_cache, sin_cache, position_ids (optional)]
  std::string current_tensor = input_names[0];
  const std::string& cos_cache_name = input_names[1];
  const std::string& sin_cache_name = input_names[2];
  const bool has_position_ids = (input_names.size() == 4);

  // Step 1: Normalize layout to [B, S, NH, HS]
  std::string normalized_tensor = current_tensor;
  std::vector<uint32_t> normalized_shape = {batch_size, seq_len, num_heads_val, head_size};

  if (is_4d_input) {
    // Transpose from [B, NH, S, HS] to [B, S, NH, HS]
    normalized_tensor = utils::GetUniqueName(node_unit, "_transpose_input");
    std::vector<uint32_t> transpose_perm = {0, 2, 1, 3};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(
        node_unit.Index(), current_tensor, normalized_tensor,
        input_shape, transpose_perm, normalized_shape,
        qnn_data_type, input_info.quant_param, do_op_validation,
        false, false));
    current_tensor = normalized_tensor;
  } else {
    // Reshape from [B, S, NH*HS] to [B, S, NH, HS]
    normalized_tensor = utils::GetUniqueName(node_unit, "_reshape_input");
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(
        current_tensor, normalized_tensor,
        input_shape, normalized_shape,
        qnn_data_type, input_info.quant_param, do_op_validation,
        false, false));
    current_tensor = normalized_tensor;
  }

  // Step 2: Slice the head dimension into rotated and non-rotated parts
  // x_rotate = x[..., :rotary_dim]
  // x_tail = x[..., rotary_dim:] (if rotary_dim < head_size)

  std::string x_rotate_name = utils::GetUniqueName(node_unit, "_x_rotate");
  std::vector<uint32_t> x_rotate_shape = {batch_size, seq_len, num_heads_val, rotary_dim};

  // Create StridedSlice for x_rotate: slice last dimension [0:rotary_dim]
  {
    std::vector<std::string> slice_input_names = {current_tensor};
    std::vector<std::string> slice_param_names;

    // ranges: [rank, 3] array where each row is [start, end, step]
    std::vector<uint32_t> ranges_data;
    ranges_data.reserve(12);  // 4 dimensions * 3 values
    // Dimension 0: [0, batch_size, 1]
    ranges_data.push_back(0);
    ranges_data.push_back(batch_size);
    ranges_data.push_back(1);
    // Dimension 1: [0, seq_len, 1]
    ranges_data.push_back(0);
    ranges_data.push_back(seq_len);
    ranges_data.push_back(1);
    // Dimension 2: [0, num_heads_val, 1]
    ranges_data.push_back(0);
    ranges_data.push_back(num_heads_val);
    ranges_data.push_back(1);
    // Dimension 3: [0, rotary_dim, 1]
    ranges_data.push_back(0);
    ranges_data.push_back(rotary_dim);
    ranges_data.push_back(1);

    std::vector<uint32_t> ranges_shape = {4, 3};
    QnnParamWrapper ranges_param(node_unit.Index(), node_unit.Name() + "_rotate_ranges",
                                 QNN_OP_STRIDED_SLICE_PARAM_RANGES,
                                 std::move(ranges_shape), std::move(ranges_data), true);
    slice_param_names.push_back(ranges_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(ranges_param));

    // Create output tensor
    QnnTensorWrapper x_rotate_tensor(x_rotate_name, QNN_TENSOR_TYPE_NATIVE,
                                     qnn_data_type, input_info.quant_param.Copy(),
                                     std::move(x_rotate_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(x_rotate_tensor)),
                      "Failed to add x_rotate tensor");

    // Create StridedSlice node
    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                          utils::GetUniqueName(node_unit, "_slice_rotate"),
                          QNN_OP_PACKAGE_NAME_QTI_AISW,
                          QNN_OP_STRIDED_SLICE,
                          std::move(slice_input_names),
                          {x_rotate_name},
                          std::move(slice_param_names),
                          do_op_validation),
                      "Failed to create StridedSlice node for x_rotate");
  }

  // Create x_tail if rotary_dim < head_size
  std::string x_tail_name;
  if (rotary_dim < head_size) {
    x_tail_name = utils::GetUniqueName(node_unit, "_x_tail");
    std::vector<uint32_t> x_tail_shape = {batch_size, seq_len, num_heads_val, head_size - rotary_dim};

    std::vector<std::string> slice_input_names = {current_tensor};
    std::vector<std::string> slice_param_names;

    // ranges: [rank, 3] array where each row is [start, end, step]
    std::vector<uint32_t> ranges_data;
    ranges_data.reserve(12);
    // Dimension 0: [0, batch_size, 1]
    ranges_data.push_back(0);
    ranges_data.push_back(batch_size);
    ranges_data.push_back(1);
    // Dimension 1: [0, seq_len, 1]
    ranges_data.push_back(0);
    ranges_data.push_back(seq_len);
    ranges_data.push_back(1);
    // Dimension 2: [0, num_heads_val, 1]
    ranges_data.push_back(0);
    ranges_data.push_back(num_heads_val);
    ranges_data.push_back(1);
    // Dimension 3: [rotary_dim, head_size, 1]
    ranges_data.push_back(rotary_dim);
    ranges_data.push_back(head_size);
    ranges_data.push_back(1);

    std::vector<uint32_t> ranges_shape = {4, 3};
    QnnParamWrapper ranges_param(node_unit.Index(), node_unit.Name() + "_tail_ranges",
                                 QNN_OP_STRIDED_SLICE_PARAM_RANGES,
                                 std::move(ranges_shape), std::move(ranges_data), true);
    slice_param_names.push_back(ranges_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(ranges_param));

    // Create output tensor
    QnnTensorWrapper x_tail_tensor(x_tail_name, QNN_TENSOR_TYPE_NATIVE,
                                   qnn_data_type, input_info.quant_param.Copy(),
                                   std::move(x_tail_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(x_tail_tensor)),
                      "Failed to add x_tail tensor");

    // Create StridedSlice node
    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                          utils::GetUniqueName(node_unit, "_slice_tail"),
                          QNN_OP_PACKAGE_NAME_QTI_AISW,
                          QNN_OP_STRIDED_SLICE,
                          std::move(slice_input_names),
                          {x_tail_name},
                          std::move(slice_param_names),
                          do_op_validation),
                      "Failed to create StridedSlice node for x_tail");
  }

  // Step 3: Split x_rotate into x1 and x2 based on interleaved flag
  std::string x1_name = utils::GetUniqueName(node_unit, "_x1");
  std::string x2_name = utils::GetUniqueName(node_unit, "_x2");
  std::vector<uint32_t> x1_x2_shape = {batch_size, seq_len, num_heads_val, rotary_half_dim};

  if (interleaved) {
    // Use StridedSlice with stride=2 for interleaved split
    // x1 = x_rotate[..., 0::2]
    {
      std::vector<std::string> slice_input_names = {x_rotate_name};
      std::vector<std::string> slice_param_names;

      // ranges: [rank, 3] array where each row is [start, end, step]
      std::vector<uint32_t> ranges_data;
      ranges_data.reserve(12);
      // Dimension 0: [0, batch_size, 1]
      ranges_data.push_back(0);
      ranges_data.push_back(batch_size);
      ranges_data.push_back(1);
      // Dimension 1: [0, seq_len, 1]
      ranges_data.push_back(0);
      ranges_data.push_back(seq_len);
      ranges_data.push_back(1);
      // Dimension 2: [0, num_heads_val, 1]
      ranges_data.push_back(0);
      ranges_data.push_back(num_heads_val);
      ranges_data.push_back(1);
      // Dimension 3: [0, rotary_dim, 2] - stride=2 for interleaved
      ranges_data.push_back(0);
      ranges_data.push_back(rotary_dim);
      ranges_data.push_back(2);

      std::vector<uint32_t> ranges_shape = {4, 3};
      QnnParamWrapper ranges_param(node_unit.Index(), node_unit.Name() + "_x1_ranges",
                                   QNN_OP_STRIDED_SLICE_PARAM_RANGES,
                                   std::move(ranges_shape), std::move(ranges_data), true);
      slice_param_names.push_back(ranges_param.GetParamTensorName());
      qnn_model_wrapper.AddParamWrapper(std::move(ranges_param));

      QnnTensorWrapper x1_tensor(x1_name, QNN_TENSOR_TYPE_NATIVE,
                                 qnn_data_type, input_info.quant_param.Copy(),
                                 std::vector<uint32_t>(x1_x2_shape));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(x1_tensor)),
                        "Failed to add x1 tensor");

      ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                            utils::GetUniqueName(node_unit, "_slice_x1"),
                            QNN_OP_PACKAGE_NAME_QTI_AISW,
                            QNN_OP_STRIDED_SLICE,
                            std::move(slice_input_names),
                            {x1_name},
                            std::move(slice_param_names),
                            do_op_validation),
                        "Failed to create StridedSlice node for x1");
    }

    // x2 = x_rotate[..., 1::2]
    {
      std::vector<std::string> slice_input_names = {x_rotate_name};
      std::vector<std::string> slice_param_names;

      // ranges: [rank, 3] array where each row is [start, end, step]
      std::vector<uint32_t> ranges_data;
      ranges_data.reserve(12);
      // Dimension 0: [0, batch_size, 1]
      ranges_data.push_back(0);
      ranges_data.push_back(batch_size);
      ranges_data.push_back(1);
      // Dimension 1: [0, seq_len, 1]
      ranges_data.push_back(0);
      ranges_data.push_back(seq_len);
      ranges_data.push_back(1);
      // Dimension 2: [0, num_heads_val, 1]
      ranges_data.push_back(0);
      ranges_data.push_back(num_heads_val);
      ranges_data.push_back(1);
      // Dimension 3: [1, rotary_dim, 2] - start at 1, stride=2 for interleaved
      ranges_data.push_back(1);
      ranges_data.push_back(rotary_dim);
      ranges_data.push_back(2);

      std::vector<uint32_t> ranges_shape = {4, 3};
      QnnParamWrapper ranges_param(node_unit.Index(), node_unit.Name() + "_x2_ranges",
                                   QNN_OP_STRIDED_SLICE_PARAM_RANGES,
                                   std::move(ranges_shape), std::move(ranges_data), true);
      slice_param_names.push_back(ranges_param.GetParamTensorName());
      qnn_model_wrapper.AddParamWrapper(std::move(ranges_param));

      QnnTensorWrapper x2_tensor(x2_name, QNN_TENSOR_TYPE_NATIVE,
                                 qnn_data_type, input_info.quant_param.Copy(),
                                 std::vector<uint32_t>(x1_x2_shape));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(x2_tensor)),
                        "Failed to add x2 tensor");

      ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                            utils::GetUniqueName(node_unit, "_slice_x2"),
                            QNN_OP_PACKAGE_NAME_QTI_AISW,
                            QNN_OP_STRIDED_SLICE,
                            std::move(slice_input_names),
                            {x2_name},
                            std::move(slice_param_names),
                            do_op_validation),
                        "Failed to create StridedSlice node for x2");
    }
  } else {
    // Use Split to divide into two contiguous halves
    std::vector<std::string> split_input_names = {x_rotate_name};
    std::vector<std::string> split_param_names;

    // axis = 3 (last dimension)
    Qnn_Scalar_t axis_scalar = QNN_SCALAR_INIT;
    axis_scalar.dataType = QNN_DATATYPE_INT_32;
    axis_scalar.int32Value = 3;
    QnnParamWrapper axis_param(node_unit.Index(), node_unit.Name() + "_split_axis",
                               QNN_OP_SPLIT_PARAM_AXIS, axis_scalar);
    split_param_names.push_back(axis_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(axis_param));

    // split_index = [rotary_half_dim] (implicit 0 at start)
    std::vector<uint32_t> split_index_data = {rotary_half_dim};
    std::vector<uint32_t> split_index_shape = {1};
    QnnParamWrapper split_index_param(node_unit.Index(), node_unit.Name() + "_split_index",
                                      QNN_OP_SPLIT_PARAM_SPLIT_INDEX,
                                      std::move(split_index_shape), std::move(split_index_data));
    split_param_names.push_back(split_index_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(split_index_param));

    // Create output tensors
    QnnTensorWrapper x1_tensor(x1_name, QNN_TENSOR_TYPE_NATIVE,
                               qnn_data_type, input_info.quant_param.Copy(),
                               std::vector<uint32_t>(x1_x2_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(x1_tensor)),
                      "Failed to add x1 tensor");

    QnnTensorWrapper x2_tensor(x2_name, QNN_TENSOR_TYPE_NATIVE,
                               qnn_data_type, input_info.quant_param.Copy(),
                               std::vector<uint32_t>(x1_x2_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(x2_tensor)),
                      "Failed to add x2 tensor");

    // Create Split node
    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                          utils::GetUniqueName(node_unit, "_split"),
                          QNN_OP_PACKAGE_NAME_QTI_AISW,
                          QNN_OP_SPLIT,
                          std::move(split_input_names),
                          {x1_name, x2_name},
                          std::move(split_param_names),
                          do_op_validation),
                      "Failed to create Split node");
  }

  // Step 4: Prepare cos/sin caches
  // If position_ids provided, gather rows; then reshape/broadcast to [B, S, 1, rotary_half_dim]
  std::string cos_prepared = cos_cache_name;
  std::string sin_prepared = sin_cache_name;

  if (has_position_ids) {
    // Gather cos/sin using position_ids
    const std::string& position_ids_name = input_names[3];

    // Gather cos_cache
    {
      std::string cos_gathered = utils::GetUniqueName(node_unit, "_cos_gathered");
      std::vector<std::string> gather_input_names = {cos_cache_name, position_ids_name};
      std::vector<std::string> gather_param_names;

      // axis = 0 (gather along sequence dimension)
      Qnn_Scalar_t axis_scalar = QNN_SCALAR_INIT;
      axis_scalar.dataType = QNN_DATATYPE_INT_32;
      axis_scalar.int32Value = 0;
      QnnParamWrapper axis_param(node_unit.Index(), node_unit.Name() + "_cos_gather_axis",
                                 QNN_OP_GATHER_PARAM_AXIS, axis_scalar);
      gather_param_names.push_back(axis_param.GetParamTensorName());
      qnn_model_wrapper.AddParamWrapper(std::move(axis_param));

      // Output shape: [B, S, rotary_half_dim]
      std::vector<uint32_t> cos_gathered_shape = {batch_size, seq_len, rotary_half_dim};
      QnnTensorWrapper cos_gathered_tensor(cos_gathered, QNN_TENSOR_TYPE_NATIVE,
                                           qnn_data_type, input_info.quant_param.Copy(),
                                           std::move(cos_gathered_shape));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(cos_gathered_tensor)),
                        "Failed to add cos_gathered tensor");

      ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                            utils::GetUniqueName(node_unit, "_gather_cos"),
                            QNN_OP_PACKAGE_NAME_QTI_AISW,
                            QNN_OP_GATHER,
                            std::move(gather_input_names),
                            {cos_gathered},
                            std::move(gather_param_names),
                            do_op_validation),
                        "Failed to create Gather node for cos");

      cos_prepared = cos_gathered;
    }

    // Gather sin_cache
    {
      std::string sin_gathered = utils::GetUniqueName(node_unit, "_sin_gathered");
      std::vector<std::string> gather_input_names = {sin_cache_name, position_ids_name};
      std::vector<std::string> gather_param_names;

      Qnn_Scalar_t axis_scalar = QNN_SCALAR_INIT;
      axis_scalar.dataType = QNN_DATATYPE_INT_32;
      axis_scalar.int32Value = 0;
      QnnParamWrapper axis_param(node_unit.Index(), node_unit.Name() + "_sin_gather_axis",
                                 QNN_OP_GATHER_PARAM_AXIS, axis_scalar);
      gather_param_names.push_back(axis_param.GetParamTensorName());
      qnn_model_wrapper.AddParamWrapper(std::move(axis_param));

      std::vector<uint32_t> sin_gathered_shape = {batch_size, seq_len, rotary_half_dim};
      QnnTensorWrapper sin_gathered_tensor(sin_gathered, QNN_TENSOR_TYPE_NATIVE,
                                           qnn_data_type, input_info.quant_param.Copy(),
                                           std::move(sin_gathered_shape));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(sin_gathered_tensor)),
                        "Failed to add sin_gathered tensor");

      ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                            utils::GetUniqueName(node_unit, "_gather_sin"),
                            QNN_OP_PACKAGE_NAME_QTI_AISW,
                            QNN_OP_GATHER,
                            std::move(gather_input_names),
                            {sin_gathered},
                            std::move(gather_param_names),
                            do_op_validation),
                        "Failed to create Gather node for sin");

      sin_prepared = sin_gathered;
    }
  }

  // Reshape cos/sin to [B, S, 1, rotary_half_dim] for broadcasting
  std::string cos_broadcast = utils::GetUniqueName(node_unit, "_cos_broadcast");
  std::string sin_broadcast = utils::GetUniqueName(node_unit, "_sin_broadcast");
  std::vector<uint32_t> broadcast_shape = {batch_size, seq_len, 1, rotary_half_dim};

  // Get cos/sin current shape
  std::vector<uint32_t> cos_current_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[1].node_arg, cos_current_shape),
                    "Cannot get cos_cache shape");

  // Reshape cos
  {
    std::vector<uint32_t> cos_input_shape = cos_current_shape;
    if (has_position_ids) {
      cos_input_shape = {batch_size, seq_len, rotary_half_dim};
    }

    ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(
        cos_prepared, cos_broadcast,
        cos_input_shape, broadcast_shape,
        qnn_data_type, input_info.quant_param, do_op_validation,
        false, false));
  }

  // Reshape sin
  {
    std::vector<uint32_t> sin_input_shape = cos_current_shape;
    if (has_position_ids) {
      sin_input_shape = {batch_size, seq_len, rotary_half_dim};
    }

    ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(
        sin_prepared, sin_broadcast,
        sin_input_shape, broadcast_shape,
        qnn_data_type, input_info.quant_param, do_op_validation,
        false, false));
  }

  // Step 5: Compute rotation
  // real = cos * x1 - sin * x2
  // imag = sin * x1 + cos * x2

  // cos * x1
  std::string cos_x1 = utils::GetUniqueName(node_unit, "_cos_x1");
  {
    std::vector<std::string> mul_input_names = {cos_broadcast, x1_name};
    QnnTensorWrapper cos_x1_tensor(cos_x1, QNN_TENSOR_TYPE_NATIVE,
                                   qnn_data_type, input_info.quant_param.Copy(),
                                   std::vector<uint32_t>(x1_x2_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(cos_x1_tensor)),
                      "Failed to add cos_x1 tensor");

    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                          utils::GetUniqueName(node_unit, "_mul_cos_x1"),
                          QNN_OP_PACKAGE_NAME_QTI_AISW,
                          QNN_OP_ELEMENT_WISE_MULTIPLY,
                          std::move(mul_input_names),
                          {cos_x1},
                          {},
                          do_op_validation),
                      "Failed to create Multiply node for cos*x1");
  }

  // sin * x2
  std::string sin_x2 = utils::GetUniqueName(node_unit, "_sin_x2");
  {
    std::vector<std::string> mul_input_names = {sin_broadcast, x2_name};
    QnnTensorWrapper sin_x2_tensor(sin_x2, QNN_TENSOR_TYPE_NATIVE,
                                   qnn_data_type, input_info.quant_param.Copy(),
                                   std::vector<uint32_t>(x1_x2_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(sin_x2_tensor)),
                      "Failed to add sin_x2 tensor");

    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                          utils::GetUniqueName(node_unit, "_mul_sin_x2"),
                          QNN_OP_PACKAGE_NAME_QTI_AISW,
                          QNN_OP_ELEMENT_WISE_MULTIPLY,
                          std::move(mul_input_names),
                          {sin_x2},
                          {},
                          do_op_validation),
                      "Failed to create Multiply node for sin*x2");
  }

  // real = cos * x1 - sin * x2
  std::string real = utils::GetUniqueName(node_unit, "_real");
  {
    std::vector<std::string> sub_input_names = {cos_x1, sin_x2};
    QnnTensorWrapper real_tensor(real, QNN_TENSOR_TYPE_NATIVE,
                                 qnn_data_type, input_info.quant_param.Copy(),
                                 std::vector<uint32_t>(x1_x2_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(real_tensor)),
                      "Failed to add real tensor");

    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                          utils::GetUniqueName(node_unit, "_sub_real"),
                          QNN_OP_PACKAGE_NAME_QTI_AISW,
                          QNN_OP_ELEMENT_WISE_SUBTRACT,
                          std::move(sub_input_names),
                          {real},
                          {},
                          do_op_validation),
                      "Failed to create Subtract node for real");
  }

  // sin * x1
  std::string sin_x1 = utils::GetUniqueName(node_unit, "_sin_x1");
  {
    std::vector<std::string> mul_input_names = {sin_broadcast, x1_name};
    QnnTensorWrapper sin_x1_tensor(sin_x1, QNN_TENSOR_TYPE_NATIVE,
                                   qnn_data_type, input_info.quant_param.Copy(),
                                   std::vector<uint32_t>(x1_x2_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(sin_x1_tensor)),
                      "Failed to add sin_x1 tensor");

    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                          utils::GetUniqueName(node_unit, "_mul_sin_x1"),
                          QNN_OP_PACKAGE_NAME_QTI_AISW,
                          QNN_OP_ELEMENT_WISE_MULTIPLY,
                          std::move(mul_input_names),
                          {sin_x1},
                          {},
                          do_op_validation),
                      "Failed to create Multiply node for sin*x1");
  }

  // cos * x2
  std::string cos_x2 = utils::GetUniqueName(node_unit, "_cos_x2");
  {
    std::vector<std::string> mul_input_names = {cos_broadcast, x2_name};
    QnnTensorWrapper cos_x2_tensor(cos_x2, QNN_TENSOR_TYPE_NATIVE,
                                   qnn_data_type, input_info.quant_param.Copy(),
                                   std::vector<uint32_t>(x1_x2_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(cos_x2_tensor)),
                      "Failed to add cos_x2 tensor");

    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                          utils::GetUniqueName(node_unit, "_mul_cos_x2"),
                          QNN_OP_PACKAGE_NAME_QTI_AISW,
                          QNN_OP_ELEMENT_WISE_MULTIPLY,
                          std::move(mul_input_names),
                          {cos_x2},
                          {},
                          do_op_validation),
                      "Failed to create Multiply node for cos*x2");
  }

  // imag = sin * x1 + cos * x2
  std::string imag = utils::GetUniqueName(node_unit, "_imag");
  {
    std::vector<std::string> add_input_names = {sin_x1, cos_x2};
    QnnTensorWrapper imag_tensor(imag, QNN_TENSOR_TYPE_NATIVE,
                                 qnn_data_type, input_info.quant_param.Copy(),
                                 std::vector<uint32_t>(x1_x2_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(imag_tensor)),
                      "Failed to add imag tensor");

    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                          utils::GetUniqueName(node_unit, "_add_imag"),
                          QNN_OP_PACKAGE_NAME_QTI_AISW,
                          QNN_OP_ELEMENT_WISE_ADD,
                          std::move(add_input_names),
                          {imag},
                          {},
                          do_op_validation),
                      "Failed to create Add node for imag");
  }

  // Step 6: Recombine real and imag based on interleaved flag
  std::string x_rotated = utils::GetUniqueName(node_unit, "_x_rotated");
  std::vector<uint32_t> x_rotated_shape = {batch_size, seq_len, num_heads_val, rotary_dim};

  if (interleaved) {
    // Interleave real and imag: stack along new axis then reshape
    // Stack to create [..., rotary_half_dim, 2]
    std::string stacked = utils::GetUniqueName(node_unit, "_stacked");
    std::vector<uint32_t> stacked_shape = {batch_size, seq_len, num_heads_val, rotary_half_dim, 2};

    // QNN doesn't have a direct Stack op, so we'll use Concat with Reshape
    // First reshape real and imag to add a dimension
    std::string real_reshaped = utils::GetUniqueName(node_unit, "_real_reshaped");
    std::string imag_reshaped = utils::GetUniqueName(node_unit, "_imag_reshaped");
    std::vector<uint32_t> reshaped_for_stack = {batch_size, seq_len, num_heads_val, rotary_half_dim, 1};

    ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(
        real, real_reshaped,
        x1_x2_shape, reshaped_for_stack,
        qnn_data_type, input_info.quant_param, do_op_validation,
        false, false));

    ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(
        imag, imag_reshaped,
        x1_x2_shape, reshaped_for_stack,
        qnn_data_type, input_info.quant_param, do_op_validation,
        false, false));

    // Concat along last dimension
    {
      std::vector<std::string> concat_input_names = {real_reshaped, imag_reshaped};
      std::vector<std::string> concat_param_names;

      Qnn_Scalar_t axis_scalar = QNN_SCALAR_INIT;
      axis_scalar.dataType = QNN_DATATYPE_INT_32;
      axis_scalar.int32Value = 4;  // last dimension
      QnnParamWrapper axis_param(node_unit.Index(), node_unit.Name() + "_stack_axis",
                                 QNN_OP_CONCAT_PARAM_AXIS, axis_scalar);
      concat_param_names.push_back(axis_param.GetParamTensorName());
      qnn_model_wrapper.AddParamWrapper(std::move(axis_param));

      QnnTensorWrapper stacked_tensor(stacked, QNN_TENSOR_TYPE_NATIVE,
                                     qnn_data_type, input_info.quant_param.Copy(),
                                     std::move(stacked_shape));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(stacked_tensor)),
                        "Failed to add stacked tensor");

      ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                            utils::GetUniqueName(node_unit, "_concat_stack"),
                            QNN_OP_PACKAGE_NAME_QTI_AISW,
                            QNN_OP_CONCAT,
                            std::move(concat_input_names),
                            {stacked},
                            std::move(concat_param_names),
                            do_op_validation),
                        "Failed to create Concat node for stacking");
    }

    // Reshape to flatten last two dimensions
    std::vector<uint32_t> stacked_shape_for_reshape = {batch_size, seq_len, num_heads_val, rotary_half_dim, 2};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(
        stacked, x_rotated,
        stacked_shape_for_reshape, x_rotated_shape,
        qnn_data_type, input_info.quant_param, do_op_validation,
        false, false));
  } else {
    // Concat real and imag along last dimension
    std::vector<std::string> concat_input_names = {real, imag};
    std::vector<std::string> concat_param_names;

    Qnn_Scalar_t axis_scalar = QNN_SCALAR_INIT;
    axis_scalar.dataType = QNN_DATATYPE_INT_32;
    axis_scalar.int32Value = 3;  // last dimension
    QnnParamWrapper axis_param(node_unit.Index(), node_unit.Name() + "_concat_axis",
                               QNN_OP_CONCAT_PARAM_AXIS, axis_scalar);
    concat_param_names.push_back(axis_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(axis_param));

    QnnTensorWrapper x_rotated_tensor(x_rotated, QNN_TENSOR_TYPE_NATIVE,
                                     qnn_data_type, input_info.quant_param.Copy(),
                                     std::move(x_rotated_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(x_rotated_tensor)),
                      "Failed to add x_rotated tensor");

    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                          utils::GetUniqueName(node_unit, "_concat_rotated"),
                          QNN_OP_PACKAGE_NAME_QTI_AISW,
                          QNN_OP_CONCAT,
                          std::move(concat_input_names),
                          {x_rotated},
                          std::move(concat_param_names),
                          do_op_validation),
                      "Failed to create Concat node for rotated");
  }

  // Step 7: Concatenate x_rotated with x_tail (if exists)
  std::string output_normalized = x_rotated;
  if (rotary_dim < head_size) {
    output_normalized = utils::GetUniqueName(node_unit, "_output_normalized");
    std::vector<std::string> concat_input_names = {x_rotated, x_tail_name};
    std::vector<std::string> concat_param_names;

    Qnn_Scalar_t axis_scalar = QNN_SCALAR_INIT;
    axis_scalar.dataType = QNN_DATATYPE_INT_32;
    axis_scalar.int32Value = 3;  // last dimension
    QnnParamWrapper axis_param(node_unit.Index(), node_unit.Name() + "_concat_tail_axis",
                               QNN_OP_CONCAT_PARAM_AXIS, axis_scalar);
    concat_param_names.push_back(axis_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(axis_param));

    QnnTensorWrapper output_normalized_tensor(output_normalized, QNN_TENSOR_TYPE_NATIVE,
                                              qnn_data_type, input_info.quant_param.Copy(),
                                              std::vector<uint32_t>(normalized_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_normalized_tensor)),
                      "Failed to add output_normalized tensor");

    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                          utils::GetUniqueName(node_unit, "_concat_tail"),
                          QNN_OP_PACKAGE_NAME_QTI_AISW,
                          QNN_OP_CONCAT,
                          std::move(concat_input_names),
                          {output_normalized},
                          std::move(concat_param_names),
                          do_op_validation),
                      "Failed to create Concat node for tail");
  }

  // Step 8: Restore original layout
  const std::string& output_name = outputs[0].node_arg.Name();
  const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);

  if (is_4d_input) {
    // Transpose back from [B, S, NH, HS] to [B, NH, S, HS]
    std::vector<uint32_t> transpose_perm = {0, 2, 1, 3};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(
        node_unit.Index(), output_normalized, output_name,
        normalized_shape, transpose_perm, input_shape,
        qnn_data_type, input_info.quant_param, do_op_validation,
        false, is_graph_output));
  } else {
    // Reshape back from [B, S, NH, HS] to [B, S, NH*HS]
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(
        output_normalized, output_name,
        normalized_shape, input_shape,
        qnn_data_type, input_info.quant_param, do_op_validation,
        false, is_graph_output));
  }

  LOGS(logger, VERBOSE) << "Successfully decomposed RotaryEmbedding into QNN ops";
  return Status::OK();
}

void CreateRopeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<RopeOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
