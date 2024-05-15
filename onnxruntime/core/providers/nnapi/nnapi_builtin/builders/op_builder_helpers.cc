// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nnapi/nnapi_builtin/builders/op_builder_helpers.h"

#include <algorithm>
#include <functional>
#include <optional>
#include <utility>

#include "core/common/gsl.h"

#include "core/common/inlined_containers_fwd.h"
#include "core/common/logging/logging.h"
#include "core/common/narrow.h"
#include "core/common/safeint.h"
#include "core/common/span_utils.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/node_arg.h"
#include "core/graph/graph_viewer.h"
#include "core/optimizer/initializer.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/nnapi/nnapi_builtin/builders/impl/base_op_builder.h"

namespace onnxruntime::nnapi::op_builder_helpers {

using android::nn::wrapper::OperandType, android::nn::wrapper::Type;

Status AddNnapiTranspose(ModelBuilder& model_builder,
                         const std::string& data_input,
                         const std::string& perm_input,
                         const gsl::span<const int32_t> perm,
                         const std::string& output) {
  auto& shaper(model_builder.GetShaper());

  // Calculate transpose output shape

  const Shape input_dimen = shaper[data_input];
  ORT_RETURN_IF_NOT(perm.size() == input_dimen.size(), "Invalid perm is given!");
  size_t size = input_dimen.size();
  Shape output_dimen(size);
  for (size_t i = 0; i < size; i++)
    output_dimen[i] = input_dimen[perm[i]];

  shaper.AddShape(output, output_dimen);

  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  InlinedVector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(data_input));  // input

  Shape perm_dimen = {SafeInt<uint32_t>(perm.size())};
  OperandType perm_operand_type(Type::TENSOR_INT32, perm_dimen);
  ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(perm_input, perm.data(), perm_operand_type));
  uint32_t perm_idx = operand_indices.at(perm_input);
  input_indices.push_back(perm_idx);  // permutation

  OperandType output_operand_type = operand_types.at(data_input);
  output_operand_type.SetDimensions(output_dimen);
  return model_builder.AddOperation(ANEURALNETWORKS_TRANSPOSE, input_indices, {output},
                                    {output_operand_type});
}

Status AddNnapiReshape(ModelBuilder& model_builder,
                       const std::string& data_input,
                       const std::string& shape_input, const std::vector<int32_t>& shape_value,
                       const std::string& output) {
  auto& shaper = model_builder.GetShaper();
  // Calculate reshape output shape

  const Shape input_dimen = shaper[data_input];
  uint32_t input_size = ShapeSize(input_dimen);
  Shape output_dimen(shape_value.size());

  int64_t capacity = 1;
  int unk_dim_idx = -1;
  for (size_t i = 0; i < shape_value.size(); i++) {
    int32_t dim_i = shape_value[i];
    ORT_RETURN_IF_NOT(dim_i != 0, "NNAPI does not support 0 reshape dimension");
    if (dim_i == -1) {
      ORT_RETURN_IF_NOT(unk_dim_idx == -1, "Only one input dimension of Attr(shape) can be unknown!");
      unk_dim_idx = static_cast<int>(i);
    } else {
      capacity *= dim_i;
      output_dimen[i] = static_cast<uint32_t>(dim_i);
    }
  }

  if (unk_dim_idx != -1) {
    if (input_size == 0)
      output_dimen[unk_dim_idx] = 0;
    else
      output_dimen[unk_dim_idx] = static_cast<uint32_t>(input_size / capacity);

    capacity *= output_dimen[unk_dim_idx];
  }

  ORT_RETURN_IF_NOT(capacity == input_size, "Invalid shape is given!");

  shaper.AddShape(output, output_dimen);

  const auto& operand_indices = model_builder.GetOperandIndices();
  const auto& operand_types = model_builder.GetOperandTypes();

  // Add input
  InlinedVector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(data_input));

  // Add new shape
  const Shape shape_dimen{static_cast<uint32_t>(shape_value.size())};
  const OperandType shape_operand_type{Type::TENSOR_INT32, shape_dimen};
  ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(shape_input, shape_value.data(),
                                                                      shape_operand_type));
  input_indices.push_back(operand_indices.at(shape_input));

  // For reshape, the output type should be the same as the input type except the shape is different
  OperandType output_operand_type{operand_types.at(data_input)};
  output_operand_type.SetDimensions(output_dimen);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_RESHAPE,
                                                 input_indices, {output}, {output_operand_type}));

  return Status::OK();
}

Status AddNnapiSplit(ModelBuilder& model_builder,
                     const std::string& input,
                     int32_t axis,
                     const std::vector<std::string>& outputs) {
  const auto& operand_indices = model_builder.GetOperandIndices();
  const auto& operand_types = model_builder.GetOperandTypes();
  auto& shaper = model_builder.GetShaper();

  const auto input_rank = shaper[input].size();
  axis = static_cast<int32_t>(HandleNegativeAxis(axis, input_rank));

  const auto count = narrow<int32_t>(outputs.size());

  // Calculate split output shape
  {
    const auto input_shape = shaper[input];
    ORT_RETURN_IF_NOT(input_shape[axis] % count == 0,
                      "count [", count, "] does not evenly divide dimension ", axis, " [", input_shape[axis], "]");

    Shape output_shape = input_shape;
    output_shape[axis] = SafeInt<uint32_t>(input_shape[axis] / count);
    for (const auto& output_name : outputs) {
      shaper.AddShape(output_name, output_shape);
    }
  }

  InlinedVector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  ORT_RETURN_IF_ERROR(AddScalarOperand(model_builder, input_indices, axis));
  ORT_RETURN_IF_ERROR(AddScalarOperand(model_builder, input_indices, count));

  const OperandType& input_operand_type = operand_types.at(input);
  std::vector<OperandType> output_operand_types;
  output_operand_types.reserve(count);
  std::transform(outputs.begin(), outputs.end(), std::back_inserter(output_operand_types),
                 [&](const std::string& output) {
                   OperandType output_operand_type = input_operand_type;
                   output_operand_type.SetDimensions(shaper[output]);
                   return output_operand_type;
                 });

  ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_SPLIT,
                                                 input_indices, outputs, output_operand_types));

  return Status::OK();
}

Status AddNnapiBatchNormalization(ModelBuilder& model_builder,
                                  const std::string& input1,
                                  const std::string& input2,
                                  const std::string& input3,
                                  const std::string& output1,
                                  const std::string& output2,
                                  int32_t fuse_code,
                                  float output_scale,
                                  int32_t output_zero_point) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  // Add Nnapi Mul
  InlinedVector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input1));
  input_indices.push_back(operand_indices.at(input2));

  ADD_SCALAR_OPERAND(model_builder, input_indices, ANEURALNETWORKS_FUSED_NONE);

  const Shape& shape1 = shaper[input1];
  const Shape& shape2 = shaper[input2];
  Shape output1_shape;
  // broadcasting support for eltwise shape operation
  ORT_RETURN_IF_ERROR(op_builder_helpers::PerformBroadcasting(shape1, shape2, output1_shape));
  const OperandType output_operand_type(operand_types.at(input1).type, output1_shape,
                                        output_scale, output_zero_point);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_MUL, input_indices,
                                                 {output1}, {output_operand_type}));

  // Add Nnapi Add
  input_indices.clear();
  input_indices.push_back(operand_indices.at(output1));
  input_indices.push_back(operand_indices.at(input3));

  ADD_SCALAR_OPERAND(model_builder, input_indices, fuse_code);

  const Shape& shape3 = shaper[input3];
  Shape output2_shape;
  ORT_RETURN_IF_ERROR(op_builder_helpers::PerformBroadcasting(output1_shape, shape3, output2_shape));
  const OperandType output_operand_type2(operand_types.at(input3).type, output2_shape,
                                         output_scale, output_zero_point);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_ADD, input_indices,
                                                 {output2}, {output_operand_type2}));
  return Status::OK();
}

bool IsSupportedBatchMatMul(const NodeUnit& node_unit, int32_t nnapi_feature_level) {
  // Currently, batch MatMul is composed of various operations including ANEURALNETWORKS_SPLIT which requires
  // ANEURALNETWORKS_FEATURE_LEVEL_3.
  const auto min_nnapi_feature_level = ANEURALNETWORKS_FEATURE_LEVEL_3;
  if (nnapi_feature_level < min_nnapi_feature_level) {
    LOGS_DEFAULT(VERBOSE) << "Minimum NNAPI feature level required: " << min_nnapi_feature_level
                          << ", actual: " << nnapi_feature_level;
    return false;
  }

  // Only support non-QDQ MatMul for now.
  // TODO could be expanded to support QLinearMatMul and QDQ MatMul
  if (node_unit.UnitType() != NodeUnit::Type::SingleNode ||
      node_unit.OpType() != "MatMul") {
    LOGS_DEFAULT(VERBOSE) << "Unsupported op type: "
                          << (node_unit.UnitType() == NodeUnit::Type::QDQGroup ? "QDQ " : "") << node_unit.OpType();
    return false;
  }

  const auto& inputs = node_unit.Inputs();

  // Verify shapes
  // A and B should have at least three dimensions* and have the same leading dimensions except for the last two.
  // [*] Having two dimensions is valid for a MatMul but for simplicity we don't support it in the current batch
  // MatMul implementation. That case is handled by the regular Gemm/MatMul op building logic.
  Shape a_shape;
  if (!GetShape(inputs[0].node_arg, a_shape)) {
    return false;
  }

  Shape b_shape;
  if (!GetShape(inputs[1].node_arg, b_shape)) {
    return false;
  }

  if (a_shape.size() < 3 ||
      a_shape.size() != b_shape.size() ||
      !std::equal(a_shape.begin(), a_shape.end() - 2,
                  b_shape.begin(), b_shape.end() - 2)) {
    LOGS_DEFAULT(VERBOSE)
        << "A and B must have at least three dimensions and have the same leading dimensions except for the last two. "
        << "A shape: " << Shape2String(a_shape) << ", B shape: " << Shape2String(b_shape);
    return false;
  }

  // Verify type
  int32_t a_type;
  if (!GetType(inputs[0].node_arg, a_type)) {
    return false;
  }

  // Only support float for now.
  // TODO could be expanded to support other types
  if (a_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    LOGS_DEFAULT(VERBOSE) << "Unsupported element data type: " << a_type;
    return false;
  }

  return true;
}

Status BuildBatchMatMul(ModelBuilder& model_builder, const NodeUnit& node_unit) {
  // we will implement batch MatMul by composing NNAPI operations
  // this could be replaced with ANEURALNETWORKS_BATCH_MATMUL when that is more widely supported

  // assuming A and B have at least three dimensions and the same leading dimensions other than the last two

  const auto& inputs = node_unit.Inputs();

  Shape a_shape;
  ORT_RETURN_IF_NOT(GetShape(inputs[0].node_arg, a_shape), "Failed to get A's shape.");

  Shape b_shape;
  ORT_RETURN_IF_NOT(GetShape(inputs[1].node_arg, b_shape), "Failed to get B's shape.");

  const std::string& a = inputs[0].node_arg.Name();
  const std::string& b = inputs[1].node_arg.Name();

  const std::string& output = node_unit.Outputs()[0].node_arg.Name();

  std::vector<std::string> gemm_a_inputs{a};
  std::vector<std::string> gemm_b_inputs{b};

  const auto m = a_shape[a_shape.size() - 2],
             k = a_shape[a_shape.size() - 1],
             n = b_shape[b_shape.size() - 1];

  const bool reshape_leading_dimensions = a_shape.size() > 3;
  const auto batch_size = ShapeSize(a_shape, 0, a_shape.size() - 2);

  auto add_reshape = [&model_builder](const std::string& input, const Shape& new_shape,
                                      const std::string& output) -> Status {
    const std::string new_shape_name = model_builder.GetUniqueName(input + "/new_shape");
    std::vector<int32_t> new_shape_i32{};
    new_shape_i32.reserve(new_shape.size());
    std::transform(new_shape.begin(), new_shape.end(), std::back_inserter(new_shape_i32),
                   [](uint32_t d) { return narrow<int32_t>(d); });
    ORT_RETURN_IF_ERROR(AddNnapiReshape(model_builder, input, new_shape_name, new_shape_i32, output));
    return Status::OK();
  };

  auto add_reshape_generate_output = [&model_builder, &add_reshape](const std::string& input, const Shape& new_shape,
                                                                    std::string& output) -> Status {
    std::string reshaped = model_builder.GetUniqueName(input + "/reshaped");
    ORT_RETURN_IF_ERROR(add_reshape(input, new_shape, reshaped));
    output = std::move(reshaped);
    return Status::OK();
  };

  // collapse leading dimensions to a single one
  if (reshape_leading_dimensions) {
    const Shape a_new_shape_value = {batch_size, m, k},
                b_new_shape_value = {batch_size, k, n};
    std::string a_reshaped, b_reshaped;

    ORT_RETURN_IF_ERROR(add_reshape_generate_output(gemm_a_inputs.front(), a_new_shape_value, a_reshaped));
    gemm_a_inputs.front() = a_reshaped;

    ORT_RETURN_IF_ERROR(add_reshape_generate_output(gemm_b_inputs.front(), b_new_shape_value, b_reshaped));
    gemm_b_inputs.front() = b_reshaped;
  }

  // transpose B
  {
    const std::string b_new_perm = model_builder.GetUniqueName(b + "/new_perm"),
                      b_transposed = model_builder.GetUniqueName(b + "/transposed");
    ORT_RETURN_IF_ERROR(AddNnapiTranspose(model_builder, gemm_b_inputs.front(), b_new_perm,
                                          AsSpan<int32_t>({0, 2, 1}), b_transposed));
    gemm_b_inputs.front() = b_transposed;
  }

  // split batch
  {
    auto add_split = [&model_builder, batch_size](const std::string& input,
                                                  std::vector<std::string>& outputs_result) -> Status {
      std::vector<std::string> outputs;
      outputs.reserve(batch_size);
      for (size_t i = 0; i < batch_size; ++i) {
        outputs.push_back(model_builder.GetUniqueName(MakeString(input, "/split_", i)));
      }
      ORT_RETURN_IF_ERROR(AddNnapiSplit(model_builder, input, 0, outputs));
      outputs_result = std::move(outputs);
      return Status::OK();
    };

    std::vector<std::string> a_split_outputs;
    ORT_RETURN_IF_ERROR(add_split(gemm_a_inputs.front(), a_split_outputs));
    gemm_a_inputs = std::move(a_split_outputs);

    std::vector<std::string> b_split_outputs;
    ORT_RETURN_IF_ERROR(add_split(gemm_b_inputs.front(), b_split_outputs));
    gemm_b_inputs = std::move(b_split_outputs);
  }

  // GEMM per matrix pair
  std::vector<std::string> gemm_outputs;
  gemm_outputs.reserve(batch_size);
  {
    const std::string bias = model_builder.GetUniqueName(node_unit.Name() + "/zero_bias");
    {
      if (model_builder.GetOperandTypes().at(b).type != Type::TENSOR_FLOAT32) {
        ORT_NOT_IMPLEMENTED("Only float input is supported now.");
      }
      const Shape bias_shape{n};
      const std::vector<float> buffer(n, 0.0f);
      const OperandType bias_operand_type(Type::TENSOR_FLOAT32, bias_shape);
      ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(bias, buffer.data(), bias_operand_type));
    }

    auto add_fc = [&model_builder, &bias](const std::string& a, const std::string& b_transposed,
                                          const std::string& output) -> Status {
      const auto& operand_indices = model_builder.GetOperandIndices();
      const auto& operand_types = model_builder.GetOperandTypes();
      auto& shaper = model_builder.GetShaper();
      InlinedVector<uint32_t> input_indices;
      input_indices.push_back(operand_indices.at(a));             // A
      input_indices.push_back(operand_indices.at(b_transposed));  // B'
      input_indices.push_back(operand_indices.at(bias));          // C
      int32_t fuse_code = ANEURALNETWORKS_FUSED_NONE;
      ORT_RETURN_IF_ERROR(AddScalarOperand(model_builder, input_indices, fuse_code));

      const auto a_dimen = shaper[a];
      const auto b_transposed_dimen = shaper[b_transposed];  // num_units, input_size
      Shape output_dimen{a_dimen[0], b_transposed_dimen[0]};
      shaper.AddShape(output, output_dimen);
      const OperandType output_operand_type(operand_types.at(a).type, output_dimen);
      ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_FULLY_CONNECTED, input_indices,
                                                     {output}, {output_operand_type}));
      return Status::OK();
    };

    for (uint32_t i = 0; i < batch_size; ++i) {
      const auto &gemm_a_input = gemm_a_inputs[i],
                 &gemm_b_input = gemm_b_inputs[i];

      // make inputs 2D ([1, x, y] -> [x, y])
      std::string a_2d, b_transposed_2d;
      ORT_RETURN_IF_ERROR(add_reshape_generate_output(gemm_a_input, Shape{m, k}, a_2d));
      ORT_RETURN_IF_ERROR(add_reshape_generate_output(gemm_b_input, Shape{n, k}, b_transposed_2d));

      const std::string gemm_output = model_builder.GetUniqueName(MakeString(node_unit.Name(), "/gemm_", i));
      ORT_RETURN_IF_ERROR(add_fc(a_2d, b_transposed_2d, gemm_output));

      // reshape output for concatenation ([x, y] -> [1, x, y])
      std::string gemm_output_3d;
      ORT_RETURN_IF_ERROR(add_reshape_generate_output(gemm_output, Shape{1, m, n}, gemm_output_3d));

      gemm_outputs.push_back(gemm_output_3d);
    }
  }

  // concat batch
  const std::string joined_gemm_output =
      reshape_leading_dimensions ? model_builder.GetUniqueName(node_unit.Name() + "/joined_gemm_output") : output;
  {
    auto add_concat = [&model_builder](const std::vector<std::string>& inputs,
                                       const std::string& output) -> Status {
      const auto& operand_indices = model_builder.GetOperandIndices();
      const auto& operand_types = model_builder.GetOperandTypes();
      auto& shaper = model_builder.GetShaper();
      InlinedVector<uint32_t> input_indices;
      input_indices.reserve(inputs.size() + 1);
      std::transform(inputs.begin(), inputs.end(), std::back_inserter(input_indices),
                     [&operand_indices](const std::string& input) { return operand_indices.at(input); });
      const int32_t axis = 0;
      ORT_RETURN_IF_ERROR(AddScalarOperand(model_builder, input_indices, axis));

      // Calculate_concat_output_shape
      {
        std::vector<Shape> dimens;
        for (const auto& input_name : inputs) {
          const Shape dimen = shaper[input_name];
          dimens.push_back(dimen);
        }

        // If one of the inputs has dynamic shape (at axis), we will keep the dimen[axis] as 0 (dynamic)
        auto output_dimen = dimens[0];
        if (output_dimen[axis] != 0) {
          for (size_t i = 1; i < dimens.size(); i++) {
            if (dimens[i][axis] == 0) {
              output_dimen[axis] = 0;
              break;
            }
            output_dimen[axis] += dimens[i][axis];
          }
        }
        shaper.AddShape(output, output_dimen);
      }

      OperandType output_operand_type = operand_types.at(inputs[0]);
      output_operand_type.SetDimensions(shaper[output]);
      ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_CONCATENATION,
                                                     input_indices, {output}, {output_operand_type}));
      return Status::OK();
    };

    ORT_RETURN_IF_ERROR(add_concat(gemm_outputs, joined_gemm_output));
  }

  // reshape to original dimensions
  if (reshape_leading_dimensions) {
    Shape new_shape = a_shape;
    new_shape[new_shape.size() - 2] = m;
    new_shape[new_shape.size() - 1] = n;
    ORT_RETURN_IF_ERROR(add_reshape(joined_gemm_output, new_shape, output));
  }

  return Status::OK();
}

Status AddInitializerInNewLayout(ModelBuilder& model_builder,
                                 const std::string& name,
                                 const OperandType& source_operand_type,
                                 DataLayout new_layout,
                                 bool is_per_tensor_u8s8) {
  const auto& tensor = *model_builder.GetInitializerTensors().at(name);
  const Shape& shape = source_operand_type.dimensions;
  ORT_RETURN_IF_NOT(shape.size() == 4,
                    "The initializer is not 4D: ", name, " actual dim ", shape.size());

  // TODO support other data types
  const uint8_t* src = nullptr;

  switch (tensor.data_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
    case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
      break;
    }
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "The initializer of graph ", name,
                             " doesn't have valid type: ", tensor.data_type());
  }
  Initializer unpacked_tensor(tensor, model_builder.GetGraphViewer().ModelPath());
  src = unpacked_tensor.DataAsByteSpan().data();
  const auto out_t = shape[0], in_t = shape[1],
             h_t = shape[2], w_t = shape[3];
  Shape dest_shape;
  if (new_layout == L_0231)
    dest_shape = {out_t, h_t, w_t, in_t};  // L_0231
  else
    dest_shape = {in_t, h_t, w_t, out_t};  // L_1230 for depthwise conv weight

  OperandType operand_type = source_operand_type;
  operand_type.SetDimensions(dest_shape);
  std::unique_ptr<uint8_t[]> buffer_holder(new uint8_t[operand_type.GetOperandBlobByteSize()]);
  uint8_t* buffer = buffer_holder.get();
  size_t element_size = operand_type.GetElementByteSize();

  uint8_t bit_flip_val = is_per_tensor_u8s8 ? 0x80 : 0;
  for (uint32_t out = 0; out < out_t; out++) {
    for (uint32_t in = 0; in < in_t; in++) {
      for (uint32_t h = 0; h < h_t; h++) {
        for (uint32_t w = 0; w < w_t; w++) {
          auto onnx_idx = out * in_t * h_t * w_t +
                          in * h_t * w_t +
                          h * w_t +
                          w;

          uint32_t nnapi_idx;
          if (new_layout == L_0231) {  // L_0231
            nnapi_idx = out * h_t * w_t * in_t +
                        h * w_t * in_t +
                        w * in_t +
                        in;
          } else {  // L_1230 for depthwise conv weight
            nnapi_idx = in * h_t * w_t * out_t +
                        h * w_t * out_t +
                        w * out_t +
                        out;
          }

          for (size_t i = 0; i < element_size; i++) {
            buffer[element_size * nnapi_idx + i] = src[element_size * onnx_idx + i] ^ bit_flip_val;
          }
        }
      }
    }
  }

  return model_builder.AddOperandFromPersistMemoryBuffer(name, &buffer[0], operand_type);
}

Status AddInitializerTransposed(ModelBuilder& model_builder,
                                const OperandType& source_operand_type,
                                const std::string& name,
                                bool is_per_tensor_u8s8) {
  const auto& tensor = *model_builder.GetInitializerTensors().at(name);
  const Shape& shape = source_operand_type.dimensions;

  ORT_RETURN_IF_NOT(shape.size() == 2,
                    "The initializer is not 2D: ", name, " actual dim ", shape.size());

  // TODO support other data types
  const uint8_t* src = nullptr;
  switch (tensor.data_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
    case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
      break;
    }
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "The initializer of graph ", name,
                             " doesn't have valid type: ", tensor.data_type());
  }
  Initializer unpacked_tensor(tensor, model_builder.GetGraphViewer().ModelPath());
  // could be float/u8/s8, so we have to use raw data here.
  src = unpacked_tensor.DataAsByteSpan().data();
  const auto x_t = shape[0], y_t = shape[1];
  Shape dest_shape = {y_t, x_t};
  OperandType operand_type = source_operand_type;
  operand_type.SetDimensions(dest_shape);
  std::unique_ptr<uint8_t[]> buffer_holder(new uint8_t[operand_type.GetOperandBlobByteSize()]);
  uint8_t* buffer = buffer_holder.get();
  size_t element_size = operand_type.GetElementByteSize();
  uint8_t bit_flip_val = is_per_tensor_u8s8 ? 0x80 : 0;
  for (uint32_t x = 0; x < x_t; x++) {
    for (uint32_t y = 0; y < y_t; y++) {
      for (size_t i = 0; i < element_size; i++) {
        buffer[element_size * (y * x_t + x) + i] = src[element_size * (x * y_t + y) + i] ^ bit_flip_val;
      }
    }
  }

  return model_builder.AddOperandFromPersistMemoryBuffer(name, &buffer[0], operand_type);
}

Status ComputeConvPads(
    const Shape& input_dimen,
    const uint32_t weight_size_y, const uint32_t weight_size_x,
    const std::vector<int32_t>& onnx_pads, const std::vector<int32_t>& onnx_strides, const std::vector<int32_t>& onnx_dilations,
    AutoPadType auto_pad_type, bool nchw,
    std::vector<int32_t>& pads_out) {
  const int32_t input_size_y = nchw ? input_dimen[2] : input_dimen[1];
  const int32_t input_size_x = nchw ? input_dimen[3] : input_dimen[2];
  const int32_t stride_y = onnx_strides[0];
  const int32_t stride_x = onnx_strides[1];
  const int32_t dilation_y = onnx_dilations[0];
  const int32_t dilation_x = onnx_dilations[1];

  int64_t padding_top = onnx_pads[0];
  int64_t padding_bottom = onnx_pads[2];
  int64_t padding_left = onnx_pads[1];
  int64_t padding_right = onnx_pads[3];

  ORT_RETURN_IF_ERROR(ComputePad(input_size_y,
                                 stride_y, weight_size_y, dilation_y,
                                 auto_pad_type,
                                 padding_top, padding_bottom));
  ORT_RETURN_IF_ERROR(ComputePad(input_size_x,
                                 stride_x, weight_size_x, dilation_x,
                                 auto_pad_type,
                                 padding_left, padding_right));

  pads_out = {static_cast<int32_t>(padding_top), static_cast<int32_t>(padding_left),
              static_cast<int32_t>(padding_bottom), static_cast<int32_t>(padding_right)};

  return Status::OK();
}

Status HandleAutoPad(const Shape& input_shape,
                     const uint32_t weight_size_y,
                     const uint32_t weight_size_x,
                     const std::vector<int32_t>& onnx_strides,
                     const std::vector<int32_t>& onnx_dilations,
                     AutoPadType auto_pad_type,
                     bool use_nchw,
                     std::vector<int32_t>& onnx_pads,
                     int32_t& nnapi_padding_code,
                     bool& use_auto_pad) {
  use_auto_pad = false;
  if (auto_pad_type != AutoPadType::NOTSET) {
    ORT_RETURN_IF_ERROR(ComputeConvPads(input_shape, weight_size_y, weight_size_x,
                                        onnx_pads, onnx_strides, onnx_dilations,
                                        auto_pad_type, use_nchw,
                                        onnx_pads));

    if (AutoPadType::VALID == auto_pad_type || AutoPadType::SAME_UPPER == auto_pad_type) {
      use_auto_pad = true;
      nnapi_padding_code = (AutoPadType::VALID == auto_pad_type) ? ANEURALNETWORKS_PADDING_VALID
                                                                 : ANEURALNETWORKS_PADDING_SAME;
    }
  } else if (onnx_dilations == std::vector<int32_t>{1, 1}) {
    // Since NNAPI runs more efficiently using auto_pad, we try to map the NOTSET padding to auto_pad
    std::vector<int32_t> same_upper_pads;
    ORT_RETURN_IF_ERROR(ComputeConvPads(input_shape, weight_size_y, weight_size_x,
                                        onnx_pads, onnx_strides, onnx_dilations,
                                        AutoPadType::SAME_UPPER, use_nchw,
                                        same_upper_pads));
    if (onnx_pads == same_upper_pads) {
      use_auto_pad = true;
      nnapi_padding_code = ANEURALNETWORKS_PADDING_SAME;
    }
  }

  return Status::OK();
}

Status GetBinaryOpQuantizationScaleAndZeroPoint(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                                                float& a_scale, float& b_scale, float& y_scale,
                                                int32_t& a_zero_point, int32_t& b_zero_point, int32_t& y_zero_point) {
  ORT_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(
      graph_viewer, node_unit.Inputs()[0], node_unit.ModelPath(), a_scale, a_zero_point));
  ORT_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(
      graph_viewer, node_unit.Inputs()[1], node_unit.ModelPath(), b_scale, b_zero_point));
  ORT_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(
      graph_viewer, node_unit.Outputs()[0], node_unit.ModelPath(), y_scale, y_zero_point));

  return Status::OK();
}

Status GetConvMatMulOpQuantizationScaleAndZeroPoint(
    const ModelBuilder& model_builder, const NodeUnit& node_unit,
    float& a_scale, float& w_scale, float& y_scale,
    int32_t& a_zero_point, int32_t& w_zero_point, int32_t& y_zero_point,
    std::optional<std::vector<float>>& w_scales, bool& is_per_tensor_u8s8) {
  is_per_tensor_u8s8 = false;
  const auto& graph_viewer(model_builder.GetGraphViewer());

  // Get scale and zero points
  // We will handle per-channel weight scale and zero point later
  ORT_RETURN_IF_ERROR(
      GetBinaryOpQuantizationScaleAndZeroPoint(graph_viewer, node_unit,
                                               a_scale, w_scale, y_scale,
                                               a_zero_point, w_zero_point, y_zero_point));

  const auto& inputs = node_unit.Inputs();
  // all these were checked to be constant in GemmOpBuilder::IsOpSupportedImpl
  const auto& weight_tensor = *graph_viewer.GetConstantInitializer(inputs[1].node_arg.Name());

  // We are done here if this is u8u8 QLinearConv
  if (weight_tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_UINT8)
    return Status::OK();

  // This is per-tensor u8s8
  // NNAPI does not support per-tensor u8s8
  // For this case we will need to convert the int8 weight tensor to uint8
  // And have same scale and 128 as zero point
  // The conversion of the weight tensor itself will be done in the OpBuilder
  const auto& scale_tensor = *graph_viewer.GetConstantInitializer(inputs[1].quant_param->scale.Name());
  int64_t scale_dim = scale_tensor.dims().empty() ? 1 : scale_tensor.dims()[0];
  if (scale_dim == 1) {
    w_zero_point = 128;
    is_per_tensor_u8s8 = true;
    return Status::OK();
  }

  // Now we have u8s8 per-channel QlinearConv
  // u8s8 QlinearConv always have 0 as zero point so we are not getting it here
  // and we do not use w_scale here, so we reset them back to 0
  w_scale = 0.0f;
  w_zero_point = 0;

  // We need to copy the 1d scales array for per-channel quantization
  Initializer unpacked_tensor(scale_tensor);
  auto scales = unpacked_tensor.DataAsSpan<float>();
  const size_t scales_size = scale_tensor.dims().empty() ? 1 : narrow<size_t>(scale_tensor.dims()[0]);
  std::vector<float> scales_vec(scales.begin(), scales.begin() + scales_size);
  w_scales = onnxruntime::make_optional(std::move(scales_vec));

  return Status::OK();
}

Status IsValidInputQuantizedType(const ModelBuilder& model_builder,
                                 const std::string& input_name,
                                 float scale,
                                 int32_t zero_point) {
  const OperandType& input_operand_type = model_builder.GetOperandTypes().at(input_name);
  if (input_operand_type.operandType.scale != scale) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input [", input_name,
                           "] NNAPI input scale: ", input_operand_type.operandType.scale,
                           ", ONNX input scale: ", scale);
  }

  if (input_operand_type.operandType.zeroPoint != zero_point) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input [", input_name,
                           "] NNAPI input zero point: ", input_operand_type.operandType.zeroPoint,
                           ", ONNX input zero point: ", zero_point);
  }

  return Status::OK();
}

Status IsValidConvWeightQuantizedType(const ModelBuilder& model_builder,
                                      const std::string& input_name,
                                      float scale,
                                      int32_t zero_point,
                                      const std::optional<std::vector<float>>& scales) {
  // first verify as the weight has no per-channel quantization
  ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input_name, scale, zero_point));

  if (scales) {
    const OperandType& input_operand_type = model_builder.GetOperandTypes().at(input_name);
    if (!input_operand_type.channelQuant) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input [", input_name, "] has no channelQuant");
    }

    if (input_operand_type.channelQuant.value().scales != scales.value()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input [", input_name, "] has mismatch scales between onnx and NNAPI");
    }
  }

  return Status::OK();
}

Status IsOpInRequiredLayout(bool use_nchw, const NodeUnit& node_unit) {
  bool is_op_nhwc = node_unit.Domain() == kMSInternalNHWCDomain;
  if (is_op_nhwc && use_nchw) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Expected layout and operator layout do not match. Possible bug in layout optimizer.");
  }

  return Status::OK();
}
void AddQuantizationScaleAndZeroPointToSkip(ModelBuilder& model_builder,
                                            const NodeUnitIODef::QuantParam& quant_param) {
  // If we reach here, we assume the io_def has quant_param
  model_builder.AddInitializerToSkip(quant_param.scale.Name());  // scale
  LOGS_DEFAULT(VERBOSE) << quant_param.scale.Name() << " is skipped";
  if (quant_param.zero_point) {
    model_builder.AddInitializerToSkip(quant_param.zero_point->Name());  // zero_point
    LOGS_DEFAULT(VERBOSE) << quant_param.zero_point->Name() << " is skipped";
  }
}

void AddInputToSkip(ModelBuilder& model_builder, const NodeUnitIODef& io_def) {
  model_builder.AddInitializerToSkip(io_def.node_arg.Name());  // main input
  if (io_def.quant_param)
    AddQuantizationScaleAndZeroPointToSkip(model_builder, *io_def.quant_param);
}

Status AddBinaryOperator(int32_t op_type,
                         ModelBuilder& model_builder,
                         const std::string& input1,
                         const std::string& input2,
                         bool add_activation,
                         int32_t fuse_code,
                         const std::string& output,
                         float output_scale,
                         int32_t output_zero_point) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  InlinedVector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input1));  // input 1
  input_indices.push_back(operand_indices.at(input2));  // input 2

  if (add_activation) {
    ADD_SCALAR_OPERAND(model_builder, input_indices, fuse_code);
  }

  const OperandType output_operand_type(operand_types.at(input1).type, shaper[output],
                                        output_scale, output_zero_point);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(op_type, input_indices,
                                                 {output}, {output_operand_type}));
  return Status::OK();
}

Status AddSqueezeOp(ModelBuilder& model_builder,
                    const std::string& node_name,
                    const std::string& input, const std::string& output,
                    std::vector<int32_t> axes) {
  if (model_builder.GetEffectiveFeatureLevel() < ANEURALNETWORKS_FEATURE_LEVEL_2) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, FAIL, "Squeeze is not supported on API level ", model_builder.GetEffectiveFeatureLevel());
  }

  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  const auto input_shape = shaper[input];
  auto input_dims = input_shape.size();
  for (auto& axis : axes) {
    axis = static_cast<int32_t>(HandleNegativeAxis(axis, input_dims));
  }

  // Despite the spec of ANEURALNETWORKS_SQUEEZE at
  // https://developer.android.com/ndk/reference/group/neural-networks
  // states, that the axes (input 1 of ANEURALNETWORKS_SQUEEZE) is optional.
  //
  // The actual code of NNAPI requires the axes to be provided
  // https://android.googlesource.com/platform/frameworks/ml/+/master/nn/common/operations/Squeeze.cpp#31
  if (axes.empty()) {  // Squeeze all
    for (size_t i = 0; i < input_dims; i++) {
      if (input_shape[i] == 1)
        axes.push_back(static_cast<int32_t>(i));
    }
  }

  const auto axes_name = model_builder.GetUniqueName(node_name + input + "_axes");
  Shape axes_dimen = {static_cast<uint32_t>(axes.size())};
  const OperandType axes_operand_type(Type::TENSOR_INT32, axes_dimen);
  ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(axes_name, axes.data(), axes_operand_type));

  InlinedVector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));      // input
  input_indices.push_back(operand_indices.at(axes_name));  // axes

  // Shape inference calculation for squeeze
  int32_t input_size = static_cast<int32_t>(input_shape.size());
  std::unordered_set<int32_t> axes_to_be_squeezed;

  // If the Op is squeezing all by not specifying axes, the axes is pre-populate
  // with axes of all single dimensions by the caller
  for (const auto& axis : axes)
    axes_to_be_squeezed.insert(axis);

  // Make output dimensions
  InlinedVector<uint32_t> output_dimen;
  output_dimen.reserve(input_size - axes_to_be_squeezed.size());
  for (int32_t i = 0; i < input_size; i++) {
    if (!Contains(axes_to_be_squeezed, i))
      output_dimen.push_back(input_shape[i]);
  }

  // In case of a tensor has all 1's in dimension such as {1,1,1,1} and gets squeezed all
  // the output shape will be {1}
  if (output_dimen.empty())
    output_dimen.push_back(1);

  shaper.AddShape(output, output_dimen);
  const OperandType output_operand_type(operand_types.at(input).type, output_dimen);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_SQUEEZE, input_indices,
                                                 {output}, {output_operand_type}));
  return Status::OK();
}

Status GetAxesForSqueezeAndUnSqueeze(ModelBuilder& model_builder, const NodeUnit& node_unit,
                                     std::vector<int32_t>& axes) {
  // Squeeze/Unsqueeze opset 13 uses input1 as axes
  if (node_unit.SinceVersion() > 12) {
    // For squeeze, axes is an optional input.If it is not supplied, return an empty axes as default to squeeze all
    // For unsqueeze, axes is a required input. This check has no effect for it
    if (node_unit.Inputs().size() > 1) {
      const auto& initializers(model_builder.GetInitializerTensors());
      const auto& axes_tensor = *initializers.at(node_unit.Inputs()[1].node_arg.Name());
      Initializer unpacked_tensor(axes_tensor);
      auto raw_axes = unpacked_tensor.DataAsSpan<int64_t>();
      axes = OnnxAxesToNnapi(raw_axes, std::nullopt);
    }
  } else {
    NodeAttrHelper helper(node_unit);
    const auto axes_int64 = helper.Get("axes", std::vector<int64_t>{});
    axes = OnnxAxesToNnapi(axes_int64, std::nullopt);
  }

  return Status::OK();
}

Status AddMinMaxOperator(ModelBuilder& model_builder, const NodeUnit& node_unit,
                         const std::string& input1, const std::string& input2) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  const auto& output = node_unit.Outputs()[0].node_arg.Name();

  const auto& op_type(node_unit.OpType());
  int32_t op_code;
  if (op_type == "Min")
    op_code = ANEURALNETWORKS_MINIMUM;
  else if (op_type == "Max")
    op_code = ANEURALNETWORKS_MAXIMUM;
  else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "MinMaxOpBuilder, unknown op: ", op_type);
  }

  InlinedVector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input1));  // input 1
  input_indices.push_back(operand_indices.at(input2));  // input 2

  const OperandType output_operand_type(operand_types.at(input1).type, shaper[output]);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(op_code, input_indices,
                                                 {output}, {output_operand_type}));

  return Status::OK();
}

// NOTE: Skipping Reshape results in invalid output on some SnapDragon chipsets. Whilst the NNAPI spec says the input
// to FullyConnnected can be > 2D, those chipsets don't handle this correctly.
//
// CanSkipReshape could potentially be re-enabled in the future if we no longer want to support those old chipsets.
// However, the Reshape of newer chipsets may not run on CPU so there may not be a performance issue to try and avoid,
// so CanSkipReshape could be redundant anyway.
//
// Known bad chipsets: Qualcomm Snapdragon 850, 855, 865, 870.
//
// See https://github.com/microsoft/onnxruntime/issues/19518

/*
// We can skip the Reshape if all the output edges satisfies both the following conditions
// 1. The output of the reshape/flatten is not an output of the graph
// 2. The output of the reshape/flatten is the input 0 of one or more GEMM/Matmul operators,
//    and not any other types of operator,
//    and the input rank >= 2 and output_rank == 2
//    This is because Gemm/Matmul will map to ANEURALNETWORKS_FULLY_CONNECTED in NNAPI,
//    ANEURALNETWORKS_FULLY_CONNECTED will flatten the 2+ dim input 0 to 2d
// The reason we want to skip Reshape is that Reshape is not running on Hardware (NPU,...) in NNAPI for
// some CPU (e.g. Qualcomm SD for now), skipping unnecessary Reshape will prevent context switching
// between NNAPI CPU impl and Hardware Accelerator impl and will speed up the execution
// If we are going to skip the reshape, we will still add correct shape and operand type for the output in
// onnxruntime::nnapi::Model.
static bool CanSkipReshape(const ModelBuilder& model_builder, const NodeUnit& node_unit,
                    size_t input_rank, size_t output_rank) {
  // Since we know this is a Reshape NodeUnit, so we can safely assume there is only 1 output
  // and the node_unit has only one output node.
  const auto& output_node_arg = node_unit.Outputs()[0].node_arg;
  const auto& output_name = output_node_arg.Name();

  // Check if the Reshape output is a graph output, if so we cannot skip the Reshape
  // We do not care the case where the Reshape output is a dead end
  for (const auto* node_arg : model_builder.GetGraphViewer().GetOutputs()) {
    if (node_arg == &output_node_arg) {
      LOGS_DEFAULT(VERBOSE) << "Reshape/Flatten can not be skipped when the output is a graph output"
                            << ", output name, " << output_name;
      return false;
    }
  }

  // We will go through all the output edges
  for (auto it = node_unit.OutputEdgesBegin(0), end = node_unit.OutputEdgesEnd(0); it != end; ++it) {
    const auto& dest_node_unit = model_builder.GetNodeUnit(&it->GetNode());
    const auto& op_type = dest_node_unit.OpType();
    // TODO add quantized matmul when reshape support quantized input
    if (op_type != "Gemm" && op_type != "MatMul") {
      LOGS_DEFAULT(VERBOSE) << "Reshape/Flatten can only be skipped when the output is Gemm/Matmul"
                            << " or no op is using the output (output is graph output)"
                            << ", output name, " << output_name
                            << " is used by " << op_type;
      return false;
    }

    // Now the dest node is Gemm/Matmul, we want to make sure it is supported
    OpSupportCheckParams params{
        model_builder.GetEffectiveFeatureLevel(),
        model_builder.UseNCHW(),
    };

    if (!IsNodeSupported(node_unit, model_builder.GetGraphViewer(), params)) {
      return false;
    }

    // NNAPI ANEURALNETWORKS_FULLY_CONNECTED will only flatten the input 0
    if (&output_node_arg != &dest_node_unit.Inputs()[0].node_arg) {
      LOGS_DEFAULT(VERBOSE) << "Reshape/Flatten can only be skipped when the output is input 0 of Gemm/Matmul"
                            << ", output name, " << output_name;
      return false;
    }

    // We only support 2d matmul/gemm here
    // And NNAPI ANEURALNETWORKS_FULLY_CONNECTED will only flatten input rank >= 2
    if (input_rank < 2 || output_rank != 2) {
      LOGS_DEFAULT(VERBOSE) << "Reshape/Flatten can only be skipped when input_rank >= 2 and output_rank == 2"
                            << ", output name, " << output_name
                            << ", the actual input_rank, " << input_rank
                            << ", the actual output_rank, " << output_rank;
      return false;
    }
  }

  LOGS_DEFAULT(VERBOSE) << "Skipping Reshape/Flatten node ["
                        << node_unit.Name() << "] with output, " << output_name;
  return true;
}
*/

Status AddReshapeOperator(ModelBuilder& model_builder,
                          const NodeUnit& node_unit,
                          const std::string& input,
                          const std::vector<int32_t>& shape) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& output = node_unit.Outputs()[0].node_arg.Name();

  const auto input_shape = shaper[input];
  const auto output_shape = shaper[output];

  // For reshape, the output type should be the same as the input type except the shape is different
  auto output_operand_type = operand_types.at(input);
  output_operand_type.SetDimensions(output_shape);

  /* See CanSkipReshape definition above for explanation of why this is disabled.
  // Since Reshape is not running using hardware in NNAPI for some CPU (e.g. Qualcomm SD for now)
  // We will try to see if we the skip the Reshape to prevent context switching between
  // NNAPI CPU impl and NNAPI hardware accelerator impl
  if (CanSkipReshape(model_builder, node_unit, input_rank, output_rank)) {
    const auto& operand_indices(model_builder.GetOperandIndices());
    const auto input_rank = input_shape.size();
    const auto output_rank = output_shape.size();
    // Since reshape can be skipped, only register the dimension and type, with same index and new name.
    // This essentially redirects the downstream operator builders to the input of the skipped Reshape node,
    // but with the output shape of the Reshape node.
    model_builder.RegisterOperand(output, operand_indices.at(input), output_operand_type);
  } else */
  {
    std::string shape_name = model_builder.GetUniqueName(node_unit.Name() + input + "newshape");
    ORT_RETURN_IF_ERROR(op_builder_helpers::AddNnapiReshape(model_builder, input, shape_name, shape, output));
  }

  return Status::OK();
}

bool IsQuantizationScaleSupported(const GraphViewer& graph_viewer,
                                  const NodeUnitIODef& io_def,
                                  const OpSupportCheckParams& params,
                                  const std::string& op_type,
                                  bool is_quant_matmul,
                                  bool is_conv_matmul_u8s8_weight) {
  const auto scale_name = io_def.quant_param->scale.Name();
  const auto* scale = graph_viewer.GetConstantInitializer(scale_name);
  if (!scale) {
    LOGS_DEFAULT(VERBOSE) << "The scale of " << op_type << " must be a constant initializer";
    return false;
  }

  const auto& scale_tensor = *scale;
  int64_t scales_dim = scale_tensor.dims().empty() ? 1 : scale_tensor.dims()[0];
  if (!is_conv_matmul_u8s8_weight) {
    if (scales_dim != 1) {
      LOGS_DEFAULT(VERBOSE) << op_type << " does not support per-channel quantization, "
                            << " for now, only u8s8 QlinearConv supports per-channel quantization on API 29+";
      return false;
    }
  } else if (scales_dim != 1) {
    // For u8s8 Qlinear[Conv/MatMul], we support
    // 1. Per-tensor, the weight will be transformed to uint8 later
    // 2. Per-channel, only from Android API level 29
    if (is_quant_matmul) {
      LOGS_DEFAULT(VERBOSE) << "QLinearMatMul does not support per-channel quantization";
      return false;
    }

    if (params.android_feature_level < ANEURALNETWORKS_FEATURE_LEVEL_3) {
      LOGS_DEFAULT(VERBOSE) << op_type << " only supports per-channel quantization on Android API 29+, "
                            << "system NNAPI feature level: " << params.android_feature_level;
      return false;
    }

    Shape weight_shape;
    if (!GetShape(io_def.node_arg, weight_shape))
      return false;

    if (weight_shape[0] != scales_dim) {
      LOGS_DEFAULT(VERBOSE) << op_type << " mismatch int8 per-channel quantization weight,"
                            << " weight dimension[0] " << weight_shape[0]
                            << " scale dimension " << scales_dim;
      return false;
    }
  }

  return true;
}

bool IsQuantizationZeroPointSupported(const GraphViewer& graph_viewer,
                                      const NodeUnitIODef& io_def,
                                      const std::string& op_type,
                                      const Path& model_path,
                                      bool is_quant_matmul,
                                      bool is_conv_matmul_u8s8_weight) {
  // zero point is optional here
  if (!io_def.quant_param->zero_point)
    return true;

  const auto& zero_point_name = io_def.quant_param->zero_point->Name();
  const auto* zero_point = graph_viewer.GetConstantInitializer(zero_point_name);
  if (!zero_point) {
    LOGS_DEFAULT(VERBOSE) << "The zero point of " << op_type << " must be a constant initializer";
    return false;
  }

  const auto& zero_tensor = *zero_point;
  int64_t zero_dim = zero_tensor.dims().empty() ? 1 : zero_tensor.dims()[0];

  if (!is_conv_matmul_u8s8_weight) {
    if (zero_dim != 1) {
      LOGS_DEFAULT(VERBOSE) << op_type << " does not support per-channel quantization, "
                            << " for now, only u8s8 QlinearConv supports per-channel quantization on API 29+";
      return false;
    }
  } else {
    // For u8s8 Qlinear[Conv/MatMul], we support
    // 1. Per-tensor, the weight will be transformed to uint8 later
    // 2. Per-channel, only from Android API level 29
    if (zero_tensor.data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT8) {
      LOGS_DEFAULT(VERBOSE) << "u8s8 Qlinear[Conv/MatMul] only supports int8 zero point for weight, "
                            << "actual zero point type: [" << zero_tensor.data_type() << "]";
      return false;
    }

    if (zero_dim != 1) {
      if (is_quant_matmul) {
        LOGS_DEFAULT(VERBOSE) << "QLinearMatMul does not support per-channel quantization";
        return false;
      }
    }

    // For onnx, u8s8 QlinearConv, the weight zero point can be a scalar,
    // or a tensor with same channel as weight, for NNAPI we only support it be
    // 0 (scalar) or all 0 (tensor), NNAPI will assume the zero point for per-channel
    // quantization is 0 there is no input for it
    Shape weight_shape;
    if (!GetShape(io_def.node_arg, weight_shape))
      return false;

    if (weight_shape[0] != zero_dim && zero_dim != 1) {
      LOGS_DEFAULT(VERBOSE) << op_type << " mismatch int8 per-channel quantization weight,"
                            << " weight dimension[0] " << weight_shape[0]
                            << " zero point dimension " << zero_dim;
      return false;
    }
    Initializer unpacked_tensor(zero_tensor, model_path);
    // Verify all onnx weight zero point(s) are 0(s)
    auto zero_points = unpacked_tensor.DataAsSpan<int8_t>();
    for (size_t i = 0; i < unpacked_tensor.size(); i++) {
      if (zero_points[i] != 0) {
        LOGS_DEFAULT(VERBOSE) << "u8s8 Qlinear[Conv/MatMul]  only support 0 as zero point, "
                              << "zero_points[" << i << "] has value: " << zero_points[i];
        return false;
      }
    }
  }

  return true;
}

bool IsQuantizedIOSupported(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                            const std::vector<size_t>& indices, const OpSupportCheckParams& params,
                            ArgType arg_type) {
  const auto& op_type = node_unit.OpType();
  auto quant_op_type = GetQuantizedOpType(node_unit);

  ORT_ENFORCE(quant_op_type != QuantizedOpType::Unknown, "[", op_type, "] is not a quantized op");

  const bool is_input = arg_type == ArgType::kInput;
  const bool is_quant_conv = IsQuantizedConv(quant_op_type);
  const bool is_quant_matmul = (quant_op_type == QuantizedOpType::QLinearMatMul) || (quant_op_type == QuantizedOpType::QDQMatMul);
  const bool is_quant_gemm = (quant_op_type == QuantizedOpType::QDQGemm);
  const bool is_quant_matmul_or_gemm = is_quant_matmul || is_quant_gemm;
  const auto& io_defs = is_input ? node_unit.Inputs() : node_unit.Outputs();

  for (const auto idx : indices) {
    if (idx >= io_defs.size()) {
      LOGS_DEFAULT(VERBOSE) << (is_input ? "Input" : "Output") << " index,  " << idx
                            << " >= size, " << io_defs.size()
                            << " of NodeUnit: " << node_unit.Name();
      return false;
    }

    const auto& io_def = io_defs[idx];
    ORT_ENFORCE(io_def.quant_param.has_value(), "Input index,  ", idx, " has no quant_param");

    // If this op is Qlinear[Conv/MatMul], we want to check u8s8 support for weight tensor (or B tensor for QlinearMatMul)
    const bool is_conv_matmul_weight = is_input && (is_quant_conv || is_quant_matmul_or_gemm) && idx == 1;
    bool is_conv_matmul_u8s8_weight = false;

    if (is_conv_matmul_weight) {
      int32_t weight_type;
      if (!GetType(io_def.node_arg, weight_type))
        return false;
      is_conv_matmul_u8s8_weight = weight_type == ONNX_NAMESPACE::TensorProto_DataType_INT8;
    }

    int32_t input_type;
    if (!GetType(io_def.node_arg, input_type))
      return false;

    // We only support u8 for most of the inputs and all outputs, with the exception for Quantized MatMul and Conv,
    // which allows s8 weight (u8s8)
    // TODO, add support of s8s8
    if (input_type != ONNX_NAMESPACE::TensorProto_DataType_UINT8 &&
        !(input_type == ONNX_NAMESPACE::TensorProto_DataType_INT8 && is_conv_matmul_u8s8_weight)) {
      LOGS_DEFAULT(VERBOSE) << op_type << "NodeUnit [" << node_unit.Name()
                            << "], type [" << op_type << "]'s "
                            << (is_input ? "Input" : "Output") << " index  [" << idx
                            << "] has unsupported type [" << input_type << "]";
      return false;
    }

    // Check scale and zero point
    if (!IsQuantizationScaleSupported(graph_viewer, io_def, params, op_type,
                                      is_quant_matmul, is_conv_matmul_u8s8_weight)) {
      return false;
    }

    if (!IsQuantizationZeroPointSupported(graph_viewer, io_def, op_type, node_unit.ModelPath(),
                                          is_quant_matmul, is_conv_matmul_u8s8_weight)) {
      return false;
    }
  }

  return true;
}

bool HasRequiredScaleAndZeroPoint(const GraphViewer& graph_viewer,
                                  const std::string& op_desc,
                                  const NodeUnitIODef& io_def,
                                  const Path& path,
                                  float required_scale, int32_t required_zp) {
  float scale = 0.0f;
  int32_t zp = 0;
  auto status = GetQuantizationScaleAndZeroPoint(graph_viewer, io_def, path, scale, zp);
  if (!status.IsOK()) {
    LOGS_DEFAULT(ERROR) << op_desc << " GetQuantizationScaleAndZeroPoint failed, message: " << status.ErrorMessage();
    return false;
  }

  if (scale != required_scale) {
    LOGS_DEFAULT(VERBOSE) << op_desc << " scale can only be [" << required_scale << "], actual scale: " << scale;
    return false;
  }

  if (zp != required_zp) {
    LOGS_DEFAULT(VERBOSE) << op_desc << "] zero point can only be [" << required_zp << "], actual zero point: "
                          << zp;
    return false;
  }

  return true;
}

Status PerformBroadcasting(const Shape& shape1, const Shape& shape2, Shape& output_shape) {
  bool shape1_is_bigger = shape1.size() >= shape2.size();
  auto max_shape = shape1_is_bigger ? shape1 : shape2;
  const auto& min_shape = shape1_is_bigger ? shape2 : shape1;
  for (int i = narrow<int>(max_shape.size()) - 1,
           j = narrow<int>(min_shape.size()) - 1;
       i >= 0 && j >= 0;
       i--, j--) {
    int dim_max_shape = max_shape[i];
    int dim_min_shape = min_shape[j];
    if (dim_max_shape != dim_min_shape) {
      ORT_RETURN_IF_NOT(dim_max_shape == 1 || dim_min_shape == 1,
                        "Dimensions are not compatible, dim1: ", dim_max_shape,
                        "dim2: ", dim_min_shape);
      if (dim_max_shape == 1) {
        max_shape[i] = dim_min_shape;
      }
    }
  }
  output_shape = std::move(max_shape);
  return Status::OK();
}

}  // namespace onnxruntime::nnapi::op_builder_helpers
