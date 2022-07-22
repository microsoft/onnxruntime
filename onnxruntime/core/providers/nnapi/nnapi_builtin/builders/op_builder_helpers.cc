// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nnapi/nnapi_builtin/builders/op_builder_helpers.h"

#include <algorithm>

#include "gsl/gsl"

#include "core/common/safeint.h"
#include "core/graph/node_arg.h"
#include "core/providers/common.h"
#include "core/providers/nnapi/nnapi_builtin/builders/helper.h"
#include "core/providers/nnapi/nnapi_builtin/nnapi_lib/NeuralNetworksWrapper.h"

namespace onnxruntime::nnapi::op_builder_helpers {

using android::nn::wrapper::OperandType, android::nn::wrapper::Type;

Status AddNnapiTranspose(ModelBuilder& model_builder,
                         const std::string& data_input,
                         const std::string& perm_input, const std::vector<int32_t>& perm,
                         const std::string& output) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(data_input));  // input

  Shape perm_dimen = {SafeInt<uint32_t>(perm.size())};
  OperandType perm_operand_type(Type::TENSOR_INT32, perm_dimen);
  ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(perm_input, perm.data(), perm_operand_type));
  uint32_t perm_idx = operand_indices.at(perm_input);

  input_indices.push_back(perm_idx);  // permutation
  ORT_RETURN_IF_ERROR(shaper.Transpose(data_input, perm, output));
  OperandType output_operand_type = operand_types.at(data_input);
  output_operand_type.SetDimensions(shaper[output]);
  return model_builder.AddOperation(ANEURALNETWORKS_TRANSPOSE, input_indices, {output},
                                    {output_operand_type});
}

Status AddNnapiReshape(ModelBuilder& model_builder,
                       const std::string& data_input,
                       const std::string& shape_input, const std::vector<int32_t>& shape_value,
                       const std::string& output, const Shape* output_shape) {
  if (output_shape == nullptr) {
    auto& shaper = model_builder.GetShaper();
    ORT_RETURN_IF_ERROR(shaper.Reshape(data_input, shape_value, output));
    output_shape = &shaper[output];
  }

  const auto& operand_indices = model_builder.GetOperandIndices();
  const auto& operand_types = model_builder.GetOperandTypes();

  // Add input
  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(data_input));

  // Add new shape
  const Shape shape_dimen{static_cast<uint32_t>(shape_value.size())};
  const OperandType shape_operand_type{Type::TENSOR_INT32, shape_dimen};
  ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(shape_input, shape_value.data(),
                                                                      shape_operand_type));
  input_indices.push_back(operand_indices.at(shape_input));

  // For reshape, the output type should be the same as the input type except the shape is different
  OperandType output_operand_type{operand_types.at(data_input)};
  output_operand_type.SetDimensions(*output_shape);
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

  ORT_RETURN_IF_ERROR(shaper.Split(input, axis, outputs));

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  ORT_RETURN_IF_ERROR(AddScalarOperand(model_builder, input_indices, axis));
  const auto count = gsl::narrow<int32_t>(outputs.size());
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
                   [](uint32_t d) { return gsl::narrow<int32_t>(d); });
    ORT_RETURN_IF_ERROR(AddNnapiReshape(model_builder, input, new_shape_name, new_shape_i32, output, nullptr));
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
    ORT_RETURN_IF_ERROR(AddNnapiTranspose(model_builder, gemm_b_inputs.front(), b_new_perm, {0, 2, 1}, b_transposed));
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
      std::vector<uint32_t> input_indices;
      input_indices.push_back(operand_indices.at(a));             // A
      input_indices.push_back(operand_indices.at(b_transposed));  // B'
      input_indices.push_back(operand_indices.at(bias));          // C
      int32_t fuse_code = ANEURALNETWORKS_FUSED_NONE;
      ORT_RETURN_IF_ERROR(AddScalarOperand(model_builder, input_indices, fuse_code));

      ORT_RETURN_IF_ERROR(shaper.FC(a, b_transposed, output));
      const OperandType output_operand_type(operand_types.at(a).type, shaper[output]);
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
      std::vector<uint32_t> input_indices;
      input_indices.reserve(inputs.size() + 1);
      std::transform(inputs.begin(), inputs.end(), std::back_inserter(input_indices),
                     [&operand_indices](const std::string& input) { return operand_indices.at(input); });
      const int32_t axis = 0;
      ORT_RETURN_IF_ERROR(AddScalarOperand(model_builder, input_indices, axis));
      ORT_RETURN_IF_ERROR(shaper.Concat(inputs, axis, output));
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

}  // namespace onnxruntime::nnapi::op_builder_helpers
