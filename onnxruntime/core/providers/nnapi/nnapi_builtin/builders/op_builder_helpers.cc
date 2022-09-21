// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nnapi/nnapi_builtin/builders/op_builder_helpers.h"

#include <algorithm>
#include <optional>

#include "gsl/gsl"

#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/node_arg.h"
#include "core/graph/graph_viewer.h"
#include "core/providers/common.h"
#include "core/providers/nnapi/nnapi_builtin/builders/helper.h"

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
  std::vector<uint8_t> unpacked_tensor;

  switch (tensor.data_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
    case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
      ORT_RETURN_IF_ERROR(
          onnxruntime::utils::UnpackInitializerData(tensor, model_builder.GetGraphViewer().ModelPath(),
                                                    unpacked_tensor));
      src = unpacked_tensor.data();
      break;
    }
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "The initializer of graph ", name,
                             " doesn't have valid type: ", tensor.data_type());
  }

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
  std::vector<uint8_t> unpacked_tensor;
  switch (tensor.data_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
    case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
      ORT_RETURN_IF_ERROR(
          onnxruntime::utils::UnpackInitializerData(tensor, model_builder.GetGraphViewer().ModelPath(),
                                                    unpacked_tensor));
      src = unpacked_tensor.data();
      break;
    }
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "The initializer of graph ", name,
                             " doesn't have valid type: ", tensor.data_type());
  }

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

Status GetBinaryOpQuantizationScaleAndZeroPoint(
    const InitializedTensorSet& initializers, const NodeUnit& node_unit,
    float& a_scale, float& b_scale, float& y_scale,
    int32_t& a_zero_point, int32_t& b_zero_point, int32_t& y_zero_point) {
  ORT_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(
      initializers, node_unit.Inputs()[0], node_unit.ModelPath(), a_scale, a_zero_point));
  ORT_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(
      initializers, node_unit.Inputs()[1], node_unit.ModelPath(), b_scale, b_zero_point));
  ORT_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(
      initializers, node_unit.Outputs()[0], node_unit.ModelPath(), y_scale, y_zero_point));

  return Status::OK();
}

Status GetConvMatMulOpQuantizationScaleAndZeroPoint(
    const ModelBuilder& model_builder, const NodeUnit& node_unit,
    float& a_scale, float& w_scale, float& y_scale,
    int32_t& a_zero_point, int32_t& w_zero_point, int32_t& y_zero_point,
    std::optional<std::vector<float>>& w_scales, bool& is_per_tensor_u8s8) {
  is_per_tensor_u8s8 = false;
  const auto& initializers(model_builder.GetInitializerTensors());
  // Get scale and zero points
  // We will handle per-channel weight scale and zero point later
  ORT_RETURN_IF_ERROR(
      GetBinaryOpQuantizationScaleAndZeroPoint(initializers, node_unit,
                                               a_scale, w_scale, y_scale,
                                               a_zero_point, w_zero_point, y_zero_point));

  const auto& inputs = node_unit.Inputs();
  const auto& weight_tensor = *initializers.at(inputs[1].node_arg.Name());

  // We are done here if this is u8u8 QLinearConv
  if (weight_tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_UINT8)
    return Status::OK();

  // This is per-tensor u8s8
  // NNAPI does not support per-tensor u8s8
  // For this case we will need to convert the int8 weight tensor to uint8
  // And have same scale and 128 as zero point
  // The conversion of the weight tensor itself will be done in the OpBuilder
  const auto& scale_tensor = *initializers.at(inputs[1].quant_param->scale.Name());
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
  std::vector<uint8_t> unpacked_tensor;
  ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(scale_tensor, unpacked_tensor));
  const float* scales = reinterpret_cast<const float*>(unpacked_tensor.data());
  const size_t scales_size = scale_tensor.dims().empty() ? 1 : scale_tensor.dims()[0];
  std::vector<float> scales_vec(scales, scales + scales_size);
  w_scales = std::make_optional(std::move(scales_vec));
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

Status IsOpInRequiredLayout(bool use_nchw, const NodeUnit& node_unit) {
  bool is_op_nhwc = node_unit.Domain() == kMSInternalNHWCDomain;
  if (is_op_nhwc && use_nchw) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Expected layout and operator layout do not match. Possible bug in layout optimizer.");
  }

  return Status::OK();
}

}  // namespace onnxruntime::nnapi::op_builder_helpers
