// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "op_builder.h"

#include <onnx/onnx_pb.h>

#include "core/common/logging/logging.h"
#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_viewer.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/shared/node_unit/node_unit.h"
#include "core/providers/cpu/tensor/slice_helper.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_builder_helpers.h"
#include "helper.h"
#include "model_builder.h"
#include "op_support_checker.h"
#include "core/providers/nnapi/nnapi_builtin/builders/impl/base_op_builder.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_builder_factory.h"

using namespace android::nn::wrapper;

namespace onnxruntime {
namespace nnapi {

using namespace op_builder_helpers;

#pragma region helpers

static Status AddBinaryOperator(int32_t op_type,
                                ModelBuilder& model_builder,
                                const std::string& input1,
                                const std::string& input2,
                                bool add_activation,
                                int32_t fuse_code,
                                const std::string& output,
                                float output_scale = 0.0f,
                                int32_t output_zero_point = 0) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input1));  // input 1
  input_indices.push_back(operand_indices.at(input2));  // input 2

  if (add_activation) {
    ADD_SCALAR_OPERAND(model_builder, input_indices, fuse_code);
  }

  ORT_RETURN_IF_ERROR(shaper.Eltwise(input1, input2, output));
  const OperandType output_operand_type(operand_types.at(input1).type, shaper[output],
                                        output_scale, output_zero_point);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(op_type, input_indices,
                                                 {output}, {output_operand_type}));
  return Status::OK();
}

static Status AddSqueezeOp(ModelBuilder& model_builder,
                           const std::string& node_name,
                           const std::string& input, const std::string& output,
                           std::vector<int32_t> axes) {
  if (model_builder.GetNNAPIFeatureLevel() < ANEURALNETWORKS_FEATURE_LEVEL_2) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, FAIL, "Squeeze is not supported on API level ", model_builder.GetNNAPIFeatureLevel());
  }

  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  const auto& input_shape(shaper[input]);
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

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));      // input
  input_indices.push_back(operand_indices.at(axes_name));  // axes

  ORT_RETURN_IF_ERROR(shaper.Squeeze(input, axes, output));
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_SQUEEZE, input_indices,
                                                 {output}, {output_operand_type}));
  return Status::OK();
}

static Status GetAxesForSqueezeAndUnSqueeze(ModelBuilder& model_builder, const NodeUnit& node_unit,
                                            std::vector<int32_t>& axes) {
  // Squeeze/Unsqueeze opset 13 uses input1 as axes
  if (node_unit.SinceVersion() > 12) {
    // For squeeze, axes is an optional input.If it is not supplied, return an empty axes as default to squeeze all
    // For unsqueeze, axes is a required input. This check has no effect for it
    // TODO: Add helper function to handle the following conversion from int64 initializer to int32
    if (node_unit.Inputs().size() > 1) {
      const auto& initializers(model_builder.GetInitializerTensors());
      const auto& axes_tensor = *initializers.at(node_unit.Inputs()[1].node_arg.Name());
      std::vector<uint8_t> unpacked_tensor;
      ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(axes_tensor, unpacked_tensor));
      const int64_t* raw_axes = reinterpret_cast<const int64_t*>(unpacked_tensor.data());
      const auto size = SafeInt<uint32_t>(axes_tensor.dims()[0]);
      axes.resize(size);
      for (uint32_t i = 0; i < size; i++) {
        // it is unlikely we have an axis value overflow for int32
        axes[i] = static_cast<int32_t>(raw_axes[i]);
      }
    }
  } else {
    NodeAttrHelper helper(node_unit);
    axes = helper.Get("axes", std::vector<int32_t>());
  }

  return Status::OK();
}

#pragma endregion helpers

#pragma region op_binary

class BinaryOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
  static void CreateSharedOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);

 private:
  bool IsQuantizedOp(const NodeUnit& node_unit) const override;
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

void CreateBinaryOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<BinaryOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

bool BinaryOpBuilder::IsQuantizedOp(const NodeUnit& node_unit) const {
  const auto quant_type = GetQuantizedOpType(node_unit);
  return quant_type == QuantizedOpType::QLinearAdd ||
         quant_type == QuantizedOpType::QLinearMul ||
         quant_type == QuantizedOpType::QDQAdd ||
         quant_type == QuantizedOpType::QDQMul;
}

void BinaryOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  if (!IsQuantizedOp(node_unit))
    return;

  const auto& inputs = node_unit.Inputs();
  AddQuantizationScaleAndZeroPointToSkip(model_builder, *inputs[0].quant_param);               // a_scale, a_zp
  AddQuantizationScaleAndZeroPointToSkip(model_builder, *inputs[1].quant_param);               // b_scale, b_zp
  AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Outputs()[0].quant_param);  // y_scale, y_zp
}

/* static */ void BinaryOpBuilder::CreateSharedOpBuilder(
    const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  CreateSharedOpBuilderImpl<BinaryOpBuilder>(
      op_type, op_registrations,
      {
          "Add",
          "Sub",
          "Mul",
          "Div",
          "QLinearAdd",
          "QLinearMul",
          "Pow",
          "PRelu",
      });
}

Status BinaryOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  const auto& op_type(node_unit.OpType());
  const auto& inputs = node_unit.Inputs();

  int32_t op_code;
  bool add_activation = true;
  bool is_quant_op = IsQuantizedOp(node_unit);
  if (op_type == "Add" || op_type == "QLinearAdd") {  // Add/QLinearAdd/QDQAdd
    op_code = ANEURALNETWORKS_ADD;
  } else if (op_type == "Sub") {
    op_code = ANEURALNETWORKS_SUB;
  } else if (op_type == "Mul" || op_type == "QLinearMul") {  // Mul/QLinearMul/QDQMul
    op_code = ANEURALNETWORKS_MUL;
  } else if (op_type == "Div") {
    op_code = ANEURALNETWORKS_DIV;
  } else if (op_type == "Pow") {
    add_activation = false;  // ANEURALNETWORKS_POW does not have activation
    op_code = ANEURALNETWORKS_POW;
  } else if (op_type == "PRelu") {
    add_activation = false;  // ANEURALNETWORKS_PRELU does not have activation
    op_code = ANEURALNETWORKS_PRELU;
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "UnaryOpBuilder, unknown op: ", op_type);
  }

  std::string input1 = inputs[0].node_arg.Name();
  std::string input2 = inputs[1].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();

  float a_scale = 0.0f,
        b_scale = 0.0f,
        y_scale = 0.0f;
  int32_t a_zero_point = 0,
          b_zero_point = 0,
          y_zero_point = 0;

  if (is_quant_op) {
    ORT_RETURN_IF_ERROR(GetBinaryOpQuantizationScaleAndZeroPoint(
        model_builder.GetInitializerTensors(), node_unit,
        a_scale, b_scale, y_scale,
        a_zero_point, b_zero_point, y_zero_point));
  }

  // Verify if the scale and zero point matchs from onnx input and nnapi input match
  if (is_quant_op) {
    ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input1, a_scale, a_zero_point));
    ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input2, b_scale, b_zero_point));
  }

  int32_t fuse_code = ANEURALNETWORKS_FUSED_NONE;
  if (add_activation) {
    fuse_code = model_builder.FindActivation(node_unit);
  }

  return AddBinaryOperator(op_code, model_builder,
                           input1, input2,
                           add_activation, fuse_code,
                           output, y_scale, y_zero_point);
}

#pragma endregion

#pragma region op_relu

class ReluOpBuilder : public BaseOpBuilder {
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

void CreateReluOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ReluOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

Status ReluOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  const auto& input = node_unit.Inputs()[0].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();
  ORT_RETURN_IF_ERROR(shaper.Identity(input, output));
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);

  // skip this relu if it is some op's fuse output
  if (Contains(model_builder.GetFusedActivations(), input)) {
    LOGS_DEFAULT(VERBOSE) << "Relu Node [" << node_unit.Name() << "] fused";
    model_builder.RegisterOperand(output, operand_indices.at(input), output_operand_type);
  } else {
    std::vector<uint32_t> input_indices;
    input_indices.push_back(operand_indices.at(input));
    ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_RELU, input_indices,
                                                   {output}, {output_operand_type}));
  }

  return Status::OK();
}

#pragma endregion op_relu

#pragma region op_transpose

class TransposeOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
  bool IsQuantizedOp(const NodeUnit& node_unit) const override;
};

void CreateTransposeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<TransposeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

void TransposeOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  if (!IsQuantizedOp(node_unit))
    return;

  AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Inputs()[0].quant_param);   // x_scale, x_zp
  AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Outputs()[0].quant_param);  // y_scale, y_zp
}

bool TransposeOpBuilder::IsQuantizedOp(const NodeUnit& node_unit) const {
  return GetQuantizedOpType(node_unit) == QuantizedOpType::QDQTranspose;
}

Status TransposeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& initializers(model_builder.GetInitializerTensors());

  const auto& input = node_unit.Inputs()[0].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();
  NodeAttrHelper helper(node_unit);
  std::vector<int32_t> perm = helper.Get("perm", std::vector<int32_t>());
  auto input_dims = static_cast<int32_t>(shaper[input].size());
  if (perm.empty()) {
    for (int32_t i = input_dims - 1; i >= 0; i--)
      perm.push_back(i);
  } else {
    ORT_RETURN_IF_NOT(static_cast<int32_t>(perm.size()) == input_dims, "Perm and input should have same dimension");
  }

  // Check if the quantization scale and ZP are correct
  if (IsQuantizedOp(node_unit)) {
    float x_scale = 0.0f;
    int32_t x_zero_point = 0;
    ORT_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(
        initializers, node_unit.Inputs()[0], node_unit.ModelPath(), x_scale, x_zero_point));
    ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input, x_scale, x_zero_point));
  }

  std::string perm_name = model_builder.GetUniqueName(node_unit.Name() + input + "perm");

  ORT_RETURN_IF_ERROR(op_builder_helpers::AddNnapiTranspose(model_builder, input, perm_name, perm, output));

  return Status::OK();
}

#pragma endregion op_transpose

#pragma region op_reshape

class ReshapeOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
  static Status AddReshapeOperator(ModelBuilder& model_builder, const NodeUnit& node_unit,
                                   const std::string& input, const std::vector<int32_t>& shape);

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
  static bool CanSkipReshape(const ModelBuilder& model_builder, const NodeUnit& node_unit,
                             size_t input_rank, size_t output_rank);
  bool IsQuantizedOp(const NodeUnit& node_unit) const override;
};

void CreateReshapeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ReshapeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

void ReshapeOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  if (IsQuantizedOp(node_unit)) {
    AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Inputs()[0].quant_param);   // x_scale, x_zp
    AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Outputs()[0].quant_param);  // y_scale, y_zp
  }
  model_builder.AddInitializerToSkip(node_unit.Inputs()[1].node_arg.Name());
}

bool ReshapeOpBuilder::IsQuantizedOp(const NodeUnit& node_unit) const {
  return GetQuantizedOpType(node_unit) == QuantizedOpType::QDQReshape;
}

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
/* static */ bool ReshapeOpBuilder::CanSkipReshape(const ModelBuilder& model_builder, const NodeUnit& node_unit,
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
    if (!BaseOpBuilder::IsOpSupported(model_builder, dest_node_unit)) {
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

/* static */ Status ReshapeOpBuilder::AddReshapeOperator(ModelBuilder& model_builder,
                                                         const NodeUnit& node_unit,
                                                         const std::string& input,
                                                         const std::vector<int32_t>& shape) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& output = node_unit.Outputs()[0].node_arg.Name();
  ORT_RETURN_IF_ERROR(shaper.Reshape(input, shape, output));
  auto input_rank = shaper[input].size();
  auto output_rank = shaper[output].size();

  // For reshape, the output type should be the same as the input type except the shape is different
  auto output_operand_type = operand_types.at(input);
  output_operand_type.SetDimensions(shaper[output]);

  // Since Reshape is not running using hardware in NNAPI for some CPU (e.g. Qualcomm SD for now)
  // We will try to see if we the skip the Reshape to prevent context switching between
  // NNAPI CPU impl and NNAPI hardware accelerator impl
  if (CanSkipReshape(model_builder, node_unit, input_rank, output_rank)) {
    // Since reshape can be skipped, only register the dimension and type, with same index and new name
    model_builder.RegisterOperand(output, operand_indices.at(input), output_operand_type);
  } else {
    // We still need to perform a reshape here
    std::string shape_name = model_builder.GetUniqueName(node_unit.Name() + input + "newshape");
    ORT_RETURN_IF_ERROR(op_builder_helpers::AddNnapiReshape(model_builder, input, shape_name, shape, output,
                                                            &shaper[output]));
  }

  return Status::OK();
}

Status ReshapeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& initializers(model_builder.GetInitializerTensors());

  auto input = node_unit.Inputs()[0].node_arg.Name();

  const auto& shape_tensor = *initializers.at(node_unit.Inputs()[1].node_arg.Name());
  std::vector<uint8_t> unpacked_tensor;
  ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(shape_tensor, unpacked_tensor));
  const int64_t* raw_shape = reinterpret_cast<const int64_t*>(unpacked_tensor.data());
  const auto size = SafeInt<uint32_t>(shape_tensor.dims()[0]);

  Shape input_shape = shaper[input];
  std::vector<int32_t> shape(size);
  for (uint32_t i = 0; i < size; i++) {
    int32_t dim = SafeInt<int32_t>(raw_shape[i]);
    // NNAPI reshape does not support 0 as dimension
    shape[i] = dim == 0 ? input_shape[i] : dim;
  }

  // Check if the quantization scale and ZP are correct
  float x_scale = 0.0f;
  int32_t x_zero_point = 0;
  if (IsQuantizedOp(node_unit)) {
    ORT_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(
        initializers, node_unit.Inputs()[0], node_unit.ModelPath(), x_scale, x_zero_point));
    ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input, x_scale, x_zero_point));
  }

  return AddReshapeOperator(model_builder, node_unit, input, shape);
}

#pragma endregion op_reshape

#pragma region op_unsqueeze

class UnsqueezeOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

void CreateUnsqueezeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<UnsqueezeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

void UnsqueezeOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  // Unsqueeze opset 13 uses input 1 as axes, add it to initializer skip list
  if (node_unit.SinceVersion() > 12 && node_unit.Inputs().size() > 1) {
    model_builder.AddInitializerToSkip(node_unit.Inputs()[1].node_arg.Name());  // "axes"
  }
}

Status UnsqueezeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  const auto& shaper(model_builder.GetShaper());
  const auto& input = node_unit.Inputs()[0].node_arg.Name();

  // NNAPI does not support unsqueeze, here we utilize unsqueeze's axes input to compute output shape
  // And add equivalent operation as ANEURALNETWORKS_RESHAPE to nnapi model
  std::vector<int32_t> axes;
  ORT_RETURN_IF_ERROR(GetAxesForSqueezeAndUnSqueeze(model_builder, node_unit, axes));

  Shape input_shape = shaper[input];
  auto input_dims = input_shape.size();
  std::vector<int32_t> shape;
  const auto size = SafeInt<uint32_t>(input_dims + axes.size());  // "output rank"
  shape.reserve(size);
  for (auto& axis : axes) {
    axis = static_cast<int32_t>(HandleNegativeAxis(axis, size));
  }
  std::sort(axes.begin(), axes.end());
  std::copy(input_shape.cbegin(), input_shape.cend(), std::back_inserter(shape));
  for (size_t i = 0; i < axes.size(); i++) {
    auto iter = shape.cbegin() + axes[i];
    shape.insert(iter, SafeInt<int32_t>(1));
  }

  return ReshapeOpBuilder::AddReshapeOperator(model_builder, node_unit, input, shape);
}

#pragma endregion op_unsqueeze

#pragma region op_batchnormalization

class BatchNormalizationOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

void CreateBatchNormalizationOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<BatchNormalizationOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

void BatchNormalizationOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  // skip everything except input0 for BatchNormalization
  model_builder.AddInitializerToSkip(node_unit.Inputs()[1].node_arg.Name());  // scale
  model_builder.AddInitializerToSkip(node_unit.Inputs()[2].node_arg.Name());  // B
  model_builder.AddInitializerToSkip(node_unit.Inputs()[3].node_arg.Name());  // mean
  model_builder.AddInitializerToSkip(node_unit.Inputs()[4].node_arg.Name());  // var
}

Status BatchNormalizationOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& initializers(model_builder.GetInitializerTensors());
  NodeAttrHelper helper(node_unit);
  const auto& inputs = node_unit.Inputs();

  // For reshape we are not really doing anything but
  // register a new operand with new shape
  const auto& input = inputs[0].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();

  const auto& scale_tensor = *initializers.at(inputs[1].node_arg.Name());
  const auto& bias_tensor = *initializers.at(inputs[2].node_arg.Name());
  const auto& mean_tensor = *initializers.at(inputs[3].node_arg.Name());
  const auto& var_tensor = *initializers.at(inputs[4].node_arg.Name());
  const auto eps = helper.Get("epsilon", 1e-5f);

  const auto size = SafeInt<uint32_t>(scale_tensor.dims()[0]);
  std::vector<float> a, b;
  a.reserve(size);
  b.reserve(size);

  std::vector<uint8_t> unpacked_scale_tensor;
  ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(scale_tensor, unpacked_scale_tensor));
  const float* scale_data = reinterpret_cast<const float*>(unpacked_scale_tensor.data());

  std::vector<uint8_t> unpacked_bias_tensor;
  ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(bias_tensor, unpacked_bias_tensor));
  const float* bias_data = reinterpret_cast<const float*>(unpacked_bias_tensor.data());

  std::vector<uint8_t> unpacked_mean_tensor;
  ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(mean_tensor, unpacked_mean_tensor));
  const float* mean_data = reinterpret_cast<const float*>(unpacked_mean_tensor.data());

  std::vector<uint8_t> unpacked_var_tensor;
  ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(var_tensor, unpacked_var_tensor));
  const float* var_data = reinterpret_cast<const float*>(unpacked_var_tensor.data());

  for (int64_t i = 0; i < size; i++) {
    a.push_back(scale_data[i] / sqrt(var_data[i] + eps));
    b.push_back((scale_data[i] * -mean_data[i]) / sqrt(var_data[i] + eps) +
                bias_data[i]);
  }

  const auto tensor_a_name = model_builder.GetUniqueName(node_unit.Name() + input + "_imm_a");
  const auto tensor_b_name = model_builder.GetUniqueName(node_unit.Name() + input + "_imm_b");
  const auto tensor_imm_product_name = model_builder.GetUniqueName(node_unit.Name() + input + "_imm_mul");
  Shape tensor_a_dimen = {size};

  bool use_nchw = model_builder.UseNCHW();
  ORT_RETURN_IF_ERROR(IsOpInRequiredLayout(use_nchw, node_unit));

  if (use_nchw) {
    // the batch normalization is applied on C channel,
    // if the input is NC[HW], will need correct shape for tensor_a/b
    // to make sure we are broadcasting on the correct channel,
    // input shape {N, C}       ==> tensor_a/b's shape {size}
    // input shape {N, C, H}    ==> tensor_a/b's shape {size, 1}
    // input shape {N, C, H, W} ==> tensor_a/b's shape {size, 1, 1}
    const auto input_rank = shaper[input].size();
    for (size_t i = 2; i < input_rank; i++)
      tensor_a_dimen.push_back(1);
  }

  shaper.AddShape(tensor_a_name, tensor_a_dimen);
  shaper.AddShape(tensor_b_name, tensor_a_dimen);
  const OperandType a_operand_type(operand_types.at(input).type, tensor_a_dimen);
  ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(tensor_a_name, a.data(), a_operand_type));
  const OperandType b_operand_type(operand_types.at(input).type, tensor_a_dimen);
  ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(tensor_b_name, b.data(), b_operand_type));

  // Mul
  ORT_RETURN_IF_ERROR(AddBinaryOperator(ANEURALNETWORKS_MUL,
                                        model_builder,
                                        input, tensor_a_name,
                                        true /* add_activation */, ANEURALNETWORKS_FUSED_NONE,
                                        tensor_imm_product_name));

  // Add
  int32_t fuse_code = model_builder.FindActivation(node_unit);
  ORT_RETURN_IF_ERROR(AddBinaryOperator(ANEURALNETWORKS_ADD,
                                        model_builder,
                                        tensor_imm_product_name, tensor_b_name,
                                        true /* add_activation */, fuse_code,
                                        output));

  return Status::OK();
}

#pragma endregion op_batchnormalization

#pragma region op_pool

class PoolOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  bool IsQuantizedOp(const NodeUnit& node_unit) const override;
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

void CreatePoolOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  CreateSharedOpBuilderImpl<PoolOpBuilder>(
      op_type, op_registrations,
      {
          "GlobalAveragePool",
          "GlobalMaxPool",
          "AveragePool",
          "MaxPool",
          "QLinearAveragePool",
      });
}

void PoolOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  if (!IsQuantizedOp(node_unit))
    return;

  // skip input/output scales and zeropoints
  AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Inputs()[0].quant_param);   // x_scale, x_zp
  AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Outputs()[0].quant_param);  // y_scale, y_zp
}

bool PoolOpBuilder::IsQuantizedOp(const NodeUnit& node_unit) const {
  return IsQuantizedPool(GetQuantizedOpType(node_unit));
}

Status PoolOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  NodeAttrHelper helper(node_unit);

  auto input = node_unit.Inputs()[0].node_arg.Name();
  bool use_nchw = model_builder.UseNCHW();
  ORT_RETURN_IF_ERROR(IsOpInRequiredLayout(use_nchw, node_unit));

  const auto& output = node_unit.Outputs()[0].node_arg.Name();
  const auto& op_type = node_unit.OpType();

  int32_t op_code;
  bool is_quant_pool = IsQuantizedOp(node_unit);
  bool is_average_pool = op_type == "AveragePool" || op_type == "QLinearAveragePool";
  if (is_average_pool || op_type == "GlobalAveragePool")
    op_code = ANEURALNETWORKS_AVERAGE_POOL_2D;
  else  // (op_type == "MaxPool" || op_type == "GlobalMaxPool")
    op_code = ANEURALNETWORKS_MAX_POOL_2D;

  std::vector<int32_t> onnx_pads, onnx_strides, kernel_shape;
  bool use_auto_pad = false;
  int32_t nnapi_padding_code = ANEURALNETWORKS_PADDING_VALID;
  const auto& input_shape = shaper[input];
  if (is_average_pool || op_type == "MaxPool") {
    const auto auto_pad_type = StringToAutoPadType(helper.Get("auto_pad", "NOTSET"));
    kernel_shape = helper.Get("kernel_shape", std::vector<int32_t>{0, 0});
    onnx_strides = helper.Get("strides", std::vector<int>{1, 1});
    onnx_pads = helper.Get("pads", std::vector<int>{0, 0, 0, 0});
    const auto weight_size_y = static_cast<uint32_t>(kernel_shape[0]);
    const auto weight_size_x = static_cast<uint32_t>(kernel_shape[1]);
    ORT_RETURN_IF_ERROR(
        HandleAutoPad(input_shape, weight_size_y, weight_size_x,
                      onnx_strides, {1, 1} /* onnx_dilations */,
                      auto_pad_type, use_nchw,
                      onnx_pads, nnapi_padding_code, use_auto_pad));
  } else {  // (op_type == "GlobalAveragePool" || op_type == "GlobalMaxPool")
    use_auto_pad = true;
    nnapi_padding_code = ANEURALNETWORKS_PADDING_VALID;
    onnx_strides = std::vector<int32_t>{1, 1};
    onnx_pads = std::vector<int32_t>{0, 0, 0, 0};
    if (use_nchw) {
      kernel_shape = std::vector<int32_t>{static_cast<int32_t>(input_shape[2]),
                                          static_cast<int32_t>(input_shape[3])};
    } else {
      kernel_shape = std::vector<int32_t>{static_cast<int32_t>(input_shape[1]),
                                          static_cast<int32_t>(input_shape[2])};
    }
  }

  int32_t fuse_code = model_builder.FindActivation(node_unit);

  // Get output scale and zero point if this is QLinearAveragePool
  // Otherwise we will use the scale and zero point of the input
  const OperandType& input_operand_type = operand_types.at(input);
  float y_scale = input_operand_type.operandType.scale;
  int32_t y_zero_point = input_operand_type.operandType.zeroPoint;
  if (is_quant_pool) {
    const auto& initializers = model_builder.GetInitializerTensors();
    float x_scale = 0.0f;
    int32_t x_zero_point = 0;
    ORT_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(
        initializers, node_unit.Inputs()[0], node_unit.ModelPath(), x_scale, x_zero_point));

    // Verify if the scale and zero point values from onnx input and nnapi input match
    ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input, x_scale, x_zero_point));
    ORT_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(
        initializers, node_unit.Outputs()[0], node_unit.ModelPath(), y_scale, y_zero_point));
  }

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));

  if (use_auto_pad) {
    ADD_SCALAR_OPERAND(model_builder, input_indices, nnapi_padding_code);
  } else {
    ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_pads[1]);
    ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_pads[3]);
    ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_pads[0]);
    ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_pads[2]);
  }

  ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_strides[1]);
  ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_strides[0]);
  ADD_SCALAR_OPERAND(model_builder, input_indices, kernel_shape[1]);
  ADD_SCALAR_OPERAND(model_builder, input_indices, kernel_shape[0]);
  ADD_SCALAR_OPERAND(model_builder, input_indices, fuse_code);

  if (model_builder.GetNNAPIFeatureLevel() > ANEURALNETWORKS_FEATURE_LEVEL_2) {  // nchw only supported on api 29+
    ADD_SCALAR_OPERAND(model_builder, input_indices, use_nchw);
  }

  ORT_RETURN_IF_ERROR(shaper.Pool(input,
                                  onnx_pads, onnx_strides, kernel_shape,
                                  use_nchw,
                                  output));
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output], y_scale, y_zero_point);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(op_code, input_indices,
                                                 {output}, {output_operand_type}));
  return Status::OK();
}

#pragma endregion op_pool

#pragma region op_cast

class CastOpBuilder : public BaseOpBuilder {
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

void CreateCastOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<CastOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

Status CastOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  NodeAttrHelper helper(node_unit);

  const auto& input = node_unit.Inputs()[0].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();

  auto to = helper.Get("to", 0);
  Type type;
  switch (to) {
    case ONNX_NAMESPACE::TensorProto::FLOAT:
      type = Type::TENSOR_FLOAT32;
      break;
    case ONNX_NAMESPACE::TensorProto::INT32:
      type = Type::TENSOR_INT32;
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid cast to type: ", to);
  }

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  ORT_RETURN_IF_ERROR(shaper.Identity(input, output));
  const OperandType output_operand_type(type, shaper[output]);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_CAST, input_indices, {output},
                                                 {output_operand_type}));
  return Status::OK();
}

#pragma endregion

#pragma region op_depthtospace
class DepthToSpaceOpBuilder : public BaseOpBuilder {
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

void CreateDepthToSpaceOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<DepthToSpaceOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

Status DepthToSpaceOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto android_feature_level = model_builder.GetNNAPIFeatureLevel();
  NodeAttrHelper helper(node_unit);

  const auto& input = node_unit.Inputs()[0].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();

  bool use_nchw = model_builder.UseNCHW();
  ORT_RETURN_IF_ERROR(IsOpInRequiredLayout(use_nchw, node_unit));

  int32_t blocksize = SafeInt<int32_t>(node_unit.GetNode().GetAttributes().at("blocksize").i());

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  ADD_SCALAR_OPERAND(model_builder, input_indices, blocksize);

  if (use_nchw && android_feature_level > ANEURALNETWORKS_FEATURE_LEVEL_2) {
    // optional input to use nchw is available starting NNAPI feature level 3
    ADD_SCALAR_OPERAND(model_builder, input_indices, use_nchw);
  }

  ORT_RETURN_IF_ERROR(shaper.DepthToSpace(input, blocksize, use_nchw, output));
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_DEPTH_TO_SPACE, input_indices, {output},
                                                 {output_operand_type}));
  return Status::OK();
}

#pragma endregion

#pragma region op_softmax

class SoftMaxOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
  bool IsQuantizedOp(const NodeUnit& node_unit) const override;
};

void CreateSoftMaxOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<SoftMaxOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

bool SoftMaxOpBuilder::IsQuantizedOp(const NodeUnit& node_unit) const {
  return GetQuantizedOpType(node_unit) == QuantizedOpType::QDQSoftmax;
}

void SoftMaxOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  if (IsQuantizedOp(node_unit)) {
    AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Inputs()[0].quant_param);   // x_scale, x_zp
    AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Outputs()[0].quant_param);  // y_scale, y_zp
  }
}

Status SoftMaxOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto android_feature_level = model_builder.GetNNAPIFeatureLevel();
  NodeAttrHelper helper(node_unit);

  auto input = node_unit.Inputs()[0].node_arg.Name();

  // TODO: Needs fix.
  if (android_feature_level < ANEURALNETWORKS_FEATURE_LEVEL_3) {
    ORT_ENFORCE(model_builder.UseNCHW(),
                "For Android API Level < 29 input for softmax needs to be NCHW.");
  }

  int32_t axis = helper.Get("axis", 1);

  // Check if the quantization scale and ZP are correct
  float x_scale = 0.0f;
  int32_t x_zero_point = 0;
  float y_scale = 0.0f;
  int32_t y_zero_point = 0;
  if (IsQuantizedOp(node_unit)) {
    ORT_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(
        model_builder.GetInitializerTensors(), node_unit.Inputs()[0], node_unit.ModelPath(),
        x_scale, x_zero_point));

    ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input, x_scale, x_zero_point));

    y_scale = 1.f / 256;
  }

  const auto& output = node_unit.Outputs()[0].node_arg.Name();
  float beta = 1.f;
  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  ADD_SCALAR_OPERAND(model_builder, input_indices, beta);

  if (android_feature_level > ANEURALNETWORKS_FEATURE_LEVEL_2) {
    // you can only specify axis for android api level 29+
    ADD_SCALAR_OPERAND(model_builder, input_indices, axis);
  }

  ORT_RETURN_IF_ERROR(shaper.Identity(input, output));
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output], y_scale, y_zero_point);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_SOFTMAX, input_indices,
                                                 {output}, {output_operand_type}));
  return Status::OK();
}

#pragma endregion

#pragma region op_identity

class IdentityOpBuilder : public BaseOpBuilder {
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

void CreateIdentityOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<IdentityOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

Status IdentityOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  // Identity is not really going to do anything
  // Just register the dimension and type, with same index and new name
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  const auto& input = node_unit.Inputs()[0].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));  // input

  ORT_RETURN_IF_ERROR(shaper.Identity(input, output));
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);
  model_builder.RegisterOperand(output, operand_indices.at(input), output_operand_type);
  return Status::OK();
}

#pragma endregion

#pragma region op_gemm

class GemmOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  bool IsQuantizedOp(const NodeUnit& node_unit) const override;
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

bool GemmOpBuilder::IsQuantizedOp(const NodeUnit& node_unit) const {
  return IsQuantizedGemm(GetQuantizedOpType(node_unit));
}

void CreateGemmOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  CreateSharedOpBuilderImpl<GemmOpBuilder>(
      op_type, op_registrations,
      {
          "Gemm",
          "MatMul",
          "QLinearMatMul",
      });
}

void GemmOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  if (op_builder_helpers::IsSupportedBatchMatMul(node_unit, model_builder.GetNNAPIFeatureLevel())) {
    // no initializers to skip for batch matmul
    return;
  }

  const auto& inputs = node_unit.Inputs();
  if (IsQuantizedOp(node_unit)) {
    if (node_unit.OpType() == "QLinearMatMul" || node_unit.OpType() == "MatMul") {                 // QLinearMatMul/QDQMatMul
      AddQuantizationScaleAndZeroPointToSkip(model_builder, *inputs[0].quant_param);               // a_scale, a_zp
      AddInputToSkip(model_builder, inputs[1]);                                                    // b, b_scale, b_zp
      AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Outputs()[0].quant_param);  // y_scale, y_zp
    } else if (node_unit.OpType() == "Gemm") {                                                     // QDQGemm
      AddQuantizationScaleAndZeroPointToSkip(model_builder, *inputs[0].quant_param);               // a_scale, a_zp
      AddQuantizationScaleAndZeroPointToSkip(model_builder, *inputs[1].quant_param);               // b_scale, b_zp

      NodeAttrHelper helper(node_unit);
      const auto transB = helper.Get("transB", 0);
      // For transB == 0, we need to transpose it and add transposed initializer later into nnapi model,
      // not directly using it here, so add to skip list.
      if (transB == 0)
        model_builder.AddInitializerToSkip(inputs[1].node_arg.Name());

      if (inputs.size() > 2) {
        AddInputToSkip(model_builder, inputs[2]);  // c, c_scale, c_zp (bias)
      }
      AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Outputs()[0].quant_param);  // y_scale, y_zp
    }
  } else {
    const auto& op = node_unit.OpType();
    if (op == "MatMul") {
      model_builder.AddInitializerToSkip(inputs[1].node_arg.Name());
    } else if (op == "Gemm") {
      NodeAttrHelper helper(node_unit);
      const auto transB = helper.Get("transB", 0);
      if (transB == 0)
        model_builder.AddInitializerToSkip(inputs[1].node_arg.Name());
    }
  }
}

Status GemmOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  if (op_builder_helpers::IsSupportedBatchMatMul(node_unit, model_builder.GetNNAPIFeatureLevel())) {
    return op_builder_helpers::BuildBatchMatMul(model_builder, node_unit);
  }

  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& initializers(model_builder.GetInitializerTensors());

  const auto& op = node_unit.OpType();
  const auto& inputs = node_unit.Inputs();
  NodeAttrHelper helper(node_unit);

  const auto quant_type = GetQuantizedOpType(node_unit);
  const bool is_quant_matmul = (quant_type == QuantizedOpType::QDQMatMul || quant_type == QuantizedOpType::QLinearMatMul);
  const bool is_quant_gemm = quant_type == QuantizedOpType::QDQGemm;

  const auto& input1 = inputs[0].node_arg.Name();
  const auto& input2 = inputs[1].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();
  const auto transB = helper.Get("transB", 0);

  float a_scale = 0.0f,
        b_scale = 0.0f,
        y_scale = 0.0f;
  int32_t a_zero_point = 0,
          b_zero_point = 0,
          y_zero_point = 0;

  bool is_per_tensor_u8s8 = false;
  if (is_quant_matmul || is_quant_gemm) {
    std::optional<std::vector<float>> w_scales;
    ORT_RETURN_IF_ERROR(
        GetConvMatMulOpQuantizationScaleAndZeroPoint(model_builder, node_unit,
                                                     a_scale, b_scale, y_scale,
                                                     a_zero_point, b_zero_point, y_zero_point,
                                                     w_scales, is_per_tensor_u8s8));
  }

  uint32_t input_2_idx;
  if (transB == 0) {
    Type onnx_mat_b_type;
    if (!is_quant_matmul && !is_quant_gemm)
      onnx_mat_b_type = Type::TENSOR_FLOAT32;
    else
      onnx_mat_b_type = Type::TENSOR_QUANT8_ASYMM;

    const auto& mat_b_tensor = *initializers.at(input2);
    Shape onnx_mat_b_shape;
    for (auto dim : mat_b_tensor.dims())
      onnx_mat_b_shape.push_back(SafeInt<uint32_t>(dim));

    const OperandType onnx_mat_b_operand_type(onnx_mat_b_type, onnx_mat_b_shape, b_scale, b_zero_point);
    ORT_RETURN_IF_ERROR(AddInitializerTransposed(model_builder, onnx_mat_b_operand_type, input2, is_per_tensor_u8s8));
  }

  input_2_idx = operand_indices.at(input2);
  // Verify if the scale and zero point matchs from onnx input and nnapi input
  if (is_quant_matmul || is_quant_gemm) {
    ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input1, a_scale, a_zero_point));
    ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input2, b_scale, b_zero_point));
  }

  uint32_t bias_idx;
  bool has_bias = inputs.size() > 2;
  if (has_bias) {
    const auto& bias = inputs[2].node_arg.Name();
    if (!is_quant_gemm) {
      // We need squeeze the input tensor to 1d if necessary
      if (shaper[bias].size() > 1) {
        std::string bias_squeezed = model_builder.GetUniqueName(node_unit.Name() + op + "_bias_squeezed");
        // We will use squeeze all here
        ORT_RETURN_IF_ERROR(AddSqueezeOp(model_builder, node_unit.Name(),
                                         bias, bias_squeezed,
                                         {} /* axes */));
        bias_idx = operand_indices.at(bias_squeezed);
        LOGS_DEFAULT(VERBOSE) << "GemmOpBuilder - Operand [" << bias << "] squeezed from "
                              << Shape2String(shaper[bias])
                              << " to "
                              << Shape2String(shaper[bias_squeezed]);
      } else {
        bias_idx = operand_indices.at(bias);
      }
    } else {  // is_quant_gemm
      const auto& bias_tensor = *model_builder.GetInitializerTensors().at(bias);
      // QGemm has a contraint on input C to be int32 type
      ORT_RETURN_IF_NOT(bias_tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT32,
                        "bias of QDQGemm should be int32, actual type: ", bias_tensor.data_type());
      Shape bias_dimen;
      for (auto dim : bias_tensor.dims())
        bias_dimen.push_back(SafeInt<uint32_t>(dim));
      std::vector<uint8_t> unpacked_tensor;
      ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(bias_tensor, unpacked_tensor));
      OperandType bias_operand_type(Type::TENSOR_INT32, bias_dimen, a_scale * b_scale);
      ORT_RETURN_IF_ERROR(
          model_builder.AddOperandFromPersistMemoryBuffer(bias, unpacked_tensor.data(), bias_operand_type));

      bias_idx = operand_indices.at(bias);
    }

  } else {
    // No C supplied, we need a vector of 0
    std::string bias = model_builder.GetUniqueName(node_unit.Name() + op + "_bias");
    const auto& bias_type = operand_types.at(input2).type;
    const Shape& bias_dimen = {shaper[input2][0]};
    if (bias_type == Type::TENSOR_FLOAT32) {
      std::vector<float> buffer(bias_dimen[0], 0.f);
      OperandType bias_operand_type(Type::TENSOR_FLOAT32, bias_dimen);
      ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(bias, buffer.data(), bias_operand_type));
    } else if (bias_type == Type::TENSOR_QUANT8_ASYMM) {
      std::vector<int32_t> buffer(bias_dimen[0], 0);
      OperandType bias_operand_type(Type::TENSOR_INT32, bias_dimen, a_scale * b_scale, 0);
      ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(bias, buffer.data(), bias_operand_type));
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unknown weight type ", TypeToStr(bias_type));
    }

    bias_idx = operand_indices.at(bias);
  }

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input1));  // A
  input_indices.push_back(input_2_idx);                 // B
  input_indices.push_back(bias_idx);                    // C
  int32_t fuse_code = model_builder.FindActivation(node_unit);
  ADD_SCALAR_OPERAND(model_builder, input_indices, fuse_code);

  ORT_RETURN_IF_ERROR(shaper.FC(input1, input2, output));
  const OperandType output_operand_type(operand_types.at(input1).type, shaper[output], y_scale, y_zero_point);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_FULLY_CONNECTED, input_indices,
                                                 {output}, {output_operand_type}));
  return Status::OK();
}

#pragma endregion

#pragma region op_unary

class UnaryOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  bool IsQuantizedOp(const NodeUnit& node_unit) const override;
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

bool UnaryOpBuilder::IsQuantizedOp(const NodeUnit& node_unit) const {
  // TODO, add support for QDQ NodeUnit
  return node_unit.OpType() == "QLinearSigmoid";
}

void UnaryOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  if (!IsQuantizedOp(node_unit))
    return;

  AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Inputs()[0].quant_param);   // x_scale, x_zp
  AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Outputs()[0].quant_param);  // y_scale, y_zp
}

void CreateUnaryOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  CreateSharedOpBuilderImpl<UnaryOpBuilder>(
      op_type, op_registrations,
      {
          "Abs",
          "Exp",
          "Floor",
          "Log",
          "Sigmoid",
          "Neg",
          "Sin",
          "Sqrt",
          "Tanh",
          "QLinearSigmoid",
      });
}

Status UnaryOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& op_type(node_unit.OpType());

  const auto& input = node_unit.Inputs()[0].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();
  ORT_RETURN_IF_ERROR(shaper.Identity(input, output));
  bool is_qlinear_sigmoid = op_type == "QLinearSigmoid";

  int32_t op_code;
  if (op_type == "Abs")
    op_code = ANEURALNETWORKS_ABS;
  else if (op_type == "Exp")
    op_code = ANEURALNETWORKS_EXP;
  else if (op_type == "Floor")
    op_code = ANEURALNETWORKS_FLOOR;
  else if (op_type == "Log")
    op_code = ANEURALNETWORKS_LOG;
  else if (op_type == "Sigmoid" || is_qlinear_sigmoid)
    op_code = ANEURALNETWORKS_LOGISTIC;
  else if (op_type == "Neg")
    op_code = ANEURALNETWORKS_NEG;
  else if (op_type == "Sin")
    op_code = ANEURALNETWORKS_SIN;
  else if (op_type == "Sqrt")
    op_code = ANEURALNETWORKS_SQRT;
  else if (op_type == "Tanh")
    op_code = ANEURALNETWORKS_TANH;
  else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "UnaryOpBuilder, unknown op: ", op_type);
  }

  float y_scale = 0.0f;
  int32_t y_zero_point = 0;
  if (is_qlinear_sigmoid) {
    const auto& initializers = model_builder.GetInitializerTensors();
    float x_scale = 0.0f;
    int32_t x_zero_point = 0;
    ORT_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(
        initializers, node_unit.Inputs()[0], node_unit.ModelPath(), x_scale, x_zero_point));

    // Verify if the scale and zero point values from onnx input and nnapi input match
    ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input, x_scale, x_zero_point));

    // We already verified this in  UnaryOpSupportChecker::IsOpSupportedImpl
    y_scale = 1.f / 256;
    y_zero_point = 0;
  }

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output], y_scale, y_zero_point);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(op_code, input_indices,
                                                 {output}, {output_operand_type}));
  return Status::OK();
}

#pragma endregion

#pragma region op_concat

class ConcatOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
  bool IsQuantizedOp(const NodeUnit& node_unit) const override;
};

void CreateConcatOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ConcatOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

bool ConcatOpBuilder::IsQuantizedOp(const NodeUnit& node_unit) const {
  // TODO: add support of QLinearConcat
  return GetQuantizedOpType(node_unit) == QuantizedOpType::QDQConcat;
}

void ConcatOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  if (IsQuantizedOp(node_unit)) {
    for (size_t i = 0; i < node_unit.Inputs().size(); ++i) {
      AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Inputs()[i].quant_param);
    }

    AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Outputs()[0].quant_param);  // y_scale, y_zp
  }
}

Status ConcatOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  NodeAttrHelper helper(node_unit);
  const auto& inputs = node_unit.Inputs();

  std::vector<uint32_t> input_indices;
  const auto& input0 = inputs[0].node_arg.Name();
  const auto node_input_size = inputs.size();

  bool is_quant_op = IsQuantizedOp(node_unit);

  if (!is_quant_op) {
    // If the inputs are uint8 and this is not a quantized Concat, we need to verify all the inputs have the
    // same scale and zero points.
    // [Side note: int8 input is not supported currently by the NNAPI EP (enforced in ConcatOpSupportChecker).
    // it is supported by NNAPI though and int8 input is allowed to have different scale  and zp values.]
    //
    // ONNX allows Concat (not QlinearConcat, not QDQ concat) to run directly on uint8 without scales and zps.
    // NNAPI requires all uint8 inputs to have scale values > 0. (zero point can be 0.)
    // See https://android.googlesource.com/platform/frameworks/ml/+/master/nn/common/Validation.cpp#486
    //
    // We need to use the scales and zps from the NNAPI input directly, there is no easy way to get the input
    // scales and zps in OpSupportChecker, so we need to verify here.
    // Also we have to assume the output scale and zp are the same as input 0
    if (operand_types.at(input0).type == android::nn::wrapper::Type::TENSOR_QUANT8_ASYMM) {
      auto scale = operand_types.at(input0).operandType.scale;
      auto zero_point = operand_types.at(input0).operandType.zeroPoint;

      // TODO: if we see scale == 0 in real models we could consider using 1 as a default. This is what TF does
      // https://github.com/tensorflow/tensorflow/blob/7737c518a864e54be9b676fe063436ccbbef21b9/tensorflow/lite/delegates/nnapi/nnapi_delegate.cc#L468-L471
      ORT_RETURN_IF_NOT(scale > 0, "NNAPI requires scale to be > 0.");

      // Compare scale and zp of input0 to input1~n
      for (size_t i = 1; i < node_input_size; i++) {
        const auto& type = operand_types.at(inputs[i].node_arg.Name());
        ORT_RETURN_IF_NOT(scale == type.operandType.scale,
                          "Input[", i, "]'s scale: ", type.operandType.scale,
                          " is different than input[0]'s scale: ", scale);

        ORT_RETURN_IF_NOT(zero_point == type.operandType.zeroPoint,
                          "Input[", i, "]'s zero_point: ", type.operandType.zeroPoint,
                          " is different than input[0]'s zero_point: ", zero_point);
      }
    }
  }

  std::vector<std::string> input_names;
  input_names.reserve(node_input_size);
  for (size_t i = 0; i < node_input_size; i++) {
    const auto& input = inputs[i].node_arg.Name();

    if (is_quant_op) {
      // scale and zp values consistency was checked in ConcatOpSupportChecker
      float scale = 0.0f;
      int32_t zero_point = 0;
      ORT_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(
          model_builder.GetInitializerTensors(), node_unit.Inputs()[i], node_unit.ModelPath(),
          scale, zero_point));

      ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input, scale, zero_point));
    }

    input_indices.push_back(operand_indices.at(input));
    input_names.push_back(input);
  }

  // Get the output scale and zp for quantized concat, default value is from input 0
  float y_scale = operand_types.at(input0).operandType.scale;
  int32_t y_zero_point = operand_types.at(input0).operandType.zeroPoint;
  if (is_quant_op) {
    ORT_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(
        model_builder.GetInitializerTensors(), node_unit.Outputs()[0], node_unit.ModelPath(),
        y_scale, y_zero_point));
  }

  int32_t rank = static_cast<int32_t>(shaper[input0].size());
  int32_t axis = static_cast<int32_t>(HandleNegativeAxis(helper.Get("axis", 1), rank));

  ADD_SCALAR_OPERAND(model_builder, input_indices, axis);

  const auto& output = node_unit.Outputs()[0].node_arg.Name();
  ORT_RETURN_IF_ERROR(shaper.Concat(input_names, axis, output));
  OperandType output_operand_type(operand_types.at(input0).type, shaper[output], y_scale, y_zero_point);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_CONCATENATION, input_indices,
                                                 {output}, {output_operand_type}));
  return Status::OK();
}

#pragma endregion

#pragma region op_squeeze

class SqueezeOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

void CreateSqueezeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<SqueezeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

void SqueezeOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  if (node_unit.SinceVersion() > 12 && node_unit.Inputs().size() > 1) {
    model_builder.AddInitializerToSkip(node_unit.Inputs()[1].node_arg.Name());
  }
}

Status SqueezeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto input = node_unit.Inputs()[0].node_arg.Name();

  std::vector<int32_t> axes;
  ORT_RETURN_IF_ERROR(GetAxesForSqueezeAndUnSqueeze(model_builder, node_unit, axes));
  return AddSqueezeOp(model_builder, node_unit.Name(), input, node_unit.Outputs()[0].node_arg.Name(), axes);
}

#pragma endregion

#pragma region op_quantizelinear

class QuantizeLinearOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

void CreateQuantizeLinearOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<QuantizeLinearOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

void QuantizeLinearOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Outputs()[0].quant_param);  // y_scale, y_zp
}

Status QuantizeLinearOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());

  const auto& input = node_unit.Inputs()[0].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();

  float scale = 0.0f;
  int32_t zero_point = 0;
  ORT_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(
      model_builder.GetInitializerTensors(), node_unit.Outputs()[0], node_unit.ModelPath(), scale, zero_point));

  Type output_type = Type::TENSOR_QUANT8_ASYMM;
  ORT_RETURN_IF_ERROR(shaper.Identity(input, output));
  const OperandType output_operand_type(output_type, shaper[output], scale, zero_point);
  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_QUANTIZE, input_indices,
                                                 {output}, {output_operand_type}));
  return Status::OK();
}

#pragma endregion

#pragma region op_dequantizelinear

class DequantizeLinearOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

void CreateDequantizeLinearOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<DequantizeLinearOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

void DequantizeLinearOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Inputs()[0].quant_param);  // x_scale, x_zp
}

Status DequantizeLinearOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& inputs = node_unit.Inputs();

  const auto& input = inputs[0].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();

  float scale = 0.0;
  int32_t zero_point = 0;
  ORT_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(
      model_builder.GetInitializerTensors(), node_unit.Inputs()[0], node_unit.ModelPath(), scale, zero_point));

  ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input, scale, zero_point));

  ORT_RETURN_IF_ERROR(shaper.Identity(input, output));
  const OperandType output_operand_type(Type::TENSOR_FLOAT32, shaper[output]);

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_DEQUANTIZE, input_indices,
                                                 {output}, {output_operand_type}));
  return Status::OK();
}

#pragma endregion

#pragma region op_LRN

class LRNOpBuilder : public BaseOpBuilder {
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

void CreateLRNOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<LRNOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

Status LRNOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  NodeAttrHelper helper(node_unit);
  const auto android_feature_level = model_builder.GetNNAPIFeatureLevel();

  auto input = node_unit.Inputs()[0].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();
  auto use_nchw = model_builder.UseNCHW();
  ORT_RETURN_IF_ERROR(IsOpInRequiredLayout(use_nchw, node_unit));

  if (android_feature_level < ANEURALNETWORKS_FEATURE_LEVEL_3) {
    // on android api level 28, we need to transpose the nchw input to nhwc
    // it is very rare that users set nchw format when using nnapi. Therefore, instead of
    // adding the ability to support conversion we fail and stop.
    ORT_ENFORCE(!use_nchw, "NCHW format is not supported on android api level 28");
  }

  auto alpha = helper.Get("alpha", 0.0001f);
  const auto beta = helper.Get("beta", 0.75f);
  const auto bias = helper.Get("bias", 1.0f);
  const auto size = helper.Get("size", 1);

  const auto radius = (size - 1) / 2;
  alpha /= size;  // NNAPI's alpha is different than ONNX's alpha

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  ADD_SCALAR_OPERAND(model_builder, input_indices, radius);
  ADD_SCALAR_OPERAND(model_builder, input_indices, bias);
  ADD_SCALAR_OPERAND(model_builder, input_indices, alpha);
  ADD_SCALAR_OPERAND(model_builder, input_indices, beta);

  // specify axis is only available on api level >= 29
  if (android_feature_level > ANEURALNETWORKS_FEATURE_LEVEL_2) {
    // ONNX LRN is always performed on C dimension
    int32_t axis = use_nchw
                       ? 1   // nchw
                       : 3;  // nhwc
    ADD_SCALAR_OPERAND(model_builder, input_indices, axis);
  }

  ORT_RETURN_IF_ERROR(shaper.Identity(input, output));
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION, input_indices,
                                                 {output}, {output_operand_type}));
  return Status::OK();
}

#pragma endregion

#pragma region op_clip

class ClipOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

void CreateClipOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ClipOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

void ClipOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  const auto& inputs = node_unit.Inputs();
  if (inputs.size() > 1)
    model_builder.AddInitializerToSkip(inputs[1].node_arg.Name());  // min

  if (inputs.size() > 2)
    model_builder.AddInitializerToSkip(inputs[2].node_arg.Name());  // max
}

Status ClipOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  const auto& input = node_unit.Inputs()[0].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();

  ORT_RETURN_IF_ERROR(shaper.Identity(input, output));
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);

  if (Contains(model_builder.GetFusedActivations(), input)) {
    LOGS_DEFAULT(VERBOSE) << "Clip Node [" << node_unit.Name() << "] fused";
    model_builder.RegisterOperand(output, operand_indices.at(input), output_operand_type);
    return Status::OK();
  }

  float min, max;
  GetClipMinMax(model_builder.GetInitializerTensors(), node_unit.GetNode(), min, max,
                logging::LoggingManager::DefaultLogger());

  int32_t op_code;
  if (min == 0.0f && max == 6.0f)
    op_code = ANEURALNETWORKS_RELU6;
  else if (min == -1.0f && max == 1.0f)
    op_code = ANEURALNETWORKS_RELU1;
  else
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "ClipOpBuilder, unsupported input [", min, ", ", max, "].",
                           "We should not reach here, ClipOpBuilder::IsOpSupportedImpl should have caught this.");

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(op_code, input_indices,
                                                 {output}, {output_operand_type}));
  return Status::OK();
}

#pragma endregion

#pragma region op_Resize

class ResizeOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
  bool IsQuantizedOp(const NodeUnit& node_unit) const override;
};

void CreateResizeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ResizeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

bool ResizeOpBuilder::IsQuantizedOp(const NodeUnit& node_unit) const {
  return GetQuantizedOpType(node_unit) == QuantizedOpType::QDQResize;
}

void ResizeOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  const auto& inputs = node_unit.Inputs();
  if (IsQuantizedOp(node_unit)) {
    AddQuantizationScaleAndZeroPointToSkip(model_builder, *inputs[0].quant_param);               // x_scale, x_zp
    AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Outputs()[0].quant_param);  // y_scale, y_zp
  }

  // We don't really use ROI here, so add them to skipped list
  model_builder.AddInitializerToSkip(inputs[1].node_arg.Name());  // ROI

  // We will still add scales to the skipped list even sizes are present
  // since there is no use of it, we will not process it later
  model_builder.AddInitializerToSkip(inputs[2].node_arg.Name());  // scales

  if (inputs.size() > 3)
    model_builder.AddInitializerToSkip(inputs[3].node_arg.Name());  // sizes
}

Status ResizeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& initializers(model_builder.GetInitializerTensors());
  NodeAttrHelper helper(node_unit);
  const auto& inputs = node_unit.Inputs();
  const auto android_feature_level = model_builder.GetNNAPIFeatureLevel();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();

  auto input = inputs[0].node_arg.Name();
  bool use_nchw = model_builder.UseNCHW();
  ORT_RETURN_IF_ERROR(IsOpInRequiredLayout(use_nchw, node_unit));

  // Check if the quantization scale and ZP is correct
  if (IsQuantizedOp(node_unit)) {
    float x_scale = 0.0f;
    int32_t x_zero_point = 0;
    ORT_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(
        initializers, node_unit.Inputs()[0], node_unit.ModelPath(), x_scale, x_zero_point));
    ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input, x_scale, x_zero_point));
  }

  bool is_linear_resize = helper.Get("mode", "nearest") == "linear";

  int32_t operationCode = is_linear_resize ? ANEURALNETWORKS_RESIZE_BILINEAR
                                           : ANEURALNETWORKS_RESIZE_NEAREST_NEIGHBOR;

  const auto coord_trans_mode = helper.Get("coordinate_transformation_mode", "half_pixel");
  bool using_half_pixel = coord_trans_mode == "half_pixel";
  bool using_align_corners = coord_trans_mode == "align_corners";

  // if the node domain is NHWC it means all the node inputs are converted to NHWC format by the layout transformer.
  // pick the index for height and width based on the format.
  int h_idx = use_nchw ? 2 : 1;
  int w_idx = use_nchw ? 3 : 2;

  if (inputs.size() == 3) {  // we are using scales
    const auto& scales_name = inputs[2].node_arg.Name();
    const auto& scales_tensor = *initializers.at(scales_name);
    std::vector<uint8_t> unpacked_tensor;
    ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(scales_tensor, unpacked_tensor));
    const float* scales_data = reinterpret_cast<const float*>(unpacked_tensor.data());
    ORT_RETURN_IF_ERROR(
        shaper.ResizeUsingScales(input, scales_data[h_idx], scales_data[w_idx], use_nchw, output));
  } else {  // we are using sizes
    const auto& sizes_name = inputs[3].node_arg.Name();
    const auto& sizes_tensor = *initializers.at(sizes_name);
    std::vector<uint8_t> unpacked_tensor;
    ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(sizes_tensor, unpacked_tensor));
    const int64_t* sizes_data = reinterpret_cast<const int64_t*>(unpacked_tensor.data());
    ORT_RETURN_IF_ERROR(
        shaper.ResizeUsingOutputSizes(input, SafeInt<uint32_t>(sizes_data[h_idx]), SafeInt<uint32_t>(sizes_data[w_idx]), use_nchw, output));
  }

  const auto& output_shape = shaper[output];
  int32_t output_h = output_shape[h_idx];
  int32_t output_w = output_shape[w_idx];

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  ADD_SCALAR_OPERAND(model_builder, input_indices, output_w);
  ADD_SCALAR_OPERAND(model_builder, input_indices, output_h);

  if (android_feature_level > ANEURALNETWORKS_FEATURE_LEVEL_2) {
    // using nchw is only available on API level 29
    ADD_SCALAR_OPERAND(model_builder, input_indices, use_nchw);
  }

  // Currently we only support align_corners and half_pixel on bilinear resize
  // TODO, investigate nearest neighbor resize difference between NNAPI(based on TF) and ONNX
  if (is_linear_resize) {
    if (android_feature_level > ANEURALNETWORKS_FEATURE_LEVEL_3 && (using_align_corners || using_half_pixel)) {
      ADD_SCALAR_OPERAND(model_builder, input_indices, using_align_corners);
      if (using_half_pixel)
        ADD_SCALAR_OPERAND(model_builder, input_indices, using_half_pixel);
    }
  }

  OperandType output_operand_type = operand_types.at(input);
  output_operand_type.SetDimensions(output_shape);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(operationCode, input_indices,
                                                 {output}, {output_operand_type}));

  return Status::OK();
}

#pragma endregion

#pragma region op_flatten

class FlattenOpBuilder : public BaseOpBuilder {
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

void CreateFlattenOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<FlattenOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

Status FlattenOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto input = node_unit.Inputs()[0].node_arg.Name();

  // Flatten is basically a reshape to 2d tensor
  // Get the shape for Reshape here
  Shape input_shape;
  GetShape(node_unit.Inputs()[0].node_arg, input_shape);
  int32_t dim_1 = 1;
  int32_t dim_2 = 1;
  GetFlattenOutputShape(node_unit, input_shape, dim_1, dim_2);
  // If the input is of dynamic shape, replace 0 (dynamic) dimension with -1
  // We cannot have dim_1 and dim_2 both be 0 here, it was checked in IsOpSupportedImpl
  dim_1 = dim_1 == 0 ? -1 : dim_1;
  dim_2 = dim_2 == 0 ? -1 : dim_2;
  std::vector<int32_t> shape{dim_1, dim_2};
  return ReshapeOpBuilder::AddReshapeOperator(model_builder, node_unit, input, shape);
}

#pragma endregion

#pragma region op_gather

class GatherOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

void CreateGatherOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<GatherOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

void GatherOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  const auto& inputs = node_unit.Inputs();
  const auto& indices_name = inputs[1].node_arg.Name();
  int32_t indices_data_type;
  GetType(node_unit.Inputs()[1].node_arg, indices_data_type);
  if (Contains(model_builder.GetInitializerTensors(), indices_name) &&
      indices_data_type != ONNX_NAMESPACE::TensorProto_DataType_INT32) {
    // Skip the second input `indices` for Gather if it is an initializer
    model_builder.AddInitializerToSkip(indices_name);
  }
}

Status GatherOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& initializers(model_builder.GetInitializerTensors());
  const auto& input1 = node_unit.Inputs()[0].node_arg.Name();
  const auto& input2 = node_unit.Inputs()[1].node_arg.Name();  // "indices"
  const auto& output = node_unit.Outputs()[0].node_arg.Name();

  NodeAttrHelper helper(node_unit);
  int32_t rank = static_cast<int32_t>(shaper[input1].size());
  int32_t axis = static_cast<int32_t>(HandleNegativeAxis(helper.Get("axis", 0), rank));

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input1));
  ADD_SCALAR_OPERAND(model_builder, input_indices, axis);

  int32_t indices_data_type;
  GetType(node_unit.Inputs()[1].node_arg, indices_data_type);
  if (Contains(model_builder.GetInitializerTensors(), input2) &&
      indices_data_type != ONNX_NAMESPACE::TensorProto_DataType_INT32) {
    // Add indices operand into nnapi
    const auto& indices_tensor = *initializers.at(input2);
    std::vector<uint8_t> unpacked_tensor;
    ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(indices_tensor, unpacked_tensor));

    const auto data_type = indices_tensor.data_type();
    const auto indices_shape = indices_tensor.dims();
    uint32_t size = 1;
    Shape indices_dimen;
    indices_dimen.reserve(indices_tensor.dims_size());
    for (auto i = 0; i < indices_tensor.dims_size(); i++) {
      size *= SafeInt<uint32_t>(indices_shape[i]);
      indices_dimen.push_back(static_cast<uint32_t>(indices_shape[i]));
    }

    std::vector<int32_t> indices(size);
    // see https://gist.github.com/shafik/848ae25ee209f698763cffee272a58f8#type-punning-arrays for the usage of memcpy here
    if (data_type == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
      for (uint32_t i = 0; i < size; i++) {
        int64_t index_i64;
        memcpy(&index_i64, unpacked_tensor.data() + i * sizeof(int64_t), sizeof(int64_t));
        indices[i] = SafeInt<int32_t>(index_i64);
      }
    } else if (data_type == ONNX_NAMESPACE::TensorProto_DataType_INT32) {
      for (uint32_t i = 0; i < size; i++) {
        int32_t index;
        memcpy(&index, unpacked_tensor.data() + i * sizeof(int32_t), sizeof(int32_t));
        indices[i] = SafeInt<int32_t>(index);
      }
    }

    OperandType indices_operand_type(Type::TENSOR_INT32, indices_dimen);
    ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(input2, indices.data(), indices_operand_type));
  }
  input_indices.push_back(operand_indices.at(input2));
  ORT_RETURN_IF_ERROR(shaper.Gather(input1, input2, axis, output));
  const OperandType output_operand_type(operand_types.at(input1).type, shaper[output]);

  return model_builder.AddOperation(ANEURALNETWORKS_GATHER, input_indices,
                                    {output}, {output_operand_type});
}

#pragma endregion

#pragma region op_minmax

class MinMaxOpBuilder : public BaseOpBuilder {
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
  static Status AddMinMaxOperator(ModelBuilder& model_builder, const NodeUnit& node_unit,
                                  const std::string& input1, const std::string& input2);
};

void CreateMinMaxOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  CreateSharedOpBuilderImpl<MinMaxOpBuilder>(
      op_type, op_registrations,
      {
          "Min",
          "Max",
      });
}

/* static */ Status MinMaxOpBuilder::AddMinMaxOperator(ModelBuilder& model_builder, const NodeUnit& node_unit,
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

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input1));  // input 1
  input_indices.push_back(operand_indices.at(input2));  // input 2
  ORT_RETURN_IF_ERROR(shaper.Eltwise(input1, input2, output));
  const OperandType output_operand_type(operand_types.at(input1).type, shaper[output]);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(op_code, input_indices,
                                                 {output}, {output_operand_type}));

  return Status::OK();
}

Status MinMaxOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  const auto& inputs = node_unit.Inputs();
  std::string input1 = inputs[0].node_arg.Name();
  std::string input2 = inputs[1].node_arg.Name();

  return AddMinMaxOperator(model_builder, node_unit, input1, input2);
}

#pragma endregion

#pragma region op_elu

class EluOpBuilder : public BaseOpBuilder {
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

void CreateEluOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<EluOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

Status EluOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& input = node_unit.Inputs()[0].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();
  ORT_RETURN_IF_ERROR(shaper.Identity(input, output));
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);
  NodeAttrHelper helper(node_unit);
  const auto alpha = helper.Get("alpha", 1.0f);
  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  ADD_SCALAR_OPERAND(model_builder, input_indices, alpha);
  return model_builder.AddOperation(ANEURALNETWORKS_ELU, input_indices,
                                    {output}, {output_operand_type});
}

#pragma endregion

#pragma region op_slice

class SliceOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

void CreateSliceOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<SliceOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

void SliceOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  // Skip everything except input0 for Slice
  const auto& inputs = node_unit.Inputs();
  model_builder.AddInitializerToSkip(inputs[1].node_arg.Name());  // starts
  model_builder.AddInitializerToSkip(inputs[2].node_arg.Name());  // ends
  if (inputs.size() > 3) {
    model_builder.AddInitializerToSkip(inputs[3].node_arg.Name());  // axes
    if (inputs.size() > 4) {
      model_builder.AddInitializerToSkip(inputs[4].node_arg.Name());  // steps
    }
  }
}

Status SliceOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& inputs = node_unit.Inputs();
  const auto& input_shape = shaper[inputs[0].node_arg.Name()];
  TensorShapeVector input_shape_64(input_shape.cbegin(), input_shape.cend());
  SliceOp::PrepareForComputeMetadata compute_metadata(input_shape_64);

  {
    // We need to copy the data from the starts/ends/axes/steps initializers to int64 vectors
    // to be used in shared PrepareForCompute function to calculate the output shape
    // and normalize inputs, for example, input can be starts/ends/steps for certain axes,
    // PrepareForCompute can generate standard starts/ends/steps/axes for each axes
    TensorShapeVector input_starts;
    TensorShapeVector input_ends;
    TensorShapeVector input_axes;
    TensorShapeVector input_steps;

    const auto CopyInputData = [&inputs, &model_builder](size_t input_idx, TensorShapeVector& data) {
      data.clear();

      // This is an optional input, return empty vector
      if (inputs.size() <= input_idx)
        return Status::OK();

      const auto& input_name = inputs[input_idx].node_arg.Name();
      const auto& initializers(model_builder.GetInitializerTensors());

      const auto& tensor = *initializers.at(input_name);
      std::vector<uint8_t> unpacked_tensor;
      ORT_RETURN_IF_ERROR(
          onnxruntime::utils::UnpackInitializerData(tensor, model_builder.GetGraphViewer().ModelPath(),
                                                    unpacked_tensor));
      size_t tensor_byte_size = unpacked_tensor.size();
      const auto data_type = tensor.data_type();
      if (data_type == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
        const int64_t* tensor_data = reinterpret_cast<const int64_t*>(unpacked_tensor.data());
        size_t size = tensor_byte_size / sizeof(int64_t);
        data.insert(data.end(), tensor_data, tensor_data + size);
      } else if (data_type == ONNX_NAMESPACE::TensorProto_DataType_INT32) {
        const int32_t* tensor_data = reinterpret_cast<const int32_t*>(unpacked_tensor.data());
        size_t size = tensor_byte_size / sizeof(int32_t);
        data.insert(data.end(), tensor_data, tensor_data + size);
      } else {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Data type for starts and ends inputs' is not supported in this build. Got ",
                               data_type);
      }

      return Status::OK();
    };

    ORT_RETURN_IF_ERROR(CopyInputData(1, input_starts));
    ORT_RETURN_IF_ERROR(CopyInputData(2, input_ends));
    ORT_RETURN_IF_ERROR(CopyInputData(3, input_axes));
    ORT_RETURN_IF_ERROR(CopyInputData(4, input_steps));
    ORT_RETURN_IF_ERROR(
        SliceOp::PrepareForComputeHelper(input_starts, input_ends, input_axes, input_steps, compute_metadata));
  }

  // output shape is of type uint32_t, convert from int64 compute_metadata.output_dims_
  Shape nnapi_output_shape;
  nnapi_output_shape.reserve(compute_metadata.output_dims_.size());
  std::transform(compute_metadata.output_dims_.cbegin(), compute_metadata.output_dims_.cend(),
                 std::back_inserter(nnapi_output_shape),
                 [](int64_t i) { return SafeInt<uint32_t>(i); });

  const auto& input = inputs[0].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();

  // No shape inference for Slice, everything is calculated here, we only need to add the output shape
  // to the shaper
  shaper.AddShape(output, nnapi_output_shape);
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));

  // begin/end/strides of ANEURALNETWORKS_STRIDED_SLICE have the same shape
  Shape param_dimen = {static_cast<uint32_t>(input_shape.size())};

  // helper function to add begin/end/strides of ANEURALNETWORKS_STRIDED_SLICE
  const auto AddOperand = [&model_builder, &node_unit, &input_indices, &operand_indices](
                              const char* name, const Shape& shape, const gsl::span<const int64_t>& param_raw_data) {
    std::vector<int32_t> param_data;
    param_data.reserve(param_raw_data.size());
    std::transform(param_raw_data.cbegin(), param_raw_data.cend(),
                   std::back_inserter(param_data),
                   [](int64_t i) { return SafeInt<int32_t>(i); });
    std::string param_name = model_builder.GetUniqueName(node_unit.Name() + name);
    OperandType param_operand_type(Type::TENSOR_INT32, shape);
    ORT_RETURN_IF_ERROR(
        model_builder.AddOperandFromPersistMemoryBuffer(param_name, param_data.data(), param_operand_type));
    input_indices.push_back(operand_indices.at(param_name));
    return Status::OK();
  };

  ORT_RETURN_IF_ERROR(AddOperand("starts", param_dimen, compute_metadata.starts_));  // nnapi_begin

  // NNAPI has 2 slice operations
  // - ANEURALNETWORKS_SLICE
  //    Simpler and faster version of slice without steps, available from ANEURALNETWORKS_FEATURE_LEVEL_3
  //    Use this one if no step other than 1 is used in ONNX slice
  // - ANEURALNETWORKS_STRIDED_SLICE
  //    More comprehensive version, available from ANEURALNETWORKS_FEATURE_LEVEL_2
  int op_code = ANEURALNETWORKS_STRIDED_SLICE;
  if (std::all_of(compute_metadata.steps_.cbegin(),
                  compute_metadata.steps_.cend(),
                  [](int64_t i) { return i == 1; }) &&
      model_builder.GetNNAPIFeatureLevel() > ANEURALNETWORKS_FEATURE_LEVEL_2) {
    op_code = ANEURALNETWORKS_SLICE;
    // the nnapi size of the slice in this case is the output shape
    ORT_RETURN_IF_ERROR(AddOperand("sizes", param_dimen, compute_metadata.output_dims_));  // nnapi_sizes
  } else {
    // ** The special treatment of ends **
    // The nnapi_end need some special handling, based on the current undocumented design of
    // ANEURALNETWORKS_STRIDED_SLICE
    // For ORT, for a single axis, after SliceOp::PrepareForCompute, and the step is negative,
    // and the last element for slice is at the beginning of the axis (we are slicing backwards)
    // The end for this axis will be -1
    // For NNAPI, it is not documented that end can be negative,
    // see https://developer.android.com/ndk/reference/group/neural-networks#group___neural_networks_1ggaabbe492c60331b13038e39d4207940e0a89695302f8b1e7ae7ce8f4d8c0b8a752
    // However, the actual NNAPI StridedSlice has some odd implementations,
    // See https://android.googlesource.com/platform/frameworks/ml/+/5b525d4d9100819d87447bd2c2a0bcfdd62899ee/nn/common/operations/StridedSlice.cpp#177
    // and, https://android.googlesource.com/platform/frameworks/ml/+/5b525d4d9100819d87447bd2c2a0bcfdd62899ee/nn/common/include/OperationsUtils.h#262
    // If a negative end is no less than -dim (dimension of the axis), it will be treated as an index counting from
    // the end, for example, dim = 5, and end = -1, the end will be normalized to 4, which will cause
    // incorrect result, so here we have to make the end = -dim - 1 such that it will not be treated as
    // an index counting from the end.
    auto ends = compute_metadata.ends_;
    for (size_t i = 0, limit = ends.size(); i < limit; ++i) {
      if (ends[i] == -1) {
        ends[i] = -static_cast<int32_t>(input_shape[i] + 1);
      }
    }
    ORT_RETURN_IF_ERROR(AddOperand("ends", param_dimen, ends));                      // nnapi_end
    ORT_RETURN_IF_ERROR(AddOperand("steps", param_dimen, compute_metadata.steps_));  // nnapi_strides
    // We do not use the following inputs in ANEURALNETWORKS_STRIDED_SLICE, set them all to 0
    ADD_SCALAR_OPERAND(model_builder, input_indices, 0);  // begin_mask
    ADD_SCALAR_OPERAND(model_builder, input_indices, 0);  // end_mask
    ADD_SCALAR_OPERAND(model_builder, input_indices, 0);  // shrink_axis_mask
  }
  return model_builder.AddOperation(op_code, input_indices, {output}, {output_operand_type});
}

#pragma endregion

#pragma region op_pad

class PadOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

void CreatePadOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<PadOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

void PadOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  const auto& inputs = node_unit.Inputs();
  model_builder.AddInitializerToSkip(inputs[1].node_arg.Name());  // pads
  if (inputs.size() > 2) {
    model_builder.AddInitializerToSkip(inputs[2].node_arg.Name());  // constant_value
  }
}

Status PadOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper = model_builder.GetShaper();
  const auto& operand_indices = model_builder.GetOperandIndices();
  const auto& operand_types = model_builder.GetOperandTypes();
  const auto& inputs = node_unit.Inputs();
  const auto& outputs = node_unit.Outputs();

  std::vector<uint32_t> input_indices{};

  // `data` input
  const auto& data = inputs[0].node_arg.Name();
  input_indices.push_back(operand_indices.at(data));

  // `pads` input
  // convert from [begin_1, begin_2, ..., end_1, end_2, ...] to [begin_1, end_1, begin_2, end_2, ...]
  // convert from int64_t to int32_t
  const auto& data_shape = shaper[data];
  const uint32_t data_rank = SafeInt<uint32_t>(data_shape.size());

  const auto& pads = inputs[1].node_arg.Name();
  const auto* pads_initializer = model_builder.GetConstantInitializer(pads);
  ORT_RETURN_IF_NOT(pads_initializer, "pads must be a constant");

  std::vector<uint8_t> pads_initializer_raw_data{};
  ORT_RETURN_IF_ERROR(utils::UnpackInitializerData(*pads_initializer, pads_initializer_raw_data));
  // assume pads_initializer has int64 data, per ONNX spec
  ORT_RETURN_IF_NOT(pads_initializer_raw_data.size() == 2 * data_rank * sizeof(int64_t),
                    "Expected pads initializer size in bytes: ", 2 * data_rank * sizeof(int64_t),
                    ", actual: ", pads_initializer_raw_data.size());

  std::vector<int32_t> converted_pads_data{};
  converted_pads_data.reserve(2 * data_rank);

  auto copy_and_convert = [](const void* raw_i64_src,
                             std::back_insert_iterator<decltype(converted_pads_data)> i32_dst) {
    int64_t i64;
    memcpy(&i64, raw_i64_src, sizeof(i64));
    *i32_dst = SafeInt<int32_t>(i64);
  };

  for (size_t i = 0; i < data_rank; ++i) {
    copy_and_convert(&pads_initializer_raw_data[i * sizeof(int64_t)],
                     std::back_inserter(converted_pads_data));

    copy_and_convert(&pads_initializer_raw_data[(i + data_rank) * sizeof(int64_t)],
                     std::back_inserter(converted_pads_data));
  }

  const Shape converted_pads_shape{data_rank, 2};
  const OperandType converted_pads_operand_type{Type::TENSOR_INT32, converted_pads_shape};
  ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(pads, converted_pads_data.data(),
                                                                      converted_pads_operand_type));
  input_indices.push_back(operand_indices.at(pads));

  // `constant_value` input
  float pad_value = 0.0f;
  if (inputs.size() > 2 && inputs[2].node_arg.Exists()) {
    const auto& constant_value = inputs[2].node_arg.Name();
    const auto* constant_value_initializer = model_builder.GetConstantInitializer(constant_value);
    ORT_RETURN_IF_NOT(constant_value_initializer, "constant_value must be a constant");

    std::vector<uint8_t> pad_value_raw_data{};
    ORT_RETURN_IF_ERROR(utils::UnpackInitializerData(*constant_value_initializer, pad_value_raw_data));
    // assume constant_value_initializer has float data
    // ONNX spec says it matches `data` input type, and op support checker limits that to float
    ORT_RETURN_IF_NOT(pad_value_raw_data.size() == sizeof(float),
                      "Expected constant_value initializer size in bytes: ", sizeof(float),
                      ", actual size: ", pad_value_raw_data.size());
    memcpy(&pad_value, pad_value_raw_data.data(), sizeof(float));
  }

  ADD_SCALAR_OPERAND(model_builder, input_indices, pad_value);

  const auto& output = outputs[0].node_arg.Name();

  ORT_RETURN_IF_ERROR(shaper.Pad(data, converted_pads_data, output));

  const OperandType output_operand_type{operand_types.at(data).type, shaper[output]};
  const auto op_code = ANEURALNETWORKS_PAD_V2;

  return model_builder.AddOperation(op_code, input_indices, {output}, {output_operand_type});
}

#pragma endregion op_pad

/* #pragma region CreateGetOpBuilders

// The reason we use macros to create OpBuilders is for easy exclusion in build if certain op(s) are not used
// such that we can reduce binary size.
// This is for multiple ops share the same OpBuilder, we only need create one for all of them
#define NNAPI_EP_ADD_SHARED_OP_BUILDER(OP_TYPE, BUILDER_NAME) \
  BUILDER_NAME::CreateSharedOpBuilder(OP_TYPE, op_registrations);

// This is for ops with dedicated OpBuilder
#define NNAPI_EP_ADD_SINGLE_OP_BUILDER(OP_TYPE, BUILDER_NAME)                                 \
  do {                                                                                        \
    op_registrations.builders.push_back(std::make_unique<BUILDER_NAME>());                    \
    op_registrations.op_builder_map.emplace(OP_TYPE, op_registrations.builders.back().get()); \
  } while (0)

static OpBuilderRegistrations CreateOpBuilderRegistrations() {
  OpBuilderRegistrations op_registrations;

  // Builders handle a single op
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("BatchNormalization", BatchNormalizationOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Cast", CastOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Clip", ClipOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Concat", ConcatOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("DepthToSpace", DepthToSpaceOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("DequantizeLinear", DequantizeLinearOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Elu", EluOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Flatten", FlattenOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Gather", GatherOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Identity", IdentityOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("LRN", LRNOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Pad", PadOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("QuantizeLinear", QuantizeLinearOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Relu", ReluOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Reshape", ReshapeOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Resize", ResizeOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Slice", SliceOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Softmax", SoftMaxOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Squeeze", SqueezeOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Transpose", TransposeOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Unsqueeze", UnsqueezeOpBuilder);

  // Builders shared among similar ops
  {
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Add", BinaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Div", BinaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Mul", BinaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Pow", BinaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("PRelu", BinaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("QLinearAdd", BinaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("QLinearMul", BinaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Sub", BinaryOpBuilder);
  }

  {
    NNAPI_EP_ADD_SHARED_OP_BUILDER("AveragePool", PoolOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("GlobalAveragePool", PoolOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("GlobalMaxPool", PoolOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("MaxPool", PoolOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("QLinearAveragePool", PoolOpBuilder);
  }

  {
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Conv", ConvOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("QLinearConv", ConvOpBuilder);
  }

  {
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Gemm", GemmOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("MatMul", GemmOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("QLinearMatMul", GemmOpBuilder);
  }

  {
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Abs", UnaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Exp", UnaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Floor", UnaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Log", UnaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Neg", UnaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("QLinearSigmoid", UnaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Sigmoid", UnaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Sin", UnaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Sqrt", UnaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Tanh", UnaryOpBuilder);
  }

  {
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Max", MinMaxOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Min", MinMaxOpBuilder);
  }

  return op_registrations;
}

const std::unordered_map<std::string, const IOpBuilder*>& GetOpBuilders() {
  static const OpBuilderRegistrations op_registrations = CreateOpBuilderRegistrations();
  return op_registrations.op_builder_map;
}

#pragma endregion */

}  // namespace nnapi
}  // namespace onnxruntime
