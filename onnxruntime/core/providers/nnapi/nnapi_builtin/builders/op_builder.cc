// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/common/logging/logging.h>
#include <onnx/onnx_pb.h>

#include "core/common/safeint.h"
#include "helper.h"
#include "model_builder.h"
#include "node_attr_helper.h"
#include "op_builder.h"

namespace onnxruntime {
namespace nnapi {

using namespace android::nn::wrapper;
using std::vector;

#pragma region helpers

const int64_t* GetTensorInt64Data(const ONNX_NAMESPACE::TensorProto& tensor) {
  return tensor.int64_data().empty()
             ? reinterpret_cast<const int64_t*>(tensor.raw_data().data())
             : tensor.int64_data().data();
}

const float* GetTensorFloatData(const ONNX_NAMESPACE::TensorProto& tensor) {
  return tensor.float_data().empty()
             ? reinterpret_cast<const float*>(tensor.raw_data().data())
             : tensor.float_data().data();
}

void AddBinaryOperator(int32_t op_type,
                       ModelBuilder& model_builder,
                       const std::string& input1,
                       const std::string& input2,
                       int32_t fuse_code,
                       const std::string& output) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input1));  // input 1
  input_indices.push_back(operand_indices.at(input2));  // input 2
  input_indices.push_back(model_builder.AddOperandFromScalar(fuse_code));
  shaper.Eltwise(input1, input2, output);
  const OperandType output_operand_type(operand_types.at(input1).type, shaper[output]);
  model_builder.AddOperation(op_type, input_indices, {output}, {output_operand_type});
}

void AddPoolOperator(int32_t op_type,
                     ModelBuilder& model_builder,
                     const std::string& input,
                     const vector<int32_t>& onnx_pads,
                     const vector<int32_t>& onnx_strides,
                     const vector<int32_t>& kernel_shape,
                     int32_t fuse_code,
                     const std::string& output) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  bool use_nchw = model_builder.UseNCHW();

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  input_indices.push_back(model_builder.AddOperandFromScalar(onnx_pads[1]));
  input_indices.push_back(model_builder.AddOperandFromScalar(onnx_pads[3]));
  input_indices.push_back(model_builder.AddOperandFromScalar(onnx_pads[0]));
  input_indices.push_back(model_builder.AddOperandFromScalar(onnx_pads[2]));
  input_indices.push_back(model_builder.AddOperandFromScalar(onnx_strides[1]));
  input_indices.push_back(model_builder.AddOperandFromScalar(onnx_strides[0]));
  input_indices.push_back(model_builder.AddOperandFromScalar(kernel_shape[1]));
  input_indices.push_back(model_builder.AddOperandFromScalar(kernel_shape[0]));
  input_indices.push_back(model_builder.AddOperandFromScalar(fuse_code));
  input_indices.push_back(model_builder.AddOperandFromScalar(use_nchw));

  shaper.Pool(input,
              onnx_pads, onnx_strides, kernel_shape,
              use_nchw,
              output);
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);
  model_builder.AddOperation(op_type, input_indices, {output}, {output_operand_type});
}

int GetType(const ONNX_NAMESPACE::ModelProto& model_proto,
            const std::string& name) {
  int invalid_type = ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
  for (const auto& input : model_proto.graph().input()) {
    if (input.name() != name)
      continue;

    return input.type().tensor_type().elem_type();
  }

  for (const auto& value_info : model_proto.graph().value_info()) {
    if (value_info.name() != name)
      continue;

    if (!value_info.has_type()) {
      return invalid_type;
    } else if (!value_info.type().has_tensor_type()) {
      return invalid_type;
    }

    return value_info.type().tensor_type().elem_type();
  }

  return invalid_type;
}

Shaper::Shape GetShape(const ONNX_NAMESPACE::ModelProto& model_proto,
                       const std::string& name) {
  Shaper::Shape empty_shape;
  for (const auto& input : model_proto.graph().input()) {
    if (input.name() != name)
      continue;

    Shaper::Shape shape;
    for (const auto& dim : input.type().tensor_type().shape().dim())
      shape.push_back(dim.dim_value());

    return shape;
  }

  for (const auto& tensor : model_proto.graph().initializer()) {
    if (tensor.name() != name)
      continue;

    Shaper::Shape shape;
    for (auto dim : tensor.dims())
      shape.push_back(SafeInt<uint32_t>(dim));

    return shape;
  }

  for (const auto& value_info : model_proto.graph().value_info()) {
    if (value_info.name() != name)
      continue;

    if (!value_info.has_type()) {
      return empty_shape;
    } else if (!value_info.type().has_tensor_type()) {
      return empty_shape;
    } else if (!value_info.type().tensor_type().has_shape()) {
      return empty_shape;
    } else if (value_info.type().tensor_type().shape().dim_size() == 0) {
      return empty_shape;
    }

    Shaper::Shape shape;
    for (const auto& dim : value_info.type().tensor_type().shape().dim())
      shape.push_back(dim.dim_value());

    return shape;
  }

  return empty_shape;
}

enum DataLayout {
  L_NCHW = 0,
  L_1230 = 1,
};

// TODO, replace this with more efficient code in optimizers
uint32_t AddInitializerInNewLayout(ModelBuilder& model_builder,
                                   const std::string& name,
                                   DataLayout new_layout) {
  const auto& tensor = model_builder.GetInitializerTensors().at(name);
  ModelBuilder::Shape shape;
  for (auto dim : tensor.dims())
    shape.push_back(SafeInt<uint32_t>(dim));

  ORT_ENFORCE(shape.size() == 4, "The initializer is not 4D: " +
                                     name + " actual dim " +
                                     std::to_string(shape.size()));

  // TODO support other data types
  Type type;
  if (tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    type = Type::TENSOR_FLOAT32;
  } else {
    ORT_THROW("The initializer of graph doesn't have valid type: " + name);
  }

  auto out_t = shape[0], in_t = shape[1],
       h_t = shape[2], w_t = shape[3];
  ModelBuilder::Shape dest_shape;
  if (new_layout == L_NCHW)
    dest_shape = {out_t, h_t, w_t, in_t};  // L_NCHW
  else
    dest_shape = {in_t, h_t, w_t, out_t};  // L_1230 for depthwise conv weight

  const float* src = GetTensorFloatData(tensor);
  float* buffer = new float[Product(shape)];
  const OperandType operandType(type, dest_shape);
  for (uint32_t out = 0; out < out_t; out++) {
    for (uint32_t in = 0; in < in_t; in++) {
      for (uint32_t h = 0; h < h_t; h++) {
        for (uint32_t w = 0; w < w_t; w++) {
          auto onnx_idx = out * in_t * h_t * w_t +
                          in * h_t * w_t +
                          h * w_t +
                          w;

          uint32_t nnapi_idx;
          if (new_layout == L_NCHW) {  // L_NCHW
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

          buffer[nnapi_idx] = src[onnx_idx];
        }
      }
    }
  }

  auto operand_idx = model_builder.AddOperandFromPersistMemoryBuffer(name, &buffer[0], operandType);
  delete[] buffer;
  return operand_idx;
}

// TODO, replace this with more efficient code in optimizers
uint32_t AddInitializerTransposed(ModelBuilder& model_builder,
                                  const std::string& name) {
  const auto& tensor = model_builder.GetInitializerTensors().at(name);
  ModelBuilder::Shape shape;
  for (auto dim : tensor.dims())
    shape.push_back(SafeInt<uint32_t>(dim));

  ORT_ENFORCE(shape.size() == 2, "The initializer is not 2D: " +
                                     name + " actual dim " +
                                     std::to_string(shape.size()));

  // TODO support other data types
  Type type;
  if (tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    type = Type::TENSOR_FLOAT32;
  } else {
    ORT_THROW("The initializer of graph doesn't have valid type: " + name);
  }

  auto x_t = shape[0], y_t = shape[1];
  ModelBuilder::Shape dest_shape = {y_t, x_t};
  const OperandType operandType(type, dest_shape);
  const float* src = GetTensorFloatData(tensor);
  float* buffer = new float[Product(shape)];
  for (uint32_t x = 0; x < x_t; x++) {
    for (uint32_t y = 0; y < y_t; y++) {
      buffer[y * x_t + x] = src[x * y_t + y];
    }
  }
  auto operand_idx = model_builder.AddOperandFromPersistMemoryBuffer(name, &buffer[0], operandType);

  delete[] buffer;
  return operand_idx;
}

#pragma endregion helpers

#pragma region op_base

class BaseOpBuilder : public IOpBuilder {
 public:
  virtual ~BaseOpBuilder() = default;
  virtual void AddInitializersToSkip(ModelBuilder& /* model_builder */,
                                     const ONNX_NAMESPACE::NodeProto& /* node */) override {}

  bool IsOpSupported(ModelBuilder& model_builder,
                     const ONNX_NAMESPACE::NodeProto& node) override final;

  void AddToModelBuilder(ModelBuilder& model_builder,
                         const ONNX_NAMESPACE::NodeProto& node) override final;

 protected:
  virtual bool IsOpSupportedImpl(
      ModelBuilder& model_builder, const ONNX_NAMESPACE::NodeProto& node);

  virtual int32_t GetMinSupportedSdkVer(
      ModelBuilder& /* model_builder */,
      const ONNX_NAMESPACE::NodeProto& /* node */) const { return 27; }

  virtual bool HasSupportedInputs(
      ModelBuilder& model_builder, const ONNX_NAMESPACE::NodeProto& node);

  virtual void AddToModelBuilderImpl(
      ModelBuilder& model_builder, const ONNX_NAMESPACE::NodeProto& node);
};

bool BaseOpBuilder::IsOpSupported(ModelBuilder& model_builder,
                                  const ONNX_NAMESPACE::NodeProto& node) {
#ifdef __ANDROID__
  int32_t android_sdk_ver = model_builder.GetAndroidSdkVer();
  int32_t required_sdk_ver = GetMinSupportedSdkVer(model_builder, node);
  if (required_sdk_ver > android_sdk_ver) {
    LOGS_DEFAULT(VERBOSE) << "Current Android API level [" << android_sdk_ver
                          << "], Operator [" << node.op_type()
                          << "] is only supported on API >" << required_sdk_ver;
    return false;
  }
#endif

  if (!HasSupportedInputs(model_builder, node))
    return false;

  return IsOpSupportedImpl(model_builder, node);
}  // namespace nnapi

bool BaseOpBuilder::HasSupportedInputs(
    ModelBuilder& model_builder,
    const ONNX_NAMESPACE::NodeProto& node) {
  // We only check the type of input 0 by default
  // specific op builder can override this
  auto input_type = GetType(model_builder.GetOnnxModel(), node.input(0));
  if (input_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    LOGS_DEFAULT(VERBOSE) << "[" << node.op_type()
                          << "] Input type: [" << input_type
                          << "] is not supported for now";
    return false;
  }

  return true;
}

bool BaseOpBuilder::IsOpSupportedImpl(ModelBuilder& /* model_builder */,
                                      const ONNX_NAMESPACE::NodeProto& /* node */) {
  return true;
}

void BaseOpBuilder::AddToModelBuilder(ModelBuilder& model_builder,
                                      const ONNX_NAMESPACE::NodeProto& node) {
  ORT_ENFORCE(IsOpSupported(model_builder, node),
              "Unsupported operator " + node.op_type());

  AddToModelBuilderImpl(model_builder, node);
  LOGS_DEFAULT(VERBOSE) << "Operator name: [" << node.name()
                        << "] type: [" << node.op_type() << "] was added";
}

void BaseOpBuilder::AddToModelBuilderImpl(ModelBuilder& /* model_builder */,
                                          const ONNX_NAMESPACE::NodeProto& node) {
  ORT_NOT_IMPLEMENTED("Unsupported operator " + node.op_type());
}

#pragma endregion op_base

#pragma region op_binary

class BinaryOpBuilder : public BaseOpBuilder {
 private:
  int32_t GetMinSupportedSdkVer(ModelBuilder& model_builder,
                                const ONNX_NAMESPACE::NodeProto& node) const override;

 private:
  void AddToModelBuilderImpl(ModelBuilder& model_builder,
                             const ONNX_NAMESPACE::NodeProto& node) override;
};

int32_t BinaryOpBuilder::GetMinSupportedSdkVer(ModelBuilder& /* model_builder */,
                                               const ONNX_NAMESPACE::NodeProto& node) const {
  const auto& op(node.op_type());
  if (op == "Sub" || op == "Div") {
    return 28;
  }

  return 27;
}

void BinaryOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                            const ONNX_NAMESPACE::NodeProto& node) {
  const auto& op(node.op_type());
  int32_t op_code;
  if (op == "Add")
    op_code = ANEURALNETWORKS_ADD;
  else if (op == "Sub")
    op_code = ANEURALNETWORKS_SUB;
  else if (op == "Mul")
    op_code = ANEURALNETWORKS_MUL;
  else if (op == "Div")
    op_code = ANEURALNETWORKS_DIV;
  else {
    ORT_THROW("UnaryOpBuilder, unknown op: " + op);
  }
  const auto& input1 = node.input(0);
  const auto& input2 = node.input(1);
  const auto& output = node.output(0);
  int32_t fuse_code = model_builder.FindActivation(output);
  AddBinaryOperator(op_code, model_builder,
                    input1, input2, fuse_code, output);
}

#pragma endregion

#pragma region op_relu

class ReluOpBuilder : public BaseOpBuilder {
 private:
  void AddToModelBuilderImpl(ModelBuilder& model_builder,
                             const ONNX_NAMESPACE::NodeProto& node) override;
};

void ReluOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                          const ONNX_NAMESPACE::NodeProto& node) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  const auto& input = node.input(0);
  const auto& output = node.output(0);
  shaper.Identity(input, output);
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);

  // skip this relu if it is some op's fuse output
  if (Contains(model_builder.GetFusedActivations(), node.name())) {
    model_builder.RegisterOperand(output, operand_indices.at(input), output_operand_type);
  } else {
    std::vector<uint32_t> input_indices;
    input_indices.push_back(operand_indices.at(input));
    model_builder.AddOperation(ANEURALNETWORKS_RELU, input_indices, {output}, {output_operand_type});
  }
}

#pragma endregion op_relu

#pragma region op_transpose

class TransposeOpBuilder : public BaseOpBuilder {
 private:
  bool IsOpSupportedImpl(
      ModelBuilder& model_builder,
      const ONNX_NAMESPACE::NodeProto& node) override;

  int32_t GetMinSupportedSdkVer(ModelBuilder& /* model_builder */,
                                const ONNX_NAMESPACE::NodeProto& /* node */) const override {
    return 28;
  }

  void AddToModelBuilderImpl(ModelBuilder& model_builder,
                             const ONNX_NAMESPACE::NodeProto& node) override;
};

bool TransposeOpBuilder::IsOpSupportedImpl(ModelBuilder& model_builder,
                                           const ONNX_NAMESPACE::NodeProto& node) {
  const auto input_size = GetShape(model_builder.GetOnnxModel(), node.input(0)).size();
  if (input_size > 4) {
    LOGS_DEFAULT(VERBOSE) << "Transpose only supports up to 4d shape, input is "
                          << input_size << "d shape";
    return false;
  }

  return true;
}

void TransposeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                               const ONNX_NAMESPACE::NodeProto& node) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  NodeAttrHelper helper(node);

  const auto& input = node.input(0);
  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));  // input

  vector<int32_t> perm = helper.Get("perm", vector<int32_t>());
  auto input_dims = shaper[input].size();
  if (perm.empty()) {
    for (int32_t i = input_dims - 1; i >= 0; i--)
      perm.push_back(i);
  }

  ModelBuilder::Shape perm_dimen = {SafeInt<uint32_t>(input_dims)};
  std::string perm_name = model_builder.GetUniqueName(node.name() + input + "perm");
  OperandType perm_operand_type(Type::TENSOR_INT32, perm_dimen);
  uint32_t perm_idx = model_builder.AddOperandFromPersistMemoryBuffer(perm_name, perm.data(), perm_operand_type);
  input_indices.push_back(perm_idx);

  const auto& output = node.output(0);
  shaper.Transpose(input, perm, output);
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);
  model_builder.AddOperation(ANEURALNETWORKS_TRANSPOSE, input_indices, {output}, {output_operand_type});
}

#pragma endregion op_transpose

#pragma region op_reshape

class ReshapeOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder,
                             const ONNX_NAMESPACE::NodeProto& node) override;

 private:
  bool IsOpSupportedImpl(
      ModelBuilder& model_builder,
      const ONNX_NAMESPACE::NodeProto& node) override;
  void AddToModelBuilderImpl(ModelBuilder& model_builder,
                             const ONNX_NAMESPACE::NodeProto& node) override;
};

void ReshapeOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder,
                                             const ONNX_NAMESPACE::NodeProto& node) {
  model_builder.AddInitializerToSkip(node.input(1));
}

bool ReshapeOpBuilder::IsOpSupportedImpl(
    ModelBuilder& model_builder,
    const ONNX_NAMESPACE::NodeProto& node) {
  const auto& initializers(model_builder.GetInitializerTensors());
  if (!Contains(initializers, node.input(1))) {
    LOGS_DEFAULT(VERBOSE) << "New shape of reshape must be known";
    return false;
  }

  const auto input_size = GetShape(model_builder.GetOnnxModel(), node.input(0)).size();
  if (input_size > 4) {
    LOGS_DEFAULT(VERBOSE) << "Reshape only supports up to 4d shape, input is "
                          << input_size << "d shape";
    return false;
  }

  const auto& shape_tensor = initializers.at(node.input(1));
  const int64_t* rawShape = GetTensorInt64Data(shape_tensor);
  const auto size = SafeInt<uint32_t>(shape_tensor.dims()[0]);
  const auto input_shape = GetShape(model_builder.GetOnnxModel(), node.input(0));

  for (uint32_t i = 0; i < size; i++) {
    // NNAPI reshape does not support 0 as dimension
    if (rawShape[i] == 0 && i < input_shape.size() && input_shape[i] == 0) {
      LOGS_DEFAULT(VERBOSE)
          << "Reshape doesn't suppport 0 reshape dimension on a dynamic dimension";
      return false;
    }
  }

  return true;
}

void ReshapeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                             const ONNX_NAMESPACE::NodeProto& node) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& initializers(model_builder.GetInitializerTensors());

  const auto& input = node.input(0);
  const auto& output = node.output(0);
  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));  // input

  const auto& shape_tensor = initializers.at(node.input(1));
  const int64_t* rawShape = GetTensorInt64Data(shape_tensor);
  const auto size = SafeInt<uint32_t>(shape_tensor.dims()[0]);

  ModelBuilder::Shape input_shape = shaper[input];
  std::vector<int32_t> shape(size);
  for (uint32_t i = 0; i < size; i++) {
    int32_t dim = SafeInt<int32_t>(rawShape[i]);
    // NNAPI reshape does not support 0 as dimension
    shape[i] = dim == 0 ? input_shape[i] : dim;
  }

  ModelBuilder::Shape shape_dimen = {size};
  std::string shape_name = model_builder.GetUniqueName(node.name() + input + "newshape");
  OperandType shape_operand_type(Type::TENSOR_INT32, shape_dimen);
  uint32_t shape_idx = model_builder.AddOperandFromPersistMemoryBuffer(shape_name, shape.data(), shape_operand_type);
  input_indices.push_back(shape_idx);

  shaper.Reshape(input, shape, output);
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);
  model_builder.AddOperation(ANEURALNETWORKS_RESHAPE, input_indices, {output}, {output_operand_type});
}

#pragma endregion op_reshape

#pragma region op_batchnormalization

class BatchNormalizationOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder,
                             const ONNX_NAMESPACE::NodeProto& node) override;

 private:
  bool IsOpSupportedImpl(ModelBuilder& model_builder,
                         const ONNX_NAMESPACE::NodeProto& node) override;
  void AddToModelBuilderImpl(ModelBuilder& model_builder,
                             const ONNX_NAMESPACE::NodeProto& node) override;
};

void BatchNormalizationOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder,
                                                        const ONNX_NAMESPACE::NodeProto& node) {
  // skip everything except input0 for BatchNormalization
  model_builder.AddInitializerToSkip(node.input(1));  // scale
  model_builder.AddInitializerToSkip(node.input(2));  // B
  model_builder.AddInitializerToSkip(node.input(3));  // mean
  model_builder.AddInitializerToSkip(node.input(4));  //var
}

bool BatchNormalizationOpBuilder::IsOpSupportedImpl(
    ModelBuilder& model_builder,
    const ONNX_NAMESPACE::NodeProto& node) {
  if (node.output_size() != 1) {
    LOGS_DEFAULT(VERBOSE) << "Your onnx model may be in training mode, please export "
                             "it in test mode.";
    return false;
  }

  const auto& initializers(model_builder.GetInitializerTensors());
  const auto& scale_name = node.input(1);
  const auto& b_name = node.input(2);
  const auto& mean_name = node.input(3);
  const auto& var_name = node.input(4);
  if (!Contains(initializers, scale_name)) {
    LOGS_DEFAULT(VERBOSE) << "Scale of BN must be known";
    return false;
  }
  if (!Contains(initializers, b_name)) {
    LOGS_DEFAULT(VERBOSE) << "B of BN must be known";
    return false;
  }
  if (!Contains(initializers, mean_name)) {
    LOGS_DEFAULT(VERBOSE) << "Mean of BN must be known";
    return false;
  }
  if (!Contains(initializers, var_name)) {
    LOGS_DEFAULT(VERBOSE) << "Var of BN must be known";
    return false;
  }

  return true;
}

void BatchNormalizationOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                        const ONNX_NAMESPACE::NodeProto& node) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& initializers(model_builder.GetInitializerTensors());
  NodeAttrHelper helper(node);

  // For reshape we are not really doing anything but
  // register a new operand with new shape
  const auto& input = node.input(0);
  const auto& output = node.output(0);

  const auto& scale_tensor = initializers.at(node.input(1));
  const auto& bias_tensor = initializers.at(node.input(2));
  const auto& mean_tensor = initializers.at(node.input(3));
  const auto& var_tensor = initializers.at(node.input(4));
  const auto eps = helper.Get("epsilon", 1e-5f);

  const auto size = SafeInt<uint32_t>(scale_tensor.dims()[0]);
  vector<float> a, b;
  a.reserve(size);
  b.reserve(size);

  const float* scale_data = GetTensorFloatData(scale_tensor);
  const float* bias_data = GetTensorFloatData(bias_tensor);
  const float* mean_data = GetTensorFloatData(mean_tensor);
  const float* var_data = GetTensorFloatData(var_tensor);

  for (int64_t i = 0; i < size; i++) {
    a.push_back(scale_data[i] / sqrt(var_data[i] + eps));
    b.push_back((scale_data[i] * -mean_data[i]) / sqrt(var_data[i] + eps) +
                bias_data[i]);
  }

  const auto tensor_a_name = model_builder.GetUniqueName(node.name() + input + "_imm_a");
  const auto tensor_b_name = model_builder.GetUniqueName(node.name() + input + "_imm_b");
  const auto tensor_imm_product_name = model_builder.GetUniqueName(node.name() + input + "_imm_mul");
  ModelBuilder::Shape tensor_a_dimen;
  if (model_builder.UseNCHW())
    tensor_a_dimen = {size, 1, 1};  // {C, H, W}
  else
    tensor_a_dimen = {size};

  shaper.AddShape(tensor_a_name, tensor_a_dimen);
  shaper.AddShape(tensor_b_name, tensor_a_dimen);
  const OperandType operandType_a(operand_types.at(input).type, tensor_a_dimen);
  model_builder.AddOperandFromPersistMemoryBuffer(tensor_a_name, a.data(), operandType_a);
  const OperandType operandType_b(operand_types.at(input).type, tensor_a_dimen);
  model_builder.AddOperandFromPersistMemoryBuffer(tensor_b_name, b.data(), operandType_b);

  // Mul
  AddBinaryOperator(ANEURALNETWORKS_MUL,
                    model_builder,
                    input, tensor_a_name,
                    ANEURALNETWORKS_FUSED_NONE,
                    tensor_imm_product_name);

  // Add
  int32_t fuse_code = model_builder.FindActivation(output);
  AddBinaryOperator(ANEURALNETWORKS_ADD,
                    model_builder,
                    tensor_imm_product_name, tensor_b_name,
                    fuse_code,
                    output);
}

#pragma endregion op_batchnormalization

#pragma region op_pool

class PoolOpBuilder : public BaseOpBuilder {
 private:
  bool IsOpSupportedImpl(
      ModelBuilder& model_builder,
      const ONNX_NAMESPACE::NodeProto& node) override;

  int32_t GetMinSupportedSdkVer(ModelBuilder& /* model_builder */,
                                const ONNX_NAMESPACE::NodeProto& /* node */) const override {
    return 29;
  }

  void AddToModelBuilderImpl(ModelBuilder& model_builder,
                             const ONNX_NAMESPACE::NodeProto& node) override;
};

bool PoolOpBuilder::IsOpSupportedImpl(ModelBuilder& model_builder,
                                      const ONNX_NAMESPACE::NodeProto& node) {
  const auto& op = node.op_type();
  if (op == "AveragePool" || op == "MaxPool") {
    NodeAttrHelper helper(node);

    const auto count_include_pad = helper.Get("count_include_pad", 0);
    if (count_include_pad == 1) {
      LOGS_DEFAULT(VERBOSE) << "count_include_pad == 1 is not supported";
      return false;
    }

    const auto storage_order = helper.Get("storage_order", 0);
    if (storage_order == 1) {
      LOGS_DEFAULT(VERBOSE) << "storage_order == 1 is not supported";
      return false;
    }

    if (helper.Get("auto_pad", "NOTSET") != "NOTSET") {
      LOGS_DEFAULT(VERBOSE) << "auto_pad is not supported";
      return false;
    }

    if (helper.Get("kernel_shape", std::vector<int32_t>{1, 1}).size() != 2) {
      LOGS_DEFAULT(VERBOSE) << "Only pooling 2d is supported";
      return false;
    }

    if (helper.Get("ceil_mode", 0) == 1) {
      LOGS_DEFAULT(VERBOSE) << "ceil_mode == 1 is not supported for pooling";
      return false;
    }

    if (helper.Get("dilations", std::vector<int32_t>{1, 1}) !=
        std::vector<int32_t>{1, 1}) {
      LOGS_DEFAULT(VERBOSE) << "Dilations of pooling is not supported";
      return false;
    }

    if (node.output_size() != 1) {
      LOGS_DEFAULT(VERBOSE) << "Argmax in maxpooling is not supported";
      return false;
    }
  } else if (op == "GlobalAveragePool" || op == "GlobalMaxPool") {
    const auto input_shape = GetShape(model_builder.GetOnnxModel(), node.input(0));
    if (input_shape.size() > 4) {
      LOGS_DEFAULT(VERBOSE)
          << "GlobalAveragePool/GlobalMaxPool Only rank-4 tensor is supported in "
          << node.input(0) << ", actual dim count " << input_shape.size();
      return false;
    }
  }

  return true;
}

void PoolOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                          const ONNX_NAMESPACE::NodeProto& node) {
  auto& shaper(model_builder.GetShaper());
  NodeAttrHelper helper(node);

  const auto& input = node.input(0);
  const auto& output = node.output(0);
  const auto& op = node.op_type();

  int32_t operationType;
  if (op == "AveragePool" || op == "GlobalAveragePool")
    operationType = ANEURALNETWORKS_AVERAGE_POOL_2D;
  else  // (op == "MaxPool" || op == "GlobalMaxPool")
    operationType = ANEURALNETWORKS_MAX_POOL_2D;

  vector<int32_t> onnx_pads, onnx_strides, kernel_shape;
  if (op == "AveragePool" || op == "MaxPool") {
    kernel_shape = helper.Get("kernel_shape", vector<int32_t>{0, 0});
    onnx_strides = helper.Get("strides", vector<int>{1, 1});
    onnx_pads = helper.Get("pads", vector<int>{0, 0, 0, 0});
  } else {  // (op == "GlobalAveragePool" || op == "GlobalMaxPool")
    onnx_strides = vector<int32_t>{1, 1};
    onnx_pads = vector<int32_t>{0, 0, 0, 0};
    if (model_builder.UseNCHW())
      kernel_shape = vector<int32_t>{static_cast<int32_t>(shaper[input][2]),
                                     static_cast<int32_t>(shaper[input][3])};
    else
      kernel_shape = vector<int32_t>{static_cast<int32_t>(shaper[input][1]),
                                     static_cast<int32_t>(shaper[input][2])};
  }

  int32_t fuse_code = model_builder.FindActivation(output);
  AddPoolOperator(operationType,
                  model_builder,
                  input,
                  onnx_pads, onnx_strides, kernel_shape,
                  fuse_code,
                  output);
}

#pragma endregion op_pool

#pragma region op_conv

class ConvOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder,
                             const ONNX_NAMESPACE::NodeProto& node) override;

 private:
  bool IsOpSupportedImpl(ModelBuilder& model_builder,
                         const ONNX_NAMESPACE::NodeProto& node) override;
  void AddToModelBuilderImpl(ModelBuilder& model_builder,
                             const ONNX_NAMESPACE::NodeProto& node) override;
};

void ConvOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder,
                                          const ONNX_NAMESPACE::NodeProto& node) {
  // skip the weight for conv as we need to transpose
  model_builder.AddInitializerToSkip(node.input(1));
}

bool ConvOpBuilder::IsOpSupportedImpl(
    ModelBuilder& model_builder,
    const ONNX_NAMESPACE::NodeProto& node) {
  NodeAttrHelper helper(node);
  if (helper.Get("auto_pad", "NOTSET") != "NOTSET") {
    LOGS_DEFAULT(VERBOSE) << "SAME_LOWER auto_pad is not supported";
    return false;
  }

  const auto group = helper.Get("group", 1);
  const auto weight_name = node.input(1);
  if (Contains(model_builder.GetInitializerTensors(), weight_name)) {
    const auto& tensor = model_builder.GetInitializerTensors().at(weight_name);
    if (tensor.dims().size() != 4) {
      LOGS_DEFAULT(VERBOSE) << "Only conv 2d is supported.";
      return false;
    }
    if (group != 1 && tensor.dims()[1] != 1) {
      LOGS_DEFAULT(VERBOSE) << "group != 1 is not supported";
      return false;
    }
  } else {
    LOGS_DEFAULT(VERBOSE) << "The weight of convolution must be known";
    return false;
  }

  return true;
}

void ConvOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                          const ONNX_NAMESPACE::NodeProto& node) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& initializers(model_builder.GetInitializerTensors());
  NodeAttrHelper helper(node);
  bool use_nchw = model_builder.UseNCHW();

  // onnx strides are in the order height, width
  // while nnapi strides are in the order width, height
  const auto onnx_strides = helper.Get("strides", vector<int>{1, 1});

  // onnx pads are in the order top, left, bottom, right
  // while nnapi pads is in the order left, right, top, bottom
  const auto onnx_pads = helper.Get("pads", vector<int>{0, 0, 0, 0});

  // onnx dilations is in the order height, width
  // while nnapi dilations are in the order width, height
  const auto onnx_dilations = helper.Get("dilations", vector<int>{1, 1});
  const auto group = helper.Get("group", 1);

  const auto& input = node.input(0);
  const auto& weight = node.input(1);
  const auto& output = node.output(0);

  bool conv2d = (group == 1);
  const auto& weight_tensor = initializers.at(weight);
  bool depthwise_conv2d = (weight_tensor.dims()[1] == 1);

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));

  if (conv2d) {
    input_indices.push_back(AddInitializerInNewLayout(
        model_builder, weight, L_NCHW));
  } else {  // depthwise_conv2d
    input_indices.push_back(AddInitializerInNewLayout(
        model_builder, weight, L_1230));
  }

  bool hasBias = (node.input_size() >= 3);
  std::string bias = hasBias ? node.input(2) : weight + "_bias";

  uint32_t bias_idx_val;
  if (hasBias) {
    bias_idx_val = operand_indices.at(bias);
  } else {
    const auto weight_dimen = shaper[weight];
    ModelBuilder::Shape bias_dimen;
    if (conv2d)
      bias_dimen = {weight_dimen[0]};
    else
      bias_dimen = {weight_dimen[3]};

    const auto& weight_type = operand_types.at(weight).type;
    if (weight_type == Type::TENSOR_FLOAT32) {
      float buffer[bias_dimen[0]];
      for (uint32_t i = 0; i < bias_dimen[0]; i++) {
        buffer[i] = 0.f;
      }
      OperandType operandType(Type::TENSOR_FLOAT32, bias_dimen);
      bias_idx_val = model_builder.AddOperandFromPersistMemoryBuffer(
          bias, &buffer[0], operandType);
    } else {
      ORT_THROW("Unknown weight type " + TypeToStr(weight_type));
    }
  }

  input_indices.push_back(bias_idx_val);
  input_indices.push_back(model_builder.AddOperandFromScalar(onnx_pads[1]));
  input_indices.push_back(model_builder.AddOperandFromScalar(onnx_pads[3]));
  input_indices.push_back(model_builder.AddOperandFromScalar(onnx_pads[0]));
  input_indices.push_back(model_builder.AddOperandFromScalar(onnx_pads[2]));
  input_indices.push_back(model_builder.AddOperandFromScalar(onnx_strides[1]));
  input_indices.push_back(model_builder.AddOperandFromScalar(onnx_strides[0]));
  if (!conv2d && depthwise_conv2d) {
    int32_t depthwiseMultiplier = shaper[weight][3] / group;
    input_indices.push_back(model_builder.AddOperandFromScalar(depthwiseMultiplier));
  }
  int32_t fuse_code = model_builder.FindActivation(output);
  input_indices.push_back(model_builder.AddOperandFromScalar(fuse_code));
  // TODO support API 27
  input_indices.push_back(model_builder.AddOperandFromScalar(use_nchw));
  input_indices.push_back(model_builder.AddOperandFromScalar(onnx_dilations[1]));
  input_indices.push_back(model_builder.AddOperandFromScalar(onnx_dilations[0]));

  int32_t operationCode;
  if (conv2d) {
    operationCode = ANEURALNETWORKS_CONV_2D;
    shaper.Conv(input, weight,
                onnx_pads, onnx_strides, onnx_dilations,
                use_nchw,
                output);
  } else {  // depthwise_conv2d
    operationCode = ANEURALNETWORKS_DEPTHWISE_CONV_2D;
    shaper.DepthwiseConv(input, weight,
                         onnx_pads, onnx_strides, onnx_dilations,
                         use_nchw,
                         output);
  }

  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);
  model_builder.AddOperation(operationCode, input_indices, {output}, {output_operand_type});
}

#pragma endregion op_conv

#pragma region op_cast

class CastOpBuilder : public BaseOpBuilder {
 private:
  bool IsOpSupportedImpl(
      ModelBuilder& model_builder,
      const ONNX_NAMESPACE::NodeProto& node) override;

  int32_t GetMinSupportedSdkVer(ModelBuilder& /* model_builder */,
                                const ONNX_NAMESPACE::NodeProto& /* node */) const override {
    return 29;
  }

  void AddToModelBuilderImpl(ModelBuilder& model_builder,
                             const ONNX_NAMESPACE::NodeProto& node) override;
};

bool CastOpBuilder::IsOpSupportedImpl(
    ModelBuilder& /* model_builder */,
    const ONNX_NAMESPACE::NodeProto& node) {
  NodeAttrHelper helper(node);
  auto to = helper.Get("to", 0);
  if (to != ONNX_NAMESPACE::TensorProto::FLOAT &&
      to != ONNX_NAMESPACE::TensorProto::INT32) {
    LOGS_DEFAULT(VERBOSE) << "[Cast] Only support cast to int32 or float";
    return false;
  }

  return true;
}

void CastOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                          const ONNX_NAMESPACE::NodeProto& node) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  NodeAttrHelper helper(node);

  const auto& input = node.input(0);
  const auto& output = node.output(0);
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
      ORT_THROW("Invalid cast to type: " +
                std::to_string(to));
  }

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  shaper.Identity(input, output);
  const OperandType output_operand_type(type, shaper[output]);
  model_builder.AddOperation(ANEURALNETWORKS_CAST, input_indices, {output}, {output_operand_type});
}

#pragma endregion

#pragma region op_softmax

class SoftMaxOpBuilder : public BaseOpBuilder {
 private:
  bool IsOpSupportedImpl(
      ModelBuilder& model_builder,
      const ONNX_NAMESPACE::NodeProto& node) override;

  int32_t GetMinSupportedSdkVer(ModelBuilder& /* model_builder */,
                                const ONNX_NAMESPACE::NodeProto& /* node */) const override {
    return 29;
  }

  void AddToModelBuilderImpl(ModelBuilder& model_builder,
                             const ONNX_NAMESPACE::NodeProto& node) override;
};

bool SoftMaxOpBuilder::IsOpSupportedImpl(
    ModelBuilder& model_builder,
    const ONNX_NAMESPACE::NodeProto& node) {
  const auto input_size = GetShape(model_builder.GetOnnxModel(), node.input(0)).size();
  if (input_size != 2 && input_size != 4) {
    LOGS_DEFAULT(VERBOSE) << "SoftMax only support 2d/4d shape, input is "
                          << input_size << "d shape";
    return false;
  }
  return true;
}

void SoftMaxOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                             const ONNX_NAMESPACE::NodeProto& node) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  NodeAttrHelper helper(node);

  const auto& input = node.input(0);
  const auto& output = node.output(0);
  float beta = 1.f;
  int32_t axis = helper.Get("axis", 1);
  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  input_indices.push_back(model_builder.AddOperandFromScalar(beta));
  input_indices.push_back(model_builder.AddOperandFromScalar(axis));

  shaper.Identity(input, output);
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);
  model_builder.AddOperation(ANEURALNETWORKS_SOFTMAX, input_indices, {output}, {output_operand_type});
}

#pragma endregion

#pragma region op_identity

class IdentityOpBuilder : public BaseOpBuilder {
 private:
  void AddToModelBuilderImpl(ModelBuilder& model_builder,
                             const ONNX_NAMESPACE::NodeProto& node) override;
};

void IdentityOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                              const ONNX_NAMESPACE::NodeProto& node) {
  // Identity is not really going to do anything
  // Just register the dimension and type, with same index and new name
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  const auto& input = node.input(0);
  const auto& output = node.output(0);
  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));  // input

  shaper.Identity(input, output);
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);
  model_builder.RegisterOperand(output, operand_indices.at(input), output_operand_type);
}

#pragma endregion

#pragma region op_gemm

class GemmOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder,
                             const ONNX_NAMESPACE::NodeProto& node) override;

 private:
  bool IsOpSupportedImpl(ModelBuilder& model_builder,
                         const ONNX_NAMESPACE::NodeProto& node) override;

  void AddToModelBuilderImpl(ModelBuilder& model_builder,
                             const ONNX_NAMESPACE::NodeProto& node) override;
};

bool GemmOpBuilder::IsOpSupportedImpl(
    ModelBuilder& model_builder,
    const ONNX_NAMESPACE::NodeProto& node) {
  const auto& op = node.op_type();
  const auto& initializers(model_builder.GetInitializerTensors());

  if (op == "MatMul") {  // Only support A*B B is an initializer
    if (!Contains(initializers, node.input(1))) {
      LOGS_DEFAULT(VERBOSE) << "B of MatMul must be known";
      return false;
    }
  } else if (op == "Gemm") {
    // Only support
    // 1. A*B'+C
    // 2. A*B+C and B is an initializer
    NodeAttrHelper helper(node);
    const auto transA = helper.Get("transA", 0);
    const auto transB = helper.Get("transB", 0);
    const auto alpha = helper.Get("alpha", 1.0f);
    const auto beta = helper.Get("beta", 1.0f);

    if (!(transA == 0 && alpha == 1.f && beta == 1.f)) {
      LOGS_DEFAULT(VERBOSE) << "Only transA == 0, alpha == 1.0 "
                            << "and beta == 1.0 is supported.";
      return false;
    }

    if (transB == 0 && !Contains(initializers, node.input(1))) {
      LOGS_DEFAULT(VERBOSE) << "B of Gemm must be known if transB != 1";
      return false;
    }

    if (node.input_size() == 3) {
      const auto b_shape = GetShape(model_builder.GetOnnxModel(), node.input(1));
      const auto c_shape = GetShape(model_builder.GetOnnxModel(), node.input(2));
      if (c_shape.size() != 1 ||
          c_shape[0] != (transB == 0 ? b_shape[1] : b_shape[0])) {
        LOGS_DEFAULT(VERBOSE) << "C of Gemm must be a vector of b_shape[0]";
        return false;
      }
    }
  }

  return true;
}

void GemmOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder,
                                          const ONNX_NAMESPACE::NodeProto& node) {
  const auto& op = node.op_type();
  if (op == "MatMul") {
    model_builder.AddInitializerToSkip(node.input(1));
  } else if (op == "Gemm") {
    NodeAttrHelper helper(node);
    const auto transB = helper.Get("transB", 0);
    if (transB == 0)
      model_builder.AddInitializerToSkip(node.input(1));
  }
}

void GemmOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                          const ONNX_NAMESPACE::NodeProto& node) {
  const auto& op = node.op_type();
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  NodeAttrHelper helper(node);

  const auto& input1 = node.input(0);
  const auto& input2 = node.input(1);
  const auto& output = node.output(0);
  const auto transB = helper.Get("transB", 0);

  uint32_t input_2_idx;
  if (transB == 0) {
    input_2_idx = AddInitializerTransposed(model_builder, input2);
  } else {
    input_2_idx = operand_indices.at(input2);
  }

  uint32_t bias_idx;
  if (node.input_size() == 2) {
    std::string bias = node.name() + op + "_bias";
    const auto& B_type = operand_types.at(input2).type;
    ModelBuilder::Shape bias_dimen = {shaper[input2][0]};
    if (B_type == Type::TENSOR_FLOAT32) {
      float buffer[bias_dimen[0]];
      for (uint32_t i = 0; i < bias_dimen[0]; i++) {
        buffer[i] = 0.f;
      }
      OperandType operandType(Type::TENSOR_FLOAT32, bias_dimen);
      bias_idx = model_builder.AddOperandFromPersistMemoryBuffer(
          bias, &buffer[0], operandType);
    } else {
      ORT_THROW("Unknown weight type " + TypeToStr(B_type));
    }
  } else {
    bias_idx = operand_indices.at(node.input(2));
  }

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input1));  // A
  input_indices.push_back(input_2_idx);                 // B
  input_indices.push_back(bias_idx);                    // C
  int32_t fuse_code = model_builder.FindActivation(output);
  input_indices.push_back(model_builder.AddOperandFromScalar(fuse_code));

  shaper.FC(input1, input2, output);
  const OperandType output_operand_type(operand_types.at(input1).type, shaper[output]);
  model_builder.AddOperation(ANEURALNETWORKS_FULLY_CONNECTED,
                             input_indices, {output}, {output_operand_type});
}

#pragma endregion

#pragma region op_unary

class UnaryOpBuilder : public BaseOpBuilder {
 private:
  int32_t GetMinSupportedSdkVer(ModelBuilder& model_builder,
                                const ONNX_NAMESPACE::NodeProto& node) const override;

  void AddToModelBuilderImpl(
      ModelBuilder& model_builder,
      const ONNX_NAMESPACE::NodeProto& node) override;
};

int32_t UnaryOpBuilder::GetMinSupportedSdkVer(ModelBuilder& /* model_builder */,
                                              const ONNX_NAMESPACE::NodeProto& node) const {
  const auto& op(node.op_type());
  if (op == "Abs" ||
      op == "Exp" ||
      op == "Neg" ||
      op == "Sin" ||
      op == "Sqrt" ||
      op == "Log") {
    return 29;
  }

  return 27;
}

void UnaryOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                           const ONNX_NAMESPACE::NodeProto& node) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& op(node.op_type());

  const auto& input = node.input(0);
  const auto& output = node.output(0);
  shaper.Identity(input, output);
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);

  int32_t op_code;
  if (op == "Abs")
    op_code = ANEURALNETWORKS_ABS;
  else if (op == "Exp")
    op_code = ANEURALNETWORKS_EXP;
  else if (op == "Floor")
    op_code = ANEURALNETWORKS_FLOOR;
  else if (op == "Log")
    op_code = ANEURALNETWORKS_LOG;
  else if (op == "Sigmoid")
    op_code = ANEURALNETWORKS_LOGISTIC;
  else if (op == "Neg")
    op_code = ANEURALNETWORKS_NEG;
  else if (op == "Sin")
    op_code = ANEURALNETWORKS_SIN;
  else if (op == "Sqrt")
    op_code = ANEURALNETWORKS_SQRT;
  else if (op == "Tanh")
    op_code = ANEURALNETWORKS_TANH;
  else {
    ORT_THROW("UnaryOpBuilder, unknown op: " + op);
  }
  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  model_builder.AddOperation(op_code, input_indices, {output}, {output_operand_type});
}

#pragma endregion

#pragma region CreateOpBuilders

std::unordered_map<std::string, std::shared_ptr<IOpBuilder>>
CreateOpBuilders() {
  std::unordered_map<std::string, std::shared_ptr<IOpBuilder>> op_map;

  {
    auto binary_op_builder = std::make_shared<BinaryOpBuilder>();
    op_map.emplace("Add", binary_op_builder);
    op_map.emplace("Sub", binary_op_builder);
    op_map.emplace("Mul", binary_op_builder);
    op_map.emplace("Div", binary_op_builder);
  }

  op_map.emplace("Relu", std::make_shared<ReluOpBuilder>());
  op_map.emplace("Transpose", std::make_shared<TransposeOpBuilder>());
  op_map.emplace("Reshape", std::make_shared<ReshapeOpBuilder>());
  op_map.emplace("BatchNormalization", std::make_shared<BatchNormalizationOpBuilder>());

  {
    auto pool_op_builder = std::make_shared<PoolOpBuilder>();
    op_map.emplace("GlobalAveragePool", pool_op_builder);
    op_map.emplace("GlobalMaxPool", pool_op_builder);
    op_map.emplace("AveragePool", pool_op_builder);
    op_map.emplace("MaxPool", pool_op_builder);
  }

  op_map.emplace("Conv", std::make_shared<ConvOpBuilder>());
  op_map.emplace("Cast", std::make_shared<CastOpBuilder>());
  op_map.emplace("Softmax", std::make_shared<SoftMaxOpBuilder>());
  op_map.emplace("Identity", std::make_shared<IdentityOpBuilder>());

  {
    auto gemm_op_builder = std::make_shared<GemmOpBuilder>();
    op_map.emplace("Gemm", gemm_op_builder);
    op_map.emplace("MatMul", gemm_op_builder);
  }

  {
    auto unary_op_builder = std::make_shared<UnaryOpBuilder>();
    op_map.emplace("Abs", unary_op_builder);
    op_map.emplace("Exp", unary_op_builder);
    op_map.emplace("Floor", unary_op_builder);
    op_map.emplace("Log", unary_op_builder);
    op_map.emplace("Sigmoid", unary_op_builder);
    op_map.emplace("Neg", unary_op_builder);
    op_map.emplace("Sin", unary_op_builder);
    op_map.emplace("Sqrt", unary_op_builder);
    op_map.emplace("Tanh", unary_op_builder);
  }

  return op_map;
}

#pragma endregion

}  // namespace nnapi
}  // namespace onnxruntime