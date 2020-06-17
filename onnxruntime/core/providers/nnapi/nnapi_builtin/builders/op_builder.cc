// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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

  ModelBuilder::IndexSeq input_indices;
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

  ModelBuilder::IndexSeq input_indices;
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

uint32_t AddInitializerInNewLayout(ModelBuilder& model_builder,
                                   const std::string& name,
                                   DataLayout new_layout) {
  const auto& tensor = model_builder.GetInitializerTensors().at(name);
  ModelBuilder::Shape shape;
  for (auto dim : tensor.dims())
    shape.push_back(SafeInt<uint32_t>(dim));

  if (shape.size() != 4)
    throw std::invalid_argument(
        "The initializer is not 4D: " + name +
        " actual dim " + std::to_string(shape.size()));

  // TODO support other data types
  Type type;
  if (tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    type = Type::TENSOR_FLOAT32;
  } else {
    throw std::invalid_argument(
        "The initializer of graph doesn't have valid type: " + name);
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
                          in * h_t * w_t + h * w_t +
                          w;

          uint32_t nnapi_idx;
          if (new_layout == L_NCHW) {  // L_NCHW
            nnapi_idx = out * h_t * w_t * in_t +
                        h * w_t * in_t + w * in_t +
                        in;
          } else {  // L_1230 for depthwise conv weight
            nnapi_idx = in * h_t * w_t * out_t +
                        h * w_t * out_t + w * out_t +
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

uint32_t AddInitializerTransposed(ModelBuilder& model_builder,
                                  const std::string& name) {
  const auto& tensor = model_builder.GetInitializerTensors().at(name);
  ModelBuilder::Shape shape;
  for (auto dim : tensor.dims())
    shape.push_back(SafeInt<uint32_t>(dim));

  if (shape.size() != 2)
    throw std::invalid_argument(
        "The initializer is not 2D: " + name +
        " actual dim " + std::to_string(shape.size()));

  // TODO support other data types
  Type type;
  if (tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    type = Type::TENSOR_FLOAT32;
  } else {
    throw std::invalid_argument(
        "The initializer of graph doesn't have valid type: " + name);
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
  BaseOpBuilder(ModelBuilder& model_builder, const ONNX_NAMESPACE::NodeProto& node)
      : model_builder_(model_builder),
        node_(node) {}

  virtual ~BaseOpBuilder() = default;
  virtual void AddInitializersToSkip() override {}
  std::pair<bool, std::string> IsOpSupported() override final;
  void AddOperator() override final;

 protected:
  ModelBuilder& model_builder_;
  const ONNX_NAMESPACE::NodeProto& node_;

  virtual std::pair<bool, std::string> IsOpSupportedImpl();
  virtual int32_t GetMinSupportedSdkVer() const { return 27; }
  virtual void AddOperatorImpl();
};

std::pair<bool, std::string> BaseOpBuilder::IsOpSupported() {
#ifdef __ANDROID__
  int32_t android_sdk_ver = model_builder_.GetAndroidSdkVer();
  int32_t required_sdk_ver = GetMinSupportedSdkVer();
  if (required_sdk_ver > android_sdk_ver) {
    LOGV("Android API level %d is lower than %d", android_sdk_ver, required_sdk_ver);
    return {false, "Operator " + node_.op_type() + " is only supported on API > " + std::to_string(required_sdk_ver)};
  }
#endif

  return IsOpSupportedImpl();
}  // namespace nnapi

std::pair<bool, std::string> BaseOpBuilder::IsOpSupportedImpl() {
  return {false, "Unsupported operator " + node_.op_type()};
}

void BaseOpBuilder::AddOperator() {
  bool supported;
  std::string error_msg;
  std::tie(supported, error_msg) = IsOpSupported();
  if (!supported)
    throw std::invalid_argument(
        "Unsupported operator " + node_.op_type() + ",msg: " + error_msg);

  AddOperatorImpl();

  LOGV("Operator %s type %s added", node_.name().c_str(), node_.op_type().c_str());
}

void BaseOpBuilder::AddOperatorImpl() {
  throw std::logic_error(
      "Unsupported operator " + node_.op_type());
}

#pragma endregion op_base

#pragma region op_add

class AddOpBuilder : public BaseOpBuilder {
 public:
  AddOpBuilder(ModelBuilder& model_builder, const ONNX_NAMESPACE::NodeProto& node)
      : BaseOpBuilder(model_builder, node) {}

 private:
  std::pair<bool, std::string> IsOpSupportedImpl() override;
  void AddOperatorImpl() override;
};

std::pair<bool, std::string> AddOpBuilder::IsOpSupportedImpl() {
  return {true, ""};
}

void AddOpBuilder::AddOperatorImpl() {
  const auto& input1 = node_.input(0);
  const auto& input2 = node_.input(1);
  const auto& output = node_.output(0);
  int32_t fuse_code = model_builder_.FindActivation(output);
  AddBinaryOperator(ANEURALNETWORKS_ADD, model_builder_,
                    input1, input2, fuse_code, output);
}

#pragma endregion op_add

#pragma region op_mul

class MulOpBuilder : public BaseOpBuilder {
 public:
  MulOpBuilder(ModelBuilder& model_builder, const ONNX_NAMESPACE::NodeProto& node)
      : BaseOpBuilder(model_builder, node) {}

 private:
  std::pair<bool, std::string> IsOpSupportedImpl() override;
  void AddOperatorImpl() override;
};

std::pair<bool, std::string> MulOpBuilder::IsOpSupportedImpl() {
  return {true, ""};
}
void MulOpBuilder::AddOperatorImpl() {
  const auto& input1 = node_.input(0);
  const auto& input2 = node_.input(1);
  const auto& output = node_.output(0);
  int32_t fuse_code = model_builder_.FindActivation(output);
  AddBinaryOperator(ANEURALNETWORKS_MUL, model_builder_,
                    input1, input2, fuse_code, output);
}

#pragma endregion op_mul

#pragma region op_relu

class ReluOpBuilder : public BaseOpBuilder {
 public:
  ReluOpBuilder(ModelBuilder& model_builder, const ONNX_NAMESPACE::NodeProto& node)
      : BaseOpBuilder(model_builder, node) {}

 private:
  std::pair<bool, std::string> IsOpSupportedImpl() override;
  void AddOperatorImpl() override;
};

std::pair<bool, std::string> ReluOpBuilder::IsOpSupportedImpl() {
  return {true, ""};
}

void ReluOpBuilder::AddOperatorImpl() {
  auto& shaper(model_builder_.GetShaper());
  const auto& operand_indices(model_builder_.GetOperandIndices());
  const auto& operand_types(model_builder_.GetOperandTypes());

  const auto& input = node_.input(0);
  const auto& output = node_.output(0);
  shaper.Identity(input, output);
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);

  // skip this relu if it is some op's fuse output
  if (Contains(model_builder_.GetFusedActivations(), node_.name())) {
    model_builder_.RegisterOperand(output, operand_indices.at(input), output_operand_type);
  } else {
    ModelBuilder::IndexSeq input_indices;
    input_indices.push_back(operand_indices.at(input));
    model_builder_.AddOperation(ANEURALNETWORKS_RELU, input_indices, {output}, {output_operand_type});
  }
}

#pragma endregion op_relu

#pragma region op_transpose

class TransposeOpBuilder : public BaseOpBuilder {
 public:
  TransposeOpBuilder(ModelBuilder& model_builder, const ONNX_NAMESPACE::NodeProto& node)
      : BaseOpBuilder(model_builder, node) {}

 private:
  std::pair<bool, std::string> IsOpSupportedImpl() override;
  int32_t GetMinSupportedSdkVer() const override { return 28; }
  void AddOperatorImpl() override;
};

std::pair<bool, std::string> TransposeOpBuilder::IsOpSupportedImpl() {
  const auto input_size = GetShape(model_builder_.GetOnnxModel(), node_.input(0)).size();
  if (input_size > 4)
    return {false, "Transpose only supports up to 4d shape, input is " +
                       std::to_string(input_size) + "d shape"};

  return {true, ""};
}

void TransposeOpBuilder::AddOperatorImpl() {
  auto& shaper(model_builder_.GetShaper());
  const auto& operand_indices(model_builder_.GetOperandIndices());
  const auto& operand_types(model_builder_.GetOperandTypes());
  NodeAttrHelper helper(node_);

  const auto& input = node_.input(0);
  ModelBuilder::IndexSeq input_indices;
  input_indices.push_back(operand_indices.at(input));  // input

  vector<int32_t> perm = helper.get("perm", vector<int32_t>());
  auto input_dims = shaper[input].size();
  if (perm.empty()) {
    for (int32_t i = input_dims - 1; i >= 0; i--)
      perm.push_back(i);
  }

  ModelBuilder::Shape perm_dimen = {SafeInt<uint32_t>(input_dims)};
  std::string perm_name = node_.name() + input + "perm";
  OperandType perm_operand_type(Type::TENSOR_INT32, perm_dimen);
  uint32_t perm_idx = model_builder_.AddOperandFromPersistMemoryBuffer(perm_name, perm.data(), perm_operand_type);
  input_indices.push_back(perm_idx);

  const auto& output = node_.output(0);
  shaper.Transpose(input, perm, output);
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);
  model_builder_.AddOperation(ANEURALNETWORKS_TRANSPOSE, input_indices, {output}, {output_operand_type});
}

#pragma endregion op_transpose

#pragma region op_reshape

class ReshapeOpBuilder : public BaseOpBuilder {
 public:
  ReshapeOpBuilder(ModelBuilder& model_builder, const ONNX_NAMESPACE::NodeProto& node)
      : BaseOpBuilder(model_builder, node) {}
  void AddInitializersToSkip() override;

 private:
  std::pair<bool, std::string> IsOpSupportedImpl() override;
  void AddOperatorImpl() override;
};

void ReshapeOpBuilder::AddInitializersToSkip() {
  model_builder_.AddSkippedInitializer(node_.input(1));
}

std::pair<bool, std::string> ReshapeOpBuilder::IsOpSupportedImpl() {
  const auto& initializers(model_builder_.GetInitializerTensors());
  if (!Contains(initializers, node_.input(1)))
    return {false, "New shape of reshape must be known"};

  const auto input_size = GetShape(model_builder_.GetOnnxModel(), node_.input(0)).size();
  if (input_size > 4)
    return {false, "Reshape only supports up to 4d shape, input is " +
                       std::to_string(input_size) + "d shape"};

  const auto& shape_tensor = initializers.at(node_.input(1));
  const int64_t* rawShape = GetTensorInt64Data(shape_tensor);
  const auto size = SafeInt<uint32_t>(shape_tensor.dims()[0]);
  const auto input_shape = GetShape(model_builder_.GetOnnxModel(), node_.input(0));

  for (uint32_t i = 0; i < size; i++) {
    // NNAPI reshape does not support 0 as dimension
    if (rawShape[i] == 0 && i < input_shape.size() && input_shape[i] == 0)
      return {false, "Reshape doesn't suppport 0 reshape dimension on a dynamic dimension"};
  }

  return {true, ""};
}

void ReshapeOpBuilder::AddOperatorImpl() {
  auto& shaper(model_builder_.GetShaper());
  const auto& operand_indices(model_builder_.GetOperandIndices());
  const auto& operand_types(model_builder_.GetOperandTypes());
  const auto& initializers(model_builder_.GetInitializerTensors());

  const auto& input = node_.input(0);
  const auto& output = node_.output(0);
  ModelBuilder::IndexSeq input_indices;
  input_indices.push_back(operand_indices.at(input));  // input

  const auto& shape_tensor = initializers.at(node_.input(1));
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
  std::string shape_name = node_.name() + input + "newshape";
  OperandType shape_operand_type(Type::TENSOR_INT32, shape_dimen);
  uint32_t shape_idx = model_builder_.AddOperandFromPersistMemoryBuffer(shape_name, shape.data(), shape_operand_type);
  input_indices.push_back(shape_idx);

  shaper.Reshape(input, shape, output);
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);
  model_builder_.AddOperation(ANEURALNETWORKS_RESHAPE, input_indices, {output}, {output_operand_type});
}

#pragma endregion op_reshape

#pragma region op_batchnormalization

class BatchNormalizationOpBuilder : public BaseOpBuilder {
 public:
  BatchNormalizationOpBuilder(ModelBuilder& model_builder, const ONNX_NAMESPACE::NodeProto& node)
      : BaseOpBuilder(model_builder, node) {}
  void AddInitializersToSkip() override;

 private:
  std::pair<bool, std::string> IsOpSupportedImpl() override;
  void AddOperatorImpl() override;
};

void BatchNormalizationOpBuilder::AddInitializersToSkip() {
  // skip everything except input0 for BatchNormalization
  model_builder_.AddSkippedInitializer(node_.input(1));  // scale
  model_builder_.AddSkippedInitializer(node_.input(2));  // B
  model_builder_.AddSkippedInitializer(node_.input(3));  // mean
  model_builder_.AddSkippedInitializer(node_.input(4));  //var
}

std::pair<bool, std::string> BatchNormalizationOpBuilder::IsOpSupportedImpl() {
  if (node_.output_size() != 1) {
    return {false,
            "Your onnx model may be in training mode, please export "
            "it in test mode."};
  }

  const auto& initializers(model_builder_.GetInitializerTensors());
  const auto& scale_name = node_.input(1);
  const auto& b_name = node_.input(2);
  const auto& mean_name = node_.input(3);
  const auto& var_name = node_.input(4);
  if (!Contains(initializers, scale_name)) {
    return {false, "Scale of BN must be known"};
  }
  if (!Contains(initializers, b_name)) {
    return {false, "B of BN must be known"};
  }
  if (!Contains(initializers, mean_name)) {
    return {false, "Mean of BN must be known"};
  }
  if (!Contains(initializers, var_name)) {
    return {false, "Var of BN must be known"};
  }

  return {true, ""};
}

void BatchNormalizationOpBuilder::AddOperatorImpl() {
  auto& shaper(model_builder_.GetShaper());
  const auto& operand_types(model_builder_.GetOperandTypes());
  const auto& initializers(model_builder_.GetInitializerTensors());
  NodeAttrHelper helper(node_);

  // For reshape we are not really doing anything but
  // register a new operand with new shape
  const auto input = node_.input(0);
  const auto output = node_.output(0);

  const auto& scale_tensor = initializers.at(node_.input(1));
  const auto& bias_tensor = initializers.at(node_.input(2));
  const auto& mean_tensor = initializers.at(node_.input(3));
  const auto& var_tensor = initializers.at(node_.input(4));
  const auto eps = helper.get("epsilon", 1e-5f);

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

  const auto tensor_a_name = input + "_imm_a";
  const auto tensor_b_name = input + "_imm_b";
  const auto tensor_imm_product_name = input + "_imm_mul";
  ModelBuilder::Shape tensor_a_dimen;
  if (model_builder_.UseNCHW())
    tensor_a_dimen = {size, 1, 1};  // {C, H, W}
  else
    tensor_a_dimen = {size};

  shaper.AddShape(tensor_a_name, tensor_a_dimen);
  shaper.AddShape(tensor_b_name, tensor_a_dimen);
  const OperandType operandType_a(operand_types.at(input).type, tensor_a_dimen);
  model_builder_.AddOperandFromPersistMemoryBuffer(tensor_a_name, a.data(), operandType_a);
  const OperandType operandType_b(operand_types.at(input).type, tensor_a_dimen);
  model_builder_.AddOperandFromPersistMemoryBuffer(tensor_b_name, b.data(), operandType_b);

  // Mul
  AddBinaryOperator(ANEURALNETWORKS_MUL,
                    model_builder_,
                    input, tensor_a_name,
                    ANEURALNETWORKS_FUSED_NONE,
                    tensor_imm_product_name);

  // Add
  int32_t fuse_code = model_builder_.FindActivation(output);
  AddBinaryOperator(ANEURALNETWORKS_ADD,
                    model_builder_,
                    tensor_imm_product_name, tensor_b_name,
                    fuse_code,
                    output);
}

#pragma endregion op_batchnormalization

#pragma region op_pool

class PoolOpBuilder : public BaseOpBuilder {
 public:
  PoolOpBuilder(ModelBuilder& model_builder, const ONNX_NAMESPACE::NodeProto& node)
      : BaseOpBuilder(model_builder, node) {}

 private:
  std::pair<bool, std::string> IsOpSupportedImpl() override;
  int32_t GetMinSupportedSdkVer() const override { return 29; }
  void AddOperatorImpl() override;
};

std::pair<bool, std::string> PoolOpBuilder::IsOpSupportedImpl() {
  const auto& op = node_.op_type();
  if (op == "AveragePool" || op == "MaxPool") {
    NodeAttrHelper helper(node_);

    const auto count_include_pad = helper.get("count_include_pad", 0);
    if (count_include_pad == 1) {
      return {false, "count_include_pad == 1 is not supported"};
    }

    const auto storage_order = helper.get("storage_order", 0);
    if (storage_order == 1) {
      return {false, "storage_order == 1 is not supported"};
    }

    if (helper.get("auto_pad", "NOTSET") != "NOTSET") {
      return {false, "auto_pad is not supported"};
    }

    if (helper.get("kernel_shape", std::vector<int32_t>{1, 1}).size() != 2) {
      return {false, "Only pooling 2d is supported"};
    }

    if (helper.get("ceil_mode", 0) == 1) {
      return {false, "ceil_mode == 1 is not supported for pooling"};
    }

    if (helper.get("dilations", std::vector<int32_t>{1, 1}) !=
        std::vector<int32_t>{1, 1}) {
      return {false, "Dilations of pooling is not supported"};
    }

    if (node_.output_size() != 1) {
      return {false, "Argmax in maxpooling is not supported"};
    }
  } else if (op == "GlobalAveragePool" || op == "GlobalMaxPool") {
    const auto input_shape = GetShape(model_builder_.GetOnnxModel(), node_.input(0));
    if (input_shape.size() != 4) {
      return {false,
              "GlobalAveragePool/GlobalMaxPool Only rank-4 tensor is supported in " + op};
    }
  }

  return {true, ""};
}

void PoolOpBuilder::AddOperatorImpl() {
  auto& shaper(model_builder_.GetShaper());
  NodeAttrHelper helper(node_);

  const auto& input = node_.input(0);
  const auto& output = node_.output(0);
  const auto& op = node_.op_type();

  int32_t operationType;
  if (op == "AveragePool" || op == "GlobalAveragePool")
    operationType = ANEURALNETWORKS_AVERAGE_POOL_2D;
  else  // (op == "MaxPool" || op == "GlobalMaxPool")
    operationType = ANEURALNETWORKS_MAX_POOL_2D;

  vector<int32_t> onnx_pads, onnx_strides, kernel_shape;
  if (op == "AveragePool" || op == "MaxPool") {
    kernel_shape = helper.get("kernel_shape", vector<int32_t>{0, 0});
    onnx_strides = helper.get("strides", vector<int>{1, 1});
    onnx_pads = helper.get("pads", vector<int>{0, 0, 0, 0});
  } else {  // (op == "GlobalAveragePool" || op == "GlobalMaxPool")
    onnx_strides = vector<int32_t>{1, 1};
    onnx_pads = vector<int32_t>{0, 0, 0, 0};
    if (model_builder_.UseNCHW())
      kernel_shape = vector<int32_t>{static_cast<int32_t>(shaper[input][2]),
                                     static_cast<int32_t>(shaper[input][3])};
    else
      kernel_shape = vector<int32_t>{static_cast<int32_t>(shaper[input][1]),
                                     static_cast<int32_t>(shaper[input][2])};
  }

  int32_t fuse_code = model_builder_.FindActivation(output);
  AddPoolOperator(operationType,
                  model_builder_,
                  input,
                  onnx_pads, onnx_strides, kernel_shape,
                  fuse_code,
                  output);
}

#pragma endregion op_pool

#pragma region op_conv

class ConvOpBuilder : public BaseOpBuilder {
 public:
  ConvOpBuilder(ModelBuilder& model_builder, const ONNX_NAMESPACE::NodeProto& node)
      : BaseOpBuilder(model_builder, node) {}
  void AddInitializersToSkip() override;

 private:
  std::pair<bool, std::string> IsOpSupportedImpl() override;
  void AddOperatorImpl() override;
};

void ConvOpBuilder::AddInitializersToSkip() {
  // skip the weight for conv as we need to transpose
  model_builder_.AddSkippedInitializer(node_.input(1));
}

std::pair<bool, std::string> ConvOpBuilder::IsOpSupportedImpl() {
  NodeAttrHelper helper(node_);
  if (helper.get("auto_pad", "NOTSET") != "NOTSET")
    return {false, "SAME_LOWER auto_pad is not supported"};

  const auto group = helper.get("group", 1);
  const auto weight_name = node_.input(1);
  if (Contains(model_builder_.GetInitializerTensors(), weight_name)) {
    const auto& tensor = model_builder_.GetInitializerTensors().at(weight_name);
    if (tensor.dims().size() != 4) {
      return {false, "Only conv 2d is supported."};
    }
    if (group != 1 && tensor.dims()[1] != 1) {
      return {false, "group != 1 is not supported"};
    }
  } else {
    return {false, "The weight of convolution must be known"};
  }

  return {true, ""};
}

void ConvOpBuilder::AddOperatorImpl() {
  auto& shaper(model_builder_.GetShaper());
  const auto& operand_indices(model_builder_.GetOperandIndices());
  const auto& operand_types(model_builder_.GetOperandTypes());
  const auto& initializers(model_builder_.GetInitializerTensors());
  NodeAttrHelper helper(node_);
  bool use_nchw = model_builder_.UseNCHW();

  // onnx strides are in the order height, width
  // while nnapi strides are in the order width, height
  const auto onnx_strides = helper.get("strides", vector<int>{1, 1});

  // onnx pads are in the order top, left, bottom, right
  // while nnapi pads is in the order left, right, top, bottom
  const auto onnx_pads = helper.get("pads", vector<int>{0, 0, 0, 0});

  // onnx dilations is in the order height, width
  // while nnapi dilations are in the order width, height
  const auto onnx_dilations = helper.get("dilations", vector<int>{1, 1});
  const auto group = helper.get("group", 1);

  const auto& input = node_.input(0);
  const auto& weight = node_.input(1);
  const auto& output = node_.output(0);

  bool conv2d = (group == 1);
  const auto& weight_tensor = initializers.at(weight);
  bool depthwiseConv2D = (weight_tensor.dims()[1] == 1);

  ModelBuilder::IndexSeq input_indices;
  input_indices.push_back(operand_indices.at(input));

  if (conv2d) {
    input_indices.push_back(AddInitializerInNewLayout(
        model_builder_, weight, L_NCHW));
  } else {  // depthwiseConv2D
    input_indices.push_back(AddInitializerInNewLayout(
        model_builder_, weight, L_1230));
  }

  bool hasBias = (node_.input_size() >= 3);
  std::string bias = hasBias ? node_.input(2) : weight + "_bias";

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
      bias_idx_val = model_builder_.AddOperandFromPersistMemoryBuffer(
          bias, &buffer[0], operandType);
    } else {
      throw std::invalid_argument("Unknown weight type " + typeToStr(weight_type));
    }
  }

  input_indices.push_back(bias_idx_val);
  input_indices.push_back(model_builder_.AddOperandFromScalar(onnx_pads[1]));
  input_indices.push_back(model_builder_.AddOperandFromScalar(onnx_pads[3]));
  input_indices.push_back(model_builder_.AddOperandFromScalar(onnx_pads[0]));
  input_indices.push_back(model_builder_.AddOperandFromScalar(onnx_pads[2]));
  input_indices.push_back(model_builder_.AddOperandFromScalar(onnx_strides[1]));
  input_indices.push_back(model_builder_.AddOperandFromScalar(onnx_strides[0]));
  if (!conv2d && depthwiseConv2D) {
    int32_t depthwiseMultiplier = shaper[weight][3] / group;
    input_indices.push_back(model_builder_.AddOperandFromScalar(depthwiseMultiplier));
  }
  int32_t fuse_code = model_builder_.FindActivation(output);
  input_indices.push_back(model_builder_.AddOperandFromScalar(fuse_code));
  // TODO support API 27
  input_indices.push_back(model_builder_.AddOperandFromScalar(use_nchw));
  input_indices.push_back(model_builder_.AddOperandFromScalar(onnx_dilations[1]));
  input_indices.push_back(model_builder_.AddOperandFromScalar(onnx_dilations[0]));

  int32_t operationCode;
  if (conv2d) {
    operationCode = ANEURALNETWORKS_CONV_2D;
    shaper.Conv(input, weight,
                onnx_pads, onnx_strides, onnx_dilations,
                use_nchw,
                output);
  } else {  // depthwiseConv2D
    operationCode = ANEURALNETWORKS_DEPTHWISE_CONV_2D;
    shaper.DepthwiseConv(input, weight,
                         onnx_pads, onnx_strides, onnx_dilations,
                         use_nchw,
                         output);
  }

  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);
  model_builder_.AddOperation(operationCode, input_indices, {output}, {output_operand_type});
}

#pragma endregion op_conv

#pragma region op_cast

class CastOpBuilder : public BaseOpBuilder {
 public:
  CastOpBuilder(ModelBuilder& model_builder, const ONNX_NAMESPACE::NodeProto& node)
      : BaseOpBuilder(model_builder, node) {}

 private:
  std::pair<bool, std::string> IsOpSupportedImpl() override;
  int32_t GetMinSupportedSdkVer() const override { return 29; }
  void AddOperatorImpl() override;
};

std::pair<bool, std::string> CastOpBuilder::IsOpSupportedImpl() {
  NodeAttrHelper helper(node_);
  auto to = helper.get("to", 0);
  if (to != ONNX_NAMESPACE::TensorProto::FLOAT &&
      to != ONNX_NAMESPACE::TensorProto::INT32) {
    return {false, "Only support cast to int32 or float"};
  }

  return {true, ""};
}

void CastOpBuilder::AddOperatorImpl() {
  auto& shaper(model_builder_.GetShaper());
  const auto& operand_indices(model_builder_.GetOperandIndices());
  NodeAttrHelper helper(node_);

  const auto& input = node_.input(0);
  const auto& output = node_.output(0);
  auto to = helper.get("to", 0);
  Type type;
  switch (to) {
    case ONNX_NAMESPACE::TensorProto::FLOAT:
      type = Type::TENSOR_FLOAT32;
      break;
    case ONNX_NAMESPACE::TensorProto::INT32:
      type = Type::TENSOR_INT32;
      break;
    default:
      throw std::invalid_argument(
          "Invalid cast to type: " + std::to_string(to));
  }

  ModelBuilder::IndexSeq input_indices;
  input_indices.push_back(operand_indices.at(input));
  shaper.Identity(input, output);
  const OperandType output_operand_type(type, shaper[output]);
  model_builder_.AddOperation(ANEURALNETWORKS_CAST, input_indices, {output}, {output_operand_type});
}

#pragma endregion

#pragma region op_softmax

class SoftMaxOpBuilder : public BaseOpBuilder {
 public:
  SoftMaxOpBuilder(ModelBuilder& model_builder, const ONNX_NAMESPACE::NodeProto& node)
      : BaseOpBuilder(model_builder, node) {}

 private:
  std::pair<bool, std::string> IsOpSupportedImpl() override;
  int32_t GetMinSupportedSdkVer() const override { return 29; }
  void AddOperatorImpl() override;
};

std::pair<bool, std::string> SoftMaxOpBuilder::IsOpSupportedImpl() {
  const auto input_size = GetShape(model_builder_.GetOnnxModel(), node_.input(0)).size();
  if (input_size != 2 || input_size != 4)
    return {false, "SoftMax only support 2d/4d shape, input is " +
                       std::to_string(input_size) + "d shape"};

  return {true, ""};
}

void SoftMaxOpBuilder::AddOperatorImpl() {
  auto& shaper(model_builder_.GetShaper());
  const auto& operand_indices(model_builder_.GetOperandIndices());
  const auto& operand_types(model_builder_.GetOperandTypes());
  NodeAttrHelper helper(node_);

  const auto& input = node_.input(0);
  const auto& output = node_.output(0);
  float beta = 1.f;
  int32_t axis = helper.get("axis", 1);
  ModelBuilder::IndexSeq input_indices;
  input_indices.push_back(operand_indices.at(input));
  input_indices.push_back(model_builder_.AddOperandFromScalar(beta));
  input_indices.push_back(model_builder_.AddOperandFromScalar(axis));

  shaper.Identity(input, output);
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);
  model_builder_.AddOperation(ANEURALNETWORKS_SOFTMAX, input_indices, {output}, {output_operand_type});
}

#pragma endregion

#pragma region op_identity

class IdentityOpBuilder : public BaseOpBuilder {
 public:
  IdentityOpBuilder(ModelBuilder& model_builder, const ONNX_NAMESPACE::NodeProto& node)
      : BaseOpBuilder(model_builder, node) {}

 private:
  std::pair<bool, std::string> IsOpSupportedImpl() override;
  void AddOperatorImpl() override;
};

std::pair<bool, std::string> IdentityOpBuilder::IsOpSupportedImpl() {
  return {true, ""};
}

void IdentityOpBuilder::AddOperatorImpl() {
  // Identity is not really going to do anything
  // Just register the dimension and type, with same index and new name
  auto& shaper(model_builder_.GetShaper());
  const auto& operand_indices(model_builder_.GetOperandIndices());
  const auto& operand_types(model_builder_.GetOperandTypes());

  const auto& input = node_.input(0);
  const auto& output = node_.output(0);
  ModelBuilder::IndexSeq input_indices;
  input_indices.push_back(operand_indices.at(input));  // input

  shaper.Identity(input, output);
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);
  model_builder_.RegisterOperand(output, operand_indices.at(input), output_operand_type);
}

#pragma endregion

#pragma region op_gemm

class GemmOpBuilder : public BaseOpBuilder {
 public:
  GemmOpBuilder(ModelBuilder& model_builder, const ONNX_NAMESPACE::NodeProto& node)
      : BaseOpBuilder(model_builder, node) {}
  void AddInitializersToSkip() override;

 private:
  std::pair<bool, std::string> IsOpSupportedImpl() override;
  void AddOperatorImpl() override;
};

std::pair<bool, std::string> GemmOpBuilder::IsOpSupportedImpl() {
  const auto& op = node_.op_type();
  const auto& initializers(model_builder_.GetInitializerTensors());

  if (op == "MatMul") {  // Only support A*B B is an initializer
    if (!Contains(initializers, node_.input(1)))
      return {false, "B of MatMul must be known"};
  } else if (op == "Gemm") {
    // Only support
    // 1. A*B'+C
    // 2. A*B+C and B is an initializer
    NodeAttrHelper helper(node_);
    const auto transA = helper.get("transA", 0);
    const auto transB = helper.get("transB", 0);
    const auto alpha = helper.get("alpha", 1.0f);
    const auto beta = helper.get("beta", 1.0f);

    if (!(transA == 0 && alpha == 1.f && beta == 1.f)) {
      return {false,
              "Only transA == 0, alpha == 1.0 and beta == "
              "1.0 is supported."};
    }

    if (transB == 0 && !Contains(initializers, node_.input(1))) {
      return {false, "B of MatMul must be known if transB != 1"};
    }

    if (node_.input_size() == 3) {
      const auto b_shape = GetShape(model_builder_.GetOnnxModel(), node_.input(1));
      const auto c_shape = GetShape(model_builder_.GetOnnxModel(), node_.input(2));
      if (c_shape.size() != 1 || c_shape[0] != b_shape[0])
        return {false, "C of MatMul must be a vector of b_shape[0]"};
    }
  }

  return {true, ""};
}

void GemmOpBuilder::AddInitializersToSkip() {
  const auto& op = node_.op_type();
  if (op == "MatMul") {
    model_builder_.AddSkippedInitializer(node_.input(1));
  } else if (op == "Gemm") {
    NodeAttrHelper helper(node_);
    const auto transB = helper.get("transB", 0);
    if (transB == 0)
      model_builder_.AddSkippedInitializer(node_.input(1));
  }
}

void GemmOpBuilder::AddOperatorImpl() {
  const auto& op = node_.op_type();
  auto& shaper(model_builder_.GetShaper());
  const auto& operand_indices(model_builder_.GetOperandIndices());
  const auto& operand_types(model_builder_.GetOperandTypes());
  NodeAttrHelper helper(node_);

  const auto& input1 = node_.input(0);
  const auto& input2 = node_.input(1);
  const auto& output = node_.output(0);
  const auto transB = helper.get("transB", 0);

  uint32_t input_2_idx;
  if (transB == 0) {
    input_2_idx = AddInitializerTransposed(model_builder_, input2);
  } else {
    input_2_idx = operand_indices.at(input2);
  }

  uint32_t bias_idx;
  if (node_.input_size() == 2) {
    std::string bias = node_.name() + op + "_bias";
    const auto& B_type = operand_types.at(input2).type;
    ModelBuilder::Shape bias_dimen = {shaper[input2][0]};
    if (B_type == Type::TENSOR_FLOAT32) {
      float buffer[bias_dimen[0]];
      for (uint32_t i = 0; i < bias_dimen[0]; i++) {
        buffer[i] = 0.f;
      }
      OperandType operandType(Type::TENSOR_FLOAT32, bias_dimen);
      bias_idx = model_builder_.AddOperandFromPersistMemoryBuffer(
          bias, &buffer[0], operandType);
    } else {
      throw std::invalid_argument("Unknown weight type " + typeToStr(B_type));
    }
  } else {
    bias_idx = operand_indices.at(node_.input(2));
  }

  ModelBuilder::IndexSeq input_indices;
  input_indices.push_back(operand_indices.at(input1));  // A
  input_indices.push_back(input_2_idx);                 // B
  input_indices.push_back(bias_idx);                    // C
  int32_t fuse_code = model_builder_.FindActivation(output);
  input_indices.push_back(model_builder_.AddOperandFromScalar(fuse_code));

  shaper.FC(input1, input2, output);
  const OperandType output_operand_type(operand_types.at(input1).type, shaper[output]);
  model_builder_.AddOperation(ANEURALNETWORKS_FULLY_CONNECTED,
                              input_indices, {output}, {output_operand_type});
}

#pragma endregion

#pragma region op_unary

class UnaryOpBuilder : public BaseOpBuilder {
 public:
  UnaryOpBuilder(ModelBuilder& model_builder, const ONNX_NAMESPACE::NodeProto& node)
      : BaseOpBuilder(model_builder, node) {}

 private:
  int32_t GetMinSupportedSdkVer() const override;
  std::pair<bool, std::string> IsOpSupportedImpl() override;
  void AddOperatorImpl() override;
};

int32_t UnaryOpBuilder::GetMinSupportedSdkVer() const {
  const auto& op(node_.op_type());
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

std::pair<bool, std::string> UnaryOpBuilder::IsOpSupportedImpl() {
  return {true, ""};
}

void UnaryOpBuilder::AddOperatorImpl() {
  auto& shaper(model_builder_.GetShaper());
  const auto& operand_indices(model_builder_.GetOperandIndices());
  const auto& operand_types(model_builder_.GetOperandTypes());
  const auto& op(node_.op_type());

  const auto& input = node_.input(0);
  const auto& output = node_.output(0);
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
    throw std::invalid_argument(
        "UnaryOpBuilder, unknown op: " + op);
  }
  ModelBuilder::IndexSeq input_indices;
  input_indices.push_back(operand_indices.at(input));
  model_builder_.AddOperation(op_code, input_indices, {output}, {output_operand_type});
}

#pragma endregion

#pragma region CreateOpBuilder

std::unique_ptr<IOpBuilder> CreateOpBuilder(ModelBuilder& model_builder,
                                            const ONNX_NAMESPACE::NodeProto& node) {
  const auto& op = node.op_type();
  if (op == "Add") {
    return std::make_unique<AddOpBuilder>(model_builder, node);
  } else if (op == "Mul") {
    return std::make_unique<MulOpBuilder>(model_builder, node);
  } else if (op == "Relu") {
    return std::make_unique<ReluOpBuilder>(model_builder, node);
  } else if (op == "Transpose") {
    return std::make_unique<TransposeOpBuilder>(model_builder, node);
  } else if (op == "Reshape") {
    return std::make_unique<ReshapeOpBuilder>(model_builder, node);
  } else if (op == "BatchNormalization") {
    return std::make_unique<BatchNormalizationOpBuilder>(model_builder, node);
  } else if (op == "GlobalAveragePool" ||
             op == "GlobalMaxPool" ||
             op == "AveragePool" ||
             op == "MaxPool") {
    return std::make_unique<PoolOpBuilder>(model_builder, node);
  } else if (op == "Conv") {
    return std::make_unique<ConvOpBuilder>(model_builder, node);
  } else if (op == "Cast") {
    return std::make_unique<CastOpBuilder>(model_builder, node);
  } else if (op == "Softmax") {
    return std::make_unique<SoftMaxOpBuilder>(model_builder, node);
  } else if (op == "Identity") {
    return std::make_unique<IdentityOpBuilder>(model_builder, node);
  } else if (op == "Gemm" ||
             op == "MatMul") {
    return std::make_unique<GemmOpBuilder>(model_builder, node);
  } else if (op == "Abs" ||
             op == "Exp" ||
             op == "Floor" ||
             op == "Log" ||
             op == "Sigmoid" ||
             op == "Neg" ||
             op == "Sin" ||
             op == "Sqrt" ||
             op == "Tanh") {
    return std::make_unique<UnaryOpBuilder>(model_builder, node);
  }

  return std::make_unique<BaseOpBuilder>(model_builder, node);
}

#pragma endregion CreateOpBuilder

}  // namespace nnapi
}  // namespace onnxruntime