// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/common/logging/logging.h>
#include <core/common/safeint.h>

#include <core/graph/graph.h>

#include "helper.h"
#include "op_support_checker.h"

namespace onnxruntime {
namespace nnapi {

using std::vector;

#pragma region helpers

bool HasExternalInitializer(const InitializedTensorSet& initializers, const Node& node) {
  for (const auto* node_arg : node.InputDefs()) {
    const auto& input_name(node_arg->Name());
    if (!Contains(initializers, input_name))
      continue;

    const auto& tensor = *initializers.at(input_name);
    if (tensor.has_data_location() &&
        tensor.data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL) {
      LOGS_DEFAULT(VERBOSE) << "Initializer [" << input_name
                            << "] with external data location are not currently supported";
      return true;
    }
  }

  return false;
}

#pragma endregion helpers

#pragma region op_base

class BaseOpSupportChecker : public IOpSupportChecker {
 public:
  virtual ~BaseOpSupportChecker() = default;
  bool IsOpSupported(const InitializedTensorSet& initializers, const Node& node,
                     const OpSupportCheckParams& params) const override;

 protected:
  virtual bool IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const Node& /* node */,
                                 const OpSupportCheckParams& /* params */) const {
    return true;
  }

  virtual int32_t GetMinSupportedSdkVer(const Node& /* node */, const OpSupportCheckParams& /* params */) const {
    // Android API level 27 is the baseline version of NNAPI,
    // There is no NNAPI support for Android API level 26-
    return 27;
  }

  virtual bool HasSupportedInputs(const Node& node) const;

  virtual int GetMinSupportedOpSet(const Node& /* node */) const { return 1; }
  virtual int GetMaxSupportedOpSet(const Node& /* node */) const { return 13; }
  bool HasSupportedOpSet(const Node& node) const;
};

bool BaseOpSupportChecker::IsOpSupported(const InitializedTensorSet& initializers, const Node& node,
                                         const OpSupportCheckParams& params) const {
  int32_t required_sdk_ver = GetMinSupportedSdkVer(node, params);
  if (required_sdk_ver > params.android_sdk_ver) {
    LOGS_DEFAULT(VERBOSE) << "Current Android API level [" << params.android_sdk_ver
                          << "], Operator [" << node.OpType()
                          << "] is only supported on API >" << required_sdk_ver;
    return false;
  }

  if (!HasSupportedInputs(node))
    return false;

  // We do not support external initializers for now
  if (HasExternalInitializer(initializers, node))
    return false;

  if (!HasSupportedOpSet(node))
    return false;

  return IsOpSupportedImpl(initializers, node, params);
}

bool BaseOpSupportChecker::HasSupportedInputs(const Node& node) const {
  // We only check the type of input 0 by default
  // specific op builder can override this
  const auto& input = *node.InputDefs()[0];

  if (nullptr == input.Shape()) {
    LOGS_DEFAULT(VERBOSE) << "[" << node.OpType()
                          << "] Input shape is null";
    return false;
  }

  int32_t input_type;
  if (!GetType(input, input_type))
    return false;

  if (input_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    LOGS_DEFAULT(VERBOSE) << "[" << node.OpType()
                          << "] Input type: [" << input_type
                          << "] is not supported for now";
    return false;
  }

  return true;
}

bool BaseOpSupportChecker::HasSupportedOpSet(const Node& node) const {
  auto since_version = node.SinceVersion();
  if (since_version < GetMinSupportedOpSet(node) || since_version > GetMaxSupportedOpSet(node)) {
    LOGS_DEFAULT(VERBOSE) << node.OpType() << "is only supported for opset ["
                          << GetMinSupportedOpSet(node) << ", "
                          << GetMaxSupportedOpSet(node) << "]";
    return false;
  }

  return true;
}

#pragma endregion op_base

#pragma region op_binary

class BinaryOpSupportChecker : public BaseOpSupportChecker {
 private:
  int32_t GetMinSupportedSdkVer(const Node& node, const OpSupportCheckParams& params) const override;
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const OpSupportCheckParams& params) const override;
  bool HasSupportedInputs(const Node& node) const override;
  int GetMinSupportedOpSet(const Node& node) const override;
};

int32_t BinaryOpSupportChecker::GetMinSupportedSdkVer(
    const Node& node, const OpSupportCheckParams& /* params */) const {
  const auto& op(node.OpType());
  if (op == "Sub" || op == "Div") {
    return 28;
  }
  return 27;
}

int BinaryOpSupportChecker::GetMinSupportedOpSet(const Node& node) const {
  const auto& op(node.OpType());

  // Add/Sub/Mul/Div opset 6- has broadcast attributes we do not support now
  if (op != "QLinearAdd")
    return 7;

  return 1;
}

bool BinaryOpSupportChecker::HasSupportedInputs(const Node& node) const {
  if (node.OpType() != "QLinearAdd")
    return BaseOpSupportChecker::HasSupportedInputs(node);

  // QLinearAdd
  if (!HasValidBinaryOpQuantizedInputs(node))
    return false;

  return true;
}

bool BinaryOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                                               const OpSupportCheckParams& /* params */) const {
  const auto& op_type(node.OpType());
  const auto input_defs(node.InputDefs());
  bool op_is_qlinear = op_type == "QLinearAdd";
  size_t a_idx = 0, b_idx = 1;
  if (op_is_qlinear) {
    b_idx = 3;
  }
  Shape input1_shape, input2_shape;
  if (!GetShape(*input_defs[a_idx], input1_shape) ||
      !GetShape(*input_defs[b_idx], input2_shape))
    return false;

  const auto input1_size = input1_shape.size();
  const auto input2_size = input2_shape.size();
  if (input1_size > 4 || input2_size > 4) {
    LOGS_DEFAULT(VERBOSE) << node.OpType() << " only support up to 4d shape, input1 is "
                          << input1_size << "d shape, input 2 is "
                          << input2_size << "d shape";
    return false;
  }

  if (op_is_qlinear) {
    // For QLinearAdd, we only support uint8 output now
    int32_t output_type;
    if (!GetType(*node.OutputDefs()[0], output_type))
      return false;

    if (output_type != ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
      LOGS_DEFAULT(VERBOSE) << "[" << op_type
                            << "] output type: [" << output_type
                            << "] is not supported for now";
      return false;
    }

    // All scale/zero points are initializer scalars
    // a/b/y_scale
    if (!HasValidQuantizationScales(initializers, node, {1, 4, 6}))
      return false;

    // a/b/y_zero_point
    if (!HasValidQuantizationZeroPoints(initializers, node, {2, 5, 7}))
      return false;
  }

  return true;
}

#pragma endregion

#pragma region op_transpose

class TransposeOpSupportChecker : public BaseOpSupportChecker {
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const OpSupportCheckParams& params) const override;

  int32_t GetMinSupportedSdkVer(const Node& /* node */, const OpSupportCheckParams& /* params */) const override {
    return 28;
  }
};

bool TransposeOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const Node& node,
                                                  const OpSupportCheckParams& /* params */) const {
  Shape input_shape;
  if (!GetShape(*node.InputDefs()[0], input_shape))
    return false;

  const auto input_size = input_shape.size();
  if (input_size > 4 || input_size == 0) {
    LOGS_DEFAULT(VERBOSE) << "Transpose only supports 1-4d shape, input is "
                          << input_size << "d shape";
    return false;
  }

  return true;
}

#pragma endregion

#pragma region op_reshape

class ReshapeOpSupportChecker : public BaseOpSupportChecker {
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const OpSupportCheckParams& params) const override;

  // Reshape opset 4- uses attributes for new shape which we do not support for now
  int GetMinSupportedOpSet(const Node& /* node */) const override { return 5; }
};

bool ReshapeOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                                                const OpSupportCheckParams& /* params */) const {
  const auto& perm_name = node.InputDefs()[1]->Name();
  if (!Contains(initializers, perm_name)) {
    LOGS_DEFAULT(VERBOSE) << "New shape of reshape must be known";
    return false;
  }

  Shape input_shape;
  if (!GetShape(*node.InputDefs()[0], input_shape))
    return false;

  if (input_shape.size() > 4 || input_shape.empty()) {
    LOGS_DEFAULT(VERBOSE) << "Reshape only supports up to 1-4d shape, input is "
                          << input_shape.size() << "d shape";
    return false;
  }

  const auto& shape_tensor = *initializers.at(perm_name);
  const int64_t* raw_shape = GetTensorInt64Data(shape_tensor);
  const auto size = SafeInt<uint32_t>(shape_tensor.dims()[0]);

  for (uint32_t i = 0; i < size; i++) {
    // NNAPI reshape does not support 0 as dimension
    if (raw_shape[i] == 0 && i < input_shape.size() && input_shape[i] == 0) {
      LOGS_DEFAULT(VERBOSE) << "Reshape doesn't suppport 0 reshape dimension on a dynamic dimension";
      return false;
    }
  }

  return true;
}

#pragma endregion

#pragma region op_batchnormalization

class BatchNormalizationOpSupportChecker : public BaseOpSupportChecker {
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const OpSupportCheckParams& params) const override;

  // BatchNormalization opset 6- has unsupported attributes
  int GetMinSupportedOpSet(const Node& /* node */) const override { return 7; }
};

bool BatchNormalizationOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                                                           const OpSupportCheckParams& /* params */) const {
  if (node.OutputDefs().size() != 1) {
    LOGS_DEFAULT(VERBOSE) << "Your onnx model may be in training mode, please export "
                             "it in test mode.";
    return false;
  }

  const auto& input_defs = node.InputDefs();
  Shape input_shape;
  if (!GetShape(*input_defs[0], input_shape))
    return false;

  const auto input_size = input_shape.size();
  if (input_size > 4) {
    LOGS_DEFAULT(VERBOSE) << "BN only support up to 4d shape, input is "
                          << input_size << "d shape";
    return false;
  }

  NodeAttrHelper helper(node);
  const auto spatial = helper.Get("spatial", 1);
  if (spatial != 1) {
    LOGS_DEFAULT(VERBOSE) << "Non-spatial BN is not supported";
    return false;
  }

  const auto& scale_name = input_defs[1]->Name();
  const auto& b_name = input_defs[2]->Name();
  const auto& mean_name = input_defs[3]->Name();
  const auto& var_name = input_defs[4]->Name();
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

#pragma endregion

#pragma region op_pool

class PoolOpSupportChecker : public BaseOpSupportChecker {
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const OpSupportCheckParams& params) const override;

  int32_t GetMinSupportedSdkVer(const Node& /* node */, const OpSupportCheckParams& params) const override {
    return params.use_nchw ? 29 : 28;
  }
};

bool PoolOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const Node& node,
                                             const OpSupportCheckParams& /* params */) const {
  const auto& op_type = node.OpType();
  Shape input_shape;
  if (!GetShape(*node.InputDefs()[0], input_shape))
    return false;

  const auto input_size = input_shape.size();
  if (input_size != 4) {
    LOGS_DEFAULT(VERBOSE)
        << op_type << " only supports rank-4 tensor, input ["
        << node.InputDefs()[0]->Name() << "] has actual dim count " << input_size;
    return false;
  }

  if (op_type == "AveragePool" || op_type == "MaxPool") {
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

    if (node.OutputDefs().size() != 1) {
      LOGS_DEFAULT(VERBOSE) << "Argmax in maxpooling is not supported";
      return false;
    }
  } else if (op_type != "GlobalAveragePool" && op_type != "GlobalMaxPool") {
    LOGS_DEFAULT(VERBOSE) << "PoolOpBuilder, unknown op: " << op_type;
    return false;
  }

  return true;
}

#pragma endregion op_pool

#pragma region op_conv

class ConvOpSupportChecker : public BaseOpSupportChecker {
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const OpSupportCheckParams& params) const override;

  int32_t GetMinSupportedSdkVer(const Node& /* node */, const OpSupportCheckParams& params) const override {
    return params.use_nchw ? 29 : 28;
  }

  bool HasSupportedInputs(const Node& node) const override;
};

bool ConvOpSupportChecker::HasSupportedInputs(const Node& node) const {
  if (node.OpType() != "QLinearConv")
    return BaseOpSupportChecker::HasSupportedInputs(node);

  // QLinearConv only supports input of uint8 for now
  if (!HasValidBinaryOpQuantizedInputs(node))
    return false;

  return true;
}

bool ConvOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                                             const OpSupportCheckParams& params) const {
  const auto& op_type = node.OpType();
  const auto input_defs = node.InputDefs();
  NodeAttrHelper helper(node);

  bool is_qlinear_conv = (op_type == "QLinearConv");
  size_t w_idx = is_qlinear_conv ? 3 : 1;
  const auto group = helper.Get("group", 1);
  const auto weight_name = input_defs[w_idx]->Name();
  if (Contains(initializers, weight_name)) {
    const auto& tensor = *initializers.at(weight_name);
    if (tensor.dims().size() != 4) {
      LOGS_DEFAULT(VERBOSE) << "Only conv 2d is supported.";
      return false;
    }

    const auto onnx_dilations = helper.Get("dilations", vector<int>{1, 1});
    if (onnx_dilations != vector<int>{1, 1}) {
      if (group != 1 && tensor.dims()[1] != 1) {
        LOGS_DEFAULT(VERBOSE) << "dilation is not supported on grouped conv";
        return false;
      }

      if (params.android_sdk_ver < 29) {
        LOGS_DEFAULT(VERBOSE) << op_type << " dilations is only supported on Android API level 29+, "
                              << "actual API level: " << params.android_sdk_ver;
        return false;
      }
    }
  } else {
    LOGS_DEFAULT(VERBOSE) << "The weight of convolution must be known";
    return false;
  }

  if (is_qlinear_conv) {
    // For QLinearConv, we only support uint8 output now
    int32_t output_type;
    if (!GetType(*node.OutputDefs()[0], output_type))
      return false;

    if (output_type != ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
      LOGS_DEFAULT(VERBOSE) << "[" << op_type
                            << "] output type: [" << output_type
                            << "] is not supported for now";
      return false;
    }

    if (input_defs.size() > 8 && !Contains(initializers, input_defs[8]->Name())) {
      LOGS_DEFAULT(VERBOSE) << "Bias of QLinearConv must be known";
      return false;
    }

    // a/b/y_scale
    if (!HasValidQuantizationScales(initializers, node, {1, 4, 6}))
      return false;

    // a/b/y_zero_point
    if (!HasValidQuantizationZeroPoints(initializers, node, {2, 5, 7}))
      return false;
  }

  return true;
}

#pragma endregion

#pragma region op_cast

class CastOpSupportChecker : public BaseOpSupportChecker {
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const OpSupportCheckParams& params) const override;

  int32_t GetMinSupportedSdkVer(const Node& /* node */, const OpSupportCheckParams& /* params */) const override {
    return 29;
  }

  // Cast opset 5- uses string attribute for to type, is not supported for now
  int GetMinSupportedOpSet(const Node& /* node */) const override { return 6; }
};

bool CastOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const Node& node,
                                             const OpSupportCheckParams& /* params */) const {
  NodeAttrHelper helper(node);
  const auto to = helper.Get("to", 0);
  if (to != ONNX_NAMESPACE::TensorProto::FLOAT &&
      to != ONNX_NAMESPACE::TensorProto::INT32) {
    LOGS_DEFAULT(VERBOSE) << "[Cast] Only support cast to int32 or float, actual to type, " << to;
    return false;
  }

  return true;
}

#pragma endregion

#pragma region op_softmax

class SoftMaxOpSupportChecker : public BaseOpSupportChecker {
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const OpSupportCheckParams& params) const override;

  int32_t GetMinSupportedSdkVer(const Node& /* node */, const OpSupportCheckParams& /* params */) const override {
    return 28;
  }
};

bool SoftMaxOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const Node& node,
                                                const OpSupportCheckParams& params) const {
  Shape input_shape;
  if (!GetShape(*node.InputDefs()[0], input_shape))
    return false;

  const auto input_size = input_shape.size();
  if (input_size != 2 && input_size != 4) {
    LOGS_DEFAULT(VERBOSE) << "SoftMax only support 2d/4d shape, input is "
                          << input_size << "d shape";
    return false;
  }

  if (params.android_sdk_ver < 29) {
    NodeAttrHelper helper(node);
    int32_t axis = helper.Get("axis", 1);
    if (axis != 1) {
      LOGS_DEFAULT(VERBOSE)
          << "SoftMax only support axis 1 on Android API level: " << params.android_sdk_ver
          << " input axis: " << axis;
      return false;
    }
  }

  return true;
}

#pragma endregion

#pragma region op_gemm

class GemmOpSupportChecker : public BaseOpSupportChecker {
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const OpSupportCheckParams& params) const override;
  bool HasSupportedInputs(const Node& node) const override;
  int GetMinSupportedOpSet(const Node& node) const override;
};

bool GemmOpSupportChecker::HasSupportedInputs(const Node& node) const {
  if (node.OpType() != "QLinearMatMul")
    return BaseOpSupportChecker::HasSupportedInputs(node);

  // QLinearMatMul
  if (!HasValidBinaryOpQuantizedInputs(node))
    return false;

  return true;
}

int GemmOpSupportChecker::GetMinSupportedOpSet(const Node& node) const {
  const auto& op(node.OpType());

  // Gemm opset 6- has broadcast attributes we do not support now
  if (op == "Gemm")
    return 7;

  return 1;
}

bool GemmOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                                             const OpSupportCheckParams& /* params */) const {
  const auto& op_type = node.OpType();
  const auto input_defs(node.InputDefs());
  size_t a_idx = 0, b_idx = 1, c_idx = 2;  // A*B+C
  bool is_qlinear_matmul = op_type == "QLinearMatMul";
  if (is_qlinear_matmul) {
    a_idx = 0;
    b_idx = 3;
  }

  Shape a_shape;
  {
    if (!GetShape(*input_defs[a_idx], a_shape))
      return false;

    if (a_shape.size() != 2) {
      LOGS_DEFAULT(VERBOSE) << "A must be 2D";
      return false;
    }
  }

  Shape b_shape;
  {
    if (!GetShape(*input_defs[b_idx], b_shape))
      return false;

    if (b_shape.size() != 2) {
      LOGS_DEFAULT(VERBOSE) << "B must be 2D";
      return false;
    }
  }

  if (op_type == "Gemm") {
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

    if (transB == 0 && !Contains(initializers, input_defs[b_idx]->Name())) {
      LOGS_DEFAULT(VERBOSE) << "B of Gemm must be known if transB != 1";
      return false;
    }

    if (input_defs.size() == 3) {
      Shape c_shape;
      if (!GetShape(*input_defs[c_idx], c_shape))
        return false;

      if (c_shape.size() != 1 ||
          c_shape[0] != (transB == 0 ? b_shape[1] : b_shape[0])) {
        LOGS_DEFAULT(VERBOSE) << "C of Gemm must be a vector of b_shape[0]"
                              << " b_shape: " << Shape2String(b_shape)
                              << " c_shape: " << Shape2String(c_shape);

        return false;
      }
    }
  } else if (op_type == "MatMul" || is_qlinear_matmul) {
    // Only support A*B B is an initializer
    if (!Contains(initializers, input_defs[b_idx]->Name())) {
      LOGS_DEFAULT(VERBOSE) << "B of MatMul must be known";
      return false;
    }

    if (is_qlinear_matmul) {
      // For QLinearMatMul, we only support uint8 output now
      int32_t output_type;
      if (!GetType(*node.OutputDefs()[0], output_type))
        return false;

      if (output_type != ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
        LOGS_DEFAULT(VERBOSE) << "[" << op_type
                              << "] output type: [" << output_type
                              << "] is not supported for now";
        return false;
      }

      // All scale/zero points are initializer scalars
      // a/b/y_scale
      if (!HasValidQuantizationScales(initializers, node, {1, 4, 6}))
        return false;

      // a/b/y_zero_point
      if (!HasValidQuantizationZeroPoints(initializers, node, {2, 5, 7}))
        return false;
    }
  } else {
    LOGS_DEFAULT(VERBOSE) << "GemmOpSupportChecker, unknown op: " << op_type;
  }

  return true;
}

#pragma endregion

#pragma region op_unary

class UnaryOpSupportChecker : public BaseOpSupportChecker {
 private:
  int32_t GetMinSupportedSdkVer(const Node& node, const OpSupportCheckParams& params) const override;

  // All ops except "Sin" opset 5- uses consumed_inputs attribute which is not supported for now
  // "Sin" op has support from opset 7, return 6 here for all ops
  int GetMinSupportedOpSet(const Node& /* node */) const override { return 6; }
};

int32_t UnaryOpSupportChecker::GetMinSupportedSdkVer(
    const Node& node, const OpSupportCheckParams& /* params */) const {
  const auto& op(node.OpType());
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

#pragma endregion

#pragma region op_concat

class ConcatOpSupportChecker : public BaseOpSupportChecker {
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const OpSupportCheckParams& params) const override;
};

bool ConcatOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const Node& node,
                                               const OpSupportCheckParams& /* params */) const {
  Shape input_shape;
  if (!GetShape(*node.InputDefs()[0], input_shape))
    return false;

  const auto input_size = input_shape.size();
  if (input_size > 4 || input_size == 0) {
    LOGS_DEFAULT(VERBOSE) << "Concat only supports up to 1-4d shape, input is "
                          << input_size << "d shape";
    return false;
  }

  return true;
}

#pragma endregion

#pragma region op_squeeze

class SqueezeOpSupportChecker : public BaseOpSupportChecker {
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const OpSupportCheckParams& params) const override;

  int32_t GetMinSupportedSdkVer(const Node& /* node */, const OpSupportCheckParams& /* params */) const override {
    return 28;
  }

  // Squeeze opset 13+ uses input for axes, which is not supported yet
  // TODO add support for squeeze opset 13+
  int GetMaxSupportedOpSet(const Node& /* node */) const override { return 12; }
};

bool SqueezeOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const Node& node,
                                                const OpSupportCheckParams& /* params */) const {
  Shape input_shape;
  if (!GetShape(*node.InputDefs()[0], input_shape))
    return false;

  const auto input_size = input_shape.size();
  if (input_size > 4 || input_size == 0) {
    LOGS_DEFAULT(VERBOSE) << "Squeeze only supports 1-4d shape, input is "
                          << input_size << "d shape";
    return false;
  }

  return true;
}

#pragma endregion

#pragma region op_quantizelinear

class QuantizeLinearOpSupportChecker : public BaseOpSupportChecker {
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const OpSupportCheckParams& params) const override;

  int32_t GetMinSupportedSdkVer(const Node& /* node */, const OpSupportCheckParams& /* params */) const override {
    return 27;
  }
};

bool QuantizeLinearOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                                                       const OpSupportCheckParams& /* params */) const {
  const auto input_defs(node.InputDefs());
  const auto output_defs(node.OutputDefs());

  int32_t output_type;
  if (!GetType(*output_defs[0], output_type))
    return false;

  if (output_type != ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
    LOGS_DEFAULT(VERBOSE) << "[" << node.OpType()
                          << "] output type: [" << output_type
                          << "] is not supported for now";
    return false;
  }

  if (!HasValidQuantizationScales(initializers, node, {1}))
    return false;

  if (input_defs.size() == 3) {  // has zero_point input
    if (!HasValidQuantizationZeroPoints(initializers, node, {2}))
      return false;
  }

  return true;
}

#pragma endregion

#pragma region op_dequantizelinear

class DequantizeLinearOpSupportChecker : public BaseOpSupportChecker {
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const OpSupportCheckParams& params) const override;

  int32_t GetMinSupportedSdkVer(const Node& /* node */, const OpSupportCheckParams& /* params */) const override {
    return 29;
  }
  bool HasSupportedInputs(const Node& node) const override;
};

bool DequantizeLinearOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                                                         const OpSupportCheckParams& /* params */) const {
  const auto input_defs(node.InputDefs());
  if (!HasValidQuantizationScales(initializers, node, {1}))
    return false;

  if (input_defs.size() == 3) {  // has zero_point input
    if (!HasValidQuantizationZeroPoints(initializers, node, {2}))
      return false;
  }

  return true;
}

bool DequantizeLinearOpSupportChecker::HasSupportedInputs(const Node& node) const {
  int32_t input_type;
  if (!GetType(*node.InputDefs()[0], input_type))
    return false;

  if (input_type != ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
    LOGS_DEFAULT(VERBOSE) << "[" << node.OpType()
                          << "] Input type: [" << input_type
                          << "] is not supported for now";
    return false;
  }

  return true;
}

#pragma endregion

#pragma region op_LRN

class LRNOpSupportChecker : public BaseOpSupportChecker {
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const OpSupportCheckParams& params) const override;

  int32_t GetMinSupportedSdkVer(const Node& /* node */, const OpSupportCheckParams& /* params */) const override {
    return 28;
  }
};

bool LRNOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const Node& node,
                                            const OpSupportCheckParams& /* params */) const {
  Shape input_shape;
  if (!GetShape(*node.InputDefs()[0], input_shape))
    return false;

  const auto input_size = input_shape.size();
  if (input_size != 4) {
    LOGS_DEFAULT(VERBOSE) << "LRN only support 4d shape, input is "
                          << input_size << "d shape";
    return false;
  }

  return true;
}

#pragma endregion

#pragma region op_clip

class ClipOpSupportChecker : public BaseOpSupportChecker {
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const OpSupportCheckParams& params) const override;
};

bool ClipOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                                             const OpSupportCheckParams& /* params */) const {
  float min, max;
  if (!GetClipMinMax(initializers, node, min, max))
    return false;

  // We only supoort relu6 or relu1
  // TODO, support clip between 2 arbitrary numbers
  if ((min == 0.0f && max == 6.0f) || (min == -1.0f && max == 1.0f)) {
    return true;
  } else {
    LOGS_DEFAULT(VERBOSE) << "Clip only supports [min, max] = [0, 6] or [-1, 1], the input is ["
                          << min << ", " << max << "]";
    return false;
  }

  return true;
}

#pragma endregion

#pragma region op_Resize

class ResizeOpSupportChecker : public BaseOpSupportChecker {
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const OpSupportCheckParams& params) const override;

  int32_t GetMinSupportedSdkVer(const Node& /* node */, const OpSupportCheckParams& /* params */) const override {
    return 28;
  }

  // Resize opset 10- is very different than Resize opset 11+, with many key attributes missing
  // We only support Resize opset 11+ here
  int GetMinSupportedOpSet(const Node& /* node */) const override { return 11; }
};

bool ResizeOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                                               const OpSupportCheckParams& params) const {
  Shape input_shape;
  if (!GetShape(*node.InputDefs()[0], input_shape))
    return false;

  const auto input_size = input_shape.size();
  if (input_size != 4) {
    LOGS_DEFAULT(VERBOSE) << "Resize only support 4d shape, input is "
                          << input_size << "d shape";
    return false;
  }

  {  // check attributes
    NodeAttrHelper helper(node);
    const auto mode = helper.Get("mode", "nearest");
    if (mode != "linear") {
      LOGS_DEFAULT(VERBOSE) << "Resize unsupported input mode, " << mode;
      return false;
    }

    const auto coord_trans_mode = helper.Get("coordinate_transformation_mode", "half_pixel");
    bool using_half_pixel = coord_trans_mode == "half_pixel";
    bool using_align_corners = coord_trans_mode == "align_corners";
    if (!using_half_pixel && !using_align_corners && coord_trans_mode != "asymmetric") {
      LOGS_DEFAULT(VERBOSE) << "Resize, unsupported coord_trans_mode, " << coord_trans_mode;
      return false;
    }

    if (params.android_sdk_ver < 30 && (using_half_pixel || using_align_corners)) {
      LOGS_DEFAULT(VERBOSE) << "Resize only support half_pixel/align_corners on API level 30+, current API level is "
                            << params.android_sdk_ver;
      return false;
    }

    const auto exclude_outside = helper.Get("exclude_outside", 0);
    if (exclude_outside != 0) {
      LOGS_DEFAULT(VERBOSE) << "Resize does not support exclude_outside for now";
      return false;
    }
  }

  {  // scales and sizes (if present) must be initializers
    const auto input_defs = node.InputDefs();
    // scales
    if (input_defs.size() < 3 || !Contains(initializers, input_defs[2]->Name())) {
      LOGS_DEFAULT(VERBOSE) << "Input scales of Resize must be known";
      return false;
    }

    // sizes
    if (input_defs.size() > 3 && !Contains(initializers, input_defs[3]->Name())) {
      LOGS_DEFAULT(VERBOSE) << "Input sizes of Resize must be known";
      return false;
    }

    // We want to check if the scales or sizes are not trying to resize on N/C channels here
    if (input_defs.size() == 3) {  // we are using scales
      const auto& scales_tensor = *initializers.at(input_defs[2]->Name());
      const float* scales_data = GetTensorFloatData(scales_tensor);
      float scale_n = scales_data[0];
      float scale_c = scales_data[1];
      if (scale_n != 1.0f || scale_c != 1.0f) {
        LOGS_DEFAULT(VERBOSE) << "Scales of N/C channel should be 1"
                              << "Resize of N/C channels are not supported"
                              << ", scale_n, " << scale_n << ", scale_c, " << scale_c;
        return false;
      }
    } else {
      // we are using sizes
      const auto& sizes_name = input_defs[3]->Name();
      const auto& sizes_tensor = *initializers.at(sizes_name);
      const int64_t* sizes_data = GetTensorInt64Data(sizes_tensor);
      uint32_t size_n = SafeInt<uint32_t>(sizes_data[0]);
      uint32_t size_c = SafeInt<uint32_t>(sizes_data[1]);
      if (size_n != input_shape[0] || size_c != input_shape[1]) {
        LOGS_DEFAULT(VERBOSE) << "Output sizes of N/C chanel should match the input sizes, "
                              << "Resize of N/C channels are not supported"
                              << ", input_size_n, " << input_shape[0] << ", output_size_n, " << size_n
                              << ". input_size_c, " << input_shape[1] << ", output_size_c, " << size_c;
        return false;
      }
    }
  }
  return true;
}

#pragma endregion

#pragma region op_flatten

class FlattenOpSupportChecker : public BaseOpSupportChecker {
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const OpSupportCheckParams& params) const override;
};

bool FlattenOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const Node& node,
                                                const OpSupportCheckParams& /* params */) const {
  Shape input_shape;
  if (!GetShape(*node.InputDefs()[0], input_shape))
    return false;

  if (input_shape.size() > 4 || input_shape.empty()) {
    LOGS_DEFAULT(VERBOSE) << "Flatten only supports up to 1-4d shape, input is "
                          << input_shape.size() << "d shape";
    return false;
  }

  int32_t dim_1 = 1;
  int32_t dim_2 = 1;
  GetFlattenOutputShape(node, input_shape, dim_1, dim_2);

  if (dim_1 == 0 && dim_2 == 0) {
    LOGS_DEFAULT(VERBOSE) << "The dynamical input shape " << Shape2String(input_shape)
                          << " is not supported";
    return false;
  }

  return true;
}

#pragma endregion

#pragma region CreateGetOpSupportCheckers

static std::unordered_map<std::string, std::shared_ptr<IOpSupportChecker>> CreateOpSupportCheckers() {
  std::unordered_map<std::string, std::shared_ptr<IOpSupportChecker>> op_map;

  // If an OP is always supported, we use BaseOpSupportChecker as default
  // Relu, Identity are using base_op_support_checker
  auto base_op_support_checker = std::make_shared<BaseOpSupportChecker>();

  {
    auto binary_op_support_checker = std::make_shared<BinaryOpSupportChecker>();
    op_map.emplace("Add", binary_op_support_checker);
    op_map.emplace("Sub", binary_op_support_checker);
    op_map.emplace("Mul", binary_op_support_checker);
    op_map.emplace("Div", binary_op_support_checker);
    op_map.emplace("QLinearAdd", binary_op_support_checker);
  }

  op_map.emplace("Relu", base_op_support_checker);
  op_map.emplace("Transpose", std::make_shared<TransposeOpSupportChecker>());
  op_map.emplace("Reshape", std::make_shared<ReshapeOpSupportChecker>());
  op_map.emplace("BatchNormalization", std::make_shared<BatchNormalizationOpSupportChecker>());

  {
    auto pool_op_support_checker = std::make_shared<PoolOpSupportChecker>();
    op_map.emplace("GlobalAveragePool", pool_op_support_checker);
    op_map.emplace("GlobalMaxPool", pool_op_support_checker);
    op_map.emplace("AveragePool", pool_op_support_checker);
    op_map.emplace("MaxPool", pool_op_support_checker);
  }

  {
    auto conv_op_support_checker = std::make_shared<ConvOpSupportChecker>();
    op_map.emplace("Conv", conv_op_support_checker);
    op_map.emplace("QLinearConv", conv_op_support_checker);
  }

  op_map.emplace("Cast", std::make_shared<CastOpSupportChecker>());
  op_map.emplace("Softmax", std::make_shared<SoftMaxOpSupportChecker>());
  op_map.emplace("Identity", base_op_support_checker);

  {
    auto gemm_op_support_checker = std::make_shared<GemmOpSupportChecker>();
    op_map.emplace("Gemm", gemm_op_support_checker);
    op_map.emplace("MatMul", gemm_op_support_checker);
    op_map.emplace("QLinearMatMul", gemm_op_support_checker);
  }

  {
    auto unary_op_support_checker = std::make_shared<UnaryOpSupportChecker>();
    op_map.emplace("Abs", unary_op_support_checker);
    op_map.emplace("Exp", unary_op_support_checker);
    op_map.emplace("Floor", unary_op_support_checker);
    op_map.emplace("Log", unary_op_support_checker);
    op_map.emplace("Sigmoid", unary_op_support_checker);
    op_map.emplace("Neg", unary_op_support_checker);
    op_map.emplace("Sin", unary_op_support_checker);
    op_map.emplace("Sqrt", unary_op_support_checker);
    op_map.emplace("Tanh", unary_op_support_checker);
  }

  op_map.emplace("Concat", std::make_shared<ConcatOpSupportChecker>());
  op_map.emplace("Squeeze", std::make_shared<SqueezeOpSupportChecker>());
  op_map.emplace("QuantizeLinear", std::make_shared<QuantizeLinearOpSupportChecker>());
  op_map.emplace("DequantizeLinear", std::make_shared<DequantizeLinearOpSupportChecker>());
  op_map.emplace("LRN", std::make_shared<LRNOpSupportChecker>());
  op_map.emplace("Clip", std::make_shared<ClipOpSupportChecker>());
  op_map.emplace("Resize", std::make_shared<ResizeOpSupportChecker>());
  op_map.emplace("Flatten", std::make_shared<FlattenOpSupportChecker>());

  return op_map;
}

const std::unordered_map<std::string, std::shared_ptr<IOpSupportChecker>>& GetOpSupportCheckers() {
  static std::unordered_map<std::string, std::shared_ptr<IOpSupportChecker>> op_map = CreateOpSupportCheckers();
  return op_map;
}

#pragma endregion

}  // namespace nnapi
}  // namespace onnxruntime