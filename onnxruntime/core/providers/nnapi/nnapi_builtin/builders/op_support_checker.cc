// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/common/logging/logging.h>
#include <core/common/safeint.h>

#include <core/graph/graph.h>

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "helper.h"
#include "op_support_checker.h"

namespace onnxruntime {
namespace nnapi {

using std::vector;

#pragma region helpers

struct OpSupportCheckerRegistrations {
  std::vector<std::unique_ptr<IOpSupportChecker>> support_checkers;
  std::unordered_map<std::string, const IOpSupportChecker*> op_support_checker_map;
};

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

template <class T>
void CreateSharedOpSupportCheckerImpl(const std::string& op_type,
                                      OpSupportCheckerRegistrations& op_registrations,
                                      const std::vector<std::string>& op_types) {
  // The shared OpSupportChecker is already in the OpSupportCheckerRegistrations
  if (op_registrations.op_support_checker_map.find(op_type) != op_registrations.op_support_checker_map.cend())
    return;

  op_registrations.support_checkers.push_back(std::make_unique<T>());
  for (const auto& op : op_types) {
    op_registrations.op_support_checker_map.emplace(op, op_registrations.support_checkers.back().get());
  }
}

#pragma endregion helpers

#pragma region op_base

class BaseOpSupportChecker : public IOpSupportChecker {
 public:
  virtual ~BaseOpSupportChecker() = default;
  bool IsOpSupported(const InitializedTensorSet& initializers, const Node& node,
                     const OpSupportCheckParams& params) const override;

  // This is for ops which are by default supported and do not have their own impl of OpSupportChecker
  // for those ops (Relu, Identity) we use BaseOpSupportChecker
  static void CreateSharedOpSupportChecker(
      const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);

 protected:
  virtual bool IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const Node& /* node */,
                                 const OpSupportCheckParams& /* params */) const {
    return true;
  }

  virtual int32_t GetMinSupportedNNAPIFeatureLevel(const Node& /* node */, const OpSupportCheckParams& /* params */) const {
    // ANEURALNETWORKS_FEATURE_LEVEL_1 is the baseline version of NNAPI,
    // There is no NNAPI support for Android API level 26-
    return ANEURALNETWORKS_FEATURE_LEVEL_1;
  }

  virtual bool HasSupportedInputsImpl(const Node& node) const;

  virtual int GetMinSupportedOpSet(const Node& /* node */) const { return 1; }
  virtual int GetMaxSupportedOpSet(const Node& /* node */) const { return 14; }

 private:
  bool HasSupportedOpSet(const Node& node) const;
  bool HasSupportedInputs(const Node& node) const;
};

/* static */ void BaseOpSupportChecker::CreateSharedOpSupportChecker(
    const std::string& op_type, OpSupportCheckerRegistrations& op_registrations) {
  CreateSharedOpSupportCheckerImpl<BaseOpSupportChecker>(
      op_type, op_registrations,
      {
          "Relu",
          "Identity",
      });
}

bool BaseOpSupportChecker::IsOpSupported(const InitializedTensorSet& initializers, const Node& node,
                                         const OpSupportCheckParams& params) const {
  int32_t required_feature_level = GetMinSupportedNNAPIFeatureLevel(node, params);
  if (required_feature_level > params.android_feature_level) {
    LOGS_DEFAULT(VERBOSE) << "Current Android API level [" << params.android_feature_level
                          << "], Operator [" << node.OpType()
                          << "] is only supported on API >" << required_feature_level;
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
  // We do not support unknown(null) input shape
  for (const auto* input : node.InputDefs()) {
    if (!input->Shape()) {
      LOGS_DEFAULT(VERBOSE) << "Node [" << node.Name() << "] type [" << node.OpType()
                            << "] Input [" << input->Name() << "] has no shape";
      return false;
    }
  }

  return HasSupportedInputsImpl(node);
}

bool BaseOpSupportChecker::HasSupportedInputsImpl(const Node& node) const {
  // We only check the type of input 0 by default
  // specific op builder can override this
  const auto& input = *node.InputDefs()[0];

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
    LOGS_DEFAULT(VERBOSE) << node.OpType() << " opset [" << since_version
                          << "] is only supported for opset ["
                          << GetMinSupportedOpSet(node) << ", "
                          << GetMaxSupportedOpSet(node) << "]";
    return false;
  }

  return true;
}

#pragma endregion op_base

#pragma region op_binary

class BinaryOpSupportChecker : public BaseOpSupportChecker {
 public:
  static void CreateSharedOpSupportChecker(
      const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);

 private:
  int32_t GetMinSupportedNNAPIFeatureLevel(const Node& node, const OpSupportCheckParams& params) const override;
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const OpSupportCheckParams& params) const override;
  bool HasSupportedInputsImpl(const Node& node) const override;
  int GetMinSupportedOpSet(const Node& node) const override;
};

/* static */ void BinaryOpSupportChecker::CreateSharedOpSupportChecker(
    const std::string& op_type, OpSupportCheckerRegistrations& op_registrations) {
  CreateSharedOpSupportCheckerImpl<BinaryOpSupportChecker>(
      op_type, op_registrations,
      {
          "Add",
          "Sub",
          "Mul",
          "Div",
          "QLinearAdd",
          "Pow",
      });
}

int32_t BinaryOpSupportChecker::GetMinSupportedNNAPIFeatureLevel(
    const Node& node, const OpSupportCheckParams& /* params */) const {
  const auto& op(node.OpType());
  if (op == "Sub" || op == "Div") {
    return ANEURALNETWORKS_FEATURE_LEVEL_2;
  }

  if (op == "Pow") {
    return ANEURALNETWORKS_FEATURE_LEVEL_3;
  }

  return ANEURALNETWORKS_FEATURE_LEVEL_1;
}

int BinaryOpSupportChecker::GetMinSupportedOpSet(const Node& node) const {
  const auto& op(node.OpType());

  // Add/Sub/Mul/Div/Pow opset 6- has broadcast attributes we do not support now
  if (op != "QLinearAdd")
    return 7;

  return 1;
}

bool BinaryOpSupportChecker::HasSupportedInputsImpl(const Node& node) const {
  bool is_qlinear_add = node.OpType() == "QLinearAdd";
  bool is_pow = node.OpType() == "Pow";
  if (!is_qlinear_add && !is_pow)
    return BaseOpSupportChecker::HasSupportedInputsImpl(node);

  if (is_qlinear_add) {
    // QLinearAdd
    if (!HasValidBinaryOpQuantizedInputs(node))
      return false;
  }

  // Pow we only support both input as fp32 now
  if (is_pow) {
    const auto& input1 = *node.InputDefs()[0];
    const auto& input2 = *node.InputDefs()[1];

    int32_t input_type_1;
    if (!GetType(input1, input_type_1))
      return false;

    int32_t input_type_2;
    if (!GetType(input2, input_type_2))
      return false;

    if (input_type_1 != ONNX_NAMESPACE::TensorProto_DataType_FLOAT || input_type_1 != input_type_2) {
      LOGS_DEFAULT(VERBOSE) << "Pow only supports fp32 inputs, actual input type"
                            << ", Input type 1: " << input_type_1
                            << ", Input type 2: " << input_type_2;
      return false;
    }
  }

  return true;
}

bool BinaryOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                                               const OpSupportCheckParams& params) const {
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
    if (!HasValidQuantizationScales(initializers, node, {1, 4, 6}, params))
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

  int32_t GetMinSupportedNNAPIFeatureLevel(const Node& /* node */, const OpSupportCheckParams& /* params */) const override {
    return ANEURALNETWORKS_FEATURE_LEVEL_2;
  }

  bool HasSupportedInputsImpl(const Node& node) const override;
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

bool TransposeOpSupportChecker::HasSupportedInputsImpl(const Node& node) const {
  int32_t input_type;
  if (!GetType(*node.InputDefs()[0], input_type))
    return false;

  if (input_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
      input_type != ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
    LOGS_DEFAULT(VERBOSE) << "[" << node.OpType()
                          << "] Input type: [" << input_type
                          << "] is not supported for now";
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

  const auto& perm_tensor = *initializers.at(perm_name);
  const int64_t* raw_perm = GetTensorInt64Data(perm_tensor);
  const auto perm_size = SafeInt<uint32_t>(perm_tensor.dims()[0]);

  NodeAttrHelper helper(node);
  const bool allow_zero = helper.Get("allowzero ", 0) == 1;
  for (uint32_t i = 0; i < perm_size; i++) {
    // NNAPI reshape does not support 0 as dimension
    if (raw_perm[i] == 0) {
      if (i < input_shape.size() && input_shape[i] == 0) {
        LOGS_DEFAULT(VERBOSE) << "Reshape doesn't support 0 reshape dimension on a dynamic dimension";
        return false;
      }

      if (allow_zero) {
        LOGS_DEFAULT(VERBOSE) << "Reshape doesn't support 0 reshape dimension when allowzero is enabled";
        return false;
      }
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
 public:
  static void CreateSharedOpSupportChecker(
      const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);

 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const OpSupportCheckParams& params) const override;

  int32_t GetMinSupportedNNAPIFeatureLevel(const Node& /* node */, const OpSupportCheckParams& params) const override {
    return params.use_nchw ? ANEURALNETWORKS_FEATURE_LEVEL_3 : ANEURALNETWORKS_FEATURE_LEVEL_2;
  }

  bool HasSupportedInputsImpl(const Node& node) const override;
};

/* static */ void PoolOpSupportChecker::CreateSharedOpSupportChecker(
    const std::string& op_type, OpSupportCheckerRegistrations& op_registrations) {
  CreateSharedOpSupportCheckerImpl<PoolOpSupportChecker>(
      op_type, op_registrations,
      {
          "GlobalAveragePool",
          "GlobalMaxPool",
          "AveragePool",
          "MaxPool",
          "QLinearAveragePool",
      });
}

bool PoolOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                                             const OpSupportCheckParams& params) const {
  const auto& op_name = node.Name();
  const auto& op_type = node.OpType();
  const auto& input_defs = node.InputDefs();
  Shape input_shape;
  if (!GetShape(*input_defs[0], input_shape))
    return false;

  const auto input_size = input_shape.size();
  if (input_size != 4) {
    LOGS_DEFAULT(VERBOSE)
        << op_type << " only supports rank-4 tensor, input ["
        << input_defs[0]->Name() << "] has actual dim count " << input_size;
    return false;
  }

  bool is_qlinear_average_pool = op_type == "QLinearAveragePool";
  if (op_type == "AveragePool" || op_type == "MaxPool" || is_qlinear_average_pool) {
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

  // We need to check if we have valid scales and zero points for QLinearAveragePool
  if (is_qlinear_average_pool) {
    if (input_defs.size() < 4)
      return false;

    // the output zero point can be optional
    bool has_output_zp = input_defs.size() == 5;

    if (!HasValidQuantizationScales(initializers, node, {1, 3}, params))
      return false;

    if (!HasValidQuantizationZeroPoints(initializers, node,
                                        has_output_zp
                                            ? std::vector<size_t>{2}
                                            : std::vector<size_t>{2, 4})) {
      return false;
    }

    // NNAPI requires Quantized Average Pool has same scale and zero point for both input and output
    auto input_scale = GetQuantizationScale(initializers, node, 1);
    auto output_scale = GetQuantizationScale(initializers, node, 3);
    if (input_scale != output_scale) {
      LOGS_DEFAULT(VERBOSE) << "Op [" << op_type << "] name [" << op_name
                            << "] has different input_scale: " << input_scale
                            << " than the output_scale: " << output_scale;
      return false;
    }

    int32_t input_zp = 0;
    int32_t output_zp = 0;
    auto status = GetQuantizationZeroPoint(initializers, node, 2, input_zp);
    if (!status.IsOK()) {
      LOGS_DEFAULT(ERROR) << "Op [" << op_type << "] name [" << op_name
                          << "] GetQuantizationZeroPoint for input_zp failed, message: "
                          << status.ErrorMessage();
      return false;
    }

    if (has_output_zp) {
      status = GetQuantizationZeroPoint(initializers, node, 4, output_zp);
      if (!status.IsOK()) {
        LOGS_DEFAULT(ERROR) << "Op [" << op_type << "] name [" << op_name
                            << "] GetQuantizationZeroPoint for output_zp failed, message: "
                            << status.ErrorMessage();
        return false;
      }
    }

    if (input_zp != output_zp) {
      LOGS_DEFAULT(VERBOSE) << "Op [" << op_type << "] name [" << op_name
                            << "] has different input_zp: " << input_zp
                            << " than the output_zp: " << output_zp;
      return false;
    }
  }

  return true;
}

bool PoolOpSupportChecker::HasSupportedInputsImpl(const Node& node) const {
  bool is_max_pool = node.OpType() == "MaxPool";
  bool is_qlinear_average_pool = node.OpType() == "QLinearAveragePool";
  if (!is_max_pool && !is_qlinear_average_pool)
    return BaseOpSupportChecker::HasSupportedInputsImpl(node);

  if (is_qlinear_average_pool) {
    return HasValidUnaryOpQuantizedInputs(node);
  }

  // is_max_pool
  // For max pool, we can support both float and uint8 input
  int32_t input_type;
  if (!GetType(*node.InputDefs()[0], input_type))
    return false;

  if (input_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
      input_type != ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
    LOGS_DEFAULT(VERBOSE) << "[" << node.OpType()
                          << "] Input type: [" << input_type
                          << "] is not supported for now";
    return false;
  }

  return true;
}

#pragma endregion op_pool

#pragma region op_conv

class ConvOpSupportChecker : public BaseOpSupportChecker {
 public:
  static void CreateSharedOpSupportChecker(
      const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);

 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const OpSupportCheckParams& params) const override;

  int32_t GetMinSupportedNNAPIFeatureLevel(const Node& /* node */, const OpSupportCheckParams& params) const override {
    return params.use_nchw ? ANEURALNETWORKS_FEATURE_LEVEL_3 : ANEURALNETWORKS_FEATURE_LEVEL_2;
  }

  bool HasSupportedInputsImpl(const Node& node) const override;
};

/* static */ void ConvOpSupportChecker::CreateSharedOpSupportChecker(
    const std::string& op_type, OpSupportCheckerRegistrations& op_registrations) {
  CreateSharedOpSupportCheckerImpl<ConvOpSupportChecker>(
      op_type, op_registrations,
      {
          "Conv",
          "QLinearConv",
      });
}

bool ConvOpSupportChecker::HasSupportedInputsImpl(const Node& node) const {
  if (node.OpType() != "QLinearConv")
    return BaseOpSupportChecker::HasSupportedInputsImpl(node);

  // QLinearConv only supports input of uint8 for now
  if (!HasValidBinaryOpQuantizedInputs(node))
    return false;

  return true;
}

bool ConvOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                                             const OpSupportCheckParams& params) const {
  const auto& op_type = node.OpType();
  const bool is_qlinear_conv = (op_type == "QLinearConv");

  // We don't support nhwc com.microsoft.QLinearConv for now
  if (is_qlinear_conv && node.Domain() == kMSDomain) {
    LOGS_DEFAULT(VERBOSE) << "com.microsoft.QLinearConv is not supported";
    return false;
  }

  const auto input_defs = node.InputDefs();
  NodeAttrHelper helper(node);
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

      if (params.android_feature_level < ANEURALNETWORKS_FEATURE_LEVEL_3) {
        LOGS_DEFAULT(VERBOSE) << op_type << " dilations is only supported on Android API level 29+, "
                              << "actual API level: " << params.android_feature_level;
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
    if (!HasValidQuantizationScales(initializers, node, {1, 4, 6}, params))
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

  int32_t GetMinSupportedNNAPIFeatureLevel(const Node& /* node */, const OpSupportCheckParams& /* params */) const override {
    return ANEURALNETWORKS_FEATURE_LEVEL_3;
  }

  // Cast opset 5- uses string attribute for to type, is not supported for now
  int GetMinSupportedOpSet(const Node& /* node */) const override {
    return 6;
  }
};  // namespace nnapi

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

  int32_t GetMinSupportedNNAPIFeatureLevel(const Node& /* node */, const OpSupportCheckParams& /* params */) const override {
    return ANEURALNETWORKS_FEATURE_LEVEL_2;
  }
};  // namespace onnxruntime

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

  if (params.android_feature_level < ANEURALNETWORKS_FEATURE_LEVEL_3) {
    NodeAttrHelper helper(node);
    int32_t axis = helper.Get("axis", 1);
    if (axis != 1) {
      LOGS_DEFAULT(VERBOSE)
          << "SoftMax only support axis 1 on Android API level: " << params.android_feature_level
          << " input axis: " << axis;
      return false;
    }
  }

  return true;
}

#pragma endregion

#pragma region op_gemm

class GemmOpSupportChecker : public BaseOpSupportChecker {
 public:
  static void CreateSharedOpSupportChecker(
      const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);

 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const OpSupportCheckParams& params) const override;
  bool HasSupportedInputsImpl(const Node& node) const override;
  int GetMinSupportedOpSet(const Node& node) const override;
};

bool GemmOpSupportChecker::HasSupportedInputsImpl(const Node& node) const {
  if (node.OpType() != "QLinearMatMul")
    return BaseOpSupportChecker::HasSupportedInputsImpl(node);

  // QLinearMatMul
  if (!HasValidBinaryOpQuantizedInputs(node))
    return false;

  return true;
}

/* static */ void GemmOpSupportChecker::CreateSharedOpSupportChecker(
    const std::string& op_type, OpSupportCheckerRegistrations& op_registrations) {
  CreateSharedOpSupportCheckerImpl<GemmOpSupportChecker>(
      op_type, op_registrations,
      {
          "Gemm",
          "MatMul",
          "QLinearMatMul",
      });
}

// Get the bias size (C) of Gemm op
// ANEURALNETWORKS_FULLY_CONNECTED only supports 1d bias
// Will test if C of Gemm can be squeezed and return the 1d vector size after squeeze
static bool GetBiasSize(const Shape& c_shape, int32_t android_feature_level, uint32_t& size) {
  // TODO add support of scalar C for Gemm
  size_t c_dim = c_shape.size();
  if (c_dim == 0) {
    LOGS_DEFAULT(VERBOSE) << "C of Gemm cannot be a scalar";
    return false;
  }

  if (c_dim != 1 && android_feature_level < ANEURALNETWORKS_FEATURE_LEVEL_2) {
    LOGS_DEFAULT(VERBOSE) << "C of Gemm can only be 1d tensor for API level " << android_feature_level
                          << " shape of C, " << Shape2String(c_shape);
    return false;
  }

  if (c_dim != 1) {
    // If C is a (2+)d tensor, it must have the format {1, 1, ..., 1, n}
    // where every except the last dimension should be 1
    for (size_t i = 0; i < c_dim - 1; ++i) {
      if (c_shape[i] != 1) {
        LOGS_DEFAULT(VERBOSE) << "C of Gemm must be a vector or a tensor with only last dimension != 1"
                              << " c_shape: " << Shape2String(c_shape);
        return false;
      }
    }
  }

  size = c_shape[c_dim - 1];
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
                                             const OpSupportCheckParams& params) const {
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
                            << "and beta == 1.0 is supported."
                            << " transA " << transA
                            << " transB " << transB
                            << " alpha " << alpha
                            << " beta " << beta;
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

      uint32_t c_size;
      if (!GetBiasSize(c_shape, params.android_feature_level, c_size))
        return false;

      if (c_size != (transB == 0 ? b_shape[1] : b_shape[0])) {
        LOGS_DEFAULT(VERBOSE) << "C of Gemm must be a vector of b_shape["
                              << (transB == 0 ? "1" : "0") << "]"
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
      if (!HasValidQuantizationScales(initializers, node, {1, 4, 6}, params))
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
 public:
  static void CreateSharedOpSupportChecker(
      const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);

 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const OpSupportCheckParams& params) const override;

  int32_t GetMinSupportedNNAPIFeatureLevel(const Node& node, const OpSupportCheckParams& params) const override;

  bool HasSupportedInputsImpl(const Node& node) const override;

  int GetMinSupportedOpSet(const Node& node) const override;

  static bool IsQuantizedOpSupported(const InitializedTensorSet& initializers, const Node& node,
                                     const OpSupportCheckParams& params);
};

/* static */ void UnaryOpSupportChecker::CreateSharedOpSupportChecker(
    const std::string& op_type, OpSupportCheckerRegistrations& op_registrations) {
  CreateSharedOpSupportCheckerImpl<UnaryOpSupportChecker>(
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

bool UnaryOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                                              const OpSupportCheckParams& params) const {
  if (node.OpType() == "QLinearSigmoid")
    return IsQuantizedOpSupported(initializers, node, params);
  else  // Everything except "QLinearSigmoid" are by default supported
    return true;
}

int32_t UnaryOpSupportChecker::GetMinSupportedNNAPIFeatureLevel(
    const Node& node, const OpSupportCheckParams& /* params */) const {
  const auto& op(node.OpType());
  if (op == "Abs" ||
      op == "Exp" ||
      op == "Neg" ||
      op == "Sin" ||
      op == "Sqrt" ||
      op == "Log") {
    return ANEURALNETWORKS_FEATURE_LEVEL_3;
  }

  return ANEURALNETWORKS_FEATURE_LEVEL_1;
}

bool UnaryOpSupportChecker::HasSupportedInputsImpl(const Node& node) const {
  // We only need to override input check for QLinearSigmoid
  if (node.OpType() != "QLinearSigmoid")
    return BaseOpSupportChecker::HasSupportedInputsImpl(node);

  return HasValidUnaryOpQuantizedInputs(node);
}

// All ops except "Sin" opset 5- uses consumed_inputs attribute which is not supported for now
// "Sin" op has support from opset 7, return 6 here for all ops
// "QLinearSigmoid" is a contrib op, OpSet will always be 1
int UnaryOpSupportChecker::GetMinSupportedOpSet(const Node& node) const {
  if (node.OpType() == "QLinearSigmoid")
    return 1;

  return 6;
}

/* static */ bool UnaryOpSupportChecker::IsQuantizedOpSupported(
    const InitializedTensorSet& initializers, const Node& node, const OpSupportCheckParams& params) {
  const auto& op_type = node.OpType();
  ORT_ENFORCE(op_type == "QLinearSigmoid");

  const auto& op_name = node.Name();
  const auto input_defs(node.InputDefs());
  // const auto output_defs(node.OutputDefs());

  if (input_defs.size() < 4)
    return false;

  bool has_output_zp = input_defs.size() == 5;

  if (!HasValidQuantizationScales(initializers, node, {1, 3}, params))
    return false;

  if (!HasValidQuantizationZeroPoints(initializers, node,
                                      has_output_zp
                                          ? std::vector<size_t>{2}
                                          : std::vector<size_t>{2, 4}))
    return false;

  // NNAPI requires the scale be 1.f/256 and zero point to be 0
  // See https://android.googlesource.com/platform/frameworks/ml/+/refs/heads/android10-c2f2-release/nn/common/operations/Activation.cpp#180
  auto output_scale = GetQuantizationScale(initializers, node, 3);
  if (output_scale != 1.f / 256) {
    LOGS_DEFAULT(VERBOSE) << "Op [" << op_type << "] name [" << op_name
                          << "] output scale can only be 1.f/256, actual scale: " << output_scale;
    return false;
  }

  int32_t output_zp;
  if (has_output_zp) {
    auto status = GetQuantizationZeroPoint(initializers, node, 4, output_zp);
    if (!status.IsOK()) {
      LOGS_DEFAULT(ERROR) << "Op [" << op_type << "] name [" << op_name
                          << "] GetQuantizationZeroPoint failed, message: " << status.ErrorMessage();
      return false;
    }

    if (output_zp != 0) {
      LOGS_DEFAULT(VERBOSE) << "Op [" << op_type << "] name [" << op_name
                            << "] output zero point can only be 0, actual zero point: " << output_scale;
      return false;
    }
  }

  return true;
}

#pragma endregion

#pragma region op_concat

class ConcatOpSupportChecker : public BaseOpSupportChecker {
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const OpSupportCheckParams& params) const override;

  bool HasSupportedInputsImpl(const Node& node) const override;
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

bool ConcatOpSupportChecker::HasSupportedInputsImpl(const Node& node) const {
  int32_t input_type;
  if (!GetType(*node.InputDefs()[0], input_type))
    return false;

  if (input_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
      input_type != ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
    LOGS_DEFAULT(VERBOSE) << "[" << node.OpType()
                          << "] Input type: [" << input_type
                          << "] is not supported for now";
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

  int32_t GetMinSupportedNNAPIFeatureLevel(const Node& /* node */, const OpSupportCheckParams& /* params */) const override {
    return ANEURALNETWORKS_FEATURE_LEVEL_2;
  }
};

bool SqueezeOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
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

  // Squeeze opset 13 use input 1 as axes, if we have input 1 then it need to be an initializer
  if (node.SinceVersion() > 12 && node.InputDefs().size() > 1) {
    const auto& axes_name = node.InputDefs()[1]->Name();
    if (!Contains(initializers, axes_name)) {
      LOGS_DEFAULT(VERBOSE) << "Input axes of Squeeze must be known";
      return false;
    }
  }

  return true;
}

#pragma endregion

#pragma region op_quantizelinear

class QuantizeLinearOpSupportChecker : public BaseOpSupportChecker {
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const OpSupportCheckParams& params) const override;

  int32_t GetMinSupportedNNAPIFeatureLevel(const Node& /* node */, const OpSupportCheckParams& /* params */) const override {
    return ANEURALNETWORKS_FEATURE_LEVEL_3;
  }
};

bool QuantizeLinearOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                                                       const OpSupportCheckParams& params) const {
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

  if (!HasValidQuantizationScales(initializers, node, {1}, params))
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

  int32_t GetMinSupportedNNAPIFeatureLevel(const Node& /* node */, const OpSupportCheckParams& /* params */) const override {
    return ANEURALNETWORKS_FEATURE_LEVEL_1;
  }
  bool HasSupportedInputsImpl(const Node& node) const override;
};

bool DequantizeLinearOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                                                         const OpSupportCheckParams& params) const {
  const auto input_defs(node.InputDefs());
  if (!HasValidQuantizationScales(initializers, node, {1}, params))
    return false;

  if (input_defs.size() == 3) {  // has zero_point input
    if (!HasValidQuantizationZeroPoints(initializers, node, {2}))
      return false;
  }

  return true;
}

bool DequantizeLinearOpSupportChecker::HasSupportedInputsImpl(const Node& node) const {
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

  int32_t GetMinSupportedNNAPIFeatureLevel(const Node& /* node */, const OpSupportCheckParams& /* params */) const override {
    return ANEURALNETWORKS_FEATURE_LEVEL_2;
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
  if (!GetClipMinMax(initializers, node, min, max, logging::LoggingManager::DefaultLogger()))
    return false;

  // We only supoort relu6 or relu1
  // TODO, support clip between 2 arbitrary numbers
  if ((min == 0.0f && max == 6.0f) || (min == -1.0f && max == 1.0f)) {
    return true;
  }

  LOGS_DEFAULT(VERBOSE) << "Clip only supports [min, max] = [0, 6] or [-1, 1], the input is ["
                        << min << ", " << max << "]";
  return false;
}

#pragma endregion

#pragma region op_Resize

class ResizeOpSupportChecker : public BaseOpSupportChecker {
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const OpSupportCheckParams& params) const override;

  int32_t GetMinSupportedNNAPIFeatureLevel(const Node& /* node */, const OpSupportCheckParams& /* params */) const override;

  // Resize opset 10- is very different than Resize opset 11+, with many key attributes missing
  // We only support Resize opset 11+ here
  int GetMinSupportedOpSet(const Node& /* node */) const override { return 11; }

  bool HasSupportedInputsImpl(const Node& node) const override;
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
    bool is_linear_resize = mode == "linear";
    bool is_nearest_resize = mode == "nearest";
    if (!is_linear_resize && !is_nearest_resize) {
      LOGS_DEFAULT(VERBOSE) << "Resize unsupported input mode, " << mode;
      return false;
    }

    const auto exclude_outside = helper.Get("exclude_outside", 0);
    if (exclude_outside != 0) {
      LOGS_DEFAULT(VERBOSE) << "Resize does not support exclude_outside for now";
      return false;
    }

    const auto coord_trans_mode = helper.Get("coordinate_transformation_mode", "half_pixel");
    bool using_half_pixel = coord_trans_mode == "half_pixel";
    bool using_align_corners = coord_trans_mode == "align_corners";
    bool using_asymmetric = coord_trans_mode == "asymmetric";
    if (is_linear_resize) {
      if (!using_half_pixel && !using_align_corners && !using_asymmetric) {
        LOGS_DEFAULT(VERBOSE) << "Resize bilinear, unsupported coord_trans_mode, " << coord_trans_mode;
        return false;
      }

      if (params.android_feature_level < 30 && (using_half_pixel || using_align_corners)) {
        LOGS_DEFAULT(VERBOSE) << "Resize bilinear only support half_pixel/align_corners on API level 30+, current API level is "
                              << params.android_feature_level;
        return false;
      }
    } else {
      // nearest neighbor resizing
      // For resize using nearest neighbor, we only support coord_trans_mode == "asymmetric" && nearest_mode == "floor"
      if (!using_asymmetric) {
        LOGS_DEFAULT(VERBOSE) << "Resize nearest neighbor, unsupported coord_trans_mode, " << coord_trans_mode;
        return false;
      }

      const auto nearest_mode = helper.Get("nearest_mode", "round_prefer_floor");
      if (nearest_mode != "floor") {
        LOGS_DEFAULT(VERBOSE) << "Resize nearest neighbor, unsupported nearest_mode, " << nearest_mode;
        return false;
      }
    }
  }

  {  // scales and sizes (if present) must be initializers
    const auto input_defs = node.InputDefs();
    if (input_defs.size() < 3) {
      LOGS_DEFAULT(VERBOSE) << "Input scales or sizes of Resize must be known";
      return false;
    }

    // scales
    if (input_defs.size() == 3 && !Contains(initializers, input_defs[2]->Name())) {
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

int32_t ResizeOpSupportChecker::GetMinSupportedNNAPIFeatureLevel(const Node& node, const OpSupportCheckParams& /* params */) const {
  int32_t input_type;

  // This should not happen, but if it happens make sure this will require an impossible version
  if (!GetType(*node.InputDefs()[0], input_type))
    return std::numeric_limits<int32_t>::max();

  if (input_type != ONNX_NAMESPACE::TensorProto_DataType_UINT8)
    return ANEURALNETWORKS_FEATURE_LEVEL_3;

  return ANEURALNETWORKS_FEATURE_LEVEL_2;
}

bool ResizeOpSupportChecker::HasSupportedInputsImpl(const Node& node) const {
  int32_t input_type;
  if (!GetType(*node.InputDefs()[0], input_type))
    return false;

  if (input_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
      input_type != ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
    LOGS_DEFAULT(VERBOSE) << "[" << node.OpType()
                          << "] Input type: [" << input_type
                          << "] is not supported for now";
    return false;
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

#pragma region op_minmax

class MinMaxOpSupportChecker : public BaseOpSupportChecker {
 public:
  static void CreateSharedOpSupportChecker(
      const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);

 private:
  int32_t GetMinSupportedNNAPIFeatureLevel(const Node& /* node */, const OpSupportCheckParams& /* params */) const override {
    return ANEURALNETWORKS_FEATURE_LEVEL_3;
  }

  // Min/Max opset 5- uses consumed_inputs attribute which is not supported for now
  int GetMinSupportedOpSet(const Node& /* node */) const override { return 6; }

  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const OpSupportCheckParams& params) const override;
};

/* static */ void MinMaxOpSupportChecker::CreateSharedOpSupportChecker(
    const std::string& op_type, OpSupportCheckerRegistrations& op_registrations) {
  CreateSharedOpSupportCheckerImpl<MinMaxOpSupportChecker>(
      op_type, op_registrations,
      {
          "Min",
          "Max",
      });
}

bool MinMaxOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const Node& node,
                                               const OpSupportCheckParams& /* params */) const {
  // TODO support 2+ inputs for Min/Max op
  if (node.InputDefs().size() != 2) {
    LOGS_DEFAULT(VERBOSE) << "[" << node.OpType() << "] only supports 2 inputs, "
                          << "actual input number, " << node.InputDefs().size();
    return false;
  }

  return true;
}

#pragma endregion

#pragma region op_elu

class EluOpSupportChecker : public BaseOpSupportChecker {
 private:
  int32_t GetMinSupportedNNAPIFeatureLevel(const Node& /* node */, const OpSupportCheckParams& /* params */) const override {
    return ANEURALNETWORKS_FEATURE_LEVEL_4;
  }

  // Elu opset 5- uses consumed_inputs attribute which is not supported for now
  int GetMinSupportedOpSet(const Node& /* node */) const override { return 6; }
};

#pragma endregion

#pragma region op_slice

class SliceOpSupportChecker : public BaseOpSupportChecker {
 private:
  int32_t GetMinSupportedNNAPIFeatureLevel(const Node& /* node */, const OpSupportCheckParams& /* params */) const override {
    return ANEURALNETWORKS_FEATURE_LEVEL_2;
  }

  // We only support slice from opset 10
  int GetMinSupportedOpSet(const Node& /* node */) const override { return 10; }

  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const OpSupportCheckParams& params) const override;
};

bool SliceOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                                              const OpSupportCheckParams& /* params */) const {
  Shape input_shape;
  if (!GetShape(*node.InputDefs()[0], input_shape))
    return false;

  if (input_shape.size() > 4) {
    LOGS_DEFAULT(VERBOSE) << "Slice only supports 1-4d shape, input is "
                          << input_shape.size() << "d shape";
    return false;
  }

  // TODO, replace with std::find when we switch to c++17
  if (std::any_of(input_shape.cbegin(), input_shape.cend(), [](int32_t i) { return i == 0; })) {
    LOGS_DEFAULT(VERBOSE) << "Slice doesn't dynamic input shape";
    return false;
  }

  if (!CheckIsInitializer(initializers, node, 1, "starts")) {
    return false;
  }
  if (!CheckIsInitializer(initializers, node, 2, "ends")) {
    return false;
  }
  const auto& input_defs = node.InputDefs();
  if (input_defs.size() > 3) {
    if (!CheckIsInitializer(initializers, node, 3, "axes")) {
      return false;
    }
    if (input_defs.size() > 4) {
      if (!CheckIsInitializer(initializers, node, 4, "steps")) {
        return false;
      }
    }
  }

  return true;
}

#pragma endregion

#pragma region CreateGetOpSupportCheckers

// The reason we use macros to create OpBuilders is for easy exclusion in build if certain op(s) are not used
// such that we can reduce binary size.
// This is for multiple ops share the same OpSupportChecker, we only need create one for all of them
#define NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER(OP_TYPE, SUPPORT_CHECKER_NAME) \
  SUPPORT_CHECKER_NAME::CreateSharedOpSupportChecker(OP_TYPE, op_registrations);

// This is for ops with dedicated OpSupportChecker
#define NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER(OP_TYPE, SUPPORT_CHECKER_NAME)                                 \
  {                                                                                                           \
    op_registrations.support_checkers.push_back(std::make_unique<SUPPORT_CHECKER_NAME>());                    \
    op_registrations.op_support_checker_map.emplace(OP_TYPE, op_registrations.support_checkers.back().get()); \
  }

static OpSupportCheckerRegistrations CreateOpSupportCheckerRegistrations() {
  OpSupportCheckerRegistrations op_registrations;

  {
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("Add", BinaryOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("Sub", BinaryOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("Mul", BinaryOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("Div", BinaryOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("QLinearAdd", BinaryOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("Pow", BinaryOpSupportChecker);
  }

  // Relu is always supported, we use BaseOpSupportChecker as default
  NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("Relu", BaseOpSupportChecker);

  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("Transpose", TransposeOpSupportChecker);
  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("Reshape", ReshapeOpSupportChecker);
  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("BatchNormalization", BatchNormalizationOpSupportChecker);

  {
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("GlobalAveragePool", PoolOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("GlobalMaxPool", PoolOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("AveragePool", PoolOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("MaxPool", PoolOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("QLinearAveragePool", PoolOpSupportChecker);
  }

  {
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("Conv", ConvOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("QLinearConv", ConvOpSupportChecker);
  }

  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("Cast", CastOpSupportChecker);
  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("Softmax", SoftMaxOpSupportChecker);

  // Identity is always supported, we use BaseOpSupportChecker as default
  NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("Identity", BaseOpSupportChecker);

  {
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("Gemm", GemmOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("MatMul", GemmOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("QLinearMatMul", GemmOpSupportChecker);
  }

  {
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("Abs", UnaryOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("Exp", UnaryOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("Floor", UnaryOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("Log", UnaryOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("Sigmoid", UnaryOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("Neg", UnaryOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("Sin", UnaryOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("Sqrt", UnaryOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("Tanh", UnaryOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("QLinearSigmoid", UnaryOpSupportChecker);
  }

  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("Concat", ConcatOpSupportChecker);
  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("Squeeze", SqueezeOpSupportChecker);
  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("QuantizeLinear", QuantizeLinearOpSupportChecker);
  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("DequantizeLinear", DequantizeLinearOpSupportChecker);
  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("LRN", LRNOpSupportChecker);
  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("Clip", ClipOpSupportChecker);
  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("Resize", ResizeOpSupportChecker);
  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("Flatten", FlattenOpSupportChecker);

  {
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("Min", MinMaxOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("Max", MinMaxOpSupportChecker);
  }

  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("Elu", EluOpSupportChecker);
  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("Slice", SliceOpSupportChecker);

  return op_registrations;
}

const std::unordered_map<std::string, const IOpSupportChecker*>& GetOpSupportCheckers() {
  static const OpSupportCheckerRegistrations op_registrations = CreateOpSupportCheckerRegistrations();
  return op_registrations.op_support_checker_map;
}

#pragma endregion

}  // namespace nnapi
}  // namespace onnxruntime
