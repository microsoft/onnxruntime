// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "QnnOpDef.h"

#include "core/providers/qnn-abi/builder/op_builder.h"
#include "core/providers/qnn-abi/builder/qnn_model_wrapper.h"
#include "core/providers/qnn-abi/builder/qnn_quant_params_wrapper.h"
#include "core/providers/qnn-abi/builder/qnn_utils.h"
#include "core/providers/qnn-abi/ort_api.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;

class BaseOpBuilder : public IOpBuilder {
 public:
  BaseOpBuilder(const std::string& op_builder_type) : op_builder_type_(op_builder_type) {}
  virtual ~BaseOpBuilder() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(BaseOpBuilder);

  Ort::Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger) const override ORT_MUST_USE_RESULT;

  Ort::Status AddToModelBuilder(QnnModelWrapper& qnn_model_wrapper,
                                const OrtNodeUnit& node_unit,
                                const Ort::Logger& logger,
                                bool do_op_validation) const override final ORT_MUST_USE_RESULT;

  std::string GetOpBuilderType() const override;

 protected:
  virtual Qnn_DataType_t GetSupportedOutputDataType(size_t index, Qnn_DataType_t qnn_data_type) const {
    ORT_UNUSED_PARAMETER(index);
    return qnn_data_type;
  }

  /**
   * Allows operator builders that override this function to override output quantization parameters.
   * Called by BaseOpBuilder::ProcessOutputs().
   *
   * \param qnn_model_wrapper The QNN model that is being built.
   * \param node_unit The node unit for which to return output information.
   * \param logger The logger.
   * \param input_names Names of all inputs consumed by this QNN node.
   * \param output_index The index in node_unit.Outputs() of the output for which to return information.
   * \param qnn_data_type The output's data type.
   * \param quant_param The quantization parameter object that is overridden.
   * \return An onnxruntime::Status object indicating failure or success.
   */
  virtual Ort::Status OverrideOutputQuantParam(QnnModelWrapper& qnn_model_wrapper,
                                               const OrtNodeUnit& node_unit,
                                               const Ort::Logger& logger,
                                               const std::vector<std::string>& input_names,
                                               size_t output_index,
                                               Qnn_DataType_t qnn_data_type,
                                               QnnQuantParamsWrapper& quant_param) const ORT_MUST_USE_RESULT {
    // Do nothing by default. Op builders like Split implement this function to override output quant params.
    ORT_UNUSED_PARAMETER(qnn_model_wrapper);
    ORT_UNUSED_PARAMETER(node_unit);
    ORT_UNUSED_PARAMETER(logger);
    ORT_UNUSED_PARAMETER(input_names);
    ORT_UNUSED_PARAMETER(output_index);
    ORT_UNUSED_PARAMETER(qnn_data_type);
    ORT_UNUSED_PARAMETER(quant_param);
    return Ort::Status();
  }

  Ort::Status ProcessDataTypes(QnnModelWrapper& qnn_model_wrapper,
                               const OrtNodeUnit& node_unit) const ORT_MUST_USE_RESULT;

  virtual Ort::Status CheckCpuDataTypes(const std::vector<Qnn_DataType_t>,
                                        const std::vector<Qnn_DataType_t>) const ORT_MUST_USE_RESULT;

  virtual Ort::Status CheckHtpDataTypes(const std::vector<Qnn_DataType_t>,
                                        const std::vector<Qnn_DataType_t>) const ORT_MUST_USE_RESULT;

  virtual Ort::Status CheckGpuDataTypes(const std::vector<Qnn_DataType_t>,
                                        const std::vector<Qnn_DataType_t>) const ORT_MUST_USE_RESULT;

  virtual Ort::Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                    const OrtNodeUnit& node_unit,
                                    const Ort::Logger& logger,
                                    std::vector<std::string>& input_names,
                                    bool do_op_validation = false) const ORT_MUST_USE_RESULT;

  Ort::Status ProcessInt64Tensors(QnnModelWrapper& qnn_model_wrapper,
                                  const OrtNodeUnit& node_unit,
                                  std::vector<std::string>& input_names) const ORT_MUST_USE_RESULT;

  virtual Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                  const OrtNodeUnit& node_unit,
                                                  std::vector<std::string>&& input_names,
                                                  const Ort::Logger& logger,
                                                  bool do_op_validation = false) const ORT_MUST_USE_RESULT;

  virtual Ort::Status ProcessOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const OrtNodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     std::vector<std::string>&& param_tensor_names,
                                     const Ort::Logger& logger,
                                     bool do_op_validation,
                                     const std::string& qnn_op_type) const ORT_MUST_USE_RESULT;

  Ort::Status ProcessInput(QnnModelWrapper& qnn_model_wrapper,
                           const OrtNodeUnitIODef& input,
                           const Ort::Logger& logger,
                           std::vector<std::string>& input_names) const ORT_MUST_USE_RESULT;

  Ort::Status AddZeroBiasInput(QnnModelWrapper& qnn_model_wrapper,
                               const QnnQuantParamsWrapper& input0_qparams,
                               const QnnQuantParamsWrapper& input1_qparams,
                               std::vector<uint32_t>&& bias_shape,
                               const std::string& bias_name,
                               const Ort::Logger& logger,
                               std::vector<std::string>& input_names) const ORT_MUST_USE_RESULT;

  Ort::Status SetOutputQParamEqualToInputIfNearlyEqual(QnnModelWrapper& qnn_model_wrapper,
                                                       const OrtNodeUnit& node_unit,
                                                       const Ort::Logger& logger,
                                                       const std::vector<std::string>& input_names,
                                                       size_t input_index,
                                                       size_t output_index,
                                                       Qnn_DataType_t qnn_data_type,
                                                       QnnQuantParamsWrapper& quant_param) const ORT_MUST_USE_RESULT;

  static const std::string& GetQnnOpType(const std::string& onnx_op_type) {
    static const std::unordered_map<std::string, std::string> onnx_op_type_to_qnn_op_type = {
        {"Add", QNN_OP_ELEMENT_WISE_ADD},
        {"Mul", QNN_OP_ELEMENT_WISE_MULTIPLY},
        {"Abs", QNN_OP_ELEMENT_WISE_ABS},
        {"And", QNN_OP_ELEMENT_WISE_AND},
        {"Asin", QNN_OP_ELEMENT_WISE_ASIN},
        {"Atan", QNN_OP_ELEMENT_WISE_ATAN},
        {"Ceil", QNN_OP_ELEMENT_WISE_CEIL},
        {"Sign", QNN_OP_ELEMENT_WISE_SIGN},
        {"Cast", QNN_OP_CAST},
        {"Clip", QNN_OP_RELU_MIN_MAX},
        {"Cos", QNN_OP_ELEMENT_WISE_COS},
        {"Div", QNN_OP_ELEMENT_WISE_DIVIDE},
        {"Equal", QNN_OP_ELEMENT_WISE_EQUAL},
        {"Exp", QNN_OP_ELEMENT_WISE_EXP},
        {"Floor", QNN_OP_ELEMENT_WISE_FLOOR},
        {"Gather", QNN_OP_GATHER},
        {"GatherElements", QNN_OP_GATHER_ELEMENTS},
        {"Greater", QNN_OP_ELEMENT_WISE_GREATER},
        {"GreaterOrEqual", QNN_OP_ELEMENT_WISE_GREATER_EQUAL},
        {"Less", QNN_OP_ELEMENT_WISE_LESS},
        {"LessOrEqual", QNN_OP_ELEMENT_WISE_LESS_EQUAL},
        {"Log", QNN_OP_ELEMENT_WISE_LOG},
        {"LSTM", QNN_OP_LSTM},
        {"Max", QNN_OP_ELEMENT_WISE_MAXIMUM},
        {"Min", QNN_OP_ELEMENT_WISE_MINIMUM},
        {"Neg", QNN_OP_ELEMENT_WISE_NEG},
        {"Not", QNN_OP_ELEMENT_WISE_NOT},
        {"Or", QNN_OP_ELEMENT_WISE_OR},
        {"Pow", QNN_OP_ELEMENT_WISE_POWER},
        {"PRelu", QNN_OP_PRELU},
        {"LeakyRelu", QNN_OP_PRELU},
        {"ReduceMax", QNN_OP_REDUCE_MAX},
        {"ReduceMean", QNN_OP_REDUCE_MEAN},
        {"ReduceMin", QNN_OP_REDUCE_MIN},
        {"ReduceProd", QNN_OP_REDUCE_PROD},
        {"ReduceSum", QNN_OP_REDUCE_SUM},
        {"Round", QNN_OP_ELEMENT_WISE_ROUND},
        {"Where", QNN_OP_ELEMENT_WISE_SELECT},
        {"ScatterND", QNN_OP_SCATTER_ND},
        {"Sigmoid", QNN_OP_SIGMOID},
        {"Sin", QNN_OP_ELEMENT_WISE_SIN},
        {"Slice", QNN_OP_STRIDED_SLICE},
        {"Split", QNN_OP_SPLIT},
        {"Softmax", QNN_OP_SOFTMAX},
        {"Sqrt", QNN_OP_ELEMENT_WISE_SQUARE_ROOT},
        {"Sub", QNN_OP_ELEMENT_WISE_SUBTRACT},
        {"Sum", QNN_OP_ELEMENT_WISE_ADD},
        {"Tanh", QNN_OP_TANH},
        {"Transpose", QNN_OP_TRANSPOSE},
        {"GridSample", QNN_OP_GRID_SAMPLE},
        {"LpNormalization", QNN_OP_L2_NORM},

        {"DequantizeLinear", QNN_OP_DEQUANTIZE},
        {"QuantizeLinear", QNN_OP_QUANTIZE},

        {"MatMul", QNN_OP_MAT_MUL},

        {"Elu", QNN_OP_ELU},
        {"Relu", QNN_OP_RELU},
        {"Gelu", QNN_OP_GELU},

        {"HardSigmoid", QNN_OP_ELEMENT_WISE_NEURON},
        {"HardSwish", QNN_OP_HARD_SWISH},
        {"DepthToSpace", QNN_OP_DEPTH_TO_SPACE},
        {"SpaceToDepth", QNN_OP_SPACE_TO_DEPTH},

        {"Conv", QNN_OP_CONV_2D},
        {"ConvTranspose", QNN_OP_TRANSPOSE_CONV_2D},

        {"GlobalAveragePool", QNN_OP_POOL_AVG_2D},
        {"AveragePool", QNN_OP_POOL_AVG_2D},
        {"MaxPool", QNN_OP_POOL_MAX_2D},
        {"GlobalMaxPool", QNN_OP_POOL_MAX_2D},

        {"Reshape", QNN_OP_RESHAPE},
        {"Resize", QNN_OP_RESIZE},
        {"Upsample", QNN_OP_RESIZE},
        {"Flatten", QNN_OP_RESHAPE},
        {"Squeeze", QNN_OP_RESHAPE},
        {"Unsqueeze", QNN_OP_RESHAPE},

        {"LogSoftmax", QNN_OP_LOG_SOFTMAX},
        {"Concat", QNN_OP_CONCAT},
        {"CumSum", QNN_OP_CUMULATIVE_SUM},

        {"Gemm", QNN_OP_FULLY_CONNECTED},

        {"ArgMax", QNN_OP_ARGMAX},
        {"ArgMin", QNN_OP_ARGMIN},
        {"Tile", QNN_OP_TILE},
        {"TopK", QNN_OP_TOP_K},
        {"InstanceNormalization", QNN_OP_INSTANCE_NORM},
        {"BatchNormalization", QNN_OP_BATCHNORM},
        {"LayerNormalization", QNN_OP_LAYER_NORM},

        {"LRN", QNN_OP_LRN},

        {"Pad", QNN_OP_PAD},

        {"ScatterElements", QNN_OP_SCATTER_ELEMENTS},

        {"Expand", QNN_OP_ELEMENT_WISE_MULTIPLY}};
    auto it = onnx_op_type_to_qnn_op_type.find(onnx_op_type);
    if (it == onnx_op_type_to_qnn_op_type.end()) {
      ORT_CXX_API_THROW(("Unable to map given ONNX op type to QNN" + onnx_op_type).c_str(), ORT_EP_FAIL);
    }
    return it->second;
  }

  // Onnx Pads is [x1_begin, x2_begin, x1_end, x2_end], QNN requires [x1_begin, x1_end, x2_begin, x2_end]
  void ReArrangePads(std::vector<uint32_t>& pads) const {
    auto pads_size = pads.size();
    auto middle_pos = pads_size / 2;
    std::vector<uint32_t> first_half(pads.begin(), pads.begin() + middle_pos);
    for (size_t i = 0; i < middle_pos; ++i) {
      pads[2 * i] = first_half[i];
      pads[2 * i + 1] = pads[middle_pos + i];
    }
  }

  Ort::Status ProcessAxisAttribute(const QnnModelWrapper& qnn_model_wrapper,
                                   const OrtNodeUnit& node_unit,
                                   Qnn_Scalar_t& axis_qnn_scalar,
                                   int32_t& default_axis_value) const;

  size_t GetInputCountQnnRequired(const OrtNodeUnit& node_unit) const {
    auto input_output_cout = GetInputOutputCountQnnRequired(node_unit.OpType());

    return 0 == input_output_cout.first ? node_unit.Inputs().size() : input_output_cout.first;
  }

  size_t GetOutputCountQnnRequired(const OrtNodeUnit& node_unit) const {
    auto input_output_cout = GetInputOutputCountQnnRequired(node_unit.OpType());

    return 0 == input_output_cout.second ? node_unit.Outputs().size() : input_output_cout.second;
  }

 private:
  static const std::pair<size_t, size_t> GetInputOutputCountQnnRequired(std::string onnx_op_type) {
    static const std::unordered_map<std::string, std::pair<size_t, size_t>> input_output_count_qnn_required = {
        {"GlobalAveragePool", {0, 1}},
        {"MaxPool", {0, 1}},
        {"BatchNormalization", {3, 1}},
        {"LayerNormalization", {0, 1}}};

    auto pos = input_output_count_qnn_required.find(onnx_op_type);
    if (pos == input_output_count_qnn_required.end()) {
      return std::make_pair<size_t, size_t>(0, 0);
    } else {
      return pos->second;
    }
  }

 private:
  std::string op_builder_type_;
};

// Type that holds information about an ONNX attribute.
template <typename ValType>
struct OnnxAttrInfo {
  std::string name;     // Attribute's name.
  ValType default_val;  // Attribute's default value.
};

template <typename ValType>
inline ValType GetOnnxAttr(const OrtNodeAttrHelper& node_helper, const OnnxAttrInfo<ValType>& attr_info) {
  return node_helper.Get(attr_info.name, attr_info.default_val);
}

// QNN-EP COPY START
// Utility functions copied from core/providers/common.h.
enum class AutoPadType {
  NOTSET = 0,
  VALID = 1,
  SAME_UPPER = 2,
  SAME_LOWER = 3,
};

inline AutoPadType StringToAutoPadType(const std::string& str) {
  if (str.empty()) {
    return AutoPadType::NOTSET;
  }
  if (str == "NOTSET") {  // in onnx spec, default value is "NOTSET"
    return AutoPadType::NOTSET;
  }
  if (str == "VALID") {
    return AutoPadType::VALID;
  }
  if (str == "SAME_UPPER") {
    return AutoPadType::SAME_UPPER;
  }
  if (str == "SAME_LOWER") {
    return AutoPadType::SAME_LOWER;
  }
  ORT_CXX_API_THROW("Unknown AutoPadType String", ORT_EP_FAIL);
}

constexpr inline int64_t ComputeOutputShape(const int64_t in_dim,
                                            const int64_t stride,
                                            const int64_t kernel,
                                            const int64_t dilation,
                                            const int64_t pad_head,
                                            const int64_t pad_tail) {
  const SafeInt<int64_t> dkernel = SafeInt<int64_t>(dilation) * (kernel - 1) + 1;
  int64_t dkernel_value = SafeInt<int64_t>(in_dim) + pad_head + pad_tail - dkernel;
  return static_cast<int64_t>(static_cast<double>(dkernel_value) / stride + 1);
}

inline Ort::Status ComputePad(const int64_t in_dim,
                              const int64_t stride,
                              const int64_t kernel,
                              const int64_t dilation,
                              AutoPadType pad_type,
                              int64_t& pad_head,
                              int64_t& pad_tail,
                              bool force_symmetric_auto_padding = false) {
  switch (pad_type) {
    case AutoPadType::NOTSET:
      break;
    case AutoPadType::VALID: {
      pad_head = 0;
      pad_tail = 0;
    } break;
    case AutoPadType::SAME_UPPER:
    case AutoPadType::SAME_LOWER: {
      if (1 != dilation)
        return MAKE_EP_FAIL("Dilation not supported for AutoPadType::SAME_UPPER or AutoPadType::SAME_LOWER.");

      // The ONNX spec says if `auto_pad` attribute is set, pad until the `legacy_target_size`
      // is `ceil (in_dim / stride)`. The following line of code is essentially just that and
      // is retained as is
      SafeInt<int64_t> legacy_target_size = (SafeInt<int64_t>(in_dim) + stride - 1) / stride;
      SafeInt<int64_t> pad_needed = (legacy_target_size - 1) * stride + kernel - in_dim;
      // out_dim = floor((in_dim + 2p - k) / s) + 1
      // => if (in_dim + 2p - k) is not divisible by s we can remove the floor with following equation:
      // out_dim + eps = ((in_dim + 2p - k) / s) + 1 ;where eps is in [0.0, 1.0]
      // therefore in edge cases padding can lower calculated above than it should be
      SafeInt<int64_t> actual_out_size = qnn::ComputeOutputShape(in_dim, stride, kernel, /*dilation*/ 1,
                                                                 pad_needed, pad_needed);
      if (actual_out_size < legacy_target_size) {
        pad_needed += 1;
      }
      // make sure padding is symmetric
      if (force_symmetric_auto_padding) {
        // Inlining math::roundUpPow2() from util/math.h to avoid bringing in the transitive dependencies.
        pad_needed = (pad_needed + 1) & ~1;
      }

      if (pad_type == AutoPadType::SAME_LOWER)
        pad_head = (pad_needed + 1) / 2;
      else
        pad_head = pad_needed / 2;

      pad_tail = pad_needed - pad_head;
    } break;
    default:
      return MAKE_EP_FAIL("ComputePad: pad type not supported.");
  }

  return Ort::Status();
}

inline Ort::Status ComputePadAndOutputShape(const int64_t in_dim,
                                            const int64_t stride,
                                            const int64_t kernel,
                                            const int64_t dilation,
                                            AutoPadType pad_type,
                                            int64_t& pad_head, int64_t& pad_tail,
                                            int64_t& out_dim,
                                            bool force_symmetric_auto_padding = false) {
  RETURN_IF_ERROR(
      qnn::ComputePad(in_dim, stride, kernel, dilation, pad_type, pad_head, pad_tail, force_symmetric_auto_padding));
  out_dim = qnn::ComputeOutputShape(in_dim, stride, kernel, dilation, pad_head, pad_tail);
  return Ort::Status();
}

constexpr inline int64_t ComputeTotalPad(int64_t in_size,
                                         int64_t stride,
                                         int64_t adj,
                                         int64_t kernel,
                                         int64_t dilation,
                                         int64_t out_size) {
  return std::max<int64_t>(0, (in_size - 1) * stride + adj + (kernel - 1) * dilation + 1 - out_size);
}

inline void DistributePadding(AutoPadType pad_type,
                              const int64_t& total_pad,
                              int64_t& pad_head,
                              int64_t& pad_tail) {
  if (pad_type == AutoPadType::SAME_UPPER) {
    // pad more on tail when total_pad is odd.
    pad_head = total_pad / 2;
    pad_tail = total_pad - total_pad / 2;
  } else {
    // When pad_type is NOTSET, SAME_LOWER or VALID,
    // pad more on head when total_pad is odd.
    pad_head = total_pad - total_pad / 2;
    pad_tail = total_pad / 2;
  }
}
// QNN-EP COPY END

// Layout sensitive op can't use Qnn Op validation API to verify Op support before layout transformation
// Need to check this explicitly
Ort::Status DataTypeCheckForCpuBackend(QnnModelWrapper& qnn_model_wrapper,
                                       ONNXTensorElementDataType onnx_tensor_data_type,
                                       std::string error_msg);

}  // namespace qnn
}  // namespace onnxruntime
