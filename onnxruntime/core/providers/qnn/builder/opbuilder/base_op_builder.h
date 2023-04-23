// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared/utils/utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder.h"
#include "core/framework/allocator.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;

class BaseOpBuilder : public IOpBuilder {
 public:
  BaseOpBuilder(const std::string& op_builder_type) : op_builder_type_(op_builder_type) {}
  virtual ~BaseOpBuilder() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(BaseOpBuilder);

  Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger,
                       bool is_quantized_model) const override ORT_MUST_USE_RESULT;

  Status AddToModelBuilder(QnnModelWrapper& qnn_model_wrapper,
                           const NodeUnit& node_unit,
                           const logging::Logger& logger,
                           bool is_quantized_model,
                           bool do_op_validation) const override final ORT_MUST_USE_RESULT;

  std::string GetOpBuilderType() const override;

 protected:
  virtual Qnn_DataType_t GetSupportedOutputDataType(size_t index, Qnn_DataType_t qnn_data_type) const {
    ORT_UNUSED_PARAMETER(index);
    return qnn_data_type;
  }

  virtual Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                               const NodeUnit& node_unit,
                               const logging::Logger& logger,
                               bool is_quantized_model,
                               std::vector<std::string>& input_names,
                               bool do_op_validation = false) const ORT_MUST_USE_RESULT;

  virtual Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                             const NodeUnit& node_unit,
                                             std::vector<std::string>&& input_names,
                                             const logging::Logger& logger,
                                             bool is_quantized_model,
                                             bool do_op_validation = false) const ORT_MUST_USE_RESULT;

  virtual Status ProcessOutputs(QnnModelWrapper& qnn_model_wrapper,
                                const NodeUnit& node_unit,
                                std::vector<std::string>&& input_names,
                                std::vector<std::string>&& param_tensor_names,
                                const logging::Logger& logger,
                                bool is_quantized_model,
                                bool do_op_validation) const ORT_MUST_USE_RESULT;

  Status ProcessInput(QnnModelWrapper& qnn_model_wrapper,
                      const NodeUnitIODef& input,
                      const logging::Logger& logger,
                      bool is_quantized_model,
                      std::vector<std::string>& input_names) const ORT_MUST_USE_RESULT;

  bool OnnxDataTypeToQnnDataType(const int32_t data_type, Qnn_DataType_t& qnn_data_type, bool is_quantized = false) const;

  Status GetQnnDataType(const bool is_quantized_node, const ONNX_NAMESPACE::TypeProto* type_proto,
                        Qnn_DataType_t& tensor_data_type) const {
    if (!type_proto || !type_proto->tensor_type().has_elem_type()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "The tensor doesn't have elem_type.");
    }

    int32_t onnx_data_type = type_proto->tensor_type().elem_type();
    ORT_RETURN_IF_NOT(OnnxDataTypeToQnnDataType(onnx_data_type, tensor_data_type, is_quantized_node),
                      "Failed to map Onnx data type to Qnn data type!");

    return Status::OK();
  }

  const std::string& GetNodeName(const NodeUnit& node_unit) const {
    const std::string& node_name(node_unit.Name());
    if (node_name.empty()) {
      return node_unit.Outputs()[0].node_arg.Name();
    }

    return node_name;
  }

  static const std::string& GetQnnOpType(const std::string& onnx_op_type) {
    static const std::unordered_map<std::string, std::string> onnx_op_type_to_qnn_op_type = {
        {"Add", "ElementWiseAdd"},
        {"Mul", "ElementWiseMultiply"},
        {"Abs", "ElementWiseAbs"},
        {"And", "ElementWiseAnd"},
        {"Ceil", "ElementWiseCeil"},
        {"Cast", "Cast"},
        {"Clip", "ReluMinMax"},
        {"Cos", "ElementWiseCos"},
        {"Div", "ElementWiseDivide"},
        {"Equal", "ElementWiseEqual"},
        {"Exp", "ElementWiseExp"},
        {"Floor", "ElementWiseFloor"},
        {"Gather", "Gather"},
        {"Greater", "ElementWiseGreater"},
        {"GreaterOrEqual", "ElementWiseGreaterEqual"},
        {"Less", "ElementWiseLess"},
        {"LessOrEqual", "ElementWiseLessEqual"},
        {"Log", "ElementWiseLog"},
        {"Max", "ElementWiseMaximum"},
        {"Min", "ElementWiseMinimum"},
        {"Neg", "ElementWiseNeg"},
        {"Not", "ElementWiseNot"},
        {"Or", "ElementWiseOr"},
        {"Pow", "ElementWisePower"},
        {"PRelu", "Prelu"},
        {"LeakyRelu", "Prelu"},
        {"ReduceMax", "ReduceMax"},
        {"ReduceMean", "ReduceMean"},
        {"ReduceMin", "ReduceMin"},
        {"ReduceProd", "ReduceProd"},
        {"ReduceSum", "ReduceSum"},
        {"Round", "ElementWiseRound"},
        {"Where", "ElementWiseSelect"},
        {"Sigmoid", "Sigmoid"},
        {"Sin", "ElementWiseSin"},
        {"Slice", "StridedSlice"},
        {"Split", "Split"},
        {"Softmax", "Softmax"},
        {"Sqrt", "ElementWiseSquareRoot"},
        {"Sub", "ElementWiseSubtract"},
        {"Tanh", "Tanh"},
        {"Transpose", "Transpose"},

        {"DequantizeLinear", "Dequantize"},
        {"QuantizeLinear", "Quantize"},

        {"MatMul", "MatMul"},

        {"Relu", "Relu"},
        {"Gelu", "Gelu"},
        {"Sigmoid", "Sigmoid"},

        {"Conv", "Conv2d"},

        {"GlobalAveragePool", "PoolAvg2d"},
        {"AveragePool", "PoolAvg2d"},
        {"MaxPool", "PoolMax2d"},

        {"Reshape", "Reshape"},
        {"Resize", "Resize"},
        {"Flatten", "Reshape"},
        {"Squeeze", "Reshape"},
        {"Unsqueeze", "Reshape"},

        {"LogSoftmax", "LogSoftmax"},
        {"Concat", "Concat"},

        {"Gemm", "FullyConnected"},

        {"ArgMax", "Argmax"},
        {"ArgMin", "Argmin"},
        {"ConvTranspose", "TransposeConv2d"},
        {"Tile", "Tile"},
        {"TopK", "TopK"},
        {"InstanceNormalization", "InstanceNorm"},
        {"BatchNormalization", "Batchnorm"}};
    auto it = onnx_op_type_to_qnn_op_type.find(onnx_op_type);
    ORT_ENFORCE(it != onnx_op_type_to_qnn_op_type.end());
    return it->second;
  }

  // NCHW shape to channel last
  Status NchwShapeToNhwc(const std::vector<uint32_t>& nchw_shape, std::vector<uint32_t>& nhwc_shape) const {
    ORT_ENFORCE(nchw_shape.size() == 4, "shape should have 4 dimension NCHW.");
    nhwc_shape[0] = nchw_shape[0];
    nhwc_shape[1] = nchw_shape[2];
    nhwc_shape[2] = nchw_shape[3];
    nhwc_shape[3] = nchw_shape[1];

    return Status::OK();
  }

  // NCHW shape to HWCN shape, required for Conv weight
  Status NchwShapeToHwcn(const std::vector<uint32_t>& nchw_shape, std::vector<uint32_t>& hwcn_shape) const {
    ORT_ENFORCE(nchw_shape.size() == 4, "shape should have 4 dimension NCHW.");
    hwcn_shape[0] = nchw_shape[2];
    hwcn_shape[1] = nchw_shape[3];
    hwcn_shape[2] = nchw_shape[1];
    hwcn_shape[3] = nchw_shape[0];

    return Status::OK();
  }

  // CNHW shape to HWCN shape, required for Conv weight
  Status CnhwShapeToHwcn(const std::vector<uint32_t>& cnhw_shape, std::vector<uint32_t>& hwcn_shape) const {
    ORT_ENFORCE(cnhw_shape.size() == 4, "shape should have 4 dimension CNHW.");
    hwcn_shape[0] = cnhw_shape[2];
    hwcn_shape[1] = cnhw_shape[3];
    hwcn_shape[2] = cnhw_shape[0];
    hwcn_shape[3] = cnhw_shape[1];

    return Status::OK();
  }
  Status TransposeInitializer(const onnx::TensorProto& initializer,
                              const std::vector<size_t>& perm,
                              const AllocatorPtr& cpu_allocator,
                              std::vector<uint8_t>& transposed_data) const;

  Status TransposeFromNchwToHwcn(const onnx::TensorProto& initializer,
                                 const AllocatorPtr& cpu_allocator,
                                 std::vector<uint8_t>& transposed_data) const {
    return TransposeInitializer(initializer, nchw2hwcn_perm, cpu_allocator, transposed_data);
  }

  Status TransposeFromCnhwToHwcn(const onnx::TensorProto& initializer,
                                 const AllocatorPtr& cpu_allocator,
                                 std::vector<uint8_t>& transposed_data) const {
    return TransposeInitializer(initializer, cnhw2hwcn_perm, cpu_allocator, transposed_data);
  }

  Status TwoDimensionTranspose(std::vector<uint32_t>& data_shape,
                               const onnx::TensorProto& initializer,
                               const AllocatorPtr& cpu_allocator,
                               std::vector<uint8_t>& transposed_data) const {
    auto tmp = data_shape[0];
    data_shape[0] = data_shape[1];
    data_shape[1] = tmp;
    std::vector<size_t> two_dim_trans_perm{1, 0};
    return TransposeInitializer(initializer, two_dim_trans_perm, cpu_allocator, transposed_data);
  }

  void InitializeQuantizeParam(Qnn_QuantizeParams_t& quantize_param, bool is_quantized_model, float scale = 0.0f, int32_t offset = 0) const {
    quantize_param.encodingDefinition = is_quantized_model ? QNN_DEFINITION_DEFINED : QNN_DEFINITION_UNDEFINED;
    quantize_param.quantizationEncoding = is_quantized_model ? QNN_QUANTIZATION_ENCODING_SCALE_OFFSET : QNN_QUANTIZATION_ENCODING_UNDEFINED;
    quantize_param.scaleOffsetEncoding.scale = scale;
    quantize_param.scaleOffsetEncoding.offset = offset;
  }

  // Onnx Pads is [x1_begin, x2_begin, x1_end, x2_end], QNN requires [x1_begin, x1_end, x2_begin, x2_end]
  void ReArranagePads(std::vector<int32_t>& pads) const {
    auto pads_size = pads.size();
    auto middle_pos = pads_size / 2;
    std::vector<int32_t> first_half(pads.begin(), pads.begin() + middle_pos);
    for (size_t i = 0; i < middle_pos; ++i) {
      pads[2 * i] = first_half[i];
      pads[2 * i + 1] = pads[middle_pos + i];
    }
  }

  Status ProcessAxisAttribute(const QnnModelWrapper& qnn_model_wrapper,
                              const NodeUnit& node_unit,
                              Qnn_Scalar_t& axis_qnn_scalar,
                              int32_t& default_axis_value) const;
  Qnn_TensorType_t GetInputTensorType(const QnnModelWrapper& qnn_model_wrapper, const std::string& input_name) const;

  size_t GetInputCountQnnRequired(const NodeUnit& node_unit) const {
    auto input_output_cout = GetInputOutputCountQnnRequired(node_unit.OpType());

    return 0 == input_output_cout.first ? node_unit.Inputs().size() : input_output_cout.first;
  }

  size_t GetOutputCountQnnRequired(const NodeUnit& node_unit) const {
    auto input_output_cout = GetInputOutputCountQnnRequired(node_unit.OpType());

    return 0 == input_output_cout.second ? node_unit.Outputs().size() : input_output_cout.second;
  }

 private:
  static const std::pair<size_t, size_t> GetInputOutputCountQnnRequired(std::string onnx_op_type) {
    static const std::unordered_map<std::string, std::pair<size_t, size_t>> input_output_count_qnn_required = {
        {"GlobalAveragePool", {0, 1}},
        {"MaxPool", {0, 1}},
        {"BatchNormalization", {3, 1}}};

    auto pos = input_output_count_qnn_required.find(onnx_op_type);
    if (pos == input_output_count_qnn_required.end()) {
      return std::make_pair<size_t, size_t>(0, 0);
    } else {
      return pos->second;
    }
  }

 private:
  std::string op_builder_type_;
  const std::vector<size_t> nchw2nhwc_perm{0, 2, 3, 1};
  const std::vector<size_t> nchw2hwcn_perm{2, 3, 1, 0};
  const std::vector<size_t> cnhw2hwcn_perm{2, 3, 0, 1};
};

// Type that holds information about an ONNX attribute.
template <typename ValType>
struct OnnxAttrInfo {
  std::string name;     // Attribute's name.
  ValType default_val;  // Attribute's default value.
};

template <typename ValType>
inline ValType GetOnnxAttr(const NodeAttrHelper& node_helper, const OnnxAttrInfo<ValType>& attr_info) {
  return node_helper.Get(attr_info.name, attr_info.default_val);
}

}  // namespace qnn
}  // namespace onnxruntime
