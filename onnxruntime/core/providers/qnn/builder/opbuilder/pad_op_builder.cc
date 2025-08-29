// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {
class PadOpBuilder : public BaseOpBuilder {
 public:
  PadOpBuilder() : BaseOpBuilder("PadOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(PadOpBuilder);

 protected:
  Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger,
                       std::vector<std::string>& input_names,
                       bool do_op_validation) const override ORT_MUST_USE_RESULT;

  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     const logging::Logger& logger,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

Status PadOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                   const NodeUnit& node_unit,
                                   const logging::Logger& logger,
                                   std::vector<std::string>& input_names,
                                   bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  // QNN Pad only has 1 input, the pads input & constant_value input need to be initializer and set as Qnn node parameter, axes input is not supported.
  if (do_op_validation) {
    ORT_RETURN_IF(inputs.size() > 3, "QNN Pad doesn't support axes.");
    ORT_RETURN_IF(inputs.size() < 2, "QNN Pad requires the pads input.");

    std::vector<uint32_t> input_shape;
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].node_arg, input_shape), "Cannot get shape of input 0.");
    ORT_RETURN_IF(input_shape.size() > 5, "QNN Pad doesn't support more than 5 dimension");

    auto& pads_input_name = inputs[1].node_arg.Name();
    ORT_RETURN_IF_NOT(qnn_model_wrapper.IsConstantInput(pads_input_name),
                      "Qnn doesn't support dynamic pad input");
    if (inputs.size() > 2 && inputs[2].node_arg.Exists()) {
      auto& constant_value_input_name = inputs[2].node_arg.Name();
      ORT_RETURN_IF_NOT(qnn_model_wrapper.IsConstantInput(constant_value_input_name),
                        "Qnn doesn't support dynamic constant_value input");
    }
  }

  ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[0], logger, input_names));

  return Status::OK();
}

Status ProcessConstantValue(QnnModelWrapper& qnn_model_wrapper,
                            std::vector<std::string>& param_tensor_names,
                            const NodeUnit& node_unit,
                            const NodeUnitIODef& input) {
  TensorInfo input_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(input, input_info));
  std::vector<uint8_t> unpacked_tensor;
  // Already confirmed constant_value input is initializer in ProcessInputs()
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*input_info.initializer_tensor, unpacked_tensor));
  Qnn_Scalar_t constant_value_qnn_scalar = QNN_SCALAR_INIT;
  // constant_value is quantized
  if (input.quant_param.has_value()) {
    // QNN prefers pad_constant_value quantized with quantization params same as in[0], and data stored as 32-bit signed integer
    // Onnx doesn't guarantee it has same quantization parameter as in[0], so get back the float32 value and use non-quantized data directly
    ORT_RETURN_IF_NOT(input_info.quant_param.IsPerTensor(),
                      "Pad's constant value must use per-tensor quantization");
    const Qnn_QuantizeParams_t& quant_param = input_info.quant_param.Get();
    constant_value_qnn_scalar.dataType = QNN_DATATYPE_FLOAT_32;
    float constant_value = 0;
    switch (input_info.qnn_data_type) {
      case QNN_DATATYPE_SFIXED_POINT_8: {
        auto int8_span = ReinterpretAsSpan<const int8_t>(gsl::make_span(unpacked_tensor));
        constant_value = static_cast<float>(utils::Dequantize(quant_param.scaleOffsetEncoding.offset,
                                                              quant_param.scaleOffsetEncoding.scale,
                                                              static_cast<double>(int8_span.data()[0])));
        break;
      }
      case QNN_DATATYPE_SFIXED_POINT_16: {
        auto int16_span = ReinterpretAsSpan<const int16_t>(gsl::make_span(unpacked_tensor));
        constant_value = static_cast<float>(utils::Dequantize(quant_param.scaleOffsetEncoding.offset,
                                                              quant_param.scaleOffsetEncoding.scale,
                                                              static_cast<double>(int16_span.data()[0])));
        break;
      }
      case QNN_DATATYPE_SFIXED_POINT_32: {
        auto int32_span = ReinterpretAsSpan<const int32_t>(gsl::make_span(unpacked_tensor));
        constant_value = static_cast<float>(utils::Dequantize(quant_param.scaleOffsetEncoding.offset,
                                                              quant_param.scaleOffsetEncoding.scale,
                                                              static_cast<double>(int32_span.data()[0])));
        break;
      }
      case QNN_DATATYPE_UFIXED_POINT_8: {
        constant_value = static_cast<float>(utils::Dequantize(quant_param.scaleOffsetEncoding.offset,
                                                              quant_param.scaleOffsetEncoding.scale,
                                                              static_cast<double>(unpacked_tensor.data()[0])));
        break;
      }
      case QNN_DATATYPE_UFIXED_POINT_16: {
        auto uint16_span = ReinterpretAsSpan<const uint16_t>(gsl::make_span(unpacked_tensor));
        constant_value = static_cast<float>(utils::Dequantize(quant_param.scaleOffsetEncoding.offset,
                                                              quant_param.scaleOffsetEncoding.scale,
                                                              static_cast<double>(uint16_span.data()[0])));
        break;
      }
      case QNN_DATATYPE_UFIXED_POINT_32: {
        auto uint32_span = ReinterpretAsSpan<const uint32_t>(gsl::make_span(unpacked_tensor));
        constant_value = static_cast<float>(utils::Dequantize(quant_param.scaleOffsetEncoding.offset,
                                                              quant_param.scaleOffsetEncoding.scale,
                                                              static_cast<double>(uint32_span.data()[0])));
        break;
      }
      default:
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for Pad constant_value.");
    }
    constant_value_qnn_scalar.floatValue = constant_value;
  } else {  // constant_value is non-quantized
    constant_value_qnn_scalar.dataType = input_info.qnn_data_type;
    switch (input_info.qnn_data_type) {
      case QNN_DATATYPE_UINT_8: {
        constant_value_qnn_scalar.uint8Value = unpacked_tensor.data()[0];
        break;
      }
      case QNN_DATATYPE_INT_8: {
        auto int8_span = ReinterpretAsSpan<const int8_t>(gsl::make_span(unpacked_tensor));
        constant_value_qnn_scalar.int8Value = int8_span.data()[0];
        break;
      }
      case QNN_DATATYPE_INT_16: {
        auto int16_span = ReinterpretAsSpan<const int16_t>(gsl::make_span(unpacked_tensor));
        constant_value_qnn_scalar.int16Value = int16_span.data()[0];
        break;
      }
      case QNN_DATATYPE_INT_32: {
        auto int32_span = ReinterpretAsSpan<const int32_t>(gsl::make_span(unpacked_tensor));
        constant_value_qnn_scalar.int32Value = int32_span.data()[0];
        break;
      }
      case QNN_DATATYPE_INT_64: {
        auto int64_span = ReinterpretAsSpan<const int64_t>(gsl::make_span(unpacked_tensor));
        constant_value_qnn_scalar.int64Value = int64_span.data()[0];
        break;
      }
      case QNN_DATATYPE_FLOAT_32: {
        auto float_span = ReinterpretAsSpan<const float>(gsl::make_span(unpacked_tensor));
        constant_value_qnn_scalar.floatValue = float_span.data()[0];
        break;
      }
      default:
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported.");
    }  // switch
  }  // if-else

  QnnParamWrapper constant_value_param(node_unit.Index(),
                                       node_unit.Name(),
                                       QNN_OP_PAD_PARAM_PAD_CONSTANT_VALUE,
                                       constant_value_qnn_scalar);
  param_tensor_names.push_back(constant_value_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(constant_value_param));

  return Status::OK();
}

Status PadOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                 const NodeUnit& node_unit,
                                                 std::vector<std::string>&& input_names,
                                                 const logging::Logger& logger,
                                                 bool do_op_validation) const {
  std::vector<std::string> param_tensor_names;
  // Process pads input
  // Already confirmed pads input is initializer in ProcessInputs()
  const auto& inputs = node_unit.Inputs();
  const auto& pads_input_name = inputs[1].node_arg.Name();

  std::vector<uint8_t> unpacked_tensor;
  const auto& input_tensor = qnn_model_wrapper.GetConstantTensor(pads_input_name);
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*input_tensor, unpacked_tensor));
  // Onnx Pads are int64, Qnn use uint32
  const int64_t* tensor_data = reinterpret_cast<const int64_t*>(unpacked_tensor.data());
  size_t tensor_byte_size = unpacked_tensor.size();
  size_t size = tensor_byte_size / sizeof(int64_t);

  bool has_negative = std::any_of(tensor_data, tensor_data + size, [](int64_t item) { return item < 0; });
  bool has_positive = std::any_of(tensor_data, tensor_data + size, [](int64_t item) { return item > 0; });

  if (!has_positive && !has_negative) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Got invalid zero only padding value.");
  }

  std::vector<int64_t> pad_amount_int64;
  pad_amount_int64.insert(pad_amount_int64.end(), tensor_data, tensor_data + size);

  std::vector<uint32_t> pad_amount;
  std::transform(tensor_data, tensor_data + size, std::back_inserter(pad_amount),
                 [](int64_t item) { return item < 0 ? SafeInt<uint32_t>(0) : SafeInt<uint32_t>(item); });

  // Onnx format is begin_0, begin_1, ..., end_0, end_1, ...
  // Qnn format is begin_0, end_0, begin_1, end_1, ...
  ReArrangePads(pad_amount);

  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].node_arg, input_shape), "Cannot get shape of input 0.");

  NodeAttrHelper node_helper(node_unit);
  std::string mode = node_helper.Get("mode", "constant");

  if ("reflect" == mode && has_negative) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "reflect mode doesn't support negative padding value.");
  }

  Qnn_Scalar_t mode_qnn_scalar = QNN_SCALAR_INIT;
  mode_qnn_scalar.dataType = QNN_DATATYPE_UINT_32;
  if ("constant" == mode) {
    mode_qnn_scalar.uint32Value = QNN_OP_PAD_SCHEME_CONSTANT;
  } else if ("reflect" == mode) {
    for (size_t i = 0; i < input_shape.size(); i++) {
      ORT_RETURN_IF(pad_amount[i * 2] > input_shape[i] - 1 || pad_amount[(i * 2) + 1] > input_shape[i] - 1,
                    "Pad amount should not be greater than shape(input[0])[i] - 1");
    }
    mode_qnn_scalar.uint32Value = QNN_OP_PAD_SCHEME_MIRROR_REFLECT;
  } else if ("edge" == mode) {
    mode_qnn_scalar.uint32Value = QNN_OP_PAD_SCHEME_EDGE;
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Pad mode only support constant.");
  }

  std::vector<uint32_t> pad_amount_dim{static_cast<uint32_t>(pad_amount.size() / 2), static_cast<uint32_t>(2)};
  QnnParamWrapper mode_param(node_unit.Index(), node_unit.Name(), QNN_OP_PAD_PARAM_SCHEME, mode_qnn_scalar);
  param_tensor_names.push_back(mode_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(mode_param));

  QnnParamWrapper pad_amount_param(node_unit.Index(), node_unit.Name(), QNN_OP_PAD_PARAM_PAD_AMOUNT,
                                   std::move(pad_amount_dim), std::move(pad_amount));
  param_tensor_names.push_back(pad_amount_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(pad_amount_param));

  // Process optional input constant_value
  if (inputs.size() > 2 && inputs[2].node_arg.Exists()) {
    ORT_RETURN_IF_ERROR(ProcessConstantValue(qnn_model_wrapper, param_tensor_names, node_unit, inputs[2]));
  }  // constant_value

  std::vector<uint32_t> pad_output_shape;
  if (has_negative) {
    for (uint32_t i = 0; i < input_shape.size(); ++i) {
      pad_output_shape.push_back(input_shape[i] + pad_amount[2 * i] + pad_amount[2 * i + 1]);
    }
  }

  std::string pad_output_name = utils::GetUniqueName(node_unit, "_Pad_output");
  // Step 1. Add pad if has_positive
  if (has_positive) {
    // Only positive.
    if (!has_negative) {
      ORT_RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit,
                                         std::move(input_names),
                                         std::move(param_tensor_names),
                                         logger, do_op_validation, GetQnnOpType(node_unit.OpType())));
      // Mixed sign pad value.
    } else {
      TensorInfo input_info = {};
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[0], input_info));

      std::string pad_name = utils::GetUniqueName(node_unit, "_Pad");
      QnnTensorWrapper pad_output(pad_output_name,
                                  QNN_TENSOR_TYPE_NATIVE,
                                  input_info.qnn_data_type,
                                  input_info.quant_param.Copy(),
                                  std::vector<uint32_t>(pad_output_shape));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(pad_output)),
                        "Failed to add Pad output tensor.");
      ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(pad_name,
                                                        QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                        GetQnnOpType(node_unit.OpType()),
                                                        std::move(input_names),
                                                        {pad_output_name},
                                                        std::move(param_tensor_names),
                                                        do_op_validation),
                        "Failed to add Pad node.");
    }
  }

  // Step 2. Add Slice if has_negative
  if (has_negative) {
    TensorInfo output_info = {};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[0], output_info));

    const std::string& org_output_name = node_unit.Outputs()[0].node_arg.Name();
    const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(org_output_name);
    Qnn_TensorType_t op_output_tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

    // Create output tensor
    std::vector<uint32_t> output_shape = output_info.shape;
    QnnTensorWrapper org_output(org_output_name,
                                op_output_tensor_type,
                                output_info.qnn_data_type,
                                output_info.quant_param.Copy(),
                                std::vector<uint32_t>(output_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(org_output)),
                      "Failed to add Pad output tensor.");

    // Create Slice param
    const size_t input_rank = input_shape.size();
    std::vector<uint32_t> ranges_dims{static_cast<uint32_t>(input_rank), 3};
    std::vector<uint32_t> ranges_data;
    ranges_data.reserve(input_rank);

    std::vector<int64_t> slice_amount;
    std::transform(tensor_data, tensor_data + size, std::back_inserter(slice_amount),
                   [](int64_t item) { return item > 0 ? SafeInt<int64_t>(0) : SafeInt<int64_t>(item); });

    // slice_amount in ONNX format: begin_0, begin_1, ..., end_0, end_1, ...
    for (size_t i = 0; i < input_rank; i++) {
      ranges_data.push_back(static_cast<uint32_t>(0 - slice_amount[i]));                                                       // starts
      ranges_data.push_back(static_cast<uint32_t>(static_cast<int64_t>(pad_output_shape[i]) + slice_amount[i + input_rank]));  // ends
      ranges_data.push_back(static_cast<uint32_t>(1));                                                                         // steps
    }

    QnnParamWrapper ranges_paramwrapper(node_unit.Index(),
                                        node_unit.Name(),
                                        QNN_OP_STRIDED_SLICE_PARAM_RANGES,
                                        std::move(ranges_dims),
                                        std::move(ranges_data),
                                        true);
    std::string slice_param_tensor_name(ranges_paramwrapper.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(ranges_paramwrapper));

    // Create Slice Node
    std::vector<std::string> slice_input_name = has_positive ? std::vector<std::string>{pad_output_name} : input_names;
    std::string slice_name = utils::GetUniqueName(node_unit, "_Slice");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(slice_name,
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_STRIDED_SLICE,
                                                      std::move(slice_input_name),
                                                      {org_output_name},
                                                      {slice_param_tensor_name},
                                                      do_op_validation),
                      "Failed to add Pad node.");
  }

  return Status::OK();
}

void CreatePadOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<PadOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
