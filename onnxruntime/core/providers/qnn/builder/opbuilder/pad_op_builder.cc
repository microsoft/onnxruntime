// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/cpu/tensor/slice_helper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/common/safeint.h"

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
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
    ORT_RETURN_IF_NOT(qnn_model_wrapper.IsInitializerInput(pads_input_name),
                      "Qnn doesn't support dynamic pad input");
    if (node_unit.Inputs().size() > 2) {
      auto& constant_value_input_name = inputs[2].node_arg.Name();
      ORT_RETURN_IF_NOT(qnn_model_wrapper.IsInitializerInput(constant_value_input_name),
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
    constant_value_qnn_scalar.dataType = QNN_DATATYPE_FLOAT_32;
    float constant_value = 0;
    switch (input_info.qnn_data_type) {
      case QNN_DATATYPE_SFIXED_POINT_8: {
        auto int8_span = ReinterpretAsSpan<const int8_t>(gsl::make_span(unpacked_tensor));
        constant_value = static_cast<float>(utils::Dequantize(input_info.quant_param.scaleOffsetEncoding.offset,
                                                              input_info.quant_param.scaleOffsetEncoding.scale,
                                                              static_cast<double>(int8_span.data()[0])));
        break;
      }
      case QNN_DATATYPE_SFIXED_POINT_16: {
        auto int16_span = ReinterpretAsSpan<const int16_t>(gsl::make_span(unpacked_tensor));
        constant_value = static_cast<float>(utils::Dequantize(input_info.quant_param.scaleOffsetEncoding.offset,
                                                              input_info.quant_param.scaleOffsetEncoding.scale,
                                                              static_cast<double>(int16_span.data()[0])));
        break;
      }
      case QNN_DATATYPE_SFIXED_POINT_32: {
        auto int32_span = ReinterpretAsSpan<const int32_t>(gsl::make_span(unpacked_tensor));
        constant_value = static_cast<float>(utils::Dequantize(input_info.quant_param.scaleOffsetEncoding.offset,
                                                              input_info.quant_param.scaleOffsetEncoding.scale,
                                                              static_cast<double>(int32_span.data()[0])));
        break;
      }
      case QNN_DATATYPE_UFIXED_POINT_8: {
        constant_value = static_cast<float>(utils::Dequantize(input_info.quant_param.scaleOffsetEncoding.offset,
                                                              input_info.quant_param.scaleOffsetEncoding.scale,
                                                              static_cast<double>(unpacked_tensor.data()[0])));
        break;
      }
      case QNN_DATATYPE_UFIXED_POINT_16: {
        auto uint16_span = ReinterpretAsSpan<const uint16_t>(gsl::make_span(unpacked_tensor));
        constant_value = static_cast<float>(utils::Dequantize(input_info.quant_param.scaleOffsetEncoding.offset,
                                                              input_info.quant_param.scaleOffsetEncoding.scale,
                                                              static_cast<double>(uint16_span.data()[0])));
        break;
      }
      case QNN_DATATYPE_UFIXED_POINT_32: {
        auto uint32_span = ReinterpretAsSpan<const uint32_t>(gsl::make_span(unpacked_tensor));
        constant_value = static_cast<float>(utils::Dequantize(input_info.quant_param.scaleOffsetEncoding.offset,
                                                              input_info.quant_param.scaleOffsetEncoding.scale,
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
  }    // if-else

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
  const auto& input_tensor = qnn_model_wrapper.GetInitializerTensors().at(pads_input_name);
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*input_tensor, unpacked_tensor));
  // Onnx Pads are int64, Qnn use uint32
  const int64_t* tensor_data = reinterpret_cast<const int64_t*>(unpacked_tensor.data());
  size_t tensor_byte_size = unpacked_tensor.size();
  size_t size = tensor_byte_size / sizeof(int64_t);

  std::vector<uint32_t> pad_amount;
  std::transform(tensor_data, tensor_data + size, std::back_inserter(pad_amount),
                 [](int64_t item) { return SafeInt<uint32_t>(item); });
  // Onnx format is begin_0, begin_1, ..., end_0, end_1, ...
  // Qnn format is begin_0, end_0, begin_1, end_1, ...
  ReArranagePads(pad_amount);

  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].node_arg, input_shape), "Cannot get shape of input 0.");

  NodeAttrHelper node_helper(node_unit);
  std::string mode = node_helper.Get("mode", "constant");
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

  QnnParamWrapper multiples_param(node_unit.Index(), node_unit.Name(), QNN_OP_PAD_PARAM_PAD_AMOUNT,
                                  std::move(pad_amount_dim), std::move(pad_amount));
  param_tensor_names.push_back(multiples_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(multiples_param));

  // Process optional input constant_value
  if (node_unit.Inputs().size() > 2) {
    ORT_RETURN_IF_ERROR(ProcessConstantValue(qnn_model_wrapper, param_tensor_names, node_unit, inputs[2]));
  }  // constant_value

  ORT_RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit,
                                     std::move(input_names),
                                     std::move(param_tensor_names),
                                     logger, do_op_validation, GetQnnOpType(node_unit.OpType())));

  return Status::OK();
}

void CreatePadOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<PadOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
