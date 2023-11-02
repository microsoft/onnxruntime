// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/common/safeint.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace qnn {

class ExpandOpBuilder : public BaseOpBuilder {
 public:
  ExpandOpBuilder() : BaseOpBuilder("ExpandOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ExpandOpBuilder);

 protected:
  Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger,
                       std::vector<std::string>& input_names,
                       bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

template <typename T>
void FillNonQuantizedInput(std::vector<uint8_t>& shape_data, int shape_size, T ini_value) {
  shape_data.resize(shape_size * sizeof(T));
  T* shape_data_float = reinterpret_cast<T*>(shape_data.data());
  std::fill(shape_data_float, shape_data_float + shape_size, ini_value);
}

// Use ElementWiseMultiply to implement data broadcast
// Get the shape data, and create a initializer input with value 1 and same shape
// input[0] * input[1]
Status ExpandOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                      const NodeUnit& node_unit,
                                      const logging::Logger& logger,
                                      std::vector<std::string>& input_names,
                                      bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);
  const auto& inputs = node_unit.Inputs();
  ORT_RETURN_IF(inputs.size() != 2, "Expand should has 2 inputs!");

  ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[0], logger, input_names));

  // Process shape input
  const auto& input_name = inputs[1].node_arg.Name();
  bool is_initializer_input = qnn_model_wrapper.IsInitializerInput(input_name);
  ORT_RETURN_IF_NOT(is_initializer_input, "QNN doesn't support dynamic shape.");

  std::vector<uint32_t> shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[1].node_arg, shape), "Cannot get shape");
  uint32_t shape_rank = shape[0];
  std::vector<uint8_t> unpacked_tensor;
  const auto& input_tensor = qnn_model_wrapper.GetInitializerTensors().at(input_name);
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*input_tensor, unpacked_tensor));
  const int64_t* shape_data_int64 = reinterpret_cast<const int64_t*>(unpacked_tensor.data());
  std::vector<uint32_t> input_shape(shape_rank, 0);
  std::transform(shape_data_int64, shape_data_int64 + shape_rank, input_shape.begin(),
                 [](int64_t item) { return SafeInt<uint32_t>(item); });
  int shape_size = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<uint32_t>());

  std::vector<uint8_t> shape_data;
  bool is_quantized_tensor = inputs[0].quant_param.has_value();
  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;
  const auto* type_proto = inputs[0].node_arg.TypeAsProto();
  Qnn_QuantizeParams_t quantize_param = QNN_QUANTIZE_PARAMS_INIT;
  if (is_quantized_tensor) {
    ORT_RETURN_IF_ERROR(utils::GetQnnDataType(true, type_proto, qnn_data_type));
    float scale = 0.0f;
    int zero_point = 0;
    float rmax = 1.0f;
    float rmin = 1.0f;
    ORT_RETURN_IF_ERROR(utils::GetQuantParams(rmin,
                                              rmax,
                                              qnn_data_type,
                                              scale,
                                              zero_point));
    utils::InitializeQuantizeParam(quantize_param, true, scale, zero_point);
    int quant_value_int = 0;
    double ini_value = 1.0;
    ORT_RETURN_IF_ERROR(utils::Quantize(ini_value, scale, zero_point, qnn_data_type, quant_value_int));
    switch (qnn_data_type) {
      case QNN_DATATYPE_SFIXED_POINT_8: {
        FillNonQuantizedInput(shape_data, shape_size, static_cast<int8_t>(quant_value_int));
        break;
      }
      case QNN_DATATYPE_UFIXED_POINT_8: {
        FillNonQuantizedInput(shape_data, shape_size, static_cast<uint8_t>(quant_value_int));
        break;
      }
      case QNN_DATATYPE_UFIXED_POINT_16: {
        FillNonQuantizedInput(shape_data, shape_size, static_cast<uint16_t>(quant_value_int));
        break;
      }
      default:
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported.");
    }  // switch
  } else {
    ORT_RETURN_IF_ERROR(utils::GetQnnDataType(false, type_proto, qnn_data_type));
    switch (qnn_data_type) {
      case QNN_DATATYPE_FLOAT_32: {
        FillNonQuantizedInput(shape_data, shape_size, static_cast<float>(1.0));
        break;
      }
      case QNN_DATATYPE_INT_32: {
        FillNonQuantizedInput(shape_data, shape_size, static_cast<int32_t>(1));
        break;
      }
      case QNN_DATATYPE_UINT_32: {
        FillNonQuantizedInput(shape_data, shape_size, static_cast<uint32_t>(1));
        break;
      }
      default:
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported.");
    }  // switch
  }    // if-else

  std::string shape_input_name(input_name + "_mul");
  QnnTensorWrapper input_tensorwrapper(shape_input_name, QNN_TENSOR_TYPE_STATIC, qnn_data_type, quantize_param,
                                       std::move(input_shape), std::move(shape_data));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");

  input_names.push_back(shape_input_name);

  return Status::OK();
}

void CreateExpandOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<ExpandOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
