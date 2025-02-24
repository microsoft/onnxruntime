// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cassert>
#include <limits>

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {
class ClipOpBuilder : public BaseOpBuilder {
 public:
  ClipOpBuilder() : BaseOpBuilder("ClipOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ClipOpBuilder);

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

 private:
  Status ExplictOpCheck(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit) const;
};

static Status ProcessClipMinMax(QnnModelWrapper& qnn_model_wrapper,
                                const NodeUnitIODef& input,
                                float& float_value) {
  TensorInfo input_info = {};
  std::vector<uint8_t> val_bytes;
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(input, input_info));
  assert(input_info.is_initializer);  // Checked by ExplicitOpCheck().
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*input_info.initializer_tensor, val_bytes));
  switch (input_info.qnn_data_type) {
    case QNN_DATATYPE_INT_8: {
      float_value = static_cast<float>(*reinterpret_cast<int8_t*>(val_bytes.data()));
      break;
    }
    case QNN_DATATYPE_INT_16: {
      float_value = static_cast<float>(*reinterpret_cast<int16_t*>(val_bytes.data()));
      break;
    }
    case QNN_DATATYPE_INT_32: {
      float_value = static_cast<float>(*reinterpret_cast<int32_t*>(val_bytes.data()));
      break;
    }
    case QNN_DATATYPE_INT_64: {
      float_value = static_cast<float>(*reinterpret_cast<int64_t*>(val_bytes.data()));
      break;
    }
    case QNN_DATATYPE_UINT_8: {
      float_value = static_cast<float>(*val_bytes.data());
      break;
    }
    case QNN_DATATYPE_UINT_16: {
      float_value = static_cast<float>(*reinterpret_cast<uint16_t*>(val_bytes.data()));
      break;
    }
    case QNN_DATATYPE_UINT_32: {
      float_value = static_cast<float>(*reinterpret_cast<uint32_t*>(val_bytes.data()));
      break;
    }
    case QNN_DATATYPE_UINT_64: {
      float_value = static_cast<float>(*reinterpret_cast<uint64_t*>(val_bytes.data()));
      break;
    }
    case QNN_DATATYPE_FLOAT_16: {
      MLFloat16 fp16_value = *reinterpret_cast<const MLFloat16*>(val_bytes.data());
      float_value = fp16_value.ToFloat();
      break;
    }
    case QNN_DATATYPE_FLOAT_32: {
      float_value = *reinterpret_cast<const float*>(val_bytes.data());
      break;
    }
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "min/max input data type not supported.");
  }

  return Status::OK();
}

Status ClipOpBuilder::ExplictOpCheck(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit) const {
  if (node_unit.Inputs().size() > 1) {
    const auto& min_input_name = node_unit.Inputs()[1].node_arg.Name();
    if (!min_input_name.empty() && !qnn_model_wrapper.IsConstantInput(min_input_name)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN desn't support dynamic min/max.");
    }
  }
  if (node_unit.Inputs().size() > 2) {
    const auto& max_input_name = node_unit.Inputs()[2].node_arg.Name();
    if (!max_input_name.empty() && !qnn_model_wrapper.IsConstantInput(max_input_name)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN desn't support dynamic min/max.");
    }
  }
  return Status::OK();
}

Status ClipOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                    const NodeUnit& node_unit,
                                    const logging::Logger& logger,
                                    std::vector<std::string>& input_names,
                                    bool do_op_validation) const {
  if (do_op_validation) {
    ORT_RETURN_IF_ERROR(ExplictOpCheck(qnn_model_wrapper, node_unit));
  }

  return ProcessInput(qnn_model_wrapper, node_unit.Inputs()[0], logger, input_names);
}

Status ClipOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                  const NodeUnit& node_unit,
                                                  std::vector<std::string>&& input_names,
                                                  const logging::Logger& logger,
                                                  bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  const size_t num_inputs = inputs.size();

  const Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;
  std::vector<std::string> param_tensor_names;

  // Set the 'min' parameter.
  Qnn_Scalar_t min_qnn_scalar = QNN_SCALAR_INIT;
  min_qnn_scalar.dataType = qnn_data_type;

  if (num_inputs > 1 && !inputs[1].node_arg.Name().empty()) {
    ORT_RETURN_IF_ERROR(ProcessClipMinMax(qnn_model_wrapper, inputs[1], min_qnn_scalar.floatValue));
  } else {
    min_qnn_scalar.floatValue = std::numeric_limits<float>::lowest();
  }

  QnnParamWrapper min_value_param(node_unit.Index(), node_unit.Name(), QNN_OP_RELU_MIN_MAX_PARAM_MIN_VALUE,
                                  min_qnn_scalar);
  param_tensor_names.push_back(min_value_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(min_value_param));

  // Set the 'max' parameter.
  Qnn_Scalar_t max_qnn_scalar = QNN_SCALAR_INIT;
  max_qnn_scalar.dataType = qnn_data_type;

  if (num_inputs > 2 && !inputs[2].node_arg.Name().empty()) {
    ORT_RETURN_IF_ERROR(ProcessClipMinMax(qnn_model_wrapper, inputs[2], max_qnn_scalar.floatValue));
  } else {
    max_qnn_scalar.floatValue = std::numeric_limits<float>::max();
  }

  QnnParamWrapper max_value_param(node_unit.Index(), node_unit.Name(), QNN_OP_RELU_MIN_MAX_PARAM_MAX_VALUE,
                                  max_qnn_scalar);
  param_tensor_names.push_back(max_value_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(max_value_param));

  ORT_RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit,
                                     std::move(input_names),
                                     std::move(param_tensor_names),
                                     logger, do_op_validation, GetQnnOpType(node_unit.OpType())));

  return Status::OK();
}

void CreateClipOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<ClipOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
