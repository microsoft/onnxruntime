// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/common/safeint.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace qnn {

// Operator which only need to hanle node inputs & outputs, no attributes or no need to handle attributes
class SimpleOpBuilder : public BaseOpBuilder {
 public:
  SimpleOpBuilder() : BaseOpBuilder("SimpleOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SimpleOpBuilder);

 protected:
  Status ProcessAttributesAndOutputs(QnnModelWrapper* qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     const std::vector<std::string>& input_names,
                                     const logging::Logger& logger,
                                     bool is_quantized_model,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;

 private:
  Status ExplictOpCheck(const QnnModelWrapper* qnn_model_wrapper, const NodeUnit& node_unit) const;
  Status ProcessPermAttribute(QnnModelWrapper* qnn_model_wrapper,
                              const NodeUnit& node_unit, std::vector<QnnParamWrapper>& node_params) const;
  Status ProcessAxesAttribute(QnnModelWrapper* qnn_model_wrapper,
                              const NodeUnit& node_unit, std::vector<QnnParamWrapper>& node_params) const;
  Status ProcessAlphaAttribute(QnnModelWrapper* qnn_model_wrapper,
                               const NodeUnit& node_unit,
                               const std::string input_name) const;
};

Status SimpleOpBuilder::ExplictOpCheck(const QnnModelWrapper* qnn_model_wrapper, const NodeUnit& node_unit) const {
  if (node_unit.OpType() == "ReduceSum") {
    // TODO: still can handle it if axes input is initializer
    if (node_unit.Inputs().size() > 1) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN ReduceSum doesn't support dynamic axes.");
    }
  }

  if (node_unit.OpType() == "Softmax" && node_unit.SinceVersion() < 13) {
    int32_t default_axis = -1;
    std::vector<QnnParamWrapper> node_params;
    ORT_RETURN_IF_ERROR(ProcessAxisAttribute(qnn_model_wrapper, node_unit, node_params, default_axis));
    std::vector<uint32_t> input_shape;
    ORT_RETURN_IF_NOT(qnn_model_wrapper->GetOnnxShape(node_unit.Inputs()[0].node_arg, input_shape),
                     "Cannot get shape");
    // For Softmax opset < 13, it's still supported if axis=rank-1
    if (default_axis == static_cast<int32_t>(input_shape.size() - 1)) {
      return Status::OK();
    }

    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN Softmax only supports opset >= 13 or axis=input_rank-1.");
  }

  return Status::OK();
}

Status SimpleOpBuilder::ProcessPermAttribute(QnnModelWrapper* qnn_model_wrapper,
                                             const NodeUnit& node_unit,
                                             std::vector<QnnParamWrapper>& node_params) const {
  std::vector<uint32_t> perm_shape;
  std::vector<uint32_t> perm_data;
  auto inputs = node_unit.Inputs();
  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper->GetOnnxShape(inputs[0].node_arg, input_shape), "Cannot get shape");
  // set default perm
  uint32_t rank = static_cast<uint32_t>(input_shape.size());
  std::vector<int64_t> transpose_perm(rank);
  for (uint32_t i = 0; i < rank; ++i) {
    transpose_perm[i] = rank - 1 - i;
  }

  NodeAttrHelper node_helper(node_unit);
  transpose_perm = node_helper.Get(qnn_def::perm, transpose_perm);
  auto perm_size = static_cast<uint32_t>(transpose_perm.size());
  perm_shape.push_back(perm_size);
  perm_data.resize(perm_size);
  std::transform(transpose_perm.begin(), transpose_perm.end(), perm_data.begin(),
                 [](int64_t item) { return SafeInt<uint32_t>(item); });

  QnnParamWrapper transpose_param(node_unit.Index(), node_unit.Name(), qnn_def::perm, std::move(perm_shape), std::move(perm_data));
  node_params.push_back(std::move(transpose_param));

  return Status::OK();
}

Status SimpleOpBuilder::ProcessAxesAttribute(QnnModelWrapper* qnn_model_wrapper,
                                             const NodeUnit& node_unit,
                                             std::vector<QnnParamWrapper>& node_params) const {
  std::vector<uint32_t> axes_shape;
  std::vector<uint32_t> axes_data;
  auto inputs = node_unit.Inputs();
  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper->GetOnnxShape(inputs[0].node_arg, input_shape), "Cannot get shape");
  uint32_t rank = static_cast<uint32_t>(input_shape.size());
  std::vector<int64_t> reduce_axes(rank);
  for (uint32_t i = 0; i < rank; ++i) {
    reduce_axes[i] = i;
  }

  NodeAttrHelper node_helper(node_unit);
  reduce_axes = node_helper.Get(qnn_def::axes, reduce_axes);
  auto axex_size = static_cast<uint32_t>(reduce_axes.size());
  for (size_t i = 0; i < axex_size; ++i) {
    if (reduce_axes.at(i) < 0) {
      reduce_axes[i] += rank;
    }
  }
  axes_shape.push_back(axex_size);
  axes_data.resize(axex_size);
  std::transform(reduce_axes.begin(), reduce_axes.end(), axes_data.begin(),
                 [](int64_t item) { return SafeInt<uint32_t>(item); });
  QnnParamWrapper axes_param(node_unit.Index(), node_unit.Name(), qnn_def::axes, std::move(axes_shape), std::move(axes_data));
  node_params.push_back(std::move(axes_param));

  return Status::OK();
}

Status SimpleOpBuilder::ProcessAlphaAttribute(QnnModelWrapper* qnn_model_wrapper, const NodeUnit& node_unit, const std::string input_name) const {
  NodeAttrHelper node_helper(node_unit);
  union {
    float alpha;
    uint8_t unpack[sizeof(float)];
  } tensor_data;
  tensor_data.alpha = node_helper.Get("alpha", 0.01f);
  std::vector<uint8_t> unpacked_data(tensor_data.unpack, tensor_data.unpack + sizeof(float));
  Qnn_QuantizeParams_t quantize_param = QNN_QUANTIZE_PARAMS_INIT;
  InitializeQuantizeParam(quantize_param, false);
  std::vector<uint32_t> input_shape{1};
  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;
  Qnn_TensorType_t tensor_type = QNN_TENSOR_TYPE_STATIC;
  Qnn_TensorDataFormat_t data_format = 0;
  QnnTensorWrapper input_tensorwrapper(input_name, tensor_type, data_format, qnn_data_type, quantize_param, std::move(input_shape), std::move(unpacked_data));
  ORT_RETURN_IF_NOT(qnn_model_wrapper->AddTensor(input_name, std::move(input_tensorwrapper)), "Failed to add tensor.");
  return Status::OK();
}

Status SimpleOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper* qnn_model_wrapper,
                                                    const NodeUnit& node_unit,
                                                    const std::vector<std::string>& input_names,
                                                    const logging::Logger& logger,
                                                    bool is_quantized_model,
                                                    bool do_op_validation) const {
  if (input_names.size() < 1) {
    return Status::OK();
  }

  if (do_op_validation) {
    ORT_RETURN_IF_ERROR(ExplictOpCheck(qnn_model_wrapper, node_unit));
  }

  std::vector<QnnParamWrapper> node_params;

  // Add attribute
  if (node_unit.OpType() == "LogSoftmax" || node_unit.OpType() == "Softmax" || node_unit.OpType() == "Concat") {
    int32_t default_axis = ("Softmax" == node_unit.OpType()) ? -1 : 0;
    ORT_RETURN_IF_ERROR(ProcessAxisAttribute(qnn_model_wrapper, node_unit, node_params, default_axis));
  }

  if (node_unit.OpType() == "MatMul") {
    Qnn_Scalar_t scalar_param = QNN_SCALAR_INIT;
    scalar_param.dataType = QNN_DATATYPE_BOOL_8;
    scalar_param.bool8Value = 0;
    QnnParamWrapper transpose_in0_param(qnn_def::transpose_in0, scalar_param);
    node_params.push_back(std::move(transpose_in0_param));

    QnnParamWrapper transpose_in1_param(qnn_def::transpose_in1, scalar_param);
    node_params.push_back(std::move(transpose_in1_param));
  }

  if (node_unit.OpType() == "ReduceMax" || node_unit.OpType() == "ReduceMean" ||
      node_unit.OpType() == "ReduceMin" || node_unit.OpType() == "ReduceProd" ||
      node_unit.OpType() == "ReduceSum") {
    ORT_RETURN_IF_ERROR(ProcessAxesAttribute(qnn_model_wrapper, node_unit, node_params));

    NodeAttrHelper node_helper(node_unit);
    auto onnx_keepdims = node_helper.Get("keepdims", (int32_t)1);
    Qnn_Scalar_t scalar_param = QNN_SCALAR_INIT;
    scalar_param.dataType = QNN_DATATYPE_BOOL_8;
    scalar_param.bool8Value = static_cast<uint8_t>(onnx_keepdims == 0 ? 0 : 1);
    QnnParamWrapper keep_dims_param(qnn_def::keep_dims, scalar_param);
    node_params.push_back(std::move(keep_dims_param));
  }

  if (node_unit.OpType() == "Transpose") {
    ORT_RETURN_IF_ERROR(ProcessPermAttribute(qnn_model_wrapper, node_unit, node_params));
  }

  if (node_unit.OpType() == "LeakyRelu") {
    std::string input_name = "alpha";
    ORT_RETURN_IF_ERROR(ProcessAlphaAttribute(qnn_model_wrapper, node_unit, input_name));
    const_cast<std::vector<std::string>&>(input_names).push_back(input_name);
  }

  ORT_RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit, input_names, std::move(node_params),
                                     logger, is_quantized_model, do_op_validation));

  return Status::OK();
}

void CreateSimpleOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<SimpleOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
