// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <vector>

#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/common/safeint.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace qnn {

class TransposeOpBuilder : public BaseOpBuilder {
 public:
  TransposeOpBuilder() : BaseOpBuilder("TransposeOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TransposeOpBuilder);

 protected:
  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     const logging::Logger& logger,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;

 private:
  Status ProcessPermAttribute(QnnModelWrapper& qnn_model_wrapper,
                              const NodeUnit& node_unit,
                              std::vector<std::string>& param_tensor_names) const;
};

Status TransposeOpBuilder::ProcessPermAttribute(QnnModelWrapper& qnn_model_wrapper,
                                                const NodeUnit& node_unit,
                                                std::vector<std::string>& param_tensor_names) const {
  auto inputs = node_unit.Inputs();
  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].node_arg, input_shape), "Cannot get shape");
  // set default perm
  uint32_t rank = static_cast<uint32_t>(input_shape.size());
  std::vector<int64_t> transpose_perm(rank);
  for (uint32_t i = 0; i < rank; ++i) {
    transpose_perm[i] = rank - 1 - i;
  }

  NodeAttrHelper node_helper(node_unit);
  transpose_perm = node_helper.Get("perm", transpose_perm);
  auto perm_size = static_cast<uint32_t>(transpose_perm.size());
  std::vector<uint32_t> perm_shape{perm_size};
  std::vector<uint32_t> perm_data;
  perm_data.resize(perm_size);
  std::transform(transpose_perm.begin(), transpose_perm.end(), perm_data.begin(),
                 [](int64_t item) { return SafeInt<uint32_t>(item); });

  QnnParamWrapper transpose_param(node_unit.Index(), node_unit.Name(), QNN_OP_TRANSPOSE_PARAM_PERM,
                                  std::move(perm_shape), std::move(perm_data));
  param_tensor_names.push_back(transpose_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(transpose_param));

  return Status::OK();
}

Status TransposeOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                       const NodeUnit& node_unit,
                                                       std::vector<std::string>&& input_names,
                                                       const logging::Logger& logger,
                                                       bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);

  if (input_names.size() < 1) {
    return Status::OK();
  }

  std::vector<std::string> param_tensor_names;
  ORT_RETURN_IF_ERROR(ProcessPermAttribute(qnn_model_wrapper, node_unit, param_tensor_names));

  const auto& output_name = node_unit.Outputs()[0].node_arg.Name();
  std::vector<std::string> output_names;

  bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);
  Qnn_TensorType_t tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

  std::vector<uint32_t> output_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(node_unit.Outputs()[0].node_arg, output_shape),
                    "Cannot get shape");

  const QnnTensorWrapper& input_tensor_wrapper = qnn_model_wrapper.GetQnnTensorWrapper(input_names[0]);

  // Transpose output uses same data type and quantization parameter with input
  // 1. In QDQ model, the optimization may create scenario like Q -> Transpose -> DQ, Transpose is single node
  // Input tensor is created by previous node which is quantized tensor,
  // so output just copy the same data type and quantization parameters
  // 2. In QDQ model, Transpose also support non-quantized data like int32.
  QnnTensorWrapper output_tensorwrapper(output_name,
                                        tensor_type,
                                        input_tensor_wrapper.GetTensorDataType(),
                                        GetQnnTensorQParams(input_tensor_wrapper.GetQnnTensor()),
                                        std::move(output_shape));

  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)), "Failed to add tensor.");

  output_names.push_back(output_name);
  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(GetNodeName(node_unit),
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_TRANSPOSE,
                                                    std::move(input_names),
                                                    std::move(output_names),
                                                    std::move(param_tensor_names),
                                                    do_op_validation),
                    "Failed to add node.");

  return Status::OK();
}

void CreateTransposeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<TransposeOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
