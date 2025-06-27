// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <vector>

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

class NonZeroOpBuilder : public BaseOpBuilder {
 public:
  NonZeroOpBuilder() : BaseOpBuilder("NonZeroOpBuilder") {}

 protected:
  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     const logging::Logger& logger,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

Status NonZeroOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                     const NodeUnit& node_unit,
                                                     std::vector<std::string>&& input_names,
                                                     const logging::Logger& logger,
                                                     bool do_op_validation) const {
  // Handle a corner case explicitly, which can pass backend validation but in fact not executable.
  const std::vector<uint32_t>& input_shape = qnn_model_wrapper.GetQnnTensorWrapper(input_names[0]).GetTensorDims();
  for (const uint32_t& dim : input_shape) {
    ORT_RETURN_IF(dim == 0, "QNN does not support NonZero with empty input.");
  }

  const auto& output = node_unit.Outputs()[0];
  const std::string& output_name = output.node_arg.Name();

  TensorInfo output_info = {};
  Status status = qnn_model_wrapper.GetTensorInfo(output, output_info);
  if (!status.IsOK()) {
    LOGS(logger, ERROR) << "Encountering NonZero " << node_unit.Name() << " which has dynamically shaped output tensor."
                        << "QNN supports NonZero by allocating maximum possible size (i.e., all elements != 0), "
                        << "and fills only the detected nonzero elements in the output tensor."
                        << "The model must be preproceesed to eliminate the dynamic shapes first for QNN to support.";
    return status;
  }

  // ONNX NonZero has shape [input_rank, #input_elements].
  uint32_t rank = output_info.shape[0];
  uint32_t num_elements = output_info.shape[1];

  // QNN NonZero has shape [#input elements, input_rank], and thus an extra Transpose must be inserted afterwards.
  const std::string transpose_input_name = utils::GetUniqueName(output_name, +"_transpose");
  const std::vector<uint32_t> transpose_input_shape{num_elements, rank};
  QnnTensorWrapper output_tensorwrapper(transpose_input_name,
                                        QNN_TENSOR_TYPE_NATIVE,
                                        output_info.qnn_data_type,
                                        output_info.quant_param.Copy(),
                                        std::vector<uint32_t>(transpose_input_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)), "Failed to add tensor.");
  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetUniqueName(node_unit),
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    GetQnnOpType(node_unit.OpType()),
                                                    std::move(input_names),
                                                    {transpose_input_name},
                                                    {},
                                                    do_op_validation),
                    "Failed to add NonZero node.");

  // NonZero's output is indices which is INT64 dtype. If it happens to be graph output as well, add a Cast node to
  // cast the dtype back to INT64 since wrapper construction implicitly changes the dtype to INT32.
  const bool is_cast_required = output_info.qnn_data_type == QNN_DATATYPE_INT_64 &&
                                qnn_model_wrapper.IsGraphOutput(output_name);
  const std::string transpose_output_name = is_cast_required ? utils::GetUniqueName(output_name, "_cast") : output_name;

  std::vector<uint32_t> transpose_perm{1, 0};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(node_unit.Index(),
                                                         transpose_input_name,
                                                         transpose_output_name,
                                                         transpose_input_shape,
                                                         transpose_perm,
                                                         output_info.shape,
                                                         output_info.qnn_data_type,
                                                         output_info.quant_param,
                                                         do_op_validation,
                                                         false,
                                                         false));

  if (is_cast_required) {
    QnnTensorWrapper cast_output_tensorwrapper(output_name,
                                               QNN_TENSOR_TYPE_APP_READ,
                                               output_info.qnn_data_type,
                                               output_info.quant_param.Copy(),
                                               std::vector<uint32_t>(output_info.shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(cast_output_tensorwrapper)),
                      "Failed to add tensor.");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetUniqueName(node_unit, QNN_OP_CAST),
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_CAST,
                                                      {transpose_output_name},
                                                      {output_name},
                                                      {},
                                                      do_op_validation),
                      "Failed to add node");
  }

  return Status::OK();
}

void CreateNonZeroOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<NonZeroOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
