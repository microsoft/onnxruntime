// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

class UDOBuilder : public BaseOpBuilder {
 public:
  UDOBuilder(const std::string& op_type, const std::string& op_package) : BaseOpBuilder(op_type + "_UDOBuilder"), op_package_(op_package) { (void)op_type; }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(UDOBuilder);

 protected:
  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     const logging::Logger& logger,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;

 private:
  const std::string op_package_;
};

Status UDOBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                               const NodeUnit& node_unit,
                                               std::vector<std::string>&& input_names,
                                               const logging::Logger& logger,
                                               bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);
  std::string node_name = utils::GetNodeName(node_unit);
  const auto& outputs = node_unit.Outputs();
  std::vector<std::string> output_names;
  for (size_t i = 0; i < outputs.size(); ++i) {
    const auto& output_name = outputs[i].node_arg.Name();

    TensorInfo output_info = {};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(outputs[i], output_info));
    bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);

    Qnn_TensorType_t tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
    QnnTensorWrapper output_tensorwrapper(output_name,
                                          tensor_type,
                                          output_info.qnn_data_type,
                                          std::move(output_info.quant_param),
                                          std::move(output_info.shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)), "Failed to add tensor.");
    output_names.emplace_back(output_name);
  }
  std::vector<std::string> param_names;
  NodeAttrHelper node_helper(node_unit);
  auto& attrs = node_unit.GetNode().GetAttributes();
  for (auto& attr : attrs) {
    LOGS(logger, VERBOSE) << "Parse attr name: " << attr.first << " for op " << node_name;
    switch (attr.second.type()) {
      case ONNX_NAMESPACE::AttributeProto::FLOAT:
        ORT_RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), node_name, attr.second.f(), attr.first, param_names));
        break;
      case ONNX_NAMESPACE::AttributeProto::INT:
        ORT_RETURN_IF_ERROR(AddQnnScalar<int64_t>(qnn_model_wrapper, node_unit.Index(), node_name, attr.second.i(), attr.first, param_names));
        break;
      default:
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to add attr ", attr.first, " to qnn_model_wrapper.");
    }
  }

  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name,
                                                    op_package_,
                                                    GetQnnOpType(node_unit.OpType()),  // Typically GetQnnOpType(), but can be overridden.
                                                    std::move(input_names),
                                                    std::move(output_names),
                                                    std::move(param_names),
                                                    do_op_validation),
                    "Failed to add node.");
  return Status::OK();
}

void CreateUDOBuilder(const std::string& op_type, const std::string& op_package, OpBuilderRegistrations& op_registrations) {
  op_registrations.RegisterUDOBuilder(op_type, std::make_unique<UDOBuilder>(op_type, op_package));
}

}  // namespace qnn
}  // namespace onnxruntime
