// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/providers/qnn-abi/ort_api.h"
#include "core/providers/qnn-abi/builder/op_builder_factory.h"
#include "core/providers/qnn-abi/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn-abi/builder/qnn_model_wrapper.h"
#include "core/providers/qnn-abi/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

class UDOBuilder : public BaseOpBuilder {
 public:
  UDOBuilder(const std::string& op_type, const std::string& op_package)
      : BaseOpBuilder(op_type + "_UDOBuilder"), op_type_(op_type), op_package_(op_package) {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(UDOBuilder);

 protected:
  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     const logging::Logger& logger,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;

 private:
  const std::string op_type_;
  const std::string op_package_;
};

Status UDOBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                               const OrtNodeUnit& node_unit,
                                               std::vector<std::string>&& input_names,
                                               const logging::Logger& logger,
                                               bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);
  const OrtApi& ort_api = qnn_model_wrapper.GetOrtApi();
  std::string node_name = utils::GetNodeName(node_unit);
  const auto& outputs = node_unit.Outputs();
  std::vector<std::string> output_names;
  for (size_t i = 0; i < outputs.size(); ++i) {
    const auto& output_name = outputs[i].name;

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

  size_t num_attributes = 0;
  RETURN_STATUS_IF_ERROR(ort_api.Node_GetNumAttributes(&(node_unit.GetNode()), &num_attributes), ort_api);
  std::vector<const OrtOpAttr*> attributes(num_attributes);
  RETURN_STATUS_IF_ERROR(ort_api.Node_GetAttributes(&(node_unit.GetNode()), attributes.data(), attributes.size()),
                         ort_api);

  std::vector<std::string> param_names;
  OrtNodeAttrHelper node_helper(ort_api, node_unit);
  for (size_t i = 0; i < num_attributes; ++i) {
    const OrtOpAttr* attr = attributes[i];
    OrtOpAttrType attr_type = ORT_OP_ATTR_UNDEFINED;
    RETURN_STATUS_IF_ERROR(ort_api.OpAttr_GetType(attr, &attr_type), ort_api);
    const char* attribute_name;
    RETURN_STATUS_IF_ERROR(ort_api.OpAttr_GetName(attr, &attribute_name), ort_api);
    std::string attr_name = std::string(attribute_name);

    LOGS(logger, VERBOSE) << "Parse attr name: " << attr_name << " for op " << node_name;
    switch (attr_type) {
      case ORT_OP_ATTR_FLOAT: {
        auto optional_float = node_helper.GetFloat(attr_name);
        ORT_RETURN_IF_NOT(optional_float.has_value(),
                          "Failed to get values from attr ", attr_name, " in op ", node_name, " to qnn_model_wrapper.");
        ORT_RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), node_name, optional_float.value(),
                                                attr_name, param_names));
        break;
      }
      case ORT_OP_ATTR_FLOATS: {
        auto optional_floats = node_helper.GetFloats(attr_name);
        ORT_RETURN_IF_NOT(optional_floats.has_value(),
                          "Failed to get values from attr ", attr_name, " in op ", node_name, " to qnn_model_wrapper.");
        std::vector<float> floats_data(optional_floats.value().begin(), optional_floats.value().end());
        auto param_wrapper = createQnnParamWrapper<float>(node_unit.Index(), node_name, attr_name,
                                                          {static_cast<uint32_t>(floats_data.size())}, std::move(floats_data));
        param_names.push_back(param_wrapper.GetParamTensorName());
        ORT_RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(param_wrapper)),
                          "Failed to add tensor attr ", attr_name, " in op ", node_name, " to qnn_model_wrapper.");
        break;
      }
      case ORT_OP_ATTR_INT: {
        auto optional_int64 = node_helper.GetInt64(attr_name);
        ORT_RETURN_IF_NOT(optional_int64.has_value(),
                          "Failed to get values from attr ", attr_name, " in op ", node_name, " to qnn_model_wrapper.");
        ORT_RETURN_IF_ERROR(AddQnnScalar<int64_t>(qnn_model_wrapper, node_unit.Index(), node_name, optional_int64.value(),
                                                  attr_name, param_names));
        break;
      }
      case ORT_OP_ATTR_INTS: {
        auto optional_int64s = node_helper.GetInt64s(attr_name);
        ORT_RETURN_IF_NOT(optional_int64s.has_value(),
                          "Failed to get values from attr ", attr_name, " in op ", node_name, " to qnn_model_wrapper.");
        std::vector<int64_t> int64s_data(optional_int64s.value().begin(), optional_int64s.value().end());
        auto param_wrapper = createQnnParamWrapper<int64_t>(node_unit.Index(), node_name, attr_name,
                                                            {static_cast<uint32_t>(int64s_data.size())}, std::move(int64s_data));
        param_names.push_back(param_wrapper.GetParamTensorName());
        ORT_RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(param_wrapper)),
                          "Failed to add tensor attr ", attr_name, " in op ", node_name, " to qnn_model_wrapper.");
        break;
      }
      case ORT_OP_ATTR_STRING: {
        auto optional_string = node_helper.GetString(attr_name);
        ORT_RETURN_IF_NOT(optional_string.has_value(),
                          "Failed to get values from attr ", attr_name, " in op ", node_name, " to qnn_model_wrapper.");
        ORT_RETURN_IF_ERROR(AddQnnScalar(qnn_model_wrapper, node_unit.Index(), node_name, optional_string.value(),
                                         attr_name, param_names));
        break;
      }
      default: {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to add scalar attr ", attr_name, " data_type ", attr_type, " in op ", node_name, " to qnn_model_wrapper.");
      }
    }
  }

  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name,
                                                    op_package_,
                                                    op_type_,
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
