// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {
class ModOpBuilder : public BaseOpBuilder {
 public:
  ModOpBuilder() : BaseOpBuilder("ModOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ModOpBuilder);

 protected:
  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     const logging::Logger& logger,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

Status ModOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                 const NodeUnit& node_unit,
                                                 std::vector<std::string>&& input_names,
                                                 const logging::Logger& logger,
                                                 bool do_op_validation) const {
  NodeAttrHelper node_helper(node_unit);
  auto fmod = node_helper.Get("fmod", (int32_t)0);

  // QNN has separate operators for Interger Mod and FMod
  const std::string op_type = fmod == 1 ? QNN_OP_ELEMENT_WISE_FMOD : QNN_OP_ELEMENT_WISE_MOD;

  ORT_RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit,
                                     std::move(input_names), {},
                                     logger, do_op_validation, op_type));

  return Status::OK();
}

void CreateModOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<ModOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
