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

namespace onnxruntime {
namespace qnn {
class TileOpBuilder : public BaseOpBuilder {
 public:
  TileOpBuilder() : BaseOpBuilder("TileOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TileOpBuilder);

 protected:
  Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger,
                       bool is_quantized_model,
                       std::vector<std::string>& input_names,
                       bool do_op_validation) const override ORT_MUST_USE_RESULT;

  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     const logging::Logger& logger,
                                     bool is_quantized_model,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

Status TileOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                    const NodeUnit& node_unit,
                                    const logging::Logger& logger,
                                    bool is_quantized_model,
                                    std::vector<std::string>& input_names,
                                    bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  // QNN Tile only support 1 input, the 2nd input need to be initialier and set as Qnn node parameter
  if (do_op_validation) {
    auto& repeats_input_name = inputs[1].node_arg.Name();
    ORT_RETURN_IF_NOT(qnn_model_wrapper.IsInitializerInput(repeats_input_name),
                      "Qnn doesn't support dynamic repeats input");
  }

  ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[0], logger, is_quantized_model, input_names));

  return Status::OK();
}

Status TileOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                  const NodeUnit& node_unit,
                                                  std::vector<std::string>&& input_names,
                                                  const logging::Logger& logger,
                                                  bool is_quantized_model,
                                                  bool do_op_validation) const {
  std::vector<std::string> param_tensor_names;
  // Already confirmed repeats input is initailizer in ProcessInputs()
  const auto& repeats_input_name = node_unit.Inputs()[1].node_arg.Name();

  std::vector<uint8_t> unpacked_tensor;
  const auto& input_tensor = qnn_model_wrapper.GetInitializerTensors().at(repeats_input_name);
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*input_tensor, unpacked_tensor));
  // Onnx repeats are int64, Qnn use uint32
  const int64_t* tensor_data = reinterpret_cast<const int64_t*>(unpacked_tensor.data());
  size_t tensor_byte_size = unpacked_tensor.size();
  size_t size = tensor_byte_size / sizeof(int64_t);

  std::vector<uint32_t> multiples;
  std::transform(tensor_data, tensor_data + size, std::back_inserter(multiples),
                 [](int64_t item) { return SafeInt<uint32_t>(item); });

  uint32_t multiples_size = static_cast<uint32_t>(multiples.size());
  std::vector<uint32_t> multiples_dim{multiples_size};
  QnnParamWrapper multiples_param(node_unit.Index(), node_unit.Name(), qnn_def::multiples, std::move(multiples_dim),
                                  std::move(multiples));
  param_tensor_names.push_back(multiples_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(multiples_param));

  ORT_RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit,
                                     std::move(input_names),
                                     std::move(param_tensor_names),
                                     logger, is_quantized_model, do_op_validation, GetQnnOpType(node_unit.OpType())));

  return Status::OK();
}

void CreateTileOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<TileOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
