// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/framework/utils.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
namespace onnxruntime {
namespace qnn {
const int TOPK_MIN_INPUT = 2;
const int TOPK_MAX_INPUT = 2;
class TopKOpBuilder : public BaseOpBuilder {
 public:
  TopKOpBuilder() : BaseOpBuilder("TopKOpBuilder") {}

 protected:
  Qnn_DataType_t GetSupportedOutputDataType(size_t index, Qnn_DataType_t qnn_data_type) const override {
    if (index == 1) {
      if (qnn_data_type == QNN_DATATYPE_INT_64) {
        return QNN_DATATYPE_INT_32;
      } else if (qnn_data_type == QNN_DATATYPE_UINT_64) {
        return QNN_DATATYPE_UINT_32;
      }
    }
    return qnn_data_type;
  }

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

Status TopKOpBuilder::ExplictOpCheck(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit) const {
  size_t input_count = node_unit.Inputs().size();
  ORT_RETURN_IF_NOT(input_count >= TOPK_MIN_INPUT && input_count <= TOPK_MAX_INPUT,
                    "For ONNX TopK operation the expected number of inputs is 2.");
  // Skip the first input. The second input needs to be an initializer.
  const auto& input_1 = node_unit.Inputs()[1].node_arg.Name();
  if (!qnn_model_wrapper.IsInitializerInput(input_1)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "The number of top elements to retrieve must be specified as constant input.");
  }
  NodeAttrHelper node_helper(node_unit);
  auto largest = node_helper.Get("largest", 1);
  auto sorted = node_helper.Get("sorted", 1);
  if (0 == sorted) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN TopK output is always sorted");
  }
  if (0 == largest) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN TopK output is always largest values");
  }
  auto& input_0 = node_unit.Inputs()[0];
  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input_0.node_arg, input_shape), "Cannot get shape");
  auto rank = input_shape.size();
  auto axis = node_helper.Get("axis", -1);

  ORT_RETURN_IF_NOT(axis == -1 || axis == static_cast<int32_t>(rank - 1),
                    "QNN TopK's axis is always the last dimension");

  // ONNX TopK outputs int64 indices, but the equivalent QNN op outputs uint32 indices.
  // The QNN HTP backend does not generally support the int64 type, but QNN EP can just use the uint32 type
  // for TopK ops within the graph. However, if the TopK op **generates** a graph output,
  // then we cannot support it on the HTP backend.
  bool is_npu_backend = IsNpuBackend(qnn_model_wrapper.GetQnnBackendType());
  if (is_npu_backend) {
    const std::string& output_name = node_unit.Outputs()[0].node_arg.Name();
    ORT_RETURN_IF(qnn_model_wrapper.IsGraphOutput(output_name),
                  "QNN EP does not support TopK ops that generate a graph output.");
  }

  return Status::OK();
}

Status TopKOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                    const NodeUnit& node_unit,
                                    const logging::Logger& logger,
                                    std::vector<std::string>& input_names,
                                    bool do_op_validation) const {
  if (do_op_validation) {
    ORT_RETURN_IF_ERROR(ExplictOpCheck(qnn_model_wrapper, node_unit));
  }

  const auto& inputs = node_unit.Inputs();
  ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[0], logger, input_names));

  return Status::OK();
}

Status TopKOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                  const NodeUnit& node_unit,
                                                  std::vector<std::string>&& input_names,
                                                  const logging::Logger& logger,
                                                  bool do_op_validation) const {
  auto& input_name = node_unit.Inputs()[1].node_arg.Name();
  uint32_t k = 0;  // The number of elements to extract from the input tensor at each position.
  bool is_initializer_input = qnn_model_wrapper.IsInitializerInput(input_name);
  if (is_initializer_input) {
    std::vector<uint8_t> unpacked_tensor;
    const auto& input_tensor = qnn_model_wrapper.GetInitializerTensors().at(input_name);
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*input_tensor, unpacked_tensor));
    const int64_t* tensor_data = reinterpret_cast<const int64_t*>(unpacked_tensor.data());
    k = static_cast<uint32_t>(*tensor_data);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN TopK operator requires constant input parameter k.");
  }
  Qnn_Scalar_t qnn_scalar_k = QNN_SCALAR_INIT;
  qnn_scalar_k.dataType = QNN_DATATYPE_UINT_32;
  qnn_scalar_k.uint32Value = k;
  QnnParamWrapper k_param(node_unit.Index(), node_unit.Name(), QNN_OP_TOP_K_PARAM_K, qnn_scalar_k);
  std::string k_param_name = k_param.GetParamTensorName();
  qnn_model_wrapper.AddParamWrapper(std::move(k_param));
  std::vector<std::string> param_tensor_names{k_param_name};
  ORT_RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit, std::move(input_names),
                                     std::move(param_tensor_names), logger, do_op_validation,
                                     GetQnnOpType(node_unit.OpType())));
  return Status::OK();
}

void CreateTopKOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<TopKOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
