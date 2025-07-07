// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/providers/qnn-abi/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn-abi/builder/op_builder_factory.h"
#include "core/providers/qnn-abi/builder/qnn_utils.h"

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
  size_t output_count = node_unit.Outputs().size();
  ORT_RETURN_IF_NOT(input_count >= TOPK_MIN_INPUT && input_count <= TOPK_MAX_INPUT,
                    "For ONNX TopK operation the expected number of inputs is 2.");
  ORT_RETURN_IF_NOT(output_count == 2, "QNN TopK expects exactly 2 outputs.");

  // Skip the first input. The second input needs to be an initializer.
  const auto& input_1 = node_unit.Inputs()[1].node_arg.Name();
  if (!qnn_model_wrapper.IsConstantInput(input_1)) {
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

  // HTP only supports TopK at the last axis, and thus check whether extra Transpose is required.
  TensorInfo input_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[0], input_info));

  size_t input_rank = input_info.shape.size();
  int32_t axis = NodeAttrHelper(node_unit).Get("axis", -1);
  if (axis == -1 || axis == static_cast<int32_t>(input_rank - 1)) {
    return Status::OK();
  }

  // Add Transpose to permute axis to the last.
  std::string transpose_output_name = input_names[0] + "_ort_qnn_ep_transpose";
  std::vector<uint32_t> transpose_perm;
  ORT_RETURN_IF_ERROR(utils::GetPermToLastAxis(static_cast<uint32_t>(axis),
                                               static_cast<uint32_t>(input_rank),
                                               transpose_perm));

  std::vector<uint32_t> transpose_output_shape = input_info.shape;
  transpose_output_shape[input_rank - 1] = input_info.shape[axis];
  transpose_output_shape[axis] = input_info.shape[input_rank - 1];

  ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(node_unit.Index(),
                                                         input_names[0],
                                                         transpose_output_name,
                                                         input_info.shape,
                                                         transpose_perm,
                                                         transpose_output_shape,
                                                         input_info.qnn_data_type,
                                                         input_info.quant_param,
                                                         do_op_validation,
                                                         false,
                                                         false));
  input_names[0] = transpose_output_name;

  return Status::OK();
}

Status TopKOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                  const NodeUnit& node_unit,
                                                  std::vector<std::string>&& input_names,
                                                  const logging::Logger& logger,
                                                  bool do_op_validation) const {
  auto& input_name = node_unit.Inputs()[1].node_arg.Name();
  uint32_t k = 0;  // The number of elements to extract from the input tensor at each position.
  bool is_constant_input = qnn_model_wrapper.IsConstantInput(input_name);
  if (is_constant_input) {
    std::vector<uint8_t> unpacked_tensor;
    const auto& input_tensor = qnn_model_wrapper.GetConstantTensor(input_name);
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

  // HTP only supports TopK at the last axis, and thus check whether extra Transpose is required.
  TensorInfo input_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[0], input_info));

  size_t input_rank = input_info.shape.size();
  int32_t axis = NodeAttrHelper(node_unit).Get("axis", -1);
  if (axis == -1 || axis == static_cast<int32_t>(input_rank - 1)) {
    ORT_RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper,
                                       node_unit,
                                       std::move(input_names),
                                       std::move(param_tensor_names),
                                       logger,
                                       do_op_validation,
                                       GetQnnOpType(node_unit.OpType())));
    return Status::OK();
  }

  const auto& outputs = node_unit.Outputs();
  std::vector<std::string> transpose_input_names;
  std::vector<std::vector<std::uint32_t>> transpose_input_shapes;

  // Add TopK outputs.
  for (size_t output_idx = 0; output_idx < 2; ++output_idx) {
    const auto& output = outputs[output_idx];

    // Since user may not be aware of the additional Transpose, the original output name of TopK node must be used by
    // the additional Transpose node which has the same output as original TopK node.
    const std::string& output_name = output.node_arg.Name();
    std::string transpose_input_name = output_name + "_ort_qnn_ep_transpose";
    transpose_input_names.push_back(std::move(transpose_input_name));

    // Since the input of TopK node is permuted, its output shape must be manually calculated.
    TensorInfo output_info = {};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(output, output_info));
    size_t output_rank = output_info.shape.size();

    std::vector<uint32_t> transpose_input_shape = output_info.shape;
    transpose_input_shape[output_rank - 1] = output_info.shape[axis];
    transpose_input_shape[axis] = output_info.shape[output_rank - 1];
    transpose_input_shapes.push_back(std::move(transpose_input_shape));

    QnnTensorWrapper output_tensorwrapper(transpose_input_names[output_idx],
                                          QNN_TENSOR_TYPE_NATIVE,
                                          output_info.qnn_data_type,
                                          output_info.quant_param.Copy(),
                                          std::vector<uint32_t>(transpose_input_shapes[output_idx]));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)), "Failed to add tensor.");
  }

  // Add TopK node.
  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetNodeName(node_unit),
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    GetQnnOpType(node_unit.OpType()),
                                                    std::move(input_names),
                                                    std::vector<std::string>(transpose_input_names),
                                                    std::move(param_tensor_names)),
                    "Failed to add node.");

  // Add Transpose nodes for each output to permute back.
  for (size_t output_idx = 0; output_idx < 2; ++output_idx) {
    const auto& output = outputs[output_idx];
    const std::string& output_name = output.node_arg.Name();

    TensorInfo output_info = {};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(output, output_info));
    size_t output_rank = output_info.shape.size();

    std::vector<uint32_t> transpose_perm;
    ORT_RETURN_IF_ERROR(utils::GetPermToLastAxis(static_cast<uint32_t>(axis),
                                                 static_cast<uint32_t>(output_rank),
                                                 transpose_perm));

    std::string transpose_output_name = output_name;
    bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);

    // TopK's second output is indices which could be INT64 dtype, and QnnTensorWrapper directly changes the dtype to
    // INT32 during the wrapper construction. Nevertheless, if this output happens to be graph output, an additional
    // Cast must be added to cast dtype from INT32 back to INT64.
    bool is_cast_required = output_idx == 1 && output_info.qnn_data_type == QNN_DATATYPE_INT_64 && is_graph_output;
    std::string cast_input_name = "";
    if (is_cast_required) {
      cast_input_name = transpose_output_name + "_ort_qnn_ep_cast";
      // For the same reason described above, the original output name is now used by this Cast.
      transpose_output_name = cast_input_name;
      // Since additional Cast is added, below Transpose is no longer graph output.
      is_graph_output = false;
    }

    ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(node_unit.Index(),
                                                           transpose_input_names[output_idx],
                                                           transpose_output_name,
                                                           transpose_input_shapes[output_idx],
                                                           transpose_perm,
                                                           output_info.shape,
                                                           output_info.qnn_data_type,
                                                           output_info.quant_param,
                                                           do_op_validation,
                                                           false,
                                                           is_graph_output));

    if (is_cast_required) {
      QnnTensorWrapper cast_output_tensorwrapper(output_name,
                                                 QNN_TENSOR_TYPE_APP_READ,
                                                 output_info.qnn_data_type,
                                                 output_info.quant_param.Copy(),
                                                 std::vector<uint32_t>(output_info.shape));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(cast_output_tensorwrapper)),
                        "Failed to add tensor.");
      ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(cast_input_name,
                                                        QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                        "Cast",
                                                        {cast_input_name},
                                                        {output_name},
                                                        {}),
                        "Failed to add node");
    }
  }

  return Status::OK();
}

void CreateTopKOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<TopKOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
