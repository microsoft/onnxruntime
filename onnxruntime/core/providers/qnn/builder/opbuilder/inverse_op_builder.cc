// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <unordered_set>
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/cpu/tensor/slice_helper.h"

namespace onnxruntime {
namespace qnn {
class InverseOpBuilder : public BaseOpBuilder {
 public:
  InverseOpBuilder() : BaseOpBuilder("InverseOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(InverseOpBuilder);

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
};

Status InverseOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                       const NodeUnit& node_unit,
                                       const logging::Logger& logger,
                                       std::vector<std::string>& input_names,
                                       bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();

  if (do_op_validation) {
    TensorInfo input_info = {};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[0], input_info));
    Qnn_DataType_t target_tensor_type = input_info.qnn_data_type;

    if (QNN_DATATYPE_FLOAT_32 != target_tensor_type) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN Inverse Op support only float input tensor.");
    }

    std::vector<uint32_t> input_shape;
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(node_unit.Inputs()[0].node_arg, input_shape), "Cannot get shape of input 0.");

    if (input_shape.size() < 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN expects Inverse Op has rank >= 2.");
    }

    if (2 != input_shape[input_shape.size() - 2] || 2 != input_shape[input_shape.size() - 1]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN expects 2x2 Inverse Op: [..., 2, 2].");
    }
  }
  const auto input_count = GetInputCountQnnRequired(node_unit);
  for (size_t input_i = 0; input_i < input_count; ++input_i) {
    ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[input_i], logger, input_names));
  }

  return Status::OK();
}

Status InverseOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                     const NodeUnit& node_unit,
                                                     std::vector<std::string>&& input_names,
                                                     const logging::Logger& logger,
                                                     bool do_op_validation) const {
  /*
  Inverse( x = [..., 2, 2] ) is lowered to:
  1. Flattern(x, [K, 4]), K = prod(x.shape)/4
  2. Slice x to a = [..., 0], b = [..., 1], c = [..., 2], d = [..., 3]
  3. Calculate Det(x) = a*d - b*c. Shape = [K, 1]
  4. Calculate Adj(x) = [d, -b, -c, a]. Shape = [K, 4]
  5. calculate Inv(x) = Adj(x)/Det(x) (broadcast). Shape = [K, 4]
  6. Reshape back to original shape.
  */
  ORT_UNUSED_PARAMETER(logger);
  TensorInfo input_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[0], input_info));
  TensorInfo output_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[0], output_info));

  const std::string& input_name = input_names[0];
  // Hard requirement to known input tensor shape.
  std::vector<uint32_t> input_shape = input_info.shape;
  const uint32_t& total_elements = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<uint32_t>());

  constexpr uint32_t kFlatDim = 4;
  const std::vector<uint32_t>& reshaped_shape{total_elements / kFlatDim, kFlatDim};

  // Reshape input to [-1, 4]
  const std::string& reshape_output_name = utils::GetUniqueName(node_unit, "_reshape_output");
  QnnTensorWrapper reshape_output(reshape_output_name,
                                  QNN_TENSOR_TYPE_NATIVE,
                                  input_info.qnn_data_type,
                                  input_info.quant_param.Copy(),
                                  std::vector<uint32_t>(reshaped_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(reshape_output)), "Failed to add tensor.");

  bool is_graph_input = qnn_model_wrapper.IsGraphInput(input_name);
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(input_name,
                                                       reshape_output_name,
                                                       input_shape,
                                                       std::vector<uint32_t>(reshaped_shape),
                                                       input_info.qnn_data_type,
                                                       input_info.quant_param.Copy(),
                                                       do_op_validation,
                                                       is_graph_input));

  const uint32_t& channel_num = total_elements / kFlatDim;
  const std::vector<uint32_t>& sliced_shape{channel_num, 1};

  // Slice 2x2 matrix to a, b, c, d.
  constexpr uint32_t kReshapedRank = 2;
  constexpr uint32_t kSliceParamNum = 3;

  const std::vector<std::string>& sliced_output_names = {
      utils::GetUniqueName(node_unit, "_slice_00_output"),
      utils::GetUniqueName(node_unit, "_slice_01_output"),
      utils::GetUniqueName(node_unit, "_slice_10_output"),
      utils::GetUniqueName(node_unit, "_slice_11_output"),
  };
  for (uint32_t i = 0; i < 4; ++i) {
    const std::string& param_name = utils::GetUniqueName(node_unit, "slice_param");
    QnnParamWrapper ranges_paramwrapper(node_unit.Index(),
                                        param_name,
                                        QNN_OP_STRIDED_SLICE_PARAM_RANGES,
                                        {kReshapedRank, kSliceParamNum},  // [rank, len(start, end, step)]
                                        {0, 0, 1, i, i + 1, 1},           // [ignore, ignore, step_0, start_1, end_1, step_1]
                                        true);
    std::vector<std::string>&& slice_param_tensor_name{ranges_paramwrapper.GetParamTensorName()};
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(ranges_paramwrapper)), "Failed to add param.");

    const std::string& slice_name = utils::GetUniqueName(node_unit, "_slice");
    // begin_mask
    uint32_t begin_mask = 0b01U;  // ignore dim = 0
    ORT_RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper,
                                               node_unit.Index(),
                                               slice_name,
                                               begin_mask,
                                               QNN_OP_STRIDED_SLICE_PARAM_BEGIN_MASK,
                                               slice_param_tensor_name));

    // end_mask
    uint32_t end_mask = 0b01U;  // ignore dim = 0
    ORT_RETURN_IF_ERROR(AddQnnScalar<uint32_t>(
        qnn_model_wrapper,
        node_unit.Index(),
        slice_name,
        end_mask,
        QNN_OP_STRIDED_SLICE_PARAM_END_MASK,
        slice_param_tensor_name));

    QnnTensorWrapper slice_output(sliced_output_names[i],
                                  QNN_TENSOR_TYPE_NATIVE,
                                  input_info.qnn_data_type,
                                  QnnQuantParamsWrapper(),
                                  std::vector<uint32_t>(sliced_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(slice_output)), "Failed to add tensor.");

    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(slice_name,
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_STRIDED_SLICE,
                                                      {reshape_output_name},
                                                      {sliced_output_names[i]},
                                                      std::move(slice_param_tensor_name),
                                                      do_op_validation),
                      "Failed to add Inverse - slice node.");
  }

  // det([[a, b], [c, d]]) = ad - bc
  const std::string& det_mul_0_name = utils::GetUniqueName(node_unit, "_det_mul_0");
  const std::string& det_mul_0_output_name = utils::GetUniqueName(node_unit, "_det_mul_0_output");
  QnnTensorWrapper det_mul_0_output(det_mul_0_output_name,
                                    QNN_TENSOR_TYPE_NATIVE,
                                    input_info.qnn_data_type,
                                    QnnQuantParamsWrapper(),
                                    std::vector<uint32_t>(sliced_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(det_mul_0_output)),
                    "Failed to add Inverse - Det - Mul(a, d) output tensor.");

  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(det_mul_0_name,
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_ELEMENT_WISE_MULTIPLY,
                                                    {sliced_output_names[0], sliced_output_names[3]},
                                                    {det_mul_0_output_name},
                                                    {},
                                                    do_op_validation),
                    "Failed to add Inverse - Det - Mul(a, d) node.");

  const std::string& det_mul_1_name = utils::GetUniqueName(node_unit, "_det_mul_1");
  const std::string& det_mul_1_output_name = utils::GetUniqueName(node_unit, "_det_mul_1_output");
  QnnTensorWrapper det_mul_1_output(det_mul_1_output_name,
                                    QNN_TENSOR_TYPE_NATIVE,
                                    input_info.qnn_data_type,
                                    QnnQuantParamsWrapper(),
                                    std::vector<uint32_t>(sliced_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(det_mul_1_output)),
                    "Failed to add Inverse - Det - Mul(b, c) output tensor.");

  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(det_mul_1_name,
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_ELEMENT_WISE_MULTIPLY,
                                                    {sliced_output_names[1], sliced_output_names[2]},
                                                    {det_mul_1_output_name},
                                                    {},
                                                    do_op_validation),
                    "Failed to add Inverse - Det - Mul(b, c) node.");

  const std::string& det_sub_name = utils::GetUniqueName(node_unit, "_det_sub");
  const std::string& det_sub_output_name = utils::GetUniqueName(node_unit, "_det_sub_output");
  QnnTensorWrapper det_sub_output(det_sub_output_name,
                                  QNN_TENSOR_TYPE_NATIVE,
                                  input_info.qnn_data_type,
                                  QnnQuantParamsWrapper(),
                                  std::vector<uint32_t>(sliced_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(det_sub_output)),
                    "Failed to add Inverse - Det - Sub(ad, bc) output tensor.");

  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(det_sub_name,
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_ELEMENT_WISE_SUBTRACT,
                                                    {det_mul_0_output_name, det_mul_1_output_name},
                                                    {det_sub_output_name},
                                                    {},
                                                    do_op_validation),
                    "Failed to add Inverse - Det - Sub(ad, bc) node.");

  // adj([[a, b], [c, d]]) = [d, b, a, c] * [1, -1, -1, 1]
  const std::string& adj_cat_name = utils::GetUniqueName(node_unit, "_adj_concat");
  const std::string& adj_cat_output_name = utils::GetUniqueName(node_unit, "_adj_concat_output");
  std::vector<std::string> concat_param_name;  // Will be modified in AddQnnScalar
  ORT_RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), adj_cat_output_name, static_cast<uint32_t>(1), QNN_OP_CONCAT_PARAM_AXIS, concat_param_name));

  QnnTensorWrapper adj_cat_output(adj_cat_output_name,
                                  QNN_TENSOR_TYPE_NATIVE,
                                  input_info.qnn_data_type,
                                  QnnQuantParamsWrapper(),
                                  std::vector<uint32_t>(reshaped_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(adj_cat_output)),
                    "Failed to add Inverse - Adj - cat(d, b, c, a) output tensor.");

  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(adj_cat_name,
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_CONCAT,
                                                    {sliced_output_names[3], sliced_output_names[1], sliced_output_names[2], sliced_output_names[0]},
                                                    {adj_cat_output_name},
                                                    std::move(concat_param_name),
                                                    do_op_validation),
                    "Failed to add Inverse - Adj - cat(d, b, c, a) node.");

  // Create [1, -1, -1, 1] tensor for adj.
  static constexpr std::array<float, 4> adj_mul_tensor_value{1.f, -1.f, -1.f, 1.f};
  std::vector<uint8_t> adj_mul_bytes(sizeof(adj_mul_tensor_value));
  std::memcpy(adj_mul_bytes.data(), adj_mul_tensor_value.data(), adj_mul_bytes.size());

  const std::string& adj_mul_tensor_name = utils::GetUniqueName(node_unit, "_adj_mul_tensor");
  QnnTensorWrapper adj_mul_tensor(adj_mul_tensor_name,
                                  QNN_TENSOR_TYPE_STATIC,
                                  input_info.qnn_data_type,
                                  QnnQuantParamsWrapper(),
                                  std::vector<uint32_t>({kFlatDim}),
                                  std::move(adj_mul_bytes));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(adj_mul_tensor)), "Failed to add tensor.");

  const std::string& adj_mul_name = utils::GetUniqueName(node_unit, "_adj_mul");
  const std::string& adj_mul_output_name = utils::GetUniqueName(node_unit, "_adj_mul_output");
  QnnTensorWrapper adj_mul_output(adj_mul_output_name,
                                  QNN_TENSOR_TYPE_NATIVE,
                                  input_info.qnn_data_type,
                                  QnnQuantParamsWrapper(),
                                  std::vector<uint32_t>(reshaped_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(adj_mul_output)),
                    "Failed to add Inverse - Adj - Mul([d, b, a, c], [1, -1, -1, 1]) output tensor.");

  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(adj_mul_name,
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_ELEMENT_WISE_MULTIPLY,
                                                    {adj_cat_output_name, adj_mul_tensor_name},
                                                    {adj_mul_output_name},
                                                    {},
                                                    do_op_validation),
                    "Failed to add Inverse - Adj - Mul([d, b, a, c], [1, -1, -1, 1]) node.");

  // inverse = adj / det
  const std::string& inverse_div_name = utils::GetUniqueName(node_unit, "_inv_div");
  const std::string& inverse_div_output_name = utils::GetUniqueName(node_unit, "_inv_div_output");
  QnnTensorWrapper inverse_div_tensor(inverse_div_output_name,
                                      QNN_TENSOR_TYPE_NATIVE,
                                      input_info.qnn_data_type,
                                      QnnQuantParamsWrapper(),
                                      std::vector<uint32_t>(reshaped_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(inverse_div_tensor)),
                    "Failed to add Inverse - Adj / Det output tensor.");

  // Expect QNN element-wise ops to broadcast unit dimensions.
  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(inverse_div_name,
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_ELEMENT_WISE_DIVIDE,
                                                    {adj_mul_output_name, det_sub_output_name},
                                                    {inverse_div_output_name},
                                                    {},
                                                    do_op_validation),
                    "Failed to add Inverse - Adj / Det node.");

  // Reshape back to original shape input_info.shape
  const std::string& org_output_name = node_unit.Outputs()[0].node_arg.Name();
  const bool& is_graph_output = qnn_model_wrapper.IsGraphOutput(org_output_name);

  std::vector<uint32_t> output_shape = output_info.shape;
  Qnn_TensorType_t op_output_tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

  QnnTensorWrapper inverse_output(org_output_name,
                                  op_output_tensor_type,
                                  output_info.qnn_data_type,
                                  output_info.quant_param.Copy(),
                                  std::vector<uint32_t>(output_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(inverse_output)), "Failed to add tensor.");

  ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(inverse_div_output_name,
                                                       org_output_name,
                                                       std::vector<uint32_t>(reshaped_shape),
                                                       std::vector<uint32_t>(output_shape),
                                                       output_info.qnn_data_type,
                                                       output_info.quant_param.Copy(),
                                                       do_op_validation,
                                                       is_graph_output));

  return Status::OK();
}

void CreateInverseOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<InverseOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime