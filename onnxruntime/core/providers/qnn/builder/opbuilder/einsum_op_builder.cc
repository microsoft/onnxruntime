// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/cpu/tensor/slice_helper.h"

namespace {

// Represented as a tuple of 3 strings <term_1, term_2, result>.
// The equation string is expected to follow the format "term_1,term_2->result"
using Equation = std::tuple<std::string, std::string, std::string>;

/**
 * @brief Parses an equation string into its components if it adheres to the expected format.
 *
 * @param equation_string The input equation string to parse.
 * @return A std::optional containing a tuple of 3 strings (term_1, term_2, result) if the parsing is successful.
 *         Returns std::nullopt if the input string is invalid or does not conform to the expected format.
 */
std::optional<Equation> ParseEquation(std::string_view equation_string) {
  std::string equation(equation_string);
  equation.erase(std::remove(equation.begin(), equation.end(), ' '),
                 equation.end());
  if (equation.empty()) {
    return std::nullopt;
  }
  auto index_arrow = equation.find("->");
  if (index_arrow == std::string::npos) {
    return std::nullopt;
  }
  const std::string lhs = equation.substr(0, index_arrow);
  const std::string result = equation.substr(index_arrow + 2);
  if (lhs.empty() || result.empty()) {
    return std::nullopt;
  }
  auto index_comma = lhs.find(",");
  if (index_comma == std::string::npos) {
    return std::nullopt;
  }
  const std::string term_1 = lhs.substr(0, index_comma);
  const std::string term_2 = lhs.substr(index_comma + 1);
  if (term_1.empty() || term_2.empty()) {
    return std::nullopt;
  }
  if (term_1.size() < 2 || term_2.size() < 2 || result.size() < 2) {
    return std::nullopt;
  }
  if (!std::all_of(term_1.begin(), term_1.end(), [](unsigned char c) { return std::islower(c); })) {
    return std::nullopt;
  }
  if (!std::all_of(term_2.begin(), term_2.end(), [](unsigned char c) { return std::islower(c); })) {
    return std::nullopt;
  }
  if (!std::all_of(result.begin(), result.end(), [](unsigned char c) { return std::islower(c); })) {
    return std::nullopt;
  }
  return std::make_tuple(term_1, term_2, result);
}

bool IsEquationMatMul(const Equation& equation) {
  // MatMul: e.g., "ij,jk->ik"
  const auto& [term_1, term_2, result] = equation;
  const size_t num_dims = term_1.size();
  for (size_t i = 0; i < num_dims; ++i) {
    if (i >= num_dims - 2) {
      continue;
    }
    if (!(term_1[i] == term_2[i] && term_1[i] == result[i])) {
      return false;
    }
  }
  char term_1_m = term_1[num_dims - 2];
  char term_2_k = term_2[num_dims - 2];
  char result_m = result[num_dims - 2];
  char term_1_k = term_1[num_dims - 1];
  char term_2_n = term_2[num_dims - 1];
  char result_n = result[num_dims - 1];
  if (term_1_m != result_m) {
    return false;
  }
  if (term_1_k != term_2_k) {
    return false;
  }
  if (term_2_n != result_n) {
    return false;
  }
  return true;
}

bool IsEquationMatMulTransposeY(const Equation& equation) {
  // MatMul with 2nd input transposed: e.g., "id,jd->ij"
  const auto& [term_1, term_2, result] = equation;
  const size_t num_dims = term_1.size();
  for (size_t i = 0; i < num_dims; ++i) {
    if (i >= num_dims - 2) {
      continue;
    }
    if (!(term_1[i] == term_2[i] && term_1[i] == result[i])) {
      return false;
    }
  }
  char term_1_m = term_1[num_dims - 2];
  char term_2_k = term_2[num_dims - 2];
  char result_m = result[num_dims - 2];
  char term_1_k = term_1[num_dims - 1];
  char term_2_n = term_2[num_dims - 1];
  char result_n = result[num_dims - 1];
  if (term_1_m != result_m) {
    return false;
  }
  if (term_1_k != term_2_n) {
    return false;
  }
  if (term_2_k != result_n) {
    return false;
  }
  return true;
}

bool IsEquationMatMulTransposeAll(const Equation& equation) {
  // MatMul transpose both inputs and output, e.g., "bchq,bkhc->bkhq", "bkhq,bchk->bchq"
  const auto& [term_1, term_2, result] = equation;
  const size_t num_dims = term_1.size();
  if (num_dims != 4) {
    return false;
  }
  if (term_1[0] != term_2[0] || term_1[0] != result[0]) {
    return false;
  }
  char term_1_m = term_1[num_dims - 1];
  char term_1_k = term_1[num_dims - 3];
  char term_2_k = term_2[num_dims - 1];
  char term_2_n = term_2[num_dims - 3];
  char result_m = result[num_dims - 1];
  char result_n = result[num_dims - 3];
  if (term_1_m != result_m) {
    return false;
  }
  if (term_1_k != term_2_k) {
    return false;
  }
  if (term_2_n != result_n) {
    return false;
  }
  return true;
}

bool IsEquationMatMulBroadcastTransposeY(const Equation& equation) {
  // E.g., bhwc,hkc->bhwk
  const auto& [term_1, term_2, result] = equation;
  const size_t term1_dims = term_1.size();
  if (term1_dims != 4) {
    return false;
  }
  const size_t term2_dims = term_2.size();
  if (term2_dims != 3) {
    return false;
  }
  const size_t result_dims = result.size();
  if (result_dims != 4) {
    return false;
  }
  // Check matrix multiplication dimensions
  char term_1_m = term_1[term1_dims - 2];
  char term_1_k = term_1[term1_dims - 1];
  char term_2_k = term_2[term2_dims - 1];
  char term_2_n = term_2[term2_dims - 2];
  char result_m = result[result_dims - 2];
  char result_n = result[result_dims - 1];
  if (term_1_m != result_m) {
    return false;
  }
  if (term_1_k != term_2_k) {
    return false;
  }
  if (term_2_n != result_n) {
    return false;
  }
  // Check batch dimensions
  if (term_1[0] != result[0]) {
    return false;
  }
  if (term_1[1] != result[1]) {
    return false;
  }
  if (term_2[0] != result[1]) {
    return false;
  }
  return true;
}

bool IsEquationReduceSumMulBroadcastX(const Equation& equation) {
  // E.g., bhwc,wkc->bhwk
  const auto& [term_1, term_2, result] = equation;
  if (term_1.size() != 4) {
    return false;
  }
  if (term_2.size() != 3) {
    return false;
  }
  if (result.size() != 4) {
    return false;
  }

  // Check contraction over last axis (c)
  char c1 = term_1[3];
  char c2 = term_2[2];
  if (c1 != c2) {
    return false;
  }

  // Check w axis alignment
  if (term_1[2] != term_2[0]) {
    return false;
  }
  if (term_1[2] != result[2]) {
    return false;
  }

  // Check k axis alignment
  if (term_2[1] != result[3]) {
    return false;
  }

  // Check batch dimensions
  if (term_1[0] != result[0]) {
    return false;
  }
  if (term_1[1] != result[1]) {
    return false;
  }

  return true;
}

/**
 * @brief Sets the parameter tensor names for a MatMul op.
 *
 * @param qnn_model_wrapper Pointer to the QnnModelWrapper instance that manages the QNN model.
 * @param node_unit Reference to the NodeUnit representing the ONNX node for which the parameters are being set.
 * @param transpose_in0 Boolean flag indicating whether the 1st input tensor should be transposed (default: false).
 * @param transpose_in1 Boolean flag indicating whether the 2nd input tensor should be transposed (default: false).
 * @return A vector of strings containing the names of the parameter tensors added to the QNN model.
 */
std::vector<std::string> SetMatMulParamTensorNames(
    onnxruntime::qnn::QnnModelWrapper* qnn_model_wrapper,
    const onnxruntime::NodeUnit& node_unit,
    bool transpose_in0 = false,
    bool transpose_in1 = false) {
  std::vector<std::string> param_tensor_names;
  Qnn_Scalar_t scalar_params[2] = {QNN_SCALAR_INIT, QNN_SCALAR_INIT};
  scalar_params[0].dataType = QNN_DATATYPE_BOOL_8;
  scalar_params[1].dataType = QNN_DATATYPE_BOOL_8;
  scalar_params[0].bool8Value = static_cast<uint8_t>(transpose_in0);
  scalar_params[1].bool8Value = static_cast<uint8_t>(transpose_in1);
  onnxruntime::qnn::QnnParamWrapper transpose_in0_param(
      node_unit.Index(), node_unit.Name(), QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0, scalar_params[0]);
  onnxruntime::qnn::QnnParamWrapper transpose_in1_param(
      node_unit.Index(), node_unit.Name(), QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1, scalar_params[1]);
  param_tensor_names.push_back(transpose_in0_param.GetParamTensorName());
  param_tensor_names.push_back(transpose_in1_param.GetParamTensorName());
  qnn_model_wrapper->AddParamWrapper(std::move(transpose_in0_param));
  qnn_model_wrapper->AddParamWrapper(std::move(transpose_in1_param));
  return param_tensor_names;
}

/**
 * @brief Creates a MatMul operation with transposed inputs and output in a QNN model.
 *
 * @param qnn_model_wrapper Pointer to the QnnModelWrapper instance used to manage the QNN model.
 * @param node_unit The NodeUnit representing the ONNX node to be converted.
 * @param do_op_validation A boolean flag indicating whether to perform operation validation.
 * @return Status indicating success or failure of the operation.
 */
Status CreateMatMulTransposeAll(
    onnxruntime::qnn::QnnModelWrapper* qnn_model_wrapper,
    const onnxruntime::NodeUnit& node_unit,
    std::vector<std::string>&& input_names,
    bool do_op_validation) {
  onnxruntime::qnn::TensorInfo input_info0{}, input_info1{};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper->GetTensorInfo(node_unit.Inputs()[0], input_info0));
  ORT_RETURN_IF_ERROR(qnn_model_wrapper->GetTensorInfo(node_unit.Inputs()[1], input_info1));
  std::vector<uint32_t> input_shape0(input_info0.shape);
  std::vector<uint32_t> input_shape1(input_info1.shape);
  std::swap(input_shape0[1], input_shape0[2]);
  std::swap(input_shape1[1], input_shape1[2]);
  const std::string input_transpos0 = input_names[0] + "_t0";
  const std::string input_transpos1 = input_names[1] + "_t1";
  const std::vector<uint32_t> transpose_perm{0, 2, 1, 3};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper->AddTransposeNode(
      /*node_index=*/node_unit.Index(),
      /*input_name=*/input_names[0],
      /*output_name=*/input_transpos0,
      /*input_shape=*/input_info0.shape,
      /*transpose_perm=*/transpose_perm,
      /*output_shape=*/input_shape0,
      /*qnn_data_type=*/input_info0.qnn_data_type,
      /*quantize_param=*/input_info0.quant_param.Copy(),
      /*do_op_validation=*/do_op_validation,
      /*is_for_input=*/qnn_model_wrapper->IsGraphInput(input_names[0])));
  ORT_RETURN_IF_ERROR(qnn_model_wrapper->AddTransposeNode(
      /*node_index=*/node_unit.Index(),
      /*input_name=*/input_names[1],
      /*output_name=*/input_transpos1,
      /*input_shape=*/input_info1.shape,
      /*transpose_perm=*/transpose_perm,
      /*output_shape=*/input_shape1,
      /*qnn_data_type=*/input_info1.qnn_data_type,
      /*quantize_param=*/input_info1.quant_param.Copy(),
      /*do_op_validation=*/do_op_validation,
      /*is_for_input=*/qnn_model_wrapper->IsGraphInput(input_names[1])));
  onnxruntime::qnn::TensorInfo matmul_output_info{};
  const auto& output = node_unit.Outputs()[0];
  ORT_RETURN_IF_ERROR(qnn_model_wrapper->GetTensorInfo(output, matmul_output_info));
  const std::string matmul_output_name = onnxruntime::qnn::utils::GetNodeName(node_unit) + "_matmul";
  std::vector<uint32_t> matmul_output_shape(matmul_output_info.shape);
  std::swap(matmul_output_shape[1], matmul_output_shape[2]);
  onnxruntime::qnn::QnnTensorWrapper matmul_output_wrapper(
      matmul_output_name, QNN_TENSOR_TYPE_NATIVE, matmul_output_info.qnn_data_type,
      matmul_output_info.quant_param.Copy(), std::vector<uint32_t>(matmul_output_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper->AddTensorWrapper(std::move(matmul_output_wrapper)),
                    node_unit.OpType() + " failed to add tensor.");
  std::vector<std::string> param_tensor_names = SetMatMulParamTensorNames(
      qnn_model_wrapper, node_unit, /*transpose_in0=*/false, /*transpose_in1=*/false);
  ORT_RETURN_IF_NOT(qnn_model_wrapper->CreateQnnNode(/*qnn_node_name=*/onnxruntime::qnn::utils::GetNodeName(node_unit),
                                                     /*package_name=*/QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                     /*qnn_node_type=*/QNN_OP_MAT_MUL,
                                                     /*input_names=*/{input_transpos1, input_transpos0},
                                                     /*output_names=*/{matmul_output_name},
                                                     /*param_tensor_names=*/std::move(param_tensor_names),
                                                     /*do_op_validation=*/do_op_validation),
                    node_unit.OpType() + " failed to add node.");
  std::vector<uint32_t> transpose_output_shape(matmul_output_info.shape);
  ORT_RETURN_IF_ERROR(qnn_model_wrapper->AddTransposeNode(
      /*node_index=*/node_unit.Index(),
      /*input_name=*/matmul_output_name,
      /*output_name=*/output.node_arg.Name(),
      /*input_shape=*/std::move(matmul_output_shape),
      /*transpose_perm=*/transpose_perm,
      /*output_shape=*/matmul_output_info.shape,
      /*tensor_data_type=*/matmul_output_info.qnn_data_type,
      /*quantize_param=*/matmul_output_info.quant_param.Copy(),
      /*do_op_validation=*/do_op_validation,
      /*is_for_input=*/qnn_model_wrapper->IsGraphInput(output.node_arg.Name()),
      /*is_for_output=*/qnn_model_wrapper->IsGraphOutput(output.node_arg.Name())));
  return Status::OK();
}

/**
 * @brief Creates a ReduceSum, Multiply on broadcasted input X and original input Y.
 *
 * @param qnn_model_wrapper Pointer to the QnnModelWrapper instance used to manage the QNN model.
 * @param node_unit The NodeUnit representing the ONNX node to be converted.
 * @param do_op_validation A boolean flag indicating whether to perform operation validation.
 * @return Status indicating success or failure of the operation.
 */
Status CreateReduceSumMulBroadcastX(
    onnxruntime::qnn::QnnModelWrapper* qnn_model_wrapper,
    const onnxruntime::NodeUnit& node_unit,
    std::vector<std::string>&& input_names,
    bool do_op_validation) {
  // Reshape in0 to shape (b, h, w, 1, c) to expand dimension before the contraction axis 'c'.
  // Allowing broadcast with in1 for multiplication, aligning the contraction axis for reduce.
  onnxruntime::qnn::TensorInfo tensor_info_in0{}, tensor_info_in1{}, tensor_info_out{};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper->GetTensorInfo(node_unit.Inputs()[0], tensor_info_in0));
  ORT_RETURN_IF_ERROR(qnn_model_wrapper->GetTensorInfo(node_unit.Inputs()[1], tensor_info_in1));
  ORT_RETURN_IF_ERROR(qnn_model_wrapper->GetTensorInfo(node_unit.Outputs()[0], tensor_info_out));
  const std::vector<uint32_t>& shape_in0 = tensor_info_in0.shape;
  const std::vector<uint32_t>& shape_in1 = tensor_info_in1.shape;
  ORT_RETURN_IF_NOT(shape_in0.size() == 4, "CreateReduceSumMulBroadcastX expects input 0 to be rank 4");
  ORT_RETURN_IF_NOT(shape_in1.size() == 3, "CreateReduceSumMulBroadcastX expects input 1 to be rank 3");
  const std::vector<uint32_t> new_shape_in0{shape_in0[0], shape_in0[1], shape_in0[2], 1, shape_in0[3]};
  const std::string reshape_out_name = input_names[0] + "_reshaped";
  ORT_RETURN_IF_ERROR(qnn_model_wrapper->AddReshapeNode(
      /*input_name=*/input_names[0],
      /*output_name=*/reshape_out_name,
      /*input_shape=*/shape_in0,
      /*output_shape=*/new_shape_in0,
      /*tensor_data_type=*/tensor_info_in0.qnn_data_type,
      /*quantize_param=*/tensor_info_in0.quant_param.Copy(),
      /*do_op_validation=*/do_op_validation,
      /*is_for_input=*/qnn_model_wrapper->IsGraphInput(input_names[0])));

  // Multiply: reshaped in0 * in1
  // The output shape of the multiplication is determined by broadcasting the reshaped in0 of
  // (b, h, w, 1, c) and in1 (w, k, c) along the matching axes, resulting in (b, h, w, k, c).
  const std::string mul_out_name = onnxruntime::qnn::utils::GetNodeName(node_unit) + "_mul";
  std::vector<uint32_t> shape_out_mul{new_shape_in0[0], new_shape_in0[1], new_shape_in0[2], shape_in1[1], new_shape_in0[4]};
  onnxruntime::qnn::QnnTensorWrapper tensor_wrapper_mul(mul_out_name,
                                                        QNN_TENSOR_TYPE_NATIVE,
                                                        tensor_info_in0.qnn_data_type,
                                                        tensor_info_in0.quant_param.Copy(),
                                                        std::move(shape_out_mul));
  ORT_RETURN_IF_NOT(qnn_model_wrapper->AddTensorWrapper(std::move(tensor_wrapper_mul)),
                    "CreateReduceSumMulBroadcastX: failed to AddTensorWrapper");
  ORT_RETURN_IF_NOT(qnn_model_wrapper->CreateQnnNode(
                        /*qnn_node_name=*/mul_out_name,
                        /*package_name=*/QNN_OP_PACKAGE_NAME_QTI_AISW,
                        /*qnn_node_type=*/QNN_OP_ELEMENT_WISE_MULTIPLY,
                        /*input_names=*/{reshape_out_name, input_names[1]},
                        /*output_names=*/{mul_out_name},
                        /*param_tensor_names=*/{},
                        /*do_op_validation=*/do_op_validation),
                    "CreateReduceSumMulBroadcastX: failed to create Mul node");

  std::vector<std::string> param_tensor_names{};

  // ReduceSum on last axes={4}, keep_dims=False
  // Axis '4' corresponds to the last dimension ('c') of the reshaped tensor (b, h, w, k, c),
  // which is the contraction axis for reduce sum op in the einsum equation (bhwc,wkc->bhwk).
  std::vector<uint32_t> axes_shape{SafeInt<uint32_t>(1)};
  std::vector<uint32_t> axes_value{SafeInt<uint32_t>(4)};
  onnxruntime::qnn::QnnParamWrapper param_axes(node_unit.Index(),
                                               node_unit.Name(),
                                               QNN_OP_REDUCE_SUM_PARAM_AXES,
                                               std::move(axes_shape),
                                               std::move(axes_value));
  param_tensor_names.push_back(param_axes.GetParamTensorName());
  ORT_RETURN_IF_NOT(qnn_model_wrapper->AddParamWrapper(std::move(param_axes)),
                    "CreateReduceSumMulBroadcastX: failed to add param axes");

  Qnn_Scalar_t keep_dims_scalar = QNN_SCALAR_INIT;
  keep_dims_scalar.dataType = QNN_DATATYPE_BOOL_8;
  keep_dims_scalar.bool8Value = SafeInt<uint8_t>(0);
  onnxruntime::qnn::QnnParamWrapper param_keep_dims(node_unit.Index(),
                                                    node_unit.Name(),
                                                    QNN_OP_REDUCE_SUM_PARAM_KEEP_DIMS,
                                                    keep_dims_scalar);
  param_tensor_names.push_back(param_keep_dims.GetParamTensorName());
  ORT_RETURN_IF_NOT(qnn_model_wrapper->AddParamWrapper(std::move(param_keep_dims)),
                    "CreateReduceSumMulBroadcastX: failed to add param keep_dims");

  const std::string out_name = node_unit.Outputs()[0].node_arg.Name();
  Qnn_TensorType_t out_tensor_type = qnn_model_wrapper->IsGraphOutput(out_name) ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
  onnxruntime::qnn::QnnTensorWrapper tensor_wrapper_out(out_name,
                                                        out_tensor_type,
                                                        tensor_info_out.qnn_data_type,
                                                        tensor_info_out.quant_param.Copy(),
                                                        std::move(tensor_info_out.shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper->AddTensorWrapper(std::move(tensor_wrapper_out)),
                    "CreateReduceSumMulBroadcastX: failed to AddTensorWrapper");

  ORT_RETURN_IF_NOT(qnn_model_wrapper->CreateQnnNode(
                        /*qnn_node_name=*/out_name,
                        /*package_name=*/QNN_OP_PACKAGE_NAME_QTI_AISW,
                        /*qnn_node_type=*/QNN_OP_REDUCE_SUM,
                        /*input_names=*/{mul_out_name},
                        /*output_names=*/{out_name},
                        /*param_tensor_names=*/std::move(param_tensor_names),
                        /*do_op_validation=*/do_op_validation),
                    "CreateReduceSumMulBroadcastX: failed to create ReduceSum node");

  return Status::OK();
}

}  // namespace

namespace onnxruntime {
namespace qnn {

class EinsumOpBuilder : public BaseOpBuilder {
 public:
  EinsumOpBuilder() : BaseOpBuilder("EinsumOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(EinsumOpBuilder);

  Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

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

  Status OverrideOutputQuantParam(QnnModelWrapper& qnn_model_wrapper,
                                  const NodeUnit& node_unit,
                                  const logging::Logger& logger,
                                  const std::vector<std::string>& input_names,
                                  size_t output_index,
                                  Qnn_DataType_t qnn_data_type,
                                  QnnQuantParamsWrapper& quant_param) const override ORT_MUST_USE_RESULT;
};

Status EinsumOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                      const NodeUnit& node_unit,
                                      const logging::Logger& logger) const {
  if (node_unit.Inputs().size() < 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, node_unit.OpType() + " requires at least 2 inputs.");
  }
  NodeAttrHelper node_helper{node_unit};
  const std::string equation = node_helper.Get("equation", std::string(""));
  std::optional<Equation> parsed_equation = ParseEquation(equation);
  if (!parsed_equation.has_value()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, node_unit.OpType() + " unsupported equation: " + equation);
  }
  if (!IsEquationMatMul(parsed_equation.value()) &&
      !IsEquationMatMulTransposeY(parsed_equation.value()) &&
      !IsEquationMatMulBroadcastTransposeY(parsed_equation.value()) &&
      !IsEquationMatMulTransposeAll(parsed_equation.value()) &&
      !IsEquationReduceSumMulBroadcastX(parsed_equation.value())) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, node_unit.OpType() + " unsupported equation: " + equation);
  }
  if (IsEquationReduceSumMulBroadcastX(parsed_equation.value())) {
    if (IsGpuBackend(qnn_model_wrapper.GetQnnBackendType())) {
      // QAIRT 3.36.1: Failed to validate on GPU.
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, node_unit.OpType() + " unsupported equation: " + equation + " on backend GPU");
    }
    if (node_unit.Inputs()[0].quant_param.has_value()) {
      // QAIRT 3.36.1: Failed to finalize QNN graph 1002.
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, node_unit.OpType() + " unsupported equation: " + equation + " for quantized inputs");
    }
  }
  return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, true);
}

Status EinsumOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                      const NodeUnit& node_unit,
                                      const logging::Logger& logger,
                                      std::vector<std::string>& input_names,
                                      bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);
  const auto& inputs = node_unit.Inputs();
  ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[0], logger, input_names));
  ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[1], logger, input_names));
  return Status::OK();
}

Status EinsumOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                    const NodeUnit& node_unit,
                                                    std::vector<std::string>&& input_names,
                                                    const logging::Logger& logger,
                                                    bool do_op_validation) const {
  NodeAttrHelper node_helper(node_unit);
  const std::string equation = node_helper.Get("equation", std::string(""));
  std::optional<Equation> parsed_equation = ParseEquation(equation);
  if (IsEquationMatMul(parsed_equation.value())) {
    std::vector<std::string> param_tensor_names = SetMatMulParamTensorNames(
        &qnn_model_wrapper, node_unit, /*transpose_in0=*/false, /*transpose_in1=*/false);
    ORT_RETURN_IF_ERROR(ProcessOutputs(/*qnn_model_wrapper=*/qnn_model_wrapper,
                                       /*node_unit=*/node_unit,
                                       /*input_names=*/std::move(input_names),
                                       /*param_tensor_names=*/std::move(param_tensor_names),
                                       /*logger=*/logger,
                                       /*do_op_validation=*/do_op_validation,
                                       /*qnn_op_type=*/QNN_OP_MAT_MUL));
  } else if (IsEquationMatMulTransposeY(parsed_equation.value()) ||
             IsEquationMatMulBroadcastTransposeY(parsed_equation.value())) {
    std::vector<std::string> param_tensor_names = SetMatMulParamTensorNames(
        &qnn_model_wrapper, node_unit, /*transpose_in0=*/false, /*transpose_in1=*/true);
    ORT_RETURN_IF_ERROR(ProcessOutputs(/*qnn_model_wrapper=*/qnn_model_wrapper,
                                       /*node_unit=*/node_unit,
                                       /*input_names=*/std::move(input_names),
                                       /*param_tensor_names=*/std::move(param_tensor_names),
                                       /*logger=*/logger,
                                       /*do_op_validation=*/do_op_validation,
                                       /*qnn_op_type=*/QNN_OP_MAT_MUL));
  } else if (IsEquationMatMulTransposeAll(parsed_equation.value())) {
    ORT_RETURN_IF_ERROR(CreateMatMulTransposeAll(/*qnn_model_wrapper=*/&qnn_model_wrapper,
                                                 /*node_unit=*/node_unit,
                                                 /*input_names=*/std::move(input_names),
                                                 /*do_op_validation=*/do_op_validation));
  } else if (IsEquationReduceSumMulBroadcastX(parsed_equation.value())) {
    ORT_RETURN_IF_ERROR(CreateReduceSumMulBroadcastX(/*qnn_model_wrapper=*/&qnn_model_wrapper,
                                                     /*node_unit=*/node_unit,
                                                     /*input_names=*/std::move(input_names),
                                                     /*do_op_validation=*/do_op_validation));
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, node_unit.OpType() + " unsupported equation: " + equation);
  }
  return Status::OK();
}

Status EinsumOpBuilder::OverrideOutputQuantParam(QnnModelWrapper& qnn_model_wrapper,
                                                 const NodeUnit& node_unit,
                                                 const logging::Logger& logger,
                                                 const std::vector<std::string>& input_names,
                                                 size_t output_index,
                                                 Qnn_DataType_t qnn_data_type,
                                                 QnnQuantParamsWrapper& quant_param) const {
  if (!quant_param.IsPerTensor()) {
    return Status::OK();
  }

  // Force the operator output to use the same quantization parameters as the input if nearly equal.
  // This helps the HTP backend employ certain optimizations.
  return SetOutputQParamEqualToInputIfNearlyEqual(qnn_model_wrapper, node_unit, logger, input_names,
                                                  0 /*input_index*/, output_index, qnn_data_type, quant_param);
}

void CreateEinsumOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<EinsumOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
