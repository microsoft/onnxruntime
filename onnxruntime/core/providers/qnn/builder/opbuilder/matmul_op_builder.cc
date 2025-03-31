// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

/**
 * An ONNX MatMul can be translated to either a QNN MatMul or a QNN FullyConnected.
 * ONNX's MatMul suports inputs of rank 1, but neither QNN's MatMul nor FullyConnected support two rank 1 inputs.
 * So, we need to add Reshape Ops if necessary.
 * In two cases, FullyConnected (input_1's shape is [n, k]) is used instead of MatMul without extra Transpose Op:
 * 1. input_1 is a rank 2 initializer.
 * 2. input_1 is a rank 1 tensor.
 */
class MatMulOpBuilder : public BaseOpBuilder {
 public:
  MatMulOpBuilder() : BaseOpBuilder("MatMulOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MatMulOpBuilder);

 protected:
  Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit, const logging::Logger& logger,
                       std::vector<std::string>& input_names, bool do_op_validation) const override ORT_MUST_USE_RESULT;

  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names, const logging::Logger& logger,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;

 private:
  Status ProcessInputsForQnnMatMul(QnnModelWrapper& qnn_model_wrapper,
                                   const NodeUnit& node_unit,
                                   const TensorInfo& input_info_0,
                                   const TensorInfo& input_info_1,
                                   const logging::Logger& logger,
                                   std::vector<std::string>& input_names,
                                   bool do_op_validation) const ORT_MUST_USE_RESULT;
  Status ProcessInputsForQnnFullyConnected(QnnModelWrapper& qnn_model_wrapper,
                                           const NodeUnit& node_unit,
                                           const TensorInfo& input_info_0,
                                           const TensorInfo& input_info_1,
                                           const logging::Logger& logger,
                                           std::vector<std::string>& input_names,
                                           bool do_op_validation) const ORT_MUST_USE_RESULT;
};

namespace {

// Inserts a QNN Convert operator to convert from one quantization type (e.g., uint16) to another (e.g., uint8).
Status InsertConvertOp(QnnModelWrapper& qnn_model_wrapper,
                       const std::string& convert_input_name,
                       const std::string& convert_output_name,
                       Qnn_DataType_t input_qnn_data_type,
                       Qnn_DataType_t output_qnn_data_type,
                       int32_t input_offset,
                       float input_scale,
                       const std::vector<uint32_t>& output_shape,
                       bool do_op_validation) {
  // Assume input is already handled.
  float qmin = 0.0f;
  float qmax = 255.0f;
  ORT_RETURN_IF_ERROR(qnn::utils::GetQminQmax(input_qnn_data_type, qmin, qmax));
  double value_min = qnn::utils::Dequantize(input_offset, input_scale, qmin);
  double value_max = qnn::utils::Dequantize(input_offset, input_scale, qmax);
  float scale = 0.0f;
  int32_t offset = 0;
  ORT_RETURN_IF_ERROR(qnn::utils::GetQuantParams(static_cast<float>(value_min),
                                                 static_cast<float>(value_max),
                                                 output_qnn_data_type,
                                                 scale,
                                                 offset));

  std::vector<uint32_t> output_shape_copy = output_shape;
  QnnTensorWrapper convert_output_tensorwrapper(convert_output_name,
                                                QNN_TENSOR_TYPE_NATIVE,
                                                output_qnn_data_type,
                                                QnnQuantParamsWrapper(scale, offset),
                                                std::move(output_shape_copy));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(convert_output_tensorwrapper)), "Failed to add tensor.");

  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(convert_output_name,
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    "Convert",
                                                    {convert_input_name},
                                                    {convert_output_name},
                                                    {},
                                                    do_op_validation),
                    "Failed to add node.");
  return Status::OK();
}

inline bool IsQuant16bit(Qnn_DataType_t qnn_data_type) {
  return qnn_data_type == QNN_DATATYPE_UFIXED_POINT_16 || qnn_data_type == QNN_DATATYPE_SFIXED_POINT_16;
}

Status CheckInputs(const QnnModelWrapper& qnn_model_wrapper, const NodeUnitIODef& input_def_0,
                   const NodeUnitIODef& input_def_1, TensorInfo& input_info_0, TensorInfo& input_info_1,
                   bool& use_fully_connected) {
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(input_def_0, input_info_0));
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(input_def_1, input_info_1));

#if QNN_API_VERSION_MAJOR >= 2 && QNN_API_VERSION_MINOR <= 20
  // Validation crashes if use QNN FullyConnected in QNN SDK versions 2.26 - 2.27
  // Just use QNN MatMul for these older QNN SDK versions.
  use_fully_connected = false;
#else
  // Use FullyConnected if 2nd input is a rank 2 initializer or a rank 1 tensor.
  // FullyConnected cannot pass the Op validation if keep_dims is true, so if input_0 is per-channel quantized tensor
  // with rank > 2, it's not easy to set the quantization parameters for the output reshaped rank 2 tensor.
  // In this case, we will not use FullyConnected.
  use_fully_connected =
      (input_info_1.shape.size() == 2 && input_info_1.is_initializer) || input_info_1.shape.size() == 1;
  use_fully_connected =
      use_fully_connected && !(input_info_0.quant_param.IsPerChannel() && input_info_0.shape.size() > 2);
  // Don't use FullyConnected if both inputs are dynamic and uint16 (quantized)
  use_fully_connected = use_fully_connected && !(IsQuant16bit(input_info_0.qnn_data_type) &&
                                                 !input_info_0.is_initializer &&
                                                 IsQuant16bit(input_info_1.qnn_data_type) &&
                                                 !input_info_1.is_initializer);
#endif
  return Status::OK();
}

// Process input[0] for ONNX MatMul that can be translated to either a QNN MatMul or a QNN FullyConnected.
Status ProcessInput0(QnnModelWrapper& qnn_model_wrapper,
                     const TensorInfo& input_0_info,
                     const std::string& original_input_0_name,
                     std::vector<std::string>& input_names,
                     const logging::Logger& logger,
                     bool do_op_validation) {
  bool reshape_input_0 = input_0_info.shape.size() == 1;
  std::string actual_input_0_name = original_input_0_name;

  if (reshape_input_0) {
    actual_input_0_name = original_input_0_name + "_ort_qnn_ep_reshape";
    std::vector<uint32_t> shape_2d{1, input_0_info.shape[0]};
    QnnQuantParamsWrapper quant_param_2d = input_0_info.quant_param.Copy();
    ORT_RETURN_IF_ERROR(quant_param_2d.HandleUnsqueeze<uint32_t>(input_0_info.shape, shape_2d));

    // If input_0 is initializer, unpack it and add the tensor with new quantization parameter and shape.
    // Otherwise, add a Reshape node.
    if (input_0_info.is_initializer) {
      std::vector<uint8_t> unpacked_tensor;
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*input_0_info.initializer_tensor, unpacked_tensor));
      QnnTensorWrapper input_tensorwrapper(actual_input_0_name, QNN_TENSOR_TYPE_STATIC, input_0_info.qnn_data_type,
                                           std::move(quant_param_2d), std::move(shape_2d), std::move(unpacked_tensor));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
    } else {
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(original_input_0_name, actual_input_0_name,
                                                           input_0_info.shape, shape_2d,
                                                           input_0_info.qnn_data_type, input_0_info.quant_param,
                                                           quant_param_2d, do_op_validation,
                                                           qnn_model_wrapper.IsGraphInput(original_input_0_name), false));
    }
  } else {
    if (qnn_model_wrapper.IsQnnTensorWrapperExist(actual_input_0_name)) {
      LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << actual_input_0_name;
    } else {
      QnnTensorWrapper input_0_tensor;
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(input_0_info, actual_input_0_name, input_0_tensor));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_0_tensor)), "Failed to add tensor.");
    }
  }
  input_names.emplace_back(actual_input_0_name);

  return Status::OK();
}
}  // namespace

// Process operator inputs. Dispatches to other processing functions depending on whether we're
// translating an ONNX MatMul to a QNN MatMul or a QNN FullyConnected.
Status MatMulOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit,
                                      const logging::Logger& logger, std::vector<std::string>& input_names,
                                      bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  TensorInfo input_info_0{};
  TensorInfo input_info_1{};
  bool use_fully_connected = false;
  ORT_RETURN_IF_ERROR(
      CheckInputs(qnn_model_wrapper, inputs[0], inputs[1], input_info_0, input_info_1, use_fully_connected));

  if (use_fully_connected) {
    return ProcessInputsForQnnFullyConnected(qnn_model_wrapper,
                                             node_unit,
                                             input_info_0,
                                             input_info_1,
                                             logger,
                                             input_names,
                                             do_op_validation);
  }
  return ProcessInputsForQnnMatMul(qnn_model_wrapper,
                                   node_unit,
                                   input_info_0,
                                   input_info_1,
                                   logger,
                                   input_names,
                                   do_op_validation);
}

Status MatMulOpBuilder::ProcessInputsForQnnMatMul(QnnModelWrapper& qnn_model_wrapper,
                                                  const NodeUnit& node_unit,
                                                  const TensorInfo& input_info_0,
                                                  const TensorInfo& input_info_1,
                                                  const logging::Logger& logger,
                                                  std::vector<std::string>& input_names,
                                                  bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  const bool reshape_input_1 = input_info_1.shape.size() == 1;

  const std::string& org_input_0_name = inputs[0].node_arg.Name();
  ORT_RETURN_IF_ERROR(ProcessInput0(qnn_model_wrapper, input_info_0, org_input_0_name, input_names,
                                    logger, do_op_validation));

  // Process input 1.
  const std::string& org_input_1_name = inputs[1].node_arg.Name();
  std::string input_1_name = org_input_1_name;
  if (reshape_input_1) {
    // Input[1] is a rank 1 tensor that needs to be reshaped.
    std::vector<uint32_t> shape_2d;
    QnnQuantParamsWrapper quant_param_2d = input_info_1.quant_param.Copy();
    input_1_name = org_input_1_name + "_ort_qnn_ep_reshape";
    shape_2d = {input_info_1.shape[0], 1};
    ORT_RETURN_IF_ERROR(quant_param_2d.HandleUnsqueeze<uint32_t>(input_info_1.shape, shape_2d));

    // If input_1 is initializer, unpack it and add the tensor with new quantization parameter and shape.
    // Otherwise, add a Reshape node.
    if (input_info_1.is_initializer) {
      std::vector<uint8_t> unpacked_tensor;
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*input_info_1.initializer_tensor, unpacked_tensor));

      Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(org_input_1_name);
      QnnTensorWrapper input_tensorwrapper(input_1_name, tensor_type, input_info_1.qnn_data_type,
                                           std::move(quant_param_2d), std::move(shape_2d), std::move(unpacked_tensor));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
    } else {
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(org_input_1_name, input_1_name, input_info_1.shape, shape_2d,
                                                           input_info_1.qnn_data_type, input_info_1.quant_param,
                                                           quant_param_2d, do_op_validation,
                                                           qnn_model_wrapper.IsGraphInput(org_input_1_name), false));
    }
  } else {
    if (qnn_model_wrapper.IsQnnTensorWrapperExist(input_1_name)) {
      LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << input_1_name;
    } else {
      QnnTensorWrapper input_1_tensor;
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(inputs[1], input_1_tensor));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_1_tensor)), "Failed to add tensor.");
    }
  }
  input_names.emplace_back(input_1_name);

  // Workaround that inserts a QNN Convert op before input[1] (converts from quantized uint16 to quantized uint8)
  // to avoid a QNN validation failure.
  //
  // QNN graph WITHOUT workaround (fails validation):
  //     input_0_uint16 ---> MatMul ---> output_uint16
  //                         ^
  //                         |
  //     input_1_uint16 -----+
  //
  // QNN graph WITH workaround (passes validation):
  //     input_0_uint16 ----------------------> MatMul ---> output_uint16
  //                                            ^
  //                                            |
  //     input_1_uint16 --> Convert(to uint8) --+
  if (!input_info_0.is_initializer && !input_info_1.is_initializer &&
      input_info_0.qnn_data_type == input_info_1.qnn_data_type &&
      input_info_0.qnn_data_type == QNN_DATATYPE_UFIXED_POINT_16) {
    ORT_RETURN_IF_NOT(input_info_1.quant_param.IsPerTensor(),
                      "MatMul's activation inputs only support per-tensor quantization");
    const Qnn_QuantizeParams_t& quant_param = input_info_1.quant_param.Get();
    // insert Convert op after input1
    std::string convert_input_name = input_names.back();
    input_names.pop_back();
    const std::string& matmul_output_name = node_unit.Outputs()[0].node_arg.Name();
    std::string convert_output_name = convert_input_name + "_convert_" + matmul_output_name;
    std::vector<uint32_t> input_1_shape = input_info_1.shape;
    if (reshape_input_1) {
      input_1_shape = {input_info_1.shape[0], 1};
    }
    ORT_RETURN_IF_ERROR(InsertConvertOp(qnn_model_wrapper,
                                        convert_input_name,
                                        convert_output_name,
                                        input_info_1.qnn_data_type,
                                        QNN_DATATYPE_UFIXED_POINT_8,
                                        quant_param.scaleOffsetEncoding.offset,
                                        quant_param.scaleOffsetEncoding.scale,
                                        input_1_shape,
                                        do_op_validation));
    input_names.push_back(convert_output_name);
  }
  return Status::OK();
}

Status MatMulOpBuilder::ProcessInputsForQnnFullyConnected(QnnModelWrapper& qnn_model_wrapper,
                                                          const NodeUnit& node_unit,
                                                          const TensorInfo& input_info_0,
                                                          const TensorInfo& input_info_1,
                                                          const logging::Logger& logger,
                                                          std::vector<std::string>& input_names,
                                                          bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  const bool reshape_input_1 = input_info_1.shape.size() == 1;

  const std::string& org_input_0_name = inputs[0].node_arg.Name();
  ORT_RETURN_IF_ERROR(ProcessInput0(qnn_model_wrapper, input_info_0, org_input_0_name, input_names,
                                    logger, do_op_validation));

  // Process input 1.
  const std::string& org_input_1_name = inputs[1].node_arg.Name();
  std::string input_1_name = org_input_1_name;
  std::vector<uint32_t> shape_2d;
  QnnQuantParamsWrapper quant_param_2d = input_info_1.quant_param.Copy();
  if (reshape_input_1) {
    // Input[1] is a rank 1 tensor that needs to be reshaped.
    input_1_name = org_input_1_name + "_ort_qnn_ep_reshape";

    // FullyConnected requires input_1's shape to be [n, k].
    shape_2d = {1, input_info_1.shape[0]};
    ORT_RETURN_IF_ERROR(quant_param_2d.HandleUnsqueeze<uint32_t>(input_info_1.shape, shape_2d));
  } else {
    assert(input_info_1.shape.size() == 2);
    input_1_name = org_input_1_name + "_ort_qnn_ep_transpose";
    shape_2d = {input_info_1.shape[1], input_info_1.shape[0]};
    ORT_RETURN_IF_ERROR(quant_param_2d.HandleTranspose<uint32_t>(std::vector<uint32_t>({1, 0})));
  }

  // If input_1 is initializer, unpack it and add the tensor with new quantization parameter and shape.
  // Otherwise, add a Reshape node.
  if (input_info_1.is_initializer) {
    std::vector<uint8_t> unpacked_tensor;
    if (!reshape_input_1) {
      // 2D initializer should be transposed to [n, k].
      std::vector<uint32_t> original_shape_copy = input_info_1.shape;
      ORT_RETURN_IF_ERROR(utils::TwoDimensionTranspose(qnn_model_wrapper,
                                                       original_shape_copy,  // Will be modified to new shape (unnecessary)
                                                       *input_info_1.initializer_tensor,
                                                       unpacked_tensor));
    } else {
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*input_info_1.initializer_tensor, unpacked_tensor));
    }

    Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(org_input_1_name);
    QnnTensorWrapper input_tensorwrapper(input_1_name, tensor_type, input_info_1.qnn_data_type,
                                         std::move(quant_param_2d), std::move(shape_2d), std::move(unpacked_tensor));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
  } else {
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(org_input_1_name, input_1_name, input_info_1.shape, shape_2d,
                                                         input_info_1.qnn_data_type, input_info_1.quant_param,
                                                         quant_param_2d, do_op_validation,
                                                         qnn_model_wrapper.IsGraphInput(org_input_1_name), false));
  }
  input_names.emplace_back(input_1_name);
  return Status::OK();
}

Status MatMulOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit,
                                                    std::vector<std::string>&& input_names,
                                                    const logging::Logger& /*logger*/, bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  TensorInfo input_info_0{};
  TensorInfo input_info_1{};
  bool use_fully_connected = false;
  ORT_RETURN_IF_ERROR(
      CheckInputs(qnn_model_wrapper, inputs[0], inputs[1], input_info_0, input_info_1, use_fully_connected));
  bool reshape_input_0 = input_info_0.shape.size() == 1;
  bool reshape_input_1 = input_info_1.shape.size() == 1;
  bool reshape_output = reshape_input_0 || reshape_input_1 || (use_fully_connected && input_info_0.shape.size() > 2);

  // For QNN MatMul: set the input transpose parameters to their default values of 0. These parameters should be
  // optional, but older versions of QNN SDK failed validation if not explicitly provided.
  std::vector<std::string> param_tensor_names;
  if (!use_fully_connected) {
    Qnn_Scalar_t scalar_param = QNN_SCALAR_INIT;
    scalar_param.dataType = QNN_DATATYPE_BOOL_8;
    scalar_param.bool8Value = 0;
    QnnParamWrapper transpose_in0_param(node_unit.Index(), node_unit.Name(), QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0,
                                        scalar_param);
    param_tensor_names.push_back(transpose_in0_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(transpose_in0_param));

    QnnParamWrapper transpose_in1_param(node_unit.Index(), node_unit.Name(), QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1,
                                        scalar_param);
    param_tensor_names.push_back(transpose_in1_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(transpose_in1_param));
  }

  const std::string& org_output_name = node_unit.Outputs()[0].node_arg.Name();
  std::string op_output_name = org_output_name;
  TensorInfo output_info{};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[0], output_info));
  std::vector<uint32_t> op_output_shape = output_info.shape;
  QnnQuantParamsWrapper op_output_quant_param = output_info.quant_param.Copy();
  if (reshape_output) {
    op_output_name = org_output_name + "_ort_qnn_ep_reshape";
    if (use_fully_connected && input_info_0.shape.size() > 2) {
      op_output_shape = {std::accumulate(input_info_0.shape.begin(), input_info_0.shape.end() - 1,
                                         static_cast<uint32_t>(1), std::multiplies<uint32_t>()),
                         reshape_input_1 ? 1 : input_info_1.shape.back()};
      ORT_ENFORCE(!op_output_quant_param.IsPerChannel());
    } else {
      // If both inputs are 1D tensors, the output shape is [1] instead of scalar. So if both inputs are 1D tensors,
      // we only need to add one "1" to the op_output_shape.
      if (reshape_input_1) {
        op_output_shape.emplace_back(1);
      } else if (reshape_input_0) {
        op_output_shape.insert(op_output_shape.end() - 1, 1);
      }
      ORT_RETURN_IF_ERROR(op_output_quant_param.HandleUnsqueeze<uint32_t>(output_info.shape, op_output_shape));
    }
  }

  const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(org_output_name);
  const bool is_op_output_graph_output = is_graph_output && !reshape_output;
  Qnn_TensorType_t op_output_tensor_type =
      is_op_output_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
  QnnTensorWrapper op_output_tensor_wrapper(op_output_name, op_output_tensor_type, output_info.qnn_data_type,
                                            op_output_quant_param.Copy(), std::vector<uint32_t>(op_output_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(op_output_tensor_wrapper)),
                    "Failed to add output tensor.");
  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetNodeName(node_unit), QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    use_fully_connected ? QNN_OP_FULLY_CONNECTED : QNN_OP_MAT_MUL,
                                                    std::move(input_names), {op_output_name},
                                                    std::move(param_tensor_names), do_op_validation),
                    "Failed to add fused Matmul node.");

  if (reshape_output) {
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(
        op_output_name, org_output_name, op_output_shape, output_info.shape, output_info.qnn_data_type,
        op_output_quant_param, output_info.quant_param, do_op_validation, false, is_graph_output));
  }

  return Status::OK();
}

void CreateMatMulOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<MatMulOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
