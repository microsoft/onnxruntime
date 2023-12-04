// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/common/safeint.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace qnn {

class GemmOpBuilder : public BaseOpBuilder {
 public:
  GemmOpBuilder() : BaseOpBuilder("GemmOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GemmOpBuilder);

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

 private:
  Status ExplictOpCheck(const QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit) const;
};

Status GemmOpBuilder::ExplictOpCheck(const QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit) const {
  NodeAttrHelper node_helper(node_unit);
  auto alpha = node_helper.Get("alpha", (float)1.0);
  if (alpha != 1.0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN FullyConnected Op only support alpha=1.0.");
  }
  auto beta = node_helper.Get("beta", (float)1.0);
  if (beta != 1.0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN FullyConnected Op only support beta=1.0.");
  }

  // input C shape need to be [M] or [1, M]
  if (node_unit.Inputs().size() == 3) {
    const auto& input_b = node_unit.Inputs()[1];
    std::vector<uint32_t> input_b_shape;
    QnnModelWrapper::GetOnnxShape(input_b.node_arg, input_b_shape);

    const auto& input_c = node_unit.Inputs()[2];
    TensorInfo input_c_info = {};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(input_c, input_c_info));

    auto trans_b = node_helper.Get("transB", static_cast<int64_t>(0));
    auto M = (trans_b == 0) ? input_b_shape.at(1) : input_b_shape.at(0);
    if (input_c_info.shape.size() == 0 || (input_c_info.shape.size() == 1 && input_c_info.shape.at(0) != M) ||
        (input_c_info.shape.size() == 2 && (input_c_info.shape.at(0) != 1 || input_c_info.shape.at(1) != M))) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN FullyConnected Op only support C with shape [M].");
    }

    // Rank 2 input C needs to be reshaped from [1, M] to [M]. On HTP, reshape does not support
    // the SFIXED_POINT_32 data type.
    ORT_RETURN_IF(!input_c_info.is_initializer &&
                      !qnn_model_wrapper.IsGraphInput(input_c.node_arg.Name()) &&
                      (qnn_model_wrapper.GetQnnBackendType() == QnnBackendType::HTP) &&
                      (input_c_info.qnn_data_type == QNN_DATATYPE_SFIXED_POINT_32),
                  "QNN EP does not support Gemm with dynamic bias input with shape [1, M] and ",
                  "quantized data type int32 on the HTP backend (unless it is input to graph).");
  }

  return Status::OK();
}

Status GemmOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                    const NodeUnit& node_unit,
                                    const logging::Logger& logger,
                                    std::vector<std::string>& input_names,
                                    bool do_op_validation) const {
  if (do_op_validation) {
    ORT_RETURN_IF_ERROR(ExplictOpCheck(qnn_model_wrapper, node_unit));
  }
  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;

  // for Input A, B, C: 1 -- need transpose, 0 -- not needed
  std::vector<int64_t> input_trans_flag(3, 0);
  NodeAttrHelper node_helper(node_unit);
  input_trans_flag.at(0) = node_helper.Get("transA", (int64_t)0);
  auto transB = node_helper.Get("transB", (int64_t)0);
  // QNN input_1 [m, n] vs Onnx [n, m]
  input_trans_flag.at(1) = transB == 0 ? 1 : 0;

  const auto& inputs = node_unit.Inputs();

  // Process inputs A and B first.
  for (size_t input_i = 0; input_i < 2; ++input_i) {
    Qnn_QuantizeParams_t quantize_param = QNN_QUANTIZE_PARAMS_INIT;
    bool is_quantized_tensor = inputs[input_i].quant_param.has_value();
    utils::InitializeQuantizeParam(quantize_param, is_quantized_tensor);

    const auto& input_name = inputs[input_i].node_arg.Name();

    // Only skip if the input tensor has already been added (by producer op) *and* we don't need
    // to transpose it.
    if (qnn_model_wrapper.IsQnnTensorWrapperExist(input_name) && input_trans_flag[input_i] == 0) {
      LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << input_name;
      input_names.push_back(input_name);
      continue;
    }

    const auto* type_proto = inputs[input_i].node_arg.TypeAsProto();
    ORT_RETURN_IF_ERROR(utils::GetQnnDataType(is_quantized_tensor, type_proto, qnn_data_type));

    std::vector<uint32_t> input_shape;
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[input_i].node_arg, input_shape), "Cannot get shape");

    ORT_RETURN_IF_NOT(qnn_model_wrapper.ProcessQuantizationParameter(inputs[input_i].quant_param,
                                                                     quantize_param.scaleOffsetEncoding.scale,
                                                                     quantize_param.scaleOffsetEncoding.offset),
                      "Cannot get quantization parameter");

    std::vector<uint8_t> unpacked_tensor;
    bool is_initializer_input = qnn_model_wrapper.IsInitializerInput(input_name);
    if (is_initializer_input) {
      const auto& input_tensor = qnn_model_wrapper.GetInitializerTensors().at(input_name);
      if (1 == input_trans_flag.at(input_i)) {
        ORT_RETURN_IF_ERROR(TwoDimensionTranspose(qnn_model_wrapper,
                                                  input_shape,
                                                  *input_tensor,
                                                  unpacked_tensor));
      } else {
        ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*input_tensor, unpacked_tensor));
      }
    }

    std::string input_tensor_name = input_name;
    if (1 == input_trans_flag.at(input_i) && !is_initializer_input) {
      // Add Transpose node
      std::vector<uint32_t> old_input_shape(input_shape);
      input_shape[0] = old_input_shape[1];
      input_shape[1] = old_input_shape[0];
      const std::string& node_input_name(input_name);
      input_tensor_name = input_tensor_name + "_ort_qnn_ep_transpose";
      std::vector<uint32_t> perm{1, 0};
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(node_unit.Index(), node_input_name, input_tensor_name,
                                                             old_input_shape, perm, input_shape,
                                                             qnn_data_type, quantize_param, do_op_validation,
                                                             qnn_model_wrapper.IsGraphInput(node_input_name)));
    }

    input_names.push_back(input_tensor_name);
    Qnn_TensorType_t tensor_type = GetInputTensorType(qnn_model_wrapper, input_tensor_name);
    QnnTensorWrapper input_tensorwrapper(input_tensor_name, tensor_type, qnn_data_type, quantize_param,
                                         std::move(input_shape), std::move(unpacked_tensor));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
  }

  // Process input C (optional)
  constexpr size_t GEMM_INPUT_C_IDX = 2;
  if (inputs.size() > GEMM_INPUT_C_IDX && inputs[GEMM_INPUT_C_IDX].node_arg.Exists()) {
    const auto& input_c = inputs[GEMM_INPUT_C_IDX];
    std::string input_c_name = input_c.node_arg.Name();
    TensorInfo input_c_info = {};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(input_c, input_c_info));

    // QNN FullyConnected expects shape [m]. Rank 2 ONNX input c is shape [1, m].
    bool need_squeeze_shape = input_c_info.shape.size() == 2;

    // Only skip if the input tensor has already been added (by producer op) *and* we don't need
    // to squeeze it.
    if (qnn_model_wrapper.IsQnnTensorWrapperExist(input_c_name) && !need_squeeze_shape) {
      LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << input_c_name;
      input_names.push_back(input_c_name);
      return Status::OK();
    }

    if (need_squeeze_shape && !input_c_info.is_initializer) {
      std::vector<uint32_t> old_shape = input_c_info.shape;
      input_c_info.shape[0] = input_c_info.shape[1];
      input_c_info.shape.resize(1);

      const bool is_graph_input = qnn_model_wrapper.IsGraphInput(input_c_name);

      // Add Reshape before input activation. Don't need to reshape graph input since we can just override the
      // shape used by QNN.
      if (!is_graph_input) {
        std::string old_input_name = input_c_name;
        input_c_name += "_ort_qnn_ep_squeeze_gemm_input_c";

        ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(old_input_name,
                                                             input_c_name,
                                                             old_shape,
                                                             input_c_info.shape,
                                                             input_c_info.qnn_data_type,
                                                             input_c_info.quant_param,
                                                             do_op_validation,
                                                             is_graph_input));
      }
    }

    std::vector<uint8_t> unpacked_tensor;
    if (input_c_info.is_initializer) {
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*input_c_info.initializer_tensor, unpacked_tensor));

      if (need_squeeze_shape) {
        input_c_info.shape[0] = input_c_info.shape[1];
        input_c_info.shape.resize(1);
      }
    }

    Qnn_TensorType_t tensor_type = GetInputTensorType(qnn_model_wrapper, input_c_name);
    QnnTensorWrapper input_tensorwrapper(input_c_name, tensor_type, input_c_info.qnn_data_type,
                                         input_c_info.quant_param, std::move(input_c_info.shape),
                                         std::move(unpacked_tensor));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
    input_names.push_back(input_c_name);
  }

  return Status::OK();
}

Status GemmOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                  const NodeUnit& node_unit,
                                                  std::vector<std::string>&& input_names,
                                                  const logging::Logger& logger,
                                                  bool do_op_validation) const {
  ORT_RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit, std::move(input_names), {},
                                     logger, do_op_validation, GetQnnOpType(node_unit.OpType())));
  return Status::OK();
}

void CreateGemmOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<GemmOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
