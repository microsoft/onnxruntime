// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_quant_params_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {
/* Op Resolution
    --> Incoming ONNX Node
    1. MatMulNBits
    Attributes : INT64
      - accuracy_level  : 4
      - bits            : 4
      - block_size      : 32
      - K               :
      - N               :

    Inputs
      - A           :                 : (fp16/32) : [batch_size{1}, sequence_len, K]
      - B           : Init            : (uint8)   : [N, K/block_size, (block_size * bits) / 8]
      - scales      : Init            : (fp32)    : [N * K / block_size]
      - zero_points : (optional)Init  : (uint8)   : [N * K / (block_size * 2)]
      - bias        : (optional)Init  : [fp16/32] : [N]

    Outputs
      -  Y          :                 : (fp16/32) : [batch_size{1}, sequence_len, N]

  <-- Outgoing QNN Node
  1. FullyConnected
  Attributes
    -
  Inputs
    - Input           : (fp16/32)         : [batch_size{1}, sequence_len, K]
    - Weight          : Static : (qint4)  : [N, K]
      - Scales        : fp32              : [(N * K) / block_size{32}]
      - Offsets       : int32_t           : [(N * K) / block_size{32}]
    - Bias            : Static :(fp16/32) : [1, N]
  Outputs
    - Output          : (fp16/32)         : [batch_size{1} * sequence_len, N]

  2. Reshape
  Inputs
    - Input           : (fp16/32)         : [batch_size{1} * sequence_len, N]
  Outputs
    - Output          : (fp16/32)         : [batch_size{1}, sequence_len, N]
*/

class MatMulNBitsOpBuilder : public BaseOpBuilder {
 public:
  MatMulNBitsOpBuilder() : BaseOpBuilder("MatMulNBitsOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MatMulNBitsOpBuilder);

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

 private:
  void DQQToSignedFixedPoint4(std::vector<uint8_t>& quant_data, int64_t num_blocks, int64_t block_size) const;
};

void MatMulNBitsOpBuilder::DQQToSignedFixedPoint4(std::vector<uint8_t>& quant_data,
                                                  int64_t num_blocks,
                                                  int64_t block_size) const {
  for (int64_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
    uint32_t zero_point = 8;
    for (int64_t val_idx = 0; val_idx < (block_size / 2); ++val_idx) {
      SafeInt<int64_t> safe_index = block_idx;
      safe_index *= (block_size / 2);
      safe_index += val_idx;

      size_t index = gsl::narrow_cast<size_t>(safe_index);
      uint8_t quant_value_4x2 = quant_data[index];

      int8_t quant_upper_value =
          gsl::narrow_cast<int8_t>(((quant_value_4x2 >> 4) & 0xF) - zero_point);
      int8_t quant_lower_value =
          gsl::narrow_cast<int8_t>(((quant_value_4x2 >> 0) & 0xF) - zero_point);

      quant_data[index] = ((quant_upper_value & 0xF) << 4) | (quant_lower_value & 0xF);
    }
  }
}

Status MatMulNBitsOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                           const NodeUnit& node_unit,
                                           const logging::Logger& logger) const {
  bool is_gpu_backend = IsGpuBackend(qnn_model_wrapper.GetQnnBackendType());
  ORT_RETURN_IF_NOT(is_gpu_backend, "MatMulNBits Op Supported Only for Qnn Gpu Backend");

  Qnn_DataType_t input_datatype = QNN_DATATYPE_FLOAT_32;
  Qnn_DataType_t datatype = QNN_DATATYPE_FLOAT_32;

  NodeAttrHelper node_helper(node_unit);

  // Extract Parameters
  const int64_t bits = node_helper.Get("bits", static_cast<int64_t>(4));
  const int64_t block_size = node_helper.Get("block_size", static_cast<int64_t>(32));

  const int64_t K = node_helper.Get("K", static_cast<int64_t>(1));
  const int64_t N = node_helper.Get("N", static_cast<int64_t>(1));

  ORT_RETURN_IF_NOT(bits == 4, "Invalid bits. Qnn Gpu Only Supports MatMulNBits with bits == 4");
  ORT_RETURN_IF_NOT(block_size == 32, "Invalid block_size. Qnn Gpu Only Supports MatMulNBits with block_size == 32");
  ORT_RETURN_IF_NOT((K % block_size) == 0, "K must be divisible by block_size");
  ORT_RETURN_IF_NOT(((N * K) % (2 * block_size)) == 0,
                    "Invalid configuration. N * K must be divisible by 2 * block_size");

  const int64_t num_blocks = (N * K) / block_size;
  ORT_RETURN_IF_NOT(num_blocks > 0, "Invalid configuration. (N * K) / block_size must be > 0");

  const auto& inputs = node_unit.Inputs();
  // 1. input : Datatype should be float16 or float32
  // Float16 Dlc serialization failing, Skipping float16 support for this op builder
  // TODO :: Add Float16 Support
  {
    const NodeUnitIODef& input_tensor = inputs[0];
    TensorInfo input_info{};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(input_tensor, input_info));
    ORT_RETURN_IF_ERROR(utils::GetQnnDataType(input_tensor.quant_param.has_value(),
                                              input_tensor.node_arg.TypeAsProto(),
                                              input_datatype));
    ORT_RETURN_IF(input_datatype != QNN_DATATYPE_FLOAT_32, "Unsupported Input datatype");
  }

  // 2. weight : weight supported with packed int4 into int8.
  {
    const NodeUnitIODef& input_tensor = inputs[1];
    TensorInfo input_info{};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(input_tensor, input_info));

    const std::vector<uint32_t> input_shape = input_info.shape;
    SafeInt<int64_t> safe_total_elements = std::accumulate(input_shape.begin(),
                                                           input_shape.end(),
                                                           SafeInt<int64_t>{1},
                                                           std::multiplies<>());
    const int64_t total_elements = static_cast<int64_t>(safe_total_elements);
    ORT_RETURN_IF_NOT(((total_elements * 2) == (N * K)),
                      "Invalid B dimensions. Qnn Gpu Only Supports MatMulNBits with bits == 4 "
                      "in packed format");
  }

  // 3. scales : scales only float32 datatype
  {
    const NodeUnitIODef& input_tensor = inputs[2];
    TensorInfo input_info{};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(input_tensor, input_info));
    ORT_RETURN_IF_ERROR(utils::GetQnnDataType(input_tensor.quant_param.has_value(),
                                              input_tensor.node_arg.TypeAsProto(),
                                              input_datatype));
    ORT_RETURN_IF(input_datatype != QNN_DATATYPE_FLOAT_32, "Unsupported Input datatype");
  }

  // 4. If input 3 exists, it has to be zero point.
  if (inputs.size() > 3 && inputs[3].node_arg.Exists()) {
    const NodeUnitIODef& input_tensor = inputs[3];
    TensorInfo input_info{};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(input_tensor, input_info));
    ORT_RETURN_IF_ERROR(utils::GetQnnDataType(input_tensor.quant_param.has_value(),
                                              input_tensor.node_arg.TypeAsProto(),
                                              datatype));
    ORT_RETURN_IF((datatype != QNN_DATATYPE_UINT_8), "Invalid zero point datatype.");

    std::vector<uint8_t> per_block_uint8_offset;
    const auto& zero_points_tensor_name = input_tensor.node_arg.Name();
    const auto& zero_points_tensor_proto = qnn_model_wrapper.GetConstantTensor(zero_points_tensor_name);
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*zero_points_tensor_proto,
                                                                per_block_uint8_offset));

    ORT_RETURN_IF_NOT((per_block_uint8_offset.size() * 2) == (num_blocks * sizeof(uint8_t)),
                      "Only packed uint4 into uint8 offset supported by op builder");
    const uint8_t expected_offset_value = 0b10001000;
    for (size_t i = 0; i < per_block_uint8_offset.size(); i++) {
      ORT_RETURN_IF_NOT(per_block_uint8_offset[i] == expected_offset_value, "Unsupported zero point value");
    }
  }

  ORT_RETURN_IF((inputs.size() > 4 && inputs[4].node_arg.Exists()) ||
                    (inputs.size() > 5 && inputs[5].node_arg.Exists()),
                "Unsupported inputs g_idx or bias");

  // Validate Process
  std::vector<std::string> input_names;
  ORT_RETURN_IF_ERROR(ProcessInputs(qnn_model_wrapper, node_unit, logger, input_names, true));
  ORT_RETURN_IF_ERROR(ProcessAttributesAndOutputs(qnn_model_wrapper, node_unit, std::move(input_names),
                                                  logger, true));

  return Status::OK();
}

Status MatMulNBitsOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                           const NodeUnit& node_unit,
                                           const logging::Logger& logger,
                                           std::vector<std::string>& input_names,
                                           bool do_op_validation) const {
  if (do_op_validation) {
    bool is_gpu_backend = IsGpuBackend(qnn_model_wrapper.GetQnnBackendType());
    ORT_RETURN_IF_NOT(is_gpu_backend, "MatMulNBits Op Supported Only for Qnn Gpu Backend");
  }
  NodeAttrHelper node_helper(node_unit);

  // Extract Parameters
  const int64_t block_size = node_helper.Get("block_size", static_cast<int64_t>(32));
  const int64_t K = node_helper.Get("K", static_cast<int64_t>(1));
  const int64_t N = node_helper.Get("N", static_cast<int64_t>(1));

  // Prepare essential parameters
  const int64_t num_blocks = (N * K) / block_size;
  const auto& inputs = node_unit.Inputs();

  // 1. Add Input
  {
    const NodeUnitIODef& input_tensor = inputs[0];
    const std::string& input_tensor_name = input_tensor.node_arg.Name();
    if (qnn_model_wrapper.IsQnnTensorWrapperExist(input_tensor_name)) {
      LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << input_tensor_name;
    } else {
      TensorInfo input_info{};
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(input_tensor, input_info));

      QnnTensorWrapper input_tensor_wrapper;
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(input_info,
                                                              input_tensor_name,
                                                              input_tensor_wrapper));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensor_wrapper)),
                        "Failed to add tensor.");
    }
    input_names.push_back(input_tensor_name);
  }

  // 2. Add Weights and add its Quantization Data
  {
    const auto& weight_tensor = inputs[1];
    const auto& scales_tensor = inputs[2];

    const auto& weight_tensor_name = weight_tensor.node_arg.Name();
    if (qnn_model_wrapper.IsQnnTensorWrapperExist(weight_tensor_name)) {
      LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << weight_tensor_name;
    } else {
      const std::vector<uint32_t> block_sizes = {1, gsl::narrow_cast<uint32_t>(block_size)};

      // 2.1 Quantization Weight Data
      std::vector<uint8_t> quant_data;
      Qnn_TensorType_t weight_tensor_type = qnn_model_wrapper.GetTensorType(weight_tensor_name);
      const auto& weight_tensor_proto = qnn_model_wrapper.GetConstantTensor(weight_tensor_name);
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*weight_tensor_proto,
                                                                  quant_data,
                                                                  false));

      // 2.2 Quantization Scales
      std::vector<uint8_t> per_block_uint8_scale;
      const auto& scale_tensor_proto = qnn_model_wrapper.GetConstantTensor(scales_tensor.node_arg.Name());
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*scale_tensor_proto,
                                                                  per_block_uint8_scale));
      ORT_RETURN_IF_NOT(per_block_uint8_scale.size() == (num_blocks * sizeof(float)),
                        "Scale Initializer Invalid Size");
      float* per_block_float_scale_ptr = reinterpret_cast<float*>(per_block_uint8_scale.data());
      const std::vector<float> per_block_float_scale(per_block_float_scale_ptr,
                                                     per_block_float_scale_ptr + num_blocks);

      // 2.3 Quantization Offsets : QNN Support only symmetric quantization with default value of 0
      std::vector<int32_t> per_block_int32_offset(num_blocks, 0);

      // 2.4 Transform quantized weights to signed fixed point 4.
      DQQToSignedFixedPoint4(quant_data, num_blocks, block_size);

      // 2.5 Create Quantization Parameter and create Weight Tensor
      QnnQuantParamsWrapper quantize_param = QnnQuantParamsWrapper(per_block_float_scale,
                                                                   per_block_int32_offset,
                                                                   block_sizes,
                                                                   QNN_DATATYPE_SFIXED_POINT_4);

      std::vector<uint32_t> weight_shape = {static_cast<uint32_t>(N), static_cast<uint32_t>(K)};
      QnnTensorWrapper weight_tensor_wrapper(weight_tensor_name,
                                             weight_tensor_type,
                                             QNN_DATATYPE_SFIXED_POINT_4,
                                             std::move(quantize_param),
                                             std::move(weight_shape),
                                             std::move(quant_data));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(weight_tensor_wrapper)),
                        "Failed to add tensor.");
    }
    input_names.push_back(weight_tensor_name);
  }

  return Status::OK();
}

Status MatMulNBitsOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                         const NodeUnit& node_unit,
                                                         std::vector<std::string>&& input_names,
                                                         const logging::Logger& logger,
                                                         bool do_op_validation) const {
  if (do_op_validation) {
    bool is_gpu_backend = IsGpuBackend(qnn_model_wrapper.GetQnnBackendType());
    ORT_RETURN_IF_NOT(is_gpu_backend, "MatMulNBits Op Supported Only for Qnn Gpu Backend");
  }

  NodeAttrHelper node_helper(node_unit);
  // Extract Parameters
  const int64_t N = node_helper.Get("N", static_cast<int64_t>(1));

  // 1. Add Output for Reshape
  const NodeUnitIODef& output_tensor = node_unit.Outputs()[0];
  const std::string& output_tensor_name = output_tensor.node_arg.Name();
  if (qnn_model_wrapper.IsQnnTensorWrapperExist(output_tensor_name)) {
    LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << output_tensor_name;
  } else {
    QnnTensorWrapper output_tensor_wrapper;
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(output_tensor, output_tensor_wrapper));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor_wrapper)),
                      "Failed to add output");
  }

  // 2. Add Output for Pre Reshape(FullyConnected)
  const std::string pre_reshape_name = utils::GetUniqueName(output_tensor_name, "_pre_reshape");
  TensorInfo output_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(output_tensor, output_info));
  std::vector<uint32_t> pre_reshape_shape(2);
  pre_reshape_shape[0] = static_cast<uint32_t>(std::accumulate(output_info.shape.begin(),
                                                               output_info.shape.end(),
                                                               SafeInt<uint32_t>{1},
                                                               std::multiplies<>()) /
                                               N);
  pre_reshape_shape[1] = gsl::narrow_cast<uint32_t>(N);
  QnnTensorWrapper output_tensor_wrapper(pre_reshape_name,
                                         QNN_TENSOR_TYPE_NATIVE,
                                         output_info.qnn_data_type,
                                         output_info.quant_param.Copy(),
                                         std::vector<uint32_t>(pre_reshape_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor_wrapper)),
                    "Failed to add tensor.");

  // 3. Add FullyConnected Op
  const std::string fully_connected_node_name = utils::GetUniqueName(node_unit, QNN_OP_FULLY_CONNECTED);
  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(fully_connected_node_name,
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_FULLY_CONNECTED,
                                                    std::move(input_names),
                                                    {pre_reshape_name},
                                                    {},
                                                    do_op_validation),
                    "Failed to add fused Matmul node.");

  // 4. Add Reshape Op
  const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_tensor_name);
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(pre_reshape_name,
                                                       output_tensor_name,
                                                       pre_reshape_shape,
                                                       output_info.shape,
                                                       output_info.qnn_data_type,
                                                       output_info.quant_param,
                                                       do_op_validation,
                                                       false,
                                                       is_graph_output));

  return Status::OK();
}

void CreateMatMulNBitsOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<MatMulNBitsOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
