// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "model_builder.h"

#include <unordered_map>

#include "core/common/common.h"
#include "core/common/inlined_containers_fwd.h"
#include "core/common/logging/logging.h"
#include "core/common/safeint.h"
#include "core/common/status.h"
#include "core/framework/execution_provider.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_viewer.h"
#include "core/providers/common.h"
#include "core/providers/nnapi/nnapi_builtin/nnapi_api_helper.h"
#include "core/providers/shared/node_unit/node_unit.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/nnapi/nnapi_builtin/builders/helper.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_builder.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_builder_factory.h"
#include "core/providers/nnapi/nnapi_builtin/nnapi_lib/nnapi_implementation.h"
#include "core/optimizer/initializer.h"

using namespace android::nn::wrapper;

namespace onnxruntime {
namespace nnapi {

ModelBuilder::ModelBuilder(const GraphViewer& graph_viewer, const NnApi& nnapi_handle,
                           gsl::span<const DeviceWrapper> nnapi_target_devices,
                           TargetDeviceOption target_device_option)
    : nnapi_(nnapi_handle), graph_viewer_(graph_viewer), nnapi_model_{std::make_unique<Model>(nnapi_handle)}, shaper_{graph_viewer}, nnapi_target_devices_(nnapi_target_devices), target_device_option_(target_device_option), nnapi_effective_feature_level_(GetNNAPIEffectiveFeatureLevel(nnapi_handle, nnapi_target_devices_)) {
  nnapi_model_->nnapi_effective_feature_level_ = nnapi_effective_feature_level_;
}

// Scalar operand is copied into the model, no need to persist
#define DEFINE_ADD_OPERAND_FROM_SCALAR(scalar_type, op_type)                      \
  Status ModelBuilder::AddOperandFromScalar(scalar_type value, uint32_t& index) { \
    OperandType operandType(Type::op_type, InlinedVector<uint32_t>{});            \
    ORT_RETURN_IF_ERROR(AddNewNNAPIOperand(operandType, index));                  \
    RETURN_STATUS_ON_ERROR_WITH_NOTE(                                             \
        nnapi_.ANeuralNetworksModel_setOperandValue(                              \
            nnapi_model_->model_, index, &value, sizeof(value)),                  \
        "value: " + std::to_string(value));                                       \
    return Status::OK();                                                          \
  }

DEFINE_ADD_OPERAND_FROM_SCALAR(bool, BOOL);
DEFINE_ADD_OPERAND_FROM_SCALAR(int32_t, INT32);
DEFINE_ADD_OPERAND_FROM_SCALAR(float, FLOAT32);

#undef DEFINE_ADD_OPERAND_FROM_SCALAR

void ModelBuilder::AddInitializerToSkip(const std::string& tensor_name) {
  skipped_initializers_.insert(tensor_name);
}

Status ModelBuilder::Prepare() {
  RETURN_STATUS_ON_ERROR(nnapi_.ANeuralNetworksModel_create(&nnapi_model_->model_));
  // Uncomment the following line to set the execution preference to
  // [PREFER_SUSTAINED_SPEED, PREFER_FAST_SINGLE_ANSWER, PREFER_LOW_POWER]
  // SetExecutePreference(android::nn::wrapper::ExecutePreference::PREFER_SUSTAINED_SPEED);
  PreprocessNodeUnits();
  GetAllQuantizedOpInputs();
  PreprocessInitializers();
  PreprocessActivations();
  ORT_RETURN_IF_ERROR(RegisterInitializers());
  ORT_RETURN_IF_ERROR(RegisterModelInputs());
  ORT_RETURN_IF_ERROR(AddOperations());
  ORT_RETURN_IF_ERROR(RegisterModelOutputs());

  return Status::OK();
}

constexpr size_t kDefaultByteAlignmentForNNAPI = 16;
static size_t GetPaddedByteSize(size_t size) {
  // if (size_t r = size % kDefaultByteAlignmentForNNAPI)
  //   return size + kDefaultByteAlignmentForNNAPI - r;
  // else
  //   return size;
  // This does exactly the same as the logic above
  return (size + kDefaultByteAlignmentForNNAPI - 1) & ~(kDefaultByteAlignmentForNNAPI - 1);
}

void ModelBuilder::PreprocessInitializers() {
  for (const auto& node_unit : node_unit_holder_) {
    if (const auto* op_builder = GetOpBuilder(*node_unit)) {
      op_builder->AddInitializersToSkip(*this, *node_unit);
    }
  }
}

void ModelBuilder::PreprocessActivations() {
  for (const auto& node_unit : node_unit_holder_) {
    const auto& node = node_unit->GetNode();
    const auto& op_type(node.OpType());
    if (op_type == "Relu") {
      activation_node_units_.emplace(node_unit.get(), ANEURALNETWORKS_FUSED_RELU);
    } else if (op_type == "Clip") {  // Relu1 or Relu6
      float min, max;
      if (!GetClipMinMax(graph_viewer_, node, min, max, logging::LoggingManager::DefaultLogger()))
        continue;

      if (min == -1.0f && max == 1.0f) {
        activation_node_units_.emplace(node_unit.get(), ANEURALNETWORKS_FUSED_RELU1);
      } else if (min == 0.0f && max == 6.0f) {
        activation_node_units_.emplace(node_unit.get(), ANEURALNETWORKS_FUSED_RELU6);
      }
    }
  }
}

const NodeUnit& ModelBuilder::GetNodeUnit(const Node* node) const {
  const auto node_unit_it = node_unit_map_.find(node);
  ORT_ENFORCE(node_unit_it != node_unit_map_.end(), "Node does not have corresponding NodeUnit.");
  return *node_unit_it->second;
}

void ModelBuilder::PreprocessNodeUnits() {
  std::tie(node_unit_holder_, node_unit_map_) = GetAllNodeUnits(graph_viewer_);
}

// Help to get all quantized operators' input and the NodeUnit(s) using the input
void ModelBuilder::GetAllQuantizedOpInputs() {
  for (const auto& node_unit : node_unit_holder_) {
    auto quant_op_type = GetQuantizedOpType(*node_unit);

    // Not a qlinear op or qdq node group
    if (quant_op_type == QuantizedOpType::Unknown)
      continue;

    const auto add_quantized_input =
        [&all_quantized_op_inputs = all_quantized_op_inputs_](const NodeUnit& node_unit, size_t input_idx) {
          const auto& input_name = node_unit.Inputs()[input_idx].node_arg.Name();
          all_quantized_op_inputs[input_name].push_back(&node_unit);
        };

    // All quantized ops EXCEPT QuantizeLinear has quantized input
    if (quant_op_type != QuantizedOpType::QuantizeLinear) {
      add_quantized_input(*node_unit, 0);
    }

    if (IsQuantizedBinaryOp(quant_op_type)) {
      add_quantized_input(*node_unit, 1);
    }

    // TODO, add handling for varidiac nodes such as QLinearConcat
  }
}

static Status GetInputDataType(
    const GraphViewer& graph_viewer,
    const std::unordered_map<std::string, std::vector<const NodeUnit*>>& all_quantized_op_inputs,
    const std::string& name, int32_t data_type, const Shape& shape,
    OperandType& operand_type) {
  Type type = Type::TENSOR_FLOAT32;
  float scale = 0.0f;
  int32_t zero_point = 0;
  switch (data_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      type = Type::TENSOR_FLOAT32;
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8: {
      // For ONNX the quantized input/initializer does not carry scale and zero point info
      // So we will need to search the operator using this input
      // And dig out the scale and zero point as the input initializers to the operator
      type = Type::TENSOR_QUANT8_ASYMM;
      if (!Contains(all_quantized_op_inputs, name)) {
        // We current do not support uint8 input if it is not a quantized input
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "The input/initializer of graph has unsupported quantized type, name: ", name,
                               " type: ", data_type);
      }

      // TODO, verify the scale and zero point match if there are multiple op using same input
      const auto* node_unit = all_quantized_op_inputs.at(name)[0];
      ORT_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(
          graph_viewer, *node_unit, name, scale, zero_point, ArgType::kInput));
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      type = Type::TENSOR_INT32;
      break;
      // case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      // We also do not consider ONNX_NAMESPACE::TensorProto_DataType_INT8 case here, since that can only
      // be input 2 of Qlinear[Conv/MatMul], which has to be an initializer tensor and will be added
      // separately by OpBuilder, so we do not treat it as an input/initializers here
    default:
      // TODO: support other type
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "The input/initializer of graph doesn't have valid type, name: ",
                             name, " type: ", data_type);
      break;
  }

  operand_type = OperandType(type, shape, scale, zero_point);
  return Status::OK();
}

Status ModelBuilder::RegisterInitializers() {
  // First pass to get all the stats of the initializers
  const auto& initializer_tensors(GetInitializerTensors());
  auto initializer_size = initializer_tensors.size();
  std::vector<std::tuple<uint32_t, size_t, size_t>> initializers(initializer_size);
  size_t sizeAll = 0;

  int i = 0;
  for (const auto& pair : initializer_tensors) {
    const auto& tensor = *pair.second;
    const auto& name = tensor.name();
    if (Contains(skipped_initializers_, name))
      continue;

    Shape shape;
    for (auto dim : tensor.dims()) {
      shape.push_back(SafeInt<uint32_t>(dim));
    }

    // If we have an empty shape, this is a scalar initializer,
    // since NNAPI will treat empty shape input as dynamic ranking input, (onnx does not support dynamic ranking)
    // we will make the scalar initializer as a {1} tensor
    if (shape.empty()) {
      shape.push_back(1);
    }

    OperandType operand_type(Type::TENSOR_FLOAT32, shape);
    ORT_RETURN_IF_ERROR(GetInputDataType(graph_viewer_, all_quantized_op_inputs_, name, tensor.data_type(), shape,
                                         operand_type));
    shaper_.AddShape(name, operand_type.dimensions);

    uint32_t index = 0;
    ORT_RETURN_IF_ERROR(AddNewOperand(name, operand_type, index));
    const size_t size = operand_type.GetOperandBlobByteSize();
    const size_t padded_size = GetPaddedByteSize(size);
    sizeAll += padded_size;
    initializers[i++] = std::make_tuple(index, size, padded_size);
  }

  // 2nd pass copies all the initializer data into NNAPI shared memory
  i = 0;
  nnapi_model_->mem_initializers_ =
      std::make_unique<Model::NNMemory>(nnapi_, "mem_initializers_", sizeAll);

  // 2nd pass to copy all the initializers into shared memory
  size_t offset = 0;
  for (const auto& pair : initializer_tensors) {
    const auto& tensor = *pair.second;
    if (Contains(skipped_initializers_, tensor.name()))
      continue;

    auto [index, size, padded_size] = initializers[i++];
    const uint8_t* src = nullptr;
    // TensorProto_DataType_UINT8 or TensorProto_DataType_FLOAT:
    Initializer unpacked_tensor(tensor, graph_viewer_.ModelPath());
    size_t size_in_bytes = unpacked_tensor.DataAsByteSpan().size();
    ORT_RETURN_IF_NOT(size == size_in_bytes,
                      "initializer tensor: ", tensor.name(), "'s size: ",
                      size_in_bytes, " should match the calculated size: ", size);
    src = unpacked_tensor.DataAsByteSpan().data();
    uint8_t* dest = nnapi_model_->mem_initializers_->GetDataPtr() + offset;
    memcpy(dest, src, size);
    ORT_RETURN_IF_ERROR(SetOperandValue(index, nnapi_model_->mem_initializers_.get(), size, offset));
    offset += padded_size;
  }

  return Status::OK();
}

Status ModelBuilder::RegisterModelInputs() {
  for (const auto* node_arg : graph_viewer_.GetInputs()) {
    const auto& input_name = node_arg->Name();

    {  // input should not be an initializer
      if (Contains(operands_, input_name))
        continue;

      if (Contains(GetInitializerTensors(), input_name))
        continue;
    }

    const auto* shape_proto = node_arg->Shape();
    ORT_RETURN_IF_NOT(shape_proto != nullptr, "shape_proto cannot be null for input: ", input_name);
    Shaper::Shape shape;

    for (const auto& dim : shape_proto->dim()) {
      // NNAPI uses 0 for dynamic dimension, which is the default value for dim.dim_value()
      shape.push_back(SafeInt<uint32_t>(dim.dim_value()));
    }

    // If we have an empty shape, this is a scalar input,
    // since NNAPI will treat empty shape input as dynamic ranking input, (onnx does not support dynamic ranking)
    // we will make the scalar input as a {1} tensor
    if (shape.empty()) {
      shape.push_back(1);
    }

    OperandType operand_type(Type::TENSOR_FLOAT32, shape);
    const auto* type_proto = node_arg->TypeAsProto();
    if (!type_proto || !type_proto->tensor_type().has_elem_type()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "The input of graph doesn't have elem_type: ", input_name);
    } else {
      ORT_RETURN_IF_ERROR(
          GetInputDataType(graph_viewer_, all_quantized_op_inputs_,
                           input_name, type_proto->tensor_type().elem_type(), shape, operand_type));
    }

    shaper_.AddShape(input_name, operand_type.dimensions);

    uint32_t index = 0;
    ORT_RETURN_IF_ERROR(AddNewOperand(input_name, operand_type, index));
    input_index_vec_.push_back(index);
    nnapi_model_->AddInput(input_name, operand_type);
  }

  return Status::OK();
}

Status ModelBuilder::RegisterModelOutputs() {
  for (const auto* node_arg : graph_viewer_.GetOutputs()) {
    const auto& output_name = node_arg->Name();

    if (!Contains(operands_, output_name)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "The output of graph is not registered [", output_name, "]");
    }

    // Since for now all the shapes are deterministic for NNAPI, it's impossible we can have unknown output shape
    const auto* shape_proto = node_arg->Shape();
    ORT_RETURN_IF(shape_proto == nullptr, "shape_proto cannot be null for output: ", output_name);
    if (shape_proto->dim_size() == 0) {
      // In NNAPI scalar output must have {1} shape
      const auto& output_shape = shaper_[output_name];
      ORT_RETURN_IF_NOT(output_shape.size() == 1 && output_shape[0] == 1,
                        "scalar output [", output_name, "] must have {1} shape, ",
                        " actual shape, ", Shape2String(output_shape));

      // Record the scalar output
      // Since within NNAPI the scalar outputs will have {1} shapes, and for ORT scalar outputs will have {} shapes,
      // we need to change the shapes of these scalar outputs back to {} when NNAPI EP returns these values to ORT
      nnapi_model_->AddScalarOutput(output_name);
    }

    std::string nnapi_output_name = output_name;

    output_index_vec_.push_back(operand_indices_[nnapi_output_name]);
    nnapi_model_->AddOutput(output_name, nnapi_output_name, operand_types_.at(nnapi_output_name));
  }

  return Status::OK();
}

Status ModelBuilder::AddNewOperand(const std::string& name,
                                   const OperandType& operand_type,
                                   uint32_t& index) {
  LOGS_DEFAULT(VERBOSE) << "operand name: " << name;
  ORT_RETURN_IF_ERROR(AddNewNNAPIOperand(operand_type, index));
  RegisterOperand(name, index, operand_type);
  return Status::OK();
}

Status ModelBuilder::AddNewNNAPIOperand(const OperandType& operand_type, uint32_t& index) {
  RETURN_STATUS_ON_ERROR(
      nnapi_.ANeuralNetworksModel_addOperand(nnapi_model_->model_, &operand_type.operandType));
  index = next_index_++;

  if (operand_type.channelQuant) {
    if (nnapi_effective_feature_level_ < ANEURALNETWORKS_FEATURE_LEVEL_3) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Per-channel quantization is only supported on Android API level 29+,",
                             " system NNAPI feature level: ", nnapi_effective_feature_level_);
    }

    RETURN_STATUS_ON_ERROR(nnapi_.ANeuralNetworksModel_setOperandSymmPerChannelQuantParams(
        nnapi_model_->model_, index, &operand_type.channelQuant->params));
  }

  return Status::OK();
}

void ModelBuilder::RegisterOperand(const std::string& name, uint32_t index,
                                   const OperandType& operand_type) {
  operand_indices_[name] = index;
  operand_types_.emplace(name, operand_type);
  operands_.insert(name);
}

Status ModelBuilder::SetOperandValue(uint32_t index,
                                     Model::NNMemory* memory,
                                     size_t size, size_t offset) {
#ifdef USENNAPISHAREDMEM
  RETURN_STATUS_ON_ERROR(
      nnapi_.ANeuralNetworksModel_setOperandValueFromMemory(
          nnapi_model_->model_, index,
          memory->GetHandle(),
          offset, size));
#else
  RETURN_STATUS_ON_ERROR(
      nnapi_.ANeuralNetworksModel_setOperandValue(
          nnapi_model_->model_, index,
          memory->GetDataPtr() + offset,
          size));
#endif

  return Status::OK();
}

Status ModelBuilder::AddOperandFromPersistMemoryBuffer(
    const std::string& name, const void* buffer,
    const android::nn::wrapper::OperandType& operand_type) {
  shaper_.AddShape(name, operand_type.dimensions);
  uint32_t index = 0;
  ORT_RETURN_IF_ERROR(AddNewOperand(name, operand_type, index));
  const size_t size = operand_type.GetOperandBlobByteSize();

  // for small size operand, the value will be copied
  // no need to persist
  if (size < ANEURALNETWORKS_MAX_SIZE_OF_IMMEDIATELY_COPIED_VALUES) {
    RETURN_STATUS_ON_ERROR(
        nnapi_.ANeuralNetworksModel_setOperandValue(
            nnapi_model_->model_, index,
            buffer, size));
  } else {
    const size_t padded_size = GetPaddedByteSize(size);
    auto persist_buffer = std::make_unique<Model::NNMemory>(nnapi_, name.c_str(), padded_size);
    uint8_t* dest = persist_buffer->GetDataPtr();
    memcpy(dest, buffer, size);
    ORT_RETURN_IF_ERROR(SetOperandValue(index, persist_buffer.get(), size, 0));
    nnapi_model_->mem_persist_buffers_.push_back(std::move(persist_buffer));
  }

  return Status::OK();
}

Status ModelBuilder::AddOperations() {
  const auto& node_indices = graph_viewer_.GetNodesInTopologicalOrder();
  for (const auto node_idx : node_indices) {
    LOGS_DEFAULT(VERBOSE) << "Adding node [" << node_idx << "]";
    const auto* node(graph_viewer_.GetNode(node_idx));
    const NodeUnit& node_unit = GetNodeUnit(node);

    // Since we may have NodeUnit with multiple nodes, insert NodeUnit with the first occurrence of
    // its node(s) in topological order may cause the incorrect topological order while inserting
    // NodeUNits, for example,
    //  Q1
    //  |
    //  DQ1  DQ2
    //    \   |
    //     CONV
    //      |
    //      Q2
    // In the above graph, we will have 2 NodeUnits, NU1 [Q1] and NU2 [DQ1, DQ2, CONV, Q2]
    // The Q1 and DQ2 have the same topological order, if we insert DQ2 (as part of NU2) when we visit DQ2
    // first in the topological order, the input from Q1 required by NU2 is not yet inserted, this will
    // cause failure finding the inputs for NU2
    //
    // So we only insert the NodeUnit once when we hit the target node, to ensure the topological order
    // of the NodeUnits
    if (node != &node_unit.GetNode())
      continue;

    if (const auto* op_builder = GetOpBuilder(node_unit)) {
      ORT_RETURN_IF_ERROR(op_builder->AddToModelBuilder(*this, node_unit));
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Node [", node_unit.Name(), "], type [", node_unit.OpType(), "] is not supported");
    }
  }

  return Status::OK();
}

Status ModelBuilder::AddOperation(int op, const InlinedVector<uint32_t>& input_indices,
                                  const std::vector<std::string>& output_names,
                                  const std::vector<OperandType>& output_types) {
#ifndef NDEBUG
  operations_recorder_.emplace_back(current_onnx_node_index_, op);
#endif
  InlinedVector<uint32_t> output_indices;
  for (size_t i = 0; i < output_types.size(); i++) {
    uint32_t index = 0;
    ORT_RETURN_IF_ERROR(AddNewOperand(output_names[i], output_types[i], index));
    output_indices.push_back(index);
  }

  RETURN_STATUS_ON_ERROR_WITH_NOTE(
      nnapi_.ANeuralNetworksModel_addOperation(
          nnapi_model_->model_, op, static_cast<uint32_t>(input_indices.size()), &input_indices[0],
          static_cast<uint32_t>(output_indices.size()), &output_indices[0]),
      "op = " + std::to_string(op));

  num_nnapi_ops_++;

  LOGS_DEFAULT(VERBOSE) << "Added NNAPI Operation Type [" << op << "]";
  return Status::OK();
}

Status ModelBuilder::Compile(std::unique_ptr<Model>& model) {
  ORT_RETURN_IF_ERROR(Prepare());

  RETURN_STATUS_ON_ERROR_WITH_NOTE(
      nnapi_.ANeuralNetworksModel_identifyInputsAndOutputs(
          nnapi_model_->model_, static_cast<uint32_t>(input_index_vec_.size()),
          &input_index_vec_[0],
          static_cast<uint32_t>(output_index_vec_.size()),
          &output_index_vec_[0]),
      "on identifyInputsAndOutputs");

  // relax fp32tofp16 is only available on API 28+
  if (use_fp16_ && nnapi_effective_feature_level_ > ANEURALNETWORKS_FEATURE_LEVEL_1) {
    RETURN_STATUS_ON_ERROR_WITH_NOTE(
        nnapi_.ANeuralNetworksModel_relaxComputationFloat32toFloat16(
            nnapi_model_->model_, true),
        "Set fp16");
  }

  RETURN_STATUS_ON_ERROR_WITH_NOTE(
      nnapi_.ANeuralNetworksModel_finish(nnapi_model_->model_),
      "on model finish");

  InlinedVector<ANeuralNetworksDevice*> device_handles;
  for (const auto& device_wp : nnapi_target_devices_) {
    device_handles.push_back(device_wp.device);
  }
  // We have a list of target devices, try to see if the model can be run entirely
  // using the list of target devices
  // This is only available on API 29+, for API 28- the nnapi_target_devices_ will
  // be empty so we will not check API level here, see GetTargetDevices()
  bool use_create_for_devices = false;
  std::unique_ptr<bool[]> supported_ops_holder = std::make_unique<bool[]>(num_nnapi_ops_);
  if (!nnapi_target_devices_.empty()) {
    auto* supported_ops = supported_ops_holder.get();
    RETURN_STATUS_ON_ERROR_WITH_NOTE(
        nnapi_.ANeuralNetworksModel_getSupportedOperationsForDevices(
            nnapi_model_->model_, device_handles.data(),
            static_cast<uint32_t>(device_handles.size()), supported_ops),
        "on getSupportedOperationsForDevices");

    bool all_ops_supported = std::all_of(supported_ops, supported_ops + num_nnapi_ops_,
                                         [](bool is_supported) { return is_supported; });
    if (!all_ops_supported) {
      // There are some ops not supported by the list of the target devices
      // Fail the Compile
      //
      // TODO, add some logic to not fail for some cases
      // Such as, if there are some acceptable fall back to CPU (nnapi-reference)
      // and CPU is not in the target devices list
      return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                             "The model cannot run using the current set of target devices, ",
                             GetDevicesDescription(nnapi_target_devices_));
    }
    // Workaround bugs in NNAPI drives on some phones
    // where ops are passed checking by 'ANeuralNetworksModel_getSupportedOperationsForDevices'
    // but failed at compilation.
    // ANeuralNetworksCompilation_create allows falling back to CPU if compilation fails
    if (target_device_option_ != TargetDeviceOption::ALL_DEVICES) {
      use_create_for_devices = true;
    }
  }

#ifndef NDEBUG
  if (nnapi_target_devices_.size() > 1 && nnapi_target_devices_.back().type == ANEURALNETWORKS_DEVICE_CPU) {
    auto supported_ops = gsl::make_span(supported_ops_holder.get(), num_nnapi_ops_);
    RETURN_STATUS_ON_ERROR_WITH_NOTE(
        nnapi_.ANeuralNetworksModel_getSupportedOperationsForDevices(
            nnapi_model_->model_, device_handles.data(),
            static_cast<uint32_t>(device_handles.size() - 1), supported_ops.data()),
        "on getSupportedOperationsForDevices");

    ORT_ENFORCE(num_nnapi_ops_ == operations_recorder_.size(), "num_nnapi_ops_!=operations_recorder_.size()");
    std::unordered_map<std::string, std::pair<int32_t, int32_t>> optype_support_status;
    for (size_t idx = 0; idx < operations_recorder_.size(); idx++) {
      auto [onnx_node_idx, nnapi_idx] = operations_recorder_[idx];
      const auto* node(graph_viewer_.GetNode(onnx_node_idx));
      auto stat_name = node->OpType() + ".nnapi_op_" + std::to_string(nnapi_idx);

      if (supported_ops[idx]) {
        optype_support_status[stat_name].first++;
      } else {
        optype_support_status[stat_name].second++;
      }
    }
    size_t fallback_ops = 0;
    std::string fallback_op_detail, normal_op_detail;

    for (const auto& [op, ops_status] : optype_support_status) {
      auto& [support_cnt, unspport_cnt] = ops_status;
      fallback_ops += unspport_cnt;
      if (support_cnt > 0) {
        normal_op_detail += MakeString(support_cnt, "x ", op, ", ");
      }
      if (unspport_cnt > 0) {
        fallback_op_detail += MakeString(unspport_cnt, "x ", op, ", ");
      }
    }

    LOGS_DEFAULT(VERBOSE) << fallback_ops << " Ops [" << fallback_op_detail << "] out of " << num_nnapi_ops_
                          << " are falling-back to " << kNnapiCpuDeviceName << ", and ["
                          << normal_op_detail << "] is running in accelerators.";
  }
#endif
  // If an op is supported and assigned to a device, it will be compiled by that device.
  // An internal compiling error will lead to the whole compilation process failure.
  // However, ANeuralNetworksCompilation_create will fall back to CPU if compilation fails.
  if (use_create_for_devices) {
    RETURN_STATUS_ON_ERROR_WITH_NOTE(
        nnapi_.ANeuralNetworksCompilation_createForDevices(
            nnapi_model_->model_, device_handles.data(),
            static_cast<uint32_t>(device_handles.size()), &nnapi_model_->compilation_),
        "on createForDevices");
  } else {
    RETURN_STATUS_ON_ERROR_WITH_NOTE(
        nnapi_.ANeuralNetworksCompilation_create(nnapi_model_->model_, &nnapi_model_->compilation_),
        "on create");
  }

  RETURN_STATUS_ON_ERROR_WITH_NOTE(
      nnapi_.ANeuralNetworksCompilation_setPreference(
          nnapi_model_->compilation_, static_cast<int32_t>(exe_pref_)),
      "on setPreference");

  RETURN_STATUS_ON_ERROR_WITH_NOTE(
      nnapi_.ANeuralNetworksCompilation_finish(nnapi_model_->compilation_),
      "on compilation finish");

  model.reset(nnapi_model_.release());
  return Status::OK();
}

int32_t ModelBuilder::FindActivation(const NodeUnit& node_unit) {
  const auto& output_def_size = node_unit.Outputs().size();
  if (output_def_size != 1) {
    LOGS_DEFAULT(VERBOSE) << "FindActivation does not support, NodeUnit [" << node_unit.Name()
                          << "] type [" << node_unit.OpType()
                          << "], with " << output_def_size << " output nodes";
    return ANEURALNETWORKS_FUSED_NONE;
  }

  const auto& outputs = node_unit.Outputs();
  if (outputs.size() != 1) {
    LOGS_DEFAULT(VERBOSE) << "FindActivation does not support, NodeUnit [" << node_unit.Name()
                          << "] type [" << node_unit.OpType()
                          << "], with " << outputs.size() << " outputs";
    return ANEURALNETWORKS_FUSED_NONE;
  }

  const NodeArg& output = outputs[0].node_arg;

  // if output is a graph output, will add activation separately
  if (const auto& graph_outputs = graph_viewer_.GetOutputs();
      std::find(graph_outputs.cbegin(), graph_outputs.cend(), &output) != graph_outputs.cend()) {
    return ANEURALNETWORKS_FUSED_NONE;
  }

  // TODO, add support of activation fusion for quantized node group (qdq or qlinear)
  // We do not support activation fusion for quantized operators for now
  // (usually the activations are fused already in the quantization)
  if (auto quant_op_type = GetQuantizedOpType(node_unit);
      quant_op_type != QuantizedOpType::Unknown) {
    return ANEURALNETWORKS_FUSED_NONE;
  }

  int32_t fuse_code = ANEURALNETWORKS_FUSED_NONE;
  bool fuse_code_assigned_from_activation = false;
  for (auto it = node_unit.OutputEdgesBegin(0), end = node_unit.OutputEdgesEnd(0); it != end; ++it) {
    const auto& dst_node = it->GetNode();
    const auto* dst_input = dst_node.InputDefs()[it->GetDstArgIndex()];

    if (&output != dst_input) {
      continue;
    }

    const auto& dst_node_unit = GetNodeUnit(&dst_node);
    auto activation_it = activation_node_units_.find(&dst_node_unit);
    if (activation_it == activation_node_units_.end()) {
      // output node is not a fusable activation
      return ANEURALNETWORKS_FUSED_NONE;
    }

    if (fuse_code_assigned_from_activation) {
      // don't overwrite a previously assigned fuse code, just don't fuse
      return ANEURALNETWORKS_FUSED_NONE;
    }

    fuse_code = activation_it->second;
    fuse_code_assigned_from_activation = true;
  }

  if (fuse_code != ANEURALNETWORKS_FUSED_NONE) {
    LOGS_DEFAULT(VERBOSE) << "Node [" << node_unit.Name() << "] type [" << node_unit.OpType()
                          << "], fused the output [" << output.Name() << "]";

    fused_activations_.insert(output.Name());
  }

  return fuse_code;
}

/* static */ const IOpBuilder* ModelBuilder::GetOpBuilder(const NodeUnit& node_unit) {
  const auto& op_builders = GetOpBuilders();
  const auto& op_type = node_unit.GetNode().OpType();
  if (!Contains(op_builders, op_type))
    return nullptr;
  return op_builders.at(op_type);
}

std::string ModelBuilder::GetUniqueName(const std::string& base_name) {
  std::string unique_name;
  do {
    std::ostringstream os;
    os << base_name << "_token_" << name_token_++;
    unique_name = os.str();
  } while (Contains(unique_names_, unique_name));

  return unique_name;
}

DataLayout ModelBuilder::GetPreferredLayout() const {
  return use_nchw_ ? DataLayout::NCHW : DataLayout::NHWC;
}

const InitializedTensorSet& ModelBuilder::GetInitializerTensors() const {
  return graph_viewer_.GetAllInitializedTensors();
}

const ONNX_NAMESPACE::TensorProto* ModelBuilder::GetConstantInitializer(const std::string& name) const {
  return graph_viewer_.GetConstantInitializer(name, true);
}

}  // namespace nnapi
}  // namespace onnxruntime
