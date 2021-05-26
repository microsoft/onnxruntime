// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/common/logging/logging.h>
#include <core/common/safeint.h>
#include <core/framework/tensorprotoutils.h>

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/nnapi/nnapi_builtin/nnapi_lib/nnapi_implementation.h"
#include "helper.h"
#include "model_builder.h"
#include "op_builder.h"
#include "op_support_checker.h"

namespace onnxruntime {
namespace nnapi {

using namespace android::nn::wrapper;
using std::vector;

ModelBuilder::ModelBuilder(const GraphViewer& graph_viewer)
    : nnapi_(NnApiImplementation()), graph_viewer_(graph_viewer) {}

int32_t ModelBuilder::GetAndroidSdkVer() const {
  return nnapi_ ? nnapi_->android_sdk_version : 0;
}

// Scalar operand is copied into the model, no need to persist
#define DEFINE_ADD_OPERAND_FROM_SCALAR(scalar_type, op_type)                      \
  Status ModelBuilder::AddOperandFromScalar(scalar_type value, uint32_t& index) { \
    OperandType operandType(Type::op_type, vector<uint32_t>{});                   \
    ORT_RETURN_IF_ERROR(AddNewNNAPIOperand(operandType, index));                  \
    RETURN_STATUS_ON_ERROR_WITH_NOTE(                                             \
        nnapi_->ANeuralNetworksModel_setOperandValue(                             \
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

static std::unordered_map<std::string, vector<const Node*>> GetAllQuantizedOpInputs(const GraphViewer& graph_viewer);

Status ModelBuilder::Prepare() {
  nnapi_model_ = std::unique_ptr<Model>(new Model());
  RETURN_STATUS_ON_ERROR(nnapi_->ANeuralNetworksModel_create(&nnapi_model_->model_));
  ORT_RETURN_IF_ERROR(GetTargetDevices());
  all_quantized_op_inputs_ = GetAllQuantizedOpInputs(graph_viewer_);
  PreprocessInitializers();
  PreprocessActivations();
  ORT_RETURN_IF_ERROR(RegisterInitializers());
  ORT_RETURN_IF_ERROR(RegisterModelInputs());
  ORT_RETURN_IF_ERROR(AddOperations());
  ORT_RETURN_IF_ERROR(RegisterModelOutputs());
  RegisterModelShaper();

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

Status ModelBuilder::GetTargetDevices() {
  // GetTargetDevices is only supported on API 29+
  if (GetAndroidSdkVer() < 29)
    return Status::OK();

  if (target_device_option_ == TargetDeviceOption::ALL_DEVICES)
    return Status::OK();

  const std::string nnapi_cpu("nnapi-reference");
  uint32_t num_devices = 0;
  RETURN_STATUS_ON_ERROR_WITH_NOTE(
      nnapi_->ANeuralNetworks_getDeviceCount(&num_devices), "Getting count of available devices");

  for (uint32_t i = 0; i < num_devices; i++) {
    ANeuralNetworksDevice* device = nullptr;
    const char* device_name = nullptr;
    int32_t device_type;
    RETURN_STATUS_ON_ERROR_WITH_NOTE(
        nnapi_->ANeuralNetworks_getDevice(i, &device), "Getting " + std::to_string(i) + "th device");

    RETURN_STATUS_ON_ERROR_WITH_NOTE(nnapi_->ANeuralNetworksDevice_getName(device, &device_name),
                                     "Getting " + std::to_string(i) + "th device's name");

    RETURN_STATUS_ON_ERROR_WITH_NOTE(nnapi_->ANeuralNetworksDevice_getType(device, &device_type),
                                     "Getting " + std::to_string(i) + "th device's type");

    bool device_is_cpu = nnapi_cpu == device_name;
    if ((target_device_option_ == TargetDeviceOption::CPU_DISABLED && !device_is_cpu) ||
        (target_device_option_ == TargetDeviceOption::CPU_ONLY && device_is_cpu)) {
      nnapi_target_devices_.push_back(device);
      const auto device_detail = MakeString("[Name: [", device_name, "], Type [", device_type, "]], ");
      nnapi_target_devices_detail_ += device_detail;
      LOGS_DEFAULT(VERBOSE) << "Target device " << device_detail << " is added";
    }
  }

  return Status::OK();
}

void ModelBuilder::PreprocessInitializers() {
  const auto& node_indices = graph_viewer_.GetNodesInTopologicalOrder();
  for (size_t i = 0; i < node_indices.size(); i++) {
    const auto* node(graph_viewer_.GetNode(node_indices[i]));
    if (const auto* op_builder = GetOpBuilder(*node)) {
      op_builder->AddInitializersToSkip(*this, *node);
    }
  }
}

void ModelBuilder::PreprocessActivations() {
  const auto& node_indices = graph_viewer_.GetNodesInTopologicalOrder();
  for (size_t i = 0; i < node_indices.size(); i++) {
    const auto* node(graph_viewer_.GetNode(node_indices[i]));
    const auto& op_type(node->OpType());

    if (op_type == "Relu") {
      activation_nodes_.emplace(node->Index(), ANEURALNETWORKS_FUSED_RELU);
    } else if (op_type == "Clip") {  // Relu1 or Relu6
      float min, max;
      if (!GetClipMinMax(GetInitializerTensors(), *node, min, max, logging::LoggingManager::DefaultLogger()))
        continue;

      if (min == -1.0f && max == 1.0f) {
        activation_nodes_.emplace(node->Index(), ANEURALNETWORKS_FUSED_RELU1);
      } else if (min == 0.0f && max == 6.0f) {
        activation_nodes_.emplace(node->Index(), ANEURALNETWORKS_FUSED_RELU6);
      }
    }
  }
}

// Help to get all quantized operators' input and the node(s) using the input
static std::unordered_map<std::string, vector<const Node*>> GetAllQuantizedOpInputs(const GraphViewer& graph_viewer) {
  std::unordered_map<std::string, vector<const Node*>> all_quantized_op_inputs;
  const auto& node_indices = graph_viewer.GetNodesInTopologicalOrder();
  for (const auto& node_idx : node_indices) {
    const auto* node(graph_viewer.GetNode(node_idx));
    auto qlinear_op_type = GetQLinearOpType(*node);

    // Not a qlinear op
    if (qlinear_op_type == QLinearOpType::Unknown)
      continue;

    // All qlinear ops EXCEPT QuantizeLinear has quantized input
    if (qlinear_op_type != QLinearOpType::QuantizeLinear) {
      const auto& input_name = node->InputDefs()[0]->Name();
      if (Contains(all_quantized_op_inputs, input_name))
        all_quantized_op_inputs.at(input_name).push_back(node);
      else
        all_quantized_op_inputs.emplace(input_name, vector<const Node*>{node});
    }

    if (IsQLinearBinaryOp(qlinear_op_type)) {
      const auto& input_name = node->InputDefs()[3]->Name();
      if (Contains(all_quantized_op_inputs, input_name))
        all_quantized_op_inputs.at(input_name).push_back(node);
      else
        all_quantized_op_inputs.emplace(input_name, vector<const Node*>{node});
    }
  }

  return all_quantized_op_inputs;
}

static Status GetInputDataType(
    const InitializedTensorSet& initializers,
    const std::unordered_map<std::string, std::vector<const Node*>>& all_quantized_op_inputs,
    const std::string& name, int32_t data_type, const Shape& shape,
    OperandType& operand_type) {
  Type type = Type::TENSOR_FLOAT32;
  float scale = 0.0f;
  int32_t zero_point = 0;
  switch (data_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      type = Type::TENSOR_FLOAT32;
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
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
      ORT_RETURN_IF_ERROR(GetQuantizedInputScaleAndZeroPoint(
          initializers, *all_quantized_op_inputs.at(name)[0], name, scale, zero_point));
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
    ORT_RETURN_IF_ERROR(
        GetInputDataType(GetInitializerTensors(), all_quantized_op_inputs_,
                         name, tensor.data_type(), shape, operand_type));
    shaper_.AddShape(name, operand_type.dimensions);

    uint32_t index = 0;
    ORT_RETURN_IF_ERROR(AddNewOperand(name, operand_type, false /* is_nhwc */, index));
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

    uint32_t index;
    size_t size, padded_size;
    std::tie(index, size, padded_size) = initializers[i++];
    const uint8_t* src = nullptr;
    // uint8_t data need unpack, need a holder for free memory after copy
    std::unique_ptr<uint8_t[]> unpacked_tensor;
    switch (tensor.data_type()) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
        src = reinterpret_cast<const uint8_t*>(GetTensorFloatData(tensor));
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
        size_t tensor_byte_size;
        ORT_RETURN_IF_ERROR(
            onnxruntime::utils::UnpackInitializerData(tensor, graph_viewer_.ModelPath(),
                                                      unpacked_tensor, tensor_byte_size));
        ORT_RETURN_IF_NOT(size == tensor_byte_size,
                          "initializer tensor: ", tensor.name(), "'s size: ", tensor_byte_size,
                          " should match the calculated size: ", size);
        src = unpacked_tensor.get();
        break;
        // default:
        // We should not get anything else here since we already checked in the 1st pass
    }

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
          GetInputDataType(GetInitializerTensors(), all_quantized_op_inputs_,
                           input_name, type_proto->tensor_type().elem_type(), shape, operand_type));
    }

    shaper_.AddShape(input_name, operand_type.dimensions);

    uint32_t index = 0;
    ORT_RETURN_IF_ERROR(AddNewOperand(input_name, operand_type, false /* is_nhwc */, index));
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
    if (IsOperandNHWC(output_name)) {
      // We need to transpose the output still in nhwc back to nchw
      nnapi_output_name = GetUniqueName(output_name + "_nhwc_to_nchw");
      ORT_RETURN_IF_ERROR(TransposeNHWCToNCHW(*this, output_name, nnapi_output_name));
    }

    output_index_vec_.push_back(operand_indices_[nnapi_output_name]);
    nnapi_model_->AddOutput(output_name, nnapi_output_name, operand_types_.at(nnapi_output_name));
  }

  return Status::OK();
}

void ModelBuilder::RegisterModelShaper() {
  nnapi_model_->SetShaper(shaper_);
}

Status ModelBuilder::AddNewOperand(const std::string& name,
                                   const OperandType& operand_type,
                                   bool is_nhwc, uint32_t& index) {
  LOGS_DEFAULT(VERBOSE) << "operand name: " << name;
  ORT_RETURN_IF_ERROR(AddNewNNAPIOperand(operand_type, index));
  RegisterOperand(name, index, operand_type, is_nhwc);
  return Status::OK();
}

Status ModelBuilder::AddNewNNAPIOperand(const OperandType& operand_type, uint32_t& index) {
  RETURN_STATUS_ON_ERROR(
      nnapi_->ANeuralNetworksModel_addOperand(nnapi_model_->model_, &operand_type.operandType));
  index = next_index_++;

  if (operand_type.channelQuant) {
    if (GetAndroidSdkVer() < 29) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Per-channel quantization is only supported on Android API level 29+,",
                             " system API level: ", GetAndroidSdkVer());
    }

    RETURN_STATUS_ON_ERROR(nnapi_->ANeuralNetworksModel_setOperandSymmPerChannelQuantParams(
        nnapi_model_->model_, index, &operand_type.channelQuant->params));
  }

  return Status::OK();
}

void ModelBuilder::RegisterOperand(const std::string& name, uint32_t index,
                                   const OperandType& operand_type, bool is_nhwc) {
  operand_indices_[name] = index;
  operand_types_.emplace(name, operand_type);
  operands_.insert(name);

  if (is_nhwc)
    RegisterNHWCOperand(name);
}

Status ModelBuilder::SetOperandValue(uint32_t index,
                                     Model::NNMemory* memory,
                                     size_t size, size_t offset) {
#ifdef USENNAPISHAREDMEM
  RETURN_STATUS_ON_ERROR(
      nnapi_->ANeuralNetworksModel_setOperandValueFromMemory(
          nnapi_model_->model_, index,
          memory->GetHandle(),
          offset, size));
#else
  RETURN_STATUS_ON_ERROR(
      nnapi_->ANeuralNetworksModel_setOperandValue(
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
  ORT_RETURN_IF_ERROR(AddNewOperand(name, operand_type, false /* is_nhwc */, index));
  const size_t size = operand_type.GetOperandBlobByteSize();

  // for small size operand, the value will be copied
  // no need to persist
  if (size < ANEURALNETWORKS_MAX_SIZE_OF_IMMEDIATELY_COPIED_VALUES) {
    RETURN_STATUS_ON_ERROR(
        nnapi_->ANeuralNetworksModel_setOperandValue(
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
  for (size_t i = 0; i < node_indices.size(); i++) {
    const auto* node(graph_viewer_.GetNode(node_indices[i]));
    if (const auto* op_builder = GetOpBuilder(*node)) {
      ORT_RETURN_IF_ERROR(op_builder->AddToModelBuilder(*this, *node));
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Node [", node->Name(), "], type [", node->OpType(), "] is not supported");
    }
  }

  return Status::OK();
}

Status ModelBuilder::AddOperation(int op, const std::vector<uint32_t>& input_indices,
                                  const std::vector<std::string>& output_names,
                                  const std::vector<OperandType>& types,
                                  const std::vector<bool>& is_nhwc_vec) {
  std::vector<uint32_t> output_indices;
  for (size_t i = 0; i < types.size(); i++) {
    uint32_t index = 0;
    ORT_RETURN_IF_ERROR(AddNewOperand(output_names[i], types[i], is_nhwc_vec[i], index));
    output_indices.push_back(index);
  }

  RETURN_STATUS_ON_ERROR_WITH_NOTE(
      nnapi_->ANeuralNetworksModel_addOperation(
          nnapi_model_->model_, op, input_indices.size(), &input_indices[0],
          output_indices.size(), &output_indices[0]),
      "op = " + std::to_string(op));

  num_nnapi_ops_++;
  return Status::OK();
}

Status ModelBuilder::Compile(std::unique_ptr<Model>& model) {
  ORT_RETURN_IF_ERROR(Prepare());

  RETURN_STATUS_ON_ERROR_WITH_NOTE(
      nnapi_->ANeuralNetworksModel_identifyInputsAndOutputs(
          nnapi_model_->model_, static_cast<uint32_t>(input_index_vec_.size()),
          &input_index_vec_[0],
          static_cast<uint32_t>(output_index_vec_.size()),
          &output_index_vec_[0]),
      "on identifyInputsAndOutputs");

  // relax fp32tofp16 is only available on API 28+
  if (use_fp16_ && GetAndroidSdkVer() > 27) {
    RETURN_STATUS_ON_ERROR_WITH_NOTE(
        nnapi_->ANeuralNetworksModel_relaxComputationFloat32toFloat16(
            nnapi_model_->model_, true),
        "Set fp16");
  }

  RETURN_STATUS_ON_ERROR_WITH_NOTE(
      nnapi_->ANeuralNetworksModel_finish(nnapi_model_->model_),
      "on model finish");

  // We have a list of target devices, try to see if the model can be run entirely
  // using the list of target devices
  // This is only available on API 29+, for API 28- the nnapi_target_devices_ will
  // be empty so we will not check API level here, see GetTargetDevices()
  bool use_create_for_devices = false;
  if (!nnapi_target_devices_.empty()) {
    std::unique_ptr<bool[]> supported_ops_holder = std::make_unique<bool[]>(num_nnapi_ops_);
    auto* supported_ops = supported_ops_holder.get();
    RETURN_STATUS_ON_ERROR_WITH_NOTE(
        nnapi_->ANeuralNetworksModel_getSupportedOperationsForDevices(
            nnapi_model_->model_, nnapi_target_devices_.data(),
            nnapi_target_devices_.size(), supported_ops),
        "on getSupportedOperationsForDevices");

    bool all_ops_supported = std::all_of(supported_ops, supported_ops + num_nnapi_ops_,
                                         [](bool is_supported) { return is_supported; });
    if (!all_ops_supported) {
      // There are some ops not supported by the list of the target devices
      // Fail the Compile
      //
      // TODO, add some logic to not fail for some cases
      // Such as, if there are some acceptable fall back to cpu (nnapi-reference)
      // and cpu is not in the target devices list
      return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                             "The model cannot run using current set of target devices, ",
                             nnapi_target_devices_detail_);
    } else {
      use_create_for_devices = true;
    }
  }

  if (use_create_for_devices) {
    RETURN_STATUS_ON_ERROR_WITH_NOTE(
        nnapi_->ANeuralNetworksCompilation_createForDevices(
            nnapi_model_->model_, nnapi_target_devices_.data(),
            nnapi_target_devices_.size(), &nnapi_model_->compilation_),
        "on createForDevices");
  } else {
    RETURN_STATUS_ON_ERROR_WITH_NOTE(
        nnapi_->ANeuralNetworksCompilation_create(nnapi_model_->model_, &nnapi_model_->compilation_),
        "on create");
  }

  RETURN_STATUS_ON_ERROR_WITH_NOTE(
      nnapi_->ANeuralNetworksCompilation_setPreference(
          nnapi_model_->compilation_, static_cast<int32_t>(exe_pref_)),
      "on setPreference");

  RETURN_STATUS_ON_ERROR_WITH_NOTE(
      nnapi_->ANeuralNetworksCompilation_finish(nnapi_model_->compilation_),
      "on compilation finish");

  model.reset(nnapi_model_.release());
  return Status::OK();
}

int32_t ModelBuilder::FindActivation(const Node& node, const NodeArg& output) {
  int32_t fuse_code = ANEURALNETWORKS_FUSED_NONE;

  // We do not support activation fusion for quantized operators for now
  auto qlinear_op_type = GetQLinearOpType(node);
  if (qlinear_op_type != QLinearOpType::Unknown)
    return fuse_code;

  for (auto it = node.OutputEdgesBegin(), end = node.OutputEdgesEnd(); it != end; ++it) {
    const auto& dst_node = it->GetNode();
    const auto* dst_input = dst_node.InputDefs()[it->GetDstArgIndex()];
    if (Contains(activation_nodes_, dst_node.Index())) {
      if (&output == dst_input) {
        fuse_code = activation_nodes_.at(dst_node.Index());
      }
    } else {
      // if there is any other non-relu node using the output
      // will add relu separately
      if (&output == dst_input)
        return ANEURALNETWORKS_FUSED_NONE;
    }
  }

  // if output is a graph output, will add relu separately
  if (fuse_code != ANEURALNETWORKS_FUSED_NONE) {
    for (const auto* graph_output : graph_viewer_.GetOutputs()) {
      if (&output == graph_output)
        return ANEURALNETWORKS_FUSED_NONE;
    }

    LOGS_DEFAULT(VERBOSE) << "Node [" << node.Name() << "] type [" << node.OpType()
                          << "], fused the output [" << output.Name() << "]";

    fused_activations_.insert(output.Name());
  }

  return fuse_code;
}

/* static */ const IOpBuilder* ModelBuilder::GetOpBuilder(const Node& node) {
  const auto& op_builders = GetOpBuilders();
  if (!Contains(op_builders, node.OpType()))
    return nullptr;

  return op_builders.at(node.OpType());
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

void ModelBuilder::RegisterNHWCOperand(const std::string& name) {
  nhwc_operands_.insert(name);
}

bool ModelBuilder::IsOperandNHWC(const std::string& name) const {
  return Contains(nhwc_operands_, name);
}

bool ModelBuilder::GetNCHWOperand(const std::string& nhwc_name, std::string& nchw_name) {
  if (Contains(nhwc_to_nchw_map_, nhwc_name)) {
    nchw_name = nhwc_to_nchw_map_[nhwc_name];
    return true;
  }
  return false;
}

bool ModelBuilder::GetNHWCOperand(const std::string& nchw_name, std::string& nhwc_name) {
  if (Contains(nchw_to_nhwc_map_, nchw_name)) {
    nhwc_name = nchw_to_nhwc_map_[nchw_name];
    return true;
  }
  return false;
}

Status ModelBuilder::SetNHWCToNCHWOperandMap(const std::string& nhwc_name,
                                             const std::string& nchw_name) {
  ORT_RETURN_IF_NOT(!Contains(nhwc_to_nchw_map_, nhwc_name), "A previous nchw to nhwc map exists");
  nhwc_to_nchw_map_[nhwc_name] = nchw_name;
  return Status::OK();
}

Status ModelBuilder::SetNCHWToNHWCOperandMap(const std::string& nchw_name,
                                             const std::string& nhwc_name) {
  ORT_RETURN_IF_NOT(!Contains(nchw_to_nhwc_map_, nchw_name), "A previous nchw to nhwc map exists");
  nchw_to_nhwc_map_[nchw_name] = nhwc_name;
  return Status::OK();
}

}  // namespace nnapi
}  // namespace onnxruntime