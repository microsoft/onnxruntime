// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/common/logging/logging.h>
#include <core/common/safeint.h>

#include "core/providers/nnapi/nnapi_builtin/nnapi_lib/nnapi_implementation.h"
#include "helper.h"
#include "model_builder.h"
#include "op_builder.h"

namespace onnxruntime {
namespace nnapi {

using namespace android::nn::wrapper;
using std::vector;

ModelBuilder::ModelBuilder(const GraphViewer& graph_viewer)
    : nnapi_(NnApiImplementation()), graph_viewer_(graph_viewer) {
  GetAllInitializers();
  op_builders_ = CreateOpBuilders();
}

int32_t ModelBuilder::GetAndroidSdkVer() const {
  return nnapi_ ? nnapi_->android_sdk_version : 0;
}

bool ModelBuilder::IsNodeSupported(const Node& node) {
  if (auto* op_builder = GetOpBuilder(node)) {
    return op_builder->IsOpSupported(*this, node);
  } else {
    return false;
  }
}

bool IsValidSupportedNodesVec(const std::vector<int>& supported_node_vec, const GraphViewer& graph_viewer) {
  if (!supported_node_vec.empty()) {
    if (supported_node_vec.size() == 1) {
      const auto& node_indices = graph_viewer.GetNodesInTopologicalOrder();
      const auto* node(graph_viewer.GetNode(node_indices[supported_node_vec[0]]));
      const auto& op = node->OpType();
      // It is not worth it to perform a single Reshape/Dropout/Identity operator
      // which is only copying the data in NNAPI
      // If this is the case, let it fall back
      if (op == "Reshape" ||
          op == "Dropout" ||
          op == "Identity") {
        return false;
      }
    }
    return true;
  }
  return false;
}

std::vector<std::vector<int>> ModelBuilder::GetSupportedNodes() {
  std::vector<std::vector<int>> supported_node_vecs;
  int32_t android_sdk_ver = GetAndroidSdkVer();
#ifdef __ANDROID__
  if (android_sdk_ver < 27) {
    LOGS_DEFAULT(VERBOSE) << "Android API level "
                          << android_sdk_ver
                          << " is lower than 27";
    return supported_node_vecs;
  }
#endif

  std::vector<int> supported_node_vec;
  const auto& node_indices = graph_viewer_.GetNodesInTopologicalOrder();
  for (size_t i = 0; i < node_indices.size(); i++) {
    const auto* node(graph_viewer_.GetNode(node_indices[i]));
    bool supported = IsNodeSupported(*node);
    LOGS_DEFAULT(VERBOSE) << "Operator type: [" << node->OpType()
                          << "] index: [" << i
                          << "] name: [" << node->Name()
                          << "] supported: [" << supported
                          << "]";
    if (supported) {
      supported_node_vec.push_back(i);
    } else {
      if (IsValidSupportedNodesVec(supported_node_vec, graph_viewer_)) {
        supported_node_vecs.push_back(supported_node_vec);
        supported_node_vec.clear();
      }
    }
  }

  if (IsValidSupportedNodesVec(supported_node_vec, graph_viewer_))
    supported_node_vecs.push_back(supported_node_vec);

  LOGS_DEFAULT(VERBOSE) << "Support vectors size is " << supported_node_vecs.size();
  for (const auto& group : supported_node_vecs)
    LOGS_DEFAULT(VERBOSE) << "Support vector size is " << group.size();

  return supported_node_vecs;
}

// Scalar operand is copied into the model, no need to persist
#define DEFINE_ADD_OPERAND_FROM_SCALAR(scalar_type, op_type)                      \
  Status ModelBuilder::AddOperandFromScalar(scalar_type value, uint32_t& index) { \
    OperandType operandType(Type::op_type);                                       \
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

Status ModelBuilder::Prepare() {
  nnapi_model_ = std::unique_ptr<Model>(new Model());
  RETURN_STATUS_ON_ERROR(nnapi_->ANeuralNetworksModel_create(&nnapi_model_->model_));
  ORT_RETURN_IF_ERROR(GetTargetDevices());
  PreprocessInitializers();
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
    RETURN_STATUS_ON_ERROR_WITH_NOTE(
        nnapi_->ANeuralNetworks_getDevice(i, &device), "Getting " + std::to_string(i) + "th device");

    RETURN_STATUS_ON_ERROR_WITH_NOTE(nnapi_->ANeuralNetworksDevice_getName(device, &device_name),
                                     "Getting " + std::to_string(i) + "th device's name");

    bool device_is_cpu = nnapi_cpu == device_name;
    if ((target_device_option_ == TargetDeviceOption::CPU_DISABLED && !device_is_cpu) ||
        (target_device_option_ == TargetDeviceOption::CPU_ONLY && device_is_cpu)) {
      nnapi_target_devices_.push_back(device);
      LOGS_DEFAULT(VERBOSE) << "Target device [" << device_name << "] added";
    }
  }

  return Status::OK();
}

void ModelBuilder::GetAllInitializers() {
  for (const auto& pair : graph_viewer_.GetAllInitializedTensors()) {
    initializers_.emplace(pair.first, *pair.second);
  }
}

void ModelBuilder::PreprocessInitializers() {
  const auto& node_indices = graph_viewer_.GetNodesInTopologicalOrder();
  for (size_t i = 0; i < node_indices.size(); i++) {
    const auto* node(graph_viewer_.GetNode(node_indices[i]));
    if (auto* op_builder = GetOpBuilder(*node)) {
      op_builder->AddInitializersToSkip(*this, *node);
    }
  }
}

// Help to get all quantized operators' input and the node(s) using the input
std::unordered_map<std::string, vector<const Node*>> GetAllQuantizedOpInputs(const GraphViewer& graph_viewer) {
  std::unordered_map<std::string, vector<const Node*>> all_quantized_op_inputs;
  const auto& node_indices = graph_viewer.GetNodesInTopologicalOrder();
  for (const auto& node_idx : node_indices) {
    const auto* node(graph_viewer.GetNode(node_idx));
    auto qlinear_op_type = GetQLinearOpType(*node);
    if (qlinear_op_type == QLinearOpType::DequantizeLinear || IsQLinearBinaryOp(qlinear_op_type)) {
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

Status ModelBuilder::RegisterInitializers() {
  // First pass to get all the stats of the initializers
  auto initializer_size = initializers_.size();
  std::vector<std::tuple<uint32_t, size_t, size_t>> initializers(initializer_size);
  size_t sizeAll = 0;

  int i = 0;
  for (const auto& pair : initializers_) {
    const auto& tensor = pair.second;
    const auto& name = tensor.name();
    if (Contains(skipped_initializers_, name))
      continue;

    Shape shape;
    for (auto dim : tensor.dims()) {
      shape.push_back(SafeInt<uint32_t>(dim));
    }

    ORT_RETURN_IF_NOT(!shape.empty(), "NNAPI does not support scalar initializer, tensor name, ", name);

    Type type = Type::TENSOR_FLOAT32;
    switch (tensor.data_type()) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
        type = Type::TENSOR_FLOAT32;
        break;
      default:
        // TODO: support other type
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "The initializer of graph doesn't have valid type, name: ",
                               name, " type: ", tensor.data_type());
        break;
    }

    OperandType operand_type(type, shape);
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
  for (const auto& pair : initializers_) {
    const auto& tensor = pair.second;
    if (Contains(skipped_initializers_, tensor.name()))
      continue;

    uint32_t index;
    size_t size, padded_size;
    std::tie(index, size, padded_size) = initializers[i++];
    const char* src = nullptr;
    if (tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      src = tensor.float_data().empty()
                ? tensor.raw_data().data()
                : reinterpret_cast<const char*>(tensor.float_data().data());
    }  // We should not get anything else here since we already checked in the 1st pass

    uint8_t* dest = nnapi_model_->mem_initializers_->GetDataPtr() + offset;
    memcpy(dest, src, size);
    ORT_RETURN_IF_ERROR(SetOperandValue(index, nnapi_model_->mem_initializers_.get(), size, offset));
    offset += padded_size;
  }

  return Status::OK();
}

Status ModelBuilder::RegisterModelInputs() {
  const auto all_quantized_op_inputs = GetAllQuantizedOpInputs(graph_viewer_);
  for (const auto* node_arg : graph_viewer_.GetInputs()) {
    const auto& input_name = node_arg->Name();

    {  // input should not be an initializer
      if (Contains(operands_, input_name))
        continue;

      if (Contains(initializers_, input_name))
        continue;
    }

    const auto* shape_proto = node_arg->Shape();
    ORT_RETURN_IF_NOT(shape_proto != nullptr, "shape_proto cannot be null for input: ", input_name);
    Shaper::Shape shape;

    for (const auto& dim : shape_proto->dim()) {
      // NNAPI uses 0 for dynamic dimension, which is the default value for dim.dim_value()
      shape.push_back(SafeInt<uint32_t>(dim.dim_value()));
    }

    ORT_RETURN_IF_NOT(GetAndroidSdkVer() >= 29 || !shape.empty(),
                      "0-rank input is only supported on Android API level 29+");

    Type type = Type::TENSOR_FLOAT32;
    float scale = 0.0f;
    int32_t zero_point = 0;
    const auto* type_proto = node_arg->TypeAsProto();
    if (!type_proto || !type_proto->tensor_type().has_elem_type()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "The input of graph doesn't have elem_type: ", input_name);
    } else {
      switch (type_proto->tensor_type().elem_type()) {
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
          type = Type::TENSOR_FLOAT32;
          break;
        case ONNX_NAMESPACE::TensorProto_DataType_UINT8: {
          // For ONNX the quantized input does not carry scale and zero point info
          // So we will need to search the operator using this input
          // And dig out the scale and zero point as the input initializers to the operator
          type = Type::TENSOR_QUANT8_ASYMM;
          if (!Contains(all_quantized_op_inputs, input_name)) {
            // We current do not support uint8 input if it is not a quantized input
            return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                   "The input of graph doesn't have valid type, name: ", input_name,
                                   " type: ", type_proto->tensor_type().elem_type());
          }

          // TODO, verify the scale and zero point match if there are multiple op using same input
          ORT_RETURN_IF_ERROR(GetQuantizedInputScaleAndZeroPoint(
              *this, *all_quantized_op_inputs.at(input_name)[0], input_name, scale, zero_point));
          break;
        }
        default: {
          // TODO: support other type
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "The input of graph doesn't have valid type, name: ", input_name,
                                 " type: ", type_proto->tensor_type().elem_type());
        }
      }
    }

    OperandType operand_type(type, shape, scale, zero_point);
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
  ORT_RETURN_IF_ERROR(AddNewNNAPIOperand(operand_type, index));
  RegisterOperand(name, index, operand_type, is_nhwc);
  return Status::OK();
}

Status ModelBuilder::AddNewNNAPIOperand(const OperandType& operand_type, uint32_t& index) {
  RETURN_STATUS_ON_ERROR(
      nnapi_->ANeuralNetworksModel_addOperand(nnapi_model_->model_, &operand_type.operandType));
  index = next_index_++;
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
    if (auto* op_builder = GetOpBuilder(*node)) {
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

  if (!nnapi_target_devices_.empty()) {
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
  for (auto it = node.OutputEdgesBegin(), end = node.OutputEdgesEnd(); it != end; ++it) {
    const auto& dst_node = it->GetNode();
    const auto* dst_input = dst_node.InputDefs()[it->GetDstArgIndex()];
    if (dst_node.OpType() == "Relu") {
      if (&output == dst_input) {
        fuse_code = ANEURALNETWORKS_FUSED_RELU;
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

IOpBuilder* ModelBuilder::GetOpBuilder(const Node& node) {
  if (!Contains(op_builders_, node.OpType()))
    return nullptr;

  return op_builders_[node.OpType()].get();
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

bool ModelBuilder::IsOperandNHWC(const std::string& name) {
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