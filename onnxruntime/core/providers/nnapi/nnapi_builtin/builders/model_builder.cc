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

ModelBuilder::ModelBuilder(const GraphViewer& graph_view)
    : nnapi_(NnApiImplementation()), graph_view_(graph_view) {
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

bool IsValidSupportedNodesVec(const std::vector<int>& supported_node_vec, const GraphViewer& graph_view) {
  if (!supported_node_vec.empty()) {
    if (supported_node_vec.size() == 1) {
      const auto& node_indices = graph_view.GetNodesInTopologicalOrder();
      const auto* node(graph_view.GetNode(node_indices[supported_node_vec[0]]));
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
  const auto& node_indices = graph_view_.GetNodesInTopologicalOrder();
  for (size_t i = 0; i < node_indices.size(); i++) {
    const auto* node(graph_view_.GetNode(node_indices[i]));
    bool supported = IsNodeSupported(*node);
    LOGS_DEFAULT(VERBOSE) << "Operator type: [" << node->OpType()
                          << "] index: [" << i
                          << "] name: [" << node->Name()
                          << "] supported: [" << supported
                          << "]";
    if (supported) {
      supported_node_vec.push_back(i);
    } else {
      if (IsValidSupportedNodesVec(supported_node_vec, graph_view_)) {
        supported_node_vecs.push_back(supported_node_vec);
        supported_node_vec.clear();
      }
    }
  }

  if (IsValidSupportedNodesVec(supported_node_vec, graph_view_))
    supported_node_vecs.push_back(supported_node_vec);

  LOGS_DEFAULT(VERBOSE) << "Support vectors size is " << supported_node_vecs.size();
  for (const auto& group : supported_node_vecs)
    LOGS_DEFAULT(VERBOSE) << "Support vector size is " << group.size();

  return supported_node_vecs;
}

// Scalar operand is copied into the model, no need to persist
#define DEFINE_ADD_OPERAND_FROM_SCALAR(scalar_type, op_type)       \
  uint32_t ModelBuilder::AddOperandFromScalar(scalar_type value) { \
    OperandType operandType(Type::op_type);                        \
    auto index = AddNewNNAPIOperand(operandType);                  \
    THROW_ON_ERROR_WITH_NOTE(                                      \
        nnapi_->ANeuralNetworksModel_setOperandValue(              \
            nnapi_model_->model_, index, &value, sizeof(value)),   \
        "value: " + std::to_string(value));                        \
    return index;                                                  \
  }

DEFINE_ADD_OPERAND_FROM_SCALAR(bool, BOOL);
DEFINE_ADD_OPERAND_FROM_SCALAR(int32_t, INT32);
DEFINE_ADD_OPERAND_FROM_SCALAR(float, FLOAT32);

#undef DEFINE_ADD_OPERAND_FROM_SCALAR

void ModelBuilder::AddInitializerToSkip(const std::string& tensor_name) {
  skipped_initializers_.insert(tensor_name);
}

void ModelBuilder::Prepare() {
  nnapi_model_ = std::unique_ptr<Model>(new Model());
  THROW_ON_ERROR(nnapi_->ANeuralNetworksModel_create(&nnapi_model_->model_));
  GetTargetDevices();
  PreprocessInitializers();
  RegisterInitializers();
  RegisterModelInputs();
  AddOperations();
  RegisterModelOutputs();
  RegisterModelShaper();
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

void ModelBuilder::GetTargetDevices() {
  // GetTargetDevices is only supported on API 29+
  if (GetAndroidSdkVer() < 29)
    return;

  if (target_device_option_ == TargetDeviceOption::ALL_DEVICES)
    return;

  const std::string nnapi_cpu("nnapi-reference");
  uint32_t num_devices = 0;
  THROW_ON_ERROR_WITH_NOTE(nnapi_->ANeuralNetworks_getDeviceCount(&num_devices),
                           "Getting count of available devices");

  for (uint32_t i = 0; i < num_devices; i++) {
    ANeuralNetworksDevice* device = nullptr;
    const char* device_name = nullptr;
    THROW_ON_ERROR_WITH_NOTE(nnapi_->ANeuralNetworks_getDevice(i, &device),
                             "Getting " + std::to_string(i) + "th device");

    THROW_ON_ERROR_WITH_NOTE(nnapi_->ANeuralNetworksDevice_getName(device, &device_name),
                             "Getting " + std::to_string(i) + "th device's name");

    bool device_is_cpu = nnapi_cpu == device_name;
    if ((target_device_option_ == TargetDeviceOption::CPU_DISABLED && !device_is_cpu) ||
        (target_device_option_ == TargetDeviceOption::CPU_ONLY && device_is_cpu)) {
      nnapi_target_devices_.push_back(device);
      LOGS_DEFAULT(VERBOSE) << "Target device [" << device_name << "] added";
    }
  }
}

void ModelBuilder::GetAllInitializers() {
  for (const auto& pair : graph_view_.GetAllInitializedTensors()) {
    initializers_.emplace(pair.first, *pair.second);
  }
}

void ModelBuilder::PreprocessInitializers() {
  const auto& node_indices = graph_view_.GetNodesInTopologicalOrder();
  for (size_t i = 0; i < node_indices.size(); i++) {
    const auto* node(graph_view_.GetNode(node_indices[i]));
    if (auto* op_builder = GetOpBuilder(*node)) {
      op_builder->AddInitializersToSkip(*this, *node);
    }
  }
}

void ModelBuilder::RegisterInitializers() {
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

    Type type = Type::TENSOR_FLOAT32;
    switch (tensor.data_type()) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
        type = Type::TENSOR_FLOAT32;
        break;
      default:
        // TODO: support other type
        ORT_THROW("The initializer of graph doesn't have valid type, name: " +
                  name + " type: " + std::to_string(tensor.data_type()));
        break;
    }

    OperandType operand_type(type, shape);
    shaper_.AddShape(name, operand_type.dimensions);

    auto index = AddNewOperand(name, operand_type, false /* is_nhwc */);
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
    SetOperandValue(index, nnapi_model_->mem_initializers_.get(), size, offset);
    offset += padded_size;
  }
}

void ModelBuilder::RegisterModelInputs() {
  for (const auto* node_arg : graph_view_.GetInputs()) {
    const auto& input_name = node_arg->Name();

    {  // input should not be an initializer
      if (Contains(operands_, input_name))
        continue;

      if (Contains(initializers_, input_name))
        continue;
    }

    const auto* shape_proto = node_arg->Shape();
    ORT_ENFORCE(shape_proto != nullptr, "shape_proto cannot be null for input: " + input_name);
    Shaper::Shape shape;

    for (const auto& dim : shape_proto->dim()) {
      // NNAPI uses 0 for dynamic dimension, which is the default value for dim.dim_value()
      shape.push_back(SafeInt<uint32_t>(dim.dim_value()));
    }

    Type type = Type::TENSOR_FLOAT32;
    const auto* type_proto = node_arg->TypeAsProto();
    if (!type_proto || !type_proto->tensor_type().has_elem_type()) {
      ORT_THROW("The input of graph doesn't have elem_type: " + input_name);
    } else {
      switch (type_proto->tensor_type().elem_type()) {
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
          type = Type::TENSOR_FLOAT32;
          break;
        default:
          // TODO: support other type
          ORT_THROW("The input of graph doesn't have valid type, name: " +
                    input_name + " type: " +
                    std::to_string(type_proto->tensor_type().elem_type()));
      }
    }

    OperandType operand_type(type, shape);
    shaper_.AddShape(input_name, operand_type.dimensions);

    auto index = AddNewOperand(input_name, operand_type, false /* is_nhwc */);
    input_index_vec_.push_back(index);
    nnapi_model_->AddInput(input_name, operand_type);
  }
}

void ModelBuilder::RegisterModelOutputs() {
  for (const auto* node_arg : graph_view_.GetOutputs()) {
    const auto& output_name = node_arg->Name();

    if (!Contains(operands_, output_name)) {
      ORT_THROW("The output of graph is not registered" + output_name);
    }

    std::string nnapi_output_name = output_name;
    if (IsOperandNHWC(output_name)) {
      // We need to transpose the output still in nhwc back to nchw
      nnapi_output_name = GetUniqueName(output_name + "_nhwc_to_nchw");
      TransposeNHWCToNCHW(*this, output_name, nnapi_output_name);
    }

    output_index_vec_.push_back(operand_indices_[nnapi_output_name]);
    nnapi_model_->AddOutput(output_name, nnapi_output_name, operand_types_.at(nnapi_output_name));
  }
}

void ModelBuilder::RegisterModelShaper() {
  shaper_.Finalize();
  nnapi_model_->SetShaper(shaper_);
}

uint32_t ModelBuilder::AddNewOperand(const std::string& name,
                                     const OperandType& operand_type,
                                     bool is_nhwc) {
  auto idx = AddNewNNAPIOperand(operand_type);
  RegisterOperand(name, idx, operand_type, is_nhwc);
  return idx;
}

uint32_t ModelBuilder::AddNewNNAPIOperand(const OperandType& operand_type) {
  THROW_ON_ERROR(nnapi_->ANeuralNetworksModel_addOperand(
      nnapi_model_->model_, &operand_type.operandType));
  return next_index_++;
}

void ModelBuilder::RegisterOperand(const std::string& name, uint32_t index,
                                   const OperandType& operand_type, bool is_nhwc) {
  operand_indices_[name] = index;
  operand_types_.emplace(name, operand_type);
  operands_.insert(name);

  if (is_nhwc)
    RegisterNHWCOperand(name);
}

void ModelBuilder::SetOperandValue(uint32_t index,
                                   Model::NNMemory* memory,
                                   size_t size, size_t offset) {
#ifdef USENNAPISHAREDMEM
  THROW_ON_ERROR(
      nnapi_->ANeuralNetworksModel_setOperandValueFromMemory(
          nnapi_model_->model_, index,
          memory->GetHandle(),
          offset, size));
#else
  THROW_ON_ERROR(
      nnapi_->ANeuralNetworksModel_setOperandValue(
          nnapi_model_->model_, index,
          memory->GetDataPtr() + offset,
          size));
#endif
}

uint32_t ModelBuilder::AddOperandFromPersistMemoryBuffer(
    const std::string& name, const void* buffer,
    const android::nn::wrapper::OperandType& operand_type) {
  shaper_.AddShape(name, operand_type.dimensions);
  auto index = AddNewOperand(name, operand_type, false /* is_nhwc */);
  const size_t size = operand_type.GetOperandBlobByteSize();

  // for small size operand, the value will be copied
  // no need to persist
  if (size < ANEURALNETWORKS_MAX_SIZE_OF_IMMEDIATELY_COPIED_VALUES) {
    THROW_ON_ERROR(
        nnapi_->ANeuralNetworksModel_setOperandValue(
            nnapi_model_->model_, index,
            buffer, size));
  } else {
    const size_t padded_size = GetPaddedByteSize(size);
    auto persist_buffer = std::make_unique<Model::NNMemory>(nnapi_, name.c_str(), padded_size);
    uint8_t* dest = persist_buffer->GetDataPtr();
    memcpy(dest, buffer, size);
    SetOperandValue(index, persist_buffer.get(), size, 0);
    nnapi_model_->mem_persist_buffers_.push_back(std::move(persist_buffer));
  }

  return index;
}

void ModelBuilder::AddOperations() {
  const auto& node_indices = graph_view_.GetNodesInTopologicalOrder();
  for (size_t i = 0; i < node_indices.size(); i++) {
    const auto* node(graph_view_.GetNode(node_indices[i]));
    if (auto* op_builder = GetOpBuilder(*node)) {
      op_builder->AddToModelBuilder(*this, *node);
    } else {
      ORT_THROW("Node [" + node->Name() + "], type [" + node->OpType() + "] is not supported");
    }
  }
}

void ModelBuilder::AddOperation(int op, const std::vector<uint32_t>& input_indices,
                                const std::vector<std::string>& output_names,
                                const std::vector<OperandType>& types,
                                const std::vector<bool>& is_nhwc_vec) {
  std::vector<uint32_t> output_indices;
  for (size_t i = 0; i < types.size(); i++) {
    output_indices.push_back(AddNewOperand(output_names[i], types[i], is_nhwc_vec[i]));
  }

  THROW_ON_ERROR_WITH_NOTE(
      nnapi_->ANeuralNetworksModel_addOperation(
          nnapi_model_->model_, op, input_indices.size(), &input_indices[0],
          output_indices.size(), &output_indices[0]),
      "op = " + std::to_string(op));
}

std::unique_ptr<Model> ModelBuilder::Compile() {
  Prepare();

  THROW_ON_ERROR_WITH_NOTE(
      nnapi_->ANeuralNetworksModel_identifyInputsAndOutputs(
          nnapi_model_->model_, static_cast<uint32_t>(input_index_vec_.size()),
          &input_index_vec_[0],
          static_cast<uint32_t>(output_index_vec_.size()),
          &output_index_vec_[0]),
      "on identifyInputsAndOutputs");

  // relax fp32tofp16 is only available on API 28+
  if (use_fp16_ && GetAndroidSdkVer() > 27) {
    THROW_ON_ERROR_WITH_NOTE(
        nnapi_->ANeuralNetworksModel_relaxComputationFloat32toFloat16(
            nnapi_model_->model_, true),
        "Set fp16");
  }

  THROW_ON_ERROR_WITH_NOTE(
      nnapi_->ANeuralNetworksModel_finish(nnapi_model_->model_),
      "on model finish");

  if (!nnapi_target_devices_.empty()) {
    THROW_ON_ERROR_WITH_NOTE(
        nnapi_->ANeuralNetworksCompilation_createForDevices(
            nnapi_model_->model_, nnapi_target_devices_.data(),
            nnapi_target_devices_.size(), &nnapi_model_->compilation_),
        "on createForDevices");
  } else {
    THROW_ON_ERROR_WITH_NOTE(
        nnapi_->ANeuralNetworksCompilation_create(nnapi_model_->model_, &nnapi_model_->compilation_),
        "on create");
  }

  THROW_ON_ERROR_WITH_NOTE(
      nnapi_->ANeuralNetworksCompilation_setPreference(
          nnapi_model_->compilation_, static_cast<int32_t>(exe_pref_)),
      "on setPreference");

  THROW_ON_ERROR_WITH_NOTE(
      nnapi_->ANeuralNetworksCompilation_finish(nnapi_model_->compilation_),
      "on compilation finish");

  return std::move(nnapi_model_);
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
    for (const auto* graph_output : graph_view_.GetOutputs()) {
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

void ModelBuilder::SetNHWCToNCHWOperandMap(const std::string& nhwc_name,
                                           const std::string& nchw_name) {
  ORT_ENFORCE(!Contains(nhwc_to_nchw_map_, nhwc_name), "A previous nchw to nhwc map exists");
  nhwc_to_nchw_map_[nhwc_name] = nchw_name;
}

void ModelBuilder::SetNCHWToNHWCOperandMap(const std::string& nchw_name,
                                           const std::string& nhwc_name) {
  ORT_ENFORCE(!Contains(nchw_to_nhwc_map_, nchw_name), "A previous nchw to nhwc map exists");
  nchw_to_nhwc_map_[nchw_name] = nhwc_name;
}

}  // namespace nnapi
}  // namespace onnxruntime