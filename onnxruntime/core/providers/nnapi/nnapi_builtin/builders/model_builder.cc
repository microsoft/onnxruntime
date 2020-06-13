// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nnapi/nnapi_builtin/nnapi_lib/nnapi_implementation.h"
#include "helper.h"
#include "model_builder.h"
#include "NodeAttrHelper.h"
#include "OpBuilder.h"

namespace onnxruntime {
namespace nnapi {

using namespace android::nn::wrapper;
using std::vector;

const float* GetTensorFloatDataA(const ONNX_NAMESPACE::TensorProto& tensor) {
  return tensor.float_data().empty()
             ? reinterpret_cast<const float*>(tensor.raw_data().data())
             : tensor.float_data().data();
}

ModelBuilder::ModelBuilder(ONNX_NAMESPACE::ModelProto& model_proto)
    : nnapi_(NnApiImplementation()), model_proto_(model_proto) {}

int32_t ModelBuilder::GetAndroidSdkVer() const {
  return nnapi_ ? nnapi_->android_sdk_version : 0;
}

std::pair<bool, std::string> ModelBuilder::IsNodeSupported(
    const ONNX_NAMESPACE::NodeProto& node) {
  auto opBuilder = CreateOpBuilder(*this, node);
  return opBuilder->IsOpSupported();
}

bool IsValidSupportedNodesVec(const std::vector<int>& supported_node_vec,
                              const ONNX_NAMESPACE::ModelProto& model_proto) {
  if (!supported_node_vec.empty()) {
    if (supported_node_vec.size() == 1) {
      const auto& node = model_proto.graph().node(supported_node_vec[0]);
      // It is not worth it to perform a single Reshape/Dropout/Identity operator
      // which is only copying the data in NNAPI
      // If this is the case, let it fall back
      if (node.op_type() == "Reshape" || node.op_type() == "Dropout" ||
          node.op_type() == "Identity") {
        return false;
      }
    }
    return true;
  }
  return false;
}

std::vector<std::vector<int>> ModelBuilder::GetSupportedNodes() {
  std::vector<std::vector<int>> supported_node_vecs;
  int32_t android_sdk_ver = nnapi_ ? nnapi_->android_sdk_version : 0;
#ifdef __ANDROID__
  if (android_sdk_ver < 27) {
    LOGI("Android API level %d is lower than 27", android_sdk_ver);
    return supported_node_vecs;
  }
#endif

  GetAllIntializers();
  std::vector<int> supported_node_vec;
  for (int i = 0; i < model_proto_.graph().node_size(); i++) {
    bool supported;
    std::string error_msg;
    std::tie(supported, error_msg) = IsNodeSupported(model_proto_.graph().node(i));
    LOGV("Node %s, name %s, supported %d, message: %s",
         model_proto_.graph().node(i).op_type().c_str(),
         model_proto_.graph().node(i).name().c_str(),
         supported, error_msg.c_str());
    if (supported) {
      supported_node_vec.push_back(i);
    } else {
      if (IsValidSupportedNodesVec(supported_node_vec, model_proto_)) {
        supported_node_vecs.push_back(supported_node_vec);
        supported_node_vec.clear();
      }
    }
  }

  if (IsValidSupportedNodesVec(supported_node_vec, model_proto_))
    supported_node_vecs.push_back(supported_node_vec);

  return supported_node_vecs;
}

#define DEFINE_ADD_OPERAND_FROM_SCALAR(scalar_type, op_type)                  \
  ModelBuilder::Index ModelBuilder::AddOperandFromScalar(scalar_type value) { \
    OperandType operandType(Type::op_type);                                   \
    auto index = AddNewOperand(operandType);                                  \
    THROW_ON_ERROR_WITH_NOTE(                                                 \
        nnapi_->ANeuralNetworksModel_setOperandValue(                         \
            nnapi_model_->model_, index, &value, sizeof(value)),              \
        "value: " + std::to_string(value));                                   \
    return index;                                                             \
  }

DEFINE_ADD_OPERAND_FROM_SCALAR(bool, BOOL);
DEFINE_ADD_OPERAND_FROM_SCALAR(int32_t, INT32);
DEFINE_ADD_OPERAND_FROM_SCALAR(float, FLOAT32);

#undef DEFINE_ADD_OPERAND_FROM_SCALAR

void ModelBuilder::AddSkippedInitializer(const std::string& tensor_name) {
  skipped_initializers_.insert(tensor_name);
}

// Scalar operand is copied into the model, no need to persist
ModelBuilder::Index ModelBuilder::SetOperandFromScalar(Type type, const void* value, size_t size) {
  OperandType operandType(type);
  auto index = AddNewOperand(operandType);
  THROW_ON_ERROR(
      nnapi_->ANeuralNetworksModel_setOperandValue(
          nnapi_model_->model_, index, value, size));
  return index;
}

void ModelBuilder::Prepare() {
  ClearData();
  nnapi_model_ = std::unique_ptr<Model>(new Model());
  THROW_ON_ERROR(nnapi_->ANeuralNetworksModel_create(&nnapi_model_->model_));
  GetAllIntializers();
  PreprocessIntializers();
  RegisterInitializers();
  RegisterModelInputs();
  AddOperations();
  RegisterModelOutputs();
}

void ModelBuilder::ClearData() {
  shaper_.Clear();

  initializers_.clear();
  skipped_initializers_.clear();

  operand_indices_.clear();
  operand_types_.clear();

  operands_.clear();
  skipped_activations_.clear();

  input_index_vec_.clear();
  output_index_vec_.clear();

  next_index_ = 0;
}

constexpr size_t kDefaultByteAlignmentForNNAPI = 16;
static size_t getPaddedByteSize(size_t size) {
  if (size_t r = size % kDefaultByteAlignmentForNNAPI)
    return size + kDefaultByteAlignmentForNNAPI - r;
  else
    return size;
}

void ModelBuilder::GetAllIntializers() {
  for (const auto& tensor : model_proto_.graph().initializer()) {
    initializers_.insert({tensor.name(), tensor});
  }
}

void ModelBuilder::PreprocessIntializers() {
  for (const auto& node : model_proto_.graph().node()) {
    auto addOpBuilder = CreateOpBuilder(*this, node);
    addOpBuilder->SkipInitializers();
  }
}

void ModelBuilder::RegisterInitializers() {
  // First pass to get all the stats of the initializers
  std::vector<std::tuple<uint32_t, size_t, size_t>> initializers;
  initializers.reserve(model_proto_.graph().initializer_size());
  size_t sizeAll = 0;

  for (int i = 0; i < model_proto_.graph().initializer_size(); ++i) {
    const auto& tensor = model_proto_.graph().initializer(i);
    const auto& name = tensor.name();
    if (HAS(skipped_initializers_, name))
      continue;

    Shape shape;
    for (auto dim : tensor.dims()) {
      shape.push_back(static_cast<uint32_t>(dim));
    }

    shaper_.AddShape(name, shape);

    Type type = Type::TENSOR_FLOAT32;
    switch (tensor.data_type()) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
        type = Type::TENSOR_FLOAT32;
        break;
      default:
        // TODO: support other type
        throw std::invalid_argument(
            "The initializer of graph doesn't have valid type: " + name);
        break;
    }

    OperandType operand_type(type, shape);
    auto index = AddNewOperand(operand_type);
    RegisterOperand(name, index, operand_type);
    const size_t size = operand_type.GetOperandBlobByteSize();
    const size_t paddedSize = getPaddedByteSize(size);
    sizeAll += paddedSize;
    initializers[i] = std::make_tuple(index, size, paddedSize);
  }

  // 2nd pass copies all the initializer data into NNAPI shared memory
  nnapi_model_->mem_initializers_ =
      std::make_unique<Model::NNMemory>(nnapi_, "mem_initializers_", sizeAll);

  // 2nd pass to copy all the initializers into shared memory
  size_t offset = 0;
  for (int i = 0; i < model_proto_.graph().initializer_size(); ++i) {
    const auto& tensor = model_proto_.graph().initializer(i);
    if (HAS(skipped_initializers_, tensor.name()))
      continue;

    Index index;
    size_t size, paddedSize;
    std::tie(index, size, paddedSize) = initializers[i];
    const char* src = nullptr;
    if (tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      src = tensor.float_data().empty()
                ? tensor.raw_data().data()
                : reinterpret_cast<const char*>(tensor.float_data().data());
    } else {
      throw std::invalid_argument(
          "The initializer of graph doesn't have valid type: " + tensor.name());
    }

    uint8_t* dest = nnapi_model_->mem_initializers_->get_data_ptr() + offset;
    memcpy(dest, src, size);
    SetOperandValue(index, nnapi_model_->mem_initializers_.get(), size, offset);
    offset += paddedSize;
  }
}

void ModelBuilder::RegisterModelInputs() {
  for (int32_t input_idx = 0; input_idx < model_proto_.graph().input_size(); input_idx++) {
    const auto& input(model_proto_.graph().input(input_idx));
    std::string input_name = input.name();

    {  // input should not be an initializer
      if (HAS(operands_, input_name))
        continue;

      if (HAS(initializers_, input_name))
        continue;
    }

    Shaper::Shape shape;
    for (int32_t dim_idx = 0; dim_idx < input.type().tensor_type().shape().dim_size(); dim_idx++) {
      const auto& dim = input.type().tensor_type().shape().dim(dim_idx);
      if (dim.value_case() == ONNX_NAMESPACE::TensorShapeProto_Dimension::kDimParam) {
        nnapi_model_->SetInputDimensionMap(input_idx, dim_idx, dim.dim_param());
      }

      shape.push_back(static_cast<uint32_t>(dim.dim_value()));
    }

    shaper_.AddShape(input_name, shape);

    Type type = Type::TENSOR_FLOAT32;
    if (input.type().tensor_type().has_elem_type()) {
      switch (input.type().tensor_type().elem_type()) {
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
          type = Type::TENSOR_FLOAT32;
          break;
        default:
          // TODO: support other type
          throw std::invalid_argument(
              "The input of graph doesn't have valid type: " + input_name);
      }
    } else {
      throw std::invalid_argument(
          "The input of graph doesn't have elem_type: " + input_name);
    }

    OperandType operand_type(type, shape);
    auto index = AddNewOperand(operand_type);
    RegisterOperand(input_name, index, operand_type);

    input_index_vec_.push_back(index);
    nnapi_model_->AddInput(input_name, shape, operand_type);
  }
}

void ModelBuilder::RegisterModelOutputs() {
  for (int32_t output_idx = 0; output_idx < model_proto_.graph().output_size(); output_idx++) {
    const auto& output(model_proto_.graph().output(output_idx));
    const std::string& output_name(output.name());
    if (!HAS(operands_, output_name)) {
      throw std::invalid_argument(
          "The output of graph is not registered" + output_name);
    }

    for (int32_t dim_idx = 0; dim_idx < output.type().tensor_type().shape().dim_size(); dim_idx++) {
      const auto& dim = output.type().tensor_type().shape().dim(dim_idx);
      if (dim.value_case() == ONNX_NAMESPACE::TensorShapeProto_Dimension::kDimParam) {
        nnapi_model_->SetOutputDimensionMap(output_name, dim_idx, dim.dim_param());
      }
    }

    output_index_vec_.push_back(operand_indices_[output_name]);
    nnapi_model_->AddOutput(output_name, shaper_[output_name], operand_types_.at(output_name));
  }
}

ModelBuilder::Index ModelBuilder::AddNewOperand(const OperandType& operand_type) {
  THROW_ON_ERROR(nnapi_->ANeuralNetworksModel_addOperand(
      nnapi_model_->model_, &operand_type.operandType));
  return next_index_++;
}

void ModelBuilder::RegisterOperand(const std::string& name,
                                   Index index,
                                   const OperandType& operand_type) {
  operand_indices_[name] = index;
  operand_types_.insert({name, operand_type});
  operands_.insert(name);
}

void ModelBuilder::SetOperandValue(ModelBuilder::Index index,
                                   Model::NNMemory* memory,
                                   size_t size, size_t offset) {
#ifdef USENNAPISHAREDMEM
  THROW_ON_ERROR(
      nnapi_->ANeuralNetworksModel_setOperandValueFromMemory(
          nnapi_model_->model_, index,
          memory->get_handle(),
          offset, size));
#else
  THROW_ON_ERROR(
      nnapi_->ANeuralNetworksModel_setOperandValue(
          nnapi_model_->model_, index,
          memory->get_data_ptr() + offset,
          size));
#endif
}

uint32_t ModelBuilder::AddOperandFromPersistMemoryBuffer(
    const std::string& name, const void* buffer,
    const android::nn::wrapper::OperandType& operand_type) {
  shaper_.AddShape(name, operand_type.dimensions);
  auto index = AddNewOperand(operand_type);
  RegisterOperand(name, index, operand_type);
  const size_t size = operand_type.GetOperandBlobByteSize();

  // for small size operand, the value will be copied
  // no need to persist
  if (size < ANEURALNETWORKS_MAX_SIZE_OF_IMMEDIATELY_COPIED_VALUES) {
    THROW_ON_ERROR(
        nnapi_->ANeuralNetworksModel_setOperandValue(
            nnapi_model_->model_, index,
            buffer, size));
  } else {
    const size_t paddedSize = getPaddedByteSize(size);
    auto persist_buffer = std::make_unique<Model::NNMemory>(nnapi_, name.c_str(), paddedSize);
    uint8_t* dest = persist_buffer->get_data_ptr();
    memcpy(dest, buffer, size);
    SetOperandValue(index, persist_buffer.get(), size, 0);
    nnapi_model_->mem_persist_buffers_.push_back(std::move(persist_buffer));
  }

  return index;
}

void ModelBuilder::AddOperations() {
  for (const auto& node : model_proto_.graph().node()) {
    auto addOpBuilder = CreateOpBuilder(*this, node);
    addOpBuilder->AddOperator();
  }
}

void ModelBuilder::AddOperation(int op, IndexSeq input_indices,
                                std::vector<std::string> output_names,
                                std::vector<android::nn::wrapper::OperandType> types) {
  IndexSeq output_indices;
  for (const auto& type : types) {
    auto index = AddNewOperand(type);
    output_indices.push_back(index);
  }

  THROW_ON_ERROR_WITH_NOTE(
      nnapi_->ANeuralNetworksModel_addOperation(
          nnapi_model_->model_, op, input_indices.size(), &input_indices[0],
          output_indices.size(), &output_indices[0]),
      "op = " + std::to_string(op));

  for (size_t i = 0; i < types.size(); i++)
    RegisterOperand(output_names[i], output_indices[i], types[i]);
}

std::unique_ptr<Model> ModelBuilder::Compile() {
  Prepare();

  // THROW_ON_ERROR_WITH_NOTE(
  //     nnapi_->ANeuralNetworksModel_relaxComputationFloat32toFloat16(
  //         nnapi_model_->model_, true),
  //     "Set fp16");

  THROW_ON_ERROR_WITH_NOTE(
      nnapi_->ANeuralNetworksModel_identifyInputsAndOutputs(
          nnapi_model_->model_, static_cast<uint32_t>(input_index_vec_.size()),
          &input_index_vec_[0],
          static_cast<uint32_t>(output_index_vec_.size()),
          &output_index_vec_[0]),
      "on identifyInputsAndOutputs");

  THROW_ON_ERROR_WITH_NOTE(
      nnapi_->ANeuralNetworksModel_finish(nnapi_model_->model_),
      "on model finish");

  THROW_ON_ERROR_WITH_NOTE(
      nnapi_->ANeuralNetworksCompilation_create(nnapi_model_->model_, &nnapi_model_->compilation_),
      "on create");

  // THROW_ON_ERROR_WITH_NOTE(
  //     nnapi_->ANeuralNetworksCompilation_setPreference(
  //         nnapi_model_->compilation_, ANEURALNETWORKS_PREFER_SUSTAINED_SPEED),
  //     "on setPreference");

  THROW_ON_ERROR_WITH_NOTE(
      nnapi_->ANeuralNetworksCompilation_finish(nnapi_model_->compilation_),
      "on compilation finish");

  ClearData();
  return std::move(nnapi_model_);
}

int32_t ModelBuilder::FindActivation(const std::string& output) {
  int32_t fuse_code = ANEURALNETWORKS_FUSED_NONE;
  const ONNX_NAMESPACE::NodeProto* activationNode{nullptr};
  std::string node_name;
  for (const auto& _node : model_proto_.graph().node()) {
    if (_node.op_type() == "Relu" && output == _node.input(0)) {
      fuse_code = ANEURALNETWORKS_FUSED_RELU;
      activationNode = &_node;
    }
  }

  if (fuse_code != ANEURALNETWORKS_FUSED_NONE) {
    for (const auto& _node : model_proto_.graph().node()) {
      if (&_node == activationNode)
        continue;

      // if there is any other node using the output
      // will add relu separately
      for (int i = 0; i < _node.input_size(); i++) {
        if (output == _node.input(i))
          return ANEURALNETWORKS_FUSED_NONE;
      }
    }

    skipped_activations_.insert(activationNode->name());
  }

  return fuse_code;
}
}  // namespace nnapi
}  // namespace onnxruntime