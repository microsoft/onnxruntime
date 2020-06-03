// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "helper.h"
#include "model_builder.h"
#include "NodeAttrHelper.h"

#include "core/providers/nnapi/nnapi_builtin/nnapi_lib/nnapi_implementation.h"
#include <onnx/shape_inference/implementation.h>

namespace onnxruntime {
namespace nnapi {

using namespace android::nn::wrapper;
using std::vector;

#define HAS(map, key) \
  (map.find(key) != map.end())

ModelBuilder::ModelBuilder(ONNX_NAMESPACE::ModelProto& model_proto)
    : nnapi_(NnApiImplementation()), model_proto_(model_proto) {}

std::pair<bool, std::string> ModelBuilder::IsNodeSupported(
    const ONNX_NAMESPACE::NodeProto& node) {
  int32_t android_sdk_ver = nnapi_ ? nnapi_->android_sdk_version : 0;

#ifdef __ANDROID__
  if (android_sdk_ver < 27) {
    LOGI("Android API level %d is lower than 27", android_sdk_ver);
    return {false, "Android API level is lower than 27"};
  }
#endif

  const auto& op = node.op_type();
  std::map<std::string, int>
      supported_ops{
          {"Add", 27},
          {"Relu", 27},
          {"Conv", 29},
          {"BatchNormalization", 27},
          {"Mul", 27},
      };

  if (supported_ops.find(op) == supported_ops.end()) {
    LOGI("Unsupported operator %s", op.c_str());
    return {false, "Unsupported operator " + op};
  }

#ifdef __ANDROID__
  if (supported_ops[op] > android_sdk_ver) {
    LOGI("Android API level %d is lower than %d", android_sdk_ver, supported_ops[op]);
    return {false, "Operator " + op + " is only supported on API > " + std::to_string(supported_ops[op])};
  }
#endif

  GetAllIntializers();
  NodeAttrHelper helper(node);
  if (op == "Conv") {
    const auto group = helper.get("group", 1);
    const auto weight_name = node.input(1);
    if (HAS(initializers_, weight_name)) {
      const auto& tensor = initializers_.at(weight_name);
      if (tensor.dims().size() != 4) {
        LOGI("1 Unsupported operator %s", op.c_str());
        return {false, "Only conv 2d is supported."};
      }
      if (group != 1 && tensor.dims()[1] != 1) {
        LOGI("2 Unsupported operator %s", op.c_str());
        return {false, "group != 1 is not supported"};
      }
    } else {
      LOGI("3 Unsupported operator %s", op.c_str());
      return {false, "The weight of convolution must be known"};
    }
    if (helper.get("auto_pad", "NOTSET") != "NOTSET") {
      LOGI("4 Unsupported operator %s", op.c_str());
      return {false, "SAME_LOWER auto_pad is not supported"};
    }
  } else if (op == "BatchNormalization") {
    if (node.output_size() != 1) {
      return {false,
              "Your onnx model may be in training mode, please export "
              "it in test mode."};
    }
    const auto& scale_name = node.input(1);
    const auto& b_name = node.input(2);
    const auto& mean_name = node.input(3);
    const auto& var_name = node.input(4);
    if (HAS(initializers_, scale_name)) {
      return {false, "Scale of BN must be known"};
    }
    if (HAS(initializers_, b_name)) {
      return {false, "B of BN must be known"};
    }
    if (HAS(initializers_, mean_name)) {
      return {false, "Mean of BN must be known"};
    }
    if (HAS(initializers_, var_name)) {
      return {false, "Var of BN must be known"};
    }
  }

  LOGI("Supported operator %s", op.c_str());
  return {true, ""};
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
  std::vector<int> supported_node_vec;

  for (int i = 0; i < model_proto_.graph().node_size(); i++) {
    bool supported;
    std::string error_msg;
    std::tie(supported, error_msg) = IsNodeSupported(model_proto_.graph().node(i));
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

  operand_indexes_.clear();
  operand_types_.clear();

  operands_.clear();

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
    const auto& op = node.op_type();
    if (op == "Conv") {
      // skip the weight for conv as we need to transpose
      skipped_initializers_.insert(node.input(1));
    } else if (op == "BatchNormalization") {
      // skip everything except input0 for BatchNormalization
      const auto& scale_name = node.input(1);
      const auto& b_name = node.input(2);
      const auto& mean_name = node.input(3);
      const auto& var_name = node.input(4);
      skipped_initializers_.insert(scale_name);
      skipped_initializers_.insert(b_name);
      skipped_initializers_.insert(mean_name);
      skipped_initializers_.insert(var_name);
    }
  }
}

void ModelBuilder::RegisterInitializers() {
  // First pass to get all the stats of the initializers
  std::vector<std::tuple<uint32_t, size_t, size_t>> initializers;
  initializers.reserve(model_proto_.graph().initializer_size());
  size_t sizeAll = 0;

  for (const auto& tensor : model_proto_.graph().initializer()) {
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
        if (shape.empty())  // scalar
          type = Type::FLOAT32;
        break;
      default:
        // TODO: support other type
        throw std::invalid_argument(
            "The initializer of graph doesn't have valid type: " + name);
    }

    OperandType operand_type(type, shape);
    auto index = AddNewOperand(operand_type);
    RegisterOperand(name, index, operand_type);
    const size_t size = operand_type.GetOperandByteSize();
    const size_t paddedSize = getPaddedByteSize(size);
    sizeAll += paddedSize;
    initializers.push_back(std::make_tuple(index, size, paddedSize));
  }

  // 2nd pass copies all the initializer data into NNAPI shared memory
  nnapi_model_->mem_initializers_ =
      std::make_unique<NNMemory>(nnapi_, "mem_initializers_", sizeAll);

  // 2nd pass to copy all the initializers into shared memory
  size_t offset = 0;
  for (int i = 0; i < model_proto_.graph().initializer_size(); ++i) {
    const auto& tensor = model_proto_.graph().initializer(i);
    if (skipped_initializers_.find(tensor.name()) != skipped_initializers_.end())
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
    THROW_ON_ERROR(
        nnapi_->ANeuralNetworksModel_setOperandValueFromMemory(
            nnapi_model_->model_, index,
            nnapi_model_->mem_initializers_->get_handle(),
            offset, size));
    offset += paddedSize;
  }
}

void ModelBuilder::RegisterModelInputs() {
  for (const auto& input : model_proto_.graph().input()) {
    const std::string& name(input.name());

    {  // input should not be an initializer
      if (HAS(operands_, name))
        continue;

      if (HAS(initializers_, name))
        continue;
    }

    Shaper::Shape shape;
    for (const auto& dim : input.type().tensor_type().shape().dim()) {
      if (dim.value_case() ==
          ONNX_NAMESPACE::TensorShapeProto_Dimension::kDimValue) {
        shape.push_back(static_cast<uint32_t>(dim.dim_value()));
      } else {
        throw std::invalid_argument(
            "The input of graph doesn't have dim_value");
      }
    }

    shaper_.AddShape(name, shape);

    Type type = Type::TENSOR_FLOAT32;
    if (input.type().tensor_type().has_elem_type()) {
      switch (input.type().tensor_type().elem_type()) {
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
          if (shape.empty())  // scalar
            type = Type::FLOAT32;
          break;
        default:
          // TODO: support other type
          throw std::invalid_argument(
              "The input of graph doesn't have valid type: " + name);
      }
    }

    OperandType operand_type(type, shape);
    auto index = AddNewOperand(operand_type);
    RegisterOperand(name, index, operand_type);

    input_index_vec_.push_back(index);
    nnapi_model_->AddInput(name, shape, operand_type);
  }
}

void ModelBuilder::RegisterModelOutputs() {
  for (const auto& output : model_proto_.graph().output()) {
    const std::string& name(output.name());
    if (operands_.find(name) == operands_.end()) {
      throw std::invalid_argument(
          "The output of graph is not registered" + name);
    }

    output_index_vec_.push_back(operand_indexes_[name]);
    nnapi_model_->AddOutput(name, shaper_[name], operand_types_.at(name));
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
  operand_indexes_[name] = index;
  operand_types_.insert({name, operand_type});
  operands_.insert(name);
}

uint32_t ModelBuilder::AddOperandFromPersistMemoryBuffer(
    const std::string& name, const void* buffer,
    const android::nn::wrapper::OperandType& operand_type) {
  const size_t size = operand_type.GetOperandByteSize();
  const size_t paddedSize = getPaddedByteSize(size);
  auto persist_buffer = std::make_unique<NNMemory>(nnapi_, name.c_str(), paddedSize);
  shaper_.AddShape(name, operand_type.dimensions);
  auto index = AddNewOperand(operand_type);
  RegisterOperand(name, index, operand_type);
  uint8_t* dest = persist_buffer->get_data_ptr();
  memcpy(dest, buffer, size);
  THROW_ON_ERROR(
      nnapi_->ANeuralNetworksModel_setOperandValueFromMemory(
          nnapi_model_->model_, index,
          persist_buffer->get_handle(),
          0, size));
  nnapi_model_->mem_persist_buffers_.push_back(std::move(persist_buffer));
  return index;
}

void ModelBuilder::AddOperations() {
  for (const auto& node : model_proto_.graph().node()) {
    const auto& op = node.op_type();

    // process skips (already used as activation)
    if (op == "Add") {
      const auto input1 = node.input(0);
      const auto input2 = node.input(1);
      const auto output = node.output(0);

      IndexSeq input_indices;
      input_indices.push_back(operand_indexes_.at(input1));  // input 1
      input_indices.push_back(operand_indexes_.at(input2));  // input 2
      int32_t fuse_code = ANEURALNETWORKS_FUSED_NONE;
      input_indices.push_back(
          SetOperandFromScalar(Type::INT32,
                               static_cast<const void*>(&fuse_code),
                               sizeof(fuse_code)));  // fusecode
      shaper_.Eltwise(input1, input2, output);
      const OperandType output_operand_type(operand_types_.at(input1).type, shaper_[output]);
      AddOperation(ANEURALNETWORKS_ADD, input_indices, {output}, {output_operand_type});
    } else if (op == "Mul") {
      const auto input1 = node.input(0);
      const auto input2 = node.input(1);
      const auto output = node.output(0);

      IndexSeq input_indices;
      input_indices.push_back(operand_indexes_.at(input1));  // input 1
      input_indices.push_back(operand_indexes_.at(input2));  // input 2
      int32_t fuse_code = ANEURALNETWORKS_FUSED_NONE;
      input_indices.push_back(
          SetOperandFromScalar(Type::INT32,
                               static_cast<const void*>(&fuse_code),
                               sizeof(fuse_code)));  // fusecode
      shaper_.Eltwise(input1, input2, output);
      const OperandType output_operand_type(operand_types_.at(input1).type, shaper_[output]);
      AddOperation(ANEURALNETWORKS_MUL, input_indices, {output}, {output_operand_type});
    } else if (op == "Relu") {
      const auto input = node.input(0);
      const auto output = node.output(0);
      IndexSeq input_indices;
      input_indices.push_back(operand_indexes_.at(input));
      shaper_.Identity(input, output);
      const OperandType output_operand_type(operand_types_.at(input).type, shaper_[output]);
      AddOperation(ANEURALNETWORKS_RELU, input_indices, {output}, {output_operand_type});
    } else if (op == "Conv") {
      NodeAttrHelper helper(node);

      // onnx strides are in the order height, width
      // while nnapi strides are in the order width, height
      const auto onnx_strides = helper.get("strides", vector<int>{1, 1});

      // onnx pads are in the order top, left, bottom, right
      // while nnapi pads is in the order left, right, top, bottom
      const auto onnx_pads = helper.get("pads", vector<int>{0, 0, 0, 0});

      // onnx dilations is in the order height, width
      // while nnapi dilations are in the order width, height
      const auto onnx_dilations = helper.get("dilations", vector<int>{1, 1});

      const auto group = helper.get("group", 1);
      const auto input = node.input(0);
      const auto weight = node.input(1);

      IndexSeq input_indices;
      input_indices.push_back(operand_indexes_.at(input));
      input_indices.push_back(operand_indexes_.at(weight));

      bool hasBias = (node.input_size() >= 3);
      std::string bias;
      if (hasBias) {
        bias = node.input(2);
      }

      uint32_t bias_idx_val;
      if (group == 1) {
        if (hasBias) {
          bias_idx_val = operand_indexes_.at(bias);
        } else {
          bias = weight + "_bias";
          const auto weight_dimen = shaper_[weight];
          const Shape bias_dimen{weight_dimen[0]};
          const auto& weight_type = operand_types_.at(weight).type;
          if (weight_type == Type::TENSOR_FLOAT32) {
            float buffer[bias_dimen[0]];
            for (uint32_t i = 0; i < bias_dimen[0]; i++) {
              buffer[i] = 0.f;
            }
            OperandType operandType(Type::TENSOR_FLOAT32, bias_dimen);
            bias_idx_val = AddOperandFromPersistMemoryBuffer(bias, &buffer[0], operandType);
          } else {
            throw std::invalid_argument("Unknown type " + typeToStr(weight_type));
          }
        }
        input_indices.push_back(bias_idx_val);
        input_indices.push_back(SetOperandFromScalar(Type::INT32, static_cast<const void*>(&onnx_pads[1]), sizeof(int)));
        input_indices.push_back(SetOperandFromScalar(Type::INT32, static_cast<const void*>(&onnx_pads[3]), sizeof(int)));
        input_indices.push_back(SetOperandFromScalar(Type::INT32, static_cast<const void*>(&onnx_pads[0]), sizeof(int)));
        input_indices.push_back(SetOperandFromScalar(Type::INT32, static_cast<const void*>(&onnx_pads[2]), sizeof(int)));
        input_indices.push_back(SetOperandFromScalar(Type::INT32, static_cast<const void*>(&onnx_strides[1]), sizeof(int)));
        input_indices.push_back(SetOperandFromScalar(Type::INT32, static_cast<const void*>(&onnx_strides[0]), sizeof(int)));
        int32_t fuse_code = ANEURALNETWORKS_FUSED_NONE;
        input_indices.push_back(SetOperandFromScalar(Type::INT32, static_cast<const void*>(&fuse_code), sizeof(fuse_code)));
        bool nchw = true;
        input_indices.push_back(SetOperandFromScalar(Type::BOOL, static_cast<const void*>(&nchw), sizeof(bool)));
        input_indices.push_back(SetOperandFromScalar(Type::INT32, static_cast<const void*>(&onnx_dilations[1]), sizeof(int)));
        input_indices.push_back(SetOperandFromScalar(Type::INT32, static_cast<const void*>(&onnx_dilations[0]), sizeof(int)));

        const auto output = node.output(0);
        shaper_.Conv(input, weight, onnx_pads[1], onnx_pads[3], onnx_pads[0],
                     onnx_pads[2], onnx_strides[1], onnx_strides[0], nchw, onnx_dilations[1],
                     onnx_dilations[0], output);
        const OperandType output_operand_type(operand_types_.at(input).type, shaper_[output]);
        AddOperation(ANEURALNETWORKS_CONV_2D, input_indices, {output}, {output_operand_type});
      } else {
        // TODO: Support it
        throw std::invalid_argument("group != 1 is not supported");
      }
    } else {
      throw std::invalid_argument("Unsupported operator " + op);
    }
  }
}  // namespace nnapi

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

  THROW_ON_ERROR_WITH_NOTE(
      nnapi_->ANeuralNetworksCompilation_finish(nnapi_model_->compilation_),
      "on compilation finish");

  ClearData();
  return std::move(nnapi_model_);
}
}  // namespace nnapi
}  // namespace onnxruntime