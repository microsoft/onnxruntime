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

const float* GetTensorFloatData(const ONNX_NAMESPACE::TensorProto& tensor) {
  return tensor.float_data().empty()
             ? reinterpret_cast<const float*>(tensor.raw_data().data())
             : tensor.float_data().data();
}

const int64_t* GetTensorInt64Data(const ONNX_NAMESPACE::TensorProto& tensor) {
  return tensor.int64_data().empty()
             ? reinterpret_cast<const int64_t*>(tensor.raw_data().data())
             : tensor.int64_data().data();
}

Shaper::Shape GetShape(const ONNX_NAMESPACE::ModelProto& model_proto,
                       const std::string& name) {
  Shaper::Shape emptyShape;
  for (const auto& value_info : model_proto.graph().value_info()) {
    if (value_info.name() == name) {
      if (!value_info.has_type()) {
        return emptyShape;
      } else if (!value_info.type().has_tensor_type()) {
        return emptyShape;
      } else if (!value_info.type().tensor_type().has_shape()) {
        return emptyShape;
      } else if (value_info.type().tensor_type().shape().dim_size() == 0) {
        return emptyShape;
      }

      Shaper::Shape shape;
      for (const auto& dim : value_info.type().tensor_type().shape().dim()) {
        if (dim.has_dim_value()) {
          shape.push_back(dim.dim_value());
        } else {
          return emptyShape;
        }
      }

      return shape;
    }
  }

  return emptyShape;
}

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
          {"GlobalAveragePool", 29},
          {"GlobalMaxPool", 29},
          {"Reshape", 27},
      };

  if (supported_ops.find(op) == supported_ops.end()) {
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
        return {false, "Only conv 2d is supported."};
      }
      if (group != 1 && tensor.dims()[1] != 1) {
        return {false, "group != 1 is not supported"};
      }
    } else {
      return {false, "The weight of convolution must be known"};
    }
    if (helper.get("auto_pad", "NOTSET") != "NOTSET") {
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
    if (!HAS(initializers_, scale_name)) {
      return {false, "Scale of BN must be known"};
    }
    if (!HAS(initializers_, b_name)) {
      return {false, "B of BN must be known"};
    }
    if (!HAS(initializers_, mean_name)) {
      return {false, "Mean of BN must be known"};
    }
    if (!HAS(initializers_, var_name)) {
      return {false, "Var of BN must be known"};
    }
  } else if (op == "GlobalAveragePool" || op == "GlobalMaxPool") {
    const auto& input_shape = GetShape(model_proto_, node.input(0));
    if (input_shape.size() != 4) {
      return {false,
              "GlobalAveragePool/GlobalMaxPool Only rank-4 tensor is supported in " + op};
    }
  } else if (op == "Reshape") {
    if (!HAS(initializers_, node.input(1))) {
      return {false, "shape of reshape must be known"};
    }
  }

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
    } else if (op == "Reshape") {
      // skip the shape for reshape
      skipped_initializers_.insert(node.input(1));
    }
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
        if (shape.empty())  // scalar
          type = Type::FLOAT32;
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
    const size_t size = operand_type.GetOperandByteSize();
    const size_t paddedSize = getPaddedByteSize(size);
    sizeAll += paddedSize;
    initializers[i] = std::make_tuple(index, size, paddedSize);
  }

  // 2nd pass copies all the initializer data into NNAPI shared memory
  nnapi_model_->mem_initializers_ =
      std::make_unique<NNMemory>(nnapi_, "mem_initializers_", sizeAll);

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

void ModelBuilder::SetOperandValue(ModelBuilder::Index index,
                                   NNMemory* memory,
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
  const size_t size = operand_type.GetOperandByteSize();
  const size_t paddedSize = getPaddedByteSize(size);
  auto persist_buffer = std::make_unique<NNMemory>(nnapi_, name.c_str(), paddedSize);
  shaper_.AddShape(name, operand_type.dimensions);
  auto index = AddNewOperand(operand_type);
  RegisterOperand(name, index, operand_type);
  uint8_t* dest = persist_buffer->get_data_ptr();
  memcpy(dest, buffer, size);
  SetOperandValue(index, persist_buffer.get(), size, 0);
  nnapi_model_->mem_persist_buffers_.push_back(std::move(persist_buffer));
  return index;
}

uint32_t ModelBuilder::AddNHWCInitializer(const std::string& name) {
  const auto& tensor = initializers_.at(name);
  Shape shape;
  for (auto dim : tensor.dims())
    shape.push_back(static_cast<uint32_t>(dim));

  if (shape.size() != 4)
    throw std::invalid_argument(
        "The initializer is not 4D: " + name +
        " actual dim " + std::to_string(shape.size()));

  // TODO support other data types
  Type type = Type::TENSOR_FLOAT32;
  if (tensor.data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT)
    throw std::invalid_argument(
        "The initializer of graph doesn't have valid type: " + name);
  const float* src = GetTensorFloatData(tensor);
  float buffer[Product(shape)];
  auto out_t = shape[0], in_t = shape[1],
       h_t = shape[2], w_t = shape[3];
  Shape dest_shape = {out_t, h_t, w_t, in_t};
  const OperandType operandType(type, dest_shape);

  for (uint32_t out = 0; out < out_t; out++) {
    for (uint32_t in = 0; in < in_t; in++) {
      for (uint32_t h = 0; h < h_t; h++) {
        for (uint32_t w = 0; w < w_t; w++) {
          auto onnx_idx = out * in_t * h_t * w_t +
                          in * h_t * w_t + h * w_t +
                          w;
          auto nnapi_idx = out * h_t * w_t * in_t +
                           h * w_t * in_t + w * in_t +
                           in;
          buffer[nnapi_idx] = src[onnx_idx];
        }
      }
    }
  }

  return AddOperandFromPersistMemoryBuffer(name, &buffer[0], operandType);
}

uint32_t ModelBuilder::Add1230Initializer(const std::string& name) {
  const auto& tensor = initializers_.at(name);
  Shape shape;
  for (auto dim : tensor.dims())
    shape.push_back(static_cast<uint32_t>(dim));

  if (shape.size() != 4)
    throw std::invalid_argument(
        "The initializer is not 4D: " + name +
        " actual dim " + std::to_string(shape.size()));

  // TODO support other data types
  Type type = Type::TENSOR_FLOAT32;
  if (tensor.data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT)
    throw std::invalid_argument(
        "The initializer of graph doesn't have valid type: " + name);
  const float* src = GetTensorFloatData(tensor);
  float buffer[Product(shape)];
  auto out_t = shape[0], in_t = shape[1],
       h_t = shape[2], w_t = shape[3];
  Shape dest_shape = {in_t, h_t, w_t, out_t};
  const OperandType operandType(type, dest_shape);

  for (uint32_t out = 0; out < out_t; out++) {
    for (uint32_t in = 0; in < in_t; in++) {
      for (uint32_t h = 0; h < h_t; h++) {
        for (uint32_t w = 0; w < w_t; w++) {
          auto onnx_idx = out * in_t * h_t * w_t +
                          in * h_t * w_t + h * w_t +
                          w;
          auto nnapi_idx = h * w_t * out_t +
                           w * out_t +
                           out;
          buffer[nnapi_idx] = src[onnx_idx];
        }
      }
    }
  }

  return AddOperandFromPersistMemoryBuffer(name, &buffer[0], operandType);
}

void ModelBuilder::AddOperations() {
  for (const auto& node : model_proto_.graph().node()) {
    const auto& op = node.op_type();

    bool nchw = true;
    NodeAttrHelper helper(node);
    // process skips (already used as activation)
    if (op == "Add") {
      const auto input1 = node.input(0);
      const auto input2 = node.input(1);
      const auto output = node.output(0);

      IndexSeq input_indices;
      input_indices.push_back(operand_indexes_.at(input1));  // input 1
      input_indices.push_back(operand_indexes_.at(input2));  // input 2
      int32_t fuse_code = FindActivation(output);
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
      int32_t fuse_code = FindActivation(output);
      input_indices.push_back(
          SetOperandFromScalar(Type::INT32,
                               static_cast<const void*>(&fuse_code),
                               sizeof(fuse_code)));  // fusecode
      shaper_.Eltwise(input1, input2, output);
      const OperandType output_operand_type(operand_types_.at(input1).type, shaper_[output]);
      AddOperation(ANEURALNETWORKS_MUL, input_indices, {output}, {output_operand_type});
    } else if (op == "Relu") {
      const auto& input = node.input(0);
      const auto& output = node.output(0);
      shaper_.Identity(input, output);
      const OperandType output_operand_type(operand_types_.at(input).type, shaper_[output]);

      // skip this relu if it is some op's fuse output
      if (HAS(skipped_activations_, node.name())) {
        RegisterOperand(output, operand_indexes_.at(input), output_operand_type);
      } else {
        IndexSeq input_indices;
        input_indices.push_back(operand_indexes_.at(input));
        AddOperation(ANEURALNETWORKS_RELU, input_indices, {output}, {output_operand_type});
      }
    } else if (op == "Conv") {
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
      const auto& input = node.input(0);
      const auto& weight = node.input(1);

      // TODO: Support it
      bool conv2d = group == 1;
      const auto& weight_tensor = initializers_.at(weight);
      bool depthwiseConv2D = weight_tensor.dims()[1] == 1;

      if (!conv2d && !depthwiseConv2D)
        throw std::invalid_argument(
            "Conv group != 1 and not depthwise is not supported");

      IndexSeq input_indices;
      input_indices.push_back(operand_indexes_.at(input));
      if (conv2d)
        input_indices.push_back(AddNHWCInitializer(weight));
      else  // depthwiseConv2D
        input_indices.push_back(Add1230Initializer(weight));

      bool hasBias = (node.input_size() >= 3);
      std::string bias;
      if (hasBias) {
        bias = node.input(2);
      }

      uint32_t bias_idx_val;
      if (hasBias) {
        bias_idx_val = operand_indexes_.at(bias);
      } else {
        bias = weight + "_bias";
        const auto weight_dimen = shaper_[weight];
        Shape bias_dimen;
        if (conv2d)
          bias_dimen = {weight_dimen[0]};
        else
          bias_dimen = {weight_dimen[3]};

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
      const auto& output = node.output(0);
      input_indices.push_back(bias_idx_val);
      input_indices.push_back(SetOperandFromScalar(Type::INT32, static_cast<const void*>(&onnx_pads[1]), sizeof(int32_t)));
      input_indices.push_back(SetOperandFromScalar(Type::INT32, static_cast<const void*>(&onnx_pads[3]), sizeof(int32_t)));
      input_indices.push_back(SetOperandFromScalar(Type::INT32, static_cast<const void*>(&onnx_pads[0]), sizeof(int32_t)));
      input_indices.push_back(SetOperandFromScalar(Type::INT32, static_cast<const void*>(&onnx_pads[2]), sizeof(int32_t)));
      input_indices.push_back(SetOperandFromScalar(Type::INT32, static_cast<const void*>(&onnx_strides[1]), sizeof(int32_t)));
      input_indices.push_back(SetOperandFromScalar(Type::INT32, static_cast<const void*>(&onnx_strides[0]), sizeof(int32_t)));
      if (!conv2d && depthwiseConv2D) {
        int32_t depthwiseMultiplier = shaper_[weight][3] / group;
        input_indices.push_back(SetOperandFromScalar(Type::INT32, static_cast<const void*>(&depthwiseMultiplier), sizeof(int32_t)));
      }
      int32_t fuse_code = FindActivation(output);
      input_indices.push_back(SetOperandFromScalar(Type::INT32, static_cast<const void*>(&fuse_code), sizeof(int32_t)));
      input_indices.push_back(SetOperandFromScalar(Type::BOOL, static_cast<const void*>(&nchw), sizeof(bool)));
      input_indices.push_back(SetOperandFromScalar(Type::INT32, static_cast<const void*>(&onnx_dilations[1]), sizeof(int32_t)));
      input_indices.push_back(SetOperandFromScalar(Type::INT32, static_cast<const void*>(&onnx_dilations[0]), sizeof(int32_t)));

      int32_t operationCode = ANEURALNETWORKS_CONV_2D;
      if (conv2d) {
        operationCode = ANEURALNETWORKS_CONV_2D;
        shaper_.Conv(input, weight,
                     onnx_pads[1], onnx_pads[3], onnx_pads[0], onnx_pads[2],
                     onnx_strides[1], onnx_strides[0],
                     nchw,
                     onnx_dilations[1], onnx_dilations[0],
                     output);
      } else if (depthwiseConv2D) {
        operationCode = ANEURALNETWORKS_DEPTHWISE_CONV_2D;
        shaper_.DepthwiseConv(input, weight,
                              onnx_pads[1], onnx_pads[3], onnx_pads[0], onnx_pads[2],
                              onnx_strides[1], onnx_strides[0],
                              nchw,
                              onnx_dilations[1], onnx_dilations[0],
                              output);
      }

      const OperandType output_operand_type(operand_types_.at(input).type, shaper_[output]);
      AddOperation(operationCode, input_indices, {output}, {output_operand_type});

    } else if (op == "BatchNormalization") {
      if (node.output_size() != 1) {
        throw std::invalid_argument(
            "Your onnx model may be in training mode, \
            please export it in test mode.");
      }

      const auto input = node.input(0);
      const auto output = node.output(0);

      const auto& scale_tensor = initializers_.at(node.input(1));
      const auto& bias_tensor = initializers_.at(node.input(2));
      const auto& mean_tensor = initializers_.at(node.input(3));
      const auto& var_tensor = initializers_.at(node.input(4));
      const auto eps = helper.get("epsilon", 1e-5f);

      const auto size = static_cast<uint32_t>(scale_tensor.dims()[0]);
      vector<float> a, b;
      a.reserve(size);
      b.reserve(size);

      const float* scale_data = GetTensorFloatData(scale_tensor);
      const float* bias_data = GetTensorFloatData(bias_tensor);
      const float* mean_data = GetTensorFloatData(mean_tensor);
      const float* var_data = GetTensorFloatData(var_tensor);

      for (int64_t i = 0; i < size; i++) {
        a.push_back(scale_data[i] / sqrt(var_data[i] + eps));
        b.push_back((scale_data[i] * -mean_data[i]) / sqrt(var_data[i] + eps) +
                    bias_data[i]);
      }

      const auto tensor_a_name = input + "_imm_a";
      const auto tensor_b_name = input + "_imm_b";
      const auto tensor_imm_product_name = input + "_imm_mul";
      Shape tensor_a_dimen;
      if (nchw)
        tensor_a_dimen = {size, 1, 1};  // {C, H, W}
      else
        tensor_a_dimen = {size};

      shaper_.AddShape(tensor_a_name, tensor_a_dimen);
      shaper_.AddShape(tensor_b_name, tensor_a_dimen);
      const OperandType operandType_a(operand_types_.at(input).type, tensor_a_dimen);
      const auto tensor_a_idx = AddOperandFromPersistMemoryBuffer(tensor_a_name, a.data(), operandType_a);
      const OperandType operandType_b(operand_types_.at(input).type, tensor_a_dimen);
      const auto tensor_b_idx = AddOperandFromPersistMemoryBuffer(tensor_b_name, b.data(), operandType_b);

      // mul
      {
        IndexSeq input_indices;
        input_indices.push_back(operand_indexes_.at(input));  // input 1
        input_indices.push_back(tensor_a_idx);                // input 2
        int32_t fuse_code = FindActivation(tensor_imm_product_name);
        input_indices.push_back(
            SetOperandFromScalar(Type::INT32,
                                 static_cast<const void*>(&fuse_code),
                                 sizeof(fuse_code)));  // fusecode
        shaper_.Eltwise(input, tensor_a_name, tensor_imm_product_name);
        const OperandType output_operand_type(operand_types_.at(input).type, shaper_[tensor_imm_product_name]);
        AddOperation(ANEURALNETWORKS_MUL, input_indices, {tensor_imm_product_name}, {output_operand_type});
      }
      // add
      {
        IndexSeq input_indices;
        input_indices.push_back(operand_indexes_.at(tensor_imm_product_name));  // input 1
        input_indices.push_back(tensor_b_idx);                                  // input 2
        int32_t fuse_code = FindActivation(output);
        input_indices.push_back(
            SetOperandFromScalar(Type::INT32,
                                 static_cast<const void*>(&fuse_code),
                                 sizeof(fuse_code)));  // fusecode
        shaper_.Eltwise(tensor_imm_product_name, tensor_b_name, output);
        const OperandType output_operand_type(operand_types_.at(tensor_imm_product_name).type, shaper_[output]);
        AddOperation(ANEURALNETWORKS_ADD, input_indices, {output}, {output_operand_type});
      }

    } else if (op == "GlobalAveragePool" || op == "GlobalMaxPool") {
      const auto input = node.input(0);
      int32_t operationCode = ANEURALNETWORKS_CONV_2D;

      vector<int> onnx_strides, onnx_pads, kernel_shape;
      if (op == "AveragePool" || op == "MaxPool") {
        if (op == "AveragePool")
          operationCode = ANEURALNETWORKS_AVERAGE_POOL_2D;
        else  //MaxPool
          operationCode = ANEURALNETWORKS_MAX_POOL_2D;

        kernel_shape = helper.get("kernel_shape", vector<int>{0, 0});
        if (helper.get("count_include_pad", 0) == 1)
          throw std::invalid_argument("count_include_pad == 1 is not supported");

        onnx_strides = helper.get("strides", vector<int>{1, 1});
        onnx_pads = helper.get("pads", vector<int>{0, 0, 0, 0});

        if (helper.get("storage_order", 0) == 1)
          throw std::invalid_argument("storage_order == 1 is not supported");

        if (helper.get("auto_pad", "NOTSET") != "NOTSET")
          throw std::invalid_argument("auto_pad is not supported");
      } else {
        if (op == "GlobalAveragePool")
          operationCode = ANEURALNETWORKS_AVERAGE_POOL_2D;
        else  //GlobalMaxPool
          operationCode = ANEURALNETWORKS_MAX_POOL_2D;

        onnx_strides = vector<int>{1, 1};
        onnx_pads = vector<int>{0, 0, 0, 0};

        if (nchw)
          kernel_shape = vector<int>{static_cast<int32_t>(shaper_[input][2]),
                                     static_cast<int32_t>(shaper_[input][3])};
        else
          kernel_shape = vector<int>{static_cast<int32_t>(shaper_[input][1]),
                                     static_cast<int32_t>(shaper_[input][2])};
      }

      IndexSeq input_indices;
      input_indices.push_back(operand_indexes_.at(input));
      const auto& output = node.output(0);

      input_indices.push_back(SetOperandFromScalar(Type::INT32, static_cast<const void*>(&onnx_pads[1]), sizeof(int32_t)));
      input_indices.push_back(SetOperandFromScalar(Type::INT32, static_cast<const void*>(&onnx_pads[3]), sizeof(int32_t)));
      input_indices.push_back(SetOperandFromScalar(Type::INT32, static_cast<const void*>(&onnx_pads[0]), sizeof(int32_t)));
      input_indices.push_back(SetOperandFromScalar(Type::INT32, static_cast<const void*>(&onnx_pads[2]), sizeof(int32_t)));
      input_indices.push_back(SetOperandFromScalar(Type::INT32, static_cast<const void*>(&onnx_strides[1]), sizeof(int32_t)));
      input_indices.push_back(SetOperandFromScalar(Type::INT32, static_cast<const void*>(&onnx_strides[0]), sizeof(int32_t)));
      input_indices.push_back(SetOperandFromScalar(Type::INT32, static_cast<const void*>(&kernel_shape[1]), sizeof(int32_t)));
      input_indices.push_back(SetOperandFromScalar(Type::INT32, static_cast<const void*>(&kernel_shape[0]), sizeof(int32_t)));
      int32_t fuse_code = FindActivation(output);
      input_indices.push_back(SetOperandFromScalar(Type::INT32, static_cast<const void*>(&fuse_code), sizeof(int32_t)));
      input_indices.push_back(SetOperandFromScalar(Type::BOOL, static_cast<const void*>(&nchw), sizeof(bool)));

      shaper_.Pool(input,
                   onnx_pads[1], onnx_pads[3], onnx_pads[0], onnx_pads[2],
                   onnx_strides[1], onnx_strides[0],
                   nchw,
                   kernel_shape[1], kernel_shape[0],
                   output);
      const OperandType output_operand_type(operand_types_.at(input).type, shaper_[output]);
      AddOperation(operationCode, input_indices, {output}, {output_operand_type});
    } else if (op == "Reshape") {
      // For reshape we are not really doing anything but
      // register a new operand with new shape
      const auto input = node.input(0);
      const auto shape_name = node.input(1);
      const auto output = node.output(0);

      const auto& shape_tensor = initializers_.at(shape_name);
      const int64_t* rawShape = GetTensorInt64Data(shape_tensor);
      const auto size = static_cast<uint32_t>(shape_tensor.dims()[0]);
      std::vector<int32_t> shape(size);
      for (uint32_t i = 0; i < size; i++) {
        shape[i] = static_cast<int32_t>(rawShape[i]);
      }

      shaper_.Reshape(input, shape, output);
      const OperandType output_operand_type(operand_types_.at(input).type, shaper_[output]);
      RegisterOperand(output, operand_indexes_.at(input), output_operand_type);
    }

    else {
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