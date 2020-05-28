// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "model_builder.h"
#include "core/providers/nnapi/nnapi_builtin/nnapi_lib/nnapi_implementation.h"
#include "helper.h"

namespace onnxruntime {
namespace nnapi {

using namespace android::nn::wrapper;

ModelBuilder::ModelBuilder(ONNX_NAMESPACE::ModelProto& model_proto)
    : nnapi_(NnApiImplementation()), model_proto_(model_proto) {}

std::pair<bool, std::string> ModelBuilder::IsNodeSupported(
    const ONNX_NAMESPACE::NodeProto& node) {
  (void)node;
  int32_t android_sdk_ver = nnapi_ ? nnapi_->android_sdk_version : 0;

#ifdef __ANDROID__
  if (android_sdk_ver < 27) {
    LOGI("Android API level %d is lower than 27", android_sdk_ver);
    return {false, "Android API level is lower than 27"};
  }
#endif

  const auto& op = node.op_type();
  std::map<std::string, int> supported_ops{{"Add", 27}};

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
  // ONNX_NAMESPACE::shape_inference::InferShapes(model_proto_);

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

#define DEFINE_OPERAND_FROM_SCALAR(scalar_type, op_type)                   \
  ModelBuilder::Index ModelBuilder::OperandFromScalar(scalar_type value) { \
    const auto index = AddNewOperand({Type::op_type});                     \
    THROW_ON_ERROR_WITH_NOTE(                                              \
        nnapi_->ANeuralNetworksModel_setOperandValue(                      \
            nnapi_model_->model_, index, &value, sizeof(value)),           \
        "value: " + std::to_string(value));                                \
    return index;                                                          \
  }  // namespace nnapi

DEFINE_OPERAND_FROM_SCALAR(bool, BOOL);
DEFINE_OPERAND_FROM_SCALAR(uint32_t, UINT32);
DEFINE_OPERAND_FROM_SCALAR(int32_t, INT32);
DEFINE_OPERAND_FROM_SCALAR(float, FLOAT32);

#undef DEFINE_OPERAND_FROM_SCALAR

void ModelBuilder::prepare() {
  clearData();
  nnapi_model_ = std::unique_ptr<Model>(new Model());
  THROW_ON_ERROR(nnapi_->ANeuralNetworksModel_create(&nnapi_model_->model_));
  addInitializers();
  registerModelInputs();
  addOperations();
  registerModelOutputs();
}

void ModelBuilder::clearData() {
  shaper_.Clear();
  operand_indexes_.clear();
  operands_.clear();
  input_index_vec_.clear();
  output_index_vec_.clear();

  next_index_ = 0;
}

void ModelBuilder::addInitializers() {}

void ModelBuilder::registerModelInputs() {
  for (const auto& input : model_proto_.graph().input()) {
    if (operands_.find(input.name()) != operands_.end())
      continue;

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

    shaper_.AddShape(input.name(), shape);

    Type type = Type::TENSOR_FLOAT32;
    if (input.type().tensor_type().has_elem_type()) {
      switch (input.type().tensor_type().elem_type()) {
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
          type = Type::TENSOR_FLOAT32;
          break;
        default:
          // TODO: support other type
          throw std::invalid_argument(
              "The input of graph doesn't have valid type");
      }
    }

    auto index = AddNewOperand({type, shape});
    RegisterOperand(input.name(), index, {type, shape});

    input_index_vec_.push_back(index);
    nnapi_model_->AddInput(input.name(), shape);
  }
}

void ModelBuilder::registerModelOutputs() {
  for (const auto& output : model_proto_.graph().output()) {
    if (operands_.find(output.name()) == operands_.end()) {
      throw std::invalid_argument(
          "The output of graph is not registered" + output.name());
    }

    output_index_vec_.push_back(operand_indexes_[output.name()]);
    nnapi_model_->AddOutput(output.name(), shaper_[output.name()]);
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

void ModelBuilder::addOperations() {
  for (int i = 0; i < model_proto_.graph().node_size(); i++) {
    const auto& node = model_proto_.graph().node(i);
    const auto& op = node.op_type();

    // process skips (already used as activation)
    if (op == "Add") {
      const auto input1 = node.input(0);
      const auto input2 = node.input(1);
      const auto output = node.output(0);

      std::vector<uint32_t> input_indices;
      input_indices.push_back(operand_indexes_.at(input1));                    // input 1
      input_indices.push_back(operand_indexes_.at(input2));                    // input 2
      input_indices.push_back(OperandFromScalar(ANEURALNETWORKS_FUSED_NONE));  // fusecode
      shaper_.Eltwise(input1, input2, output);
      const OperandType output_operand_type = {operand_types_.at(input1).type, shaper_[output]};
      auto output_idx = AddOperation(ANEURALNETWORKS_ADD, input_indices, {output_operand_type})[0];
      RegisterOperand(output, output_idx, output_operand_type);
      //   AddLayerAdd(input1, input2, output);
    } else {
      throw std::invalid_argument("Unsupported operator " + op);
    }
  }
}

ModelBuilder::IndexSeq ModelBuilder::AddOperation(
    int op, IndexSeq input_indexes,
    std::vector<android::nn::wrapper::OperandType> types) {
  IndexSeq output_indexes;
  for (const auto& type : types) {
    auto index = AddNewOperand(type);
    output_indexes.push_back(index);
  }

  THROW_ON_ERROR_WITH_NOTE(
      nnapi_->ANeuralNetworksModel_addOperation(
          nnapi_model_->model_, op, input_indexes.size(), &input_indexes[0],
          output_indexes.size(), &output_indexes[0]),
      "op = " + std::to_string(op));

  return output_indexes;
}

std::unique_ptr<Model> ModelBuilder::Compile() {
  prepare();
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

  clearData();
  return std::move(nnapi_model_);
}
}  // namespace nnapi
}  // namespace onnxruntime