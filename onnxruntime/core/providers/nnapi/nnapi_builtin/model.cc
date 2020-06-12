// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "model.h"
#include "core/providers/nnapi/nnapi_builtin/nnapi_lib/nnapi_implementation.h"
#include "core/providers/nnapi/nnapi_builtin/builders/helper.h"

// Android only?
#include <sys/mman.h>
#include <unistd.h>

namespace onnxruntime {
namespace nnapi {

Model::Model() : nnapi_(NnApiImplementation()) {}

Model::~Model() {
  if (execution_)
    nnapi_->ANeuralNetworksExecution_free(execution_);

  nnapi_->ANeuralNetworksCompilation_free(compilation_);
  nnapi_->ANeuralNetworksModel_free(model_);
}

void Model::AddInput(const std::string& name, const Shaper::Shape& shape,
                     const android::nn::wrapper::OperandType& operand_type) {
  input_names_.push_back(name);
  operand_types_.insert({name, operand_type});
  shaper_.AddShape(name, shape);
}

void Model::AddOutput(const std::string& name, const Shaper::Shape& shape,
                      const android::nn::wrapper::OperandType& operand_type) {
  output_names_.push_back(name);
  operand_types_.insert({name, operand_type});
  shaper_.AddShape(name, shape);
}

const std::vector<std::string>& Model::GetInputs() const {
  return input_names_;
}

const std::vector<std::string>& Model::GetOutputs() const {
  return output_names_;
}

const android::nn::wrapper::OperandType& Model::GetInputType(const std::string& name) const {
  return operand_types_.at(name);
}

const android::nn::wrapper::OperandType Model::GetOutputType(
    const std::string& name, std::unordered_map<std::string, uint32_t> input_dim_param_values) const {
  const auto& output_type = operand_types_.at(name);
  if (HAS(output_dimension_map_, name)) {
    auto dimensions = output_type.dimensions;
    for (const auto& dim_map_entry : output_dimension_map_.at(name)) {
      const auto& dim_param = dim_map_entry.second;
      const auto dim_idx = dim_map_entry.first;

      if (dimensions.size() <= static_cast<size_t>(dim_idx))
        throw std::invalid_argument(
            "Invalid dim_idx " + std::to_string(dim_idx));
      if (HAS(input_dim_param_values, dim_param))
        dimensions[dim_idx] = input_dim_param_values[dim_param];
    }

    android::nn::wrapper::OperandType type(
        output_type.type, dimensions, output_type.operandType.scale, output_type.operandType.zeroPoint);

    return type;
  }

  return output_type;
}

void Model::SetInputMap(std::unordered_map<std::string, size_t>&& input_map) {
  input_map_ = std::move(input_map);
}

void Model::SetOutputMap(std::unordered_map<std::string, size_t>&& output_map) {
  output_map_ = std::move(output_map);
}

void Model::SetInputDimensionMap(int32_t input_idx, int32_t dim_idx, const std::string& dim_param) {
  input_dimension_map_[dim_param] = std::make_pair(input_idx, dim_idx);
}

void Model::SetOutputDimensionMap(const std::string& output_name, int32_t dim_idx, const std::string& dim_param) {
  output_dimension_map_[output_name][dim_idx] = dim_param;
}

size_t Model::GetMappedInputIdx(const std::string& name) const {
  return input_map_.at(name);
}

size_t Model::GetMappedOutputIdx(const std::string& name) const {
  return output_map_.at(name);
}

void Model::SetInputBuffer(const int32_t index, const InputOutputInfo& input) {
  if (!prepared_for_exe_) PrepareForExecution();
  THROW_ON_ERROR(nnapi_->ANeuralNetworksExecution_setInput(
      execution_, index, &input.type.operandType, input.buffer, input.type.GetOperandBlobByteSize()));
}

void Model::SetOutputBuffer(const int32_t index, const InputOutputInfo& output) {
  if (!prepared_for_exe_) PrepareForExecution();
  THROW_ON_ERROR(nnapi_->ANeuralNetworksExecution_setOutput(
      execution_, index, &output.type.operandType, output.buffer, output.type.GetOperandBlobByteSize()));
}

void Model::PrepareForExecution() {
  if (compilation_ == nullptr) {
    throw std::invalid_argument(
        "Error in PrepareForExecution, compilation_ == nullptr");
  }
  THROW_ON_ERROR(
      nnapi_->ANeuralNetworksExecution_create(compilation_, &execution_));
  prepared_for_exe_ = true;
}

void Model::ResetExecution() {
  nnapi_->ANeuralNetworksExecution_free(execution_);
  execution_ = nullptr;
  prepared_for_exe_ = false;
}

std::unordered_map<std::string, uint32_t>
Model::GetInputDimParamValues(const std::vector<InputOutputInfo>& inputs) {
  std::unordered_map<std::string, uint32_t> input_dim_param_values;
  for (const auto& entry : input_dimension_map_) {
    const auto& dim_param = entry.first;
    size_t input_idx;
    size_t dim_idx;
    std::tie(input_idx, dim_idx) = entry.second;

    if (inputs.size() <= input_idx)
      throw std::invalid_argument(
          "Invalid input_idx " + std::to_string(input_idx));

    const auto& input_info(inputs[input_idx]);
    if (input_info.type.dimensions.size() <= dim_idx)
      throw std::invalid_argument(
          "Invalid dim_idx " + std::to_string(dim_idx));

    input_dim_param_values[dim_param] = input_info.type.dimensions[dim_idx];
  }

  return input_dim_param_values;
}

void Model::Predict(const std::vector<InputOutputInfo>& inputs,
                    const std::vector<InputOutputInfo>& outputs) {
  if (!prepared_for_exe_) PrepareForExecution();

  SetInputs(inputs);
  SetOutputs(outputs);

  ANeuralNetworksEvent* event = nullptr;

  THROW_ON_ERROR(
      nnapi_->ANeuralNetworksExecution_startCompute(execution_, &event));

  THROW_ON_ERROR(
      nnapi_->ANeuralNetworksEvent_wait(event));

  nnapi_->ANeuralNetworksEvent_free(event);
  ResetExecution();
}

void Model::SetInputs(const std::vector<InputOutputInfo>& inputs) {
  for (size_t i = 0; i < inputs.size(); i++) {
    SetInputBuffer(i, inputs[i]);
  }
}

void Model::SetOutputs(const std::vector<InputOutputInfo>& outputs) {
  for (size_t i = 0; i < outputs.size(); i++) {
    SetOutputBuffer(i, outputs[i]);
  }
}

#ifdef USENNAPISHAREDMEM
Model::NNMemory::NNMemory(const NnApi* nnapi, const char* name, size_t size) {
  if (name && size > 0) {
    nnapi_ = nnapi;
    byte_size_ = size;
    fd_ = nnapi_->ASharedMemory_create(name, size);
    data_ptr_ = reinterpret_cast<uint8_t*>(
        mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
    THROW_ON_ERROR(nnapi_->ANeuralNetworksMemory_createFromFd(size, PROT_READ | PROT_WRITE,
                                                              fd_, 0, &nn_memory_handle_));
  }
}

Model::NNMemory::~NNMemory() {
  if (nn_memory_handle_) {
    nnapi_->ANeuralNetworksMemory_free(nn_memory_handle_);
  }
  if (data_ptr_) {
    munmap(data_ptr_, byte_size_);
  }

  if (fd_ > 0) close(fd_);
}
#else
Model::NNMemory::NNMemory(const NnApi* /*nnapi*/, const char* name, size_t size) {
  if (name && size > 0) {
    data_ = std::make_unique<std::vector<uint8_t> >(size);
  }
}
#endif

}  // namespace nnapi
}  // namespace onnxruntime