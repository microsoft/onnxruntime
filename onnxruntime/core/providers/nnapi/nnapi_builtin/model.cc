// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "model.h"
#include "core/providers/nnapi/nnapi_builtin/builders/helper.h"
#include "core/providers/nnapi/nnapi_builtin/nnapi_lib/nnapi_implementation.h"

#ifdef USENNAPISHAREDMEM
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace onnxruntime {
namespace nnapi {

Model::Model() : nnapi_(NnApiImplementation()) {}

Model::~Model() {
  if (execution_)
    nnapi_->ANeuralNetworksExecution_free(execution_);

  nnapi_->ANeuralNetworksCompilation_free(compilation_);
  nnapi_->ANeuralNetworksModel_free(model_);
}

void Model::AddInput(const std::string& name, const android::nn::wrapper::OperandType& operand_type) {
  input_names_.push_back(name);
  operand_types_.insert({name, operand_type});
}

void Model::AddOutput(const std::string& name, const android::nn::wrapper::OperandType& operand_type) {
  output_names_.push_back(name);
  operand_types_.insert({name, operand_type});
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

const android::nn::wrapper::OperandType Model::GetOutputType(const std::string& name) const {
  const auto& output_type = operand_types_.at(name);
  android::nn::wrapper::OperandType type(
      output_type.type, shaper_for_exeuction_[name], output_type.operandType.scale, output_type.operandType.zeroPoint);

  return type;
}

void Model::SetInputMap(std::unordered_map<std::string, size_t>&& input_map) {
  input_map_ = std::move(input_map);
}

void Model::SetOutputMap(std::unordered_map<std::string, size_t>&& output_map) {
  output_map_ = std::move(output_map);
}

size_t Model::GetMappedInputIdx(const std::string& name) const {
  return input_map_.at(name);
}

size_t Model::GetMappedOutputIdx(const std::string& name) const {
  return output_map_.at(name);
}

void Model::SetInputBuffer(const int32_t index, const InputBuffer& input) {
  PrepareForExecution();

  THROW_ON_ERROR(nnapi_->ANeuralNetworksExecution_setInput(
      execution_, index, &input.type.operandType, input.buffer, input.type.GetOperandBlobByteSize()));
}

void Model::SetOutputBuffer(const int32_t index, const OutputBuffer& output) {
  PrepareForExecution();

  THROW_ON_ERROR(nnapi_->ANeuralNetworksExecution_setOutput(
      execution_, index, &output.type.operandType, output.buffer, output.type.GetOperandBlobByteSize()));
}

void Model::PrepareForExecution() {
  if (prepared_for_exe_)
    return;

  ORT_ENFORCE(nullptr != compilation_,
              "Error in PrepareForExecution, compilation_ is null");

  // Copy the shaper for calculate the dynamic output shape
  // based on the input shape
  shaper_for_exeuction_ = shaper_;

  THROW_ON_ERROR(
      nnapi_->ANeuralNetworksExecution_create(compilation_, &execution_));
  prepared_for_exe_ = true;
}

void Model::ResetExecution() {
  nnapi_->ANeuralNetworksExecution_free(execution_);
  execution_ = nullptr;
  shaper_for_exeuction_.Clear();
  prepared_for_exe_ = false;
}

void Model::Predict() {
  PrepareForExecution();

  ANeuralNetworksEvent* event = nullptr;
  THROW_ON_ERROR(nnapi_->ANeuralNetworksExecution_startCompute(execution_, &event));

  THROW_ON_ERROR(nnapi_->ANeuralNetworksEvent_wait(event));

  nnapi_->ANeuralNetworksEvent_free(event);

  ResetExecution();
}

void Model::SetInputBuffers(const std::vector<InputBuffer>& inputs) {
  PrepareForExecution();

  for (size_t i = 0; i < inputs.size(); i++) {
    SetInputBuffer(i, inputs[i]);
    shaper_for_exeuction_.UpdateShape(input_names_[i], inputs[i].type.dimensions);
  }

  shaper_for_exeuction_.UpdateDynamicDimensions();
}

void Model::SetOutputBuffers(const std::vector<OutputBuffer>& outputs) {
  PrepareForExecution();

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
    data_.resize(size);
  }
}
#endif

}  // namespace nnapi
}  // namespace onnxruntime