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
  nnapi_->ANeuralNetworksExecution_free(execution_);
  nnapi_->ANeuralNetworksCompilation_free(compilation_);
  nnapi_->ANeuralNetworksModel_free(model_);
}

Shaper::Shape Model::GetShape(const std::string& name) {
  return shaper_[name];
}

void Model::AddInput(const std::string& name, const Shaper::Shape& shape) {
  input_names_.push_back(name);
  shaper_.AddShape(name, shape);
}

void Model::AddOutput(const std::string& name, const Shaper::Shape& shape) {
  output_names_.push_back(name);
  shaper_.AddShape(name, shape);
}

std::vector<std::string> Model::GetInputs() {
  return input_names_;
}

std::vector<std::string> Model::GetOutputs() {
  return output_names_;
}

void Model::SetInputBuffer(const int32_t index, const float* buffer) {
  SetInputBuffer(index, buffer, 4);
}

void Model::SetInputBuffer(const int32_t index, const uint8_t* buffer) {
  SetInputBuffer(index, buffer, 1);
}

void Model::SetInputBuffer(const int32_t index, const void* buffer,
                           const size_t elemsize) {
  if (!prepared_for_exe_) PrepareForExecution();
  auto size = shaper_.GetSize(input_names_[index]) * elemsize;
  THROW_ON_ERROR(nnapi_->ANeuralNetworksExecution_setInput(
      execution_, index, nullptr, buffer, size))
}

void Model::SetOutputBuffer(const int32_t index, float* buffer) {
  SetOutputBuffer(index, buffer, 4);
}

void Model::SetOutputBuffer(const int32_t index, uint8_t* buffer) {
  SetOutputBuffer(index, buffer, 1);
}

void Model::SetOutputBuffer(const int32_t index, char* buffer) {
  SetOutputBuffer(index, reinterpret_cast<uint8_t*>(buffer));
}

void Model::SetOutputBuffer(const int32_t index, void* buffer,
                            const size_t elemsize) {
  if (!prepared_for_exe_) PrepareForExecution();
  auto size = shaper_.GetSize(output_names_[index]) * elemsize;
  THROW_ON_ERROR(nnapi_->ANeuralNetworksExecution_setOutput(
      execution_, index, nullptr, buffer, size))
}

void Model::PredictAfterSetInputBuffer() {
  ANeuralNetworksEvent* event = nullptr;
  THROW_ON_ERROR(
      nnapi_->ANeuralNetworksExecution_startCompute(execution_, &event));
  THROW_ON_ERROR(nnapi_->ANeuralNetworksEvent_wait(event));

  nnapi_->ANeuralNetworksEvent_free(event);
  nnapi_->ANeuralNetworksExecution_free(execution_);
  prepared_for_exe_ = false;
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

template <typename T>
void Model::Predict(const std::vector<T>& input) {
  // const_cast is a ugly workaround, vector<const T*> causes strange errors
  Predict<T>({const_cast<T*>(input.data())});
}

template <typename T>
void Model::Predict(const std::vector<std::vector<T>>& inputs) {
  std::vector<T*> input_ptrs;
  for (size_t i = 0; i < inputs.size(); i++) {
    auto& input = inputs[i];
    // const_cast is a ugly workaround, vector<const T*> causes strange
    // errors
    input_ptrs.push_back(const_cast<T*>(input.data()));
  }
  Predict<T>(input_ptrs);
}

template <typename T>
void Model::Predict(const T* input) {
  // Predict<T>({input}) doesn't compile. Have no idea why.
  std::vector<T*> inputs;
  inputs.push_back(const_cast<T*>(input));
  Predict<T>(inputs);
}

template <typename T>
void Model::Predict(const std::vector<T*>& inputs) {
  if (!prepared_for_exe_) PrepareForExecution();
  for (size_t i = 0; i < inputs.size(); i++) {
    SetInputBuffer(i, inputs[i]);
  }
  PredictAfterSetInputBuffer();
}

void Model::Predict(const std::vector<float*>& inputs) {
  if (!prepared_for_exe_) PrepareForExecution();
  for (size_t i = 0; i < inputs.size(); i++) {
    SetInputBuffer(i, inputs[i]);
  }
  PredictAfterSetInputBuffer();
}

NNMemory::NNMemory(const NnApi* nnapi, const char* name, size_t size) {
  LOGI("NNMemory ctor name %s size %zu", name, size);

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

NNMemory::~NNMemory() {
  if (nn_memory_handle_) {
    nnapi_->ANeuralNetworksMemory_free(nn_memory_handle_);
  }
  if (data_ptr_) {
    munmap(data_ptr_, byte_size_);
  }

  if (fd_ > 0) close(fd_);
}

}  // namespace nnapi
}  // namespace onnxruntime