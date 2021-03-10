// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/common/logging/logging.h>

#include "model.h"
#include "core/providers/common.h"
#include "core/providers/nnapi/nnapi_builtin/builders/helper.h"
#include "core/providers/nnapi/nnapi_builtin/nnapi_lib/nnapi_implementation.h"

#ifdef USENNAPISHAREDMEM
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace onnxruntime {
namespace nnapi {

#pragma region Model

Model::Model() : nnapi_(NnApiImplementation()) {}

Model::~Model() {
  nnapi_->ANeuralNetworksCompilation_free(compilation_);
  nnapi_->ANeuralNetworksModel_free(model_);
}

void Model::AddInput(const std::string& name, const android::nn::wrapper::OperandType& operand_type) {
  input_names_.push_back(name);
  operand_types_.emplace(name, operand_type);
}

void Model::AddOutput(const std::string& onnx_output_name,
                      const std::string& nnapi_output_name,
                      const android::nn::wrapper::OperandType& operand_type) {
  LOGS_DEFAULT(VERBOSE) << "Model::AddOutput output name " << onnx_output_name
                        << " shape " << Shape2String(operand_type.dimensions);

  output_names_.push_back(onnx_output_name);
  onnx_to_nnapi_output_map_.emplace(onnx_output_name, nnapi_output_name);
  operand_types_.emplace(nnapi_output_name, operand_type);
}

bool Model::IsScalarOutput(const std::string& output_name) const {
  return Contains(scalar_outputs_, output_name);
}

void Model::AddScalarOutput(const std::string& output_name) {
  scalar_outputs_.insert(output_name);
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

android::nn::wrapper::OperandType Model::GetOutputType(const std::string& name, const Execution& execution) const {
  const auto& nnapi_output_name = onnx_to_nnapi_output_map_.at(name);
  const auto& output_type = operand_types_.at(nnapi_output_name);
  android::nn::wrapper::OperandType type(
      output_type.type, execution.GetShaper()[nnapi_output_name], output_type.operandType.scale, output_type.operandType.zeroPoint);

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

bool Model::SupportsDynamicOutputShape() const {
  // dynamic output shape is only supported on Android API level 29+
  return GetAndroidSdkVer() >= 29 && dynamic_output_buffer_size_ > 0;
}

Status Model::PrepareForExecution(std::unique_ptr<Execution>& execution) {
  ORT_RETURN_IF_NOT(nullptr != compilation_,
                    "Error in PrepareForExecution, compilation_ is null");

  ANeuralNetworksExecution* nnapi_execution;
  RETURN_STATUS_ON_ERROR(
      nnapi_->ANeuralNetworksExecution_create(compilation_, &nnapi_execution));

  execution.reset(new Execution(*nnapi_execution, shaper_));
  return Status::OK();
}

int32_t Model::GetAndroidSdkVer() const {
  return nnapi_ ? nnapi_->android_sdk_version : 0;
}

#pragma region Model::NNMemory

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

  if (fd_ >= 0) close(fd_);
}
#else
Model::NNMemory::NNMemory(const NnApi* /*nnapi*/, const char* name, size_t size) {
  if (name && size > 0) {
    data_.resize(size);
  }
}
#endif

#pragma endregion

#pragma endregion

#pragma region Execution

Execution::Execution(ANeuralNetworksExecution& execution, const Shaper& shaper)
    : nnapi_(NnApiImplementation()),
      execution_(&execution),
      shaper_(shaper) {
}

Execution::~Execution() {
  nnapi_->ANeuralNetworksExecution_free(execution_);
}

Status Execution::SetInputBuffers(const std::vector<InputBuffer>& inputs) {
  for (size_t i = 0; i < inputs.size(); i++) {
    const auto& input(inputs[i]);
    ORT_RETURN_IF_ERROR(SetInputBuffer(i, input));
    ORT_RETURN_IF_ERROR(shaper_.UpdateShape(input.name, input.type.dimensions));
  }

  ORT_RETURN_IF_ERROR(shaper_.UpdateDynamicDimensions());
  return Status::OK();
}

Status Execution::SetOutputBuffers(const std::vector<OutputBuffer>& outputs) {
  for (size_t i = 0; i < outputs.size(); i++) {
    ORT_RETURN_IF_ERROR(SetOutputBuffer(i, outputs[i]));
  }

  return Status::OK();
}

Status Execution::SetInputBuffer(const int32_t index, const InputBuffer& input) {
  RETURN_STATUS_ON_ERROR(nnapi_->ANeuralNetworksExecution_setInput(
      execution_, index, &input.type.operandType, input.buffer, input.type.GetOperandBlobByteSize()));

  return Status::OK();
}

Status Execution::SetOutputBuffer(const int32_t index, const OutputBuffer& output) {
  LOGS_DEFAULT(VERBOSE) << "Model::SetOutputBuffer, output shape "
                        << Shape2String(output.type.dimensions);

  RETURN_STATUS_ON_ERROR(nnapi_->ANeuralNetworksExecution_setOutput(
      execution_, index, &output.type.operandType, output.buffer, output.buffer_byte_size));

  return Status::OK();
}

Status Execution::Predict(const std::vector<int32_t>& dynamic_outputs, std::vector<Shaper::Shape>& dynamic_output_shapes) {
  ANeuralNetworksEvent* event = nullptr;
  RETURN_STATUS_ON_ERROR(nnapi_->ANeuralNetworksExecution_startCompute(execution_, &event));
  RETURN_STATUS_ON_ERROR(nnapi_->ANeuralNetworksEvent_wait(event));
  nnapi_->ANeuralNetworksEvent_free(event);

  dynamic_output_shapes.clear();
  dynamic_output_shapes.reserve(dynamic_outputs.size());
  for (const int32_t i : dynamic_outputs) {
    uint32_t output_rank = 0;
    RETURN_STATUS_ON_ERROR(nnapi_->ANeuralNetworksExecution_getOutputOperandRank(execution_, i, &output_rank));

    std::vector<uint32_t> output_shape(output_rank);
    RETURN_STATUS_ON_ERROR(nnapi_->ANeuralNetworksExecution_getOutputOperandDimensions(execution_, i, output_shape.data()));

    dynamic_output_shapes.push_back(output_shape);
  }

  return Status::OK();
}

#pragma endregion

}  // namespace nnapi
}  // namespace onnxruntime