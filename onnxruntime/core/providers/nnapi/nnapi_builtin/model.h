// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "builders/Shaper.h"
#include "nnapi_lib/NeuralNetworksWrapper.h"

struct ANeuralNetworksModel;
struct ANeuralNetworksCompilation;
struct ANeuralNetworksExecution;
struct ANeuralNetworksMemory;
struct NnApi;

namespace onnxruntime {
namespace nnapi {

// Manage NNAPI shared memory handle
class NNMemory {
 public:
  NNMemory(const NnApi* nnapi, const char* name, size_t size);
  ~NNMemory();

  ANeuralNetworksMemory* get_handle() { return nn_memory_handle_; }
  uint8_t* get_data_ptr() { return data_ptr_; }

 private:
  // NnApi instance to use. Not owned by this object.
  const NnApi* nnapi_{nullptr};
  int fd_{0};
  size_t byte_size_{0};
  uint8_t* data_ptr_{nullptr};
  ANeuralNetworksMemory* nn_memory_handle_{nullptr};
};

struct InputOutputInfo {
  void* buffer{nullptr};
  android::nn::wrapper::OperandType type;
};

class Model {
  friend class ModelBuilder;

 public:
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  const std::vector<std::string>& GetInputs() const;
  const std::vector<std::string>& GetOutputs() const;
  const android::nn::wrapper::OperandType& GetType(const std::string& name) const;
  Shaper::Shape GetShape(const std::string& name);

  void SetOutputBuffer(const int32_t index, float* buffer);
  void SetOutputBuffer(const int32_t index, uint8_t* buffer);
  void SetOutputBuffer(const int32_t index, char* buffer);
  void SetOutputBuffer(const int32_t index, void* buffer,
                       const size_t elemsize);

  void Predict(const std::vector<InputOutputInfo>& inputs);

 private:
  const NnApi* nnapi_{nullptr};

  ANeuralNetworksModel* model_{nullptr};
  ANeuralNetworksCompilation* compilation_{nullptr};
  ANeuralNetworksExecution* execution_{nullptr};

  std::unique_ptr<NNMemory> mem_initializers_;
  std::vector<std::unique_ptr<NNMemory> > mem_persist_buffers_;

  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::map<std::string, android::nn::wrapper::OperandType> operand_types_;
  Shaper shaper_;

  Model();
  void AddInput(const std::string& name, const Shaper::Shape& shape,
                const android::nn::wrapper::OperandType& operand_type);
  void AddOutput(const std::string& name, const Shaper::Shape& shape,
                 const android::nn::wrapper::OperandType& operand_type);

  void SetInputBuffer(const int32_t index, const InputOutputInfo& input);
  void PrepareForExecution();
  void ResetExecution();
  void PredictAfterSetInputBuffer();
  bool prepared_for_exe_ = false;
};

}  // namespace nnapi
}  // namespace onnxruntime