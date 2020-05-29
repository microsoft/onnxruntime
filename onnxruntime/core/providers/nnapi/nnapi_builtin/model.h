// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "builders/Shaper.h"

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

class Model {
  friend class ModelBuilder;

 public:
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;
  std::vector<std::string> GetInputs();
  std::vector<std::string> GetOutputs();
  Shaper::Shape GetShape(const std::string& name);
  void SetOutputBuffer(const int32_t index, float* buffer);
  void SetOutputBuffer(const int32_t index, uint8_t* buffer);
  void SetOutputBuffer(const int32_t index, char* buffer);
  void SetOutputBuffer(const int32_t index, void* buffer,
                       const size_t elemsize);

  template <typename T>
  void Predict(const std::vector<T>& input);
  template <typename T>
  void Predict(const std::vector<std::vector<T>>& inputs);
  template <typename T>
  void Predict(const T* input);
  template <typename T>
  void Predict(const std::vector<T*>& inputs);
  void Predict(const std::vector<float*>& inputs);

 private:
  const NnApi* nnapi_{nullptr};

  ANeuralNetworksModel* model_{nullptr};
  ANeuralNetworksCompilation* compilation_{nullptr};
  ANeuralNetworksExecution* execution_{nullptr};

  std::unique_ptr<NNMemory> mem_initializers;

  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  Shaper shaper_;

  Model();
  void AddInput(const std::string& name, const Shaper::Shape& shape);
  void AddOutput(const std::string& name, const Shaper::Shape& shape);

  void SetInputBuffer(const int32_t index, const float* buffer);
  void SetInputBuffer(const int32_t index, const uint8_t* buffer);
  void SetInputBuffer(const int32_t index, const void* buffer,
                      const size_t elemsize);
  void PrepareForExecution();
  void ResetExecution();
  void PredictAfterSetInputBuffer();
  bool prepared_for_exe_ = false;
};

}  // namespace nnapi
}  // namespace onnxruntime