// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "builders/Shaper.h"

struct ANeuralNetworksModel;
struct ANeuralNetworksCompilation;
struct ANeuralNetworksExecution;
struct NnApi;

namespace onnxruntime {
namespace nnapi {

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
  ANeuralNetworksModel* model_{nullptr};
  ANeuralNetworksCompilation* compilation_{nullptr};
  ANeuralNetworksExecution* execution_{nullptr};
  const NnApi* nnapi_{nullptr};

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
  void PredictAfterSetInputBuffer();
  bool prepared_for_exe_ = false;
};

}  // namespace nnapi
}  // namespace onnxruntime