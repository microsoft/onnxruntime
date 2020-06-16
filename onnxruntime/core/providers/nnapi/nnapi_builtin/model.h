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

#define USENNAPISHAREDMEM 1

class Model {
  friend class ModelBuilder;

 public:
  struct InputOutputInfo {
    void* buffer{nullptr};
    android::nn::wrapper::OperandType type;
  };

#ifdef USENNAPISHAREDMEM
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
#else
  class NNMemory {
   public:
    NNMemory(const NnApi* /*nnapi*/, const char* name, size_t size);
    ~NNMemory() = default;
    uint8_t* get_data_ptr() { return data_->data(); }

   private:
    std::unique_ptr<std::vector<uint8_t>> data_;
  };
#endif

 public:
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  const std::vector<std::string>& GetInputs() const;
  const std::vector<std::string>& GetOutputs() const;
  const android::nn::wrapper::OperandType& GetInputType(const std::string& name) const;
  const android::nn::wrapper::OperandType GetOutputType(const std::string& name) const;

  void SetInputMap(std::unordered_map<std::string, size_t>&& input_map);
  void SetOutputMap(std::unordered_map<std::string, size_t>&& output_map);

  size_t GetMappedInputIdx(const std::string& name) const;
  size_t GetMappedOutputIdx(const std::string& name) const;

  void SetInputs(const std::vector<InputOutputInfo>& inputs);
  void SetOutputs(const std::vector<InputOutputInfo>& outputs);

  void Predict();

 private:
  const NnApi* nnapi_{nullptr};
  bool prepared_for_exe_ = false;

  ANeuralNetworksModel* model_{nullptr};
  ANeuralNetworksCompilation* compilation_{nullptr};
  ANeuralNetworksExecution* execution_{nullptr};

  std::unique_ptr<NNMemory> mem_initializers_;
  std::vector<std::unique_ptr<NNMemory>> mem_persist_buffers_;

  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::unordered_map<std::string, android::nn::wrapper::OperandType>
      operand_types_;

  Shaper shaper_;
  Shaper shaper_for_exeuction_;

  std::unordered_map<std::string, size_t> input_map_;
  std::unordered_map<std::string, size_t> output_map_;

  Model();
  void AddInput(const std::string& name, const android::nn::wrapper::OperandType& operand_type);
  void AddOutput(const std::string& name, const android::nn::wrapper::OperandType& operand_type);
  void SetShaper(const Shaper shaper) { shaper_ = shaper; }

  void SetInputBuffer(const int32_t index, const InputOutputInfo& input);
  void SetOutputBuffer(const int32_t index, const InputOutputInfo& output);

  void PrepareForExecution();
  void ResetExecution();
};

}  // namespace nnapi
}  // namespace onnxruntime