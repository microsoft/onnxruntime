// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "builders/shaper.h"
#include "nnapi_lib/NeuralNetworksWrapper.h"

namespace onnxruntime {
namespace nnapi {

#define USENNAPISHAREDMEM 1

class Model {
  friend class ModelBuilder;

 public:
  struct InputBuffer {
    const void* buffer{nullptr};
    android::nn::wrapper::OperandType type;
  };

  struct OutputBuffer {
    void* buffer{nullptr};
    android::nn::wrapper::OperandType type;
  };

  // Memory for persist data such as initializers and intermediate result
#ifdef USENNAPISHAREDMEM
  // Use NNAPI shared memory
  class NNMemory {
   public:
    NNMemory(const NnApi* nnapi, const char* name, size_t size);
    ~NNMemory();

    ANeuralNetworksMemory* GetHandle() { return nn_memory_handle_; }
    uint8_t* GetDataPtr() { return data_ptr_; }

   private:
    // NnApi instance to use. Not owned by this object.
    const NnApi* nnapi_{nullptr};
    int fd_{0};
    size_t byte_size_{0};
    uint8_t* data_ptr_{nullptr};
    ANeuralNetworksMemory* nn_memory_handle_{nullptr};
  };
#else
  // Use system memory buffer
  class NNMemory {
   public:
    NNMemory(const NnApi* /*nnapi*/, const char* name, size_t size);
    ~NNMemory() = default;
    uint8_t* GetDataPtr() { return data_.data(); }

   private:
    std::vector<uint8_t> data_;
  };
#endif

 public:
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  // Get the names of inputs/outputs
  // in the order of NNAPI inputs/outputs order
  const std::vector<std::string>& GetInputs() const;
  const std::vector<std::string>& GetOutputs() const;

  // Get the input/output type of a particular input/output
  // Returns the data type and dimension of the given input/output
  // Please note the output type will have updated dimensions
  const android::nn::wrapper::OperandType& GetInputType(const std::string& name) const;
  const android::nn::wrapper::OperandType GetOutputType(const std::string& name) const;

  // Set the mapping between input/output name and ORT kernel context
  // input/output index, at execution time
  void SetInputMap(std::unordered_map<std::string, size_t>&& input_map);
  void SetOutputMap(std::unordered_map<std::string, size_t>&& output_map);

  // Get the ORT kernel context input/output index with given name
  size_t GetMappedInputIdx(const std::string& name) const;
  size_t GetMappedOutputIdx(const std::string& name) const;

  // Set the input/output data buffers
  // These need to be called before calling Predict()
  void SetInputBuffers(const std::vector<InputBuffer>& inputs);
  void SetOutputBuffers(const std::vector<OutputBuffer>& outputs);

  // Execute the NNAPI model
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

  void SetInputBuffer(const int32_t index, const InputBuffer& input);
  void SetOutputBuffer(const int32_t index, const OutputBuffer& output);

  void PrepareForExecution();
  void ResetExecution();
};

}  // namespace nnapi
}  // namespace onnxruntime