// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_set>

#include "builders/shaper.h"
#include "core/platform/ort_mutex.h"
#include "nnapi_lib/NeuralNetworksWrapper.h"

struct NnApi;

namespace onnxruntime {
namespace nnapi {

#define USENNAPISHAREDMEM 1

class Execution;

class Model {
  friend class ModelBuilder;

 public:
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
    int fd_{-1};
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
  android::nn::wrapper::OperandType GetOutputType(const std::string& name, const Execution& execution) const;

  // Set the mapping between input/output name and ORT kernel context
  // input/output index, at execution time
  void SetInputMap(std::unordered_map<std::string, size_t>&& input_map);
  void SetOutputMap(std::unordered_map<std::string, size_t>&& output_map);

  // Get the ORT kernel context input/output index with given name
  size_t GetMappedInputIdx(const std::string& name) const;
  size_t GetMappedOutputIdx(const std::string& name) const;

  // If we support the dynamic output shape,
  // This is only for the case where output size cannot be determined at model execution time
  // Do not use this for case a determined output shape can be returned from GetOutputType()
  bool SupportsDynamicOutputShape() const;

  // Set and Get the number of elements in the buffer for a dynamic output
  // If the buffer is not big enough, ANEURALNETWORKS_OUTPUT_INSUFFICIENT_SIZE will be returned by execution
  // Note: this will return number of elements of the buffer not the byte size of the buffer
  //       and each output will have its separated buffer
  // TODO:
  // 1. Consider an adaptive approach to automatically increase the buffer size if the execution reports
  //    insufficient size
  // 2. Experiment with bigger initial buffer size (currently 1024)
  size_t GetDynamicOutputBufferSize() const { return dynamic_output_buffer_size_; }
  void SetDynamicOutputBufferSize(size_t size) { dynamic_output_buffer_size_ = size; }

  // Mutex for exclusive lock to this model object
  OrtMutex& GetMutex() { return mutex_; }

  // If the given output is a scalar output
  // Since NNAPI does not support tensor with empty shape (scalar), we use {1} tensor for scalar in NNAPI
  // this output may need special handling
  bool IsScalarOutput(const std::string& output_name) const;

  Status PrepareForExecution(std::unique_ptr<Execution>& execution) ORT_MUST_USE_RESULT;

 private:
  const NnApi* nnapi_{nullptr};

  ANeuralNetworksModel* model_{nullptr};
  ANeuralNetworksCompilation* compilation_{nullptr};

  size_t dynamic_output_buffer_size_{1024};

  std::unique_ptr<NNMemory> mem_initializers_;
  std::vector<std::unique_ptr<NNMemory>> mem_persist_buffers_;

  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::unordered_map<std::string, android::nn::wrapper::OperandType> operand_types_;
  std::unordered_set<std::string> scalar_outputs_;

  Shaper shaper_;

  std::unordered_map<std::string, size_t> input_map_;
  std::unordered_map<std::string, size_t> output_map_;

  // We may transpose the nnapi output to nchw with a different name
  // This is map is to lookup the nnapi output from the onnx output
  std::unordered_map<std::string, std::string> onnx_to_nnapi_output_map_;

  OrtMutex mutex_;

  Model();
  void AddInput(const std::string& name, const android::nn::wrapper::OperandType& operand_type);

  // It is possible that the actual output from NNAPI model is not the same as the name of
  // the output from the onnx model, need to have both names and add mapping between them
  void AddOutput(const std::string& onnx_output_name,
                 const std::string& nnapi_output_name,
                 const android::nn::wrapper::OperandType& operand_type);

  void AddScalarOutput(const std::string& output_name);

  void SetShaper(const Shaper shaper) { shaper_ = shaper; }

  int32_t GetNNAPIFeatureLevel() const;
};

class Execution {
 public:
  struct InputBuffer {
    const std::string& name;
    const void* buffer{nullptr};
    android::nn::wrapper::OperandType type;
  };

  struct OutputBuffer {
    void* buffer{nullptr};
    android::nn::wrapper::OperandType type;
    size_t buffer_byte_size;
  };

 public:
  explicit Execution(ANeuralNetworksExecution& execution, const Shaper& shaper);
  ~Execution();
  Execution(const Execution&) = delete;
  Execution& operator=(const Execution&) = delete;

  const Shaper& GetShaper() const { return shaper_; }

  // Set the input/output data buffers
  // These need to be called before calling Predict()
  Status SetInputBuffers(const std::vector<InputBuffer>& inputs) ORT_MUST_USE_RESULT;
  Status SetOutputBuffers(const std::vector<OutputBuffer>& outputs) ORT_MUST_USE_RESULT;

  // Execute the NNAPI model
  // if there is dynamic output shape, will output the actual output shapes
  Status Predict(const std::vector<int32_t>& dynamic_outputs, std::vector<Shaper::Shape>& dynamic_output_shapes)
      ORT_MUST_USE_RESULT;

 private:
  Status SetInputBuffer(const int32_t index, const InputBuffer& input) ORT_MUST_USE_RESULT;
  Status SetOutputBuffer(const int32_t index, const OutputBuffer& output) ORT_MUST_USE_RESULT;

  const NnApi* nnapi_{nullptr};
  ANeuralNetworksExecution* execution_;
  Shaper shaper_;
};

}  // namespace nnapi
}  // namespace onnxruntime