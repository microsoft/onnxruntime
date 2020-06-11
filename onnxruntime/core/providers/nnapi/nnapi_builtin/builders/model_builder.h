// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <onnx/onnx_pb.h>
#include <unordered_set>

#include "core/providers/nnapi/nnapi_builtin/model.h"
#include "core/providers/nnapi/nnapi_builtin/nnapi_lib/NeuralNetworksWrapper.h"
#include "Shaper.h"

namespace onnxruntime {
namespace nnapi {

class ModelBuilder {
 public:
  using Index = uint32_t;
  using IndexSeq = std::vector<Index>;
  using Shape = Shaper::Shape;

  ModelBuilder(ONNX_NAMESPACE::ModelProto& model_proto);
  ~ModelBuilder() = default;
  std::vector<std::vector<int>> GetSupportedNodes();
  std::unique_ptr<Model> Compile();

  int32_t GetAndroidSdkVer() const;

  void AddOperation(int op, IndexSeq input_indices, std::vector<std::string> output_names,
                    std::vector<android::nn::wrapper::OperandType> types);
  void RegisterOperand(const std::string& name, Index index,
                       const android::nn::wrapper::OperandType& operand_type);

  uint32_t AddOperandFromPersistMemoryBuffer(const std::string& name, const void* buffer,
                                             const android::nn::wrapper::OperandType& operand_type);

  int32_t FindActivation(const std::string& output);
  Shaper& GetShaper() { return shaper_; }

  uint32_t AddOperandFromScalar(bool value);
  uint32_t AddOperandFromScalar(float value);
  uint32_t AddOperandFromScalar(int32_t value);

  void AddSkippedInitializer(const std::string& tensor_name);

  // Accessors for members
  const std::unordered_map<std::string, uint32_t>&
  GetOperandIndices() const { return operand_indices_; }

  const std::unordered_map<std::string, android::nn::wrapper::OperandType>&
  GetOperandTypes() const { return operand_types_; }

  const std::unordered_set<std::string>&
  GetSkippedActivations() const { return skipped_activations_; }

  const std::unordered_map<std::string,
                           const ONNX_NAMESPACE::TensorProto&>&
  GetInitializerTensors() const { return initializers_; }

  const ONNX_NAMESPACE::ModelProto& GetOnnxModel() const { return model_proto_; }
  bool UseNCHW() const { return use_nchw_; }

 private:
  const NnApi* nnapi_{nullptr};
  ONNX_NAMESPACE::ModelProto& model_proto_;
  std::unique_ptr<Model> nnapi_model_;

  bool use_nchw_{true};

  Shaper shaper_;

  std::unordered_map<std::string, uint32_t> operand_indices_;
  std::unordered_map<std::string, android::nn::wrapper::OperandType> operand_types_;

  std::unordered_set<std::string> operands_;
  std::unordered_set<std::string> skipped_activations_;

  std::unordered_map<std::string, const ONNX_NAMESPACE::TensorProto&> initializers_;
  std::unordered_set<std::string> skipped_initializers_;

  IndexSeq input_index_vec_;
  IndexSeq output_index_vec_;

  uint32_t next_index_ = 0;

  std::pair<bool, std::string> IsNodeSupported(const ONNX_NAMESPACE::NodeProto& node);

  // Convert the onnx model to ANeuralNetworksModel
  void Prepare();

  void GetAllIntializers();
  void PreprocessIntializers();
  void RegisterInitializers();
  void RegisterModelInputs();
  void AddOperations();
  void RegisterModelOutputs();
  void ClearData();

  void SetOperandValue(ModelBuilder::Index index,
                       NNMemory* memory,
                       size_t size, size_t offset);

  uint32_t SetOperandFromScalar(android::nn::wrapper::Type type, const void* value, size_t size);
  uint32_t AddNewOperand(const android::nn::wrapper::OperandType& type);
};

}  // namespace nnapi
}  // namespace onnxruntime