// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <onnx/onnx_pb.h>
#include <unordered_set>
#include "core/providers/nnapi/nnapi_builtin/nnapi_lib/NeuralNetworksWrapper.h"
#include "Shaper.h"
#include "core/providers/nnapi/nnapi_builtin/model.h"

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

 private:
  const NnApi* nnapi_{nullptr};
  ONNX_NAMESPACE::ModelProto& model_proto_;
  std::unique_ptr<Model> nnapi_model_;

  Shaper shaper_;

  std::map<std::string, uint32_t> operand_indexes_;
  std::map<std::string, android::nn::wrapper::OperandType> operand_types_;

  std::unordered_set<std::string> operands_;
  std::map<std::string, const ONNX_NAMESPACE::TensorProto&> initializers_;
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

  uint32_t AddOperandFromPersistMemoryBuffer(
      const std::string& name, const void* buffer,
      const android::nn::wrapper::OperandType& operand_type);

  uint32_t AddNHWCInitializer(const std::string& name);
  uint32_t Add1230Initializer(const std::string& name);

  uint32_t SetOperandFromScalar(android::nn::wrapper::Type type, const void* value, size_t size);
  uint32_t AddNewOperand(const android::nn::wrapper::OperandType& type);
  void RegisterOperand(const std::string& name,
                       Index index, const android::nn::wrapper::OperandType& operand_type);

  void AddOperation(int op, IndexSeq input_indices, std::vector<std::string> output_names,
                    std::vector<android::nn::wrapper::OperandType> types);
};

}  // namespace nnapi
}  // namespace onnxruntime