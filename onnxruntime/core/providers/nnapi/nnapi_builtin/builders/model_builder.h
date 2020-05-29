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

  IndexSeq input_index_vec_;
  IndexSeq output_index_vec_;

  uint32_t next_index_ = 0;

  std::pair<bool, std::string> IsNodeSupported(const ONNX_NAMESPACE::NodeProto& node);

  // Convert the onnx model to ANeuralNetworksModel
  void prepare();

  void addInitializers();
  void copyInitializersData();
  void registerModelInputs();
  void addOperations();
  void registerModelOutputs();
  void clearData();

  uint32_t SetOperandFromScalar(android::nn::wrapper::Type type, void* value, size_t size);
  uint32_t AddNewOperand(const android::nn::wrapper::OperandType& type);
  void RegisterOperand(const std::string& name,
                       Index index, const android::nn::wrapper::OperandType& operand_type);

  void AddOperation(int op, IndexSeq input_indices, std::vector<std::string> output_names,
                    std::vector<android::nn::wrapper::OperandType> types);
};

}  // namespace nnapi
}  // namespace onnxruntime