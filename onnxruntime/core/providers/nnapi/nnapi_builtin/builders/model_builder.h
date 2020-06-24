// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <onnx/onnx_pb.h>
#include <unordered_set>

#include "core/providers/nnapi/nnapi_builtin/model.h"
#include "core/providers/nnapi/nnapi_builtin/nnapi_lib/NeuralNetworksWrapper.h"
#include "op_builder.h"
#include "shaper.h"

namespace onnxruntime {
namespace nnapi {

class ModelBuilder {
 public:
  using Shape = Shaper::Shape;

  ModelBuilder(ONNX_NAMESPACE::ModelProto& model_proto);
  ~ModelBuilder() = default;

  std::vector<std::vector<int>> GetSupportedNodes();

  std::unique_ptr<Model> Compile();

  int32_t GetAndroidSdkVer() const;

  // Add an NNAPI operation (operator)
  void AddOperation(int op, const std::vector<uint32_t>& input_indices,
                    const std::vector<std::string>& output_names,
                    const std::vector<android::nn::wrapper::OperandType>& types);

  // Find if an output has a fuseable activation (Relu)
  int32_t FindActivation(const std::string& output);

  // Add an NNAPI scalar operand
  uint32_t AddOperandFromScalar(bool value);
  uint32_t AddOperandFromScalar(float value);
  uint32_t AddOperandFromScalar(int32_t value);

  // Add an NNAPI tensor operand (and allocate persist buffer)
  uint32_t AddOperandFromPersistMemoryBuffer(const std::string& name, const void* buffer,
                                             const android::nn::wrapper::OperandType& operand_type);

  // The initializer will be processed separately, skip it as an initializer
  void AddInitializerToSkip(const std::string& tensor_name);

  // Register informations for a particular operand
  void RegisterOperand(const std::string& name, uint32_t index,
                       const android::nn::wrapper::OperandType& operand_type);

  // Generate an unique name for intermediate result
  std::string GetUniqueName(const std::string& base_name);

  // Enable and disable using NCHW
  void SetUseNCHW(bool use_nchw) { use_nchw_ = use_nchw; }
  bool UseNCHW() const { return use_nchw_; }

  // Relax fp32 computation to fp16
  // It is off by default
  void SetUseFp16(bool use_fp16) { use_fp16_ = use_fp16; }

  // Set NNAPI execution preference
  // Default preference is PREFER_SUSTAINED_SPEED
  void ExecutePreference(
      android::nn::wrapper::ExecutePreference pref) { exe_pref_ = pref; }

  // Accessors for members
  Shaper& GetShaper() { return shaper_; }

  const std::unordered_map<std::string, uint32_t>&
  GetOperandIndices() const { return operand_indices_; }

  const std::unordered_map<std::string, android::nn::wrapper::OperandType>&
  GetOperandTypes() const { return operand_types_; }

  const std::unordered_set<std::string>&
  GetFusedActivations() const { return fused_activations_; }

  const std::unordered_map<std::string,
                           const ONNX_NAMESPACE::TensorProto&>&
  GetInitializerTensors() const { return initializers_; }

  const ONNX_NAMESPACE::ModelProto& GetOnnxModel() const { return model_proto_; }

 private:
  const NnApi* nnapi_{nullptr};
  ONNX_NAMESPACE::ModelProto& model_proto_;
  std::unique_ptr<Model> nnapi_model_;

  uint32_t name_token_{0};

  bool use_nchw_{true};
  bool use_fp16_{false};
  android::nn::wrapper::ExecutePreference exe_pref_{
      android::nn::wrapper::ExecutePreference::PREFER_FAST_SINGLE_ANSWER};

  Shaper shaper_;

  std::unordered_map<std::string, uint32_t> operand_indices_;
  std::unordered_map<std::string, android::nn::wrapper::OperandType> operand_types_;

  std::unordered_set<std::string> operands_;
  std::unordered_set<std::string> fused_activations_;

  std::unordered_map<std::string, const ONNX_NAMESPACE::TensorProto&> initializers_;
  std::unordered_set<std::string> skipped_initializers_;

  std::unordered_map<std::string, std::shared_ptr<IOpBuilder>> op_builders_;

  std::vector<uint32_t> input_index_vec_;
  std::vector<uint32_t> output_index_vec_;

  std::unordered_set<std::string> unique_names_;

  uint32_t next_index_ = 0;

  bool IsNodeSupported(const ONNX_NAMESPACE::NodeProto& node);

  // Convert the onnx model to ANeuralNetworksModel
  void Prepare();

  void GetAllInitializers();
  void PreprocessInitializers();
  void RegisterInitializers();
  void RegisterModelInputs();
  void AddOperations();
  void RegisterModelOutputs();
  void RegisterModelShaper();

  void SetOperandValue(uint32_t index, Model::NNMemory* memory,
                       size_t size, size_t offset);

  uint32_t AddNewNNAPIOperand(const android::nn::wrapper::OperandType& type);
  uint32_t AddNewOperand(const std::string& name,
                         const android::nn::wrapper::OperandType& operand_type);

  IOpBuilder* GetOpBuilder(const ONNX_NAMESPACE::NodeProto& node);
};

}  // namespace nnapi
}  // namespace onnxruntime