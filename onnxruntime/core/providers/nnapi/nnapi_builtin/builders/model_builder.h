// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <onnx/onnx_pb.h>
#include <unordered_set>

#include "core/graph/basic_types.h"
#include "core/providers/nnapi/nnapi_builtin/model.h"
#include "core/providers/nnapi/nnapi_builtin/nnapi_lib/NeuralNetworksWrapper.h"
#include "shaper.h"

namespace onnxruntime {

class GraphViewer;
enum class DataLayout;
class NodeUnit;
class Node;
class NodeArg;

namespace nnapi {

class IOpBuilder;
class IOpSupportChecker;

class ModelBuilder {
 public:
  using Shape = Shaper::Shape;

  enum class TargetDeviceOption : int8_t {
    ALL_DEVICES,  // use all avaliable target devices

    /* TODO support these options
    PREFERRED_DEVICES,  // Use one or more preferred devices (must be given)
    EXCLUDED_DEVICES,   // Exclude one or more devices (must be given)
     */

    CPU_DISABLED,  // use all avaliable target devices except CPU
    CPU_ONLY,      // use CPU only
  };

  ModelBuilder(const GraphViewer& graph_viewer);

  common::Status Compile(std::unique_ptr<Model>& model);

  int32_t GetNNAPIFeatureLevel() const;

  // Add an NNAPI operation (operator)
  common::Status AddOperation(int op, const std::vector<uint32_t>& input_indices,
                              const std::vector<std::string>& output_names,
                              const std::vector<android::nn::wrapper::OperandType>& types);

  // Find if the given node_unit has a fuseable activation (Relu/Relu1/Relu6)
  // For now we only support node_unit with a single output
  int32_t FindActivation(const NodeUnit& node_unit);

  // Add an NNAPI scalar operand
  common::Status AddOperandFromScalar(bool value, uint32_t& index);
  common::Status AddOperandFromScalar(float value, uint32_t& index);
  common::Status AddOperandFromScalar(int32_t value, uint32_t& index);

  // Add an NNAPI tensor operand (and allocate persist buffer)
  common::Status AddOperandFromPersistMemoryBuffer(
      const std::string& name, const void* buffer,
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

  // Returns the preferred layout for this EP.
  DataLayout GetPreferredLayout() const;

  // Relax fp32 computation to fp16
  // It is off by default
  void SetUseFp16(bool use_fp16) { use_fp16_ = use_fp16; }

  void SetTargetDeviceOption(TargetDeviceOption option) { target_device_option_ = option; }

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

  const InitializedTensorSet& GetInitializerTensors() const;

  const GraphViewer& GetGraphViewer() const { return graph_viewer_; }

  // Get the NodeUnit which contains the given node
  const NodeUnit& GetNodeUnit(const Node* node) const;

 private:
  const NnApi* nnapi_{nullptr};
  const GraphViewer& graph_viewer_;
  std::unique_ptr<Model> nnapi_model_;

  uint32_t name_token_{0};

  bool use_nchw_{false};
  bool use_fp16_{false};
  android::nn::wrapper::ExecutePreference exe_pref_{
      android::nn::wrapper::ExecutePreference::PREFER_FAST_SINGLE_ANSWER};

  Shaper shaper_;

  std::unordered_map<std::string, uint32_t> operand_indices_;
  std::unordered_map<std::string, android::nn::wrapper::OperandType> operand_types_;

  std::unordered_set<std::string> operands_;
  std::unordered_set<std::string> fused_activations_;

  std::unordered_set<std::string> skipped_initializers_;

  // All activation nodes (Relu, Relu1, Relu6) as a map <const NodeUnit*, activation_code>
  std::unordered_map<const NodeUnit*, int32_t> activation_node_units_;

  std::unordered_map<std::string, std::shared_ptr<IOpSupportChecker>> op_support_checkers_;


  std::vector<uint32_t> input_index_vec_;
  std::vector<uint32_t> output_index_vec_;

  // Contains all quantized operators' input and the NodeUnit(s) using the input
  // In the form of {input_name, [NodeUnit(s) using the input]}
  std::unordered_map<std::string, std::vector<const NodeUnit*>> all_quantized_op_inputs_;

  // Holder for the NodeUnits in the graph, this will guarantee the NodeUnits is
  // valid throughout the lifetime of the ModelBuilder
  std::vector<std::unique_ptr<NodeUnit>> node_unit_holder_;
  std::unordered_map<const Node*, const NodeUnit*> node_unit_map_;

  std::unordered_set<std::string> unique_names_;

  TargetDeviceOption target_device_option_{TargetDeviceOption::ALL_DEVICES};
  std::vector<ANeuralNetworksDevice*> nnapi_target_devices_;
  std::string nnapi_target_devices_detail_;  // Debug info for target devices

  // The number of nnapi operations in this model
  size_t num_nnapi_ops_ = 0;
  uint32_t next_index_ = 0;

  // Convert the onnx model to ANeuralNetworksModel
  common::Status Prepare();

  common::Status GetTargetDevices();

  // If a NNAPI operation will use initializers directly, we will add the initializers to the skip list
  void PreprocessInitializers();
  // Preprocess all the activation nodes (Relu/Relu1/Relu6) for easy query later
  void PreprocessActivations();
  // Copy and process all the initializers to NNAPI model
  common::Status RegisterInitializers();
  common::Status RegisterModelInputs();
  common::Status AddOperations();
  common::Status RegisterModelOutputs();
  // After constructing the NNAPI model, will set the shape inferencing record to the Model
  void RegisterModelShaper();

  // Get all quantized inputs in the underlying graph_viewer
  void GetAllQuantizedOpInputs();

  // Go through the underlying graph_viewer, and generate NodeUnits, Many initializing functions are
  // using the result of PreprocessNodeUnits, this need to run early in the Prepare()
  void PreprocessNodeUnits();

  common::Status SetOperandValue(uint32_t index, Model::NNMemory* memory, size_t size, size_t offset);

  common::Status AddNewNNAPIOperand(const android::nn::wrapper::OperandType& type, uint32_t& index);
  common::Status AddNewOperand(const std::string& name,
                               const android::nn::wrapper::OperandType& operand_type,
                               uint32_t& index);

  static const IOpBuilder* GetOpBuilder(const NodeUnit& node_unit);
};

}  // namespace nnapi
}  // namespace onnxruntime