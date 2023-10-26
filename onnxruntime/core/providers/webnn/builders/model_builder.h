// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/inlined_containers.h"
#include <core/graph/graph_viewer.h>

#include "model.h"
#include "core/framework/execution_provider.h"
#include "core/providers/webnn/builders/helper.h"

#include <emscripten.h>
#include <emscripten/val.h>

namespace onnxruntime {
namespace webnn {

class IOpBuilder;

class ModelBuilder {
 public:
  ModelBuilder(const GraphViewer& graph_viewer, const logging::Logger& logger,
               const emscripten::val& context, const emscripten::val& builder,
               const DataLayout preferred_layout, const WebnnDeviceType wnn_device_type);
  ~ModelBuilder() = default;

  Status Compile(std::unique_ptr<Model>& model) ORT_MUST_USE_RESULT;

  // Accessors for members.
  const GraphViewer& GetGraphViewer() const { return graph_viewer_; }
  const InitializedTensorSet& GetInitializerTensors() const { return graph_viewer_.GetAllInitializedTensors(); }

  const emscripten::val& GetBuilder() const { return wnn_builder_; }
  const emscripten::val& GetContext() const { return wnn_context_; }
  const emscripten::val& GetOperand(const std::string& name) const { return wnn_operands_.at(name); }
  void AddOperand(const std::string& name, const emscripten::val& operand);
  const emscripten::val& GetZeroConstant(const std::string& data_type);
  // Use the buffers to persist WebNN allocated data like transposed weight.
  // It ensures the validity during inference session.
  std::vector<std::unique_ptr<uint8_t[]>> mem_persist_buffers_;
  // Add a constant operand (allocate persist buffer and move the ownership to mem_persist_buffers_).
  Status AddOperandFromPersistMemoryBuffer(
      const std::string& name, const void* buffer,
      const size_t size, const std::vector<uint32_t> shape, const int32_t data_type);
  // Find if an output has a fuseable activation (e.g., Relu).
  emscripten::val FindActivation(const Node& node, const NodeArg& output,
                                 const InlinedHashSet<std::string> supported_nodes = {});

  const InlinedHashSet<std::string>&
  GetFusedActivations() const { return fused_activations_; }

  DataLayout GetPreferredLayout() const { return preferred_layout_; }

  WebnnDeviceType GetWebnnDeviceType() const { return wnn_device_type_; }

  // The initializer will be processed separately, skip it as an initializer.
  void AddInitializerToSkip(const std::string& tensor_name);

  // There are some input which will not be used, add it to a list which will not
  // be added to CoreML model, since CoreML does not like input unused.
  void AddInputToSkip(const std::string& input_name);

  std::string GetUniqueName(const std::string& base_name);

 private:
  const GraphViewer& graph_viewer_;
  const logging::Logger& logger_;

  emscripten::val wnn_context_ = emscripten::val::object();
  emscripten::val wnn_builder_ = emscripten::val::object();
  DataLayout preferred_layout_;
  WebnnDeviceType wnn_device_type_;
  std::vector<std::vector<uint8_t>> unpacked_tensors_;
  InlinedHashMap<std::string, emscripten::val> wnn_operands_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;

  InlinedHashSet<std::string> scalar_outputs_;
  InlinedHashMap<std::string, OnnxTensorInfo> input_output_info_;

  InlinedHashSet<std::string> skipped_initializers_;
  InlinedHashSet<std::string> skipped_inputs_;

  InlinedHashSet<std::string> fused_activations_;

  uint32_t name_token_{0};
  InlinedHashSet<std::string> unique_names_;

  // All activation nodes (e.g., Relu) as a map <NodeIndex, FusionOperator>.
  InlinedHashMap<NodeIndex, emscripten::val> activation_nodes_;

  // Convert the onnx model to WebNN operands
  Status Initialize() ORT_MUST_USE_RESULT;

  void PreprocessInitializers();
  // Preprocess all the activation nodes (e.g., Relu) for easy query later.
  void PreprocessActivations();

  // Copy and process all the initializers to WebNN constants.
  Status RegisterInitializers() ORT_MUST_USE_RESULT;

  Status AddOperations() ORT_MUST_USE_RESULT;
  Status RegisterModelInputs() ORT_MUST_USE_RESULT;
  Status RegisterModelOutputs() ORT_MUST_USE_RESULT;
  Status RegisterModelInputOutput(const NodeArg& node_arg, bool is_input) ORT_MUST_USE_RESULT;

  // Record the onnx scalar output names.
  void AddScalarOutput(const std::string& output_name);

  static const IOpBuilder* GetOpBuilder(const Node& node);
};

}  // namespace webnn
}  // namespace onnxruntime
