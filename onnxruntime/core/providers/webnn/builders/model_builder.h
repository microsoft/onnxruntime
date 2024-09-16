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
               const emscripten::val& context, const DataLayout preferred_layout,
               const WebnnDeviceType wnn_device_type, const emscripten::val& wnn_limits);
  ~ModelBuilder() = default;

  Status Compile(std::unique_ptr<Model>& model) ORT_MUST_USE_RESULT;

  // Accessors for members.
  const GraphViewer& GetGraphViewer() const { return graph_viewer_; }
  InitializedTensorSet GetInitializerTensors();

  const emscripten::val& GetBuilder() const { return wnn_builder_; }
  const emscripten::val& GetContext() const { return wnn_context_; }
  const emscripten::val& GetOperand(const std::string& name) const { return wnn_operands_.at(name); }
  const emscripten::val& GetOpSupportLimits() const { return wnn_limits_; }

  void AddOperand(const std::string& name, const emscripten::val& operand);
  const emscripten::val& GetZeroConstant(const std::string& data_type);
  // Use the buffers to persist WebNN allocated data like transposed weight.
  // It ensures the validity during inference session.
  std::vector<std::unique_ptr<uint8_t[]>> mem_persist_buffers_;
  // Add a constant operand (allocate persist buffer and move the ownership to mem_persist_buffers_).
  Status AddOperandFromPersistMemoryBuffer(
      const std::string& name, const void* buffer,
      const size_t size, const std::vector<uint32_t> shape, const int32_t data_type);

  DataLayout GetPreferredLayout() const { return preferred_layout_; }

  WebnnDeviceType GetWebnnDeviceType() const { return wnn_device_type_; }

  // The initializer will be processed separately, skip it as an initializer.
  void AddInitializerToSkip(const std::string& tensor_name);

  // There are some input which will not be used, add it to a list which will not
  // be added to WebNN model, since WebNN does not like input unused.
  void AddInputToSkip(const std::string& input_name);

  std::string GetUniqueName(const std::string& base_name);

 private:
  const GraphViewer& graph_viewer_;
  const logging::Logger& logger_;

  emscripten::val wnn_context_ = emscripten::val::undefined();
  emscripten::val wnn_builder_ = emscripten::val::undefined();
  DataLayout preferred_layout_;
  WebnnDeviceType wnn_device_type_;
  emscripten::val wnn_limits_ = emscripten::val::undefined();
  InlinedHashMap<std::string, emscripten::val> wnn_operands_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<std::vector<uint8_t>> unpacked_tensors_;

  InlinedHashMap<std::string, OnnxTensorInfo> input_output_info_;

  InlinedHashSet<std::string> skipped_initializers_;
  InlinedHashSet<std::string> skipped_inputs_;

  uint32_t name_token_{0};
  InlinedHashSet<std::string> unique_names_;

  // Convert the onnx model to WebNN operands
  Status Initialize() ORT_MUST_USE_RESULT;

  void PreprocessInitializers();

  // Copy and process all the initializers to WebNN constants.
  Status RegisterInitializers() ORT_MUST_USE_RESULT;

  Status AddOperations() ORT_MUST_USE_RESULT;
  Status RegisterModelInputs() ORT_MUST_USE_RESULT;
  Status RegisterModelOutputs() ORT_MUST_USE_RESULT;
  Status RegisterModelInputOutput(const NodeArg& node_arg, bool is_input) ORT_MUST_USE_RESULT;

  static const IOpBuilder* GetOpBuilder(const Node& node);
};

}  // namespace webnn
}  // namespace onnxruntime
