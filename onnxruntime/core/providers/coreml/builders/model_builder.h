// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <core/graph/graph_viewer.h>
#include "coreml/Model.pb.h"

// namespace CoreML {
// namespace Specification {
// class Model;
// }
// }  // namespace CoreML

namespace COREML_SPEC = CoreML::Specification;

namespace onnxruntime {
namespace coreml {

class IOpBuilder;
class Model;
struct OnnxTensorInfo;

class ModelBuilder {
 public:
  ModelBuilder(const GraphViewer& graph_viewer);
  ~ModelBuilder() = default;

  Status Compile(std::unique_ptr<Model>& model, const std::string& path) ORT_MUST_USE_RESULT;
  Status SaveCoreMLModel(const std::string& path);

  // Accessors for members
  const GraphViewer& GetGraphViewer() const { return graph_viewer_; }
  const InitializedTensorSet& GetInitializerTensors() const { return graph_viewer_.GetAllInitializedTensors(); }

  void AddLayer(COREML_SPEC::NeuralNetworkLayer* layer);

 private:
  const GraphViewer& graph_viewer_;
  std::unique_ptr<CoreML::Specification::Model> coreml_model_;
  std::unordered_set<std::string> scalar_outputs_;
  std::unordered_map<std::string, OnnxTensorInfo> input_output_info_;

  // Convert the onnx model to CoreML::Specification::Model
  Status Initialize() ORT_MUST_USE_RESULT;

  // If a CoreML operation will use initializers directly, we will add the initializers to the skip list
  void PreprocessInitializers();

  // Copy and process all the initializers to CoreML model
  Status RegisterInitializers() ORT_MUST_USE_RESULT;

  Status AddOperations() ORT_MUST_USE_RESULT;
  Status RegisterModelInputs() ORT_MUST_USE_RESULT;
  Status RegisterModelOutputs() ORT_MUST_USE_RESULT;
  Status RegisterModelInputOutput(const NodeArg& node_arg, bool is_input) ORT_MUST_USE_RESULT;

  // Record the onnx scalar output names
  void AddScalarOutput(const std::string& output_name);

  static const IOpBuilder* GetOpBuilder(const Node& node);
};

}  // namespace coreml
}  // namespace onnxruntime
