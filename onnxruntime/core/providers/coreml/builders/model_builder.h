// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/span_utils.h"
#include "core/graph/graph_viewer.h"
#include "core/providers/coreml/builders/coreml_spec.h"
#include "core/providers/coreml/model/model.h"

#if defined(COREML_ENABLE_MLPROGRAM)
// coremltools classes
namespace MPL {
class ModelPackage;
}

namespace MILBlob {
namespace Blob {
class StorageWriter;
}
}  // namespace MILBlob
#endif

namespace onnxruntime {
namespace coreml {

class IOpBuilder;

class ModelBuilder {
 private:
  ModelBuilder(const GraphViewer& graph_viewer, const logging::Logger& logger,
               int32_t coreml_version, uint32_t coreml_flags,
               std::vector<std::string>&& onnx_input_names,
               std::vector<std::string>&& onnx_output_names);

 public:
  // Create the CoreML model, serialize to disk, load and compile using the CoreML API and return in `model`
  static Status Build(const GraphViewer& graph_viewer, const logging::Logger& logger,
                      int32_t coreml_version, uint32_t coreml_flags,
                      std::vector<std::string>&& onnx_input_names,
                      std::vector<std::string>&& onnx_output_names,
                      std::unique_ptr<Model>& model);

  ~ModelBuilder();

  const GraphViewer& GetGraphViewer() const { return graph_viewer_; }
  const InitializedTensorSet& GetInitializerTensors() const { return graph_viewer_.GetAllInitializedTensors(); }
  const ONNX_NAMESPACE::TensorProto* GetConstantInitializer(const std::string& name) const {
    return graph_viewer_.GetConstantInitializer(name, true);
  }

  // Since CoreML 2 the spec version is +1 as CoreML 1.1 was spec version 2.
  // We only support CoreML 3 and later so the spec version is always version + 1.
  int32_t CoreMLVersion() const { return coreml_version_; }
  int32_t CoreMLSpecVersion() const { return coreml_version_ + 1; }

  // Returns true if we are creating an ML Program
  bool CreateMLProgram() const {
#if defined(COREML_ENABLE_MLPROGRAM)
    return create_ml_program_;
#else
    return false;
#endif
  }

  /*
   * NeuralNetworkLayer helpers
   */

  // Create a NeuralNetwork layer using the node name and optional suffix for the name.
  // If Node has no name a unique name will be generated from the node index and operator.
  std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> CreateNNLayer(const Node& node, std::string_view suffix = "");

  // Add layer to the Core ML NeuralNetwork model
  void AddLayer(std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer);

#if defined(COREML_ENABLE_MLPROGRAM)
  /*
   * MLProgram helpers
   */

  // Create Operation, set type and the unique name attribute.
  std::unique_ptr<COREML_SPEC::MILSpec::Operation> CreateOperation(const Node& node, std::string_view op_type,
                                                                   std::string_view suffix = "");

  //
  // Helpers for adding attributes from ONNX nodes as inputs to an ML Program Operation
  //

  /// <summary>
  /// Add a value as a 'const' operation, generating a unique name for the value from op_type and value_type.
  /// Use for values that were not initializers in the original ONNX model. e.g. attributes from ONNX nodes.
  /// Add existing initializers using AddConstant with the TensorProto.
  ///
  /// e.g. adding the bias input of Gemm would have op_type='gemm' and value_type='bias'.
  /// </summary>
  /// <typeparam name="T">Value type.</typeparam>
  /// <param name="op_type">Typically MILSpec::Operation.type().</param>
  /// <param name="value_type">Typically the input name of the operation that will consume the value.</param>
  /// <param name="value">Value to add.</param>
  /// <param name="shape">Optional shape for the value.
  /// If T is a primitive type `shape` is ignored and the value is treated as a scalar.
  /// For a container type, if `shape` is not provided the shape is inferred to be 1-D of {value.size()}.
  /// </param>
  /// <returns>Unique name generated for value.</returns>
  template <typename T>
  std::string_view AddConstant(std::string_view op_type, std::string_view value_type, gsl::span<const T> value,
                               std::optional<gsl::span<const int64_t>> shape = std::nullopt) {
    static_assert(std::is_same_v<T, float> ||
                      std::is_same_v<T, int64_t> ||
                      std::is_same_v<T, std::string> ||
                      std::is_same_v<T, bool>,
                  // add specialization in AddConstantImpl for new types if needed
                  "AddConstant currently supports float, int64_t, std::string and bool.");
    return AddConstantImpl(op_type, value_type, value, shape);
  }

  template <typename T>
  std::string_view AddConstant(std::string_view op_type, std::string_view value_type, const std::vector<T>& value,
                               std::optional<gsl::span<const int64_t>> shape = std::nullopt) {
    return AddConstant(op_type, value_type, AsSpan(value), shape);
  }

  /// <summary>
  /// Add a scalar value as a 'const' operation. See AddConstant for details.
  /// </summary>
  template <typename T>
  std::string_view AddScalarConstant(std::string_view op_type, std::string_view value_type, const T& value) {
    return AddConstant(op_type, value_type, AsSpan({value}), AsSpan<const int64_t>({}));
  }

  // add the operation to the main function
  void AddOperation(std::unique_ptr<COREML_SPEC::MILSpec::Operation> operation);
#endif

  /*
   * General helpers
   */

  // The initializer is processed separately (e.g. layout is transformed) by the operator builder,
  // so we don't do a copy of the original initializer into the model.
  void AddInitializerToSkip(const std::string& tensor_name);

  // There are some input which will not be used, add it to a list which will not
  // be added to CoreML model, since CoreML does not like input unused
  void AddInputToSkip(const std::string& input_name);

  const std::string& GetUniqueName(const std::string& base_name);
  const std::string& GetUniqueName(const Node& node, std::string_view suffix);

  const logging::Logger& Logger() const { return logger_; }

 private:
#if defined(COREML_ENABLE_MLPROGRAM)
  template <typename T>
  std::string_view AddConstantImpl(std::string_view op_type, std::string_view value_type, gsl::span<const T> value,
                                   std::optional<gsl::span<const int64_t>> shape = std::nullopt);

  // apply the CoreML naming rules and fix any invalid names.
  const std::string& GetSafeName(const std::string& name);
  // sanitize all the names in the ML Model
  void SanitizeNames();

  // add Value as a const operation. return value name in case sanitization changed it
  const std::string& AddConstantOperation(std::string_view name, COREML_SPEC::MILSpec::Value&& initializer);
  const std::string& AddTensorValueAsConstantOperation(std::string_view op_type, std::string_view value_type,
                                                       COREML_SPEC::MILSpec::Value&& input_value);
#endif

  // Convert the ONNX model in graph_viewer_ to a CoreML::Specification::Model and serialize to disk.
  // We then load it using CoreML in order compile it.
  Status CreateModel();
  Status SaveModel();
  Status LoadModel(std::unique_ptr<Model>& model);

  // If a CoreML operation will use initializers directly, we will add the initializers to the skip list
  void PreprocessInitializers();

  // Copy and process all the initializers to CoreML model
  Status RegisterInitializers();

  Status ProcessNodes();
  Status RegisterModelInputs();
  Status RegisterModelOutputs();
  Status RegisterModelInputOutput(const NodeArg& node_arg, bool is_input);

  // Record the onnx scalar output names
  void AddScalarOutput(const std::string& output_name);

  // Record the onnx int64 type output names
  void AddInt64Output(const std::string& output_name);

  const GraphViewer& graph_viewer_;
  const logging::Logger& logger_;
  const int32_t coreml_version_;
  const uint32_t coreml_flags_;
  const bool create_ml_program_;         // ML Program (CoreML5, iOS 15+, macOS 12+) or NeuralNetwork (old)
  const std::string model_output_path_;  // create_ml_program_ ? dir for mlpackage : filename for mlmodel

  std::vector<std::string> onnx_input_names_;
  std::vector<std::string> onnx_output_names_;

  std::unique_ptr<CoreML::Specification::Model> coreml_model_;
  std::unordered_set<std::string> scalar_outputs_;
  std::unordered_set<std::string> int64_outputs_;
  std::unordered_map<std::string, OnnxTensorInfo> input_output_info_;

  std::unordered_map<std::string, int> initializer_usage_;
  std::unordered_set<std::string> skipped_inputs_;

  uint32_t name_token_{0};
  std::unordered_set<std::string> unique_names_;

#if defined(COREML_ENABLE_MLPROGRAM)
  // mlprogram_main_ is the main block of the CoreML ML Program.
  // It is set in CreateModel to the CoreML Model.mlprogram.functions['main'].block_specializations['CoreML<ver>']
  // entry we create.
  COREML_SPEC::MILSpec::Function* mlprogram_main_fn_{nullptr};  // Function that contains a Block with the operations
  COREML_SPEC::MILSpec::Block* mlprogram_main_block_{nullptr};  // Block that all the operations are added to
  std::unique_ptr<MPL::ModelPackage> mlpackage_;
  std::unique_ptr<MILBlob::Blob::StorageWriter> weights_file_writer_;

  // Values must start with [a-zA-A_]
  // Additionally they can't be in a list of reserved words.
  // If we need to sanitize an initializer name we do so during PreprocessInitializers and apply the change during
  // RegisterInitializers.
  // We also check inputs in AddOperation and apply the change there.
  // This means an op builder author doesn't need to be aware of the renaming.
  // https://github.com/apple/coremltools/blob/8b37641f243b1a3e81452feea311c6e30dcc9287/coremltools/converters/mil/mil/passes/defs/preprocess.py#L146-L149
  std::unordered_map<std::string, std::string> values_to_rename_;
#endif
};

}  // namespace coreml
}  // namespace onnxruntime
