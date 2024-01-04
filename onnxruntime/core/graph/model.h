// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <list>
#include <unordered_map>
#include <memory>
#include <climits>
#include <string>

#include "flatbuffers/flatbuffers.h"

#include "core/common/path.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/ort_format_load_options.h"
#include "core/session/onnxruntime_c_api.h"
#if !defined(ORT_MINIMAL_BUILD)
#include "core/graph/function_template.h"
#endif

namespace onnxruntime {

namespace fbs {
struct Model;
}  // namespace fbs

typedef std::unordered_map<std::string, std::string> ModelMetaData;
using IOnnxRuntimeOpSchemaRegistryList = std::list<std::shared_ptr<IOnnxRuntimeOpSchemaCollection>>;

// Options to configure Model.
struct ModelOptions {
  bool allow_released_opsets_only;
  // If true, all inconsistencies encountered during shape and type inference
  // will be exposed to the caller as failures. If false, in some cases
  // warnings will be logged but processing will continue and no error will
  // be returned.
  bool strict_shape_type_inference;

  ModelOptions(bool allow_released_opsets_only, bool strict_shape_type_inference)
      : allow_released_opsets_only(allow_released_opsets_only),
        strict_shape_type_inference(strict_shape_type_inference) {}

  ModelOptions() : ModelOptions(true, false) {}
};

// A machine learning model representation class.
// Besides a main <Graph>, it also holds basic information, say,
// model version, model domain, model author, license etc.
class Model {
 public:
  static constexpr Version kNoVersion = INT64_MAX;

#if !defined(ORT_MINIMAL_BUILD)
  explicit Model(const std::string& graph_name,
                 bool is_onnx_domain_only,
                 const logging::Logger& logger,
                 const ModelOptions& options = {})
      : Model(graph_name, is_onnx_domain_only, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {},
              {}, logger, options) {}

  // Construct model from scratch.
  explicit Model(const std::string& graph_name,
                 bool is_onnx_domain_only,
                 const ModelMetaData& model_metadata,
                 const PathString& model_path,
                 const IOnnxRuntimeOpSchemaRegistryList& local_registries,
                 const std::unordered_map<std::string, int>& domain_to_version,
                 const std::vector<ONNX_NAMESPACE::FunctionProto>& model_local_functions,
                 const logging::Logger& logger,
                 const ModelOptions& options = {});

  // NOTE: after calling this constructor, <*this> model will
  // hold a copy of <model_proto>.
  explicit Model(const ONNX_NAMESPACE::ModelProto& model_proto,
                 const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                 const logging::Logger& logger, const ModelOptions& options = {})
      : Model(model_proto, PathString(), local_registries, logger, options) {}

  // NOTE: after calling this constructor, <*this> model will
  // hold a copy of <model_proto>.
  explicit Model(const ONNX_NAMESPACE::ModelProto& model_proto,
                 const PathString& model_path,
                 const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                 const logging::Logger& logger, const ModelOptions& options = {});

  // NOTE: after calling this constructor, <*this> model will
  // own the <model_proto>.
  explicit Model(ONNX_NAMESPACE::ModelProto&& model_proto,
                 const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                 const logging::Logger& logger, const ModelOptions& options = {})
      : Model(std::move(model_proto), PathString(), local_registries, logger, options) {}

  // NOTE: after calling this constructor, <*this> model will
  // own the <model_proto>.
  explicit Model(ONNX_NAMESPACE::ModelProto&& model_proto,
                 const PathString& model_path,
                 const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                 const logging::Logger& logger,
                 const ModelOptions& options = {});

#endif  // !defined(ORT_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD)
  // Get model's IR version.
  // Return <kNoVersion> if not specified.
  Version IrVersion() const;

  // Get model's producer name.
  // Returns empty string if not specified.
  const std::string ProducerName() const;
  // Set model's producer name.
  void SetProducerName(const std::string& producer_name);

  // Get model's producer version.
  // Returns empty string if not specified.
  const std::string ProducerVersion() const;
  // Set model's producer version.
  void SetProducerVersion(const std::string& producer_version);

  // Get model's domain.
  // Returns empty string if not specified.
  const std::string Domain() const;
  // Set models' domain.
  void SetDomain(const std::string& domain);

  // Get model's version.
  // Return null pointer if not specified.
  Version ModelVersion() const;
  // Set models' version.
  void SetModelVersion(onnxruntime::Version model_version);

  // Get model's doc string.
  // Returns empty string if not specified.
  const std::string DocString() const;
  // Set models' doc string.
  void SetDocString(const std::string& doc_string);

  // Get graph's doc string.
  // Returns empty string if not specified.
  const std::string GraphDocString() const;

  const NodeHashMap<std::string, std::unique_ptr<FunctionTemplate>>& GetModelLocalFunctionTemplates() const;

#else
  // Get model's IR version.
  // Return <kNoVersion> if not specified.
  Version IrVersion() const { return ir_version_; }

  // Get model's producer name.
  // Returns empty string if not specified.
  const std::string ProducerName() const { return producer_name_; }

  // Get model's producer version.
  // Returns empty string if not specified.
  const std::string ProducerVersion() const { return producer_version_; }

  // Get model's domain.
  // Returns empty string if not specified.
  const std::string Domain() const { return domain_; }

  // Get model's version.
  // Return null pointer if not specified.
  Version ModelVersion() const { return model_version_; }

  // Get model's doc string.
  // Returns empty string if not specified.
  const std::string DocString() const { return doc_string_; }

  // Get graph's doc string.
  // Returns empty string if not specified.
  const std::string GraphDocString() const { return graph_doc_string_; }
#endif

  const ModelMetaData& MetaData() const noexcept;

  // Gets the path from which the model was loaded, if any.
  const Path& ModelPath() const noexcept { return model_path_; }

  // Get model's main graph.
  Graph& MainGraph() noexcept;
  const Graph& MainGraph() const noexcept;

#if !defined(ORT_MINIMAL_BUILD)
  // Get model's serialization proto data.
  ONNX_NAMESPACE::ModelProto ToProto() const;

  // Get model's serialization proto data.
  // Save initializer larger than the given threshold (in bytes) into an external binary file
  // with the given name. This function is useful to avoid hitting the size limit of protobuf files.
  ONNX_NAMESPACE::ModelProto ToGraphProtoWithExternalInitializers(const std::string& external_file_name,
                                                                  const PathString& file_path,
                                                                  size_t initializer_size_threshold) const;

#ifdef _WIN32
  static common::Status Save(Model& model, const std::wstring& file_path);
#endif
  static common::Status Save(Model& model, const std::string& file_path);

  static common::Status Save(Model& model, int fd);

  // Save the model to file using an external file for initializers larger than the given threshold (in bytes).
  // Notice that when on Windows the external_file_name is a plain string.
  // This is because the string is saved inside the output protobuf as a plain string, where wchar is not supported.
#ifdef _WIN32
  static common::Status SaveWithExternalInitializers(Model& model,
                                                     const std::wstring& file_path,
                                                     const std::string& external_file_name,
                                                     size_t initializer_size_threshold);
#else
  static common::Status SaveWithExternalInitializers(Model& model,
                                                     const std::string& file_path,
                                                     const std::string& external_file_name,
                                                     size_t initializer_size_threshold);
#endif

  static common::Status SaveWithExternalInitializers(Model& model,
                                                     int fd,
                                                     const PathString& file_path,
                                                     const std::string& external_file_name,
                                                     size_t initializer_size_threshold);

  static common::Status Load(std::istream& model_istream, ONNX_NAMESPACE::ModelProto* p_model_proto);

  static common::Status Load(const PathString& file_path,
                             /*out*/ ONNX_NAMESPACE::ModelProto& model_proto);

  // TODO(Task:132) Use of shared_ptr<X>* in Load/Save methods is confusing.
  static common::Status Load(const PathString& file_path,
                             /*out*/ std::shared_ptr<Model>& p_model,
                             const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                             const logging::Logger& logger,
                             const ModelOptions& options = {});

  static common::Status Load(int fd, /*out*/ ONNX_NAMESPACE::ModelProto& model_proto);

  static common::Status Load(int fd, /*out*/ std::shared_ptr<Model>& p_model,
                             const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                             const logging::Logger& logger,
                             const ModelOptions& options = {});

  static common::Status Load(int fd,
                             const PathString& model_path,
                             /*out*/ std::shared_ptr<Model>& p_model,
                             const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                             const logging::Logger& logger,
                             const ModelOptions& options = {});

  // 'int' rather than 'size_t' because of a protobuf design choice; let callers handle type checks
  static common::Status LoadFromBytes(int count, void* pBytes,
                                      /*out*/ ONNX_NAMESPACE::ModelProto& model_proto);

  // 'int' rather than 'size_t' because of a protobuf design choice; let callers handle type checks
  static common::Status LoadFromBytes(int count, void* pBytes, /*out*/ std::shared_ptr<Model>& p_model,
                                      const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                                      const logging::Logger& logger,
                                      const ModelOptions& options = {});

  // 'int' rather than 'size_t' because of a protobuf design choice; let callers handle type checks
  static common::Status LoadFromBytes(int count, void* pBytes,
                                      const PathString& model_path,
                                      /*out*/ std::shared_ptr<Model>& p_model,
                                      const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                                      const logging::Logger& logger,
                                      const ModelOptions& options = {});

  static common::Status Load(const ONNX_NAMESPACE::ModelProto& model_proto, /*out*/ std::shared_ptr<Model>& p_model,
                             const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                             const logging::Logger& logger,
                             const ModelOptions& options = {});

  static common::Status Load(const ONNX_NAMESPACE::ModelProto& model_proto,
                             const PathString& model_path,
                             /*out*/ std::shared_ptr<Model>& p_model,
                             const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                             const logging::Logger& logger,
                             const ModelOptions& options = {});

  static common::Status Load(ONNX_NAMESPACE::ModelProto&& model_proto,
                             /*out*/ std::shared_ptr<Model>& p_model,
                             const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                             const logging::Logger& logger,
                             const ModelOptions& options = {});

  static common::Status Load(ONNX_NAMESPACE::ModelProto&& model_proto,
                             const PathString& model_path,
                             /*out*/ std::shared_ptr<Model>& p_model,
                             const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                             const logging::Logger& logger,
                             const ModelOptions& options = {});

  common::Status SaveToOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                                 flatbuffers::Offset<onnxruntime::fbs::Model>& model) const;

  /// <summary>
  /// Frees local function definitions in the model, excluding those in the `retained` set.
  /// Called from GraphPartitioner::InlineFunctionsAOT.
  /// </summary>
  /// <param name="retained">contains function IDs that should not be removed.</param>
  void RemoveLocalFunctionsProtos(const InlinedHashSet<std::string>& retained);

#endif  // !defined(ORT_MINIMAL_BUILD)

  static common::Status LoadFromOrtFormat(const onnxruntime::fbs::Model& fbs_model,
#if !defined(ORT_MINIMAL_BUILD)
                                          const IOnnxRuntimeOpSchemaRegistryList* local_registries,
#endif
                                          const OrtFormatLoadOptions& load_options,
                                          const logging::Logger& logger,
                                          std::unique_ptr<Model>& model);

  Model();

 private:
  // Model data.
#if !defined(ORT_MINIMAL_BUILD)
  ONNX_NAMESPACE::ModelProto model_proto_;
  // map from function id to pointer of model local function proto
  // FunctionProto is hosted in ModelProto.
  // this map will be used for the local functions' schema's type/shape inference.
  // This container is used by ONNX code and must be an std::unordered_map.
  std::unordered_map<std::string, const ONNX_NAMESPACE::FunctionProto*> model_local_functions_;
  // this is the map from function id to the local function template.
  // this map will be used by graph to instantiate the function body.
  // Defined as a node based map so the memory is released when not all of the functions
  // are inlined and removed.
  NodeHashMap<std::string, std::unique_ptr<FunctionTemplate>> model_local_function_templates_maps_;

#else
  // properties that would normally come from ModelProto
  std::string producer_version_;
  std::string producer_name_;
  int64_t model_version_ = kNoVersion;
  int64_t ir_version_ = kNoVersion;
  std::string domain_;
  std::string doc_string_;
  std::string graph_doc_string_;
#endif

  // This is a duplication of <model_proto_.metadata_props()>.
  // It gives better accessibility.
  ModelMetaData model_metadata_;

  // Path to model file. May be empty.
  const Path model_path_;

  // Main graph of the model.
  std::unique_ptr<Graph> graph_;
};
}  // namespace onnxruntime
