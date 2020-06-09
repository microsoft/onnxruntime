// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <list>
#include <unordered_map>
#include <memory>
#include <climits>
#include <string>
#include "core/common/path.h"
#include "core/graph/graph_viewer.h"
#include "core/session/onnxruntime_c_api.h"
#include "gsl/gsl"

namespace onnxruntime {
typedef std::unordered_map<std::string, std::string> ModelMetaData;
using IOnnxRuntimeOpSchemaRegistryList = std::list<std::shared_ptr<IOnnxRuntimeOpSchemaCollection>>;

// A machine learning model representation class.
// Besides a main <Graph>, it also holds basic information, say,
// model version, model domain, model author, license etc.
class Model {
 public:
  static constexpr Version kNoVersion = INT64_MAX;

  explicit Model(const std::string& graph_name,
                 bool is_onnx_domain_only,
                 const logging::Logger& logger)
      : Model(graph_name, is_onnx_domain_only, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {},
              {}, logger) {}

  // Construct model from scratch.
  explicit Model(const std::string& graph_name,
                 bool is_onnx_domain_only,
                 const ModelMetaData& model_metadata,
                 const PathString& model_path,
                 const IOnnxRuntimeOpSchemaRegistryList& local_registries,
                 const std::unordered_map<std::string, int>& domain_to_version,
                 const std::vector<ONNX_NAMESPACE::FunctionProto>& model_specific_functions,
                 const logging::Logger& logger);

  // NOTE: after calling this constructor, <*this> model will
  // hold a copy of <model_proto>.
  explicit Model(const ONNX_NAMESPACE::ModelProto& model_proto,
                 const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                 const logging::Logger& logger)
      : Model(model_proto, PathString(), local_registries, logger) {}

  // NOTE: after calling this constructor, <*this> model will
  // hold a copy of <model_proto>.
  explicit Model(const ONNX_NAMESPACE::ModelProto& model_proto,
                 const PathString& model_path,
                 const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                 const logging::Logger& logger);

  // NOTE: after calling this constructor, <*this> model will
  // own the <model_proto>.
  explicit Model(ONNX_NAMESPACE::ModelProto&& model_proto,
                 const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                 const logging::Logger& logger)
      : Model(std::move(model_proto), PathString(), local_registries, logger) {}

  // NOTE: after calling this constructor, <*this> model will
  // own the <model_proto>.
  explicit Model(ONNX_NAMESPACE::ModelProto&& model_proto,
                 const PathString& model_path,
                 const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                 const logging::Logger& logger);

  // Get model's IR version.
  // Return <kNoVersion> if not specified.
  Version IrVersion() const;

  // Get model's producer name.
  // Return null pointer if not specified.
  const std::string& ProducerName() const;
  // Set model's producer name.
  void SetProducerName(const std::string& producer_name);

  // Get model's producer version.
  // Return null pointer if not specified.
  const std::string& ProducerVersion() const;
  // Set model's producer version.
  void SetProducerVersion(const std::string& producer_version);

  // Get model's domain.
  // Return null pointer if not specified.
  const std::string& Domain() const;
  // Set models' domain.
  void SetDomain(const std::string& domain);

  // Get model's version.
  // Return null pointer if not specified.
  Version ModelVersion() const;
  // Set models' version.
  void SetModelVersion(onnxruntime::Version model_version);

  // Get model's doc string.
  // Return null pointer if not specified.
  const std::string& DocString() const;
  // Set models' doc string.
  void SetDocString(const std::string& doc_string);

  const ModelMetaData& MetaData() const noexcept;

  // Gets the path from which the model was loaded, if any.
  const Path& ModelPath() const noexcept { return model_path_; }

  // Get model's main graph.
  Graph& MainGraph() noexcept;
  const Graph& MainGraph() const noexcept;

  // Add function proto to Model
  void AddFunction(const ONNX_NAMESPACE::FunctionProto& func_proto);

  // Get model's serialization proto data.
  ONNX_NAMESPACE::ModelProto ToProto();

#ifdef _WIN32
  static common::Status Save(Model& model, const std::wstring& file_path);
#endif
  static common::Status Save(Model& model, const std::string& file_path);

  static common::Status Save(Model& model, int fd);

  static common::Status Load(std::istream& model_istream, ONNX_NAMESPACE::ModelProto* p_model_proto);

  static common::Status Load(const PathString& file_path,
                             /*out*/ ONNX_NAMESPACE::ModelProto& model_proto);

  // TODO(Task:132) Use of shared_ptr<X>* in Load/Save methods is confusing.
  static common::Status Load(const PathString& file_path,
                             /*out*/ std::shared_ptr<Model>& p_model,
                             const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                             const logging::Logger& logger);

  static common::Status Load(int fd, /*out*/ ONNX_NAMESPACE::ModelProto& model_proto);

  static common::Status Load(int fd, /*out*/ std::shared_ptr<Model>& p_model,
                             const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                             const logging::Logger& logger);

  static common::Status Load(int fd,
                             const PathString& model_path,
                             /*out*/ std::shared_ptr<Model>& p_model,
                             const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                             const logging::Logger& logger);

  // 'int' rather than 'size_t' because of a protobuf design choice; let callers handle type checks
  static common::Status LoadFromBytes(int count, void* pBytes,
                                      /*out*/ ONNX_NAMESPACE::ModelProto& model_proto);

  // 'int' rather than 'size_t' because of a protobuf design choice; let callers handle type checks
  static common::Status LoadFromBytes(int count, void* pBytes, /*out*/ std::shared_ptr<Model>& p_model,
                                      const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                                      const logging::Logger& logger);

  // 'int' rather than 'size_t' because of a protobuf design choice; let callers handle type checks
  static common::Status LoadFromBytes(int count, void* pBytes,
                                      const PathString& model_path,
                                      /*out*/ std::shared_ptr<Model>& p_model,
                                      const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                                      const logging::Logger& logger);

  static common::Status Load(const ONNX_NAMESPACE::ModelProto& model_proto, /*out*/ std::shared_ptr<Model>& p_model,
                             const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                             const logging::Logger& logger);

  static common::Status Load(const ONNX_NAMESPACE::ModelProto& model_proto,
                             const PathString& model_path,
                             /*out*/ std::shared_ptr<Model>& p_model,
                             const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                             const logging::Logger& logger);

  static common::Status Load(ONNX_NAMESPACE::ModelProto&& model_proto,
                             /*out*/ std::shared_ptr<Model>& p_model,
                             const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                             const logging::Logger& logger);

  static common::Status Load(ONNX_NAMESPACE::ModelProto&& model_proto,
                             const PathString& model_path,
                             /*out*/ std::shared_ptr<Model>& p_model,
                             const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                             const logging::Logger& logger);

 private:
  // Model data.
  ONNX_NAMESPACE::ModelProto model_proto_;

  // This is a duplication of <model_proto_.metadata_props()>.
  // It gives better accessibility.
  ModelMetaData model_metadata_;

  // Path to model file. May be empty.
  const Path model_path_;

  // Main graph of the model.
  std::unique_ptr<Graph> graph_;
};
}  // namespace onnxruntime
