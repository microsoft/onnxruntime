// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include "core/common/logging/logging.h"
#include "core/flatbuffers/schema/ort.fbs.h"
#include "core/flatbuffers/flatbuffers_utils.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/model.h"
#include "core/graph/model_load_utils.h"

#ifdef _MSC_VER
#pragma warning(push)
// 'type' : forcing value to bool 'true' or 'false' (performance warning)
#pragma warning(disable : 4800)
#endif
#include <google/protobuf/io/coded_stream.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#include "core/util/protobuf_parsing_utils.h"

#include "gsl/gsl"

#include "core/platform/env.h"

#if !defined(ORT_MINIMAL_BUILD)
#include "core/graph/schema_registry.h"
#endif

using namespace ONNX_NAMESPACE;
using namespace onnxruntime;
using namespace onnxruntime::common;
using namespace onnxruntime::experimental;

namespace onnxruntime {

#if !defined(ORT_MINIMAL_BUILD)

static constexpr int DEFAULT_PROTOBUF_BLOCK_SIZE = 4 * 1024 * 1024;

Model::Model(const std::string& graph_name,
             bool is_onnx_domain_only,
             const ModelMetaData& model_metadata,
             const PathString& model_path,
             const IOnnxRuntimeOpSchemaRegistryList& local_registries,
             const std::unordered_map<std::string, int>& domain_to_version,
             const std::vector<ONNX_NAMESPACE::FunctionProto>&,
             const logging::Logger& logger)
    : model_path_(Path::Parse(model_path)) {
  model_proto_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  model_proto_.mutable_graph()->set_name(graph_name);
  model_metadata_ = model_metadata;
  for (auto& metadata : model_metadata_) {
    const gsl::not_null<StringStringEntryProto*> prop{model_proto_.add_metadata_props()};
    prop->set_key(metadata.first);
    prop->set_value(metadata.second);
  }

  auto schema_registry = std::make_shared<SchemaRegistryManager>();
  for (const auto& schema_collection : local_registries) {
    schema_registry->RegisterRegistry(schema_collection);
  }

  auto allow_released_opsets_only =
      model_load_utils::IsAllowReleasedONNXOpsetsOnlySet();
  auto* p_domain_to_version = &domain_to_version;
  DomainToVersionMap domain_to_version_static;
  domain_to_version_static = allow_released_opsets_only
                                 ? schema_registry->GetLastReleasedOpsetVersions(is_onnx_domain_only)
                                 : schema_registry->GetLatestOpsetVersions(is_onnx_domain_only);
  if (p_domain_to_version->empty()) {
    p_domain_to_version = &domain_to_version_static;
  }

  for (const auto& domain : *p_domain_to_version) {
    model_load_utils::ValidateOpsetForDomain(
        domain_to_version_static, logger, allow_released_opsets_only,
        domain.first, domain.second);
    const gsl::not_null<OperatorSetIdProto*> opset_id_proto{model_proto_.add_opset_import()};
    opset_id_proto->set_domain(domain.first);
    opset_id_proto->set_version(domain.second);
  }

  // need to call private ctor so can't use make_shared
  GSL_SUPPRESS(r .11)
  graph_.reset(new Graph(*this, model_proto_.mutable_graph(), *p_domain_to_version, IrVersion(), schema_registry,
                         logger));
}

Model::Model(const ModelProto& model_proto, const PathString& model_path,
             const IOnnxRuntimeOpSchemaRegistryList* local_registries, const logging::Logger& logger)
    : Model(ModelProto(model_proto), model_path, local_registries, logger) {
}

Model::Model(ModelProto&& model_proto, const PathString& model_path,
             const IOnnxRuntimeOpSchemaRegistryList* local_registries,
             const logging::Logger& logger)
    : model_path_(Path::Parse(model_path)) {
  if (!utils::HasGraph(model_proto)) {
    ORT_THROW("ModelProto does not have a graph.");
  }

  if (model_proto.opset_import_size() == 0) {
    ORT_THROW(
        "Missing opset in the model. All ModelProtos MUST have at least one entry that"
        " specifies which version of the ONNX OperatorSet is being imported.");
  }

  if (!model_proto.has_ir_version() || model_proto.ir_version() > ONNX_NAMESPACE::Version::IR_VERSION) {
    ORT_THROW("Unknown model file format version.");
  }

  model_proto_ = std::move(model_proto);
  for (auto& prop : model_proto_.metadata_props()) {
    model_metadata_[prop.key()] = prop.value();
  }

  auto schema_registry = std::make_shared<SchemaRegistryManager>();
  if (local_registries != nullptr) {
    for (const auto& schema_collection : *local_registries) {
      schema_registry->RegisterRegistry(schema_collection);
    }
  }

  bool allow_official_onnx_release_only =
      model_load_utils::IsAllowReleasedONNXOpsetsOnlySet();
  const auto onnx_released_versions =
      schema_registry->GetLastReleasedOpsetVersions(false);

  std::unordered_map<std::string, int> domain_to_version;
  for (auto& opSet : model_proto_.opset_import()) {
    const auto& domain = opSet.domain();
    const auto version = gsl::narrow_cast<int>(opSet.version());
    // empty domain and 'ai.onnx' are equivalent
    if ((domain.empty() || domain == kOnnxDomainAlias) && version < 7) {
      // TODO: Check if we can upgrade all the current opset 6 models that are being tested
      // in CI to opset 7 or above
      LOGS(logger, WARNING) << "ONNX Runtime only *guarantees* support for models stamped "
                               "with opset version 7 or above for opset domain 'ai.onnx'. "
                               "Please upgrade your model to opset 7 or higher. "
                               "For now, this opset "
                            << version
                            << " model may run depending upon legacy support "
                               "of some older opset version operators.";
    }

    model_load_utils::ValidateOpsetForDomain(onnx_released_versions, logger,
                                             allow_official_onnx_release_only, domain, version);

    // We need to overwrite the domain here with ("") or else the loop below will try to find ("")
    // in the map and if not found (when domain == kOnnxDomainAlias), adds an entry for ("", 11).
    // This effectively ignores the opset version specified by the model for the onnx domain.
    if (domain == kOnnxDomainAlias) {
      domain_to_version[kOnnxDomain] = version;
    } else {
      domain_to_version[domain] = version;
    }
  }

  auto domain_map = allow_official_onnx_release_only
                        ? schema_registry->GetLastReleasedOpsetVersions(false)
                        : schema_registry->GetLatestOpsetVersions(false);
  for (const auto& domain : domain_map) {
    if (domain_to_version.find(domain.first) == domain_to_version.end()) {
      domain_to_version[domain.first] = domain.second;
      const gsl::not_null<OperatorSetIdProto*> opset_id_proto{model_proto_.add_opset_import()};
      opset_id_proto->set_domain(domain.first);
      opset_id_proto->set_version(domain.second);
    }
  }

  // create instance. need to call private ctor so can't use make_unique
  GSL_SUPPRESS(r .11)
  graph_.reset(new Graph(*this, model_proto_.mutable_graph(), domain_to_version, IrVersion(), schema_registry, logger));
}

Version Model::IrVersion() const {
  if (utils::HasIrVersion(model_proto_)) {
    return model_proto_.ir_version();
  }
  return kNoVersion;
}

const std::string Model::ProducerName() const {
  if (model_proto_.has_producer_name()) {
    return model_proto_.producer_name();
  }
  return std::string();
}

void Model::SetProducerName(const std::string& producer_name) {
  model_proto_.set_producer_name(producer_name);
}

const std::string Model::ProducerVersion() const {
  if (model_proto_.has_producer_version()) {
    return model_proto_.producer_version();
  }
  return std::string();
}

void Model::SetProducerVersion(const std::string& producer_version) {
  model_proto_.set_producer_version(producer_version);
}

const std::string Model::Domain() const {
  if (model_proto_.has_domain()) {
    return model_proto_.domain();
  }
  return std::string();
}

void Model::SetDomain(const std::string& domain) {
  model_proto_.set_domain(domain);
}

Version Model::ModelVersion() const {
  if (utils::HasModelVersion(model_proto_)) {
    return model_proto_.model_version();
  }
  return kNoVersion;
}

void Model::SetModelVersion(onnxruntime::Version version) {
  model_proto_.set_model_version(version);
}

const std::string Model::DocString() const {
  if (model_proto_.has_doc_string()) {
    return model_proto_.doc_string();
  }
  return std::string();
}

void Model::SetDocString(const std::string& doc_string) {
  model_proto_.set_doc_string(doc_string);
}

const std::string Model::GraphDocString() const {
  if (model_proto_.has_graph() && model_proto_.graph().has_doc_string()) {
    return model_proto_.graph().doc_string();
  }
  return std::string();
}

#endif  // !defined(ORT_MINIMAL_BUILD)

const ModelMetaData& Model::MetaData() const noexcept {
  return model_metadata_;
}

Graph& Model::MainGraph() noexcept {
  return *graph_;
}

const Graph& Model::MainGraph() const noexcept {
  return *graph_;
}

#if !defined(ORT_MINIMAL_BUILD)
ModelProto Model::ToProto() {
  // We want to return back the original proto
  // To that end invoke const overload of ToGraphProto()
  // that returns by value and, therefore, allows us to filter
  // out dense duplicates of sparse initializers and leave the original
  // proto intact.
  ModelProto result(model_proto_);
  const auto& graph = *graph_;
  *(result.mutable_graph()) = graph.ToGraphProto();
  return result;
}

Status Model::Load(std::istream& model_istream, ModelProto* p_model_proto) {
  if (!model_istream.good()) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid istream object.");
  }
  if (!p_model_proto) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Null model_proto ptr.");
  }

  google::protobuf::io::IstreamInputStream zero_copy_input(&model_istream);
  const bool result = p_model_proto->ParseFromZeroCopyStream(&zero_copy_input) && model_istream.eof();
  if (!result) {
    return Status(ONNXRUNTIME, INVALID_PROTOBUF, "Failed to load model because protobuf parsing failed.");
  }
  return Status::OK();
}

Status Model::Load(const ModelProto& model_proto,
                   std::shared_ptr<Model>& model,
                   const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                   const logging::Logger& logger) {
  return Model::Load(model_proto, PathString{}, model, local_registries, logger);
}

Status Model::Load(const ModelProto& model_proto,
                   const PathString& model_path,
                   std::shared_ptr<Model>& model,
                   const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                   const logging::Logger& logger) {
  // we expect a graph to be present
  if (!utils::HasGraph(model_proto)) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "No graph was found in the protobuf.");
  }

  // need to call private ctor so can't use make_shared
  GSL_SUPPRESS(r .11)

  auto status = Status::OK();
  ORT_TRY {
    model.reset(new Model(model_proto, model_path, local_registries, logger));
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = Status(ONNXRUNTIME, INVALID_ARGUMENT, "Failed to load model with error: " + std::string(ex.what()));
    });
  }
  ORT_RETURN_IF_ERROR(status);

  Graph::ResolveOptions options;
  options.no_proto_sync_required = true;
  ORT_RETURN_IF_ERROR(model->MainGraph().Resolve(options));

  return status;
}

Status Model::Load(ModelProto&& model_proto,
                   std::shared_ptr<Model>& model,
                   const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                   const logging::Logger& logger) {
  return Model::Load(std::move(model_proto), PathString{}, model, local_registries, logger);
}

Status Model::Load(ModelProto&& model_proto,
                   const PathString& model_path,
                   std::shared_ptr<Model>& model,
                   const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                   const logging::Logger& logger) {
  // we expect a graph to be present
  if (!utils::HasGraph(model_proto)) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "No graph was found in the protobuf.");
  }

  // need to call private ctor so can't use make_shared
  GSL_SUPPRESS(r .11)
  auto status = Status::OK();
  ORT_TRY {
    model.reset(new Model(std::move(model_proto), model_path, local_registries, logger));
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = Status(ONNXRUNTIME, INVALID_ARGUMENT, "Failed to load model with error: " + std::string(ex.what()));
    });
  }
  ORT_RETURN_IF_ERROR(status);

  Graph::ResolveOptions options;
  options.no_proto_sync_required = true;
  ORT_RETURN_IF_ERROR(model->MainGraph().Resolve(options));

  return status;
}

template <typename T, typename Loader>
static Status LoadModelHelper(const T& file_path, Loader loader) {
  int fd;
  Status status = Env::Default().FileOpenRd(file_path, fd);
  if (!status.IsOK()) {
    if (status.Category() == common::SYSTEM) {
      switch (status.Code()) {
        case ENOENT:
          return ORT_MAKE_STATUS(ONNXRUNTIME, NO_SUCHFILE, "Load model ", ToMBString(file_path),
                                 " failed. File doesn't exist");
        case EINVAL:
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Load model ", ToMBString(file_path), " failed");
        default:
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "system error number ", status.Code());
      }
    }
  }

  ORT_TRY {
    status = loader(fd);
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = Status(ONNXRUNTIME, FAIL, ex.what());
    });
  }

  if (!status.IsOK()) {
    GSL_SUPPRESS(es .84)
    ORT_IGNORE_RETURN_VALUE(Env::Default().FileClose(fd));
    return status;
  }
  return Env::Default().FileClose(fd);
}

template <typename T>
static Status LoadModel(const T& file_path, ONNX_NAMESPACE::ModelProto& model_proto) {
  const auto loader = [&model_proto](int fd) {
    return Model::Load(fd, model_proto);
  };

  return LoadModelHelper(file_path, loader);
}

template <typename T>
static Status LoadModel(const T& file_path, std::shared_ptr<Model>& p_model,
                        const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                        const logging::Logger& logger) {
  const auto loader = [&file_path, &p_model, local_registries, &logger](int fd) {
    return Model::Load(fd, ToPathString(file_path), p_model, local_registries, logger);
  };

  return LoadModelHelper(file_path, loader);
}

template <typename T>
static Status SaveModel(Model& model, const T& file_path) {
  int fd;
  Status status = Env::Default().FileOpenWr(file_path, fd);
  ORT_RETURN_IF_ERROR(status);

  ORT_TRY {
    status = Model::Save(model, fd);
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = Status(ONNXRUNTIME, FAIL, ex.what());
    });
  }
  if (!status.IsOK()) {
    GSL_SUPPRESS(es .84)
    ORT_IGNORE_RETURN_VALUE(Env::Default().FileClose(fd));
    return status;
  }
  return Env::Default().FileClose(fd);
}

#ifdef _WIN32
Status Model::Save(Model& model, const std::wstring& file_path) {
  return SaveModel(model, file_path);
}
#endif

Status Model::Load(const PathString& file_path,
                   ONNX_NAMESPACE::ModelProto& model_proto) {
  return LoadModel(file_path, model_proto);
}

GSL_SUPPRESS(r .30)  // spurious warnings. p_model is potentially reset in the internal call to Load
GSL_SUPPRESS(r .35)
Status Model::Load(const PathString& file_path, std::shared_ptr<Model>& p_model,
                   const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                   const logging::Logger& logger) {
  return LoadModel(file_path, p_model, local_registries, logger);
}

Status Model::Save(Model& model, const std::string& file_path) {
  return SaveModel(model, file_path);
}

Status Model::LoadFromBytes(int count, void* p_bytes, /*out*/ ONNX_NAMESPACE::ModelProto& model_proto) {
  const bool result = model_proto.ParseFromArray(p_bytes, count);
  if (!result) {
    return Status(ONNXRUNTIME, INVALID_PROTOBUF, "Protobuf parsing failed.");
  }

  return Status::OK();
}

Status Model::LoadFromBytes(int count, void* p_bytes, /*out*/ std::shared_ptr<Model>& p_model,
                            const IOnnxRuntimeOpSchemaRegistryList* local_registries, const logging::Logger& logger) {
  return LoadFromBytes(count, p_bytes, PathString{}, p_model, local_registries, logger);
}

Status Model::LoadFromBytes(int count, void* p_bytes, const PathString& model_path,
                            std::shared_ptr<Model>& p_model, const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                            const logging::Logger& logger) {
  ModelProto model_proto;

  auto status = LoadFromBytes(count, p_bytes, model_proto);
  if (!status.IsOK()) {
    return status;
  }

  p_model = std::make_shared<Model>(std::move(model_proto), model_path, local_registries, logger);

  Graph::ResolveOptions options;
  options.no_proto_sync_required = true;
  ORT_RETURN_IF_ERROR(p_model->MainGraph().Resolve(options));

  return Status::OK();
}

using ::google::protobuf::io::CodedInputStream;
using ::google::protobuf::io::FileInputStream;
using ::google::protobuf::io::ZeroCopyInputStream;

Status Model::Load(int fd, ONNX_NAMESPACE::ModelProto& model_proto) {
  if (fd < 0) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "<p_fd> less than 0.");
  }

#if GOOGLE_PROTOBUF_VERSION >= 3002000
  size_t file_size = 0;
  int block_size = -1;
  Status st = Env::Default().GetFileLength(fd, file_size);
  if (st.IsOK()) {
    block_size = std::min(DEFAULT_PROTOBUF_BLOCK_SIZE, static_cast<int>(file_size));
  }
  FileInputStream input(fd, block_size);
  const bool result = model_proto.ParseFromZeroCopyStream(&input) && input.GetErrno() == 0;
  if (!result) {
    return Status(ONNXRUNTIME, INVALID_PROTOBUF, "Protobuf parsing failed.");
  }
#else
  // CNTK uses ORT as a submodule in order to use its GraphIR code.
  // CNTK needs to be built with protobuf 3.1.0 for its version specific features.
  // This code block is needed to support CNTK and any other
  // GraphIR client that will be built with protobuf at a version older than 3.2.0.
  FileInputStream fs(fd);
  CodedInputStream cis(&fs);

  // Allows protobuf library versions < 3.2.0 to parse messages greater than 64MB.
  cis.SetTotalBytesLimit(INT_MAX);
  if (!model_proto->ParseFromCodedStream(&cis)) {
    return Status(ONNXRUNTIME, INVALID_PROTOBUF, "Protobuf parsing failed.");
  }
#endif
  return Status::OK();
}

Status Model::Load(int fd, std::shared_ptr<Model>& p_model, const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                   const logging::Logger& logger) {
  return Load(fd, PathString{}, p_model, local_registries, logger);
}

Status Model::Load(int fd, const PathString& model_path, std::shared_ptr<Model>& p_model,
                   const IOnnxRuntimeOpSchemaRegistryList* local_registries, const logging::Logger& logger) {
  ModelProto model_proto;

  ORT_RETURN_IF_ERROR(Load(fd, model_proto));

  p_model = std::make_shared<Model>(std::move(model_proto), model_path, local_registries, logger);

  Graph::ResolveOptions options;
  options.no_proto_sync_required = true;
  ORT_RETURN_IF_ERROR(p_model->MainGraph().Resolve(options));

  return Status::OK();
}

Status Model::Save(Model& model, int p_fd) {
  if (p_fd < 0) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "<p_fd> is less than 0.");
  }

  ORT_RETURN_IF_ERROR(model.MainGraph().Resolve());

  auto model_proto = model.ToProto();
  google::protobuf::io::FileOutputStream output(p_fd);
  const bool result = model_proto.SerializeToZeroCopyStream(&output) && output.Flush();
  if (result) {
    return Status::OK();
  }
  return Status(ONNXRUNTIME, INVALID_PROTOBUF, "Protobuf serialization failed.");
}

common::Status Model::SaveToOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                                      flatbuffers::Offset<fbs::Model>& fbs_model) const {
  auto producer_name = experimental::utils::SaveStringToOrtFormat(
      builder, model_proto_.has_producer_name(), model_proto_.producer_name());
  auto producer_version = experimental::utils::SaveStringToOrtFormat(
      builder, model_proto_.has_producer_version(), model_proto_.producer_version());
  auto domain = builder.CreateSharedString(model_proto_.domain());
  auto doc_string = experimental::utils::SaveStringToOrtFormat(
      builder, model_proto_.has_doc_string(), model_proto_.doc_string());
  auto graph_doc_string = experimental::utils::SaveStringToOrtFormat(
      builder, model_proto_.has_graph() && model_proto_.graph().has_doc_string(), model_proto_.graph().doc_string());

  std::vector<flatbuffers::Offset<fbs::OperatorSetId>> op_set_ids_vec;
  op_set_ids_vec.reserve(model_proto_.opset_import().size());
  for (const auto& entry : model_proto_.opset_import()) {
    auto op_set_domain = builder.CreateSharedString(entry.domain());
    fbs::OperatorSetIdBuilder ob(builder);
    ob.add_domain(op_set_domain);
    ob.add_version(entry.version());
    op_set_ids_vec.push_back(ob.Finish());
  }
  auto op_set_ids = builder.CreateVector(op_set_ids_vec);

  flatbuffers::Offset<fbs::Graph> fbs_graph;
  ORT_RETURN_IF_ERROR(graph_->SaveToOrtFormat(builder, fbs_graph));

  fbs::ModelBuilder mb(builder);
  mb.add_ir_version(model_proto_.ir_version());
  mb.add_opset_import(op_set_ids);
  mb.add_producer_name(producer_name);
  mb.add_producer_version(producer_version);
  mb.add_domain(domain);
  mb.add_model_version(model_proto_.model_version());
  mb.add_doc_string(doc_string);
  mb.add_graph_doc_string(graph_doc_string);
  mb.add_graph(fbs_graph);

  // add graph
  fbs_model = mb.Finish();

  return Status::OK();
}

#endif  // !defined(ORT_MINIMAL_BUILD)

Model::Model() : model_path_{} {
}

#if defined(ENABLE_ORT_FORMAT_LOAD)
common::Status Model::LoadFromOrtFormat(const fbs::Model& fbs_model,
#if !defined(ORT_MINIMAL_BUILD)
                                        const IOnnxRuntimeOpSchemaRegistryList* local_registries,
#endif
                                        const logging::Logger& logger,
                                        std::unique_ptr<Model>& model) {
  model.reset(new Model());

#if !defined(ORT_MINIMAL_BUILD)
  LOAD_STR_FROM_ORT_FORMAT(model->model_proto_, producer_name, fbs_model.producer_name());
  LOAD_STR_FROM_ORT_FORMAT(model->model_proto_, producer_version, fbs_model.producer_version());
  LOAD_STR_FROM_ORT_FORMAT(model->model_proto_, domain, fbs_model.domain());
  LOAD_STR_FROM_ORT_FORMAT(model->model_proto_, doc_string, fbs_model.doc_string());
  if (fbs_model.graph_doc_string()) {
    model->model_proto_.mutable_graph()->set_doc_string(fbs_model.graph_doc_string()->c_str());
  }
  model->model_proto_.set_model_version(fbs_model.model_version());
  model->model_proto_.set_ir_version(fbs_model.ir_version());

  auto schema_registry = std::make_shared<SchemaRegistryManager>();
  if (local_registries != nullptr) {
    for (const auto& schema_collection : *local_registries) {
      schema_registry->RegisterRegistry(schema_collection);
    }
  }
#else
  experimental::utils::LoadStringFromOrtFormat(model->producer_name_, fbs_model.producer_name());
  experimental::utils::LoadStringFromOrtFormat(model->producer_version_, fbs_model.producer_version());
  experimental::utils::LoadStringFromOrtFormat(model->domain_, fbs_model.domain());
  experimental::utils::LoadStringFromOrtFormat(model->doc_string_, fbs_model.doc_string());
  experimental::utils::LoadStringFromOrtFormat(model->graph_doc_string_, fbs_model.graph_doc_string());
  model->model_version_ = fbs_model.model_version();
  model->ir_version_ = fbs_model.ir_version();
#endif

  std::unordered_map<std::string, int> domain_to_version;
  ORT_RETURN_IF_ERROR(experimental::utils::LoadOpsetImportOrtFormat(fbs_model.opset_import(), domain_to_version));

  auto fbs_graph = fbs_model.graph();
  ORT_RETURN_IF(nullptr == fbs_graph, "Graph is null. Invalid ORT format model.");

#if !defined(ORT_MINIMAL_BUILD)
  ORT_RETURN_IF_ERROR(Graph::LoadFromOrtFormat(*fbs_graph, *model, domain_to_version, schema_registry, logger,
                                               model->graph_));
#else
  ORT_RETURN_IF_ERROR(Graph::LoadFromOrtFormat(*fbs_graph, *model, domain_to_version, logger, model->graph_));
#endif
  return Status::OK();
}
#endif  // defined(ENABLE_ORT_FORMAT_LOAD)

}  // namespace onnxruntime
