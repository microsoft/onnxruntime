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

#include "core/common/gsl.h"

#include "core/platform/env.h"

#if !defined(ORT_MINIMAL_BUILD)
#include "core/graph/schema_registry.h"
#include "core/graph/function_utils.h"
#endif

#if defined(__wasm__)
#include <emscripten.h>
#endif

using namespace ONNX_NAMESPACE;
using namespace onnxruntime;
using namespace onnxruntime::common;

namespace onnxruntime {

#if !defined(ORT_MINIMAL_BUILD)

static constexpr int DEFAULT_PROTOBUF_BLOCK_SIZE = 4 * 1024 * 1024;

Model::Model(const std::string& graph_name,
             bool is_onnx_domain_only,
             const ModelMetaData& model_metadata,
             const PathString& model_path,
             const IOnnxRuntimeOpSchemaRegistryList& local_registries,
             const std::unordered_map<std::string, int>& domain_to_version,
             const std::vector<ONNX_NAMESPACE::FunctionProto>& model_local_functions,
             const logging::Logger& logger,
             const ModelOptions& options)
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

  // IsAllowReleasedONNXOpsetsOnlySet() checks for the appropriate env var in the process (i.e.) process-wide
  // `allow_released_opsets_only` is for this specific Model instance
  // We will only support released opsets iff IsAllowReleasedONNXOpsetsOnlySet() and `allow_released_opsets_only`
  // are both true
  auto allow_released_opsets_only_final =
      options.allow_released_opsets_only && model_load_utils::IsAllowReleasedONNXOpsetsOnlySet();

  auto* p_domain_to_version = &domain_to_version;
  DomainToVersionMap domain_to_version_static;
  domain_to_version_static = allow_released_opsets_only_final
                                 ? schema_registry->GetLastReleasedOpsetVersions(is_onnx_domain_only)
                                 : schema_registry->GetLatestOpsetVersions(is_onnx_domain_only);
  if (p_domain_to_version->empty()) {
    p_domain_to_version = &domain_to_version_static;
  }

  for (const auto& [domain, version] : *p_domain_to_version) {
    model_load_utils::ValidateOpsetForDomain(domain_to_version_static, logger, allow_released_opsets_only_final,
                                             domain, version);
    const gsl::not_null<OperatorSetIdProto*> opset_id_proto{model_proto_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }

  for (auto& func : model_local_functions) {
    auto func_ptr = model_proto_.add_functions();
    func_ptr->CopyFrom(func);
    model_local_functions_[function_utils::GetFunctionIdentifier(func_ptr->domain(), func_ptr->name())] = func_ptr;
  }

  model_local_function_templates_.reserve(model_proto_.functions().size());
  model_local_function_templates_maps_.reserve(model_proto_.functions().size());
  for (auto& func : model_proto_.functions()) {
    auto func_schema_ptr = function_utils::CreateSchema(func.domain(),
                                                        func.name(),
                                                        model_local_functions_,
                                                        *p_domain_to_version,
                                                        *schema_registry,
                                                        logger,
                                                        allow_released_opsets_only_final);
    auto func_template_ptr = std::make_unique<FunctionTemplate>();
    func_template_ptr->op_schema_ = std::move(func_schema_ptr);
    func_template_ptr->onnx_func_proto_ = &func;
    model_local_function_templates_.push_back(std::move(func_template_ptr));
    model_local_function_templates_maps_[function_utils::GetFunctionIdentifier(func.domain(), func.name())] = model_local_function_templates_.back().get();
  }

  // need to call private ctor so can't use make_shared
  GSL_SUPPRESS(r .11)
  graph_.reset(new Graph(*this, model_proto_.mutable_graph(), *p_domain_to_version, IrVersion(), schema_registry,
                         logger, options.strict_shape_type_inference));
}

Model::Model(const ModelProto& model_proto, const PathString& model_path,
             const IOnnxRuntimeOpSchemaRegistryList* local_registries, const logging::Logger& logger,
             const ModelOptions& options)
    : Model(ModelProto(model_proto), model_path, local_registries, logger, options) {
}

Model::Model(ModelProto&& model_proto, const PathString& model_path,
             const IOnnxRuntimeOpSchemaRegistryList* local_registries,
             const logging::Logger& logger, const ModelOptions& options)
    : model_path_(Path::Parse(model_path)) {
  if (!utils::HasGraph(model_proto)) {
    ORT_THROW("ModelProto does not have a graph.");
  }

  if (model_proto.opset_import_size() == 0) {
    ORT_THROW(
        "Missing opset in the model. All ModelProtos MUST have at least one entry that"
        " specifies which version of the ONNX OperatorSet is being imported.");
  }

  if (!model_proto.has_ir_version()) {
    ORT_THROW("Missing model IR version.");
  }

  if (const auto ir_version = model_proto.ir_version();
      ir_version > ONNX_NAMESPACE::Version::IR_VERSION) {
    ORT_THROW("Unsupported model IR version: ", ir_version,
              ", max supported IR version: ", ONNX_NAMESPACE::Version::IR_VERSION);
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

  // IsAllowReleasedONNXOpsetsOnlySet() checks for the appropriate env var in the process (i.e.) process-wide
  // `allow_released_opsets_only` is for this specific Model instance
  // We will only support released opsets iff IsAllowReleasedONNXOpsetsOnlySet() and `allow_released_opsets_only`
  // are both true
  auto allow_official_onnx_release_only_final =
      options.allow_released_opsets_only && model_load_utils::IsAllowReleasedONNXOpsetsOnlySet();

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
                                             allow_official_onnx_release_only_final, domain, version);

    // We need to overwrite the domain here with ("") or else the loop below will try to find ("")
    // in the map and if not found (when domain == kOnnxDomainAlias), adds an entry for ("", 11).
    // This effectively ignores the opset version specified by the model for the onnx domain.
    if (domain == kOnnxDomainAlias) {
      domain_to_version[kOnnxDomain] = version;
    } else {
      domain_to_version[domain] = version;
    }
  }

  auto domain_map = allow_official_onnx_release_only_final
                        ? schema_registry->GetLastReleasedOpsetVersions(false)
                        : schema_registry->GetLatestOpsetVersions(false);
  for (const auto& [domain, version] : domain_map) {
    if (domain_to_version.find(domain) == domain_to_version.end()) {
      domain_to_version[domain] = version;
      const gsl::not_null<OperatorSetIdProto*> opset_id_proto{model_proto_.add_opset_import()};
      opset_id_proto->set_domain(domain);
      opset_id_proto->set_version(version);
    }
  }

  std::vector<const ONNX_NAMESPACE::FunctionProto*> model_local_functions;
  for (auto& func : model_proto_.functions()) {
    model_local_functions_[function_utils::GetFunctionIdentifier(func.domain(), func.name())] = &func;
  }

  model_local_function_templates_.reserve(model_proto_.functions().size());
  model_local_function_templates_maps_.reserve(model_proto_.functions().size());
  for (auto& func : model_proto_.functions()) {
    auto func_schema_ptr = function_utils::CreateSchema(func.domain(),
                                                        func.name(),
                                                        model_local_functions_,
                                                        domain_to_version,
                                                        *schema_registry,
                                                        logger,
                                                        allow_official_onnx_release_only_final);
    auto func_template_ptr = std::make_unique<FunctionTemplate>();
    func_template_ptr->op_schema_ = std::move(func_schema_ptr);
    func_template_ptr->onnx_func_proto_ = &func;
    model_local_function_templates_.push_back(std::move(func_template_ptr));
    model_local_function_templates_maps_[function_utils::GetFunctionIdentifier(func.domain(), func.name())] =
        model_local_function_templates_.back().get();
  }

  // create instance. need to call private ctor so can't use make_unique
  GSL_SUPPRESS(r .11)
  graph_.reset(new Graph(*this, model_proto_.mutable_graph(), domain_to_version, IrVersion(), schema_registry,
                         logger, options.strict_shape_type_inference));
}

const InlinedHashMap<std::string, FunctionTemplate*>& Model::GetModelLocalFunctionTemplates() const {
  return model_local_function_templates_maps_;
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

ModelProto Model::ToGraphProtoWithExternalInitializers(const std::string& external_file_name,
                                                       const PathString& file_path,
                                                       size_t initializer_size_threshold) {
  ModelProto result(model_proto_);
  const auto& graph = *graph_;
  *(result.mutable_graph()) = graph.ToGraphProtoWithExternalInitializers(external_file_name,
                                                                         file_path,
                                                                         initializer_size_threshold);
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
                   const logging::Logger& logger,
                   const ModelOptions& options) {
  return Model::Load(model_proto, PathString{}, model, local_registries, logger, options);
}

Status Model::Load(const ModelProto& model_proto,
                   const PathString& model_path,
                   std::shared_ptr<Model>& model,
                   const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                   const logging::Logger& logger,
                   const ModelOptions& options) {
  // we expect a graph to be present
  if (!utils::HasGraph(model_proto)) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "No graph was found in the protobuf.");
  }

  // need to call private ctor so can't use make_shared
  GSL_SUPPRESS(r .11)

  auto status = Status::OK();
  ORT_TRY {
    model = std::make_unique<Model>(model_proto, model_path, local_registries, logger, options);
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = Status(ONNXRUNTIME, INVALID_ARGUMENT, "Failed to load model with error: " + std::string(ex.what()));
    });
  }
  ORT_RETURN_IF_ERROR(status);

  Graph::ResolveOptions resolve_options;
  resolve_options.no_proto_sync_required = true;
  ORT_RETURN_IF_ERROR(model->MainGraph().Resolve(resolve_options));

  return status;
}

Status Model::Load(ModelProto&& model_proto,
                   std::shared_ptr<Model>& model,
                   const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                   const logging::Logger& logger,
                   const ModelOptions& options) {
  return Model::Load(std::move(model_proto), PathString{}, model, local_registries, logger, options);
}

Status Model::Load(ModelProto&& model_proto,
                   const PathString& model_path,
                   std::shared_ptr<Model>& model,
                   const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                   const logging::Logger& logger,
                   const ModelOptions& options) {
  // we expect a graph to be present
  if (!utils::HasGraph(model_proto)) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "No graph was found in the protobuf.");
  }

  // need to call private ctor so can't use make_shared
  GSL_SUPPRESS(r .11)
  auto status = Status::OK();
  ORT_TRY {
    model = std::make_unique<Model>(std::move(model_proto), model_path, local_registries, logger, options);
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = Status(ONNXRUNTIME, INVALID_ARGUMENT, "Failed to load model with error: " + std::string(ex.what()));
    });
  }
  ORT_RETURN_IF_ERROR(status);

  Graph::ResolveOptions resolve_options;
  resolve_options.no_proto_sync_required = true;
  ORT_RETURN_IF_ERROR(model->MainGraph().Resolve(resolve_options));

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
          return ORT_MAKE_STATUS(ONNXRUNTIME, NO_SUCHFILE, "Load model ", ToUTF8String(file_path),
                                 " failed. File doesn't exist");
        case EINVAL:
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Load model ", ToUTF8String(file_path), " failed");
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
                        const logging::Logger& logger, const ModelOptions& options) {
  const auto loader = [&file_path, &p_model, local_registries, &logger, &options](int fd) {
    return Model::Load(fd, ToPathString(file_path), p_model, local_registries, logger, options);
  };

  return LoadModelHelper(file_path, loader);
}

template <typename T>
static Status SaveModel(Model& model, const T& file_path) {
#if defined(__wasm__) && defined(ORT_ENABLE_WEBASSEMBLY_OUTPUT_OPTIMIZED_MODEL)
  ORT_RETURN_IF_ERROR(model.MainGraph().Resolve());
  auto model_proto = model.ToProto();
  auto buffer_size = model_proto.ByteSizeLong();
  void* buffer = malloc(buffer_size);
  model_proto.SerializeToArray(buffer, buffer_size);

  EM_ASM(({
           const buffer = $0;
           const buffer_size = $1;
           const file_path = UTF8ToString($2);
           const bytes = new Uint8Array(buffer_size);
           bytes.set(HEAPU8.subarray(buffer, buffer + buffer_size));
           if (typeof process == 'object' && typeof process.versions == 'object' && typeof process.versions.node == 'string') {
             // Node.js
             require('fs').writeFileSync(file_path, bytes);
           } else {
             // Browser
             const file = new File([bytes], file_path, {type: "application/octet-stream" });
             const url = URL.createObjectURL(file);
             window.open(url, '_blank');
           }
         }),
         reinterpret_cast<int32_t>(buffer),
         static_cast<int32_t>(buffer_size),
         reinterpret_cast<int32_t>(file_path.c_str()));

  free(buffer);
  return Status::OK();

#else
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
#endif
}

#ifdef _WIN32
Status Model::Save(Model& model, const std::wstring& file_path) {
  return SaveModel(model, file_path);
}
#endif

template <typename T>
static Status SaveModelWithExternalInitializers(Model& model,
                                                const T& file_path,
                                                const std::string& external_file_name,
                                                size_t initializer_size_threshold) {
  int fd = 0;
  Status status = Env::Default().FileOpenWr(file_path, fd);
  ORT_RETURN_IF_ERROR(status);

  ORT_TRY {
    status = Model::SaveWithExternalInitializers(model, fd, file_path, external_file_name,
                                                 initializer_size_threshold);
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

Status Model::Load(const PathString& file_path,
                   ONNX_NAMESPACE::ModelProto& model_proto) {
  return LoadModel(file_path, model_proto);
}

GSL_SUPPRESS(r .30)  // spurious warnings. p_model is potentially reset in the internal call to Load
GSL_SUPPRESS(r .35)
Status Model::Load(const PathString& file_path, std::shared_ptr<Model>& p_model,
                   const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                   const logging::Logger& logger, const ModelOptions& options) {
  return LoadModel(file_path, p_model, local_registries, logger, options);
}

Status Model::Save(Model& model, const std::string& file_path) {
  return SaveModel(model, file_path);
}

Status Model::SaveWithExternalInitializers(Model& model, const PathString& file_path,
                                           const std::string& external_file_name,
                                           size_t initializer_size_threshold) {
  return SaveModelWithExternalInitializers(model, file_path, external_file_name, initializer_size_threshold);
}

Status Model::LoadFromBytes(int count, void* p_bytes, /*out*/ ONNX_NAMESPACE::ModelProto& model_proto) {
  const bool result = model_proto.ParseFromArray(p_bytes, count);
  if (!result) {
    return Status(ONNXRUNTIME, INVALID_PROTOBUF, "Protobuf parsing failed.");
  }

  return Status::OK();
}

Status Model::LoadFromBytes(int count, void* p_bytes, /*out*/ std::shared_ptr<Model>& p_model,
                            const IOnnxRuntimeOpSchemaRegistryList* local_registries, const logging::Logger& logger,
                            const ModelOptions& options) {
  return LoadFromBytes(count, p_bytes, PathString{}, p_model, local_registries, logger, options);
}

Status Model::LoadFromBytes(int count, void* p_bytes, const PathString& model_path,
                            std::shared_ptr<Model>& p_model, const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                            const logging::Logger& logger, const ModelOptions& options) {
  ModelProto model_proto;

  auto status = LoadFromBytes(count, p_bytes, model_proto);
  if (!status.IsOK()) {
    return status;
  }

  p_model = std::make_shared<Model>(std::move(model_proto), model_path, local_registries, logger, options);

  Graph::ResolveOptions resolve_options;
  resolve_options.no_proto_sync_required = true;
  ORT_RETURN_IF_ERROR(p_model->MainGraph().Resolve(resolve_options));

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
                   const logging::Logger& logger, const ModelOptions& options) {
  return Load(fd, PathString{}, p_model, local_registries, logger, options);
}

Status Model::Load(int fd, const PathString& model_path, std::shared_ptr<Model>& p_model,
                   const IOnnxRuntimeOpSchemaRegistryList* local_registries, const logging::Logger& logger,
                   const ModelOptions& options) {
  ModelProto model_proto;

  ORT_RETURN_IF_ERROR(Load(fd, model_proto));

  p_model = std::make_shared<Model>(std::move(model_proto), model_path, local_registries, logger, options);

  Graph::ResolveOptions resolve_options;
  resolve_options.no_proto_sync_required = true;
  ORT_RETURN_IF_ERROR(p_model->MainGraph().Resolve(resolve_options));

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

Status Model::SaveWithExternalInitializers(Model& model,
                                           int fd,
                                           const PathString& file_path,
                                           const std::string& external_file_name,
                                           size_t initializer_size_threshold) {
  if (fd < 0) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "<fd> is less than 0.");
  }

  ORT_RETURN_IF_ERROR(model.MainGraph().Resolve());

  auto model_proto = model.ToGraphProtoWithExternalInitializers(external_file_name, file_path, initializer_size_threshold);
  google::protobuf::io::FileOutputStream output(fd);
  const bool result = model_proto.SerializeToZeroCopyStream(&output) && output.Flush();
  if (result) {
    return Status::OK();
  }
  return Status(ONNXRUNTIME, INVALID_PROTOBUF, "Protobuf serialization failed.");
}

common::Status Model::SaveToOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                                      flatbuffers::Offset<fbs::Model>& fbs_model) const {
  auto producer_name = fbs::utils::SaveStringToOrtFormat(
      builder, model_proto_.has_producer_name(), model_proto_.producer_name());
  auto producer_version = fbs::utils::SaveStringToOrtFormat(
      builder, model_proto_.has_producer_version(), model_proto_.producer_version());
  auto domain = builder.CreateSharedString(model_proto_.domain());
  auto doc_string = fbs::utils::SaveStringToOrtFormat(
      builder, model_proto_.has_doc_string(), model_proto_.doc_string());
  auto graph_doc_string = fbs::utils::SaveStringToOrtFormat(
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

  flatbuffers::Offset<flatbuffers::Vector<
      flatbuffers::Offset<onnxruntime::fbs::StringStringEntry>>>
      metadata_props{0};

  // We will not serialize an empty metadata_props
  if (!model_metadata_.empty()) {
    std::vector<flatbuffers::Offset<onnxruntime::fbs::StringStringEntry>> metadata_props_vec;
    metadata_props_vec.reserve(model_metadata_.size());
    for (const auto& prop : model_metadata_) {
      metadata_props_vec.push_back(
          fbs::CreateStringStringEntryDirect(builder, prop.first.c_str(), prop.second.c_str()));
    }
    metadata_props = builder.CreateVector(metadata_props_vec);
  }

  flatbuffers::Offset<fbs::Graph> fbs_graph;
  ORT_RETURN_IF_ERROR(graph_->SaveToOrtFormat(builder, fbs_graph));

  fbs::ModelBuilder mb(builder);
  mb.add_ir_version(IrVersion());
  mb.add_opset_import(op_set_ids);
  mb.add_producer_name(producer_name);
  mb.add_producer_version(producer_version);
  mb.add_domain(domain);
  mb.add_model_version(ModelVersion());
  mb.add_doc_string(doc_string);
  mb.add_graph_doc_string(graph_doc_string);
  mb.add_metadata_props(metadata_props);
  mb.add_graph(fbs_graph);

  // add graph
  fbs_model = mb.Finish();

  return Status::OK();
}

#endif  // !defined(ORT_MINIMAL_BUILD)

Model::Model() : model_path_{} {
}

common::Status Model::LoadFromOrtFormat(const fbs::Model& fbs_model,
#if !defined(ORT_MINIMAL_BUILD)
                                        const IOnnxRuntimeOpSchemaRegistryList* local_registries,
#endif
                                        const OrtFormatLoadOptions& load_options,
                                        const logging::Logger& logger,
                                        std::unique_ptr<Model>& model) {
  model = std::make_unique<Model>();

  // Load the model metadata
  if (const auto* fbs_metadata_props = fbs_model.metadata_props()) {
    model->model_metadata_.reserve(fbs_metadata_props->size());
    for (const auto* prop : *fbs_metadata_props) {
      ORT_RETURN_IF(nullptr == prop, "Null entry in metadata_props. Invalid ORT format model.");
      std::string key, value;
      fbs::utils::LoadStringFromOrtFormat(key, prop->key());
      fbs::utils::LoadStringFromOrtFormat(value, prop->value());
      model->model_metadata_.insert({key, value});
    }
  }

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

  // Populate the metadata to model_proto
  for (auto& metadata : model->model_metadata_) {
    const gsl::not_null<StringStringEntryProto*> prop{model->model_proto_.add_metadata_props()};
    prop->set_key(metadata.first);
    prop->set_value(metadata.second);
  }
#else
  fbs::utils::LoadStringFromOrtFormat(model->producer_name_, fbs_model.producer_name());
  fbs::utils::LoadStringFromOrtFormat(model->producer_version_, fbs_model.producer_version());
  fbs::utils::LoadStringFromOrtFormat(model->domain_, fbs_model.domain());
  fbs::utils::LoadStringFromOrtFormat(model->doc_string_, fbs_model.doc_string());
  fbs::utils::LoadStringFromOrtFormat(model->graph_doc_string_, fbs_model.graph_doc_string());
  model->model_version_ = fbs_model.model_version();
  model->ir_version_ = fbs_model.ir_version();
#endif

  std::unordered_map<std::string, int> domain_to_version;
  ORT_RETURN_IF_ERROR(fbs::utils::LoadOpsetImportOrtFormat(fbs_model.opset_import(), domain_to_version));

  auto fbs_graph = fbs_model.graph();
  ORT_RETURN_IF(nullptr == fbs_graph, "Graph is null. Invalid ORT format model.");

#if !defined(ORT_MINIMAL_BUILD)
  // add the opset imports to the model_proto in case we're updating an ORT format model and need those to be
  // included when SaveToOrtFormat is called later
  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<OperatorSetIdProto*> opset_id_proto{model->model_proto_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }

  ORT_RETURN_IF_ERROR(Graph::LoadFromOrtFormat(*fbs_graph, *model, domain_to_version, schema_registry,
                                               load_options, logger, model->graph_));
#else
  ORT_RETURN_IF_ERROR(Graph::LoadFromOrtFormat(*fbs_graph, *model, domain_to_version,
                                               load_options, logger, model->graph_));
#endif
  return Status::OK();
}

}  // namespace onnxruntime
