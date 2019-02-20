// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/model.h"
#include <memory>

#ifdef _MSC_VER
#pragma warning(push)
// 'type' : forcing value to bool 'true' or 'false' (performance warning)
#pragma warning(disable : 4800)
#endif
#include <google/protobuf/io/coded_stream.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "gsl/pointers"
#include "gsl/gsl_util"

#include "core/platform/env.h"
#include "core/graph/schema_registry.h"
using namespace ONNX_NAMESPACE;
using namespace onnxruntime;
using namespace ::onnxruntime::common;

namespace onnxruntime {
Model::Model(const std::string& graph_name,
             bool is_onnx_domain_only,
             const ModelMetaData& model_metadata,
             const IOnnxRuntimeOpSchemaRegistryList& local_registries,
             const std::unordered_map<std::string, int>& domain_to_version,
             const std::vector<ONNX_NAMESPACE::FunctionProto>& model_functions) {
  model_proto_ = std::make_unique<ModelProto>();
  model_proto_->set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  model_proto_->mutable_graph()->set_name(graph_name);
  model_metadata_ = model_metadata;
  for (auto& metadata : model_metadata_) {
    const gsl::not_null<StringStringEntryProto*> prop{model_proto_->add_metadata_props()};
    prop->set_key(metadata.first);
    prop->set_value(metadata.second);
  }

  auto schema_registry = std::make_shared<SchemaRegistryManager>();
  for (auto schema_collection : local_registries) {
    schema_registry->RegisterRegistry(schema_collection);
  }

  auto* p_domain_to_version = &domain_to_version;
  std::unordered_map<std::string, int> domain_to_version_static;
  if (p_domain_to_version->empty()) {
    domain_to_version_static = schema_registry->GetLatestOpsetVersions(is_onnx_domain_only);
    p_domain_to_version = &domain_to_version_static;
  }

  for (auto domain : *p_domain_to_version) {
    const gsl::not_null<OperatorSetIdProto*> opset_id_proto{model_proto_->add_opset_import()};
    opset_id_proto->set_domain(domain.first);
    opset_id_proto->set_version(domain.second);
  }

  std::unordered_map<std::string, const ONNX_NAMESPACE::FunctionProto*> model_functions_map;
  for (auto& func : model_functions) {
    auto func_ptr = model_proto_->add_functions();
    func_ptr->CopyFrom(func);
    model_functions_map[func_ptr->name()] = func_ptr;
  }

  // need to call private ctor so can't use make_shared
  GSL_SUPPRESS(r .11)
  graph_.reset(new Graph(model_proto_->mutable_graph(), *p_domain_to_version, IrVersion(), schema_registry, model_functions_map));
}

Model::Model(const ModelProto& model_proto, const IOnnxRuntimeOpSchemaRegistryList* local_registries)
    : Model(std::make_unique<ModelProto>(model_proto), local_registries) {
}

Model::Model(std::unique_ptr<ModelProto> model_proto, const IOnnxRuntimeOpSchemaRegistryList* local_registries) {
  if (!model_proto) {
    throw std::invalid_argument("ModelProto was null.");
  }

  if (!model_proto->has_graph()) {
    throw std::invalid_argument("ModelProto does not have a graph.");
  }

  if (model_proto->opset_import_size() == 0) {
    throw std::invalid_argument(
        "Missing opset in the model. All ModelProtos MUST have at least one entry that"
        " specifies which version of the ONNX OperatorSet is being imported.");
  }

  model_proto_ = std::move(model_proto);
  for (auto& prop : model_proto_->metadata_props()) {
    model_metadata_[prop.key()] = prop.value();
  }

  auto schema_registry = std::make_shared<SchemaRegistryManager>();
  if (local_registries != nullptr) {
    for (auto schema_collection : *local_registries) {
      schema_registry->RegisterRegistry(schema_collection);
    }
  }

  std::unordered_map<std::string, int> domain_to_version;
  for (auto& opSet : model_proto_->opset_import()) {
    domain_to_version[opSet.domain()] = gsl::narrow_cast<int>(opSet.version());
  }

  auto domain_map = schema_registry->GetLatestOpsetVersions(false);
  for (auto domain : domain_map) {
    if (domain_to_version.find(domain.first) == domain_to_version.end()) {
      domain_to_version[domain.first] = domain.second;
      const gsl::not_null<OperatorSetIdProto*> opset_id_proto{model_proto_->add_opset_import()};
      opset_id_proto->set_domain(domain.first);
      opset_id_proto->set_version(domain.second);
    }
  }

  std::unordered_map<std::string, const ONNX_NAMESPACE::FunctionProto*> model_functions_map;
  for (auto& func : model_proto_->functions()) {
    model_functions_map[func.name()] = &func;
  }

  // create instance. need to call private ctor so can't use make_unique
  GSL_SUPPRESS(r .11)
  graph_.reset(new Graph(model_proto_->mutable_graph(), domain_to_version, IrVersion(), schema_registry, model_functions_map));
}

Version Model::IrVersion() const {
  if (model_proto_->has_ir_version()) {
    return model_proto_->ir_version();
  }
  return kNoVersion;
}

const std::string& Model::ProducerName() const {
  return model_proto_->producer_name();
}

void Model::SetProducerName(const std::string& producer_name) {
  model_proto_->set_producer_name(producer_name);
}

const std::string& Model::ProducerVersion() const {
  return model_proto_->producer_version();
}

void Model::SetProducerVersion(const std::string& producer_version) {
  model_proto_->set_producer_version(producer_version);
}

const std::string& Model::Domain() const {
  return model_proto_->domain();
}

void Model::SetDomain(const std::string& domain) {
  model_proto_->set_domain(domain);
}

Version Model::ModelVersion() const {
  if (model_proto_->has_model_version()) {
    return model_proto_->model_version();
  }
  return kNoVersion;
}

void Model::SetModelversion(onnxruntime::Version version) {
  model_proto_->set_model_version(version);
}

const std::string& Model::DocString() const {
  return model_proto_->doc_string();
}

void Model::SetDocString(const std::string& doc_string) {
  model_proto_->set_doc_string(doc_string);
}

const ModelMetaData& Model::MetaData() const noexcept {
  return model_metadata_;
}

Graph& Model::MainGraph() noexcept {
  return *graph_;
}

const Graph& Model::MainGraph() const noexcept {
  return *graph_;
}

void Model::AddFunction(const ONNX_NAMESPACE::FunctionProto& func_proto) {
  auto func_ptr = model_proto_->add_functions();
  func_ptr->CopyFrom(func_proto);
  graph_->AddFunction(func_ptr);
}

ModelProto Model::ToProto() {
  *(model_proto_->mutable_graph()) = graph_->ToGraphProto();
  return *model_proto_;
}

Status Model::Load(std::istream& model_istream, ModelProto* p_model_proto) {
  if (!model_istream.good()) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid istream object.");
  }
  if (!p_model_proto) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Null model_proto ptr.");
  }
  const bool result = p_model_proto->ParseFromIstream(&model_istream);
  if (!result) {
    return Status(ONNXRUNTIME, INVALID_PROTOBUF, "Failed to load model because protobuf parsing failed.");
  }
  return Status::OK();
}

Status Model::Load(const ModelProto& model_proto, std::shared_ptr<Model>& model, const IOnnxRuntimeOpSchemaRegistryList* local_registries) {
  // we expect a graph to be present
  if (!model_proto.has_graph()) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "No graph was found in the protobuf.");
  }

  // need to call private ctor so can't use make_shared
  GSL_SUPPRESS(r .11)
  try {
    model.reset(new Model(model_proto, local_registries));
  } catch (const std::exception& ex) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Failed to load model with error: " + std::string(ex.what()));
  }

  ORT_RETURN_IF_ERROR(model->MainGraph().Resolve(true));

  return Status::OK();
}

Status Model::Load(std::unique_ptr<ModelProto> p_model_proto, std::shared_ptr<Model>& model, const IOnnxRuntimeOpSchemaRegistryList* local_registries) {
  // we expect a graph to be present
  if (!p_model_proto->has_graph()) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "No graph was found in the protobuf.");
  }

  // need to call private ctor so can't use make_shared
  GSL_SUPPRESS(r .11)
  try {
    model.reset(new Model(std::move(p_model_proto), local_registries));
  } catch (const std::exception& ex) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Failed to load model with error: " + std::string(ex.what()));
  }

  ORT_RETURN_IF_ERROR(model->MainGraph().Resolve(true));

  return Status::OK();
}

template <typename T>
static Status LoadModel(const T& file_path, std::shared_ptr<Model>& p_model, const IOnnxRuntimeOpSchemaRegistryList* local_registries) {
  int fd;
  Status status = Env::Default().FileOpenRd(file_path, fd);
  if (!status.IsOK()) {
    if (status.Category() == common::SYSTEM) {
      switch (status.Code()) {
        case ENOENT:
          return ORT_MAKE_STATUS(ONNXRUNTIME, NO_SUCHFILE, "Load model failed. File doesn't exist");
        case EINVAL:
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT);
        default:
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "system error number ", status.Code());
      }
    }
  }
  try {
    status = Model::Load(fd, p_model, local_registries);
  } catch (std::exception& ex) {
    GSL_SUPPRESS(es .84)
    ORT_IGNORE_RETURN_VALUE(Env::Default().FileClose(fd));
    return Status(ONNXRUNTIME, FAIL, ex.what());
  }
  if (!status.IsOK()) {
    GSL_SUPPRESS(es .84)
    ORT_IGNORE_RETURN_VALUE(Env::Default().FileClose(fd));
    return status;
  }
  return Env::Default().FileClose(fd);
}

template <typename T>
static Status SaveModel(Model& model, const T& file_path) {
  int fd;
  Status status = Env::Default().FileOpenWr(file_path, fd);
  ORT_RETURN_IF_ERROR(status);
  try {
    status = Model::Save(model, fd);
  } catch (std::exception& ex) {
    GSL_SUPPRESS(es .84)
    ORT_IGNORE_RETURN_VALUE(Env::Default().FileClose(fd));
    return Status(ONNXRUNTIME, FAIL, ex.what());
  }
  if (!status.IsOK()) {
    GSL_SUPPRESS(es .84)
    ORT_IGNORE_RETURN_VALUE(Env::Default().FileClose(fd));
    return status;
  }
  return Env::Default().FileClose(fd);
}

#ifdef _WIN32
GSL_SUPPRESS(r .30)  // spurious warnings. p_model is potentially reset in the internal call to Load
GSL_SUPPRESS(r .35)
Status Model::Load(const std::wstring& file_path, std::shared_ptr<Model>& p_model, const IOnnxRuntimeOpSchemaRegistryList* local_registries) {
  return LoadModel(file_path, p_model, local_registries);
}

Status Model::Save(Model& model, const std::wstring& file_path) {
  return SaveModel(model, file_path);
}

#endif

GSL_SUPPRESS(r .30)  // spurious warnings. p_model is potentially reset in the internal call to Load
GSL_SUPPRESS(r .35)
Status Model::Load(const std::string& file_path, std::shared_ptr<Model>& p_model, const IOnnxRuntimeOpSchemaRegistryList* local_registries) {
  return LoadModel(file_path, p_model, local_registries);
}

Status Model::Save(Model& model, const std::string& file_path) {
  return SaveModel(model, file_path);
}

Status Model::LoadFromBytes(int count, void* p_bytes, /*out*/ std::shared_ptr<Model>& p_model, const IOnnxRuntimeOpSchemaRegistryList* local_registries) {
  std::unique_ptr<ModelProto> modelProto = std::make_unique<ModelProto>();
  const bool result = modelProto->ParseFromArray(p_bytes, count);
  if (!result) {
    return Status(ONNXRUNTIME, INVALID_PROTOBUF, "Protobuf parsing failed.");
  }

  p_model = std::make_shared<Model>(std::move(modelProto), local_registries);

  ORT_RETURN_IF_ERROR(p_model->MainGraph().Resolve(true));

  return Status::OK();
}

using ::google::protobuf::io::CodedInputStream;
using ::google::protobuf::io::FileInputStream;
using ::google::protobuf::io::ZeroCopyInputStream;

Status Model::Load(int fd, std::shared_ptr<Model>& p_model, const IOnnxRuntimeOpSchemaRegistryList* local_registries) {
  if (fd < 0) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "<p_fd> less than 0.");
  }

  auto raw_input = std::unique_ptr<ZeroCopyInputStream>(std::make_unique<FileInputStream>(fd));
  auto coded_input = std::make_unique<CodedInputStream>(raw_input.get());

  // Allows protobuf library versions < 3.2.0 to parse messages greater than 64MB.
  coded_input->SetTotalBytesLimit(INT_MAX, INT_MAX);

  std::unique_ptr<ModelProto> model_proto = std::make_unique<ModelProto>();
  const bool result = model_proto->ParseFromCodedStream(coded_input.get());
  coded_input.reset();
  raw_input.reset();

  if (!result) {
    return Status(ONNXRUNTIME, INVALID_PROTOBUF, "Protobuf parsing failed.");
  }

  p_model = std::make_shared<Model>(std::move(model_proto), local_registries);

  ORT_RETURN_IF_ERROR(p_model->MainGraph().Resolve(true));

  return Status::OK();
}

Status Model::Save(Model& model, int p_fd) {
  if (p_fd < 0) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "<p_fd> is less than 0.");
  }

  ORT_RETURN_IF_ERROR(model.MainGraph().Resolve());

  auto model_proto = model.ToProto();
  const bool result = model_proto.SerializeToFileDescriptor(p_fd);
  if (result) {
    return Status::OK();
  } else {
    return Status(ONNXRUNTIME, INVALID_PROTOBUF, "Protobuf serialization failed.");
  }
}
}  // namespace onnxruntime
