// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32
#pragma warning(disable : 4267)
#endif

#include "core/session/inference_session_impl.h"

ONNXTensorElementDataType MLDataTypeToOnnxRuntimeTensorElementDataType(const onnxruntime::DataTypeImpl* cpp_type);

ORT_API_STATUS_IMPL(OrtKernelInfoGetAttribute_float, _In_ OrtKernelInfo* info, _In_ const char* name, _Out_ float* out) {
  auto status = reinterpret_cast<onnxruntime::OpKernelInfo*>(info)->GetAttr<float>(name, out);
  if (status.IsOK())
    return nullptr;
  return onnxruntime::ToOrtStatus(status);
}

ORT_API_STATUS_IMPL(OrtKernelInfoGetAttribute_int64, _In_ OrtKernelInfo* info, _In_ const char* name, _Out_ int64_t* out) {
  auto status = reinterpret_cast<onnxruntime::OpKernelInfo*>(info)->GetAttr<int64_t>(name, out);
  if (status.IsOK())
    return nullptr;
  return onnxruntime::ToOrtStatus(status);
}

namespace onnxruntime {
//
// InferenceSession
//
InferenceSession::InferenceSession(const SessionOptions& session_options,
                                   logging::LoggingManager* logging_manager)
    : impl_(std::make_unique<Impl>(session_options, logging_manager)) {
}

InferenceSession::~InferenceSession() = default;

common::Status InferenceSession::Load(const std::string& model_uri) {
  return impl_->Load(model_uri);
}
#ifdef _WIN32
common::Status InferenceSession::Load(const std::wstring& model_uri) {
  return impl_->Load(model_uri);
}
#endif
common::Status InferenceSession::Load(std::istream& model_istream) {
  return impl_->Load(model_istream);
}

common::Status InferenceSession::Initialize() {
  return impl_->Initialize();
}

common::Status InferenceSession::Run(const RunOptions& run_options,
                                     const std::vector<std::string>& feed_names,
                                     const std::vector<MLValue>& feeds,
                                     const std::vector<std::string>& output_names,
                                     std::vector<MLValue>* p_fetches) {
  return impl_->Run(run_options, feed_names, feeds, output_names, p_fetches);
}

common::Status InferenceSession::Run(const NameMLValMap& feeds,
                                     const std::vector<std::string>& output_names,
                                     std::vector<MLValue>* p_fetches) {
  return Run({}, feeds, output_names, p_fetches);
}

common::Status InferenceSession::Run(const RunOptions& run_options,
                                     const NameMLValMap& feeds_map,
                                     const std::vector<std::string>& output_names,
                                     std::vector<MLValue>* p_fetches) {
  std::vector<std::string> feed_names;
  std::vector<MLValue> feeds;

  auto num_feeds = feeds_map.size();
  feed_names.reserve(num_feeds);
  feeds.reserve(num_feeds);

  for (auto& pair : feeds_map) {
    feed_names.push_back(pair.first);
    feeds.push_back(pair.second);
  }

  return Run(run_options, feed_names, feeds, output_names, p_fetches);
}

std::pair<common::Status, const ModelMetadata*> InferenceSession::GetModelMetadata() const {
  return impl_->GetModelMetadata();
}

std::pair<common::Status, const InputDefList*> InferenceSession::GetModelInputs() const {
  return impl_->GetModelInputs();
}

std::pair<common::Status, const OutputDefList*> InferenceSession::GetModelOutputs() const {
  return impl_->GetModelOutputs();
}

int InferenceSession::GetCurrentNumRuns() {
  return impl_->GetCurrentNumRuns();
}

void InferenceSession::StartProfiling(const std::string& file_prefix) {
  impl_->StartProfiling(file_prefix);
}

#ifdef _WIN32
void InferenceSession::StartProfiling(const std::wstring& file_prefix) { impl_->StartProfiling(file_prefix); }
#endif
void InferenceSession::StartProfiling(const logging::Logger* custom_logger) {
  impl_->StartProfiling(custom_logger);
}

std::string InferenceSession::EndProfiling() {
  return impl_->EndProfiling();
}

common::Status InferenceSession::RegisterExecutionProvider(std::unique_ptr<IExecutionProvider> p_exec_provider) {
  return impl_->RegisterExecutionProvider(std::move(p_exec_provider));
}

common::Status InferenceSession::RegisterGraphTransformer(std::unique_ptr<onnxruntime::GraphTransformer> p_graph_transformer) {
  return impl_->RegisterGraphTransformer(std::move(p_graph_transformer));
}

common::Status InferenceSession::RegisterCustomRegistry(std::shared_ptr<CustomRegistry> custom_registry) {
  return impl_->RegisterCustomRegistry(custom_registry);
}

common::Status InferenceSession::Load(const ModelProto& model_proto) {
  return impl_->Load(model_proto);
}

common::Status InferenceSession::Load(std::unique_ptr<ModelProto> p_model_proto) {
  return impl_->Load(std::move(p_model_proto));
}

common::Status InferenceSession::NewIOBinding(std::unique_ptr<IOBinding>* io_binding) {
  return impl_->NewIOBinding(io_binding);
}

common::Status InferenceSession::Run(const RunOptions& run_options, IOBinding& io_binding) {
  return impl_->Run(run_options, io_binding);
}

common::Status InferenceSession::Run(IOBinding& io_binding) {
  return impl_->Run(io_binding);
}

common::Status InferenceSession::LoadCustomOps(const std::vector<std::string>& dso_list) {
  return impl_->LoadCustomOps(dso_list);
}

common::Status InferenceSession::AddCustomOpDomains(const std::vector<OrtCustomOpDomain*>& ops) {
  return impl_->AddCustomOpDomains(ops);
}
}  // namespace onnxruntime
