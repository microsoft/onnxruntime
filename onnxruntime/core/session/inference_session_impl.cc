// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32
#pragma warning(disable : 4267)
#endif

#include "core/session/inference_session_impl.h"
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

common::Status InferenceSession::Run(const NameMLValMap& feeds,
                                     const std::vector<std::string>& output_names,
                                     std::vector<MLValue>* p_fetches) {
  return impl_->Run(feeds, output_names, p_fetches);
}

common::Status InferenceSession::Run(const RunOptions& run_options,
                                     const NameMLValMap& feeds,
                                     const std::vector<std::string>& output_names,
                                     std::vector<MLValue>* p_fetches) {
  return impl_->Run(run_options, feeds, output_names, p_fetches);
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
}  // namespace onnxruntime
