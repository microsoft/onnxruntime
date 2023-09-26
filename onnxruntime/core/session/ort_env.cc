// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// this file contains implementations of the C API

#include <cassert>

#include "ort_env.h"
#include "core/session/ort_apis.h"
#include "core/session/environment.h"
#include "core/session/allocator_adapters.h"
#include "core/common/logging/logging.h"
#include "core/framework/provider_shutdown.h"
#include "core/platform/logging/make_platform_default_log_sink.h"

using namespace onnxruntime;
using namespace onnxruntime::logging;

std::vector< std::shared_ptr<OrtEnv> > OrtEnv::p_instances_;
onnxruntime::OrtMutex OrtEnv::m_;

LoggingWrapper::LoggingWrapper(OrtLoggingFunction logging_function, void* logger_param)
    : logging_function_(logging_function), logger_param_(logger_param) {
}

void LoggingWrapper::SendImpl(const onnxruntime::logging::Timestamp& /*timestamp*/, const std::string& logger_id,
                              const onnxruntime::logging::Capture& message) {
  std::string s = message.Location().ToString();
  logging_function_(logger_param_, static_cast<OrtLoggingLevel>(message.Severity()), message.Category(),
                    logger_id.c_str(), s.c_str(), message.Message().c_str());
}

OrtEnv::OrtEnv(std::unique_ptr<onnxruntime::Environment> value1)
    : value_(std::move(value1)) {
}

OrtEnv::~OrtEnv() {
// We don't support any shared providers in the minimal build yet
#if !defined(ORT_MINIMAL_BUILD)
  UnloadSharedProviders();
#endif
}

OrtEnv* OrtEnv::GetInstance(const OrtEnv::LoggingManagerConstructionInfo& lm_info,
                            onnxruntime::common::Status& status,
                            const OrtThreadingOptions* tp_options) {
  std::lock_guard<onnxruntime::OrtMutex> lock(m_);
  std::unique_ptr<LoggingManager> lmgr;
  std::string name = lm_info.logid;
  if (lm_info.logging_function) {
    std::unique_ptr<ISink> logger = std::make_unique<LoggingWrapper>(lm_info.logging_function,
                                                                     lm_info.logger_param);

    auto logInstanceType = p_instances_.empty() ? LoggingManager::InstanceType::Default : LoggingManager::InstanceType::Temporal;
    lmgr = std::make_unique<LoggingManager>(std::move(logger),
                                            static_cast<Severity>(lm_info.default_warning_level),
                                            false,
                                            logInstanceType,
                                            &name);
  } else {
    auto sink = MakePlatformDefaultLogSink();

    lmgr = std::make_unique<LoggingManager>(std::move(sink),
                                            static_cast<Severity>(lm_info.default_warning_level),
                                            false,
                                            LoggingManager::InstanceType::Default,
                                            &name);
  }
  std::unique_ptr<onnxruntime::Environment> env;
  if (!tp_options) {
    status = onnxruntime::Environment::Create(std::move(lmgr), env);
  } else {
    status = onnxruntime::Environment::Create(std::move(lmgr), env, tp_options, true);
  }
  if (!status.IsOK()) {
    return nullptr;
  }
  p_instances_.push_back(std::make_shared<OrtEnv>(std::move(env)));

  return p_instances_.back().get();
}

void OrtEnv::Release(OrtEnv* env_ptr) {
  if (!env_ptr) {
    return;
  }
  std::lock_guard<onnxruntime::OrtMutex> lock(m_);
  auto sptr = std::find_if(p_instances_.begin(), p_instances_.end(), [env_ptr](std::shared_ptr<OrtEnv> i) { return i.get() == env_ptr; });
  ORT_ENFORCE(sptr != p_instances_.end());  // sanity check

  // If we are about to delete the owner of the default logger and there is another env available
  // then move the default logger owner
  if ((p_instances_.size() > 1) && (*sptr)->GetLoggingManager()->IsDefaultOwner()) {
      p_instances_[1]->GetLoggingManager()->MakeDefaultOwner((*sptr)->GetLoggingManager());
  }
  p_instances_.erase(sptr);
}

onnxruntime::logging::LoggingManager* OrtEnv::GetLoggingManager() const {
  return value_->GetLoggingManager();
}

void OrtEnv::SetLoggingManager(std::unique_ptr<onnxruntime::logging::LoggingManager> logging_manager) {
  value_->SetLoggingManager(std::move(logging_manager));
}

onnxruntime::common::Status OrtEnv::RegisterAllocator(AllocatorPtr allocator) {
  auto status = value_->RegisterAllocator(allocator);
  return status;
}

onnxruntime::common::Status OrtEnv::CreateAndRegisterAllocator(const OrtMemoryInfo& mem_info,
                                                               const OrtArenaCfg* arena_cfg) {
  auto status = value_->CreateAndRegisterAllocator(mem_info, arena_cfg);
  return status;
}

onnxruntime::common::Status OrtEnv::UnregisterAllocator(const OrtMemoryInfo& mem_info) {
  return value_->UnregisterAllocator(mem_info);
}

onnxruntime::common::Status OrtEnv::CreateAndRegisterAllocatorV2(const std::string& provider_type, const OrtMemoryInfo& mem_info, const std::unordered_map<std::string, std::string>& options, const OrtArenaCfg* arena_cfg) {
  return value_->CreateAndRegisterAllocatorV2(provider_type, mem_info, options, arena_cfg);
}
