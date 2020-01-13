// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "pch.h"

#include "winml_adapter_operator_registry.h"

#include "winml_adapter_c_api.h"
#include "core/session/ort_apis.h"
#include "winml_adapter_apis.h"
#include "core/framework/error_code_helper.h"
#include "core/session/onnxruntime_env.h"

namespace winmla = Windows::AI::MachineLearning::Adapter;

class WinmlAdapterLoggingWrapper : public LoggingWrapper {
 public:
  WinmlAdapterLoggingWrapper(OrtLoggingFunction logging_function, OrtProfilingFunction profiling_function, void* logger_param) : LoggingWrapper(logging_function, logger_param),
                                                                                                                                 profiling_function_(profiling_function) {
    ;
  }

  void SendProfileEvent(onnxruntime::profiling::EventRecord& event_record) const override {
    if (profiling_function_) {
      OrtProfilerEventRecord ort_event_record = {};
      ort_event_record.category_ = static_cast<uint32_t>(event_record.cat);
      ort_event_record.category_name_ = onnxruntime::profiling::event_categor_names_[event_record.cat];
      ort_event_record.duration_ = event_record.dur;
      ort_event_record.event_name_ = event_record.name.c_str();
      ort_event_record.execution_provider_ = (event_record.cat == onnxruntime::profiling::EventCategory::NODE_EVENT) ? event_record.args["provider"].c_str() : nullptr;
      ort_event_record.op_name_ = (event_record.cat == onnxruntime::profiling::EventCategory::NODE_EVENT) ? event_record.args["op_name"].c_str() : nullptr;
      ort_event_record.process_id_ = event_record.pid;
      ort_event_record.thread_id_ = event_record.tid;
      ort_event_record.time_span_ = event_record.ts;

      profiling_function_(&ort_event_record);
    }
  }

 private:
  OrtProfilingFunction profiling_function_{};
};

ORT_API_STATUS_IMPL(winmla::EnvConfigureCustomLoggerAndProfiler, _In_ OrtEnv* env, OrtLoggingFunction logging_function, OrtProfilingFunction profiling_function,
                    _In_opt_ void* logger_param, OrtLoggingLevel default_warning_level,
                    _In_ const char* logid, _Outptr_ OrtEnv** out) {
  std::unique_ptr<onnxruntime::logging::LoggingManager> logging_manager;
  std::string name = logid;
  std::unique_ptr<onnxruntime::logging::ISink> logger = onnxruntime::make_unique<WinmlAdapterLoggingWrapper>(logging_function, profiling_function, logger_param);
    logging_manager.reset(new onnxruntime::logging::LoggingManager(std::move(logger),
                                  static_cast<onnxruntime::logging::Severity>(default_warning_level),
                                  false,
                                  onnxruntime::logging::LoggingManager::InstanceType::Default,
                                  &name));

  env->SetLoggingManager(std::move(logging_manager));
  return nullptr;
}