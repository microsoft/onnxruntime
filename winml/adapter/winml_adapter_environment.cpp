// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "pch.h"

#include "winml_adapter_c_api.h"
#include "core/session/ort_apis.h"
#include "winml_adapter_apis.h"
#include "core/framework/error_code_helper.h"
#include "core/session/ort_env.h"

#ifdef USE_DML
#include "abi_custom_registry_impl.h"
#include "core/providers/dml/DmlExecutionProvider/inc/DmlExecutionProvider.h"
#include "core/providers/dml/OperatorAuthorHelper/SchemaInferenceOverrider.h"

#endif USE_DML
namespace winmla = Windows::AI::MachineLearning::Adapter;

class WinmlAdapterLoggingWrapper : public LoggingWrapper {
 public:
  WinmlAdapterLoggingWrapper(OrtLoggingFunction logging_function, OrtProfilingFunction profiling_function, void* logger_param) : LoggingWrapper(logging_function, logger_param),
                                                                                                                                 profiling_function_(profiling_function) {
  }

  void SendProfileEvent(onnxruntime::profiling::EventRecord& event_record) const override {
    if (profiling_function_) {
      OrtProfilerEventRecord ort_event_record = {};
      ort_event_record.category_ = static_cast<OrtProfilerEventCategory>(event_record.cat);
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
  API_IMPL_BEGIN
  std::string name = logid;
  std::unique_ptr<onnxruntime::logging::ISink> logger = std::make_unique<WinmlAdapterLoggingWrapper>(logging_function, profiling_function, logger_param);

  // Clear the logging manager, since only one default instance of logging manager can exist at a time.
  env->SetLoggingManager(nullptr);

  auto winml_logging_manager = std::make_unique<onnxruntime::logging::LoggingManager>(std::move(logger),
                                                                                      static_cast<onnxruntime::logging::Severity>(default_warning_level),
                                                                                      false,
                                                                                      onnxruntime::logging::LoggingManager::InstanceType::Default,
                                                                                      &name);

  // Set a new default logging manager
  env->SetLoggingManager(std::move(winml_logging_manager));
  return nullptr;
  API_IMPL_END
}

// Override select shape inference functions which are incomplete in ONNX with versions that are complete,
// and are also used in DML kernel registrations.  Doing this avoids kernel and shader creation being
// deferred until first evaluation.  It also prevents a situation where inference functions in externally
// registered schema are reachable only after upstream schema have been revised in a later OS release,
// which would be a compatibility risk.
ORT_API_STATUS_IMPL(winmla::OverrideSchema) {
  API_IMPL_BEGIN
#ifdef USE_DML
  static std::once_flag schema_override_once_flag;
  std::call_once(schema_override_once_flag, []() {
    SchemaInferenceOverrider::OverrideSchemaInferenceFunctions();
  });
#endif USE_DML.
  return nullptr;
  API_IMPL_END
}