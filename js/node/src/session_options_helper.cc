// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_cxx_api.h"
#include <napi.h>

#include <cmath>
#include <unordered_map>

#include "common.h"
#include "session_options_helper.h"
#ifdef USE_CUDA
#include "core/providers/cuda/cuda_provider_options.h"
#endif
#ifdef USE_DML
#include "core/providers/dml/dml_provider_factory.h"
#endif
#ifdef USE_TENSORRT
#include "core/providers/tensorrt/tensorrt_provider_options.h"
#endif
#ifdef USE_COREML
#include "core/providers/coreml/coreml_provider_factory.h"
#endif

const std::unordered_map<std::string, GraphOptimizationLevel> GRAPH_OPT_LEVEL_NAME_TO_ID_MAP = {
    {"disabled", ORT_DISABLE_ALL},
    {"basic", ORT_ENABLE_BASIC},
    {"extended", ORT_ENABLE_EXTENDED},
    {"all", ORT_ENABLE_ALL}};

const std::unordered_map<std::string, ExecutionMode> EXECUTION_MODE_NAME_TO_ID_MAP = {{"sequential", ORT_SEQUENTIAL},
                                                                                      {"parallel", ORT_PARALLEL}};

void ParseExecutionProviders(const Napi::Array epList, Ort::SessionOptions &sessionOptions) {
  for (uint32_t i = 0; i < epList.Length(); i++) {
    Napi::Value epValue = epList[i];
    std::string name;
    int deviceId = 0;
    int coreMlFlags = 0;
    if (epValue.IsString()) {
      name = epValue.As<Napi::String>().Utf8Value();
    } else if (!epValue.IsObject() || epValue.IsNull() || !epValue.As<Napi::Object>().Has("name") ||
               !epValue.As<Napi::Object>().Get("name").IsString()) {
      ORT_NAPI_THROW_TYPEERROR(epList.Env(), "Invalid argument: sessionOptions.executionProviders[", i,
                               "] must be either a string or an object with property 'name'.");
    } else {
      auto obj = epValue.As<Napi::Object>();
      name = obj.Get("name").As<Napi::String>().Utf8Value();
      if (obj.Has("deviceId")) {
        deviceId = obj.Get("deviceId").As<Napi::Number>();
      }
      if (obj.Has("coreMlFlags")) {
        coreMlFlags = obj.Get("coreMlFlags").As<Napi::Number>();
      }
    }

    // CPU execution provider
    if (name == "cpu") {
      // TODO: handling CPU EP options
#ifdef USE_CUDA
    } else if (name == "cuda") {
      OrtCUDAProviderOptionsV2 *options;
      Ort::GetApi().CreateCUDAProviderOptions(&options);
      options->device_id = deviceId;
      sessionOptions.AppendExecutionProvider_CUDA_V2(*options);
      Ort::GetApi().ReleaseCUDAProviderOptions(options);
#endif
#ifdef USE_TENSORRT
    } else if (name == "tensorrt") {
      OrtTensorRTProviderOptionsV2 *options;
      Ort::GetApi().CreateTensorRTProviderOptions(&options);
      options->device_id = deviceId;
      sessionOptions.AppendExecutionProvider_TensorRT_V2(*options);
      Ort::GetApi().ReleaseTensorRTProviderOptions(options);
#endif
#ifdef USE_DML
    } else if (name == "dml") {
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(sessionOptions, deviceId));
#endif
#ifdef USE_COREML
    } else if (name == "coreml") {
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(sessionOptions, coreMlFlags));
#endif
    } else {
      ORT_NAPI_THROW_ERROR(epList.Env(), "Invalid argument: sessionOptions.executionProviders[", i,
                           "] is unsupported: '", name, "'.");
    }
  }
}

void ParseSessionOptions(const Napi::Object options, Ort::SessionOptions &sessionOptions) {
  // Execution provider
  if (options.Has("executionProviders")) {
    auto epsValue = options.Get("executionProviders");
    ORT_NAPI_THROW_TYPEERROR_IF(!epsValue.IsArray(), options.Env(),
                                "Invalid argument: sessionOptions.executionProviders must be an array.");
    ParseExecutionProviders(epsValue.As<Napi::Array>(), sessionOptions);
  }

  // Intra threads number
  if (options.Has("intraOpNumThreads")) {
    auto numValue = options.Get("intraOpNumThreads");
    ORT_NAPI_THROW_TYPEERROR_IF(!numValue.IsNumber(), options.Env(),
                                "Invalid argument: sessionOptions.intraOpNumThreads must be a number.");
    double num = numValue.As<Napi::Number>().DoubleValue();
    ORT_NAPI_THROW_RANGEERROR_IF(std::floor(num) != num || num < 0 || num > 4294967295, options.Env(),
                                 "'intraOpNumThreads' is invalid: ", num);
    sessionOptions.SetIntraOpNumThreads(static_cast<int>(num));
  }

  // Inter threads number
  if (options.Has("interOpNumThreads")) {
    auto numValue = options.Get("interOpNumThreads");
    ORT_NAPI_THROW_TYPEERROR_IF(!numValue.IsNumber(), options.Env(),
                                "Invalid argument: sessionOptions.interOpNumThreads must be a number.");
    double num = numValue.As<Napi::Number>().DoubleValue();
    ORT_NAPI_THROW_RANGEERROR_IF(std::floor(num) != num || num < 0 || num > 4294967295, options.Env(),
                                 "'interOpNumThreads' is invalid: ", num);
    sessionOptions.SetInterOpNumThreads(static_cast<int>(num));
  }

  // Optimization level
  if (options.Has("graphOptimizationLevel")) {
    auto optLevelValue = options.Get("graphOptimizationLevel");
    ORT_NAPI_THROW_TYPEERROR_IF(!optLevelValue.IsString(), options.Env(),
                                "Invalid argument: sessionOptions.graphOptimizationLevel must be a string.");
    auto optLevelString = optLevelValue.As<Napi::String>().Utf8Value();
    auto v = GRAPH_OPT_LEVEL_NAME_TO_ID_MAP.find(optLevelString);
    ORT_NAPI_THROW_TYPEERROR_IF(v == GRAPH_OPT_LEVEL_NAME_TO_ID_MAP.end(), options.Env(),
                                "'graphOptimizationLevel' is not supported: ", optLevelString);
    sessionOptions.SetGraphOptimizationLevel(v->second);
  }

  // CPU memory arena
  if (options.Has("enableCpuMemArena")) {
    auto enableCpuMemArenaValue = options.Get("enableCpuMemArena");
    ORT_NAPI_THROW_TYPEERROR_IF(!enableCpuMemArenaValue.IsBoolean(), options.Env(),
                                "Invalid argument: sessionOptions.enableCpuMemArena must be a boolean value.");
    if (enableCpuMemArenaValue.As<Napi::Boolean>().Value()) {
      sessionOptions.EnableCpuMemArena();
    } else {
      sessionOptions.DisableCpuMemArena();
    }
  }

  // memory pattern
  if (options.Has("enableMemPattern")) {
    auto enableMemPatternValue = options.Get("enableMemPattern");
    ORT_NAPI_THROW_TYPEERROR_IF(!enableMemPatternValue.IsBoolean(), options.Env(),
                                "Invalid argument: sessionOptions.enableMemPattern must be a boolean value.");
    if (enableMemPatternValue.As<Napi::Boolean>().Value()) {
      sessionOptions.EnableMemPattern();
    } else {
      sessionOptions.DisableMemPattern();
    }
  }

  // execution mode
  if (options.Has("executionMode")) {
    auto executionModeValue = options.Get("executionMode");
    ORT_NAPI_THROW_TYPEERROR_IF(!executionModeValue.IsString(), options.Env(),
                                "Invalid argument: sessionOptions.executionMode must be a string.");
    auto executionModeString = executionModeValue.As<Napi::String>().Utf8Value();
    auto v = EXECUTION_MODE_NAME_TO_ID_MAP.find(executionModeString);
    ORT_NAPI_THROW_TYPEERROR_IF(v == EXECUTION_MODE_NAME_TO_ID_MAP.end(), options.Env(),
                                "'executionMode' is not supported: ", executionModeString);
    sessionOptions.SetExecutionMode(v->second);
  }

  // log ID
  if (options.Has("logId")) {
    auto logIdValue = options.Get("logId");
    ORT_NAPI_THROW_TYPEERROR_IF(!logIdValue.IsString(), options.Env(),
                                "Invalid argument: sessionOptions.logId must be a string.");
    auto logIdString = logIdValue.As<Napi::String>().Utf8Value();
    sessionOptions.SetLogId(logIdString.c_str());
  }

  // Log severity level
  if (options.Has("logSeverityLevel")) {
    auto logLevelValue = options.Get("logSeverityLevel");
    ORT_NAPI_THROW_TYPEERROR_IF(!logLevelValue.IsNumber(), options.Env(),
                                "Invalid argument: sessionOptions.logSeverityLevel must be a number.");
    double logLevelNumber = logLevelValue.As<Napi::Number>().DoubleValue();
    ORT_NAPI_THROW_RANGEERROR_IF(
        std::floor(logLevelNumber) != logLevelNumber || logLevelNumber < 0 || logLevelNumber > 4, options.Env(),
        "Invalid argument: sessionOptions.logSeverityLevel must be one of the following: 0, 1, 2, 3, 4.");

    sessionOptions.SetLogSeverityLevel(static_cast<int>(logLevelNumber));
  }
}
