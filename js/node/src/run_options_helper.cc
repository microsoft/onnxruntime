// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_cxx_api.h"
#include <napi.h>

#include <cmath>

#include "common.h"
#include "run_options_helper.h"

void ParseRunOptions(const Napi::Object options, Ort::RunOptions &runOptions) {
  // Log severity level
  if (options.Has("logSeverityLevel")) {
    auto logLevelValue = options.Get("logSeverityLevel");
    ORT_NAPI_THROW_TYPEERROR_IF(!logLevelValue.IsNumber(), options.Env(),
                                "Invalid argument: runOptions.logSeverityLevel must be a number.");
    double logLevelNumber = logLevelValue.As<Napi::Number>().DoubleValue();
    ORT_NAPI_THROW_RANGEERROR_IF(
        std::floor(logLevelNumber) != logLevelNumber || logLevelNumber < 0 || logLevelNumber > 4, options.Env(),
        "Invalid argument: runOptions.logSeverityLevel must be one of the following: 0, 1, 2, 3, 4.");

    runOptions.SetRunLogSeverityLevel(static_cast<int>(logLevelNumber));
  }

  // Tag
  if (options.Has("tag")) {
    auto tagValue = options.Get("tag");
    ORT_NAPI_THROW_TYPEERROR_IF(!tagValue.IsString(), options.Env(),
                                "Invalid argument: runOptions.tag must be a string.");
    runOptions.SetRunTag(tagValue.As<Napi::String>().Utf8Value().c_str());
  }
}
