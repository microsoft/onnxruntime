// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <napi.h>
#include <vector>

#include "onnxruntime_cxx_api.h"

// convert a Javascript OnnxValue object to an OrtValue object
Ort::Value NapiValueToOrtValue(Napi::Env env, Napi::Value value, OrtMemoryInfo *memory_info);

// convert an OrtValue object to a Javascript OnnxValue object
Napi::Value OrtValueToNapiValue(Napi::Env env, Ort::Value &value);
