#ifndef ONNXRUNTIME_NODE_TENSOR_H
#define ONNXRUNTIME_NODE_TENSOR_H

#pragma once

#include <napi.h>
#include <vector>

#include "onnxruntime_cxx_api.h"

// convert a Javascript OnnxValue object to an OrtValue object
Ort::Value NapiValueToOrtValue(Napi::Env env, Napi::Value value);

// convert an OrtValue object to a Javascript OnnxValue object
Napi::Value OrtValueToNapiValue(Napi::Env env, Ort::Value &value);

#endif
