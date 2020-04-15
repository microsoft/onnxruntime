#ifndef ONNXRUNTIME_NODE_TENSOR_H
#define ONNXRUNTIME_NODE_TENSOR_H

#pragma once

#include <napi.h>
#include <vector>

#include <core/session/onnxruntime_cxx_api.h>

Ort::Value NapiValueToOrtValue(Napi::Env env, Napi::Value value);
Napi::Value OrtValueToNapiValue(Napi::Env env, Ort::Value &value);

#endif
