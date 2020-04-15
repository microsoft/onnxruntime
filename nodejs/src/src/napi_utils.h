#ifndef ONNXRUNTIME_NODE_UTILS_H
#define ONNXRUNTIME_NODE_UTILS_H

#pragma once

#include <memory>
#include <sstream>
#include <stdexcept>

#include <napi.h>
#include <vector>

template <typename T> Napi::Value CreateNapiArrayFrom(napi_env env, const std::vector<T> &vec) {
  Napi::EscapableHandleScope scope(env);
  auto array = Napi::Array::New(env, vec.size());
  for (uint32_t i = 0; i < vec.size(); i++) {
    array.Set(i, Napi::Value::From(env, vec[i]));
  }
  return scope.Escape(array);
}

#endif