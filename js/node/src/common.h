// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <napi.h>

#include <sstream>
#include <vector>

inline void MakeStringInternal(std::ostringstream & /*ss*/) noexcept {}

template <typename T> inline void MakeStringInternal(std::ostringstream &ss, const T &t) noexcept { ss << t; }

template <typename T, typename... Args>
inline void MakeStringInternal(std::ostringstream &ss, const T &t, const Args &... args) noexcept {
  ::MakeStringInternal(ss, t);
  ::MakeStringInternal(ss, args...);
}

template <typename... Args> std::string MakeString(const Args &... args) {
  std::ostringstream ss;
  ::MakeStringInternal(ss, args...);
  return std::string(ss.str());
}

#define ORT_NAPI_THROW(ERROR, ENV, ...)                                                                                \
  do {                                                                                                                 \
    throw Napi::ERROR::New((ENV), MakeString(__VA_ARGS__));                                                            \
  } while (false)
#define ORT_NAPI_THROW_ERROR(ENV, ...) ORT_NAPI_THROW(Error, ENV, __VA_ARGS__)
#define ORT_NAPI_THROW_TYPEERROR(ENV, ...) ORT_NAPI_THROW(TypeError, ENV, __VA_ARGS__)
#define ORT_NAPI_THROW_RANGEERROR(ENV, ...) ORT_NAPI_THROW(RangeError, ENV, __VA_ARGS__)

#define ORT_NAPI_THROW_IF(COND, ERROR, ENV, ...)                                                                       \
  if (COND) {                                                                                                          \
    ORT_NAPI_THROW(ERROR, ENV, __VA_ARGS__);                                                                           \
  }
#define ORT_NAPI_THROW_ERROR_IF(COND, ENV, ...) ORT_NAPI_THROW_IF(COND, Error, ENV, __VA_ARGS__)
#define ORT_NAPI_THROW_TYPEERROR_IF(COND, ENV, ...) ORT_NAPI_THROW_IF(COND, TypeError, ENV, __VA_ARGS__)
#define ORT_NAPI_THROW_RANGEERROR_IF(COND, ENV, ...) ORT_NAPI_THROW_IF(COND, RangeError, ENV, __VA_ARGS__)

template <typename T> Napi::Value CreateNapiArrayFrom(napi_env env, const std::vector<T> &vec) {
  Napi::EscapableHandleScope scope(env);
  auto array = Napi::Array::New(env, vec.size());
  for (uint32_t i = 0; i < vec.size(); i++) {
    array.Set(i, Napi::Value::From(env, vec[i]));
  }
  return scope.Escape(array);
}
