#pragma once

#include <functional>
#include <jsi/jsi.h>
#include <memory>
#include <string>
#include <unordered_map>

#define BIND_METHOD(method)                                                    \
  std::bind(&method, std::placeholders::_1, std::placeholders::_2,             \
            std::placeholders::_3, std::placeholders::_4)

#define BIND_GETTER(method) std::bind(&method, std::placeholders::_1)

#define BIND_SETTER(method)                                                    \
  std::bind(&method, std::placeholders::_1, std::placeholders::_2)

#define BIND_THIS_METHOD(cls, name)                                            \
  std::bind(&cls::name##_method, this, std::placeholders::_1,                  \
            std::placeholders::_2, std::placeholders::_3,                      \
            std::placeholders::_4)

#define BIND_THIS_GETTER(cls, name)                                            \
  std::bind(&cls::name##_get, this, std::placeholders::_1)

#define BIND_THIS_SETTER(cls, name)                                            \
  std::bind(&cls::name##_set, this, std::placeholders::_1,                     \
            std::placeholders::_2)

#define METHOD_INFO(cls, name, count)                                          \
  {                                                                            \
    #name, { BIND_THIS_METHOD(cls, name), count }                              \
  }

#define GETTER_INFO(cls, name)                                                 \
  { #name, BIND_THIS_GETTER(cls, name) }

#define DEFINE_METHOD(name)                                                    \
  Value name##_method(Runtime &runtime, const Value &thisValue,                \
                      const Value *arguments, size_t count)

#define DEFINE_GETTER(name) Value name##_get(Runtime &runtime)

#define DEFINE_SETTER(name)                                                    \
  void name##_set(Runtime &runtime, const Value &value)

typedef std::function<facebook::jsi::Value(
    facebook::jsi::Runtime &runtime, const facebook::jsi::Value &thisValue,
    const facebook::jsi::Value *arguments, size_t count)>
    JsiMethod;
typedef std::function<facebook::jsi::Value(facebook::jsi::Runtime &runtime)>
    JsiGetter;
typedef std::function<void(facebook::jsi::Runtime &runtime,
                           const facebook::jsi::Value &value)>
    JsiSetter;

struct JsiMethodInfo {
  JsiMethod method;
  size_t count;
};

typedef std::unordered_map<std::string, JsiMethodInfo> JsiMethodMap;
typedef std::unordered_map<std::string, JsiGetter> JsiGetterMap;
typedef std::unordered_map<std::string, JsiSetter> JsiSetterMap;
