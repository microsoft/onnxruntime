#pragma once

#include <jsi/jsi.h>
#include <string>
#include <vector>

bool isTypedArray(facebook::jsi::Runtime& runtime,
                  const facebook::jsi::Object& jsObj);

void forEach(
    facebook::jsi::Runtime& runtime, const facebook::jsi::Object& object,
    const std::function<void(const std::string&, const facebook::jsi::Value&,
                             size_t)>& callback);

void forEach(
    facebook::jsi::Runtime& runtime, const facebook::jsi::Array& array,
    const std::function<void(const facebook::jsi::Value&, size_t)>& callback);
