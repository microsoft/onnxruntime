#pragma once

#include <jsi/jsi.h>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

namespace onnxruntimejsi {

class TensorUtils {
 public:
  static Ort::Value
  createOrtValueFromJSTensor(facebook::jsi::Runtime& runtime,
                             const facebook::jsi::Object& tensorObj,
                             const Ort::MemoryInfo& memoryInfo);

  static facebook::jsi::Object
  createJSTensorFromOrtValue(facebook::jsi::Runtime& runtime,
                             Ort::Value& ortValue,
                             const facebook::jsi::Object& tensorConstructor);

  static bool isTensor(facebook::jsi::Runtime& runtime,
                       const facebook::jsi::Object& obj);
};

}  // namespace onnxruntimejsi
