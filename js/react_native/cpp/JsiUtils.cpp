#include "JsiUtils.h"

using namespace facebook::jsi;

bool isTypedArray(Runtime &runtime, const Object &jsObj) {
  if (!jsObj.hasProperty(runtime, "buffer"))
    return false;
  if (!jsObj.getProperty(runtime, "buffer")
           .asObject(runtime)
           .isArrayBuffer(runtime))
    return false;
  return true;
}

void forEach(Runtime &runtime, const Object &object,
             const std::function<void(const std::string &, const Value &,
                                      size_t)> &callback) {
  auto names = object.getPropertyNames(runtime);
  for (size_t i = 0; i < names.size(runtime); i++) {
    auto key =
        names.getValueAtIndex(runtime, i).asString(runtime).utf8(runtime);
    auto value = object.getProperty(runtime, key.c_str());
    callback(key, value, i);
  }
}

void forEach(Runtime &runtime, const Array &array,
             const std::function<void(const Value &, size_t)> &callback) {
  for (size_t i = 0; i < array.size(runtime); i++) {
    callback(array.getValueAtIndex(runtime, i), i);
  }
}
