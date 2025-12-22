#include "TensorUtils.h"
#include "JsiUtils.h"
#include <cstring>
#include <stdexcept>
#include <unordered_map>

using namespace facebook::jsi;

namespace onnxruntimejsi {

static const std::unordered_map<ONNXTensorElementDataType, const char*>
    dataTypeToStringMap = {
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "float32"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, "uint8"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, "int8"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, "uint16"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, "int16"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, "int32"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, "int64"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, "string"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, "bool"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, "float16"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, "float64"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32, "uint32"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64, "uint64"},
};

static const std::unordered_map<ONNXTensorElementDataType, size_t>
    elementSizeMap = {
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, sizeof(float)},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, sizeof(uint8_t)},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, sizeof(int8_t)},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, sizeof(uint16_t)},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, sizeof(int16_t)},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, sizeof(int32_t)},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, sizeof(int64_t)},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, sizeof(char*)},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, sizeof(bool)},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, 2},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, sizeof(double)},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32, sizeof(uint32_t)},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64, sizeof(uint64_t)},
};

static const std::unordered_map<ONNXTensorElementDataType, const char*>
    dataTypeToTypedArrayMap = {
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "Float32Array"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, "Float64Array"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, "Int32Array"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, "BigInt64Array"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32, "Uint32Array"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64, "BigUint64Array"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, "Uint8Array"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, "Int8Array"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, "Uint16Array"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, "Int16Array"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, "Float16Array"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, "Array"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, "Uint8Array"},
};

inline size_t getElementSize(ONNXTensorElementDataType dataType) {
  auto it = elementSizeMap.find(dataType);
  if (it != elementSizeMap.end()) {
    return it->second;
  }
  throw std::invalid_argument("Unsupported or unknown tensor data type: " +
                              std::to_string(static_cast<int>(dataType)));
}

bool TensorUtils::isTensor(Runtime& runtime, const Object& obj) {
  return obj.hasProperty(runtime, "cpuData") &&
         obj.hasProperty(runtime, "dims") && obj.hasProperty(runtime, "type");
}

inline Object getTypedArrayConstructor(Runtime& runtime,
                                       const ONNXTensorElementDataType type) {
  auto it = dataTypeToTypedArrayMap.find(type);
  if (it != dataTypeToTypedArrayMap.end()) {
    auto prop = runtime.global().getProperty(runtime, it->second);
    if (prop.isObject()) {
      return prop.asObject(runtime);
    } else {
      throw JSError(runtime, "TypedArray constructor not found: " +
                                 std::string(it->second));
    }
  }
  throw JSError(runtime,
                "Unsupported tensor data type for TypedArray creation: " +
                    std::to_string(static_cast<int>(type)));
}

size_t getElementCount(const std::vector<int64_t>& shape) {
  size_t count = 1;
  for (auto dim : shape) {
    count *= dim;
  }
  return count;
}

Ort::Value
TensorUtils::createOrtValueFromJSTensor(Runtime& runtime,
                                        const Object& tensorObj,
                                        const Ort::MemoryInfo& memoryInfo) {
  if (!isTensor(runtime, tensorObj)) {
    throw JSError(
        runtime,
        "Invalid tensor object: missing cpuData, dims, or type properties");
  }

  auto dataProperty = tensorObj.getProperty(runtime, "cpuData");
  auto dimsProperty = tensorObj.getProperty(runtime, "dims");
  auto typeProperty = tensorObj.getProperty(runtime, "type");

  if (!dimsProperty.isObject() ||
      !dimsProperty.asObject(runtime).isArray(runtime)) {
    throw JSError(runtime, "Tensor dims must be array");
  }

  if (!typeProperty.isString()) {
    throw JSError(runtime, "Tensor type must be string");
  }

  ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  auto typeStr = typeProperty.asString(runtime).utf8(runtime);
  for (auto it = dataTypeToStringMap.begin(); it != dataTypeToStringMap.end();
       ++it) {
    if (it->second == typeStr) {
      type = it->first;
      break;
    }
  }
  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
    throw JSError(runtime, "Unsupported tensor data type: " + typeStr);
  }

  void* data = nullptr;
  auto dataObj = dataProperty.asObject(runtime);

  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
    if (!dataObj.isArray(runtime)) {
      throw JSError(runtime, "Tensor data must be an array of strings");
    }
    auto array = dataObj.asArray(runtime);
    auto size = array.size(runtime);
    data = new char*[size];
    for (size_t i = 0; i < size; ++i) {
      auto item = array.getValueAtIndex(runtime, i);
      static_cast<char**>(data)[i] =
          strdup(item.toString(runtime).utf8(runtime).c_str());
    }
  } else {
    if (!isTypedArray(runtime, dataObj)) {
      throw JSError(runtime, "Tensor data must be a TypedArray");
    }
    auto buffer = dataObj.getProperty(runtime, "buffer")
                      .asObject(runtime)
                      .getArrayBuffer(runtime);
    data = buffer.data(runtime);
  }

  std::vector<int64_t> shape;
  auto dimsArray = dimsProperty.asObject(runtime).asArray(runtime);
  for (size_t i = 0; i < dimsArray.size(runtime); ++i) {
    auto dim = dimsArray.getValueAtIndex(runtime, i);
    if (dim.isNumber()) {
      shape.push_back(static_cast<int64_t>(dim.asNumber()));
    }
  }

  return Ort::Value::CreateTensor(memoryInfo, data,
                                  getElementCount(shape) * getElementSize(type),
                                  shape.data(), shape.size(), type);
}

Object
TensorUtils::createJSTensorFromOrtValue(Runtime& runtime, Ort::Value& ortValue,
                                        const Object& tensorConstructor) {
  auto typeInfo = ortValue.GetTensorTypeAndShapeInfo();
  auto shape = typeInfo.GetShape();
  auto elementType = typeInfo.GetElementType();

  std::string typeStr;
  auto it = dataTypeToStringMap.find(elementType);
  if (it != dataTypeToStringMap.end()) {
    typeStr = it->second;
  } else {
    throw JSError(runtime,
                  "Unsupported tensor data type for TypedArray creation: " +
                      std::to_string(static_cast<int>(elementType)));
  }

  auto dimsArray = Array(runtime, shape.size());
  for (size_t j = 0; j < shape.size(); ++j) {
    dimsArray.setValueAtIndex(runtime, j, Value(static_cast<double>(shape[j])));
  }

  if (elementType != ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
    void* rawData = ortValue.GetTensorMutableRawData();
    size_t elementCount =
        ortValue.GetTensorTypeAndShapeInfo().GetElementCount();
    size_t elementSize = getElementSize(elementType);
    size_t dataSize = elementCount * elementSize;

    auto typedArrayCtor = getTypedArrayConstructor(runtime, elementType);
    auto typedArrayInstance =
        typedArrayCtor.asFunction(runtime).callAsConstructor(
            runtime, static_cast<double>(elementCount));

    auto buffer = typedArrayInstance.asObject(runtime)
                      .getProperty(runtime, "buffer")
                      .asObject(runtime)
                      .getArrayBuffer(runtime);
    memcpy(buffer.data(runtime), rawData, dataSize);

    auto tensorInstance =
        tensorConstructor.asFunction(runtime).callAsConstructor(
            runtime, typeStr, typedArrayInstance, dimsArray);

    return tensorInstance.asObject(runtime);
  } else {
    auto strArray = Array(runtime, shape.size());
    for (size_t j = 0; j < shape.size(); ++j) {
      strArray.setValueAtIndex(
          runtime, j, Value(runtime, String::createFromUtf8(runtime, "")));
    }

    auto tensorInstance =
        tensorConstructor.asFunction(runtime).callAsConstructor(
            runtime, typeStr, strArray, dimsArray);

    return tensorInstance.asObject(runtime);
  }
}

}  // namespace onnxruntimejsi
