#include <memory>
#include <sstream>
#include <unordered_map>

#include "napi_utils.h"
#include "tensor_helper.h"

// make sure consistent with origin definition
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED == 0, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT == 1, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 == 2, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 == 3, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 == 4, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 == 5, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 == 6, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 == 7, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING == 8, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL == 9, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 == 10, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE == 11, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 == 12, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 == 13, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64 == 14, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128 == 15, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 == 16, "definition not consistent with OnnxRuntime");
constexpr size_t ONNX_TENSOR_ELEMENT_DATA_TYPE_COUNT = 17;

// size of element in bytes for each data type. 0 indicates not supported.
constexpr size_t DATA_TYPE_ELEMENT_SIZE_MAP[] = {
    0, // ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED     not supported
    4, // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
    1, // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
    1, // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
    2, // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16
    2, // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
    4, // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
    8, // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64         INT64 not working in Javascript
    0, // ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING        N/A
    1, // ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
    0, // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16       FLOAT16 not working in Javascript
    8, // ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE
    4, // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32
    8, // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64        UINT64 not working in Javascript
    0, // ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64     not supported
    0, // ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128    not supported
    0  // ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16      not supported
};
static_assert(sizeof(DATA_TYPE_ELEMENT_SIZE_MAP) == sizeof(size_t) * ONNX_TENSOR_ELEMENT_DATA_TYPE_COUNT,
              "definition not matching");

constexpr napi_typedarray_type DATA_TYPE_TYPEDARRAY_MAP[] = {
    (napi_typedarray_type)(-1), // ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED     not supported
    napi_float32_array,         // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
    napi_uint8_array,           // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
    napi_int8_array,            // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
    napi_uint16_array,          // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16
    napi_int16_array,           // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
    napi_int32_array,           // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
    napi_bigint64_array,        // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64         INT64 not working i
    (napi_typedarray_type)(-1), // ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING        not supported
    napi_uint8_array,           // ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
    (napi_typedarray_type)(-1), // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16       FLOAT16 not working
    napi_float64_array,         // ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE
    napi_uint32_array,          // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32
    napi_biguint64_array,       // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64        UINT64 not working
    (napi_typedarray_type)(-1), // ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64     not supported
    (napi_typedarray_type)(-1), // ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128    not supported
    (napi_typedarray_type)(-1)  // ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16      not supported
};
static_assert(sizeof(DATA_TYPE_TYPEDARRAY_MAP) == sizeof(napi_typedarray_type) * ONNX_TENSOR_ELEMENT_DATA_TYPE_COUNT,
              "definition not matching");

constexpr const char *DATA_TYPE_ID_TO_NAME_MAP[] = {
    nullptr,   // ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED
    "float32", // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
    "uint8",   // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
    "int8",    // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
    "uint16",  // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16
    "int16",   // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
    "int32",   // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
    "int64",   // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
    "string",  // ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING
    "bool",    // ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
    "float16", // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
    "float64", // ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE
    "uint32",  // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32
    "uint64",  // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64
    nullptr,   // ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64
    nullptr,   // ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128
    nullptr    // ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16
};
static_assert(sizeof(DATA_TYPE_ID_TO_NAME_MAP) == sizeof(const char *) * ONNX_TENSOR_ELEMENT_DATA_TYPE_COUNT,
              "definition not matching");

const std::unordered_map<std::string, ONNXTensorElementDataType> DATA_TYPE_NAME_TO_ID_MAP = {
    {"float32", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT},  {"uint8", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8},
    {"int8", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8},      {"uint16", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16},
    {"int16", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16},    {"int32", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32},
    {"int64", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64},    {"string", ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING},
    {"bool", ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL},      {"float16", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16},
    {"float64", ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE}, {"uint32", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32},
    {"uint64", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64}};

// currently only support tensor
Ort::Value NapiValueToOrtValue(Napi::Env env, Napi::Value value) {
  if (!value.IsObject()) {
    throw Napi::TypeError::New(env, "Tensor must be an object");
  }

  // check 'dims'
  auto tensorObject = value.As<Napi::Object>();
  auto dimsValue = tensorObject.Get("dims");
  if (!dimsValue.IsArray()) {
    throw Napi::TypeError::New(env, "Tensor.dims must be an array");
  }

  auto dimsArray = dimsValue.As<Napi::Array>();
  auto len = dimsArray.Length();
  std::vector<int64_t> dims;
  if (len > 0) {
    dims.reserve(len);
    for (uint32_t i = 0; i < len; i++) {
      Napi::Value dimValue = dimsArray[i];
      if (!dimValue.IsNumber()) {
        throw Napi::TypeError::New(env, "Tensor.dims must be an array of numbers");
      }
      auto dimNumber = dimValue.As<Napi::Number>();
      double dimDouble = dimNumber.DoubleValue();
      if (floor(dimDouble) != dimDouble || dimDouble < 0 || dimDouble > 4294967295) {
        throw Napi::TypeError::New(env, "Tensor.dims contains invalid dimension");
      }
      int64_t dim = static_cast<int64_t>(dimDouble);
      dims.push_back(dim);
    }
  }

  // check 'data' and 'type'
  auto tensorDataValue = tensorObject.Get("data");
  auto tensorTypeValue = tensorObject.Get("type");
  if (!tensorTypeValue.IsString()) {
    throw Napi::TypeError::New(env, "tensor.type must be a string");
  }
  auto tensorTypeString = tensorTypeValue.As<Napi::String>().Utf8Value();

  if (tensorTypeString == "string") {
    if (!tensorDataValue.IsArray()) {
      throw Napi::TypeError::New(env, "Tensor.data must be an array for string tensor");
    }
    auto tensorDataArray = tensorDataValue.As<Napi::Array>();
    auto tensorDataSize = tensorDataArray.Length();
    std::vector<std::string> stringData;
    std::vector<const char *> stringDataCStr;
    stringData.reserve(tensorDataSize);
    stringDataCStr.reserve(tensorDataSize);
    for (uint32_t i = 0; i < tensorDataSize; i++) {
      auto currentData = tensorDataArray.Get(i);
      if (!currentData.IsString()) {
        throw Napi::TypeError::New(env, "Tensor.data must be an array for string tensor");
      }
      auto currentString = currentData.As<Napi::String>();
      stringData.emplace_back(currentString.Utf8Value());
      stringDataCStr.emplace_back(stringData[i].c_str());
    }

    Ort::AllocatorWithDefaultOptions allocator;
    auto tensor = Ort::Value::CreateTensor(allocator, dims.empty() ? nullptr : &dims[0], dims.size(),
                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
    if (stringDataCStr.size() > 0) {
      Ort::ThrowOnError(Ort::GetApi().FillStringTensor(tensor, &stringDataCStr[0], stringDataCStr.size()));
    }
    return tensor;
  } else {
    // lookup numeric tensor types
    auto &v = DATA_TYPE_NAME_TO_ID_MAP.find(tensorTypeString);
    if (v == DATA_TYPE_NAME_TO_ID_MAP.end()) {
      throw Napi::TypeError::New(env, "Tensor.type is not supported");
    }
    ONNXTensorElementDataType elemType = v->second;

    if (!tensorDataValue.IsTypedArray()) {
      throw Napi::TypeError::New(env, "Tensor.data must be a typed array for numeric tensor");
    }
    auto tensorDataTypedArray = tensorDataValue.As<Napi::TypedArray>();
    auto typedArrayType = tensorDataValue.As<Napi::TypedArray>().TypedArrayType();
    if (DATA_TYPE_TYPEDARRAY_MAP[elemType] != typedArrayType) {
      throw Napi::TypeError::New(env, "Tensor.data must be a typed array for numeric tensor");
    }

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    char *buffer = reinterpret_cast<char *>(tensorDataTypedArray.ArrayBuffer().Data());
    size_t bufferByteOffset = tensorDataTypedArray.ByteOffset();
    // there is a bug in TypedArray::ElementSize(): https://github.com/nodejs/node-addon-api/pull/705
    // TODO: change to TypedArray::ByteLength() in next node-addon-api release.
    size_t bufferByteLength = tensorDataTypedArray.ElementLength() * DATA_TYPE_ELEMENT_SIZE_MAP[elemType];
    return Ort::Value::CreateTensor(memory_info, buffer + bufferByteOffset, bufferByteLength,
                                    dims.empty() ? nullptr : &dims[0], dims.size(), elemType);
  }
}

Napi::Value OrtValueToNapiValue(Napi::Env env, Ort::Value &value) {
  Napi::EscapableHandleScope scope(env);
  auto returnValue = Napi::Object::New(env);

  auto typeInfo = value.GetTypeInfo();
  auto onnxType = typeInfo.GetONNXType();

  if (onnxType != ONNX_TYPE_TENSOR) {
    throw Napi::Error::New(env, "non tensor type is temporarily not supported");
  }

  auto tensorTypeAndShapeInfo = typeInfo.GetTensorTypeAndShapeInfo();
  auto elemType = tensorTypeAndShapeInfo.GetElementType();

  // type
  auto typeCstr = DATA_TYPE_ID_TO_NAME_MAP[elemType];
  if (typeCstr == nullptr) {
    throw Napi::Error::New(env, "tensor type is not supported");
  }
  returnValue.DefineProperty(Napi::PropertyDescriptor::Value("type", Napi::String::New(env, typeCstr)));

  // dims
  size_t dimsCount = tensorTypeAndShapeInfo.GetDimensionsCount();
  std::vector<int64_t> dims;
  if (dimsCount > 0) {
    dims.resize(dimsCount);
    tensorTypeAndShapeInfo.GetDimensions(&dims[0], dimsCount);
  }
  auto dimsArray = Napi::Array::New(env, dimsCount);
  for (uint32_t i = 0; i < dimsCount; i++) {
    dimsArray[i] = dims[i];
  }
  returnValue.DefineProperty(Napi::PropertyDescriptor::Value("dims", dimsArray));

  // size
  auto size = tensorTypeAndShapeInfo.GetElementCount();
  returnValue.DefineProperty(Napi::PropertyDescriptor::Value("size", Napi::Number::From(env, size)));

  // data
  // TODO: optimize memory
  if (elemType == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
    // string data
    auto stringArray = Napi::Array::New(env, size);
    if (size > 0) {
      auto tempBufferLength = value.GetStringTensorDataLength();
      auto tempBuffer = std::make_unique<char[]>(tempBufferLength);
      std::vector<size_t> tempOffsets;
      tempOffsets.resize(size);
      value.GetStringTensorContent(tempBuffer.get(), tempBufferLength, &tempOffsets[0], size);

      for (uint32_t i = 0; i < size; i++) {
        stringArray[i] =
            Napi::String::New(env, tempBuffer.get() + tempOffsets[i],
                              i == size - 1 ? tempBufferLength - tempOffsets[i] : tempOffsets[i + 1] - tempOffsets[i]);
      }
    }
    returnValue.DefineProperty(Napi::PropertyDescriptor::Value("data", Napi::Value(env, stringArray)));
  } else {
    // number data
    auto arrayBuffer = Napi::ArrayBuffer::New(env, size * DATA_TYPE_ELEMENT_SIZE_MAP[elemType]);
    if (size > 0) {
      memcpy(arrayBuffer.Data(), value.GetTensorMutableData<void>(), size * DATA_TYPE_ELEMENT_SIZE_MAP[elemType]);
    }
    napi_value typedArrayData;
    napi_status status =
        napi_create_typedarray(env, DATA_TYPE_TYPEDARRAY_MAP[elemType], size, arrayBuffer, 0, &typedArrayData);
    NAPI_THROW_IF_FAILED(env, status, Napi::Value);
    returnValue.DefineProperty(Napi::PropertyDescriptor::Value("data", Napi::Value(env, typedArrayData)));
  }

  return scope.Escape(returnValue);
}
