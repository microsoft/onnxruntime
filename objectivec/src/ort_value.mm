// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "src/ort_value_internal.h"

#include <algorithm>
#include <optional>

#include "safeint/SafeInt.hpp"

#include "core/session/onnxruntime_cxx_api.h"

#import "src/error_utils.h"

NS_ASSUME_NONNULL_BEGIN

namespace {
struct ValueTypeInfo {
  ORTValueType type;
  ONNXType capi_type;
};

// supported ORT value types
// define the mapping from ORTValueType to C API ONNXType here
constexpr ValueTypeInfo kValueTypeInfos[]{
    {ORTValueTypeUnknown, ONNX_TYPE_UNKNOWN},
    {ORTValueTypeTensor, ONNX_TYPE_TENSOR},
};

struct TensorElementTypeInfo {
  ORTTensorElementDataType type;
  ONNXTensorElementDataType capi_type;
  size_t element_size;
};

// supported ORT tensor element data types
// define the mapping from ORTTensorElementDataType to C API
// ONNXTensorElementDataType here
constexpr TensorElementTypeInfo kElementTypeInfos[]{
    {ORTTensorElementDataTypeUndefined, ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED, 0},
    {ORTTensorElementDataTypeFloat, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, sizeof(float)},
    {ORTTensorElementDataTypeInt32, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, sizeof(int32_t)},
};

ORTValueType CAPIToPublicValueType(ONNXType capi_type) {
  const auto it = std::find_if(
      std::begin(kValueTypeInfos), std::end(kValueTypeInfos),
      [capi_type](const auto& type_info) { return type_info.capi_type == capi_type; });
  if (it == std::end(kValueTypeInfos)) {
    throw Ort::Exception{"unsupported value type", ORT_NOT_IMPLEMENTED};
  }
  return it->type;
}

ONNXTensorElementDataType PublicToCAPITensorElementType(ORTTensorElementDataType type) {
  const auto it = std::find_if(
      std::begin(kElementTypeInfos), std::end(kElementTypeInfos),
      [type](const auto& type_info) { return type_info.type == type; });
  if (it == std::end(kElementTypeInfos)) {
    throw Ort::Exception{"unsupported tensor element type", ORT_NOT_IMPLEMENTED};
  }
  return it->capi_type;
}

ORTTensorElementDataType CAPIToPublicTensorElementType(ONNXTensorElementDataType capi_type) {
  const auto it = std::find_if(
      std::begin(kElementTypeInfos), std::end(kElementTypeInfos),
      [capi_type](const auto& type_info) { return type_info.capi_type == capi_type; });
  if (it == std::end(kElementTypeInfos)) {
    throw Ort::Exception{"unsupported tensor element type", ORT_NOT_IMPLEMENTED};
  }
  return it->type;
}

size_t SizeOfCAPITensorElementType(ONNXTensorElementDataType capi_type) {
  const auto it = std::find_if(
      std::begin(kElementTypeInfos), std::end(kElementTypeInfos),
      [capi_type](const auto& type_info) { return type_info.capi_type == capi_type; });
  if (it == std::end(kElementTypeInfos)) {
    throw Ort::Exception{"unsupported tensor element type", ORT_NOT_IMPLEMENTED};
  }
  return it->element_size;
}
}

@interface ORTValue ()

// pointer to any external tensor data to keep alive for the lifetime of the ORTValue
@property(nullable) NSMutableData* externalTensorData;

@end

@implementation ORTValue {
  std::optional<Ort::Value> _value;
  std::optional<Ort::TypeInfo> _typeInfo;
}

#pragma mark Public

- (nullable instancetype)initTensorWithData:(NSMutableData*)tensorData
                                elementType:(ORTTensorElementDataType)elementType
                                      shape:(NSArray<NSNumber*>*)shape
                                      error:(NSError**)error {
  try {
    const auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    const auto ONNXElementType = PublicToCAPITensorElementType(elementType);
    const auto shapeVector = [shape]() {
      std::vector<int64_t> result{};
      result.reserve(shape.count);
      for (NSNumber* dim in shape) {
        result.push_back(dim.longLongValue);
      }
      return result;
    }();
    Ort::Value ortValue = Ort::Value::CreateTensor(
        memoryInfo, tensorData.mutableBytes, tensorData.length,
        shapeVector.data(), shapeVector.size(), ONNXElementType);

    self = [self initWithCAPIOrtValue:ortValue.release()
                   externalTensorData:tensorData
                                error:error];
  } catch (const Ort::Exception& e) {
    ORTSaveExceptionToError(e, error);
    self = nil;
  }

  return self;
}

- (BOOL)valueType:(ORTValueType*)valueType
            error:(NSError**)error {
  try {
    const auto ortValueType = _typeInfo->GetONNXType();
    *valueType = CAPIToPublicValueType(ortValueType);
    return YES;
  } catch (const Ort::Exception& e) {
    ORTSaveExceptionToError(e, error);
    return NO;
  }
}

- (BOOL)tensorElementType:(ORTTensorElementDataType*)elementType
                    error:(NSError**)error {
  try {
    const auto tensorTypeAndShapeInfo = _typeInfo->GetTensorTypeAndShapeInfo();
    const auto ortElementType = tensorTypeAndShapeInfo.GetElementType();
    *elementType = CAPIToPublicTensorElementType(ortElementType);
    return YES;
  } catch (const Ort::Exception& e) {
    ORTSaveExceptionToError(e, error);
    return NO;
  }
}

- (nullable NSArray<NSNumber*>*)tensorShapeWithError:(NSError**)error {
  try {
    const auto tensorTypeAndShapeInfo = _typeInfo->GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> shape = tensorTypeAndShapeInfo.GetShape();
    NSMutableArray<NSNumber*>* shapeArray = [[NSMutableArray alloc] initWithCapacity:shape.size()];
    for (size_t i = 0; i < shape.size(); ++i) {
      shapeArray[i] = @(shape[i]);
    }
    return shapeArray;
  } catch (const Ort::Exception& e) {
    ORTSaveExceptionToError(e, error);
    return nil;
  }
}

- (nullable NSMutableData*)tensorDataWithError:(NSError**)error {
  try {
    const auto tensorTypeAndShapeInfo = _typeInfo->GetTensorTypeAndShapeInfo();
    const size_t elementCount = tensorTypeAndShapeInfo.GetElementCount();
    const size_t elementSize = SizeOfCAPITensorElementType(tensorTypeAndShapeInfo.GetElementType());
    size_t rawDataLength;
    if (!SafeMultiply(elementCount, elementSize, rawDataLength)) {
      throw Ort::Exception{"failed to compute tensor data length", ORT_RUNTIME_EXCEPTION};
    }
    void* rawData;
    Ort::ThrowOnError(Ort::GetApi().GetTensorMutableData(*_value, &rawData));
    return [NSMutableData dataWithBytesNoCopy:rawData
                                       length:rawDataLength
                                 freeWhenDone:NO];
  } catch (const Ort::Exception& e) {
    ORTSaveExceptionToError(e, error);
    return nil;
  }
}

#pragma mark Internal

- (nullable instancetype)initWithCAPIOrtValue:(OrtValue*)CAPIOrtValue
                           externalTensorData:(nullable NSMutableData*)externalTensorData
                                        error:(NSError**)error {
  self = [super init];
  if (self) {
    try {
      _value = Ort::Value{CAPIOrtValue};
      _typeInfo = _value->GetTypeInfo();
      _externalTensorData = externalTensorData;
    } catch (const Ort::Exception& e) {
      ORTSaveExceptionToError(e, error);
      self = nil;
    }
  }
  return self;
}

- (Ort::Value*)CXXAPIOrtValue {
  return &(*_value);
}

@end

NS_ASSUME_NONNULL_END
