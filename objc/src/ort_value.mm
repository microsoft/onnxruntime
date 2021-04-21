// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "src/ort_value_internal.h"

#include <optional>

#include "core/session/onnxruntime_cxx_api.h"

#import "src/error_utils.h"

NS_ASSUME_NONNULL_BEGIN

static ONNXTensorElementDataType GetONNXTensorElementDataType(ORTTensorElementDataType value) {
  switch (value) {
    case ORTTensorElementDataTypeFloat:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case ORTTensorElementDataTypeInt32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case ORTTensorElementDataTypeUndefined:
    default:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
}

@interface ORTValue ()

@property(nullable) NSMutableData* data;

@end

@implementation ORTValue {
  std::optional<Ort::Value> _value;
}

- (nullable instancetype)initTensorWithData:(NSMutableData*)data
                                elementType:(ORTTensorElementDataType)elementType
                                      shape:(const int64_t*)shape
                                   shapeLen:(size_t)shapeLen
                                      error:(NSError**)error {
  self = [super init];
  if (self) {
    try {
      _data = data;
      const auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
      const auto ONNXElementType = GetONNXTensorElementDataType(elementType);
      _value = Ort::Value::CreateTensor(memoryInfo, _data.mutableBytes, _data.length, shape, shapeLen, ONNXElementType);
    } catch (const Ort::Exception& e) {
      [ORTErrorUtils saveErrorCode:e.GetOrtErrorCode()
                       description:e.what()
                           toError:error];
      self = nil;
    }
  }
  return self;
}

- (Ort::Value*)internalORTValue {
  return &(*_value);
}

@end

NS_ASSUME_NONNULL_END
