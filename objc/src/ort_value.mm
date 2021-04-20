// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "onnxruntime/ort_value.h"
#import "src/ort_value_internal.h"
#import "src/error_utils.h"

#include "core/common/optional.h"
#include "core/session/onnxruntime_cxx_api.h"

NS_ASSUME_NONNULL_BEGIN

static ONNXTensorElementDataType get_onnx_tensor_element_data_type(ORTTensorElementDataType value) {
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

@property NSMutableData* data;

@end

@implementation ORTValue {
  onnxruntime::optional<Ort::Value> _value;
}

- (nullable instancetype)initTensorWithData:(NSMutableData*)data
                                elementType:(ORTTensorElementDataType)type
                                      shape:(const int64_t*)shape
                                   shapeLen:(size_t)shape_len
                                      error:(NSError**)error {
  self = [super init];
  if (self) {
    try {
      _data = data;
      const auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
      const auto element_type = get_onnx_tensor_element_data_type(type);
      _value = Ort::Value::CreateTensor(memory_info, _data.mutableBytes, _data.length, shape, shape_len, element_type);
    } catch (const Ort::Exception& e) {
      [ORTErrorUtils saveErrorCode:e.GetOrtErrorCode()
                       description:e.what()
                           toError:error];
      self = nil;
    }
  }
  return self;
}

- (Ort::Value*)handle {
  return &(*_value);
}

@end

NS_ASSUME_NONNULL_END
