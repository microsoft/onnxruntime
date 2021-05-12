// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "src/ort_value_internal.h"

#include <optional>

#include "safeint/SafeInt.hpp"

#include "core/session/onnxruntime_cxx_api.h"

#import "src/error_utils.h"
#import "src/ort_enums_internal.h"

NS_ASSUME_NONNULL_BEGIN

namespace {

ORTTensorTypeAndShapeInfo* CXXAPIToPublicTensorTypeAndShapeInfo(
    const Ort::TensorTypeAndShapeInfo& CXXAPITensorTypeAndShapeInfo) {
  auto* result = [[ORTTensorTypeAndShapeInfo alloc] init];
  const auto elementType = CXXAPITensorTypeAndShapeInfo.GetElementType();
  const std::vector<int64_t> shape = CXXAPITensorTypeAndShapeInfo.GetShape();

  result.elementType = CAPIToPublicTensorElementType(elementType);
  auto* shapeArray = [[NSMutableArray alloc] initWithCapacity:shape.size()];
  for (size_t i = 0; i < shape.size(); ++i) {
    shapeArray[i] = @(shape[i]);
  }
  result.shape = shapeArray;

  return result;
}

ORTValueTypeInfo* CXXAPIToPublicValueTypeInfo(
    const Ort::TypeInfo& CXXAPITypeInfo) {
  auto* result = [[ORTValueTypeInfo alloc] init];
  const auto valueType = CXXAPITypeInfo.GetONNXType();

  result.type = CAPIToPublicValueType(valueType);

  if (valueType == ONNX_TYPE_TENSOR) {
    const auto tensorTypeAndShapeInfo = CXXAPITypeInfo.GetTensorTypeAndShapeInfo();
    result.tensorTypeAndShapeInfo = CXXAPIToPublicTensorTypeAndShapeInfo(tensorTypeAndShapeInfo);
  }

  return result;
}

}  // namespace

@interface ORTValue ()

// pointer to any external tensor data to keep alive for the lifetime of the ORTValue
@property(nonatomic, nullable) NSMutableData* externalTensorData;

@end

@implementation ORTValue {
  std::optional<Ort::Value> _value;
  std::optional<Ort::TypeInfo> _typeInfo;
}

#pragma mark - Public

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

- (nullable ORTValueTypeInfo*)typeInfoWithError:(NSError**)error {
  ORT_OBJC_API_IMPL_BEGIN
  return CXXAPIToPublicValueTypeInfo(*_typeInfo);
  ORT_OBJC_API_IMPL_END_NULLABLE(error)
}

- (nullable ORTTensorTypeAndShapeInfo*)tensorTypeAndShapeInfoWithError:(NSError**)error {
  ORT_OBJC_API_IMPL_BEGIN
  const auto tensorTypeAndShapeInfo = _typeInfo->GetTensorTypeAndShapeInfo();
  return CXXAPIToPublicTensorTypeAndShapeInfo(tensorTypeAndShapeInfo);
  ORT_OBJC_API_IMPL_END_NULLABLE(error)
}

- (nullable NSMutableData*)tensorDataWithError:(NSError**)error {
  ORT_OBJC_API_IMPL_BEGIN

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

  ORT_OBJC_API_IMPL_END_NULLABLE(error)
}

#pragma mark - Internal

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

- (Ort::Value&)CXXAPIOrtValue {
  return *_value;
}

@end

@implementation ORTValueTypeInfo
@end

@implementation ORTTensorTypeAndShapeInfo
@end

NS_ASSUME_NONNULL_END
