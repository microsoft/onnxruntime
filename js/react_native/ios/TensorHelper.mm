// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "TensorHelper.h"
#import <Foundation/Foundation.h>

@implementation TensorHelper

/**
 * Supported tensor data type
 */
NSString *const JsTensorTypeBool = @"bool";
NSString *const JsTensorTypeUnsignedByte = @"uint8";
NSString *const JsTensorTypeByte = @"int8";
NSString *const JsTensorTypeShort = @"int16";
NSString *const JsTensorTypeInt = @"int32";
NSString *const JsTensorTypeLong = @"int64";
NSString *const JsTensorTypeFloat = @"float32";
NSString *const JsTensorTypeDouble = @"float64";
NSString *const JsTensorTypeString = @"string";

/**
 * It creates an input tensor from a map passed by react native js.
 * 'data' is blob object and the buffer is stored in RCTBlobManager. It first resolve it and creates a tensor.
 */
+ (Ort::Value)createInputTensor:(RCTBlobManager *)blobManager
                          input:(NSDictionary *)input
                   ortAllocator:(OrtAllocator *)ortAllocator
                    allocations:(std::vector<Ort::MemoryAllocation> &)allocations {
  // shape
  NSArray *dimsArray = [input objectForKey:@"dims"];
  std::vector<int64_t> dims;
  dims.reserve(dimsArray.count);
  for (NSNumber *dim in dimsArray) {
    dims.emplace_back([dim longLongValue]);
  }

  // type
  ONNXTensorElementDataType tensorType = [self getOnnxTensorType:[input objectForKey:@"type"]];

  // data
  if (tensorType == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
    NSArray *values = [input objectForKey:@"data"];
    auto inputTensor =
        Ort::Value::CreateTensor(ortAllocator, dims.data(), dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
    size_t index = 0;
    for (NSString *value in values) {
      inputTensor.FillStringTensorElement([value UTF8String], index++);
    }
    return inputTensor;
  } else {
    NSDictionary *data = [input objectForKey:@"data"];
    NSString *blobId = [data objectForKey:@"blobId"];
    long size = [[data objectForKey:@"size"] longValue];
    long offset = [[data objectForKey:@"offset"] longValue];
    auto buffer = [blobManager resolve:blobId offset:offset size:size];
    Ort::Value inputTensor = [self createInputTensor:tensorType
                                                dims:dims
                                              buffer:buffer
                                        ortAllocator:ortAllocator
                                         allocations:allocations];
    [blobManager remove:blobId];
    return inputTensor;
  }
}

/**
 * It creates an output map from an output tensor.
 * a data array is store in RCTBlobManager.
 */
+ (NSDictionary *)createOutputTensor:(RCTBlobManager *)blobManager
                         outputNames:(const std::vector<const char *> &)outputNames
                              values:(const std::vector<Ort::Value> &)values {
  if (outputNames.size() != values.size()) {
    NSException *exception = [NSException exceptionWithName:@"create output tensor"
                                                     reason:@"output name and tensor count mismatched"
                                                   userInfo:nil];
    @throw exception;
  }

  NSMutableDictionary *outputTensorMap = [NSMutableDictionary dictionary];

  for (size_t i = 0; i < outputNames.size(); ++i) {
    const auto outputName = outputNames[i];
    const Ort::Value &value = values[i];

    if (!value.IsTensor()) {
      NSException *exception = [NSException exceptionWithName:@"create output tensor"
                                                       reason:@"only tensor type is supported"
                                                     userInfo:nil];
      @throw exception;
    }

    NSMutableDictionary *outputTensor = [NSMutableDictionary dictionary];

    // dims
    NSMutableArray *outputDims = [NSMutableArray array];
    auto dims = value.GetTensorTypeAndShapeInfo().GetShape();
    for (auto dim : dims) {
      [outputDims addObject:[NSNumber numberWithLongLong:dim]];
    }
    outputTensor[@"dims"] = outputDims;

    // type
    outputTensor[@"type"] = [self getJsTensorType:value.GetTensorTypeAndShapeInfo().GetElementType()];

    // data
    if (value.GetTensorTypeAndShapeInfo().GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
      NSMutableArray *buffer = [NSMutableArray array];
      for (NSInteger i = 0; i < value.GetTensorTypeAndShapeInfo().GetElementCount(); ++i) {
        size_t elementLength = value.GetStringTensorElementLength(i);
        std::string element(elementLength, '\0');
        value.GetStringTensorElement(elementLength, i, (void *)element.data());
        [buffer addObject:[NSString stringWithUTF8String:element.data()]];
      }
      outputTensor[@"data"] = buffer;
    } else {
      NSData *data = [self createOutputTensor:value];
      NSString *blobId = [blobManager store:data];
      outputTensor[@"data"] = @{
        @"blobId" : blobId,
        @"offset" : @0,
        @"size" : @(data.length),
      };
    }

    outputTensorMap[[NSString stringWithUTF8String:outputName]] = outputTensor;
  }

  return outputTensorMap;
}

template <typename T>
static Ort::Value createInputTensorT(OrtAllocator *ortAllocator, const std::vector<int64_t> &dims, NSData *buffer,
                                     std::vector<Ort::MemoryAllocation> &allocations) {
  T *dataBuffer = static_cast<T *>(ortAllocator->Alloc(ortAllocator, [buffer length]));
  allocations.emplace_back(ortAllocator, dataBuffer, [buffer length]);
  memcpy(static_cast<void *>(dataBuffer), [buffer bytes], [buffer length]);

  return Ort::Value::CreateTensor<T>(ortAllocator->Info(ortAllocator), dataBuffer, buffer.length / sizeof(T),
                                     dims.data(), dims.size());
}

+ (Ort::Value)createInputTensor:(ONNXTensorElementDataType)tensorType
                           dims:(const std::vector<int64_t> &)dims
                         buffer:(NSData *)buffer
                   ortAllocator:(OrtAllocator *)ortAllocator
                    allocations:(std::vector<Ort::MemoryAllocation> &)allocations {
  switch (tensorType) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    return createInputTensorT<float>(ortAllocator, dims, buffer, allocations);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    return createInputTensorT<uint8_t>(ortAllocator, dims, buffer, allocations);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    return createInputTensorT<int8_t>(ortAllocator, dims, buffer, allocations);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    return createInputTensorT<int16_t>(ortAllocator, dims, buffer, allocations);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    return createInputTensorT<int32_t>(ortAllocator, dims, buffer, allocations);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    return createInputTensorT<int64_t>(ortAllocator, dims, buffer, allocations);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    return createInputTensorT<bool>(ortAllocator, dims, buffer, allocations);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    return createInputTensorT<double_t>(ortAllocator, dims, buffer, allocations);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
  default: {
    NSException *exception = [NSException exceptionWithName:@"create input tensor"
                                                     reason:@"unsupported tensor type"
                                                   userInfo:nil];
    @throw exception;
  }
  }
}

template <typename T> static NSData *createOutputTensorT(const Ort::Value &tensor) {
  const auto data = tensor.GetTensorData<T>();
  return [NSData dataWithBytesNoCopy:(void *)data
                              length:tensor.GetTensorTypeAndShapeInfo().GetElementCount() * sizeof(T)
                        freeWhenDone:false];
}

+ (NSData *)createOutputTensor:(const Ort::Value &)tensor {
  ONNXTensorElementDataType tensorType = tensor.GetTensorTypeAndShapeInfo().GetElementType();

  switch (tensorType) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    return createOutputTensorT<float>(tensor);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    return createOutputTensorT<uint8_t>(tensor);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    return createOutputTensorT<int8_t>(tensor);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    return createOutputTensorT<int16_t>(tensor);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    return createOutputTensorT<int32_t>(tensor);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    return createOutputTensorT<int64_t>(tensor);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    return createOutputTensorT<bool>(tensor);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    return createOutputTensorT<double_t>(tensor);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
  default: {
    NSException *exception = [NSException exceptionWithName:@"create output tensor"
                                                     reason:@"unsupported tensor type"
                                                   userInfo:nil];
    @throw exception;
  }
  }
}

NSDictionary *JsTensorTypeToOnnxTensorTypeMap;
NSDictionary *OnnxTensorTypeToJsTensorTypeMap;

+ (void)initialize {
  JsTensorTypeToOnnxTensorTypeMap = @{
    JsTensorTypeFloat : @(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
    JsTensorTypeUnsignedByte : @(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8),
    JsTensorTypeByte : @(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8),
    JsTensorTypeShort : @(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16),
    JsTensorTypeInt : @(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
    JsTensorTypeLong : @(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
    JsTensorTypeString : @(ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING),
    JsTensorTypeBool : @(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL),
    JsTensorTypeDouble : @(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)
  };

  OnnxTensorTypeToJsTensorTypeMap = @{
    @(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) : JsTensorTypeFloat,
    @(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) : JsTensorTypeUnsignedByte,
    @(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) : JsTensorTypeByte,
    @(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16) : JsTensorTypeShort,
    @(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) : JsTensorTypeInt,
    @(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) : JsTensorTypeLong,
    @(ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) : JsTensorTypeString,
    @(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) : JsTensorTypeBool,
    @(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) : JsTensorTypeDouble
  };
}

+ (ONNXTensorElementDataType)getOnnxTensorType:(const NSString *)type {
  if ([JsTensorTypeToOnnxTensorTypeMap objectForKey:type]) {
    return (ONNXTensorElementDataType)[JsTensorTypeToOnnxTensorTypeMap[type] intValue];
  } else {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
}

+ (NSString *)getJsTensorType:(ONNXTensorElementDataType)type {
  if ([OnnxTensorTypeToJsTensorTypeMap objectForKey:@(type)]) {
    return OnnxTensorTypeToJsTensorTypeMap[@(type)];
  } else {
    return @"undefined";
  }
}

@end
