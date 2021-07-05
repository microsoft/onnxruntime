// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef TensorHelper_h
#define TensorHelper_h

#import <Foundation/Foundation.h>
#import <onnxruntime/onnxruntime_cxx_api.h>

@interface TensorHelper : NSObject

/**
 * Supported tensor data type
 */
FOUNDATION_EXPORT NSString* const JsTensorTypeBool;
FOUNDATION_EXPORT NSString* const JsTensorTypeByte;
FOUNDATION_EXPORT NSString* const JsTensorTypeShort;
FOUNDATION_EXPORT NSString* const JsTensorTypeInt;
FOUNDATION_EXPORT NSString* const JsTensorTypeLong;
FOUNDATION_EXPORT NSString* const JsTensorTypeFloat;
FOUNDATION_EXPORT NSString* const JsTensorTypeDouble;
FOUNDATION_EXPORT NSString* const JsTensorTypeString;

/**
 * It creates an input tensor from a map passed by react native js.
 * 'data' must be a string type as data is encoded as base64. It first decodes it and creates a tensor.
 */
+(Ort::Value)createInputTensor:(NSDictionary*)input
                  ortAllocator:(OrtAllocator*)ortAllocator
                   allocations:(std::vector<Ort::MemoryAllocation>&)allocatons;

/**
 * It creates an output map from an output tensor.
 * a data array is encoded as base64 string.
 */
+(NSDictionary*)createOutputTensor:(const std::vector<const char*>&)outputNames
                            values:(const std::vector<Ort::Value>&)values;

@end

#endif /* TensorHelper_h */
