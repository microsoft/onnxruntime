// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef TensorHelper_h
#define TensorHelper_h

#import <Foundation/Foundation.h>
#import "onnxruntime_cxx_api.h"

@interface TensorHelper : NSObject

FOUNDATION_EXPORT NSString* const JsTensorTypeBool;
FOUNDATION_EXPORT NSString* const JsTensorTypeByte;
FOUNDATION_EXPORT NSString* const JsTensorTypeShort;
FOUNDATION_EXPORT NSString* const JsTensorTypeInt;
FOUNDATION_EXPORT NSString* const JsTensorTypeLong;
FOUNDATION_EXPORT NSString* const JsTensorTypeFloat;
FOUNDATION_EXPORT NSString* const JsTensorTypeDouble;
FOUNDATION_EXPORT NSString* const JsTensorTypeString;

+(Ort::Value)createInputTensor:(NSDictionary*)input
                  ortAllocator:(OrtAllocator*)ortAllocator
                   allocations:(std::vector<Ort::MemoryAllocation>&)allocatons;

+(NSDictionary*)createOutputTensor:(const std::vector<const char*>&)outputNames
                            values:(const std::vector<Ort::Value>&)values;

@end

#endif /* TensorHelper_h */
