// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef TensorHelper_h
#define TensorHelper_h

#import <Foundation/Foundation.h>
#import <React/RCTBlobManager.h>

// Note: Using below syntax for including ort c api and ort extensions headers to resolve a compiling error happened
// in an expo react native ios app (a redefinition error happened with multiple object types defined within
// ORT C API header). It's an edge case that the compiler allows both ort c api headers to be included when #include
// syntax doesn't match. For the case when extensions not enabled, it still requires a onnxruntime prefix directory for
// searching paths. Also in general, it's a convention to use #include for C/C++ headers rather then #import. See:
// https://google.github.io/styleguide/objcguide.html#import-and-include
// https://microsoft.github.io/objc-guide/Headers/ImportAndInclude.html
#ifdef ORT_ENABLE_EXTENSIONS
#include "onnxruntime_cxx_api.h"
#else
#include <onnxruntime/onnxruntime_cxx_api.h>
#endif

@interface TensorHelper : NSObject

/**
 * Supported tensor data type
 */
FOUNDATION_EXPORT NSString* const JsTensorTypeBool;
FOUNDATION_EXPORT NSString* const JsTensorTypeUnsignedByte;
FOUNDATION_EXPORT NSString* const JsTensorTypeByte;
FOUNDATION_EXPORT NSString* const JsTensorTypeShort;
FOUNDATION_EXPORT NSString* const JsTensorTypeInt;
FOUNDATION_EXPORT NSString* const JsTensorTypeLong;
FOUNDATION_EXPORT NSString* const JsTensorTypeFloat;
FOUNDATION_EXPORT NSString* const JsTensorTypeDouble;
FOUNDATION_EXPORT NSString* const JsTensorTypeString;

/**
 * It creates an input tensor from a map passed by react native js.
 * 'data' is blob object and the buffer is stored in RCTBlobManager. It first resolve it and creates a tensor.
 */
+(Ort::Value)createInputTensor:(RCTBlobManager *)blobManager
                         input:(NSDictionary*)input
                  ortAllocator:(OrtAllocator*)ortAllocator
                   allocations:(std::vector<Ort::MemoryAllocation>&)allocations;

/**
 * It creates an output map from an output tensor.
 * a data array is store in RCTBlobManager.
 */
+(NSDictionary*)createOutputTensor:(RCTBlobManager *)blobManager
                       outputNames:(const std::vector<const char*>&)outputNames
                            values:(const std::vector<Ort::Value>&)values;

@end

#endif /* TensorHelper_h */
