// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <Foundation/Foundation.h>

#include "core/session/onnxruntime_cxx_api.h"

@interface ORTErrorUtils : NSObject

+ (void)saveErrorCode:(int)code
          description:(const char*)description_cstr
              toError:(NSError**)error;

@end
