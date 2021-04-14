// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "onnxruntime/error_utils.h"

static NSString* const kOrtErrorDomain = @"org.onnxruntime";

@implementation ORTErrorUtils
+ (void)saveErrorCode:(int)code
          description:(const char*)description_cstr
              toError:(NSError**)error {
    if (!error) return;
    
    NSString* description = [NSString stringWithCString:description_cstr
                                               encoding:NSASCIIStringEncoding];
    
    *error = [NSError errorWithDomain:kOrtErrorDomain
                                 code:code
                             userInfo:@{NSLocalizedDescriptionKey : description}];
}
@end
