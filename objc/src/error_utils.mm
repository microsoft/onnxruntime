// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "src/error_utils.h"

NS_ASSUME_NONNULL_BEGIN

static NSString* const kOrtErrorDomain = @"onnxruntime";

@implementation ORTErrorUtils

+ (void)saveErrorCode:(int)code
          description:(const char*)descriptionCstr
              toError:(NSError**)error {
  if (!error) return;

  NSString* description = [NSString stringWithCString:descriptionCstr
                                             encoding:NSASCIIStringEncoding];

  *error = [NSError errorWithDomain:kOrtErrorDomain
                               code:code
                           userInfo:@{NSLocalizedDescriptionKey : description}];
}

@end

NS_ASSUME_NONNULL_END
