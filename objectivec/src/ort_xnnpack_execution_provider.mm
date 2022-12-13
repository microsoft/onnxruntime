// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "ort_xnnpack_execution_provider.h"

#import "src/cxx_api.h"
#import "src/error_utils.h"
#import "src/ort_session_internal.h"

NS_ASSUME_NONNULL_BEGIN

@implementation ORTXnnpackExecutionProviderOptions

@end

@implementation ORTSessionOptions (ORTSessionOptionsXnnpackEP)

- (BOOL)appendXnnpackExecutionProviderWithOptions:(ORTXnnpackExecutionProviderOptions*)options
                                            error:(NSError**)error {
  try {
    NSDictionary* provider_options = @{
      @"intra_thread_num" : [NSString stringWithFormat:@"%d", options.intra_thread_num]
    };
    return appendExecutionProvider(@"xnnpack", provider_options, error);
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error);
}

@end

NS_ASSUME_NONNULL_END
