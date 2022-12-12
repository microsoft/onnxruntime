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
#if ORT_OBJC_API_XNNPACK_EP_AVAILABLE
  try {
    NSDictionary * provider_option = @{
      @"intra_thread_num": [NSString stringWithFormat: @"%d", options.intra_thread_num]
    };
    AppendExecutionProvider(@"xnnpack", provider_option, error);
    return YES;
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error);
#else  // !ORT_OBJC_API_XNNPACK_EP_AVAILABLE
  static_cast<void>(options);
  ORTSaveCodeAndDescriptionToError(ORT_FAIL, "Xnnpack execution provider is not enabled.", error);
  return NO;
#endif
}

@end

NS_ASSUME_NONNULL_END
