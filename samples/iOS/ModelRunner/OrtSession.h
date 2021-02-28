// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef ort_session_h
#define ort_session_h

#import <Foundation/Foundation.h>

// @class TFLInterpreterOptions;
// @class TFLTensor;

NS_ASSUME_NONNULL_BEGIN


/**
 * An OnnxRuntime Session
 */
@interface OrtMobileSession : NSObject

- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error;
- (NSString*)run: (NSMutableData *)buff mname:(NSString*)mname error:(NSError **)error;

@end

NS_ASSUME_NONNULL_END


#endif /* ort_session_h */
