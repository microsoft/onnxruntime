// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <Foundation/Foundation.h>

@class ORTEnv;
@class ORTValue;

// TODO
@interface ORTSessionOptions : NSObject
@end

// TODO
@interface ORTRunOptions : NSObject
@end

@interface ORTSession : NSObject

- (instancetype)initWithEnv:(ORTEnv*)env
                  modelPath:(NSString*)path
                      error:(NSError**)error;

- (BOOL)runWithInputs:(NSDictionary<NSString*, ORTValue*>*)inputs
              outputs:(NSDictionary<NSString*, ORTValue*>*)outputs
                error:(NSError**)error;

@end
