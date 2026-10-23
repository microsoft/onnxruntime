// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <Foundation/Foundation.h>

@class ORTEnv;
@class ORTSessionOptions;

NS_ASSUME_NONNULL_BEGIN

// Test-only setup for registration, device selection, and compilation, which lack public Objective-C APIs.
@interface ORTEpContextFixture : NSObject
- (instancetype)initWithEnv:(ORTEnv*)env pluginPath:(NSString*)pluginPath;
- (void)configureSessionOptions:(ORTSessionOptions*)options;
- (NSData*)compileModelAtPath:(NSString*)source
              outputDirectory:(NSString*)directory
               sessionOptions:(ORTSessionOptions*)options
                  contextData:(NSData* _Nullable* _Nonnull)contextData
                  contextName:(NSString* _Nullable* _Nonnull)contextName;
- (void)unregister;
@end

NS_ASSUME_NONNULL_END
