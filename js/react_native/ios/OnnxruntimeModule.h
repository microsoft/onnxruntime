// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef OnnxruntimeModule_h
#define OnnxruntimeModule_h

#import <React/RCTBridgeModule.h>
#import <React/RCTBlobManager.h>

@interface OnnxruntimeModule : NSObject<RCTBridgeModule>

- (void)setBlobManager:(RCTBlobManager *)manager;

-(NSDictionary*)loadModel:(NSString*)modelPath
                  options:(NSDictionary*)options;

-(NSDictionary*)loadModelFromBuffer:(NSData*)modelData
                            options:(NSDictionary*)options;

-(void)dispose:(NSString*)key;

-(NSDictionary*)run:(NSString*)url
              input:(NSDictionary*)input
             output:(NSArray*)output
            options:(NSDictionary*)options;

@end

#endif /* OnnxruntimeModule_h */
