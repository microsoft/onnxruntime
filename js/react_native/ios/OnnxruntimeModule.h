// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef OnnxruntimeModule_h
#define OnnxruntimeModule_h

#import <React/RCTBridgeModule.h>

@interface OnnxruntimeModule : NSObject<RCTBridgeModule>

-(NSDictionary*)loadModel:(NSString*)modelPath
                  options:(NSDictionary*)options;

-(NSDictionary*)run:(NSString*)url
              input:(NSDictionary*)input
             output:(NSArray*)output
            options:(NSDictionary*)options;

@end

#endif /* OnnxruntimeModule_h */
