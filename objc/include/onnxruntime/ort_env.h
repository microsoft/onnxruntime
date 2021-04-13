// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <Foundation/Foundation.h>

@interface ORTEnv : NSObject

-(instancetype) init:(NSError **)error;
-(void*) handle;

@end
