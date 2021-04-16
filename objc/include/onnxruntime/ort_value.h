// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <Foundation/Foundation.h>

typedef NS_ENUM(NSUInteger, ORTElementDataType) {
    ORTElementDataTypeUndefined,
    ORTElementDataTypeFloat,
    ORTElementDataTypeInt32,
};

@interface ORTValue : NSObject

-(instancetype) initTensorWithData:(NSMutableData*) data
                       elementType:(ORTElementDataType) type
                             shape:(const int64_t*)shape
                          shapeLen:(size_t)shapeLen
                             error:(NSError**) error;

@end
