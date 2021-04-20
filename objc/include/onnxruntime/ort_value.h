// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
 * The supported ORT tensor element data types.
 */
typedef NS_ENUM(int32_t, ORTTensorElementDataType) {
  ORTTensorElementDataTypeUndefined,
  ORTTensorElementDataTypeFloat,
  ORTTensorElementDataTypeInt32,
};

/**
 * An ORT value encapsulates data used as an input or output to a model at runtime.
 * Typically, ORT values represent tensors.
 */
@interface ORTValue : NSObject

- (nullable instancetype)init NS_UNAVAILABLE;

/**
 * Creates an ORT value that is a tensor.
 * The tensor data is allocated by the caller.
 *
 * @param data The tensor data.
 * @param type The tensor data element type.
 * @param shape The tensor shape dimensions.
 * @param shapeLen The number of tensor shape dimensions.
 * @param error Optional error information set if an error occurs.
 * @return The instance, or nil if an error occurs.
 */
- (nullable instancetype)initTensorWithData:(NSMutableData*)data
                                elementType:(ORTTensorElementDataType)elementType
                                      shape:(const int64_t*)shape
                                   shapeLen:(size_t)shapeLen
                                      error:(NSError**)error NS_DESIGNATED_INITIALIZER;

@end

NS_ASSUME_NONNULL_END
