// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
 * The supported ORT value types.
 */
typedef NS_ENUM(int32_t, ORTValueType) {
  ORTValueTypeUnknown,
  ORTValueTypeTensor,
};

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
 */
@interface ORTValue : NSObject

- (nullable instancetype)init NS_UNAVAILABLE;

/**
 * Creates a value that is a tensor.
 * The tensor data is allocated by the caller.
 *
 * @param data The tensor data.
 * @param type The tensor data element type.
 * @param shape The tensor shape.
 * @param[out] error Optional error information set if an error occurs.
 * @return The instance, or nil if an error occurs.
 */
- (nullable instancetype)initTensorWithData:(NSMutableData*)tensorData
                                elementType:(ORTTensorElementDataType)elementType
                                      shape:(NSArray<NSNumber*>*)shape
                                      error:(NSError**)error;

/**
 * Gets the value type.
 *
 * @param[out] valueType The type of the value.
 * @param[out] error Optional error information set if an error occurs.
 * @return Whether the value type was retrieved successfully.
 */
- (BOOL)valueType:(ORTValueType*)valueType
            error:(NSError**)error;

/**
 * Gets the tensor data element type.
 * This assumes that the value is a tensor.
 *
 * @param[out] elementType The type of the tensor's data elements.
 * @param[out] error Optional error information set if an error occurs.
 * @return Whether the tensor data element type was retrieved successfully.
 */
- (BOOL)tensorElementType:(ORTTensorElementDataType*)elementType
                    error:(NSError**)error;

/**
 * Gets the tensor shape.
 * This assumes that the value is a tensor.
 *
 * @param[out] error Optional error information set if an error occurs.
 * @return The tensor shape, or nil if an error occurs.
 */
- (nullable NSArray<NSNumber*>*)tensorShapeWithError:(NSError**)error;

/**
 * Gets the tensor data.
 * This assumes that the value is a tensor.
 *
 * @param[out] error Optional error information set if an error occurs.
 * @return The tensor data, or nil if an error occurs.
 */
- (nullable NSMutableData*)tensorDataWithError:(NSError**)error;

@end

NS_ASSUME_NONNULL_END
