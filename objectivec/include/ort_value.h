// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <Foundation/Foundation.h>

#import "ort_enums.h"

NS_ASSUME_NONNULL_BEGIN

@class ORTValueTypeInfo;
@class ORTTensorTypeAndShapeInfo;

/**
 * An ORT value encapsulates data used as an input or output to a model at runtime.
 */
@interface ORTValue : NSObject

- (instancetype)init NS_UNAVAILABLE;

/**
 * Creates a value that is a tensor.
 * The tensor data is allocated by the caller.
 *
 * @param tensorData The tensor data.
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
 * Gets the type information.
 *
 * @param[out] error Optional error information set if an error occurs.
 * @return The type information, or nil if an error occurs.
 */
- (nullable ORTValueTypeInfo*)typeInfoWithError:(NSError**)error;

/**
 * Gets the tensor type and shape information.
 * This assumes that the value is a tensor.
 *
 * @param[out] error Optional error information set if an error occurs.
 * @return The tensor type and shape information, or nil if an error occurs.
 */
- (nullable ORTTensorTypeAndShapeInfo*)tensorTypeAndShapeInfoWithError:(NSError**)error;

/**
 * Gets the tensor data.
 * This assumes that the value is a tensor.
 *
 * @param[out] error Optional error information set if an error occurs.
 * @return The tensor data, or nil if an error occurs.
 */
- (nullable NSMutableData*)tensorDataWithError:(NSError**)error;

@end

/**
 * A value's type information.
 */
@interface ORTValueTypeInfo : NSObject

/** The value type. */
@property(nonatomic) ORTValueType type;

/** The tensor type and shape information, if the value is a tensor. */
@property(nonatomic, nullable) ORTTensorTypeAndShapeInfo* tensorTypeAndShapeInfo;

@end

/**
 * A tensor's type and shape information.
 */
@interface ORTTensorTypeAndShapeInfo : NSObject

/** The tensor element data type. */
@property(nonatomic) ORTTensorElementDataType elementType;

/** The tensor shape. */
@property(nonatomic) NSArray<NSNumber*>* shape;

@end

NS_ASSUME_NONNULL_END
