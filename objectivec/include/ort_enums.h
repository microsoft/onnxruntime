// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
 * The ORT logging verbosity levels.
 */
typedef NS_ENUM(int32_t, ORTLoggingLevel) {
  ORTLoggingLevelVerbose,
  ORTLoggingLevelInfo,
  ORTLoggingLevelWarning,
  ORTLoggingLevelError,
  ORTLoggingLevelFatal,
};

/**
 * The ORT value types.
 * Currently, a subset of all types is supported.
 */
typedef NS_ENUM(int32_t, ORTValueType) {
  ORTValueTypeUnknown,
  ORTValueTypeTensor,
};

/**
 * The ORT tensor element data types.
 * Currently, a subset of all types is supported.
 */
typedef NS_ENUM(int32_t, ORTTensorElementDataType) {
  ORTTensorElementDataTypeUndefined,
  ORTTensorElementDataTypeFloat,
  ORTTensorElementDataTypeInt32,
};

NS_ASSUME_NONNULL_END
