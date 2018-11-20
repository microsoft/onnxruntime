// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/error_code.h"
#ifdef __cplusplus
extern "C" {
#endif
struct ONNXRuntimeTensorTypeAndShapeInfo;

//copied from TensorProto::DataType
typedef enum OnnxRuntimeTensorElementDataType {
  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT = 1,   // float
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 = 2,   // uint8_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 = 3,    // int8_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 = 4,  // uint16_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 = 5,   // int16_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 = 6,   // int32_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 = 7,   // int64_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING = 8,  // string
  ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL = 9,    // bool
  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 = 10,
  ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE = 11,
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 = 12,
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 = 13,
  ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64 = 14,   // complex with float32 real and imaginary components
  ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128 = 15,  // complex with float64 real and imaginary components
  ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 = 16,    // Non-IEEE floating-point format based on IEEE754 single-precision
  ONNX_TENSOR_ELEMENT_DATA_TYPE_MAX = 17
} OnnxRuntimeTensorElementDataType;

/**
 * The retured value should be released by calling ONNXRuntimeReleaseObject
 */
ONNXRUNTIME_API(struct ONNXRuntimeTensorTypeAndShapeInfo*, ONNXRuntimeCreateTensorTypeAndShapeInfo);

ONNXRUNTIME_API_STATUS(ONNXRuntimeSetTensorElementType, _In_ struct ONNXRuntimeTensorTypeAndShapeInfo*, enum OnnxRuntimeTensorElementDataType type);

/**
 * \param info Created from ONNXRuntimeCreateTensorTypeAndShapeInfo() function
 * \param dim_values An array with length of `dim_count`. Its elements can contain negative values.
 * \param dim_count length of dim_values
 */
ONNXRUNTIME_API_STATUS(ONNXRuntimeSetDims, struct ONNXRuntimeTensorTypeAndShapeInfo* info, _In_ const int64_t* dim_values, size_t dim_count);

ONNXRUNTIME_API(enum OnnxRuntimeTensorElementDataType, ONNXRuntimeGetTensorElementType, _In_ const struct ONNXRuntimeTensorTypeAndShapeInfo*);
ONNXRUNTIME_API(size_t, ONNXRuntimeGetNumOfDimensions, _In_ const struct ONNXRuntimeTensorTypeAndShapeInfo* info);
ONNXRUNTIME_API(void, ONNXRuntimeGetDimensions, _In_ const struct ONNXRuntimeTensorTypeAndShapeInfo* info, _Out_ int64_t* dim_values, size_t dim_values_length);

/**
 * How many elements does this tensor have.
 * May return a negative value
 * e.g.
 * [] -> 1
 * [1,3,4] -> 12
 * [2,0,4] -> 0
 * [-1,3,4] -> -1
 */
ONNXRUNTIME_API(int64_t, ONNXRuntimeGetTensorShapeElementCount, _In_ const struct ONNXRuntimeTensorTypeAndShapeInfo* info);
#ifdef __cplusplus
}
#endif
