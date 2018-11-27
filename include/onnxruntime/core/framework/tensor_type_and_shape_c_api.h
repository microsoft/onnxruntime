// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/error_code.h"
#ifdef __cplusplus
extern "C" {
#endif
struct ONNXRuntimeTensorTypeAndShapeInfo;

//copied from TensorProto::DataType
//Currently, ONNXRuntime doesn't support complex64, complex128, bfloat16 types
typedef enum OnnxRuntimeTensorElementDataType {
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED = 0,
  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT = 1,   // maps to c type float
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 = 2,   // maps to c type uint8_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 = 3,    // maps to c type int8_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 = 4,  // maps to c type uint16_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 = 5,   // maps to c type int16_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 = 6,   // maps to c type int32_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 = 7,   // maps to c type int64_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING = 8,  // maps to c++ type std::string
  ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL = 9,    //
  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 = 10,
  ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE = 11,      // maps to c type double
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 = 12,      // maps to c type uint32_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 = 13,      // maps to c type uint64_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64 = 14,   // complex with float32 real and imaginary components
  ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128 = 15,  // complex with float64 real and imaginary components
  ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 = 16,    // Non-IEEE floating-point format based on IEEE754 single-precision
} OnnxRuntimeTensorElementDataType;

//sync with onnx TypeProto oneof
typedef enum ONNXRuntimeType {
  ONNXRUNTIME_TYPE_UNKNOWN,
  ONNXRUNTIME_TYPE_TENSOR,
  ONNXRUNTIME_TYPE_SEQUENCE,
  ONNXRUNTIME_TYPE_MAP,
  ONNXRUNTIME_TYPE_OPAQUE,
  ONNXRUNTIME_TYPE_SPARSETENSOR,
} ONNXRuntimeType;

struct ONNXRuntimeTypeInfo;

/**
 * Don't free the returned value
 */
ONNXRUNTIME_API(const struct ONNXRuntimeTensorTypeAndShapeInfo*, ONNXRuntimeCastTypeInfoToTensorInfo, _In_ struct ONNXRuntimeTypeInfo*);

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
 * return a negative value if unknown. (That this shape contains a symbolic variable which
 * represents an unknown dimension.)
 */
ONNXRUNTIME_API(int64_t, ONNXRuntimeGetTensorShapeElementCount, _In_ const struct ONNXRuntimeTensorTypeAndShapeInfo* info);
struct ONNXValue;

/**
 * \param out Should be freed by ONNXRuntimeReleaseObject after use
 */
ONNXRUNTIME_API_STATUS(ONNXRuntimeGetTensorShapeAndType, _In_ const struct ONNXValue* value,
                       _Out_ struct ONNXRuntimeTensorTypeAndShapeInfo** out);

/**
 * Get the type information of an ONNXValue
 * \param value
 * \param out The returned value should be freed by ONNXRuntimeReleaseObject after use
 */
ONNXRUNTIME_API_STATUS(ONNXRuntimeGetTypeInfo, _In_ const struct ONNXValue* value, struct ONNXRuntimeTypeInfo** out);

ONNXRUNTIME_API(enum ONNXRuntimeType, ONNXRuntimeGetValueType, _In_ const struct ONNXValue* value);

#ifdef __cplusplus
}
#endif
