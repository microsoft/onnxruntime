// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "core/common/visibility_macros.h"
#include "core/framework/error_code.h"
#include "core/framework/onnx_object.h"
#include "core/framework/run_options_c_api.h"
#include "core/framework/tensor_type_and_shape_c_api.h"
#include "allocator.h"
#include "session_options_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

//Any pointer marked with _In_ or _Out_, cannot be NULL. Caller should ensure that.

struct ONNXRuntimeEnv;
typedef struct ONNXRuntimeEnv ONNXRuntimeEnv;
//old name
typedef struct ONNXRuntimeEnv* ONNXEnvPtr;

typedef enum ONNXRuntimeLoggingLevel {
  ONNXRUNTIME_LOGGING_LEVEL_kVERBOSE = 0,
  ONNXRUNTIME_LOGGING_LEVEL_kINFO = 1,
  ONNXRUNTIME_LOGGING_LEVEL_kWARNING = 2,
  ONNXRUNTIME_LOGGING_LEVEL_kERROR = 3,
  ONNXRUNTIME_LOGGING_LEVEL_kFATAL = 4
} ONNXRuntimeLoggingLevel;

typedef void(ONNXRUNTIME_API_STATUSCALL* ONNXRuntimeLoggingFunction)(
    void* param, ONNXRuntimeLoggingLevel severity, const char* category, const char* logid, const char* code_location,
    const char* message);
/**
 * ONNXRuntimeEnv is process-wise. For each process, only one ONNXRuntimeEnv can be created. Don't do it multiple times
 * \param out Should be freed by `ONNXRuntimeReleaseObject` after use
 */
ONNXRUNTIME_API_STATUS(ONNXRuntimeInitialize, ONNXRuntimeLoggingLevel default_warning_level, _In_ const char* logid,
                       _Out_ ONNXRuntimeEnv** out)
ONNXRUNTIME_ALL_ARGS_NONNULL;

/**
 * ONNXRuntimeEnv is process-wise. For each process, only one ONNXRuntimeEnv can be created. Don't do it multiple times
 * \param out Should be freed by `ONNXRuntimeReleaseObject` after use
 */
ONNXRUNTIME_API_STATUS(ONNXRuntimeInitializeWithCustomLogger, ONNXRuntimeLoggingFunction logging_function,
                       _In_opt_ void* logger_param, ONNXRuntimeLoggingLevel default_warning_level,
                       _In_ const char* logid,
                       _Out_ ONNXRuntimeEnv** out);

DEFINE_RUNTIME_CLASS(ONNXSession);

//TODO: document the path separator convention? '/' vs '\'
//TODO: should specify the access characteristics of model_path. Is this read only during the
//execution of ONNXRuntimeCreateInferenceSession, or does the ONNXSession retain a handle to the file/directory
//and continue to access throughout the ONNXSession lifetime?
// What sort of access is needed to model_path : read or read/write?
//TODO:  allow loading from an in-memory byte-array
#ifdef _WIN32
ONNXRUNTIME_API_STATUS(ONNXRuntimeCreateInferenceSession, _In_ ONNXRuntimeEnv* env, _In_ const wchar_t* model_path,
                       _In_ const ONNXRuntimeSessionOptions* options, _Out_ ONNXSession** out);
#else
ONNXRUNTIME_API_STATUS(ONNXRuntimeCreateInferenceSession, _In_ ONNXRuntimeEnv* env, _In_ const char* model_path,
                       _In_ const ONNXRuntimeSessionOptions* options, _Out_ ONNXSession** out);
#endif

DEFINE_RUNTIME_CLASS(ONNXValue);

///Call ONNXRuntimeReleaseObject to release the returned value
ONNXRUNTIME_API_STATUS(ONNXRuntimeCreateDefaultAllocator, _Out_ ONNXRuntimeAllocator** out);

/**
 * Create a tensor from an allocator. ReleaseONNXValue will also release the buffer inside the output value
 * \param out will keep a reference to the allocator, without reference counting(will be fixed). Should be freed by
 *            calling ReleaseONNXValue
 * \param type must be one of TENSOR_ELEMENT_DATA_TYPE_xxxx
 */
ONNXRUNTIME_API_STATUS(ONNXRuntimeCreateTensorAsONNXValue, _Inout_ ONNXRuntimeAllocator* allocator,
                       _In_ const size_t* shape, size_t shape_len, OnnxRuntimeTensorElementDataType type,
                       _Out_ ONNXValue** out);

/**
 * Create a tensor with user's buffer. You can fill the buffer either before calling this function or after.
 * p_data is owned by caller. ReleaseONNXValue won't release p_data.
 * \param out Should be freed by calling ReleaseONNXValue
 */
ONNXRUNTIME_API_STATUS(ONNXRuntimeCreateTensorWithDataAsONNXValue, _In_ const ONNXRuntimeAllocatorInfo* info,
                       _In_ void* p_data, size_t p_data_len, _In_ const size_t* shape, size_t shape_len,
                       OnnxRuntimeTensorElementDataType type, _Out_ ONNXValue** out);

/// This function doesn't work with string tensor
/// this is a no-copy method whose pointer is only valid until the backing ONNXValuePtr is free'd.
ONNXRUNTIME_API_STATUS(ONNXRuntimeGetTensorMutableData, _Inout_ ONNXValue* value, _Out_ void** out);

/**
 * Test if an ONNXValue is a tensor
 * \return zero, false. non-zero true
 */
ONNXRUNTIME_API(int, ONNXRuntimeIsTensor, _In_ const ONNXValue* value);

/**
 * \param value A tensor created from ONNXRuntimeCreateTensor*** function.
 * \param s each A string array. Each string in this array must be null terminated.
 * \param s_len length of s
 */
ONNXRUNTIME_API_STATUS(ONNXRuntimeFillStringTensor, _In_ ONNXValue* value, _In_ const char* const* s, size_t s_len);
/**
 * \param value A tensor created from ONNXRuntimeCreateTensor*** function.
 * \param len total data length, not including the trailing '\0' chars.
 */
ONNXRUNTIME_API_STATUS(ONNXRuntimeGetStringTensorDataLength, _In_ const ONNXValue* value, _Out_ size_t* len);

/**
 * \param s string contents. Each string is NOT null-terminated.
 * \param value A tensor created from ONNXRuntimeCreateTensor*** function.
 * \param s_len total data length, get it from ONNXRuntimeGetStringTensorDataLength
 */
ONNXRUNTIME_API_STATUS(ONNXRuntimeGetStringTensorContent, _In_ const ONNXValue* value, _Out_ void* s, size_t s_len,
                       _Out_ size_t* offsets, size_t offsets_len);

DEFINE_RUNTIME_CLASS(ONNXValueList);

ONNXRUNTIME_API_STATUS(ONNXRuntimeRunInference, _Inout_ ONNXSession* sess,
                       _In_ ONNXRuntimeRunOptions* run_options,
                       _In_ const char* const* input_names, _In_ const ONNXValue* const* input, size_t input_len,
                       _In_ const char* const* output_names, size_t output_names_len, _Out_ ONNXValue** output);

ONNXRUNTIME_API_STATUS(ONNXRuntimeInferenceSessionGetInputCount, _In_ const ONNXSession* sess, _Out_ size_t* out);
ONNXRUNTIME_API_STATUS(ONNXRuntimeInferenceSessionGetOutputCount, _In_ const ONNXSession* sess, _Out_ size_t* out);

/**
 * \param out  should be freed by ONNXRuntimeReleaseObject after use
 */
ONNXRUNTIME_API_STATUS(ONNXRuntimeInferenceSessionGetInputTypeInfo, _In_ const ONNXSession* sess, size_t index, _Out_ struct ONNXRuntimeTypeInfo** out);

/**
 * \param out  should be freed by ONNXRuntimeReleaseObject after use
 */
ONNXRUNTIME_API_STATUS(ONNXRuntimeInferenceSessionGetOutputTypeInfo, _In_ const ONNXSession* sess, size_t index, _Out_ struct ONNXRuntimeTypeInfo** out);

ONNXRUNTIME_API_STATUS(ONNXRuntimeInferenceSessionGetInputName, _In_ const ONNXSession* sess, size_t index,
                       _Inout_ ONNXRuntimeAllocator* allocator, _Out_ char** value);
ONNXRUNTIME_API_STATUS(ONNXRuntimeInferenceSessionGetOutputName, _In_ const ONNXSession* sess, size_t index,
                       _Inout_ ONNXRuntimeAllocator* allocator, _Out_ char** value);

ONNXRUNTIME_API_STATUS(ONNXRuntimeTensorProtoToONNXValue, _Inout_ ONNXRuntimeAllocator* allocator,
                       _In_ const void* input, int input_len, _Out_ ONNXValue** out);

/**
 * Deprecated. Please use ONNXRuntimeReleaseObject
 */
ONNXRUNTIME_API(void, ReleaseONNXEnv, ONNXRuntimeEnv* env);

#ifdef __cplusplus
}
#endif
