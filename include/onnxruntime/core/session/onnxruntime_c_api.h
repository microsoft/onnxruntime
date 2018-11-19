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
#include "core/session/tensor_type_and_shape_c_api.h"
#include "allocator.h"
#include "session_options_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

//Any pointer marked with _In_ or _Out_, cannot be NULL. Caller should ensure that.

typedef enum ONNXRuntimeType {
  ONNXRUNTIME_TYPE_TENSOR,
  ONNXRUNTIME_TYPE_SEQUENCE,
  ONNXRUNTIME_TYPE_MAP,
  ONNXRUNTIME_TYPE_OPAQUE,
  ONNXRUNTIME_TYPE_ELEMENT,  //basic types like float/int32
} ONNXRuntimeType;

typedef struct ONNXOpaqueTypeInfo {
  char* domain;
  char* name;
} ONNXOpaqueTypeInfo;

//Each ONNX value is a n-ary tree.
//Data is only stored in leaf nodes.
//Every non-leaf node contains a field of ONNXRuntimeType
//Each leaf node is either a tensor, or an ONNXArray.

/**
 * ReleaseONNXEnv function calls ::google::protobuf::ShutdownProtobufLibrary().
 * Therefore, you should only call ReleaseONNXEnv at the end of your program.
 * Once you did that, don't call any onnxruntime, onnx or protobuf functions again.
 */
DEFINE_RUNTIME_CLASS(ONNXEnv);

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
 * ONNXEnv is process-wise. For each process, only one ONNXEnv can be created. Don't do it multiple times
 */
ONNXRUNTIME_API_STATUS(ONNXRuntimeInitialize, ONNXRuntimeLoggingLevel default_warning_level, _In_ const char* logid,
                       _Out_ ONNXEnv** out)
ONNXRUNTIME_ALL_ARGS_NONNULL;
/**
 * ONNXEnv is process-wise. For each process, only one ONNXEnv can be created. Don't do it multiple times
 */
ONNXRUNTIME_API_STATUS(ONNXRuntimeInitializeWithCustomLogger, ONNXRuntimeLoggingFunction logging_function,
                       void* logger_param, ONNXRuntimeLoggingLevel default_warning_level,
                       _In_ const char* logid,
                       _Out_ ONNXEnv** out);

DEFINE_RUNTIME_CLASS(ONNXSession);

//TODO: document the path separator convention? '/' vs '\'
//TODO: should specify the access characteristics of model_path. Is this read only during the
//execution of ONNXRuntimeCreateInferenceSession, or does the ONNXSession retain a handle to the file/directory
//and continue to access throughout the ONNXSession lifetime?
// What sort of access is needed to model_path : read or read/write?
//TODO:  allow loading from an in-memory byte-array
#ifdef _WIN32
ONNXRUNTIME_API_STATUS(ONNXRuntimeCreateInferenceSession, _In_ ONNXEnv* env, _In_ const wchar_t* model_path,
                       _In_ const ONNXRuntimeSessionOptions* options, _Out_ ONNXSessionPtr* out);
#else
ONNXRUNTIME_API_STATUS(ONNXRuntimeCreateInferenceSession, _In_ ONNXEnv* env, _In_ const char* model_path,
                       _In_ const ONNXRuntimeSessionOptions* options, _Out_ ONNXSessionPtr* out);
#endif

DEFINE_RUNTIME_CLASS(ONNXValue);

///Call ONNXRuntimeReleaseObject to release the returned value
ONNXRUNTIME_API_STATUS(ONNXRuntimeCreateDefaultAllocator, _Out_ ONNXRuntimeAllocator** out);

/**
 * This function is only for advanced users. In most cases, please use ONNXRuntimeCreateTensorWithDataAsONNXValue
 * The returned ONNXValuePtr will keep a reference to allocator, without reference counting
 * \param type must be one of TENSOR_ELEMENT_DATA_TYPE_xxxx
 */
ONNXRUNTIME_API_STATUS(ONNXRuntimeCreateTensorAsONNXValue, _Inout_ ONNXRuntimeAllocator* allocator,
                       _In_ const size_t* shape, size_t shape_len, OnnxRuntimeTensorElementDataType type,
                       _Out_ ONNXValuePtr* out);

/**
 * p_data is owned by caller. ReleaseTensor won't release p_data. 
 */
ONNXRUNTIME_API_STATUS(ONNXRuntimeCreateTensorWithDataAsONNXValue, _In_ const ONNXRuntimeAllocatorInfo* info,
                       _In_ void* p_data, size_t p_data_len, _In_ const size_t* shape, size_t shape_len,
                       OnnxRuntimeTensorElementDataType type, _Out_ ONNXValuePtr* out);

/// This function doesn't work with string tensor
/// this is a no-copy method whose pointer is only valid until the backing ONNXValuePtr is free'd.
ONNXRUNTIME_API_STATUS(ONNXRuntimeGetTensorMutableData, _In_ ONNXValuePtr value, _Out_ void** out);

/**
 * \return zero, false. non-zero true
 */
ONNXRUNTIME_API(int, ONNXRuntimeIsTensor, _In_ ONNXValuePtr value);

/**
 * \param value A tensor created from ONNXRuntimeCreateTensor*** function.
 * \param s each A string array. Each string in this array must be null terminated.
 * \param s_len length of s
 */
ONNXRUNTIME_API_STATUS(ONNXRuntimeFillStringTensor, _In_ ONNXValuePtr value, _In_ const char* s[], size_t s_len);
/**
 * \param value A tensor created from ONNXRuntimeCreateTensor*** function.
 * \param len total data length, not including the trailing '\0' chars.
 */
ONNXRUNTIME_API_STATUS(ONNXRuntimeGetStringTensorDataLength, _In_ ONNXValuePtr value, _Out_ size_t* len);

/**
 * \param s string contents. Each string is NOT null-terminated.
 * \param value A tensor created from ONNXRuntimeCreateTensor*** function.
 * \param s_len total data length, get it from ONNXRuntimeGetStringTensorDataLength
 */
ONNXRUNTIME_API_STATUS(ONNXRuntimeGetStringTensorContent, _In_ ONNXValuePtr value, _Out_ void* s, size_t s_len,
                       _Out_ size_t* offsets, size_t offsets_len);

/**
 * \param out Should be freed by ONNXRuntimeReleaseObject after use
 */
ONNXRUNTIME_API_STATUS(ONNXRuntimeGetTensorShapeAndType, _In_ const ONNXValuePtr,
                       _Out_ struct ONNXRuntimeTensorTypeAndShapeInfo** out);

//not implemented
//ONNX_RUNTIME_EXPORT int GetPONNXValueDataType(_In_ ONNXValuePtr) NO_EXCEPTION;

DEFINE_RUNTIME_CLASS(ONNXValueList);

//For InferenceSession run calls, all the input values shouldn't created by allocator
//User should manage the buffer by himself, not allocator

/**
 * \param sess created by ONNXRuntimeCreateInferenceSession function
 * \param output must be freed by ReleaseONNXValueListPtr function
 */
ONNXRUNTIME_API_STATUS(ONNXRuntimeRunInferenceAndFetchAll, _In_ ONNXSessionPtr sess,
                       _In_ const char* input_names[], _In_ ONNXValuePtr* input, size_t input_len,
                       _Out_ ONNXValueListPtr* output, _Out_ size_t* output_len);
ONNXRUNTIME_API_STATUS(ONNXRuntimeRunInferenceAndFetchAllWithRunOptions, _In_ ONNXSessionPtr sess,
                       _In_ ONNXRuntimeRunOptionsPtr run_options,
                       _In_ const char* input_names[], _In_ ONNXValuePtr* input, size_t input_len,
                       _Out_ ONNXValueListPtr* output, _Out_ size_t* output_len);
ONNXRUNTIME_API_STATUS(ONNXRuntimeRunInference, _In_ ONNXSessionPtr sess,
                       _In_ const char* input_names[], _In_ ONNXValuePtr* input, size_t input_len,
                       _In_ const char* output_names[], size_t output_names_len, _Out_ ONNXValuePtr* output);
ONNXRUNTIME_API_STATUS(ONNXRuntimeRunInferenceWithRunOptions, _In_ ONNXSessionPtr sess,
                       _In_ ONNXRuntimeRunOptionsPtr run_options,
                       _In_ const char* input_names[], _In_ ONNXValuePtr* input, size_t input_len,
                       _In_ const char* output_names[], size_t output_names_len, _Out_ ONNXValuePtr* output);

ONNXRUNTIME_API_STATUS(ONNXRuntimeInferenceSessionGetInputCount, _In_ ONNXSessionPtr sess, _Out_ size_t* out);
ONNXRUNTIME_API_STATUS(ONNXRuntimeInferenceSessionGetOutputCount, _In_ ONNXSessionPtr sess, _Out_ size_t* out);

ONNXRUNTIME_API_STATUS(ONNXRuntimeInferenceSessionGetInputName, _In_ ONNXSessionPtr sess, size_t index,
                       _Inout_ ONNXRuntimeAllocator* allocator, _Out_ char** value);
ONNXRUNTIME_API_STATUS(ONNXRuntimeInferenceSessionGetOutputName, _In_ ONNXSessionPtr sess, size_t index,
                       _Inout_ ONNXRuntimeAllocator* allocator, _Out_ char** value);

//Tree for PONNXType:
//ONNXRUNTIME_TYPE_TENSOR -> ONNXTensorTypeInfo
//ONNXRUNTIME_TYPE_SEQUENCE -> nullptr
//ONNXRUNTIME_TYPE_MAP -> nullptr
//ONNXRUNTIME_TYPE_OPAQUE-> ONNXOpaqueTypeInfo
//ONNXRUNTIME_TYPE_ELEMENT -> nullptr

//The output value must be freed by ONNXRuntimeNodeDestoryTree
//ONNXRUNTIME_API_STATUS(ONNXRuntimeInferenceSessionGetInputType, _In_ ONNXSessionPtr sess, _Out_ PONNXType* out);
//ONNXRUNTIME_API_STATUS(ONNXRuntimeInferenceSessionGetOutputType, _In_ ONNXSessionPtr sess, _Out_ PONNXType* out);

/**
 * Get the n-th value from the List
 * \param index starts from zero
 */
ONNXRUNTIME_API(ONNXValuePtr, ONNXRuntimeONNXValueListGetNthValue, _In_ ONNXValueListPtr list, size_t index);

ONNXRUNTIME_API_STATUS(ONNXRuntimeTensorProtoToONNXValue, _Inout_ ONNXRuntimeAllocator* allocator,
                       _In_ const void* input, int input_len, _Out_ ONNXValuePtr* out);

#ifdef __cplusplus
}
#endif
