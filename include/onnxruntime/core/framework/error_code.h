// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "core/common/visibility_macros.h"

#ifdef __cplusplus
//Windows user should use unicode path whenever possible, to bypass the MAX_PATH limitation
//Evevy type name started with 'P' is a pointer type, an opaque handler
//Every pointer marked with _In_ or _Out_, cannot be NULL. Caller should ensure that.
//for ReleaseXXX(...) functions, they can accept NULL pointer.
#define NO_EXCEPTION noexcept
#else
#define NO_EXCEPTION
#endif

#ifdef __cplusplus
extern "C" {
#endif
typedef enum ONNXRuntimeErrorCode {
  ONNXRUNTIME_OK = 0,
  ONNXRUNTIME_FAIL = 1,
  ONNXRUNTIME_INVALID_ARGUMENT = 2,
  ONNXRUNTIME_NO_SUCHFILE = 3,
  ONNXRUNTIME_NO_MODEL = 4,
  ONNXRUNTIME_ENGINE_ERROR = 5,
  ONNXRUNTIME_RUNTIME_EXCEPTION = 6,
  ONNXRUNTIME_INVALID_PROTOBUF = 7,
  ONNXRUNTIME_MODEL_LOADED = 8,
  ONNXRUNTIME_NOT_IMPLEMENTED = 9,
  ONNXRUNTIME_INVALID_GRAPH = 10,
  ONNXRUNTIME_SHAPE_INFERENCE_NOT_REGISTERED = 11,
  ONNXRUNTIME_REQUIREMENT_NOT_REGISTERED = 12
} ONNXRuntimeErrorCode;

//nullptr indicates success. Otherwise, this pointer must be freed by
typedef void* ONNXStatusPtr;

#ifdef _WIN32
#define ONNXRUNTIME_API_STATUSCALL _stdcall
#else
#define ONNXRUNTIME_API_STATUSCALL
#endif

//__VA_ARGS__ on Windows and Linux are different
#define ONNXRUNTIME_API(RETURN_TYPE, NAME, ...) \
  ONNX_RUNTIME_EXPORT RETURN_TYPE ONNXRUNTIME_API_STATUSCALL NAME(__VA_ARGS__) NO_EXCEPTION

#define ONNXRUNTIME_API_STATUS(NAME, ...) \
  ONNX_RUNTIME_EXPORT ONNXStatusPtr ONNXRUNTIME_API_STATUSCALL NAME(__VA_ARGS__) NO_EXCEPTION ONNX_RUNTIME_MUST_USE_RESULT

//Used in *.cc files. Almost as same as ONNXRUNTIME_API_STATUS, expect without ONNX_RUNTIME_MUST_USE_RESULT
#define ONNXRUNTIME_API_STATUS_IMPL(NAME, ...) \
  ONNX_RUNTIME_EXPORT ONNXStatusPtr ONNXRUNTIME_API_STATUSCALL NAME(__VA_ARGS__) NO_EXCEPTION

#define DEFINE_RUNTIME_CLASS2(NAME, TYPE) \
  typedef TYPE* NAME##Ptr;                \
  ONNXRUNTIME_API(void, Release##NAME, _Frees_ptr_opt_ TYPE* input);

#define DEFINE_RUNTIME_CLASS(X) \
  struct X;                     \
  typedef struct X X;           \
  DEFINE_RUNTIME_CLASS2(X, X)

//ONNXStatusPtr is pointer to something like this:
//struct ONNXStatus{
//  ONNXRuntimeErrorCode code;
//  char msg[];//a null-terminated string, var length
//}
DEFINE_RUNTIME_CLASS2(ONNXStatus, void);

/**
 * \param msg A null-terminated string. Its content will be copied into the newly created ONNXStatus
 */
ONNXRUNTIME_API(ONNXStatusPtr, CreateONNXStatus, ONNXRuntimeErrorCode code, _In_ const char* msg)
ONNXRUNTIME_ALL_ARGS_NONNULL;

ONNXRUNTIME_API(ONNXRuntimeErrorCode, ONNXRuntimeGetErrorCode, _In_ const ONNXStatusPtr status)
ONNXRUNTIME_ALL_ARGS_NONNULL;
/**
 * \param status must not be NULL
 * \return The error message inside the `status`. Don't free the returned value.
 */
ONNXRUNTIME_API(const char*, ONNXRuntimeGetErrorMessage, _In_ const ONNXStatusPtr status)
ONNXRUNTIME_ALL_ARGS_NONNULL;
#ifdef __cplusplus
}
#endif
