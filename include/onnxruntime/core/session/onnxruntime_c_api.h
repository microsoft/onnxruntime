// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// =====================================================================================================
// NOTE: This header is PRE-RELEASE and subject to change. Please do not rely on this file not changing.
// =====================================================================================================

#pragma once
#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// SAL2 staffs
#ifndef _WIN32
#define _In_
#define _In_opt_
#define _Out_
#define _Out_opt_
#define _Inout_
#define _Inout_opt_
#define _Frees_ptr_opt_
#define ORT_ALL_ARGS_NONNULL __attribute__((nonnull))
#else
#include <specstrings.h>
#define ORT_ALL_ARGS_NONNULL
#endif

#ifdef _WIN32
// Define ONNX_RUNTIME_DLL_IMPORT if your program is dynamically linked to Ort.
// dllexport is not used, we use a .def file.
#ifdef ONNX_RUNTIME_DLL_IMPORT
#define ONNX_RUNTIME_EXPORT __declspec(dllimport)
#else
#define ONNX_RUNTIME_EXPORT
#endif
#define ORT_API_CALL _stdcall
#define ONNX_RUNTIME_MUST_USE_RESULT
#else
#define ONNX_RUNTIME_EXPORT
#define ORT_API_CALL
#define ONNX_RUNTIME_MUST_USE_RESULT __attribute__((warn_unused_result))
#endif

//Any pointer marked with _In_ or _Out_, cannot be NULL. Caller should ensure that.

#ifdef __cplusplus
// Windows users should use unicode paths when possible to bypass the MAX_PATH limitation
// Every type name starting with 'P' is a pointer type, an opaque handler
// Every pointer marked with _In_ or _Out_, cannot be NULL. Caller should ensure that.
// for ReleaseXXX(...) functions, they can accept NULL pointer.
#define NO_EXCEPTION noexcept
#else
#define NO_EXCEPTION
#endif

typedef enum OrtErrorCode {
  ORT_OK = 0,
  ORT_FAIL = 1,
  ORT_INVALID_ARGUMENT = 2,
  ORT_NO_SUCHFILE = 3,
  ORT_NO_MODEL = 4,
  ORT_ENGINE_ERROR = 5,
  ORT_RUNTIME_EXCEPTION = 6,
  ORT_INVALID_PROTOBUF = 7,
  ORT_MODEL_LOADED = 8,
  ORT_NOT_IMPLEMENTED = 9,
  ORT_INVALID_GRAPH = 10,
  ORT_SHAPE_INFERENCE_NOT_REGISTERED = 11,
  ORT_REQUIREMENT_NOT_REGISTERED = 12
} OrtErrorCode;

// ONNXStatus is always returned as a pointer. nullptr indicates success
typedef void ONNXStatus;

// __VA_ARGS__ on Windows and Linux are different
#define ORT_API(RETURN_TYPE, NAME, ...) \
  ONNX_RUNTIME_EXPORT RETURN_TYPE ORT_API_CALL NAME(__VA_ARGS__) NO_EXCEPTION

#define ORT_API_STATUS(NAME, ...) \
  ONNX_RUNTIME_EXPORT ONNXStatus* ORT_API_CALL NAME(__VA_ARGS__) NO_EXCEPTION ONNX_RUNTIME_MUST_USE_RESULT

// Used in *.cc files. Almost as same as ORT_API_STATUS, except without ONNX_RUNTIME_MUST_USE_RESULT
#define ORT_API_STATUS_IMPL(NAME, ...) \
  ONNX_RUNTIME_EXPORT ONNXStatus* ORT_API_CALL NAME(__VA_ARGS__) NO_EXCEPTION

#define DEFINE_RUNTIME_CLASS2(NAME, TYPE) \
  ORT_API(void, Release##NAME, _Frees_ptr_opt_ TYPE* input);

#define DEFINE_RUNTIME_CLASS(X) \
  struct X;                     \
  typedef struct X X;           \
  DEFINE_RUNTIME_CLASS2(X, X)

// ONNXStatus* is pointer to something like this:
// struct ONNXStatus {
//   OrtErrorCode code;
//   char msg[]; // a null-terminated string, var length
// }
DEFINE_RUNTIME_CLASS2(ONNXStatus, void);

/**
 * \param msg A null-terminated string. Its content will be copied into the newly created ONNXStatus
 */
ORT_API(ONNXStatus*, CreateONNXStatus, OrtErrorCode code, _In_ const char* msg)
ORT_ALL_ARGS_NONNULL;

ORT_API(OrtErrorCode, OrtGetErrorCode, _In_ const ONNXStatus* status)
ORT_ALL_ARGS_NONNULL;
/**
 * \param status must not be NULL
 * \return The error message inside the `status`. Don't free the returned value.
 */
ORT_API(const char*, OrtGetErrorMessage, _In_ const ONNXStatus* status)
ORT_ALL_ARGS_NONNULL;

//
// Tensor Type and Shapes
//
struct OrtTensorTypeAndShapeInfo;

//copied from TensorProto::DataType
//Currently, Ort doesn't support complex64, complex128, bfloat16 types
typedef enum OrtTensorElementDataType {
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
} OrtTensorElementDataType;

//sync with onnx TypeProto oneof
typedef enum OrtType {
  ORT_TYPE_UNKNOWN,
  ORT_TYPE_TENSOR,
  ORT_TYPE_SEQUENCE,
  ORT_TYPE_MAP,
  ORT_TYPE_OPAQUE,
  ORT_TYPE_SPARSETENSOR,
} OrtType;

struct OrtTypeInfo;

/**
 * Don't free the returned value
 */
ORT_API(const struct OrtTensorTypeAndShapeInfo*, OrtCastTypeInfoToTensorInfo, _In_ struct OrtTypeInfo*);

/**
 * The retured value should be released by calling OrtReleaseObject
 */
ORT_API(struct OrtTensorTypeAndShapeInfo*, OrtCreateTensorTypeAndShapeInfo);

ORT_API_STATUS(OrtSetTensorElementType, _In_ struct OrtTensorTypeAndShapeInfo*, enum OrtTensorElementDataType type);

/**
 * \param info Created from OrtCreateTensorTypeAndShapeInfo() function
 * \param dim_values An array with length of `dim_count`. Its elements can contain negative values.
 * \param dim_count length of dim_values
 */
ORT_API_STATUS(OrtSetDims, struct OrtTensorTypeAndShapeInfo* info, _In_ const int64_t* dim_values, size_t dim_count);

ORT_API(enum OrtTensorElementDataType, OrtGetTensorElementType, _In_ const struct OrtTensorTypeAndShapeInfo*);
ORT_API(size_t, OrtGetNumOfDimensions, _In_ const struct OrtTensorTypeAndShapeInfo* info);
ORT_API(void, OrtGetDimensions, _In_ const struct OrtTensorTypeAndShapeInfo* info, _Out_ int64_t* dim_values, size_t dim_values_length);

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
ORT_API(int64_t, OrtGetTensorShapeElementCount, _In_ const struct OrtTensorTypeAndShapeInfo* info);
struct ONNXValue;

/**
 * \param out Should be freed by OrtReleaseObject after use
 */
ORT_API_STATUS(OrtGetTensorShapeAndType, _In_ const struct ONNXValue* value,
               _Out_ struct OrtTensorTypeAndShapeInfo** out);

/**
 * Get the type information of an ONNXValue
 * \param value
 * \param out The returned value should be freed by OrtReleaseObject after use
 */
ORT_API_STATUS(OrtGetTypeInfo, _In_ const struct ONNXValue* value, struct OrtTypeInfo** out);

ORT_API(enum OrtType, OrtGetValueType, _In_ const struct ONNXValue* value);

//
// RuntimeRunOptions
//
struct OrtRunOptions;
typedef struct OrtRunOptions OrtRunOptions;
/**
 * \return A pointer of the newly created object. The pointer should be freed by OrtReleaseObject after use
 */
ORT_API(OrtRunOptions*, OrtCreateRunOptions);

ORT_API_STATUS(OrtRunOptionsSetRunLogVerbosityLevel, _In_ OrtRunOptions*, unsigned int);
ORT_API_STATUS(OrtRunOptionsSetRunTag, _In_ OrtRunOptions*, _In_ const char* run_tag);

ORT_API(unsigned int, OrtRunOptionsGetRunLogVerbosityLevel, _In_ OrtRunOptions*);
ORT_API(const char*, OrtRunOptionsGetRunTag, _In_ OrtRunOptions*);

// set a flag so that any running OrtRunInference* calls that are using this instance of ORtRunOptions
// will exit as soon as possible if the flag is true.
ORT_API(void, OrtRunOptionsSetTerminate, _In_ OrtRunOptions*, _In_ bool value);

DEFINE_RUNTIME_CLASS(OrtProvider);

/**
 * Just like the IUnknown interface in COM
 * Every type inherented from ONNXObject should be deleted by OrtReleaseObject(...).
 */
typedef struct ONNXObject {
  ///returns the new reference count.
  uint32_t(ORT_API_CALL* AddRef)(void* this_);
  ///returns the new reference count.
  uint32_t(ORT_API_CALL* Release)(void* this_);
  //TODO: implement QueryInterface?
} ONNXObject;

/**
 * This function is a wrapper to "(*(ONNXObject**)ptr)->AddRef(ptr)"
 * WARNING: There is NO type checking in this function.
 * Before calling this function, caller should make sure current ref count > 0
 * \return the new reference count
 */
ORT_API(uint32_t, OrtAddRefToObject, _In_ void* ptr);

/**
 *
 * A wrapper to "(*(ONNXObject**)ptr)->Release(ptr)"
 * WARNING: There is NO type checking in this function.
 * \param ptr Can be NULL. If it's NULL, this function will return zero.
 * \return the new reference count.
 */
ORT_API(uint32_t, OrtReleaseObject, _Inout_opt_ void* ptr);

//Inherented from ONNXObject
typedef struct OrtProviderFactoryInterface {
  ONNXObject parent;
  ONNXStatus*(ORT_API_CALL* CreateProvider)(void* this_, OrtProvider** out);
} OrtProviderFactoryInterface;

struct OrtSessionOptions;
typedef struct OrtSessionOptions OrtSessionOptions;

/**
 * \return A pointer of the newly created object. The pointer should be freed by OrtReleaseObject after use
 */
ORT_API(OrtSessionOptions*, OrtCreateSessionOptions, void);

/// create a copy of an existing OrtSessionOptions
ORT_API(OrtSessionOptions*, OrtCloneSessionOptions, OrtSessionOptions*);
ORT_API(void, OrtEnableSequentialExecution, _In_ OrtSessionOptions* options);
ORT_API(void, OrtDisableSequentialExecution, _In_ OrtSessionOptions* options);

// enable profiling for this session.
ORT_API(void, OrtEnableProfiling, _In_ OrtSessionOptions* options, _In_ const char* profile_file_prefix);
ORT_API(void, OrtDisableProfiling, _In_ OrtSessionOptions* options);

// enable the memory pattern optimization.
// The idea is if the input shapes are the same, we could trace the internal memory allocation
// and generate a memory pattern for future request. So next time we could just do one allocation
// with a big chunk for all the internal memory allocation.
ORT_API(void, OrtEnableMemPattern, _In_ OrtSessionOptions* options);
ORT_API(void, OrtDisableMemPattern, _In_ OrtSessionOptions* options);

// enable the memory arena on CPU
// Arena may pre-allocate memory for future usage.
// set this option to false if you don't want it.
ORT_API(void, OrtEnableCpuMemArena, _In_ OrtSessionOptions* options);
ORT_API(void, OrtDisableCpuMemArena, _In_ OrtSessionOptions* options);

///< logger id to use for session output
ORT_API(void, OrtSetSessionLogId, _In_ OrtSessionOptions* options, const char* logid);

///< applies to session load, initialization, etc
ORT_API(void, OrtSetSessionLogVerbosityLevel, _In_ OrtSessionOptions* options, uint32_t session_log_verbosity_level);

///How many threads in the session thread pool.
ORT_API(int, OrtSetSessionThreadPoolSize, _In_ OrtSessionOptions* options, int session_thread_pool_size);

/**
  * The order of invocation indicates the preference order as well. In other words call this method
  * on your most preferred execution provider first followed by the less preferred ones.
  * Calling this API is optional in which case Ort will use its internal CPU execution provider.
  */
ORT_API(void, OrtSessionOptionsAppendExecutionProvider, _In_ OrtSessionOptions* options, _In_ OrtProviderFactoryInterface** f);

ORT_API(void, OrtAddCustomOp, _In_ OrtSessionOptions* options, const char* custom_op_path);

typedef enum OrtAllocatorType {
  OrtDeviceAllocator = 0,
  OrtArenaAllocator = 1
} OrtAllocatorType;

/**
   memory types for allocator, exec provider specific types should be extended in each provider
*/
typedef enum OrtMemType {
  OrtMemTypeCPUInput = -2,              // Any CPU memory used by non-CPU execution provider
  OrtMemTypeCPUOutput = -1,             // CPU accessible memory outputted by non-CPU execution provider, i.e. CUDA_PINNED
  OrtMemTypeCPU = OrtMemTypeCPUOutput,  // temporary CPU accessible memory allocated by non-CPU execution provider, i.e. CUDA_PINNED
  OrtMemTypeDefault = 0,                // the default allocator for execution provider
} OrtMemType;

DEFINE_RUNTIME_CLASS(OrtAllocatorInfo);

ORT_API_STATUS(OrtCreateAllocatorInfo, _In_ const char* name1, enum OrtAllocatorType type, int id1, enum OrtMemType mem_type1, _Out_ OrtAllocatorInfo** out);

/**
 * Test if two allocation info are equal
 * \return 0, equal. zero, not equal
 */
ORT_API(int, OrtCompareAllocatorInfo, _In_ const OrtAllocatorInfo* info1, _In_ const OrtAllocatorInfo* info2)
ORT_ALL_ARGS_NONNULL;
/**
 * Do not free the returned value
 */
ORT_API(const char*, OrtAllocatorInfoGetName, _In_ OrtAllocatorInfo* ptr);
ORT_API(int, OrtAllocatorInfoGetId, _In_ OrtAllocatorInfo* ptr);
ORT_API(OrtMemType, OrtAllocatorInfoGetMemType, _In_ OrtAllocatorInfo* ptr);
ORT_API(OrtAllocatorType, OrtAllocatorInfoGetType, _In_ OrtAllocatorInfo* ptr);

//inherented from ONNXObject
typedef struct OrtAllocatorInterface {
  struct ONNXObject parent;
  void*(ORT_API_CALL* Alloc)(void* this_, size_t size);
  void(ORT_API_CALL* Free)(void* this_, void* p);
  const struct OrtAllocatorInfo*(ORT_API_CALL* Info)(const void* this_);
} OrtAllocatorInterface;

typedef OrtAllocatorInterface* OrtAllocator;

ORT_API(void*, OrtAllocatorAlloc, _Inout_ OrtAllocator* ptr, size_t size);
ORT_API(void, OrtAllocatorFree, _Inout_ OrtAllocator* ptr, void* p);
ORT_API(const struct OrtAllocatorInfo*, OrtAllocatorGetInfo, _In_ const OrtAllocator* ptr);

struct OrtEnv;
typedef struct OrtEnv OrtEnv;

typedef enum OrtLoggingLevel {
  ORT_LOGGING_LEVEL_kVERBOSE = 0,
  ORT_LOGGING_LEVEL_kINFO = 1,
  ORT_LOGGING_LEVEL_kWARNING = 2,
  ORT_LOGGING_LEVEL_kERROR = 3,
  ORT_LOGGING_LEVEL_kFATAL = 4
} OrtLoggingLevel;

typedef void(ORT_API_CALL* OrtLoggingFunction)(
    void* param, OrtLoggingLevel severity, const char* category, const char* logid, const char* code_location,
    const char* message);
/**
 * OrtEnv is process-wise. For each process, only one OrtEnv can be created. Don't do it multiple times
 * \param out Should be freed by `OrtReleaseObject` after use
 */
ORT_API_STATUS(OrtInitialize, OrtLoggingLevel default_warning_level, _In_ const char* logid, _Out_ OrtEnv** out)
ORT_ALL_ARGS_NONNULL;

/**
 * OrtEnv is process-wise. For each process, only one OrtEnv can be created. Don't do it multiple times
 * \param out Should be freed by `OrtReleaseObject` after use
 */
ORT_API_STATUS(OrtInitializeWithCustomLogger, OrtLoggingFunction logging_function,
               _In_opt_ void* logger_param, OrtLoggingLevel default_warning_level,
               _In_ const char* logid,
               _Out_ OrtEnv** out);

DEFINE_RUNTIME_CLASS(ONNXSession);

//TODO: document the path separator convention? '/' vs '\'
//TODO: should specify the access characteristics of model_path. Is this read only during the
//execution of OrtCreateInferenceSession, or does the ONNXSession retain a handle to the file/directory
//and continue to access throughout the ONNXSession lifetime?
// What sort of access is needed to model_path : read or read/write?
//TODO:  allow loading from an in-memory byte-array
#ifdef _WIN32
ORT_API_STATUS(OrtCreateInferenceSession, _In_ OrtEnv* env, _In_ const wchar_t* model_path,
               _In_ const OrtSessionOptions* options, _Out_ ONNXSession** out);
#else
ORT_API_STATUS(OrtCreateInferenceSession, _In_ OrtEnv* env, _In_ const char* model_path,
               _In_ const OrtSessionOptions* options, _Out_ ONNXSession** out);
#endif

DEFINE_RUNTIME_CLASS(ONNXValue);

///Call OrtReleaseObject to release the returned value
ORT_API_STATUS(OrtCreateDefaultAllocator, _Out_ OrtAllocator** out);

/**
 * Create a tensor from an allocator. ReleaseONNXValue will also release the buffer inside the output value
 * \param out will keep a reference to the allocator, without reference counting(will be fixed). Should be freed by
 *            calling ReleaseONNXValue
 * \param type must be one of TENSOR_ELEMENT_DATA_TYPE_xxxx
 */
ORT_API_STATUS(OrtCreateTensorAsONNXValue, _Inout_ OrtAllocator* allocator,
               _In_ const size_t* shape, size_t shape_len, OrtTensorElementDataType type,
               _Out_ ONNXValue** out);

/**
 * Create a tensor with user's buffer. You can fill the buffer either before calling this function or after.
 * p_data is owned by caller. ReleaseONNXValue won't release p_data.
 * \param out Should be freed by calling ReleaseONNXValue
 */
ORT_API_STATUS(OrtCreateTensorWithDataAsONNXValue, _In_ const OrtAllocatorInfo* info,
               _In_ void* p_data, size_t p_data_len, _In_ const size_t* shape, size_t shape_len,
               OrtTensorElementDataType type, _Out_ ONNXValue** out);

/// This function doesn't work with string tensor
/// this is a no-copy method whose pointer is only valid until the backing ONNXValue is free'd.
ORT_API_STATUS(OrtGetTensorMutableData, _Inout_ ONNXValue* value, _Out_ void** out);

/**
 * Test if an ONNXValue is a tensor
 * \return zero, false. non-zero true
 */
ORT_API(int, OrtIsTensor, _In_ const ONNXValue* value);

/**
 * \param value A tensor created from OrtCreateTensor*** function.
 * \param s each A string array. Each string in this array must be null terminated.
 * \param s_len length of s
 */
ORT_API_STATUS(OrtFillStringTensor, _In_ ONNXValue* value, _In_ const char* const* s, size_t s_len);
/**
 * \param value A tensor created from OrtCreateTensor*** function.
 * \param len total data length, not including the trailing '\0' chars.
 */
ORT_API_STATUS(OrtGetStringTensorDataLength, _In_ const ONNXValue* value, _Out_ size_t* len);

/**
 * \param s string contents. Each string is NOT null-terminated.
 * \param value A tensor created from OrtCreateTensor*** function.
 * \param s_len total data length, get it from OrtGetStringTensorDataLength
 */
ORT_API_STATUS(OrtGetStringTensorContent, _In_ const ONNXValue* value, _Out_ void* s, size_t s_len,
               _Out_ size_t* offsets, size_t offsets_len);

DEFINE_RUNTIME_CLASS(ONNXValueList);

ORT_API_STATUS(OrtRunInference, _Inout_ ONNXSession* sess,
               _In_ OrtRunOptions* run_options,
               _In_ const char* const* input_names, _In_ const ONNXValue* const* input, size_t input_len,
               _In_ const char* const* output_names, size_t output_names_len, _Out_ ONNXValue** output);

ORT_API_STATUS(OrtInferenceSessionGetInputCount, _In_ const ONNXSession* sess, _Out_ size_t* out);
ORT_API_STATUS(OrtInferenceSessionGetOutputCount, _In_ const ONNXSession* sess, _Out_ size_t* out);

/**
 * \param out  should be freed by OrtReleaseObject after use
 */
ORT_API_STATUS(OrtInferenceSessionGetInputTypeInfo, _In_ const ONNXSession* sess, size_t index, _Out_ struct OrtTypeInfo** out);

/**
 * \param out  should be freed by OrtReleaseObject after use
 */
ORT_API_STATUS(OrtInferenceSessionGetOutputTypeInfo, _In_ const ONNXSession* sess, size_t index, _Out_ struct OrtTypeInfo** out);

ORT_API_STATUS(OrtInferenceSessionGetInputName, _In_ const ONNXSession* sess, size_t index,
               _Inout_ OrtAllocator* allocator, _Out_ char** value);
ORT_API_STATUS(OrtInferenceSessionGetOutputName, _In_ const ONNXSession* sess, size_t index,
               _Inout_ OrtAllocator* allocator, _Out_ char** value);

ORT_API_STATUS(OrtTensorProtoToONNXValue, _Inout_ OrtAllocator* allocator,
               _In_ const void* input, int input_len, _Out_ ONNXValue** out);

/**
 * Deprecated. Please use OrtReleaseObject
 */
ORT_API(void, ReleaseONNXEnv, OrtEnv* env);

#ifdef __cplusplus
}
#endif
