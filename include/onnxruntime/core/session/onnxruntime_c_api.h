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
#define ONNXRUNTIME_ALL_ARGS_NONNULL __attribute__((nonnull))
#else
#include <specstrings.h>
#define ONNXRUNTIME_ALL_ARGS_NONNULL
#endif

#ifdef _WIN32
// Define ONNX_RUNTIME_DLL_IMPORT if your program is dynamically linked to onnxruntime.
// dllexport is not used, we use a .def file.
#ifdef ONNX_RUNTIME_DLL_IMPORT
#define ONNX_RUNTIME_EXPORT __declspec(dllimport)
#else
#define ONNX_RUNTIME_EXPORT
#endif
#define ONNXRUNTIME_API_STATUSCALL _stdcall
#define ONNX_RUNTIME_MUST_USE_RESULT
#else
#define ONNX_RUNTIME_EXPORT
#define ONNXRUNTIME_API_STATUSCALL
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

// ONNXStatus is always returned as a pointer. nullptr indicates success
typedef void ONNXStatus;

// __VA_ARGS__ on Windows and Linux are different
#define ONNXRUNTIME_API(RETURN_TYPE, NAME, ...) \
  ONNX_RUNTIME_EXPORT RETURN_TYPE ONNXRUNTIME_API_STATUSCALL NAME(__VA_ARGS__) NO_EXCEPTION

#define ONNXRUNTIME_API_STATUS(NAME, ...) \
  ONNX_RUNTIME_EXPORT ONNXStatus* ONNXRUNTIME_API_STATUSCALL NAME(__VA_ARGS__) NO_EXCEPTION ONNX_RUNTIME_MUST_USE_RESULT

// Used in *.cc files. Almost as same as ONNXRUNTIME_API_STATUS, except without ONNX_RUNTIME_MUST_USE_RESULT
#define ONNXRUNTIME_API_STATUS_IMPL(NAME, ...) \
  ONNX_RUNTIME_EXPORT ONNXStatus* ONNXRUNTIME_API_STATUSCALL NAME(__VA_ARGS__) NO_EXCEPTION

#define DEFINE_RUNTIME_CLASS2(NAME, TYPE) \
  ONNXRUNTIME_API(void, Release##NAME, _Frees_ptr_opt_ TYPE* input);

#define DEFINE_RUNTIME_CLASS(X) \
  struct X;                     \
  typedef struct X X;           \
  DEFINE_RUNTIME_CLASS2(X, X)

// ONNXStatus* is pointer to something like this:
// struct ONNXStatus {
//   ONNXRuntimeErrorCode code;
//   char msg[]; // a null-terminated string, var length
// }
DEFINE_RUNTIME_CLASS2(ONNXStatus, void);

/**
 * \param msg A null-terminated string. Its content will be copied into the newly created ONNXStatus
 */
ONNXRUNTIME_API(ONNXStatus*, CreateONNXStatus, ONNXRuntimeErrorCode code, _In_ const char* msg)
ONNXRUNTIME_ALL_ARGS_NONNULL;

ONNXRUNTIME_API(ONNXRuntimeErrorCode, ONNXRuntimeGetErrorCode, _In_ const ONNXStatus* status)
ONNXRUNTIME_ALL_ARGS_NONNULL;
/**
 * \param status must not be NULL
 * \return The error message inside the `status`. Don't free the returned value.
 */
ONNXRUNTIME_API(const char*, ONNXRuntimeGetErrorMessage, _In_ const ONNXStatus* status)
ONNXRUNTIME_ALL_ARGS_NONNULL;

//
// Tensor Type and Shapes
//
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

//
// RuntimeRunOptions
//
struct ONNXRuntimeRunOptions;
typedef struct ONNXRuntimeRunOptions ONNXRuntimeRunOptions;
/**
 * \return A pointer of the newly created object. The pointer should be freed by ONNXRuntimeReleaseObject after use
 */
ONNXRUNTIME_API(ONNXRuntimeRunOptions*, ONNXRuntimeCreateRunOptions);

ONNXRUNTIME_API_STATUS(ONNXRuntimeRunOptionsSetRunLogVerbosityLevel, _In_ ONNXRuntimeRunOptions*, unsigned int);
ONNXRUNTIME_API_STATUS(ONNXRuntimeRunOptionsSetRunTag, _In_ ONNXRuntimeRunOptions*, _In_ const char* run_tag);

ONNXRUNTIME_API(unsigned int, ONNXRuntimeRunOptionsGetRunLogVerbosityLevel, _In_ ONNXRuntimeRunOptions*);
ONNXRUNTIME_API(const char*, ONNXRuntimeRunOptionsGetRunTag, _In_ ONNXRuntimeRunOptions*);

// set a flag so that any running ONNXRuntimeRunInference* calls that are using this instance of ONNXRuntimeRunOptions
// will exit as soon as possible if the flag is true.
ONNXRUNTIME_API(void, ONNXRuntimeRunOptionsSetTerminate, _In_ ONNXRuntimeRunOptions*, _In_ bool value);

DEFINE_RUNTIME_CLASS(ONNXRuntimeProvider);

/**
 * Just like the IUnknown interface in COM
 * Every type inherented from ONNXObject should be deleted by ONNXRuntimeReleaseObject(...).
 */
typedef struct ONNXObject {
  ///returns the new reference count.
  uint32_t(ONNXRUNTIME_API_STATUSCALL* AddRef)(void* this_);
  ///returns the new reference count.
  uint32_t(ONNXRUNTIME_API_STATUSCALL* Release)(void* this_);
  //TODO: implement QueryInterface?
} ONNXObject;

/**
 * This function is a wrapper to "(*(ONNXObject**)ptr)->AddRef(ptr)"
 * WARNING: There is NO type checking in this function.
 * Before calling this function, caller should make sure current ref count > 0
 * \return the new reference count
 */
ONNXRUNTIME_API(uint32_t, ONNXRuntimeAddRefToObject, _In_ void* ptr);

/**
 *
 * A wrapper to "(*(ONNXObject**)ptr)->Release(ptr)"
 * WARNING: There is NO type checking in this function.
 * \param ptr Can be NULL. If it's NULL, this function will return zero.
 * \return the new reference count.
 */
ONNXRUNTIME_API(uint32_t, ONNXRuntimeReleaseObject, _Inout_opt_ void* ptr);

//Inherented from ONNXObject
typedef struct ONNXRuntimeProviderFactoryInterface {
  ONNXObject parent;
  ONNXStatus*(ONNXRUNTIME_API_STATUSCALL* CreateProvider)(void* this_, ONNXRuntimeProvider** out);
} ONNXRuntimeProviderFactoryInterface;

struct ONNXRuntimeSessionOptions;
typedef struct ONNXRuntimeSessionOptions ONNXRuntimeSessionOptions;

/**
 * \return A pointer of the newly created object. The pointer should be freed by ONNXRuntimeReleaseObject after use
 */
ONNXRUNTIME_API(ONNXRuntimeSessionOptions*, ONNXRuntimeCreateSessionOptions, void);

/// create a copy of an existing ONNXRuntimeSessionOptions
ONNXRUNTIME_API(ONNXRuntimeSessionOptions*, ONNXRuntimeCloneSessionOptions, ONNXRuntimeSessionOptions*);
ONNXRUNTIME_API(void, ONNXRuntimeEnableSequentialExecution, _In_ ONNXRuntimeSessionOptions* options);
ONNXRUNTIME_API(void, ONNXRuntimeDisableSequentialExecution, _In_ ONNXRuntimeSessionOptions* options);

// enable profiling for this session.
ONNXRUNTIME_API(void, ONNXRuntimeEnableProfiling, _In_ ONNXRuntimeSessionOptions* options, _In_ const char* profile_file_prefix);
ONNXRUNTIME_API(void, ONNXRuntimeDisableProfiling, _In_ ONNXRuntimeSessionOptions* options);

// enable the memory pattern optimization.
// The idea is if the input shapes are the same, we could trace the internal memory allocation
// and generate a memory pattern for future request. So next time we could just do one allocation
// with a big chunk for all the internal memory allocation.
ONNXRUNTIME_API(void, ONNXRuntimeEnableMemPattern, _In_ ONNXRuntimeSessionOptions* options);
ONNXRUNTIME_API(void, ONNXRuntimeDisableMemPattern, _In_ ONNXRuntimeSessionOptions* options);

// enable the memory arena on CPU
// Arena may pre-allocate memory for future usage.
// set this option to false if you don't want it.
ONNXRUNTIME_API(void, ONNXRuntimeEnableCpuMemArena, _In_ ONNXRuntimeSessionOptions* options);
ONNXRUNTIME_API(void, ONNXRuntimeDisableCpuMemArena, _In_ ONNXRuntimeSessionOptions* options);

///< logger id to use for session output
ONNXRUNTIME_API(void, ONNXRuntimeSetSessionLogId, _In_ ONNXRuntimeSessionOptions* options, const char* logid);

///< applies to session load, initialization, etc
ONNXRUNTIME_API(void, ONNXRuntimeSetSessionLogVerbosityLevel, _In_ ONNXRuntimeSessionOptions* options, uint32_t session_log_verbosity_level);

///How many threads in the session thread pool.
ONNXRUNTIME_API(int, ONNXRuntimeSetSessionThreadPoolSize, _In_ ONNXRuntimeSessionOptions* options, int session_thread_pool_size);

/**
  * The order of invocation indicates the preference order as well. In other words call this method
  * on your most preferred execution provider first followed by the less preferred ones.
  * Calling this API is optional in which case onnxruntime will use its internal CPU execution provider.
  */
ONNXRUNTIME_API(void, ONNXRuntimeSessionOptionsAppendExecutionProvider, _In_ ONNXRuntimeSessionOptions* options, _In_ ONNXRuntimeProviderFactoryInterface** f);

ONNXRUNTIME_API(void, ONNXRuntimeAddCustomOp, _In_ ONNXRuntimeSessionOptions* options, const char* custom_op_path);

typedef enum ONNXRuntimeAllocatorType {
  ONNXRuntimeDeviceAllocator = 0,
  ONNXRuntimeArenaAllocator = 1
} ONNXRuntimeAllocatorType;

/**
   memory types for allocator, exec provider specific types should be extended in each provider
*/
typedef enum ONNXRuntimeMemType {
  ONNXRuntimeMemTypeCPUInput = -2,                      // Any CPU memory used by non-CPU execution provider
  ONNXRuntimeMemTypeCPUOutput = -1,                     // CPU accessible memory outputted by non-CPU execution provider, i.e. CUDA_PINNED
  ONNXRuntimeMemTypeCPU = ONNXRuntimeMemTypeCPUOutput,  // temporary CPU accessible memory allocated by non-CPU execution provider, i.e. CUDA_PINNED
  ONNXRuntimeMemTypeDefault = 0,                        // the default allocator for execution provider
} ONNXRuntimeMemType;

DEFINE_RUNTIME_CLASS(ONNXRuntimeAllocatorInfo);

ONNXRUNTIME_API_STATUS(ONNXRuntimeCreateAllocatorInfo, _In_ const char* name1, enum ONNXRuntimeAllocatorType type, int id1, enum ONNXRuntimeMemType mem_type1, _Out_ ONNXRuntimeAllocatorInfo** out);

/**
 * Test if two allocation info are equal
 * \return 0, equal. zero, not equal
 */
ONNXRUNTIME_API(int, ONNXRuntimeCompareAllocatorInfo, _In_ const ONNXRuntimeAllocatorInfo* info1, _In_ const ONNXRuntimeAllocatorInfo* info2)
ONNXRUNTIME_ALL_ARGS_NONNULL;
/**
 * Do not free the returned value
 */
ONNXRUNTIME_API(const char*, ONNXRuntimeAllocatorInfoGetName, _In_ ONNXRuntimeAllocatorInfo* ptr);
ONNXRUNTIME_API(int, ONNXRuntimeAllocatorInfoGetId, _In_ ONNXRuntimeAllocatorInfo* ptr);
ONNXRUNTIME_API(ONNXRuntimeMemType, ONNXRuntimeAllocatorInfoGetMemType, _In_ ONNXRuntimeAllocatorInfo* ptr);
ONNXRUNTIME_API(ONNXRuntimeAllocatorType, ONNXRuntimeAllocatorInfoGetType, _In_ ONNXRuntimeAllocatorInfo* ptr);

//inherented from ONNXObject
typedef struct ONNXRuntimeAllocatorInteface {
  struct ONNXObject parent;
  void*(ONNXRUNTIME_API_STATUSCALL* Alloc)(void* this_, size_t size);
  void(ONNXRUNTIME_API_STATUSCALL* Free)(void* this_, void* p);
  const struct ONNXRuntimeAllocatorInfo*(ONNXRUNTIME_API_STATUSCALL* Info)(const void* this_);
} ONNXRuntimeAllocatorInteface;

typedef ONNXRuntimeAllocatorInteface* ONNXRuntimeAllocator;

ONNXRUNTIME_API(void*, ONNXRuntimeAllocatorAlloc, _Inout_ ONNXRuntimeAllocator* ptr, size_t size);
ONNXRUNTIME_API(void, ONNXRuntimeAllocatorFree, _Inout_ ONNXRuntimeAllocator* ptr, void* p);
ONNXRUNTIME_API(const struct ONNXRuntimeAllocatorInfo*, ONNXRuntimeAllocatorGetInfo, _In_ const ONNXRuntimeAllocator* ptr);

struct ONNXRuntimeEnv;
typedef struct ONNXRuntimeEnv ONNXRuntimeEnv;

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
