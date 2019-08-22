// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// This value is used in structures passed to ORT so that a newer version of ORT will still work with
#define ORT_API_VERSION 1

#ifdef __cplusplus
extern "C" {
#endif

// SAL2 Definitions
#ifndef _WIN32
#define _In_
#define _In_opt_
#define _Out_
#define _Outptr_
#define _Out_opt_
#define _Inout_
#define _Inout_opt_
#define _Frees_ptr_opt_
#define _Ret_maybenull_
#define _Ret_notnull_
#define _Check_return_
#define _Success_(X)
#define ORT_ALL_ARGS_NONNULL __attribute__((nonnull))
#else
#include <specstrings.h>
#define ORT_ALL_ARGS_NONNULL
#endif

#ifdef _WIN32
// Define ORT_DLL_IMPORT if your program is dynamically linked to Ort.
// dllexport is not used, we use a .def file.
#ifdef ORT_DLL_IMPORT
#define ORT_EXPORT __declspec(dllimport)
#else
#define ORT_EXPORT
#endif
#define ORT_API_CALL _stdcall
#define ORT_MUST_USE_RESULT
#define ORTCHAR_T wchar_t
#else
#define ORT_EXPORT
#define ORT_API_CALL
#define ORT_MUST_USE_RESULT __attribute__((warn_unused_result))
#define ORTCHAR_T char
#endif

#ifndef ORT_TSTR
#ifdef _WIN32
#define ORT_TSTR(X) L##X
#else
#define ORT_TSTR(X) (X)
#endif
#endif

// Any pointer marked with _In_ or _Out_, cannot be NULL.

#ifdef __cplusplus
// Windows users should use unicode paths when possible to bypass the MAX_PATH limitation
// Every pointer marked with _In_ or _Out_, cannot be NULL. Caller should ensure that.
// for ReleaseXXX(...) functions, they can accept NULL pointer.
#define NO_EXCEPTION noexcept
#else
#define NO_EXCEPTION
#endif

// Copied from TensorProto::DataType
// Currently, Ort doesn't support complex64, complex128, bfloat16 types
typedef enum ONNXTensorElementDataType {
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED,
  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,   // maps to c type float
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,   // maps to c type uint8_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,    // maps to c type int8_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,  // maps to c type uint16_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,   // maps to c type int16_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,   // maps to c type int32_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,   // maps to c type int64_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,  // maps to c++ type std::string
  ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
  ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,      // maps to c type double
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,      // maps to c type uint32_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,      // maps to c type uint64_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64,   // complex with float32 real and imaginary components
  ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128,  // complex with float64 real and imaginary components
  ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,    // Non-IEEE floating-point format based on IEEE754 single-precision
} ONNXTensorElementDataType;

// Synced with onnx TypeProto oneof
typedef enum ONNXType {
  ONNX_TYPE_UNKNOWN,
  ONNX_TYPE_TENSOR,
  ONNX_TYPE_SEQUENCE,
  ONNX_TYPE_MAP,
  ONNX_TYPE_OPAQUE,
  ONNX_TYPE_SPARSETENSOR,
} ONNXType;

typedef enum OrtLoggingLevel {
  ORT_LOGGING_LEVEL_VERBOSE,
  ORT_LOGGING_LEVEL_INFO,
  ORT_LOGGING_LEVEL_WARNING,
  ORT_LOGGING_LEVEL_ERROR,
  ORT_LOGGING_LEVEL_FATAL,
} OrtLoggingLevel;

typedef enum OrtErrorCode {
  ORT_OK,
  ORT_FAIL,
  ORT_INVALID_ARGUMENT,
  ORT_NO_SUCHFILE,
  ORT_NO_MODEL,
  ORT_ENGINE_ERROR,
  ORT_RUNTIME_EXCEPTION,
  ORT_INVALID_PROTOBUF,
  ORT_MODEL_LOADED,
  ORT_NOT_IMPLEMENTED,
  ORT_INVALID_GRAPH,
  ORT_SHAPE_INFERENCE_NOT_REGISTERED,
  ORT_REQUIREMENT_NOT_REGISTERED,
} OrtErrorCode;

// __VA_ARGS__ on Windows and Linux are different
#define ORT_API(RETURN_TYPE, NAME, ...) \
  ORT_EXPORT RETURN_TYPE ORT_API_CALL NAME(__VA_ARGS__) NO_EXCEPTION

#define ORT_API_STATUS(NAME, ...) \
  ORT_EXPORT _Check_return_ _Success_(return == 0) _Ret_maybenull_ OrtStatus* ORT_API_CALL NAME(__VA_ARGS__) NO_EXCEPTION ORT_MUST_USE_RESULT

// Used in *.cc files. Almost as same as ORT_API_STATUS, except without ORT_MUST_USE_RESULT
#define ORT_API_STATUS_IMPL(NAME, ...) \
  ORT_EXPORT _Check_return_ _Success_(return == 0) _Ret_maybenull_ OrtStatus* ORT_API_CALL NAME(__VA_ARGS__) NO_EXCEPTION

#define ORT_RUNTIME_CLASS(X)    \
  struct Ort##X;                \
  typedef struct Ort##X Ort##X; \
  ORT_API(void, OrtRelease##X, _Frees_ptr_opt_ Ort##X* input);

// The actual types defined have an Ort prefix
ORT_RUNTIME_CLASS(Env);
ORT_RUNTIME_CLASS(Status);  // nullptr for Status* indicates success
ORT_RUNTIME_CLASS(Provider);
ORT_RUNTIME_CLASS(AllocatorInfo);
ORT_RUNTIME_CLASS(Session);  //Don't call OrtReleaseSession from Dllmain (because session owns a thread pool)
ORT_RUNTIME_CLASS(Value);
ORT_RUNTIME_CLASS(RunOptions);
ORT_RUNTIME_CLASS(TypeInfo);
ORT_RUNTIME_CLASS(TensorTypeAndShapeInfo);
ORT_RUNTIME_CLASS(SessionOptions);
ORT_RUNTIME_CLASS(CustomOpDomain);
ORT_RUNTIME_CLASS(Allocator);

// When passing in an allocator to any ORT function, be sure that the allocator object
// is not destroyed until the last allocated object using it is freed.
typedef struct OrtAllocator {
  uint32_t version;  // Initialize to ORT_API_VERSION
  void*(ORT_API_CALL* Alloc)(struct OrtAllocator* this_, size_t size);
  void(ORT_API_CALL* Free)(struct OrtAllocator* this_, void* p);
  const struct OrtAllocatorInfo*(ORT_API_CALL* Info)(const struct OrtAllocator* this_);
} OrtAllocator;

typedef void(ORT_API_CALL* OrtLoggingFunction)(
    void* param, OrtLoggingLevel severity, const char* category, const char* logid, const char* code_location,
    const char* message);

/**
 * \param out Should be freed by `OrtReleaseEnv` after use
 */
ORT_API_STATUS(OrtCreateEnv, OrtLoggingLevel default_logging_level, _In_ const char* logid, _Outptr_ OrtEnv** out)
ORT_ALL_ARGS_NONNULL;

/**
 * \param out Should be freed by `OrtReleaseEnv` after use
 */
ORT_API_STATUS(OrtCreateEnvWithCustomLogger, OrtLoggingFunction logging_function,
               _In_opt_ void* logger_param, OrtLoggingLevel default_logging_level,
               _In_ const char* logid,
               _Outptr_ OrtEnv** out);

// TODO: document the path separator convention? '/' vs '\'
// TODO: should specify the access characteristics of model_path. Is this read only during the
// execution of OrtCreateSession, or does the OrtSession retain a handle to the file/directory
// and continue to access throughout the OrtSession lifetime?
//  What sort of access is needed to model_path : read or read/write?
ORT_API_STATUS(OrtCreateSession, _In_ const OrtEnv* env, _In_ const ORTCHAR_T* model_path,
               _In_ const OrtSessionOptions* options, _Outptr_ OrtSession** out);

ORT_API_STATUS(OrtCreateSessionFromArray, _In_ const OrtEnv* env, _In_ const void* model_data, size_t model_data_length,
               _In_ const OrtSessionOptions* options, _Outptr_ OrtSession** out);

ORT_API_STATUS(OrtRun, _Inout_ OrtSession* sess,
               _In_opt_ const OrtRunOptions* run_options,
               _In_ const char* const* input_names, _In_ const OrtValue* const* input, size_t input_len,
               _In_ const char* const* output_names, size_t output_names_len, _Outptr_ OrtValue** out);

/**
 * \return A pointer of the newly created object. The pointer should be freed by OrtReleaseSessionOptions after use
 */
ORT_API_STATUS(OrtCreateSessionOptions, _Outptr_ OrtSessionOptions** options);

// Set filepath to save optimized model after graph level transformations.
ORT_API_STATUS(OrtSetOptimizedModelFilePath, _In_ OrtSessionOptions* options, _In_ const ORTCHAR_T* optimized_model_filepath);

// create a copy of an existing OrtSessionOptions
ORT_API_STATUS(OrtCloneSessionOptions, _In_ const OrtSessionOptions* in_options, _Outptr_ OrtSessionOptions** out_options);
ORT_API_STATUS(OrtEnableSequentialExecution, _Inout_ OrtSessionOptions* options);
ORT_API_STATUS(OrtDisableSequentialExecution, _Inout_ OrtSessionOptions* options);

// Enable profiling for this session.
ORT_API_STATUS(OrtEnableProfiling, _Inout_ OrtSessionOptions* options, _In_ const ORTCHAR_T* profile_file_prefix);
ORT_API_STATUS(OrtDisableProfiling, _Inout_ OrtSessionOptions* options);

// Enable the memory pattern optimization.
// The idea is if the input shapes are the same, we could trace the internal memory allocation
// and generate a memory pattern for future request. So next time we could just do one allocation
// with a big chunk for all the internal memory allocation.
// Note: memory pattern optimization is only available when SequentialExecution enabled.
ORT_API_STATUS(OrtEnableMemPattern, _Inout_ OrtSessionOptions* options);
ORT_API_STATUS(OrtDisableMemPattern, _Inout_ OrtSessionOptions* options);

// Enable the memory arena on CPU
// Arena may pre-allocate memory for future usage.
// set this option to false if you don't want it.
ORT_API_STATUS(OrtEnableCpuMemArena, _Inout_ OrtSessionOptions* options);
ORT_API_STATUS(OrtDisableCpuMemArena, _Inout_ OrtSessionOptions* options);

// < logger id to use for session output
ORT_API_STATUS(OrtSetSessionLogId, _Inout_ OrtSessionOptions* options, const char* logid);

// < applies to session load, initialization, etc
ORT_API_STATUS(OrtSetSessionLogVerbosityLevel, _Inout_ OrtSessionOptions* options, int session_log_verbosity_level);

// Set Graph optimization level.
// TODO Add documentation about which optimizations are enabled for each value.
typedef enum GraphOptimizationLevel {
  ORT_DISABLE_ALL = 0,
  ORT_ENABLE_BASIC = 1,
  ORT_ENABLE_EXTENDED = 2,
  ORT_ENABLE_ALL = 99
} GraphOptimizationLevel;
ORT_API_STATUS(OrtSetSessionGraphOptimizationLevel, _Inout_ OrtSessionOptions* options,
               GraphOptimizationLevel graph_optimization_level);

/**
 * How many threads in the session thread pool.
 * Set it to 0 to make onnxruntime run as single threaded.
 * \param session_thread_pool_size <0, let the runtime choose a default. =0, Don't create extra threads. 
 *                                 >0, create a thread pool with size of this value.
 */
ORT_API_STATUS(OrtSetSessionThreadPoolSize, _Inout_ OrtSessionOptions* options, int session_thread_pool_size);

/**
  * To use additional providers, you must build ORT with the extra providers enabled. Then call one of these
  * functions to enable them in the session:
  *   OrtSessionOptionsAppendExecutionProvider_CPU
  *   OrtSessionOptionsAppendExecutionProvider_CUDA
  *   OrtSessionOptionsAppendExecutionProvider_<remaining providers...>
  * The order they care called indicates the preference order as well. In other words call this method
  * on your most preferred execution provider first followed by the less preferred ones.
  * If none are called Ort will use its internal CPU execution provider.
  */

ORT_API_STATUS(OrtSessionGetInputCount, _In_ const OrtSession* sess, _Out_ size_t* out);
ORT_API_STATUS(OrtSessionGetOutputCount, _In_ const OrtSession* sess, _Out_ size_t* out);

/**
 * \param out  should be freed by OrtReleaseTypeInfo after use
 */
ORT_API_STATUS(OrtSessionGetInputTypeInfo, _In_ const OrtSession* sess, size_t index, _Outptr_ OrtTypeInfo** type_info);

/**
 * \param out  should be freed by OrtReleaseTypeInfo after use
 */
ORT_API_STATUS(OrtSessionGetOutputTypeInfo, _In_ const OrtSession* sess, size_t index, _Outptr_ OrtTypeInfo** type_info);

/**
 * \param value  is set to a null terminated string allocated using 'allocator'. The caller is responsible in freeing it.
 */
ORT_API_STATUS(OrtSessionGetInputName, _In_ const OrtSession* sess, size_t index,
               _Inout_ OrtAllocator* allocator, _Outptr_ char** value);
ORT_API_STATUS(OrtSessionGetOutputName, _In_ const OrtSession* sess, size_t index,
               _Inout_ OrtAllocator* allocator, _Outptr_ char** value);

/**
 * \return A pointer to the newly created object. The pointer should be freed by OrtReleaseRunOptions after use
 */
ORT_API_STATUS(OrtCreateRunOptions, _Outptr_ OrtRunOptions** out);

ORT_API_STATUS(OrtRunOptionsSetRunLogVerbosityLevel, _Inout_ OrtRunOptions* options, int value);
ORT_API_STATUS(OrtRunOptionsSetRunTag, _In_ OrtRunOptions*, _In_ const char* run_tag);

ORT_API_STATUS(OrtRunOptionsGetRunLogVerbosityLevel, _In_ const OrtRunOptions* options, _Out_ int* out);
ORT_API_STATUS(OrtRunOptionsGetRunTag, _In_ const OrtRunOptions*, _Out_ const char** out);

// Set a flag so that any running OrtRun* calls that are using this instance of OrtRunOptions
// will exit as soon as possible if the flag is true.
ORT_API_STATUS(OrtRunOptionsEnableTerminate, _Inout_ OrtRunOptions* options);
ORT_API_STATUS(OrtRunOptionsDisableTerminate, _Inout_ OrtRunOptions* options);

/**
 * Create a tensor from an allocator. OrtReleaseValue will also release the buffer inside the output value
 * \param out Should be freed by calling OrtReleaseValue
 * \param type must be one of TENSOR_ELEMENT_DATA_TYPE_xxxx
 */
ORT_API_STATUS(OrtCreateTensorAsOrtValue, _Inout_ OrtAllocator* allocator,
               _In_ const int64_t* shape, size_t shape_len, ONNXTensorElementDataType type,
               _Outptr_ OrtValue** out);

/**
 * Create a tensor with user's buffer. You can fill the buffer either before calling this function or after.
 * p_data is owned by caller. OrtReleaseValue won't release p_data.
 * \param out Should be freed by calling OrtReleaseValue
 */
ORT_API_STATUS(OrtCreateTensorWithDataAsOrtValue, _In_ const OrtAllocatorInfo* info,
               _Inout_ void* p_data, size_t p_data_len, _In_ const int64_t* shape, size_t shape_len,
               ONNXTensorElementDataType type, _Outptr_ OrtValue** out);

// This function doesn't work with string tensor
// this is a no-copy method whose pointer is only valid until the backing OrtValue is free'd.
ORT_API_STATUS(OrtGetTensorMutableData, _Inout_ OrtValue* value, _Outptr_ void** out);

/**
 * \Sets *out to 1 iff an OrtValue is a tensor, 0 otherwise
 */
ORT_API_STATUS(OrtIsTensor, _In_ const OrtValue* value, _Out_ int* out);

/**
 * \param value A tensor created from OrtCreateTensor... function.
 * \param s each A string array. Each string in this array must be null terminated.
 * \param s_len length of s
 */
ORT_API_STATUS(OrtFillStringTensor, _Inout_ OrtValue* value, _In_ const char* const* s, size_t s_len);
/**
 * \param value A tensor created from OrtCreateTensor... function.
 * \param len total data length, not including the trailing '\0' chars.
 */
ORT_API_STATUS(OrtGetStringTensorDataLength, _In_ const OrtValue* value, _Out_ size_t* len);

/**
 * \param s string contents. Each string is NOT null-terminated.
 * \param value A tensor created from OrtCreateTensor... function.
 * \param s_len total data length, get it from OrtGetStringTensorDataLength
 */
ORT_API_STATUS(OrtGetStringTensorContent, _In_ const OrtValue* value, _Out_ void* s, size_t s_len,
               _Out_ size_t* offsets, size_t offsets_len);

/**
 * Don't free the 'out' value
 */
ORT_API_STATUS(OrtCastTypeInfoToTensorInfo, _In_ const OrtTypeInfo*, _Out_ const OrtTensorTypeAndShapeInfo** out);

/**
 * Return OnnxType from OrtTypeInfo
 */
ORT_API_STATUS(OrtGetOnnxTypeFromTypeInfo, _In_ const OrtTypeInfo*, _Out_ enum ONNXType* out);

/**
 * The 'out' value should be released by calling OrtReleaseTensorTypeAndShapeInfo
 */
ORT_API_STATUS(OrtCreateTensorTypeAndShapeInfo, _Outptr_ OrtTensorTypeAndShapeInfo** out);

ORT_API_STATUS(OrtSetTensorElementType, _Inout_ OrtTensorTypeAndShapeInfo*, enum ONNXTensorElementDataType type);

/**
 * \param info Created from OrtCreateTensorTypeAndShapeInfo() function
 * \param dim_values An array with length of `dim_count`. Its elements can contain negative values.
 * \param dim_count length of dim_values
 */
ORT_API_STATUS(OrtSetDimensions, OrtTensorTypeAndShapeInfo* info, _In_ const int64_t* dim_values, size_t dim_count);

ORT_API_STATUS(OrtGetTensorElementType, _In_ const OrtTensorTypeAndShapeInfo*, _Out_ enum ONNXTensorElementDataType* out);
ORT_API_STATUS(OrtGetDimensionsCount, _In_ const OrtTensorTypeAndShapeInfo* info, _Out_ size_t* out);
ORT_API_STATUS(OrtGetDimensions, _In_ const OrtTensorTypeAndShapeInfo* info, _Out_ int64_t* dim_values, size_t dim_values_length);

/**
 * Return the number of elements specified by the tensor shape.
 * Return a negative value if unknown (i.e., any dimension is negative.)
 * e.g.
 * [] -> 1
 * [1,3,4] -> 12
 * [2,0,4] -> 0
 * [-1,3,4] -> -1
 */
ORT_API_STATUS(OrtGetTensorShapeElementCount, _In_ const OrtTensorTypeAndShapeInfo* info, _Out_ size_t* out);

/**
 * \param out Should be freed by OrtReleaseTensorTypeAndShapeInfo after use
 */
ORT_API_STATUS(OrtGetTensorTypeAndShape, _In_ const OrtValue* value, _Outptr_ OrtTensorTypeAndShapeInfo** out);

/**
 * Get the type information of an OrtValue
 * \param value
 * \param out The returned value should be freed by OrtReleaseTypeInfo after use
 */
ORT_API_STATUS(OrtGetTypeInfo, _In_ const OrtValue* value, _Outptr_ OrtTypeInfo** out);

ORT_API_STATUS(OrtGetValueType, _In_ const OrtValue* value, _Out_ enum ONNXType* out);

typedef enum OrtAllocatorType {
  OrtDeviceAllocator = 0,
  OrtArenaAllocator = 1
} OrtAllocatorType;

/**
 * memory types for allocator, exec provider specific types should be extended in each provider
 * Whenever this struct is updated, please also update the MakeKey function in onnxruntime/core/framework/execution_provider.cc
*/
typedef enum OrtMemType {
  OrtMemTypeCPUInput = -2,              // Any CPU memory used by non-CPU execution provider
  OrtMemTypeCPUOutput = -1,             // CPU accessible memory outputted by non-CPU execution provider, i.e. CUDA_PINNED
  OrtMemTypeCPU = OrtMemTypeCPUOutput,  // temporary CPU accessible memory allocated by non-CPU execution provider, i.e. CUDA_PINNED
  OrtMemTypeDefault = 0,                // the default allocator for execution provider
} OrtMemType;

ORT_API_STATUS(OrtCreateAllocatorInfo, _In_ const char* name1, enum OrtAllocatorType type, int id1, enum OrtMemType mem_type1, _Outptr_ OrtAllocatorInfo** out);

/**
 * Convenience function for special case of OrtCreateAllocatorInfo, for the CPU allocator. Uses name = "Cpu" and id = 0.
 */
ORT_API_STATUS(OrtCreateCpuAllocatorInfo, enum OrtAllocatorType type, enum OrtMemType mem_type1, _Outptr_ OrtAllocatorInfo** out)
ORT_ALL_ARGS_NONNULL;

/**
 * Test if two allocation info are equal
 * \Sets 'out' to 0 if equal, -1 if not equal
 */
ORT_API_STATUS(OrtCompareAllocatorInfo, _In_ const OrtAllocatorInfo* info1, _In_ const OrtAllocatorInfo* info2, _Out_ int* out)
ORT_ALL_ARGS_NONNULL;

/**
 * Do not free the returned value
 */
ORT_API_STATUS(OrtAllocatorInfoGetName, _In_ const OrtAllocatorInfo* ptr, _Out_ const char** out);
ORT_API_STATUS(OrtAllocatorInfoGetId, _In_ const OrtAllocatorInfo* ptr, _Out_ int* out);
ORT_API_STATUS(OrtAllocatorInfoGetMemType, _In_ const OrtAllocatorInfo* ptr, _Out_ OrtMemType* out);
ORT_API_STATUS(OrtAllocatorInfoGetType, _In_ const OrtAllocatorInfo* ptr, _Out_ OrtAllocatorType* out);

ORT_API_STATUS(OrtAllocatorAlloc, _Inout_ OrtAllocator* ptr, size_t size, _Outptr_ void** out);
ORT_API_STATUS(OrtAllocatorFree, _Inout_ OrtAllocator* ptr, void* p);
ORT_API_STATUS(OrtAllocatorGetInfo, _In_ const OrtAllocator* ptr, _Out_ const OrtAllocatorInfo** out);

ORT_API_STATUS(OrtCreateDefaultAllocator, _Outptr_ OrtAllocator** out);

ORT_API(const char*, OrtGetVersionString);
/**
 * \param msg A null-terminated string. Its content will be copied into the newly created OrtStatus
 */
ORT_EXPORT _Check_return_ _Ret_notnull_ OrtStatus* ORT_API_CALL OrtCreateStatus(OrtErrorCode code, _In_ const char* msg) NO_EXCEPTION
    ORT_ALL_ARGS_NONNULL;

ORT_API(OrtErrorCode, OrtGetErrorCode, _In_ const OrtStatus* status)
ORT_ALL_ARGS_NONNULL;
/**
 * \param status must not be NULL
 * \return The error message inside the `status`. Do not free the returned value.
 */
ORT_API(const char*, OrtGetErrorMessage, _In_ const OrtStatus* status)
ORT_ALL_ARGS_NONNULL;

/**
   * APIs to support non-tensor types - map and sequence.
   * Currently only the following types are supported
   * Note: the following types should be kept in sync with data_types.h
   * Map types
   * =========
   * std::map<std::string, std::string>
   * std::map<std::string, int64_t>
   * std::map<std::string, float>
   * std::map<std::string, double>
   * std::map<int64_t, std::string>
   * std::map<int64_t, int64_t>
   * std::map<int64_t, float>
   * std::map<int64_t, double>
   * 
   * Sequence types
   * ==============
   * std::vector<std::string>
   * std::vector<int64_t>
   * std::vector<float>
   * std::vector<double>
   * std::vector<std::map<std::string, float>>
   * std::vector<std::map<int64_t, float>
   */

/**
   * If input OrtValue represents a map, you need to retrieve the keys and values
   * separately. Use index=0 to retrieve keys and index=1 to retrieve values.
   * If input OrtValue represents a sequence, use index to retrieve the index'th element
   * of the sequence.
   */
ORT_API_STATUS(OrtGetValue, _In_ const OrtValue* value, int index, _Inout_ OrtAllocator* allocator, _Outptr_ OrtValue** out);

/**
   * Returns 2 for type map and N for sequence where N is the number of elements
   * in the sequence.
   */
ORT_API_STATUS(OrtGetValueCount, _In_ const OrtValue* value, _Out_ size_t* out);

/**
   * To construct a map, use num_values = 2 and 'in' should be an arrary of 2 OrtValues
   * representing keys and values.
   * To construct a sequence, use num_values = N where N is the number of the elements in the
   * sequence. 'in' should be an arrary of N OrtValues.
   * \value_type should be either map or sequence.
   */
ORT_API_STATUS(OrtCreateValue, _In_ const OrtValue* const* in, size_t num_values, enum ONNXType value_type,
               _Outptr_ OrtValue** out);

/*
 * EXPERIMENTAL APIS - Subject to change. Released as a preview to get feedback and enable early testing
*/

/*
 * Steps to use a custom op:
 *   1 Create an OrtCustomOpDomain with the domain name used by the custom ops
 *   2 Create an OrtCustomOp structure for each op and add them to the domain
 *   3 Call OrtAddCustomOpDomain to add the custom domain of ops to the session options
*/
struct OrtKernelInfo;
typedef struct OrtKernelInfo OrtKernelInfo;
struct OrtKernelContext;
typedef struct OrtKernelContext OrtKernelContext;

struct OrtCustomOpApi {
  /*
   * These allow reading node attributes during kernel creation
  */
  OrtStatus*(ORT_API_CALL* KernelInfoGetAttribute_float)(_In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ float* out);
  OrtStatus*(ORT_API_CALL* KernelInfoGetAttribute_int64)(_In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ int64_t* out);
  OrtStatus*(ORT_API_CALL* KernelInfoGetAttribute_string)(_In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ char* out, _Inout_ size_t* size);

  OrtStatus*(ORT_API_CALL* GetTensorTypeAndShape)(_In_ const OrtValue* value, _Outptr_ OrtTensorTypeAndShapeInfo** out);

  OrtStatus*(ORT_API_CALL* GetTensorShapeElementCount)(_In_ const OrtTensorTypeAndShapeInfo* info, _Out_ size_t* out);
  OrtStatus*(ORT_API_CALL* GetTensorElementType)(_In_ const OrtTensorTypeAndShapeInfo*, _Out_ enum ONNXTensorElementDataType* out);

  OrtStatus*(ORT_API_CALL* GetDimensionCount)(_In_ const OrtTensorTypeAndShapeInfo* info, _Out_ size_t* out);
  OrtStatus*(ORT_API_CALL* GetDimensions)(_In_ const OrtTensorTypeAndShapeInfo* info, _Out_ int64_t* dim_values, size_t dim_values_length);
  OrtStatus*(ORT_API_CALL* SetDimensions)(OrtTensorTypeAndShapeInfo* info, _In_ const int64_t* dim_values, size_t dim_count);
  OrtStatus*(ORT_API_CALL* GetTensorMutableData)(_Inout_ OrtValue* value, _Outptr_ void** data);

  void(ORT_API_CALL* ReleaseTensorTypeAndShapeInfo)(_In_ OrtTensorTypeAndShapeInfo* input);

  OrtStatus*(ORT_API_CALL* KernelContext_GetInputCount)(_In_ const OrtKernelContext* context, _Out_ size_t* out);
  OrtStatus*(ORT_API_CALL* KernelContext_GetInput)(_In_ const OrtKernelContext* context, _In_ size_t index, _Out_ const OrtValue** out);
  OrtStatus*(ORT_API_CALL* KernelContext_GetOutputCount)(_In_ const OrtKernelContext* context, _Out_ size_t* out);
  OrtStatus*(ORT_API_CALL* KernelContext_GetOutput)(_Inout_ OrtKernelContext* context, _In_ size_t index, _In_ const int64_t* dim_values, size_t dim_count, _Outptr_ OrtValue** out);
};
typedef struct OrtCustomOpApi OrtCustomOpApi;

/*
 * The OrtCustomOp structure defines a custom op's schema and its kernel callbacks. The callbacks are filled in by
 * the implementor of the custom op.
*/
struct OrtCustomOp {
  uint32_t version;  // Initialize to ORT_API_VERSION

  // This callback creates the kernel, which is a user defined parameter that is passed to the Kernel* callbacks below.
  void*(ORT_API_CALL* CreateKernel)(_In_ struct OrtCustomOp* op, _In_ const OrtCustomOpApi* api, _In_ const OrtKernelInfo* info);

  // Returns the name of the op
  const char*(ORT_API_CALL* GetName)(_In_ struct OrtCustomOp* op);

  // Returns the type of the execution provider, return nullptr to use CPU execution provider
  const char*(ORT_API_CALL* GetExecutionProviderType)(_In_ struct OrtCustomOp* op);

  // Returns the count and types of the input & output tensors
  ONNXTensorElementDataType(ORT_API_CALL* GetInputType)(_In_ struct OrtCustomOp* op, _In_ size_t index);
  size_t(ORT_API_CALL* GetInputTypeCount)(_In_ struct OrtCustomOp* op);
  ONNXTensorElementDataType(ORT_API_CALL* GetOutputType)(_In_ struct OrtCustomOp* op, _In_ size_t index);
  size_t(ORT_API_CALL* GetOutputTypeCount)(_In_ struct OrtCustomOp* op);

  // Op kernel callbacks
  void(ORT_API_CALL* KernelCompute)(_In_ void* op_kernel, _In_ OrtKernelContext* context);
  void(ORT_API_CALL* KernelDestroy)(_In_ void* op_kernel);
};
typedef struct OrtCustomOp OrtCustomOp;

/*
* Create a custom op domain. After all sessions using it are released, call OrtReleaseCustomOpDomain
*/
ORT_API_STATUS(OrtCreateCustomOpDomain, _In_ const char* domain, _Outptr_ OrtCustomOpDomain** out);

/*
 * Add custom ops to the OrtCustomOpDomain
 *  Note: The OrtCustomOp* pointer must remain valid until the OrtCustomOpDomain using it is released
*/
ORT_API_STATUS(OrtCustomOpDomain_Add, _Inout_ OrtCustomOpDomain* custom_op_domain, _In_ OrtCustomOp* op);

/*
 * Add a custom op domain to the OrtSessionOptions
 *  Note: The OrtCustomOpDomain* must not be deleted until the sessions using it are released
*/
ORT_API_STATUS(OrtAddCustomOpDomain, _Inout_ OrtSessionOptions* options, _In_ OrtCustomOpDomain* custom_op_domain);
/*
 * END EXPERIMENTAL
*/

#ifdef __cplusplus
}
#endif
