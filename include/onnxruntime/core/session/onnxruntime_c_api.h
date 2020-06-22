// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// This value is used in structures passed to ORT so that a newer version of ORT will still work with them
#define ORT_API_VERSION 4

#ifdef __cplusplus
extern "C" {
#endif

// SAL2 Definitions
#ifndef _WIN32
#define _In_
#define _In_z_
#define _In_opt_
#define _In_opt_z_
#define _Out_
#define _Outptr_
#define _Out_opt_
#define _Inout_
#define _Inout_opt_
#define _Frees_ptr_opt_
#define _Ret_maybenull_
#define _Ret_notnull_
#define _Check_return_
#define _Outptr_result_maybenull_
#define _In_reads_(X)
#define _Inout_updates_all_(X)
#define _Out_writes_bytes_all_(X)
#define _Out_writes_all_(X)
#define _Success_(X)
#define _Outptr_result_buffer_maybenull_(X)
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
#define ORT_TSTR(X) X
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
  ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16     // Non-IEEE floating-point format based on IEEE754 single-precision
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
  ORT_EP_FAIL,
} OrtErrorCode;

#define ORT_RUNTIME_CLASS(X) \
  struct Ort##X;             \
  typedef struct Ort##X Ort##X;

// The actual types defined have an Ort prefix
ORT_RUNTIME_CLASS(Env);
ORT_RUNTIME_CLASS(Status);  // nullptr for Status* indicates success
ORT_RUNTIME_CLASS(MemoryInfo);
ORT_RUNTIME_CLASS(Session);  //Don't call OrtReleaseSession from Dllmain (because session owns a thread pool)
ORT_RUNTIME_CLASS(Value);
ORT_RUNTIME_CLASS(RunOptions);
ORT_RUNTIME_CLASS(TypeInfo);
ORT_RUNTIME_CLASS(TensorTypeAndShapeInfo);
ORT_RUNTIME_CLASS(SessionOptions);
ORT_RUNTIME_CLASS(CustomOpDomain);
ORT_RUNTIME_CLASS(MapTypeInfo);
ORT_RUNTIME_CLASS(SequenceTypeInfo);
ORT_RUNTIME_CLASS(ModelMetadata);
ORT_RUNTIME_CLASS(ThreadPoolParams);
ORT_RUNTIME_CLASS(ThreadingOptions);

#ifdef _WIN32
typedef _Return_type_success_(return == 0) OrtStatus* OrtStatusPtr;
#else
typedef OrtStatus* OrtStatusPtr;
#endif

// __VA_ARGS__ on Windows and Linux are different
#define ORT_API(RETURN_TYPE, NAME, ...) ORT_EXPORT RETURN_TYPE ORT_API_CALL NAME(__VA_ARGS__) NO_EXCEPTION

#define ORT_API_STATUS(NAME, ...) \
  ORT_EXPORT _Check_return_ _Ret_maybenull_ OrtStatusPtr ORT_API_CALL NAME(__VA_ARGS__) NO_EXCEPTION ORT_MUST_USE_RESULT

// XXX: Unfortunately, SAL annotations are known to not work with function pointers
#define ORT_API2_STATUS(NAME, ...) \
  _Check_return_ _Ret_maybenull_ OrtStatusPtr(ORT_API_CALL* NAME)(__VA_ARGS__) NO_EXCEPTION ORT_MUST_USE_RESULT

// Used in *.cc files. Almost as same as ORT_API_STATUS, except without ORT_MUST_USE_RESULT and ORT_EXPORT
#define ORT_API_STATUS_IMPL(NAME, ...) \
  _Check_return_ _Ret_maybenull_ OrtStatusPtr ORT_API_CALL NAME(__VA_ARGS__) NO_EXCEPTION

#define ORT_CLASS_RELEASE(X) void(ORT_API_CALL * Release##X)(_Frees_ptr_opt_ Ort##X * input)

// When passing in an allocator to any ORT function, be sure that the allocator object
// is not destroyed until the last allocated object using it is freed.
typedef struct OrtAllocator {
  uint32_t version;  // Initialize to ORT_API_VERSION
  void*(ORT_API_CALL* Alloc)(struct OrtAllocator* this_, size_t size);
  void(ORT_API_CALL* Free)(struct OrtAllocator* this_, void* p);
  const struct OrtMemoryInfo*(ORT_API_CALL* Info)(const struct OrtAllocator* this_);
} OrtAllocator;

typedef void(ORT_API_CALL* OrtLoggingFunction)(
    void* param, OrtLoggingLevel severity, const char* category, const char* logid, const char* code_location,
    const char* message);

// Set Graph optimization level.
// Refer https://github.com/microsoft/onnxruntime/blob/master/docs/ONNX_Runtime_Graph_Optimizations.md
// for in-depth undersrtanding of Graph Optimizations in ORT
typedef enum GraphOptimizationLevel {
  ORT_DISABLE_ALL = 0,
  ORT_ENABLE_BASIC = 1,
  ORT_ENABLE_EXTENDED = 2,
  ORT_ENABLE_ALL = 99
} GraphOptimizationLevel;

typedef enum ExecutionMode {
  ORT_SEQUENTIAL = 0,
  ORT_PARALLEL = 1,
} ExecutionMode;

struct OrtKernelInfo;
typedef struct OrtKernelInfo OrtKernelInfo;
struct OrtKernelContext;
typedef struct OrtKernelContext OrtKernelContext;
struct OrtCustomOp;
typedef struct OrtCustomOp OrtCustomOp;

typedef enum OrtAllocatorType {
  Invalid = -1,
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

struct OrtApi;
typedef struct OrtApi OrtApi;

struct OrtApiBase {
  const OrtApi*(ORT_API_CALL* GetApi)(uint32_t version)NO_EXCEPTION;  // Pass in ORT_API_VERSION
  // nullptr will be returned if the version is unsupported, for example when using a runtime older than this header file

  const char*(ORT_API_CALL* GetVersionString)() NO_EXCEPTION;
};
typedef struct OrtApiBase OrtApiBase;

ORT_EXPORT const OrtApiBase* ORT_API_CALL OrtGetApiBase(void) NO_EXCEPTION;

struct OrtApi {
  /**
* \param msg A null-terminated string. Its content will be copied into the newly created OrtStatus
*/
  OrtStatus*(ORT_API_CALL* CreateStatus)(OrtErrorCode code, _In_ const char* msg)NO_EXCEPTION ORT_ALL_ARGS_NONNULL;

  OrtErrorCode(ORT_API_CALL* GetErrorCode)(_In_ const OrtStatus* status) NO_EXCEPTION ORT_ALL_ARGS_NONNULL;

  /**
 * \param status must not be NULL
 * \return The error message inside the `status`. Do not free the returned value.
 */
  const char*(ORT_API_CALL* GetErrorMessage)(_In_ const OrtStatus* status)NO_EXCEPTION ORT_ALL_ARGS_NONNULL;

  /**
     * \param out Should be freed by `OrtReleaseEnv` after use
     */
  ORT_API2_STATUS(CreateEnv, OrtLoggingLevel default_logging_level, _In_ const char* logid, _Outptr_ OrtEnv** out);

  /**
   * \param out Should be freed by `OrtReleaseEnv` after use
   */
  ORT_API2_STATUS(CreateEnvWithCustomLogger, OrtLoggingFunction logging_function, _In_opt_ void* logger_param,
                  OrtLoggingLevel default_warning_level, _In_ const char* logid, _Outptr_ OrtEnv** out);

  // Platform telemetry events are on by default since they are lightweight.  You can manually turn them off.
  ORT_API2_STATUS(EnableTelemetryEvents, _In_ const OrtEnv* env);
  ORT_API2_STATUS(DisableTelemetryEvents, _In_ const OrtEnv* env);

  // TODO: document the path separator convention? '/' vs '\'
  // TODO: should specify the access characteristics of model_path. Is this read only during the
  // execution of CreateSession, or does the OrtSession retain a handle to the file/directory
  // and continue to access throughout the OrtSession lifetime?
  //  What sort of access is needed to model_path : read or read/write?
  ORT_API2_STATUS(CreateSession, _In_ const OrtEnv* env, _In_ const ORTCHAR_T* model_path,
                  _In_ const OrtSessionOptions* options, _Outptr_ OrtSession** out);

  ORT_API2_STATUS(CreateSessionFromArray, _In_ const OrtEnv* env, _In_ const void* model_data, size_t model_data_length,
                  _In_ const OrtSessionOptions* options, _Outptr_ OrtSession** out);

  ORT_API2_STATUS(Run, _Inout_ OrtSession* sess, _In_opt_ const OrtRunOptions* run_options,
                  _In_reads_(input_len) const char* const* input_names,
                  _In_reads_(input_len) const OrtValue* const* input, size_t input_len,
                  _In_reads_(output_names_len) const char* const* output_names1, size_t output_names_len,
                  _Inout_updates_all_(output_names_len) OrtValue** output);

  /**
    * \return A pointer of the newly created object. The pointer should be freed by OrtReleaseSessionOptions after use
    */
  ORT_API2_STATUS(CreateSessionOptions, _Outptr_ OrtSessionOptions** options);

  // Set filepath to save optimized model after graph level transformations.
  ORT_API2_STATUS(SetOptimizedModelFilePath, _Inout_ OrtSessionOptions* options,
                  _In_ const ORTCHAR_T* optimized_model_filepath);

  // create a copy of an existing OrtSessionOptions
  ORT_API2_STATUS(CloneSessionOptions, _In_ const OrtSessionOptions* in_options,
                  _Outptr_ OrtSessionOptions** out_options);

  // Controls whether you want to execute operators in your graph sequentially or in parallel. Usually when the model
  // has many branches, setting this option to ExecutionMode.ORT_PARALLEL will give you better performance.
  // See [docs/ONNX_Runtime_Perf_Tuning.md] for more details.
  ORT_API2_STATUS(SetSessionExecutionMode, _Inout_ OrtSessionOptions* options, ExecutionMode execution_mode);

  // Enable profiling for this session.
  ORT_API2_STATUS(EnableProfiling, _Inout_ OrtSessionOptions* options, _In_ const ORTCHAR_T* profile_file_prefix);
  ORT_API2_STATUS(DisableProfiling, _Inout_ OrtSessionOptions* options);

  // Enable the memory pattern optimization.
  // The idea is if the input shapes are the same, we could trace the internal memory allocation
  // and generate a memory pattern for future request. So next time we could just do one allocation
  // with a big chunk for all the internal memory allocation.
  // Note: memory pattern optimization is only available when SequentialExecution enabled.
  ORT_API2_STATUS(EnableMemPattern, _Inout_ OrtSessionOptions* options);
  ORT_API2_STATUS(DisableMemPattern, _Inout_ OrtSessionOptions* options);

  // Enable the memory arena on CPU
  // Arena may pre-allocate memory for future usage.
  // set this option to false if you don't want it.
  ORT_API2_STATUS(EnableCpuMemArena, _Inout_ OrtSessionOptions* options);
  ORT_API2_STATUS(DisableCpuMemArena, _Inout_ OrtSessionOptions* options);

  // < logger id to use for session output
  ORT_API2_STATUS(SetSessionLogId, _Inout_ OrtSessionOptions* options, const char* logid);

  // < applies to session load, initialization, etc
  ORT_API2_STATUS(SetSessionLogVerbosityLevel, _Inout_ OrtSessionOptions* options, int session_log_verbosity_level);
  ORT_API2_STATUS(SetSessionLogSeverityLevel, _Inout_ OrtSessionOptions* options, int session_log_severity_level);

  ORT_API2_STATUS(SetSessionGraphOptimizationLevel, _Inout_ OrtSessionOptions* options,
                  GraphOptimizationLevel graph_optimization_level);

  // Sets the number of threads used to parallelize the execution within nodes
  // A value of 0 means ORT will pick a default
  // Note: If you've built ORT with OpenMP, this API has no effect on the number of threads used. In this case
  // use the OpenMP env variables to configure the number of intra op num threads.
  ORT_API2_STATUS(SetIntraOpNumThreads, _Inout_ OrtSessionOptions* options, int intra_op_num_threads);

  // Sets the number of threads used to parallelize the execution of the graph (across nodes)
  // If sequential execution is enabled this value is ignored
  // A value of 0 means ORT will pick a default
  ORT_API2_STATUS(SetInterOpNumThreads, _Inout_ OrtSessionOptions* options, int inter_op_num_threads);

  /*
  Create a custom op domain. After all sessions using it are released, call OrtReleaseCustomOpDomain
  */
  ORT_API2_STATUS(CreateCustomOpDomain, _In_ const char* domain, _Outptr_ OrtCustomOpDomain** out);

  /*
     * Add custom ops to the OrtCustomOpDomain
     *  Note: The OrtCustomOp* pointer must remain valid until the OrtCustomOpDomain using it is released
    */
  ORT_API2_STATUS(CustomOpDomain_Add, _Inout_ OrtCustomOpDomain* custom_op_domain, _In_ OrtCustomOp* op);

  /*
     * Add a custom op domain to the OrtSessionOptions
     *  Note: The OrtCustomOpDomain* must not be deleted until the sessions using it are released
    */
  ORT_API2_STATUS(AddCustomOpDomain, _Inout_ OrtSessionOptions* options, _In_ OrtCustomOpDomain* custom_op_domain);

  /*
     * Loads a DLL named 'library_path' and looks for this entry point:
     *		OrtStatus* RegisterCustomOps(OrtSessionOptions * options, const OrtApiBase* api);
     * It then passes in the provided session options to this function along with the api base.
     * The handle to the loaded library is returned in library_handle. It can be freed by the caller after all sessions using the passed in
     * session options are destroyed, or if an error occurs and it is non null.
  */
  ORT_API2_STATUS(RegisterCustomOpsLibrary, _Inout_ OrtSessionOptions* options, _In_ const char* library_path,
                  void** library_handle);

  /**
    * To use additional providers, you must build ORT with the extra providers enabled. Then call one of these
    * functions to enable them in the session:
    *   OrtSessionOptionsAppendExecutionProvider_CPU
    *   OrtSessionOptionsAppendExecutionProvider_CUDA
    *   OrtSessionOptionsAppendExecutionProvider_<remaining providers...>
    * The order they are called indicates the preference order as well. In other words call this method
    * on your most preferred execution provider first followed by the less preferred ones.
    * If none are called Ort will use its internal CPU execution provider.
    */

  ORT_API2_STATUS(SessionGetInputCount, _In_ const OrtSession* sess, _Out_ size_t* out);
  ORT_API2_STATUS(SessionGetOutputCount, _In_ const OrtSession* sess, _Out_ size_t* out);
  ORT_API2_STATUS(SessionGetOverridableInitializerCount, _In_ const OrtSession* sess, _Out_ size_t* out);

  /**
   * \param out  should be freed by OrtReleaseTypeInfo after use
   */
  ORT_API2_STATUS(SessionGetInputTypeInfo, _In_ const OrtSession* sess, size_t index, _Outptr_ OrtTypeInfo** type_info);

  /**
   * \param out  should be freed by OrtReleaseTypeInfo after use
   */
  ORT_API2_STATUS(SessionGetOutputTypeInfo, _In_ const OrtSession* sess, size_t index,
                  _Outptr_ OrtTypeInfo** type_info);

  /**
 * \param out  should be freed by OrtReleaseTypeInfo after use
 */
  ORT_API2_STATUS(SessionGetOverridableInitializerTypeInfo, _In_ const OrtSession* sess, size_t index,
                  _Outptr_ OrtTypeInfo** type_info);

  /**
   * \param value  is set to a null terminated string allocated using 'allocator'. The caller is responsible for freeing it.
   */
  ORT_API2_STATUS(SessionGetInputName, _In_ const OrtSession* sess, size_t index, _Inout_ OrtAllocator* allocator,
                  _Outptr_ char** value);
  ORT_API2_STATUS(SessionGetOutputName, _In_ const OrtSession* sess, size_t index, _Inout_ OrtAllocator* allocator,
                  _Outptr_ char** value);
  ORT_API2_STATUS(SessionGetOverridableInitializerName, _In_ const OrtSession* sess, size_t index,
                  _Inout_ OrtAllocator* allocator, _Outptr_ char** value);

  /**
   * \return A pointer to the newly created object. The pointer should be freed by OrtReleaseRunOptions after use
   */
  ORT_API2_STATUS(CreateRunOptions, _Outptr_ OrtRunOptions** out);

  ORT_API2_STATUS(RunOptionsSetRunLogVerbosityLevel, _Inout_ OrtRunOptions* options, int value);
  ORT_API2_STATUS(RunOptionsSetRunLogSeverityLevel, _Inout_ OrtRunOptions* options, int value);
  ORT_API2_STATUS(RunOptionsSetRunTag, _Inout_ OrtRunOptions*, _In_ const char* run_tag);

  ORT_API2_STATUS(RunOptionsGetRunLogVerbosityLevel, _In_ const OrtRunOptions* options, _Out_ int* out);
  ORT_API2_STATUS(RunOptionsGetRunLogSeverityLevel, _In_ const OrtRunOptions* options, _Out_ int* out);
  ORT_API2_STATUS(RunOptionsGetRunTag, _In_ const OrtRunOptions*, _Out_ const char** out);

  // Set a flag so that ALL incomplete OrtRun calls that are using this instance of OrtRunOptions
  // will exit as soon as possible.
  ORT_API2_STATUS(RunOptionsSetTerminate, _Inout_ OrtRunOptions* options);
  // Unset the terminate flag to enable this OrtRunOptions instance being used in new OrtRun calls.
  ORT_API2_STATUS(RunOptionsUnsetTerminate, _Inout_ OrtRunOptions* options);

  /**
   * Create a tensor from an allocator. OrtReleaseValue will also release the buffer inside the output value
   * \param out Should be freed by calling OrtReleaseValue
   * \param type must be one of TENSOR_ELEMENT_DATA_TYPE_xxxx
   */
  ORT_API2_STATUS(CreateTensorAsOrtValue, _Inout_ OrtAllocator* allocator, _In_ const int64_t* shape, size_t shape_len,
                  ONNXTensorElementDataType type, _Outptr_ OrtValue** out);

  /**
   * Create a tensor with user's buffer. You can fill the buffer either before calling this function or after.
   * p_data is owned by caller. OrtReleaseValue won't release p_data.
   * \param out Should be freed by calling OrtReleaseValue
   */
  ORT_API2_STATUS(CreateTensorWithDataAsOrtValue, _In_ const OrtMemoryInfo* info, _Inout_ void* p_data,
                  size_t p_data_len, _In_ const int64_t* shape, size_t shape_len, ONNXTensorElementDataType type,
                  _Outptr_ OrtValue** out);

  /**
   * \Sets *out to 1 iff an OrtValue is a tensor, 0 otherwise
   */
  ORT_API2_STATUS(IsTensor, _In_ const OrtValue* value, _Out_ int* out);

  // This function doesn't work with string tensor
  // this is a no-copy method whose pointer is only valid until the backing OrtValue is free'd.
  ORT_API2_STATUS(GetTensorMutableData, _Inout_ OrtValue* value, _Outptr_ void** out);

  /**
     * \param value A tensor created from OrtCreateTensor... function.
     * \param s each A string array. Each string in this array must be null terminated.
     * \param s_len length of s
     */
  ORT_API2_STATUS(FillStringTensor, _Inout_ OrtValue* value, _In_ const char* const* s, size_t s_len);

  /**
     * \param value A tensor created from OrtCreateTensor... function.
     * \param len total data length, not including the trailing '\0' chars.
     */
  ORT_API2_STATUS(GetStringTensorDataLength, _In_ const OrtValue* value, _Out_ size_t* len);

  /**
     * \param s string contents. Each string is NOT null-terminated.
     * \param value A tensor created from OrtCreateTensor... function.
     * \param s_len total data length, get it from OrtGetStringTensorDataLength
     */
  ORT_API2_STATUS(GetStringTensorContent, _In_ const OrtValue* value, _Out_writes_bytes_all_(s_len) void* s,
                  size_t s_len, _Out_writes_all_(offsets_len) size_t* offsets, size_t offsets_len);

  /**
     * Don't free the 'out' value
     */
  ORT_API2_STATUS(CastTypeInfoToTensorInfo, _In_ const OrtTypeInfo*,
                  _Outptr_result_maybenull_ const OrtTensorTypeAndShapeInfo** out);

  /**
     * Return OnnxType from OrtTypeInfo
     */
  ORT_API2_STATUS(GetOnnxTypeFromTypeInfo, _In_ const OrtTypeInfo*, _Out_ enum ONNXType* out);

  /**
     * The 'out' value should be released by calling OrtReleaseTensorTypeAndShapeInfo
     */
  ORT_API2_STATUS(CreateTensorTypeAndShapeInfo, _Outptr_ OrtTensorTypeAndShapeInfo** out);

  ORT_API2_STATUS(SetTensorElementType, _Inout_ OrtTensorTypeAndShapeInfo*, enum ONNXTensorElementDataType type);

  /**
 * \param info Created from CreateTensorTypeAndShapeInfo() function
 * \param dim_values An array with length of `dim_count`. Its elements can contain negative values.
 * \param dim_count length of dim_values
 */
  ORT_API2_STATUS(SetDimensions, OrtTensorTypeAndShapeInfo* info, _In_ const int64_t* dim_values, size_t dim_count);

  ORT_API2_STATUS(GetTensorElementType, _In_ const OrtTensorTypeAndShapeInfo*,
                  _Out_ enum ONNXTensorElementDataType* out);
  ORT_API2_STATUS(GetDimensionsCount, _In_ const OrtTensorTypeAndShapeInfo* info, _Out_ size_t* out);
  ORT_API2_STATUS(GetDimensions, _In_ const OrtTensorTypeAndShapeInfo* info, _Out_ int64_t* dim_values,
                  size_t dim_values_length);
  ORT_API2_STATUS(GetSymbolicDimensions, _In_ const OrtTensorTypeAndShapeInfo* info,
                  _Out_writes_all_(dim_params_length) const char* dim_params[], size_t dim_params_length);

  /**
 * Return the number of elements specified by the tensor shape.
 * Return a negative value if unknown (i.e., any dimension is negative.)
 * e.g.
 * [] -> 1
 * [1,3,4] -> 12
 * [2,0,4] -> 0
 * [-1,3,4] -> -1
 */
  ORT_API2_STATUS(GetTensorShapeElementCount, _In_ const OrtTensorTypeAndShapeInfo* info, _Out_ size_t* out);

  /**
 * \param out Should be freed by OrtReleaseTensorTypeAndShapeInfo after use
 */
  ORT_API2_STATUS(GetTensorTypeAndShape, _In_ const OrtValue* value, _Outptr_ OrtTensorTypeAndShapeInfo** out);

  /**
 * Get the type information of an OrtValue
 * \param value
 * \param out The returned value should be freed by OrtReleaseTypeInfo after use
 */
  ORT_API2_STATUS(GetTypeInfo, _In_ const OrtValue* value, _Outptr_result_maybenull_ OrtTypeInfo** out);

  ORT_API2_STATUS(GetValueType, _In_ const OrtValue* value, _Out_ enum ONNXType* out);

  ORT_API2_STATUS(CreateMemoryInfo, _In_ const char* name1, enum OrtAllocatorType type, int id1,
                  enum OrtMemType mem_type1, _Outptr_ OrtMemoryInfo** out);

  /**
 * Convenience function for special case of CreateMemoryInfo, for the CPU allocator. Uses name = "Cpu" and id = 0.
 */
  ORT_API2_STATUS(CreateCpuMemoryInfo, enum OrtAllocatorType type, enum OrtMemType mem_type1,
                  _Outptr_ OrtMemoryInfo** out);

  /**
 * Test if two memory info are equal
 * \Sets 'out' to 0 if equal, -1 if not equal
 */
  ORT_API2_STATUS(CompareMemoryInfo, _In_ const OrtMemoryInfo* info1, _In_ const OrtMemoryInfo* info2, _Out_ int* out);

  /**
 * Do not free the returned value
 */
  ORT_API2_STATUS(MemoryInfoGetName, _In_ const OrtMemoryInfo* ptr, _Out_ const char** out);
  ORT_API2_STATUS(MemoryInfoGetId, _In_ const OrtMemoryInfo* ptr, _Out_ int* out);
  ORT_API2_STATUS(MemoryInfoGetMemType, _In_ const OrtMemoryInfo* ptr, _Out_ OrtMemType* out);
  ORT_API2_STATUS(MemoryInfoGetType, _In_ const OrtMemoryInfo* ptr, _Out_ OrtAllocatorType* out);

  ORT_API2_STATUS(AllocatorAlloc, _Inout_ OrtAllocator* ptr, size_t size, _Outptr_ void** out);
  ORT_API2_STATUS(AllocatorFree, _Inout_ OrtAllocator* ptr, void* p);
  ORT_API2_STATUS(AllocatorGetInfo, _In_ const OrtAllocator* ptr, _Outptr_ const struct OrtMemoryInfo** out);

  // The returned pointer doesn't have to be freed.
  // Always returns the same instance on every invocation.
  ORT_API2_STATUS(GetAllocatorWithDefaultOptions, _Outptr_ OrtAllocator** out);

  // Override symbolic dimensions (by specific denotation strings) with actual values if known at session initialization time to enable
  // optimizations that can take advantage of fixed values (such as memory planning, etc)
  ORT_API2_STATUS(AddFreeDimensionOverride, _Inout_ OrtSessionOptions* options, _In_ const char* dim_denotation,
                  _In_ int64_t dim_value);

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
  ORT_API2_STATUS(GetValue, _In_ const OrtValue* value, int index, _Inout_ OrtAllocator* allocator,
                  _Outptr_ OrtValue** out);

  /**
   * Returns 2 for type map and N for sequence where N is the number of elements
   * in the sequence.
   */
  ORT_API2_STATUS(GetValueCount, _In_ const OrtValue* value, _Out_ size_t* out);

  /**
   * To construct a map, use num_values = 2 and 'in' should be an arrary of 2 OrtValues
   * representing keys and values.
   * To construct a sequence, use num_values = N where N is the number of the elements in the
   * sequence. 'in' should be an arrary of N OrtValues.
   * \value_type should be either map or sequence.
   */
  ORT_API2_STATUS(CreateValue, _In_reads_(num_values) const OrtValue* const* in, size_t num_values,
                  enum ONNXType value_type, _Outptr_ OrtValue** out);

  /**
     * Construct OrtValue that contains a value of non-standard type created for
     * experiments or while awaiting standardization. OrtValue in this case would contain
     * an internal representation of the Opaque type. Opaque types are distinguished between
     * each other by two strings 1) domain and 2) type name. The combination of the two
     * must be unique, so the type representation is properly identified internally. The combination
     * must be properly registered from within ORT at both compile/run time or by another API.
     *
     * To construct the OrtValue pass domain and type names, also a pointer to a data container
     * the type of which must be know to both ORT and the client program. That data container may or may
     * not match the internal representation of the Opaque type. The sizeof(data_container) is passed for
     * verification purposes.
     *
     * \domain_name - domain name for the Opaque type, null terminated.
     * \type_name   - type name for the Opaque type, null terminated.
     * \data_contianer - data to populate OrtValue
     * \data_container_size - sizeof() of the data container. Must match the sizeof() of the expected
     *                    data_container size internally.
     */
  ORT_API2_STATUS(CreateOpaqueValue, _In_z_ const char* domain_name, _In_z_ const char* type_name,
                  _In_ const void* data_container, size_t data_container_size, _Outptr_ OrtValue** out);

  /**
     * Fetch data from an OrtValue that contains a value of non-standard type created for
     * experiments or while awaiting standardization.
     * \domain_name - domain name for the Opaque type, null terminated.
     * \type_name   - type name for the Opaque type, null terminated.
     * \data_contianer - data to populate OrtValue
     * \data_container_size - sizeof() of the data container. Must match the sizeof() of the expected
     *                    data_container size internally.
     */

  ORT_API2_STATUS(GetOpaqueValue, _In_ const char* domain_name, _In_ const char* type_name, _In_ const OrtValue* in,
                  _Out_ void* data_container, size_t data_container_size);

  ORT_API2_STATUS(KernelInfoGetAttribute_float, _In_ const OrtKernelInfo* info, _In_ const char* name,
                  _Out_ float* out);
  ORT_API2_STATUS(KernelInfoGetAttribute_int64, _In_ const OrtKernelInfo* info, _In_ const char* name,
                  _Out_ int64_t* out);
  ORT_API2_STATUS(KernelInfoGetAttribute_string, _In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ char* out,
                  _Inout_ size_t* size);

  ORT_API2_STATUS(KernelContext_GetInputCount, _In_ const OrtKernelContext* context, _Out_ size_t* out);
  ORT_API2_STATUS(KernelContext_GetOutputCount, _In_ const OrtKernelContext* context, _Out_ size_t* out);
  ORT_API2_STATUS(KernelContext_GetInput, _In_ const OrtKernelContext* context, _In_ size_t index,
                  _Out_ const OrtValue** out);
  ORT_API2_STATUS(KernelContext_GetOutput, _Inout_ OrtKernelContext* context, _In_ size_t index,
                  _In_ const int64_t* dim_values, size_t dim_count, _Outptr_ OrtValue** out);

  ORT_CLASS_RELEASE(Env);
  ORT_CLASS_RELEASE(Status);  // nullptr for Status* indicates success
  ORT_CLASS_RELEASE(MemoryInfo);
  ORT_CLASS_RELEASE(Session);  //Don't call OrtReleaseSession from Dllmain (because session owns a thread pool)
  ORT_CLASS_RELEASE(Value);
  ORT_CLASS_RELEASE(RunOptions);
  ORT_CLASS_RELEASE(TypeInfo);
  ORT_CLASS_RELEASE(TensorTypeAndShapeInfo);
  ORT_CLASS_RELEASE(SessionOptions);
  ORT_CLASS_RELEASE(CustomOpDomain);

  // End of Version 1 - DO NOT MODIFY ABOVE (see above text for more information)

  // Version 2 - In development, feel free to add/remove/rearrange here

  /**
    * GetDenotationFromTypeInfo
	 * This api augments OrtTypeInfo to return denotations on the type.
	 * This is used by WinML to determine if an input/output is intended to be an Image or a Tensor.
    */
  ORT_API2_STATUS(GetDenotationFromTypeInfo, _In_ const OrtTypeInfo*, _Out_ const char** const denotation,
                  _Out_ size_t* len);

  // OrtTypeInfo Casting methods

  /**
    * CastTypeInfoToMapTypeInfo
	 * This api augments OrtTypeInfo to return an OrtMapTypeInfo when the type is a map.
	 * The OrtMapTypeInfo has additional information about the map's key type and value type.
	 * This is used by WinML to support model reflection APIs.
	 * This is used by WinML to support model reflection APIs.
	 *
	 * Don't free the 'out' value
    */
  ORT_API2_STATUS(CastTypeInfoToMapTypeInfo, _In_ const OrtTypeInfo* type_info,
                  _Outptr_result_maybenull_ const OrtMapTypeInfo** out);

  /**
    * CastTypeInfoToSequenceTypeInfo
	 * This api augments OrtTypeInfo to return an OrtSequenceTypeInfo when the type is a sequence.
	 * The OrtSequenceTypeInfo has additional information about the sequence's element type.
    * This is used by WinML to support model reflection APIs.
	 *
	 * Don't free the 'out' value
    */
  ORT_API2_STATUS(CastTypeInfoToSequenceTypeInfo, _In_ const OrtTypeInfo* type_info,
                  _Outptr_result_maybenull_ const OrtSequenceTypeInfo** out);

  // OrtMapTypeInfo Accessors

  /**
    * GetMapKeyType
	 * This api augments get the key type of a map. Key types are restricted to being scalar types and use ONNXTensorElementDataType.
	 * This is used by WinML to support model reflection APIs.
    */
  ORT_API2_STATUS(GetMapKeyType, _In_ const OrtMapTypeInfo* map_type_info, _Out_ enum ONNXTensorElementDataType* out);

  /**
    * GetMapValueType
	 * This api augments get the value type of a map.
    */
  ORT_API2_STATUS(GetMapValueType, _In_ const OrtMapTypeInfo* map_type_info, _Outptr_ OrtTypeInfo** type_info);

  // OrtSequenceTypeInfo Accessors

  /**
    * GetSequenceElementType
	 * This api augments get the element type of a sequence.
	 * This is used by WinML to support model reflection APIs.
    */
  ORT_API2_STATUS(GetSequenceElementType, _In_ const OrtSequenceTypeInfo* sequence_type_info,
                  _Outptr_ OrtTypeInfo** type_info);

  ORT_CLASS_RELEASE(MapTypeInfo);
  ORT_CLASS_RELEASE(SequenceTypeInfo);

  /**
   * \param out is set to a null terminated string allocated using 'allocator'. The caller is responsible for freeing it.
   * Profiling is turned ON automatically if enabled for the particular session by invoking EnableProfiling() 
   * on the SessionOptions instance used to create the session.  
   */
  ORT_API2_STATUS(SessionEndProfiling, _In_ OrtSession* sess, _Inout_ OrtAllocator* allocator, _Outptr_ char** out);

  /**
   * \param out is a pointer to the newly created object. The pointer should be freed by calling ReleaseModelMetadata after use.
   */
  ORT_API2_STATUS(SessionGetModelMetadata, _In_ const OrtSession* sess, _Outptr_ OrtModelMetadata** out);

  /**
   * \param value  is set to a null terminated string allocated using 'allocator'. The caller is responsible for freeing it.
   */
  ORT_API2_STATUS(ModelMetadataGetProducerName, _In_ const OrtModelMetadata* model_metadata,
                  _Inout_ OrtAllocator* allocator, _Outptr_ char** value);
  ORT_API2_STATUS(ModelMetadataGetGraphName, _In_ const OrtModelMetadata* model_metadata,
                  _Inout_ OrtAllocator* allocator, _Outptr_ char** value);
  ORT_API2_STATUS(ModelMetadataGetDomain, _In_ const OrtModelMetadata* model_metadata, _Inout_ OrtAllocator* allocator,
                  _Outptr_ char** value);
  ORT_API2_STATUS(ModelMetadataGetDescription, _In_ const OrtModelMetadata* model_metadata,
                  _Inout_ OrtAllocator* allocator, _Outptr_ char** value);
  /**
   * \param value  is set to a null terminated string allocated using 'allocator'. The caller is responsible for freeing it.
   * 'value' will be a nullptr if the given key is not found in the custom metadata map.
   */
  ORT_API2_STATUS(ModelMetadataLookupCustomMetadataMap, _In_ const OrtModelMetadata* model_metadata,
                  _Inout_ OrtAllocator* allocator, _In_ const char* key, _Outptr_result_maybenull_ char** value);

  ORT_API2_STATUS(ModelMetadataGetVersion, _In_ const OrtModelMetadata* model_metadata, _Out_ int64_t* value);

  ORT_CLASS_RELEASE(ModelMetadata);

  /*
  * Creates an environment with global threadpools that will be shared across sessions.
  * Use this in conjunction with DisablePerSessionThreads API or else the session will use
  * its own thread pools.
  */
  ORT_API2_STATUS(CreateEnvWithGlobalThreadPools, OrtLoggingLevel default_logging_level, _In_ const char* logid,
                  _In_ const OrtThreadingOptions* t_options, _Outptr_ OrtEnv** out);

  /* TODO: Should there be a version of CreateEnvWithGlobalThreadPools with custom logging function? */

  /*
  * Calling this API will make the session use the global threadpools shared across sessions.
  * This API should be used in conjunction with CreateEnvWithGlobalThreadPools API.
  */
  ORT_API2_STATUS(DisablePerSessionThreads, _Inout_ OrtSessionOptions* options);

  ORT_API2_STATUS(CreateThreadingOptions, _Outptr_ OrtThreadingOptions** out);

  ORT_CLASS_RELEASE(ThreadingOptions);

  /**
   * \param num_keys contains the number of keys in the custom metadata map
   * \param keys is an array of null terminated strings (array count = num_keys) allocated using 'allocator'. 
   * The caller is responsible for freeing each string and the pointer array.
   * 'keys' will be a nullptr if custom metadata map is empty.
   */
  ORT_API2_STATUS(ModelMetadataGetCustomMetadataMapKeys, _In_ const OrtModelMetadata* model_metadata,
                  _Inout_ OrtAllocator* allocator, _Outptr_result_buffer_maybenull_(*num_keys) char*** keys, _Out_ int64_t* num_keys);

  // Override symbolic dimensions (by specific name strings) with actual values
  // if known at session initialization time to enable optimizations that can
  // take advantage of fixed values (such as memory planning, etc)
  ORT_API2_STATUS(AddFreeDimensionOverrideByName,
                  _Inout_ OrtSessionOptions* options, _In_ const char* dim_name,
                  _In_ int64_t dim_value);

  /**
   * \param out_ptr will hold a pointer to the array of char *
   * representing available providers.
   * \param provider_length is a pointer to an int variable where
   * the number of available providers will be added.
   * The caller is responsible for freeing each char * and the pointer
   * array by calling ReleaseAvailableProviders().
   */
  ORT_API2_STATUS(GetAvailableProviders, _Outptr_ char ***out_ptr,
                  _In_ int *provider_length);

  /**
   * \param ptr is the pointer to an array of available providers you
   * get after calling GetAvailableProviders().
   * \param providers_length is the number of available providers.
   */
  ORT_API2_STATUS(ReleaseAvailableProviders, _In_ char **ptr,
                  _In_ int providers_length);
};

/*
 * Steps to use a custom op:
 *   1 Create an OrtCustomOpDomain with the domain name used by the custom ops
 *   2 Create an OrtCustomOp structure for each op and add them to the domain
 *   3 Call OrtAddCustomOpDomain to add the custom domain of ops to the session options
*/
#define OrtCustomOpApi OrtApi

/*
 * The OrtCustomOp structure defines a custom op's schema and its kernel callbacks. The callbacks are filled in by
 * the implementor of the custom op.
*/
struct OrtCustomOp {
  uint32_t version;  // Initialize to ORT_API_VERSION

  // This callback creates the kernel, which is a user defined parameter that is passed to the Kernel* callbacks below.
  void*(ORT_API_CALL* CreateKernel)(_In_ struct OrtCustomOp* op, _In_ const OrtApi* api,
                                    _In_ const OrtKernelInfo* info);

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

/*
 * END EXPERIMENTAL
*/

#ifdef __cplusplus
}
#endif
