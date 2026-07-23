#ifndef ORT_GO_CSHIM_H
#define ORT_GO_CSHIM_H

#include "onnxruntime_c_api.h"

// Preferred API version (ORT >= 1.27.0, includes getter APIs).
// Falls back to ORT_GO_API_VERSION_MIN for older libraries.
#define ORT_GO_API_VERSION_MAX 27
#define ORT_GO_API_VERSION_MIN 17

// Initialize the global OrtApi pointer from a resolved OrtGetApiBase function.
// Tries ORT_GO_API_VERSION_MAX first, falls back to ORT_GO_API_VERSION_MIN.
// Returns 0 on success, 1 if apiBase is NULL, 2 if even min version unsupported.
// Sets *actual_version to the version that was loaded.
int ort_init_api(void *get_api_base_fn, int *actual_version);

// Environment
OrtStatusPtr ort_CreateEnv(OrtLoggingLevel level, const char *logid, OrtEnv **out);
void ort_ReleaseEnv(OrtEnv *env);

// Session options
OrtStatusPtr ort_CreateSessionOptions(OrtSessionOptions **out);
void ort_ReleaseSessionOptions(OrtSessionOptions *opts);
OrtStatusPtr ort_SetIntraOpNumThreads(OrtSessionOptions *opts, int n);
OrtStatusPtr ort_SetInterOpNumThreads(OrtSessionOptions *opts, int n);
OrtStatusPtr ort_SetSessionGraphOptimizationLevel(OrtSessionOptions *opts, GraphOptimizationLevel level);
OrtStatusPtr ort_AddSessionConfigEntry(OrtSessionOptions *opts, const char *key, const char *value);
OrtStatusPtr ort_SessionOptionsAppendExecutionProvider(
    OrtSessionOptions *opts, const char *provider_name,
    const char *const *keys, const char *const *values, size_t num_keys);

// CUDA V2 provider options
OrtStatusPtr ort_CreateCUDAProviderOptions(OrtCUDAProviderOptionsV2 **out);
OrtStatusPtr ort_UpdateCUDAProviderOptions(OrtCUDAProviderOptionsV2 *opts,
    const char *const *keys, const char *const *values, size_t num_keys);
OrtStatusPtr ort_SessionOptionsAppendExecutionProvider_CUDA_V2(
    OrtSessionOptions *opts, const OrtCUDAProviderOptionsV2 *cuda_opts);
void ort_ReleaseCUDAProviderOptions(OrtCUDAProviderOptionsV2 *opts);

// TensorRT V2 provider options
OrtStatusPtr ort_CreateTensorRTProviderOptions(OrtTensorRTProviderOptionsV2 **out);
OrtStatusPtr ort_UpdateTensorRTProviderOptions(OrtTensorRTProviderOptionsV2 *opts,
    const char *const *keys, const char *const *values, size_t num_keys);
OrtStatusPtr ort_SessionOptionsAppendExecutionProvider_TensorRT_V2(
    OrtSessionOptions *opts, const OrtTensorRTProviderOptionsV2 *trt_opts);
void ort_ReleaseTensorRTProviderOptions(OrtTensorRTProviderOptionsV2 *opts);

// Session
OrtStatusPtr ort_CreateSession(const OrtEnv *env, const ORTCHAR_T *model_path,
    const OrtSessionOptions *opts, OrtSession **out);
OrtStatusPtr ort_CreateSessionFromArray(const OrtEnv *env, const void *model_data,
    size_t model_data_length, const OrtSessionOptions *opts, OrtSession **out);
void ort_ReleaseSession(OrtSession *session);

// Session introspection
OrtStatusPtr ort_SessionGetInputCount(const OrtSession *session, size_t *out);
OrtStatusPtr ort_SessionGetOutputCount(const OrtSession *session, size_t *out);
OrtStatusPtr ort_SessionGetInputName(const OrtSession *session, size_t index,
    OrtAllocator *allocator, char **out);
OrtStatusPtr ort_SessionGetOutputName(const OrtSession *session, size_t index,
    OrtAllocator *allocator, char **out);
OrtStatusPtr ort_SessionGetInputTypeInfo(const OrtSession *session, size_t index,
    OrtTypeInfo **out);
OrtStatusPtr ort_SessionGetOutputTypeInfo(const OrtSession *session, size_t index,
    OrtTypeInfo **out);

// Type info
OrtStatusPtr ort_CastTypeInfoToTensorInfo(const OrtTypeInfo *type_info,
    const OrtTensorTypeAndShapeInfo **out);
OrtStatusPtr ort_GetOnnxTypeFromTypeInfo(const OrtTypeInfo *type_info, enum ONNXType *out);
void ort_ReleaseTypeInfo(OrtTypeInfo *info);

// Tensor type and shape info
OrtStatusPtr ort_GetTensorElementType(const OrtTensorTypeAndShapeInfo *info,
    enum ONNXTensorElementDataType *out);
OrtStatusPtr ort_GetDimensionsCount(const OrtTensorTypeAndShapeInfo *info, size_t *out);
OrtStatusPtr ort_GetDimensions(const OrtTensorTypeAndShapeInfo *info,
    int64_t *dim_values, size_t dim_values_length);
OrtStatusPtr ort_GetTensorShapeElementCount(const OrtTensorTypeAndShapeInfo *info, size_t *out);
void ort_ReleaseTensorTypeAndShapeInfo(OrtTensorTypeAndShapeInfo *info);

// Tensor creation and data access
OrtStatusPtr ort_CreateTensorWithDataAsOrtValue(const OrtMemoryInfo *info,
    void *p_data, size_t p_data_len, const int64_t *shape, size_t shape_len,
    ONNXTensorElementDataType type, OrtValue **out);
OrtStatusPtr ort_CreateTensorAsOrtValue(OrtAllocator *allocator,
    const int64_t *shape, size_t shape_len, ONNXTensorElementDataType type, OrtValue **out);
OrtStatusPtr ort_GetTensorMutableData(OrtValue *value, void **out);
OrtStatusPtr ort_GetTensorTypeAndShape(const OrtValue *value,
    OrtTensorTypeAndShapeInfo **out);
OrtStatusPtr ort_IsTensor(const OrtValue *value, int *out);
void ort_ReleaseValue(OrtValue *value);

// Memory and allocator
OrtStatusPtr ort_CreateCpuMemoryInfo(enum OrtAllocatorType type,
    enum OrtMemType mem_type, OrtMemoryInfo **out);
void ort_ReleaseMemoryInfo(OrtMemoryInfo *info);
OrtStatusPtr ort_GetAllocatorWithDefaultOptions(OrtAllocator **out);

// Run
OrtStatusPtr ort_Run(OrtSession *session, const OrtRunOptions *run_options,
    const char *const *input_names, const OrtValue *const *inputs, size_t input_len,
    const char *const *output_names, size_t output_names_len, OrtValue **outputs);

// Run options
OrtStatusPtr ort_CreateRunOptions(OrtRunOptions **out);
void ort_ReleaseRunOptions(OrtRunOptions *opts);
OrtStatusPtr ort_RunOptionsSetTerminate(OrtRunOptions *opts);
OrtStatusPtr ort_RunOptionsUnsetTerminate(OrtRunOptions *opts);
OrtStatusPtr ort_RunOptionsSetRunLogVerbosityLevel(OrtRunOptions *opts, int level);
OrtStatusPtr ort_RunOptionsSetRunLogSeverityLevel(OrtRunOptions *opts, int level);
OrtStatusPtr ort_RunOptionsSetRunTag(OrtRunOptions *opts, const char *tag);
OrtStatusPtr ort_AddRunConfigEntry(OrtRunOptions *opts, const char *key, const char *value);

// Model metadata
OrtStatusPtr ort_SessionGetModelMetadata(const OrtSession *session, OrtModelMetadata **out);
OrtStatusPtr ort_ModelMetadataGetProducerName(const OrtModelMetadata *meta,
    OrtAllocator *allocator, char **out);
OrtStatusPtr ort_ModelMetadataGetGraphName(const OrtModelMetadata *meta,
    OrtAllocator *allocator, char **out);
OrtStatusPtr ort_ModelMetadataGetDomain(const OrtModelMetadata *meta,
    OrtAllocator *allocator, char **out);
OrtStatusPtr ort_ModelMetadataGetDescription(const OrtModelMetadata *meta,
    OrtAllocator *allocator, char **out);
OrtStatusPtr ort_ModelMetadataGetGraphDescription(const OrtModelMetadata *meta,
    OrtAllocator *allocator, char **out);
OrtStatusPtr ort_ModelMetadataGetVersion(const OrtModelMetadata *meta, int64_t *out);
OrtStatusPtr ort_ModelMetadataGetCustomMetadataMapKeys(const OrtModelMetadata *meta,
    OrtAllocator *allocator, char ***keys, int64_t *count);
OrtStatusPtr ort_ModelMetadataLookupCustomMetadataMap(const OrtModelMetadata *meta,
    OrtAllocator *allocator, const char *key, char **out);
void ort_ReleaseModelMetadata(OrtModelMetadata *meta);

// IO Binding
OrtStatusPtr ort_CreateIoBinding(OrtSession *session, OrtIoBinding **out);
OrtStatusPtr ort_BindInput(OrtIoBinding *binding, const char *name, const OrtValue *val);
OrtStatusPtr ort_BindOutput(OrtIoBinding *binding, const char *name, const OrtValue *val);
OrtStatusPtr ort_BindOutputToDevice(OrtIoBinding *binding, const char *name,
    const OrtMemoryInfo *mem_info);
OrtStatusPtr ort_RunWithBinding(OrtSession *session, const OrtRunOptions *run_options,
    const OrtIoBinding *binding);
OrtStatusPtr ort_GetBoundOutputNames(const OrtIoBinding *binding, OrtAllocator *allocator,
    char **buffer, size_t **lengths, size_t *count);
OrtStatusPtr ort_GetBoundOutputValues(const OrtIoBinding *binding, OrtAllocator *allocator,
    OrtValue ***output, size_t *count);
void ort_ClearBoundInputs(OrtIoBinding *binding);
void ort_ClearBoundOutputs(OrtIoBinding *binding);
void ort_ReleaseIoBinding(OrtIoBinding *binding);

// String tensors
OrtStatusPtr ort_FillStringTensor(OrtValue *value, const char *const *s, size_t s_len);
OrtStatusPtr ort_GetStringTensorDataLength(const OrtValue *value, size_t *len);
OrtStatusPtr ort_GetStringTensorContent(const OrtValue *value, void *s, size_t s_len,
    size_t *offsets, size_t offsets_len);
OrtStatusPtr ort_GetStringTensorElementLength(const OrtValue *value, size_t index, size_t *out);
OrtStatusPtr ort_GetStringTensorElement(const OrtValue *value, size_t s_len, size_t index, void *s);

// Additional session options
OrtStatusPtr ort_CloneSessionOptions(const OrtSessionOptions *in, OrtSessionOptions **out);
OrtStatusPtr ort_EnableMemPattern(OrtSessionOptions *opts);
OrtStatusPtr ort_DisableMemPattern(OrtSessionOptions *opts);
OrtStatusPtr ort_EnableCpuMemArena(OrtSessionOptions *opts);
OrtStatusPtr ort_DisableCpuMemArena(OrtSessionOptions *opts);
OrtStatusPtr ort_EnableProfiling(OrtSessionOptions *opts, const ORTCHAR_T *prefix);
OrtStatusPtr ort_DisableProfiling(OrtSessionOptions *opts);
OrtStatusPtr ort_AddFreeDimensionOverride(OrtSessionOptions *opts,
    const char *dim_denotation, int64_t dim_value);
OrtStatusPtr ort_AddFreeDimensionOverrideByName(OrtSessionOptions *opts,
    const char *dim_name, int64_t dim_value);
OrtStatusPtr ort_SetSessionExecutionMode(OrtSessionOptions *opts, ExecutionMode mode);
OrtStatusPtr ort_AddInitializer(OrtSessionOptions *opts, const char *name, const OrtValue *val);

// Session profiling
OrtStatusPtr ort_SessionEndProfiling(OrtSession *session, OrtAllocator *allocator, char **out);

// Version
const char *ort_GetVersionString(void);

// Session options getters (since 1.27)
OrtStatusPtr ort_GetMemPatternEnabled(const OrtSessionOptions *opts, int *out);
OrtStatusPtr ort_GetSessionExecutionMode(const OrtSessionOptions *opts, ExecutionMode *out);

// Value type
OrtStatusPtr ort_GetValueType(const OrtValue *value, enum ONNXType *out);
OrtStatusPtr ort_GetValueCount(const OrtValue *value, size_t *out);
OrtStatusPtr ort_GetValue(const OrtValue *value, int index, OrtAllocator *allocator, OrtValue **out);

// Value creation (sequence/map)
OrtStatusPtr ort_CreateValue(const OrtValue *const *in, size_t num_values,
    enum ONNXType value_type, OrtValue **out);

// Memory info
OrtStatusPtr ort_CreateMemoryInfo(const char *name, enum OrtAllocatorType type,
    int id, enum OrtMemType mem_type, OrtMemoryInfo **out);

// Telemetry
OrtStatusPtr ort_EnableTelemetryEvents(const OrtEnv *env);
OrtStatusPtr ort_DisableTelemetryEvents(const OrtEnv *env);

// Additional session options (continued)
OrtStatusPtr ort_SetOptimizedModelFilePath(OrtSessionOptions *opts, const ORTCHAR_T *path);
OrtStatusPtr ort_RegisterCustomOpsLibrary_V2(OrtSessionOptions *opts, const ORTCHAR_T *path);
OrtStatusPtr ort_HasSessionConfigEntry(const OrtSessionOptions *opts, const char *key, int *out);
OrtStatusPtr ort_GetSessionConfigEntry(const OrtSessionOptions *opts, const char *key,
    char *value, size_t *size);

// Error handling
OrtErrorCode ort_GetErrorCode(const OrtStatus *status);
const char *ort_GetErrorMessage(const OrtStatus *status);
void ort_ReleaseStatus(OrtStatus *status);

// Provider enumeration
OrtStatusPtr ort_GetAvailableProviders(char ***out, int *count);
OrtStatusPtr ort_ReleaseAvailableProviders(char **ptr, int count);

// Allocator free (for freeing names returned by SessionGetInputName etc.)
void ort_AllocatorFree(OrtAllocator *allocator, void *ptr);

#endif // ORT_GO_CSHIM_H
