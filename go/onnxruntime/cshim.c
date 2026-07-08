#include "cshim.h"

static const OrtApi *g_api = NULL;

int ort_init_api(void *get_api_base_fn) {
    const OrtApiBase *base = ((const OrtApiBase *(*)(void))get_api_base_fn)();
    if (base == NULL) return 1;
    g_api = base->GetApi(ORT_GO_API_VERSION);
    return g_api == NULL ? 2 : 0;
}

// Environment

OrtStatusPtr ort_CreateEnv(OrtLoggingLevel level, const char *logid, OrtEnv **out) {
    return g_api->CreateEnv(level, logid, out);
}

void ort_ReleaseEnv(OrtEnv *env) {
    g_api->ReleaseEnv(env);
}

// Session options

OrtStatusPtr ort_CreateSessionOptions(OrtSessionOptions **out) {
    return g_api->CreateSessionOptions(out);
}

void ort_ReleaseSessionOptions(OrtSessionOptions *opts) {
    g_api->ReleaseSessionOptions(opts);
}

OrtStatusPtr ort_SetIntraOpNumThreads(OrtSessionOptions *opts, int n) {
    return g_api->SetIntraOpNumThreads(opts, n);
}

OrtStatusPtr ort_SetInterOpNumThreads(OrtSessionOptions *opts, int n) {
    return g_api->SetInterOpNumThreads(opts, n);
}

OrtStatusPtr ort_SetSessionGraphOptimizationLevel(OrtSessionOptions *opts, GraphOptimizationLevel level) {
    return g_api->SetSessionGraphOptimizationLevel(opts, level);
}

OrtStatusPtr ort_AddSessionConfigEntry(OrtSessionOptions *opts, const char *key, const char *value) {
    return g_api->AddSessionConfigEntry(opts, key, value);
}

OrtStatusPtr ort_SessionOptionsAppendExecutionProvider(
    OrtSessionOptions *opts, const char *provider_name,
    const char *const *keys, const char *const *values, size_t num_keys) {
    return g_api->SessionOptionsAppendExecutionProvider(opts, provider_name, keys, values, num_keys);
}

// CUDA V2 provider options

OrtStatusPtr ort_CreateCUDAProviderOptions(OrtCUDAProviderOptionsV2 **out) {
    return g_api->CreateCUDAProviderOptions(out);
}

OrtStatusPtr ort_UpdateCUDAProviderOptions(OrtCUDAProviderOptionsV2 *opts,
    const char *const *keys, const char *const *values, size_t num_keys) {
    return g_api->UpdateCUDAProviderOptions(opts, keys, values, num_keys);
}

OrtStatusPtr ort_SessionOptionsAppendExecutionProvider_CUDA_V2(
    OrtSessionOptions *opts, const OrtCUDAProviderOptionsV2 *cuda_opts) {
    return g_api->SessionOptionsAppendExecutionProvider_CUDA_V2(opts, cuda_opts);
}

void ort_ReleaseCUDAProviderOptions(OrtCUDAProviderOptionsV2 *opts) {
    g_api->ReleaseCUDAProviderOptions(opts);
}

// TensorRT V2 provider options

OrtStatusPtr ort_CreateTensorRTProviderOptions(OrtTensorRTProviderOptionsV2 **out) {
    return g_api->CreateTensorRTProviderOptions(out);
}

OrtStatusPtr ort_UpdateTensorRTProviderOptions(OrtTensorRTProviderOptionsV2 *opts,
    const char *const *keys, const char *const *values, size_t num_keys) {
    return g_api->UpdateTensorRTProviderOptions(opts, keys, values, num_keys);
}

OrtStatusPtr ort_SessionOptionsAppendExecutionProvider_TensorRT_V2(
    OrtSessionOptions *opts, const OrtTensorRTProviderOptionsV2 *trt_opts) {
    return g_api->SessionOptionsAppendExecutionProvider_TensorRT_V2(opts, trt_opts);
}

void ort_ReleaseTensorRTProviderOptions(OrtTensorRTProviderOptionsV2 *opts) {
    g_api->ReleaseTensorRTProviderOptions(opts);
}

// Session

OrtStatusPtr ort_CreateSession(const OrtEnv *env, const char *model_path,
    const OrtSessionOptions *opts, OrtSession **out) {
    return g_api->CreateSession(env, model_path, opts, out);
}

OrtStatusPtr ort_CreateSessionFromArray(const OrtEnv *env, const void *model_data,
    size_t model_data_length, const OrtSessionOptions *opts, OrtSession **out) {
    return g_api->CreateSessionFromArray(env, model_data, model_data_length, opts, out);
}

void ort_ReleaseSession(OrtSession *session) {
    g_api->ReleaseSession(session);
}

// Session introspection

OrtStatusPtr ort_SessionGetInputCount(const OrtSession *session, size_t *out) {
    return g_api->SessionGetInputCount(session, out);
}

OrtStatusPtr ort_SessionGetOutputCount(const OrtSession *session, size_t *out) {
    return g_api->SessionGetOutputCount(session, out);
}

OrtStatusPtr ort_SessionGetInputName(const OrtSession *session, size_t index,
    OrtAllocator *allocator, char **out) {
    return g_api->SessionGetInputName(session, index, allocator, out);
}

OrtStatusPtr ort_SessionGetOutputName(const OrtSession *session, size_t index,
    OrtAllocator *allocator, char **out) {
    return g_api->SessionGetOutputName(session, index, allocator, out);
}

OrtStatusPtr ort_SessionGetInputTypeInfo(const OrtSession *session, size_t index,
    OrtTypeInfo **out) {
    return g_api->SessionGetInputTypeInfo(session, index, out);
}

OrtStatusPtr ort_SessionGetOutputTypeInfo(const OrtSession *session, size_t index,
    OrtTypeInfo **out) {
    return g_api->SessionGetOutputTypeInfo(session, index, out);
}

// Type info

OrtStatusPtr ort_CastTypeInfoToTensorInfo(const OrtTypeInfo *type_info,
    const OrtTensorTypeAndShapeInfo **out) {
    return g_api->CastTypeInfoToTensorInfo(type_info, out);
}

OrtStatusPtr ort_GetOnnxTypeFromTypeInfo(const OrtTypeInfo *type_info, enum ONNXType *out) {
    return g_api->GetOnnxTypeFromTypeInfo(type_info, out);
}

void ort_ReleaseTypeInfo(OrtTypeInfo *info) {
    g_api->ReleaseTypeInfo(info);
}

// Tensor type and shape info

OrtStatusPtr ort_GetTensorElementType(const OrtTensorTypeAndShapeInfo *info,
    enum ONNXTensorElementDataType *out) {
    return g_api->GetTensorElementType(info, out);
}

OrtStatusPtr ort_GetDimensionsCount(const OrtTensorTypeAndShapeInfo *info, size_t *out) {
    return g_api->GetDimensionsCount(info, out);
}

OrtStatusPtr ort_GetDimensions(const OrtTensorTypeAndShapeInfo *info,
    int64_t *dim_values, size_t dim_values_length) {
    return g_api->GetDimensions(info, dim_values, dim_values_length);
}

OrtStatusPtr ort_GetTensorShapeElementCount(const OrtTensorTypeAndShapeInfo *info, size_t *out) {
    return g_api->GetTensorShapeElementCount(info, out);
}

void ort_ReleaseTensorTypeAndShapeInfo(OrtTensorTypeAndShapeInfo *info) {
    g_api->ReleaseTensorTypeAndShapeInfo(info);
}

// Tensor creation and data access

OrtStatusPtr ort_CreateTensorWithDataAsOrtValue(const OrtMemoryInfo *info,
    void *p_data, size_t p_data_len, const int64_t *shape, size_t shape_len,
    ONNXTensorElementDataType type, OrtValue **out) {
    return g_api->CreateTensorWithDataAsOrtValue(info, p_data, p_data_len, shape, shape_len, type, out);
}

OrtStatusPtr ort_CreateTensorAsOrtValue(OrtAllocator *allocator,
    const int64_t *shape, size_t shape_len, ONNXTensorElementDataType type, OrtValue **out) {
    return g_api->CreateTensorAsOrtValue(allocator, shape, shape_len, type, out);
}

OrtStatusPtr ort_GetTensorMutableData(OrtValue *value, void **out) {
    return g_api->GetTensorMutableData(value, out);
}

OrtStatusPtr ort_GetTensorTypeAndShape(const OrtValue *value,
    OrtTensorTypeAndShapeInfo **out) {
    return g_api->GetTensorTypeAndShape(value, out);
}

OrtStatusPtr ort_IsTensor(const OrtValue *value, int *out) {
    return g_api->IsTensor(value, out);
}

void ort_ReleaseValue(OrtValue *value) {
    g_api->ReleaseValue(value);
}

// Memory and allocator

OrtStatusPtr ort_CreateCpuMemoryInfo(enum OrtAllocatorType type,
    enum OrtMemType mem_type, OrtMemoryInfo **out) {
    return g_api->CreateCpuMemoryInfo(type, mem_type, out);
}

void ort_ReleaseMemoryInfo(OrtMemoryInfo *info) {
    g_api->ReleaseMemoryInfo(info);
}

OrtStatusPtr ort_GetAllocatorWithDefaultOptions(OrtAllocator **out) {
    return g_api->GetAllocatorWithDefaultOptions(out);
}

// Run

OrtStatusPtr ort_Run(OrtSession *session, const OrtRunOptions *run_options,
    const char *const *input_names, const OrtValue *const *inputs, size_t input_len,
    const char *const *output_names, size_t output_names_len, OrtValue **outputs) {
    return g_api->Run(session, run_options, input_names, inputs, input_len,
                      output_names, output_names_len, outputs);
}

// Run options

OrtStatusPtr ort_CreateRunOptions(OrtRunOptions **out) {
    return g_api->CreateRunOptions(out);
}

void ort_ReleaseRunOptions(OrtRunOptions *opts) {
    g_api->ReleaseRunOptions(opts);
}

OrtStatusPtr ort_RunOptionsSetTerminate(OrtRunOptions *opts) {
    return g_api->RunOptionsSetTerminate(opts);
}

OrtStatusPtr ort_RunOptionsUnsetTerminate(OrtRunOptions *opts) {
    return g_api->RunOptionsUnsetTerminate(opts);
}

OrtStatusPtr ort_RunOptionsSetRunLogVerbosityLevel(OrtRunOptions *opts, int level) {
    return g_api->RunOptionsSetRunLogVerbosityLevel(opts, level);
}

OrtStatusPtr ort_RunOptionsSetRunLogSeverityLevel(OrtRunOptions *opts, int level) {
    return g_api->RunOptionsSetRunLogSeverityLevel(opts, level);
}

OrtStatusPtr ort_RunOptionsSetRunTag(OrtRunOptions *opts, const char *tag) {
    return g_api->RunOptionsSetRunTag(opts, tag);
}

OrtStatusPtr ort_AddRunConfigEntry(OrtRunOptions *opts, const char *key, const char *value) {
    return g_api->AddRunConfigEntry(opts, key, value);
}

// Model metadata

OrtStatusPtr ort_SessionGetModelMetadata(const OrtSession *session, OrtModelMetadata **out) {
    return g_api->SessionGetModelMetadata(session, out);
}

OrtStatusPtr ort_ModelMetadataGetProducerName(const OrtModelMetadata *meta,
    OrtAllocator *allocator, char **out) {
    return g_api->ModelMetadataGetProducerName(meta, allocator, out);
}

OrtStatusPtr ort_ModelMetadataGetGraphName(const OrtModelMetadata *meta,
    OrtAllocator *allocator, char **out) {
    return g_api->ModelMetadataGetGraphName(meta, allocator, out);
}

OrtStatusPtr ort_ModelMetadataGetDomain(const OrtModelMetadata *meta,
    OrtAllocator *allocator, char **out) {
    return g_api->ModelMetadataGetDomain(meta, allocator, out);
}

OrtStatusPtr ort_ModelMetadataGetDescription(const OrtModelMetadata *meta,
    OrtAllocator *allocator, char **out) {
    return g_api->ModelMetadataGetDescription(meta, allocator, out);
}

OrtStatusPtr ort_ModelMetadataGetGraphDescription(const OrtModelMetadata *meta,
    OrtAllocator *allocator, char **out) {
    return g_api->ModelMetadataGetGraphDescription(meta, allocator, out);
}

OrtStatusPtr ort_ModelMetadataGetVersion(const OrtModelMetadata *meta, int64_t *out) {
    return g_api->ModelMetadataGetVersion(meta, out);
}

OrtStatusPtr ort_ModelMetadataGetCustomMetadataMapKeys(const OrtModelMetadata *meta,
    OrtAllocator *allocator, char ***keys, int64_t *count) {
    return g_api->ModelMetadataGetCustomMetadataMapKeys(meta, allocator, keys, count);
}

OrtStatusPtr ort_ModelMetadataLookupCustomMetadataMap(const OrtModelMetadata *meta,
    OrtAllocator *allocator, const char *key, char **out) {
    return g_api->ModelMetadataLookupCustomMetadataMap(meta, allocator, key, out);
}

void ort_ReleaseModelMetadata(OrtModelMetadata *meta) {
    g_api->ReleaseModelMetadata(meta);
}

// IO Binding

OrtStatusPtr ort_CreateIoBinding(OrtSession *session, OrtIoBinding **out) {
    return g_api->CreateIoBinding(session, out);
}

OrtStatusPtr ort_BindInput(OrtIoBinding *binding, const char *name, const OrtValue *val) {
    return g_api->BindInput(binding, name, val);
}

OrtStatusPtr ort_BindOutput(OrtIoBinding *binding, const char *name, const OrtValue *val) {
    return g_api->BindOutput(binding, name, val);
}

OrtStatusPtr ort_BindOutputToDevice(OrtIoBinding *binding, const char *name,
    const OrtMemoryInfo *mem_info) {
    return g_api->BindOutputToDevice(binding, name, mem_info);
}

OrtStatusPtr ort_RunWithBinding(OrtSession *session, const OrtRunOptions *run_options,
    const OrtIoBinding *binding) {
    return g_api->RunWithBinding(session, run_options, binding);
}

OrtStatusPtr ort_GetBoundOutputNames(const OrtIoBinding *binding, OrtAllocator *allocator,
    char **buffer, size_t **lengths, size_t *count) {
    return g_api->GetBoundOutputNames(binding, allocator, buffer, lengths, count);
}

OrtStatusPtr ort_GetBoundOutputValues(const OrtIoBinding *binding, OrtAllocator *allocator,
    OrtValue ***output, size_t *count) {
    return g_api->GetBoundOutputValues(binding, allocator, output, count);
}

void ort_ClearBoundInputs(OrtIoBinding *binding) {
    g_api->ClearBoundInputs(binding);
}

void ort_ClearBoundOutputs(OrtIoBinding *binding) {
    g_api->ClearBoundOutputs(binding);
}

void ort_ReleaseIoBinding(OrtIoBinding *binding) {
    g_api->ReleaseIoBinding(binding);
}

// String tensors

OrtStatusPtr ort_FillStringTensor(OrtValue *value, const char *const *s, size_t s_len) {
    return g_api->FillStringTensor(value, s, s_len);
}

OrtStatusPtr ort_GetStringTensorDataLength(const OrtValue *value, size_t *len) {
    return g_api->GetStringTensorDataLength(value, len);
}

OrtStatusPtr ort_GetStringTensorContent(const OrtValue *value, void *s, size_t s_len,
    size_t *offsets, size_t offsets_len) {
    return g_api->GetStringTensorContent(value, s, s_len, offsets, offsets_len);
}

OrtStatusPtr ort_GetStringTensorElementLength(const OrtValue *value, size_t index, size_t *out) {
    return g_api->GetStringTensorElementLength(value, index, out);
}

OrtStatusPtr ort_GetStringTensorElement(const OrtValue *value, size_t s_len, size_t index, void *s) {
    return g_api->GetStringTensorElement(value, s_len, index, s);
}

// Additional session options

OrtStatusPtr ort_CloneSessionOptions(const OrtSessionOptions *in, OrtSessionOptions **out) {
    return g_api->CloneSessionOptions(in, out);
}

OrtStatusPtr ort_EnableMemPattern(OrtSessionOptions *opts) {
    return g_api->EnableMemPattern(opts);
}

OrtStatusPtr ort_DisableMemPattern(OrtSessionOptions *opts) {
    return g_api->DisableMemPattern(opts);
}

OrtStatusPtr ort_EnableCpuMemArena(OrtSessionOptions *opts) {
    return g_api->EnableCpuMemArena(opts);
}

OrtStatusPtr ort_DisableCpuMemArena(OrtSessionOptions *opts) {
    return g_api->DisableCpuMemArena(opts);
}

OrtStatusPtr ort_EnableProfiling(OrtSessionOptions *opts, const char *prefix) {
    return g_api->EnableProfiling(opts, prefix);
}

OrtStatusPtr ort_DisableProfiling(OrtSessionOptions *opts) {
    return g_api->DisableProfiling(opts);
}

OrtStatusPtr ort_AddFreeDimensionOverride(OrtSessionOptions *opts,
    const char *dim_denotation, int64_t dim_value) {
    return g_api->AddFreeDimensionOverride(opts, dim_denotation, dim_value);
}

OrtStatusPtr ort_AddFreeDimensionOverrideByName(OrtSessionOptions *opts,
    const char *dim_name, int64_t dim_value) {
    return g_api->AddFreeDimensionOverrideByName(opts, dim_name, dim_value);
}

OrtStatusPtr ort_SetSessionExecutionMode(OrtSessionOptions *opts, ExecutionMode mode) {
    return g_api->SetSessionExecutionMode(opts, mode);
}

OrtStatusPtr ort_AddInitializer(OrtSessionOptions *opts, const char *name, const OrtValue *val) {
    return g_api->AddInitializer(opts, name, val);
}

// Session profiling

OrtStatusPtr ort_SessionEndProfiling(OrtSession *session, OrtAllocator *allocator, char **out) {
    return g_api->SessionEndProfiling(session, allocator, out);
}

// Value type

OrtStatusPtr ort_GetValueType(const OrtValue *value, enum ONNXType *out) {
    return g_api->GetValueType(value, out);
}

OrtStatusPtr ort_GetValueCount(const OrtValue *value, size_t *out) {
    return g_api->GetValueCount(value, out);
}

OrtStatusPtr ort_GetValue(const OrtValue *value, int index, OrtAllocator *allocator, OrtValue **out) {
    return g_api->GetValue(value, index, allocator, out);
}

// Memory info

OrtStatusPtr ort_CreateMemoryInfo(const char *name, enum OrtAllocatorType type,
    int id, enum OrtMemType mem_type, OrtMemoryInfo **out) {
    return g_api->CreateMemoryInfo(name, type, id, mem_type, out);
}

// Session options getters (since 1.27)

OrtStatusPtr ort_GetMemPatternEnabled(const OrtSessionOptions *opts, int *out) {
    return g_api->GetMemPatternEnabled(opts, out);
}

OrtStatusPtr ort_GetSessionExecutionMode(const OrtSessionOptions *opts, ExecutionMode *out) {
    return g_api->GetSessionExecutionMode(opts, out);
}

// Error handling

OrtErrorCode ort_GetErrorCode(const OrtStatus *status) {
    return g_api->GetErrorCode(status);
}

const char *ort_GetErrorMessage(const OrtStatus *status) {
    return g_api->GetErrorMessage(status);
}

void ort_ReleaseStatus(OrtStatus *status) {
    g_api->ReleaseStatus(status);
}

// Provider enumeration

OrtStatusPtr ort_GetAvailableProviders(char ***out, int *count) {
    return g_api->GetAvailableProviders(out, count);
}

OrtStatusPtr ort_ReleaseAvailableProviders(char **ptr, int count) {
    return g_api->ReleaseAvailableProviders(ptr, count);
}

// Allocator free

void ort_AllocatorFree(OrtAllocator *allocator, void *ptr) {
    allocator->Free(allocator, ptr);
}
