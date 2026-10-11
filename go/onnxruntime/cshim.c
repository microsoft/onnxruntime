#include "cshim.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT2 == 24, "UINT2 C API ABI value must remain 24");
_Static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT2 == 25, "INT2 C API ABI value must remain 25");
_Static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E8M0 == 26, "FLOAT8E8M0 C API ABI value must remain 26");

static const OrtApi *g_api = NULL;
static int g_api_version = 0;
static const char *g_ort_version = NULL;

void ort_reset_api(void) {
    g_api = NULL;
    g_api_version = 0;
    g_ort_version = NULL;
}

static int ort_first_api_version(const char *version) {
    if (version == NULL || version[0] != '1' || version[1] != '.') return ORT_GO_API_VERSION_MAX;
    const char *p = version + 2;
    if (*p < '0' || *p > '9') return ORT_GO_API_VERSION_MAX;
    int minor = 0;
    while (*p >= '0' && *p <= '9') {
        // Clamp while parsing so an oversized minor cannot overflow.
        if (minor < ORT_GO_API_VERSION_MAX) {
            minor = minor * 10 + (*p - '0');
            if (minor > ORT_GO_API_VERSION_MAX) minor = ORT_GO_API_VERSION_MAX;
        }
        p++;
    }
    if (*p++ != '.') return ORT_GO_API_VERSION_MAX;
    if (*p < '0' || *p > '9') return ORT_GO_API_VERSION_MAX;
    while (*p >= '0' && *p <= '9') p++;
    if (*p != '\0') return ORT_GO_API_VERSION_MAX;
    return minor;
}

int ort_init_api(void *get_api_base_fn, int *actual_version) {
    ort_reset_api();
    *actual_version = 0;
    const OrtApiBase *base = ((const OrtApiBase *(*)(void))get_api_base_fn)();
    if (base == NULL) return 1;
    g_ort_version = base->GetVersionString();
    // Release 1.N provides API N. Avoid rejected requests that ORT logs to stderr.
    for (int v = ort_first_api_version(g_ort_version); v >= ORT_GO_API_VERSION_MIN; v--) {
        g_api = base->GetApi((uint32_t)v);
        if (g_api != NULL) {
            g_api_version = v;
            *actual_version = v;
            return 0;
        }
    }
    ort_reset_api();
    return 2;
}

const char *ort_GetVersionString(void) {
    return g_ort_version;
}

// Call only after successful API initialization. Check versions before slot reads.
static OrtStatusPtr ort_unsupported(const char *operation, int required) {
    char message[192];
    snprintf(message, sizeof(message), "%s requires API %d; negotiated API %d (function unavailable)",
             operation, required, g_api_version);
    return g_api->CreateStatus(ORT_NOT_IMPLEMENTED, message);
}

OrtStatusPtr ort_GetBuildInfoString(const char **out) {
    if (g_api_version < 15) return ort_unsupported("GetBuildInfoString", 15);
    if (g_api->GetBuildInfoString == NULL) return ort_unsupported("GetBuildInfoString", 15);
    *out = g_api->GetBuildInfoString();
    return NULL;
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

OrtStatusPtr ort_CreateSession(const OrtEnv *env, const ORTCHAR_T *model_path,
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

OrtStatusPtr ort_EnableProfiling(OrtSessionOptions *opts, const ORTCHAR_T *prefix) {
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

// Value creation (sequence/map)

OrtStatusPtr ort_CreateValue(const OrtValue *const *in, size_t num_values,
    enum ONNXType value_type, OrtValue **out) {
    return g_api->CreateValue(in, num_values, value_type, out);
}

// Memory info

OrtStatusPtr ort_CreateMemoryInfo(const char *name, enum OrtAllocatorType type,
    int id, enum OrtMemType mem_type, OrtMemoryInfo **out) {
    return g_api->CreateMemoryInfo(name, type, id, mem_type, out);
}

// Session options getters (since 1.27)

OrtStatusPtr ort_GetMemPatternEnabled(const OrtSessionOptions *opts, int *out) {
    if (g_api_version < 27) return ort_unsupported("GetMemPatternEnabled", 27);
    if (g_api->GetMemPatternEnabled == NULL) return ort_unsupported("GetMemPatternEnabled", 27);
    return g_api->GetMemPatternEnabled(opts, out);
}

OrtStatusPtr ort_GetSessionExecutionMode(const OrtSessionOptions *opts, ExecutionMode *out) {
    if (g_api_version < 27) return ort_unsupported("GetSessionExecutionMode", 27);
    if (g_api->GetSessionExecutionMode == NULL) return ort_unsupported("GetSessionExecutionMode", 27);
    return g_api->GetSessionExecutionMode(opts, out);
}

// Telemetry

OrtStatusPtr ort_EnableTelemetryEvents(const OrtEnv *env) {
    return g_api->EnableTelemetryEvents(env);
}

OrtStatusPtr ort_DisableTelemetryEvents(const OrtEnv *env) {
    return g_api->DisableTelemetryEvents(env);
}

// Additional session options (continued)

OrtStatusPtr ort_SetOptimizedModelFilePath(OrtSessionOptions *opts, const ORTCHAR_T *path) {
    return g_api->SetOptimizedModelFilePath(opts, path);
}

OrtStatusPtr ort_RegisterCustomOpsLibrary_V2(OrtSessionOptions *opts, const ORTCHAR_T *path) {
    return g_api->RegisterCustomOpsLibrary_V2(opts, path);
}

OrtStatusPtr ort_HasSessionConfigEntry(const OrtSessionOptions *opts, const char *key, int *out) {
    return g_api->HasSessionConfigEntry(opts, key, out);
}

OrtStatusPtr ort_GetSessionConfigEntry(const OrtSessionOptions *opts, const char *key,
    char *value, size_t *size) {
    return g_api->GetSessionConfigEntry(opts, key, value, size);
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

// Plugin execution providers. Version checks precede all API-22 slot reads.

OrtStatusPtr ort_RegisterExecutionProviderLibrary(OrtEnv* env, const char* name, const ORTCHAR_T* path) {
    if (g_api_version < 22) return ort_unsupported("RegisterExecutionProviderLibrary", 22);
    if (g_api->RegisterExecutionProviderLibrary == NULL) return ort_unsupported("RegisterExecutionProviderLibrary", 22);
    return g_api->RegisterExecutionProviderLibrary(env, name, path);
}

OrtStatusPtr ort_UnregisterExecutionProviderLibrary(OrtEnv* env, const char* name) {
    if (g_api_version < 22) return ort_unsupported("UnregisterExecutionProviderLibrary", 22);
    if (g_api->UnregisterExecutionProviderLibrary == NULL) return ort_unsupported("UnregisterExecutionProviderLibrary", 22);
    return g_api->UnregisterExecutionProviderLibrary(env, name);
}

OrtStatusPtr ort_GetEpDevices(const OrtEnv* env, const OrtEpDevice* const** out, size_t* count) {
    if (g_api_version < 22) return ort_unsupported("GetEpDevices", 22);
    if (g_api->GetEpDevices == NULL) return ort_unsupported("GetEpDevices", 22);
    return g_api->GetEpDevices(env, out, count);
}

OrtStatusPtr ort_SessionOptionsAppendExecutionProvider_V2(
    OrtSessionOptions* opts, OrtEnv* env,
    const OrtEpDevice* const* devices, size_t count,
    const char* const* keys, const char* const* values, size_t num_options) {
    if (g_api_version < 22) return ort_unsupported("SessionOptionsAppendExecutionProvider_V2", 22);
    if (g_api->SessionOptionsAppendExecutionProvider_V2 == NULL)
        return ort_unsupported("SessionOptionsAppendExecutionProvider_V2", 22);
    return g_api->SessionOptionsAppendExecutionProvider_V2(opts, env, devices, count, keys, values, num_options);
}

OrtStatusPtr ort_SessionOptionsSetEpSelectionPolicy(OrtSessionOptions* opts, OrtExecutionProviderDevicePolicy policy) {
    if (g_api_version < 22) return ort_unsupported("SessionOptionsSetEpSelectionPolicy", 22);
    if (g_api->SessionOptionsSetEpSelectionPolicy == NULL) return ort_unsupported("SessionOptionsSetEpSelectionPolicy", 22);
    return g_api->SessionOptionsSetEpSelectionPolicy(opts, policy);
}

OrtStatusPtr ort_EpDevice_EpName(const OrtEpDevice* device, const char** out) {
    if (g_api_version < 22) return ort_unsupported("EpDevice_EpName", 22);
    if (g_api->EpDevice_EpName == NULL) return ort_unsupported("EpDevice_EpName", 22);
    *out = g_api->EpDevice_EpName(device);
    return NULL;
}

OrtStatusPtr ort_EpDevice_EpVendor(const OrtEpDevice* device, const char** out) {
    if (g_api_version < 22) return ort_unsupported("EpDevice_EpVendor", 22);
    if (g_api->EpDevice_EpVendor == NULL) return ort_unsupported("EpDevice_EpVendor", 22);
    *out = g_api->EpDevice_EpVendor(device);
    return NULL;
}

OrtStatusPtr ort_EpDevice_EpMetadata(const OrtEpDevice* device, const OrtKeyValuePairs** out) {
    if (g_api_version < 22) return ort_unsupported("EpDevice_EpMetadata", 22);
    if (g_api->EpDevice_EpMetadata == NULL) return ort_unsupported("EpDevice_EpMetadata", 22);
    *out = g_api->EpDevice_EpMetadata(device);
    return NULL;
}

OrtStatusPtr ort_EpDevice_EpOptions(const OrtEpDevice* device, const OrtKeyValuePairs** out) {
    if (g_api_version < 22) return ort_unsupported("EpDevice_EpOptions", 22);
    if (g_api->EpDevice_EpOptions == NULL) return ort_unsupported("EpDevice_EpOptions", 22);
    *out = g_api->EpDevice_EpOptions(device);
    return NULL;
}

OrtStatusPtr ort_EpDevice_Device(const OrtEpDevice* device, const OrtHardwareDevice** out) {
    if (g_api_version < 22) return ort_unsupported("EpDevice_Device", 22);
    if (g_api->EpDevice_Device == NULL) return ort_unsupported("EpDevice_Device", 22);
    *out = g_api->EpDevice_Device(device);
    return NULL;
}

OrtStatusPtr ort_HardwareDevice_Type(const OrtHardwareDevice* device, OrtHardwareDeviceType* out) {
    if (g_api_version < 22) return ort_unsupported("HardwareDevice_Type", 22);
    if (g_api->HardwareDevice_Type == NULL) return ort_unsupported("HardwareDevice_Type", 22);
    *out = g_api->HardwareDevice_Type(device);
    return NULL;
}

OrtStatusPtr ort_HardwareDevice_VendorId(const OrtHardwareDevice* device, uint32_t* out) {
    if (g_api_version < 22) return ort_unsupported("HardwareDevice_VendorId", 22);
    if (g_api->HardwareDevice_VendorId == NULL) return ort_unsupported("HardwareDevice_VendorId", 22);
    *out = g_api->HardwareDevice_VendorId(device);
    return NULL;
}

OrtStatusPtr ort_HardwareDevice_Vendor(const OrtHardwareDevice* device, const char** out) {
    if (g_api_version < 22) return ort_unsupported("HardwareDevice_Vendor", 22);
    if (g_api->HardwareDevice_Vendor == NULL) return ort_unsupported("HardwareDevice_Vendor", 22);
    *out = g_api->HardwareDevice_Vendor(device);
    return NULL;
}

OrtStatusPtr ort_HardwareDevice_DeviceId(const OrtHardwareDevice* device, uint32_t* out) {
    if (g_api_version < 22) return ort_unsupported("HardwareDevice_DeviceId", 22);
    if (g_api->HardwareDevice_DeviceId == NULL) return ort_unsupported("HardwareDevice_DeviceId", 22);
    *out = g_api->HardwareDevice_DeviceId(device);
    return NULL;
}

OrtStatusPtr ort_HardwareDevice_Metadata(const OrtHardwareDevice* device, const OrtKeyValuePairs** out) {
    if (g_api_version < 22) return ort_unsupported("HardwareDevice_Metadata", 22);
    if (g_api->HardwareDevice_Metadata == NULL) return ort_unsupported("HardwareDevice_Metadata", 22);
    *out = g_api->HardwareDevice_Metadata(device);
    return NULL;
}

OrtStatusPtr ort_GetKeyValuePairs(
    const OrtKeyValuePairs* pairs, const char* const** keys,
    const char* const** values, size_t* count) {
    if (g_api_version < 22) return ort_unsupported("GetKeyValuePairs", 22);
    if (g_api->GetKeyValuePairs == NULL) return ort_unsupported("GetKeyValuePairs", 22);
    g_api->GetKeyValuePairs(pairs, keys, values, count);
    return NULL;
}

OrtStatusPtr ort_SetDeterministicCompute(OrtSessionOptions* opts, bool enabled) {
    if (g_api_version < 17) return ort_unsupported("SetDeterministicCompute", 17);
    if (g_api->SetDeterministicCompute == NULL) return ort_unsupported("SetDeterministicCompute", 17);
    return g_api->SetDeterministicCompute(opts, enabled);
}

OrtStatusPtr ort_SynchronizeBoundInputs(OrtIoBinding* binding) {
    if (g_api_version < 10) return ort_unsupported("SynchronizeBoundInputs", 10);
    if (g_api->SynchronizeBoundInputs == NULL) return ort_unsupported("SynchronizeBoundInputs", 10);
    return g_api->SynchronizeBoundInputs(binding);
}

OrtStatusPtr ort_SynchronizeBoundOutputs(OrtIoBinding* binding) {
    if (g_api_version < 10) return ort_unsupported("SynchronizeBoundOutputs", 10);
    if (g_api->SynchronizeBoundOutputs == NULL) return ort_unsupported("SynchronizeBoundOutputs", 10);
    return g_api->SynchronizeBoundOutputs(binding);
}

OrtStatusPtr ort_SessionOptionsSetWeightlessSourceModelBuffer(OrtSessionOptions* opts, const void* data, size_t size) {
    if (g_api_version < 29) return ort_unsupported("SessionOptionsSetWeightlessSourceModelBuffer", 29);
    if (g_api->SessionOptionsSetWeightlessSourceModelBuffer == NULL)
        return ort_unsupported("SessionOptionsSetWeightlessSourceModelBuffer", 29);
    return g_api->SessionOptionsSetWeightlessSourceModelBuffer(opts, data, size);
}

OrtStatusPtr ort_copy_buffer(const void* data, size_t size, void** out) {
    if (data == NULL || size == 0) return g_api->CreateStatus(ORT_INVALID_ARGUMENT, "source buffer is empty");
    void* copy = malloc(size);
    if (copy == NULL) return g_api->CreateStatus(ORT_FAIL, "source buffer allocation failed");
    memcpy(copy, data, size);
    *out = copy;
    return NULL;
}

OrtStatusPtr ort_SessionOptionsSetEpContextDataReadFunc(
    OrtSessionOptions* options, OrtReadNamedBufferFunc reader, void* state) {
    if (g_api_version < 31) return ort_unsupported("SessionOptionsSetEpContextDataReadFunc", 31);
    if (g_api->SessionOptionsSetEpContextDataReadFunc == NULL)
        return ort_unsupported("SessionOptionsSetEpContextDataReadFunc", 31);
    return g_api->SessionOptionsSetEpContextDataReadFunc(options, reader, state);
}

typedef struct { uintptr_t handle; } OrtGoReaderState;
extern OrtStatus* ortGoReadNamedBuffer(uintptr_t, char*, OrtAllocator*, void**, size_t*);

OrtStatusPtr ort_reader_state(uintptr_t handle, void** out) {
    OrtGoReaderState* state = malloc(sizeof(*state));
    if (state == NULL) return g_api->CreateStatus(ORT_FAIL, "reader state allocation failed");
    state->handle = handle;
    *out = state;
    return NULL;
}

OrtStatus* ORT_API_CALL ort_reader_trampoline(
    void* state, const char* name, OrtAllocator* allocator, void** buffer, size_t* size) {
    if (buffer != NULL) *buffer = NULL;
    if (size != NULL) *size = 0;
    if (state == NULL || name == NULL || allocator == NULL || buffer == NULL || size == NULL)
        return g_api->CreateStatus(ORT_INVALID_ARGUMENT, "invalid reader callback arguments");
    OrtGoReaderState* registration = state;
    return ortGoReadNamedBuffer(registration->handle, (char*)name, allocator, buffer, size);
}

OrtStatusPtr ort_reader_validate(OrtAllocator* allocator) {
    if (allocator == NULL || allocator->Alloc == NULL || allocator->Free == NULL || allocator->Info == NULL)
        return g_api->CreateStatus(ORT_INVALID_ARGUMENT, "invalid reader allocator");
    const OrtMemoryInfo* info = NULL;
    OrtStatus* status = g_api->AllocatorGetInfo(allocator, &info);
    if (status != NULL) return status;
    if (info == NULL) return g_api->CreateStatus(ORT_INVALID_ARGUMENT, "reader allocator has no memory info");
    OrtMemoryInfoDeviceType type;
    g_api->MemoryInfoGetDeviceType(info, &type);
    if (type != OrtMemoryInfoDeviceType_CPU &&
        g_api->MemoryInfoGetDeviceMemType(info) != OrtDeviceMemoryType_HOST_ACCESSIBLE)
        return g_api->CreateStatus(ORT_INVALID_ARGUMENT, "reader allocator memory is not host accessible");
    return NULL;
}

OrtStatusPtr ort_reader_copy(OrtAllocator* allocator, const void* data, size_t size,
                             void** out, size_t* out_size) {
    if (size == 0) return NULL;
    void* candidate = NULL;
    OrtStatus* status = g_api->AllocatorAlloc(allocator, size, &candidate);
    if (status != NULL) {
        if (candidate != NULL) {
            OrtStatus* cleanup = g_api->AllocatorFree(allocator, candidate);
            if (cleanup != NULL) g_api->ReleaseStatus(cleanup);
        }
        return status;
    }
    if (candidate == NULL) return g_api->CreateStatus(ORT_FAIL, "reader allocator returned NULL");
    memcpy(candidate, data, size);
    *out = candidate;
    *out_size = size;
    return NULL;
}

OrtStatusPtr ort_reader_error(OrtErrorCode code, const char* message, size_t size) {
    if (size == SIZE_MAX) return g_api->CreateStatus(ORT_FAIL, "reader error message too large");
    char* copy = malloc(size + 1);
    if (copy == NULL) return g_api->CreateStatus(ORT_FAIL, "reader error allocation failed");
    if (size != 0) memcpy(copy, message, size);
    copy[size] = 0;
    OrtStatus* status = g_api->CreateStatus(code, copy);
    free(copy);
    return status;
}

OrtStatusPtr ort_reader_panic(void) {
    return g_api->CreateStatus(ORT_FAIL, "EPContext data reader panicked");
}
