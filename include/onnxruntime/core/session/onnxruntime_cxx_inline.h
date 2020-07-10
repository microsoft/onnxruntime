// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Do not include this file directly. Please include "onnxruntime_cxx_api.h" instead.
// If interested in trying out features of the new experimental C++ API, include "experimental_onnxruntime_cxx_api.h" instead.
//
// These are the inline implementations of the C++ header APIs. They're in this separate file as to not clutter
// the main C++ file with implementation details.

namespace Ort {

inline void ThrowOnError(const OrtApi& ort, OrtStatus* status) {
  if (status) {
    std::string error_message = ort.GetErrorMessage(status);
    OrtErrorCode error_code = ort.GetErrorCode(status);
    ort.ReleaseStatus(status);
    throw Ort::Exception(std::move(error_message), error_code);
  }
}

inline void ThrowOnError(OrtStatus* status) {
  ThrowOnError(Global<void>::api_, status);
}

// This template converts a C++ type into it's ONNXTensorElementDataType
template <typename T>
struct TypeToTensorType;
template <>
struct TypeToTensorType<float> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };
template <>
struct TypeToTensorType<double> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE; };
template <>
struct TypeToTensorType<int8_t> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8; };
template <>
struct TypeToTensorType<int16_t> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16; };
template <>
struct TypeToTensorType<int32_t> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32; };
template <>
struct TypeToTensorType<int64_t> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64; };
template <>
struct TypeToTensorType<uint8_t> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8; };
template <>
struct TypeToTensorType<uint16_t> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16; };
template <>
struct TypeToTensorType<uint32_t> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32; };
template <>
struct TypeToTensorType<uint64_t> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64; };
template <>
struct TypeToTensorType<bool> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL; };

inline AllocatorWithDefaultOptions::AllocatorWithDefaultOptions() {
  ThrowOnError(Global<void>::api_.GetAllocatorWithDefaultOptions(&p_));
}

inline void* AllocatorWithDefaultOptions::Alloc(size_t size) {
  void* out;
  ThrowOnError(Global<void>::api_.AllocatorAlloc(p_, size, &out));
  return out;
}

inline void AllocatorWithDefaultOptions::Free(void* p) {
  ThrowOnError(Global<void>::api_.AllocatorFree(p_, p));
}

inline const OrtMemoryInfo* AllocatorWithDefaultOptions::GetInfo() const {
  const OrtMemoryInfo* out;
  ThrowOnError(Global<void>::api_.AllocatorGetInfo(p_, &out));
  return out;
}

inline MemoryInfo MemoryInfo::CreateCpu(OrtAllocatorType type, OrtMemType mem_type) {
  OrtMemoryInfo* p;
  ThrowOnError(Global<void>::api_.CreateCpuMemoryInfo(type, mem_type, &p));
  return MemoryInfo(p);
}

inline MemoryInfo::MemoryInfo(const char* name, OrtAllocatorType type, int id, OrtMemType mem_type) {
  ThrowOnError(Global<void>::api_.CreateMemoryInfo(name, type, id, mem_type, &p_));
}

inline Env::Env(OrtLoggingLevel default_warning_level, _In_ const char* logid) {
  ThrowOnError(Global<void>::api_.CreateEnv(default_warning_level, logid, &p_));
}

inline Env::Env(OrtLoggingLevel default_warning_level, const char* logid, OrtLoggingFunction logging_function, void* logger_param) {
  ThrowOnError(Global<void>::api_.CreateEnvWithCustomLogger(logging_function, logger_param, default_warning_level, logid, &p_));
}

inline Env::Env(const OrtThreadingOptions* tp_options, OrtLoggingLevel default_warning_level, _In_ const char* logid) {
  ThrowOnError(Global<void>::api_.CreateEnvWithGlobalThreadPools(default_warning_level, logid, tp_options, &p_));
}

inline Env& Env::EnableTelemetryEvents() {
  ThrowOnError(Global<void>::api_.EnableTelemetryEvents(p_));
  return *this;
}

inline Env& Env::DisableTelemetryEvents() {
  ThrowOnError(Global<void>::api_.DisableTelemetryEvents(p_));
  return *this;
}

inline CustomOpDomain::CustomOpDomain(const char* domain) {
  ThrowOnError(Global<void>::api_.CreateCustomOpDomain(domain, &p_));
}

inline void CustomOpDomain::Add(OrtCustomOp* op) {
  ThrowOnError(Global<void>::api_.CustomOpDomain_Add(p_, op));
}

inline RunOptions::RunOptions() {
  ThrowOnError(Global<void>::api_.CreateRunOptions(&p_));
}

inline RunOptions& RunOptions::SetRunLogVerbosityLevel(int level) {
  ThrowOnError(Global<void>::api_.RunOptionsSetRunLogVerbosityLevel(p_, level));
  return *this;
}

inline RunOptions& RunOptions::SetRunLogSeverityLevel(int level) {
  ThrowOnError(Global<void>::api_.RunOptionsSetRunLogSeverityLevel(p_, level));
  return *this;
}

inline int RunOptions::GetRunLogVerbosityLevel() const {
  int out;
  ThrowOnError(Global<void>::api_.RunOptionsGetRunLogVerbosityLevel(p_, &out));
  return out;
}

inline RunOptions& RunOptions::SetRunTag(const char* run_tag) {
  ThrowOnError(Global<void>::api_.RunOptionsSetRunTag(p_, run_tag));
  return *this;
}

inline const char* RunOptions::GetRunTag() const {
  const char* out;
  ThrowOnError(Global<void>::api_.RunOptionsGetRunTag(p_, &out));
  return out;
}

inline RunOptions& RunOptions::SetTerminate() {
  ThrowOnError(Global<void>::api_.RunOptionsSetTerminate(p_));
  return *this;
}

inline RunOptions& RunOptions::UnsetTerminate() {
  ThrowOnError(Global<void>::api_.RunOptionsUnsetTerminate(p_));
  return *this;
}

inline SessionOptions::SessionOptions() {
  ThrowOnError(Global<void>::api_.CreateSessionOptions(&p_));
}

inline SessionOptions SessionOptions::Clone() const {
  OrtSessionOptions* out;
  ThrowOnError(Global<void>::api_.CloneSessionOptions(p_, &out));
  return SessionOptions{out};
}

inline SessionOptions& SessionOptions::SetIntraOpNumThreads(int intra_op_num_threads) {
  ThrowOnError(Global<void>::api_.SetIntraOpNumThreads(p_, intra_op_num_threads));
  return *this;
}

inline SessionOptions& SessionOptions::SetInterOpNumThreads(int inter_op_num_threads) {
  ThrowOnError(Global<void>::api_.SetInterOpNumThreads(p_, inter_op_num_threads));
  return *this;
}

inline SessionOptions& SessionOptions::SetGraphOptimizationLevel(GraphOptimizationLevel graph_optimization_level) {
  ThrowOnError(Global<void>::api_.SetSessionGraphOptimizationLevel(p_, graph_optimization_level));
  return *this;
}

inline SessionOptions& SessionOptions::SetOptimizedModelFilePath(const ORTCHAR_T* optimized_model_filepath) {
  ThrowOnError(Global<void>::api_.SetOptimizedModelFilePath(p_, optimized_model_filepath));
  return *this;
}

inline SessionOptions& SessionOptions::EnableProfiling(const ORTCHAR_T* profile_file_prefix) {
  ThrowOnError(Global<void>::api_.EnableProfiling(p_, profile_file_prefix));
  return *this;
}

inline SessionOptions& SessionOptions::DisableProfiling() {
  ThrowOnError(Global<void>::api_.DisableProfiling(p_));
  return *this;
}

inline SessionOptions& SessionOptions::EnableMemPattern() {
  ThrowOnError(Global<void>::api_.EnableMemPattern(p_));
  return *this;
}

inline SessionOptions& SessionOptions::DisableMemPattern() {
  ThrowOnError(Global<void>::api_.DisableMemPattern(p_));
  return *this;
}

inline SessionOptions& SessionOptions::EnableCpuMemArena() {
  ThrowOnError(Global<void>::api_.EnableCpuMemArena(p_));
  return *this;
}

inline SessionOptions& SessionOptions::DisableCpuMemArena() {
  ThrowOnError(Global<void>::api_.DisableCpuMemArena(p_));
  return *this;
}

inline SessionOptions& SessionOptions::SetExecutionMode(ExecutionMode execution_mode) {
  ThrowOnError(Global<void>::api_.SetSessionExecutionMode(p_, execution_mode));
  return *this;
}

inline SessionOptions& SessionOptions::SetLogId(const char* logid) {
  ThrowOnError(Global<void>::api_.SetSessionLogId(p_, logid));
  return *this;
}
inline SessionOptions& SessionOptions::Add(OrtCustomOpDomain* custom_op_domain) {
  ThrowOnError(Global<void>::api_.AddCustomOpDomain(p_, custom_op_domain));
  return *this;
}

inline Session::Session(Env& env, const ORTCHAR_T* model_path, const SessionOptions& options) {
  ThrowOnError(Global<void>::api_.CreateSession(env, model_path, options, &p_));
}

inline Session::Session(Env& env, const void* model_data, size_t model_data_length, const SessionOptions& options) {
  ThrowOnError(Global<void>::api_.CreateSessionFromArray(env, model_data, model_data_length, options, &p_));
}

inline std::vector<Value> Session::Run(const RunOptions& run_options, const char* const* input_names, const Value* input_values, size_t input_count,
                                       const char* const* output_names, size_t output_names_count) {
  std::vector<Ort::Value> output_values;
  for (size_t i = 0; i < output_names_count; i++)
    output_values.emplace_back(nullptr);
  Run(run_options, input_names, input_values, input_count, output_names, output_values.data(), output_names_count);
  return output_values;
}

inline void Session::Run(const RunOptions& run_options, const char* const* input_names, const Value* input_values, size_t input_count,
                         const char* const* output_names, Value* output_values, size_t output_count) {
  static_assert(sizeof(Value) == sizeof(OrtValue*), "Value is really just an array of OrtValue* in memory, so we can reinterpret_cast safely");
  auto ort_input_values = reinterpret_cast<const OrtValue**>(const_cast<Value*>(input_values));
  auto ort_output_values = reinterpret_cast<OrtValue**>(output_values);
  ThrowOnError(Global<void>::api_.Run(p_, run_options, input_names, ort_input_values, input_count, output_names, output_count, ort_output_values));
}

inline size_t Session::GetInputCount() const {
  size_t out;
  ThrowOnError(Global<void>::api_.SessionGetInputCount(p_, &out));
  return out;
}

inline size_t Session::GetOutputCount() const {
  size_t out;
  ThrowOnError(Global<void>::api_.SessionGetOutputCount(p_, &out));
  return out;
}

inline size_t Session::GetOverridableInitializerCount() const {
  size_t out;
  ThrowOnError(Global<void>::api_.SessionGetOverridableInitializerCount(p_, &out));
  return out;
}

inline char* Session::GetInputName(size_t index, OrtAllocator* allocator) const {
  char* out;
  ThrowOnError(Global<void>::api_.SessionGetInputName(p_, index, allocator, &out));
  return out;
}

inline char* Session::GetOutputName(size_t index, OrtAllocator* allocator) const {
  char* out;
  ThrowOnError(Global<void>::api_.SessionGetOutputName(p_, index, allocator, &out));
  return out;
}

inline char* Session::GetOverridableInitializerName(size_t index, OrtAllocator* allocator) const {
  char* out;
  ThrowOnError(Global<void>::api_.SessionGetOverridableInitializerName(p_, index, allocator, &out));
  return out;
}

inline char* Session::EndProfiling(OrtAllocator* allocator) const {
  char* out;
  ThrowOnError(Global<void>::api_.SessionEndProfiling(p_, allocator, &out));
  return out;
}

inline ModelMetadata Session::GetModelMetadata() const {
  OrtModelMetadata* out;
  ThrowOnError(Global<void>::api_.SessionGetModelMetadata(p_, &out));
  return ModelMetadata{out};
}

inline char* ModelMetadata::GetProducerName(OrtAllocator* allocator) const {
  char* out;
  ThrowOnError(Global<void>::api_.ModelMetadataGetProducerName(p_, allocator, &out));
  return out;
}

inline char* ModelMetadata::GetGraphName(OrtAllocator* allocator) const {
  char* out;
  ThrowOnError(Global<void>::api_.ModelMetadataGetGraphName(p_, allocator, &out));
  return out;
}

inline char* ModelMetadata::GetDomain(OrtAllocator* allocator) const {
  char* out;
  ThrowOnError(Global<void>::api_.ModelMetadataGetDomain(p_, allocator, &out));
  return out;
}

inline char* ModelMetadata::GetDescription(OrtAllocator* allocator) const {
  char* out;
  ThrowOnError(Global<void>::api_.ModelMetadataGetDescription(p_, allocator, &out));
  return out;
}

inline char* ModelMetadata::LookupCustomMetadataMap(const char* key, OrtAllocator* allocator) const {
  char* out;
  ThrowOnError(Global<void>::api_.ModelMetadataLookupCustomMetadataMap(p_, allocator, key, &out));
  return out;
}

inline char** ModelMetadata::GetCustomMetadataMapKeys(OrtAllocator* allocator, _Out_ int64_t& num_keys) const {
  char** out;
  ThrowOnError(Global<void>::api_.ModelMetadataGetCustomMetadataMapKeys(p_, allocator, &out, &num_keys));
  return out;
}

inline int64_t ModelMetadata::GetVersion() const {
  int64_t out;
  ThrowOnError(Global<void>::api_.ModelMetadataGetVersion(p_, &out));
  return out;
}

inline TypeInfo Session::GetInputTypeInfo(size_t index) const {
  OrtTypeInfo* out;
  ThrowOnError(Global<void>::api_.SessionGetInputTypeInfo(p_, index, &out));
  return TypeInfo{out};
}

inline TypeInfo Session::GetOutputTypeInfo(size_t index) const {
  OrtTypeInfo* out;
  ThrowOnError(Global<void>::api_.SessionGetOutputTypeInfo(p_, index, &out));
  return TypeInfo{out};
}

inline TypeInfo Session::GetOverridableInitializerTypeInfo(size_t index) const {
  OrtTypeInfo* out;
  ThrowOnError(Global<void>::api_.SessionGetOverridableInitializerTypeInfo(p_, index, &out));
  return TypeInfo{out};
}

inline ONNXTensorElementDataType TensorTypeAndShapeInfo::GetElementType() const {
  ONNXTensorElementDataType out;
  ThrowOnError(Global<void>::api_.GetTensorElementType(p_, &out));
  return out;
}

inline size_t TensorTypeAndShapeInfo::GetElementCount() const {
  size_t out;
  ThrowOnError(Global<void>::api_.GetTensorShapeElementCount(p_, &out));
  return static_cast<size_t>(out);
}

inline size_t TensorTypeAndShapeInfo::GetDimensionsCount() const {
  size_t out;
  ThrowOnError(Global<void>::api_.GetDimensionsCount(p_, &out));
  return out;
}

inline void TensorTypeAndShapeInfo::GetDimensions(int64_t* values, size_t values_count) const {
  ThrowOnError(Global<void>::api_.GetDimensions(p_, values, values_count));
}

inline void TensorTypeAndShapeInfo::GetSymbolicDimensions(const char** values, size_t values_count) const {
  ThrowOnError(Global<void>::api_.GetSymbolicDimensions(p_, values, values_count));
}

inline std::vector<int64_t> TensorTypeAndShapeInfo::GetShape() const {
  std::vector<int64_t> out(GetDimensionsCount(), 0);
  GetDimensions(out.data(), out.size());
  return out;
}

inline Unowned<TensorTypeAndShapeInfo> TypeInfo::GetTensorTypeAndShapeInfo() const {
  const OrtTensorTypeAndShapeInfo* out;
  ThrowOnError(Global<void>::api_.CastTypeInfoToTensorInfo(p_, &out));
  return Unowned<TensorTypeAndShapeInfo>{const_cast<OrtTensorTypeAndShapeInfo*>(out)};
}

inline ONNXType TypeInfo::GetONNXType() const {
  ONNXType out;
  ThrowOnError(Global<void>::api_.GetOnnxTypeFromTypeInfo(p_, &out));
  return out;
}

template <typename T>
inline Value Value::CreateTensor(const OrtMemoryInfo* info, T* p_data, size_t p_data_element_count, const int64_t* shape, size_t shape_len) {
  return CreateTensor(info, p_data, p_data_element_count * sizeof(T), shape, shape_len, TypeToTensorType<T>::type);
}

inline Value Value::CreateTensor(const OrtMemoryInfo* info, void* p_data, size_t p_data_byte_count, const int64_t* shape, size_t shape_len,
                                 ONNXTensorElementDataType type) {
  OrtValue* out;
  ThrowOnError(Global<void>::api_.CreateTensorWithDataAsOrtValue(info, p_data, p_data_byte_count, shape, shape_len, type, &out));
  return Value{out};
}

template <typename T>
inline Value Value::CreateTensor(OrtAllocator* allocator, const int64_t* shape, size_t shape_len) {
  return CreateTensor(allocator, shape, shape_len, TypeToTensorType<T>::type);
}

inline Value Value::CreateTensor(OrtAllocator* allocator, const int64_t* shape, size_t shape_len, ONNXTensorElementDataType type) {
  OrtValue* out;
  ThrowOnError(Global<void>::api_.CreateTensorAsOrtValue(allocator, shape, shape_len, type, &out));
  return Value{out};
}

inline Value Value::CreateMap(Value& keys, Value& values) {
  OrtValue* out;
  OrtValue* inputs[2] = {keys, values};
  ThrowOnError(Global<void>::api_.CreateValue(inputs, 2, ONNX_TYPE_MAP, &out));
  return Value{out};
}

inline Value Value::CreateSequence(std::vector<Value>& values) {
  OrtValue* out;
  std::vector<OrtValue*> values_ort{values.data(), values.data() + values.size()};
  ThrowOnError(Global<void>::api_.CreateValue(values_ort.data(), values_ort.size(), ONNX_TYPE_SEQUENCE, &out));
  return Value{out};
}

template <typename T>
inline Value Value::CreateOpaque(const char* domain, const char* type_name, const T& data_container) {
  OrtValue* out;
  ThrowOnError(Global<void>::api_.CreateOpaqueValue(domain, type_name, &data_container, sizeof(T), &out));
  return Value{out};
}

template <typename T>
inline void Value::GetOpaqueData(const char* domain, const char* type_name, T& out) {
  ThrowOnError(Global<void>::api_.GetOpaqueValue(domain, type_name, p_, &out, sizeof(T)));
}

inline bool Value::IsTensor() const {
  int out;
  ThrowOnError(Global<void>::api_.IsTensor(p_, &out));
  return out != 0;
}

inline size_t Value::GetCount() const {
  size_t out;
  ThrowOnError(Global<void>::api_.GetValueCount(p_, &out));
  return out;
}

inline Value Value::GetValue(int index, OrtAllocator* allocator) const {
  OrtValue* out;
  ThrowOnError(Global<void>::api_.GetValue(p_, index, allocator, &out));
  return Value{out};
}

inline size_t Value::GetStringTensorDataLength() const {
  size_t out;
  ThrowOnError(Global<void>::api_.GetStringTensorDataLength(p_, &out));
  return out;
}

inline void Value::GetStringTensorContent(void* buffer, size_t buffer_length, size_t* offsets, size_t offsets_count) const {
  ThrowOnError(Global<void>::api_.GetStringTensorContent(p_, buffer, buffer_length, offsets, offsets_count));
}

template <typename T>
T* Value::GetTensorMutableData() {
  T* out;
  ThrowOnError(Global<void>::api_.GetTensorMutableData(p_, (void**)&out));
  return out;
}

inline TypeInfo Value::GetTypeInfo() const {
  OrtTypeInfo* output;
  ThrowOnError(Global<void>::api_.GetTypeInfo(p_, &output));
  return TypeInfo{output};
}

inline TensorTypeAndShapeInfo Value::GetTensorTypeAndShapeInfo() const {
  OrtTensorTypeAndShapeInfo* output;
  ThrowOnError(Global<void>::api_.GetTensorTypeAndShape(p_, &output));
  return TensorTypeAndShapeInfo{output};
}

//
// Custom OP API Inlines
//
inline void CustomOpApi::ThrowOnError(OrtStatus* status) {
  Ort::ThrowOnError(api_, status);
}

template <>
inline float CustomOpApi::KernelInfoGetAttribute<float>(_In_ const OrtKernelInfo* info, _In_ const char* name) {
  float out;
  ThrowOnError(api_.KernelInfoGetAttribute_float(info, name, &out));
  return out;
}

template <>
inline int64_t CustomOpApi::KernelInfoGetAttribute<int64_t>(_In_ const OrtKernelInfo* info, _In_ const char* name) {
  int64_t out;
  ThrowOnError(api_.KernelInfoGetAttribute_int64(info, name, &out));
  return out;
}

template <>
inline std::string CustomOpApi::KernelInfoGetAttribute<std::string>(_In_ const OrtKernelInfo* info, _In_ const char* name) {
  size_t size = 0;
  std::string out;
  OrtStatus* status = api_.KernelInfoGetAttribute_string(info, name, nullptr, &size);

  // The status should be ORT_INVALID_ARGUMENT because the size is insufficient to hold the string
  if (api_.GetErrorCode(status) == ORT_INVALID_ARGUMENT) {
    api_.ReleaseStatus(status);
    out.resize(size);
    ThrowOnError(api_.KernelInfoGetAttribute_string(info, name, &out[0], &size));
    out.resize(size - 1);  // remove the terminating character '\0'
  } else {
    ThrowOnError(status);
  }
  return out;
}

inline OrtTensorTypeAndShapeInfo* CustomOpApi::GetTensorTypeAndShape(_In_ const OrtValue* value) {
  OrtTensorTypeAndShapeInfo* out;
  ThrowOnError(api_.GetTensorTypeAndShape(value, &out));
  return out;
}

inline size_t CustomOpApi::GetTensorShapeElementCount(_In_ const OrtTensorTypeAndShapeInfo* info) {
  size_t out;
  ThrowOnError(api_.GetTensorShapeElementCount(info, &out));
  return out;
}

inline ONNXTensorElementDataType CustomOpApi::GetTensorElementType(const OrtTensorTypeAndShapeInfo* info) {
  ONNXTensorElementDataType out;
  ThrowOnError(api_.GetTensorElementType(info, &out));
  return out;
}

inline size_t CustomOpApi::GetDimensionsCount(_In_ const OrtTensorTypeAndShapeInfo* info) {
  size_t out;
  ThrowOnError(api_.GetDimensionsCount(info, &out));
  return out;
}

inline void CustomOpApi::GetDimensions(_In_ const OrtTensorTypeAndShapeInfo* info, _Out_ int64_t* dim_values, size_t dim_values_length) {
  ThrowOnError(api_.GetDimensions(info, dim_values, dim_values_length));
}

inline void CustomOpApi::SetDimensions(OrtTensorTypeAndShapeInfo* info, _In_ const int64_t* dim_values, size_t dim_count) {
  ThrowOnError(api_.SetDimensions(info, dim_values, dim_count));
}

template <typename T>
inline T* CustomOpApi::GetTensorMutableData(_Inout_ OrtValue* value) {
  T* data;
  ThrowOnError(api_.GetTensorMutableData(value, reinterpret_cast<void**>(&data)));
  return data;
}

template <typename T>
inline const T* CustomOpApi::GetTensorData(_Inout_ const OrtValue* value) {
  return GetTensorMutableData<T>(const_cast<OrtValue*>(value));
}

inline std::vector<int64_t> CustomOpApi::GetTensorShape(const OrtTensorTypeAndShapeInfo* info) {
  std::vector<int64_t> output(GetDimensionsCount(info));
  GetDimensions(info, output.data(), output.size());
  return output;
}

inline void CustomOpApi::ReleaseTensorTypeAndShapeInfo(OrtTensorTypeAndShapeInfo* input) {
  api_.ReleaseTensorTypeAndShapeInfo(input);
}

inline size_t CustomOpApi::KernelContext_GetInputCount(const OrtKernelContext* context) {
  size_t out;
  ThrowOnError(api_.KernelContext_GetInputCount(context, &out));
  return out;
}

inline const OrtValue* CustomOpApi::KernelContext_GetInput(const OrtKernelContext* context, _In_ size_t index) {
  const OrtValue* out;
  ThrowOnError(api_.KernelContext_GetInput(context, index, &out));
  return out;
}

inline size_t CustomOpApi::KernelContext_GetOutputCount(const OrtKernelContext* context) {
  size_t out;
  ThrowOnError(api_.KernelContext_GetOutputCount(context, &out));
  return out;
}

inline OrtValue* CustomOpApi::KernelContext_GetOutput(OrtKernelContext* context, _In_ size_t index, _In_ const int64_t* dim_values, size_t dim_count) {
  OrtValue* out;
  ThrowOnError(api_.KernelContext_GetOutput(context, index, dim_values, dim_count, &out));
  return out;
}

inline SessionOptions& SessionOptions::DisablePerSessionThreads() {
  ThrowOnError(Global<void>::api_.DisablePerSessionThreads(p_));
  return *this;
}

inline std::vector<std::string> GetAvailableProviders() {
  int len;
  char **providers;
  const OrtApi& api = GetApi();
  ThrowOnError(api.GetAvailableProviders(&providers, &len));
  std::vector<std::string> available_providers(providers, providers + len);
  ThrowOnError(api.ReleaseAvailableProviders(providers, len));
  return available_providers;
}
}  // namespace Ort
