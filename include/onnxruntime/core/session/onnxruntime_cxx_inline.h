// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Don't include this file directly. Please include "onnxruntime_cxx_api.h" instead.
// These are the inline implementations of the C++ header APIs. They're in this separate file as to not clutter
// the main C++ file with implementation details.

#define ORT_THROW_ON_ERROR(expr)                                              \
  if (OrtStatus* onnx_status = (expr)) {                                      \
    std::string ort_error_message = Ort::g_api->GetErrorMessage(onnx_status); \
    OrtErrorCode ort_error_code = Ort::g_api->GetErrorCode(onnx_status);      \
    Ort::g_api->ReleaseStatus(onnx_status);                                   \
    throw Ort::Exception(std::move(ort_error_message), ort_error_code);       \
  }

namespace Ort {

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

inline AllocatorWithDefaultOptions::AllocatorWithDefaultOptions() {
  ORT_THROW_ON_ERROR(g_api->GetAllocatorWithDefaultOptions(&p_));
}

inline void* AllocatorWithDefaultOptions::Alloc(size_t size) {
  void* out;
  ORT_THROW_ON_ERROR(g_api->AllocatorAlloc(p_, size, &out));
  return out;
}

inline void AllocatorWithDefaultOptions::Free(void* p) {
  ORT_THROW_ON_ERROR(g_api->AllocatorFree(p_, p));
}

inline const OrtAllocatorInfo* AllocatorWithDefaultOptions::GetInfo() const {
  const OrtAllocatorInfo* out;
  ORT_THROW_ON_ERROR(g_api->AllocatorGetInfo(p_, &out));
  return out;
}

inline AllocatorInfo AllocatorInfo::CreateCpu(OrtAllocatorType type, OrtMemType mem_type) {
  OrtAllocatorInfo* p;
  ORT_THROW_ON_ERROR(g_api->CreateCpuAllocatorInfo(type, mem_type, &p));
  return AllocatorInfo(p);
}

inline AllocatorInfo::AllocatorInfo(const char* name, OrtAllocatorType type, int id, OrtMemType mem_type) {
  ORT_THROW_ON_ERROR(g_api->CreateAllocatorInfo(name, type, id, mem_type, &p_));
}

inline Env::Env(OrtLoggingLevel default_warning_level, _In_ const char* logid) {
  ORT_THROW_ON_ERROR(g_api->CreateEnv(default_warning_level, logid, &p_));
}

inline Env::Env(OrtLoggingLevel default_warning_level, const char* logid, OrtLoggingFunction logging_function, void* logger_param) {
  ORT_THROW_ON_ERROR(g_api->CreateEnvWithCustomLogger(logging_function, logger_param, default_warning_level, logid, &p_));
}

inline CustomOpDomain::CustomOpDomain(const char* domain) {
  ORT_THROW_ON_ERROR(g_api->CreateCustomOpDomain(domain, &p_));
}

inline void CustomOpDomain::Add(OrtCustomOp* op) {
  ORT_THROW_ON_ERROR(g_api->CustomOpDomain_Add(p_, op));
}

inline RunOptions::RunOptions() {
  ORT_THROW_ON_ERROR(g_api->CreateRunOptions(&p_));
}

inline RunOptions& RunOptions::SetRunLogVerbosityLevel(int level) {
  ORT_THROW_ON_ERROR(g_api->RunOptionsSetRunLogVerbosityLevel(p_, level));
  return *this;
}

inline RunOptions& RunOptions::SetRunLogSeverityLevel(int level) {
  ORT_THROW_ON_ERROR(g_api->RunOptionsSetRunLogSeverityLevel(p_, level));
  return *this;
}

inline int RunOptions::GetRunLogVerbosityLevel() const {
  int out;
  ORT_THROW_ON_ERROR(g_api->RunOptionsGetRunLogVerbosityLevel(p_, &out));
  return out;
}

inline RunOptions& RunOptions::SetRunTag(const char* run_tag) {
  ORT_THROW_ON_ERROR(g_api->RunOptionsSetRunTag(p_, run_tag));
  return *this;
}

inline const char* RunOptions::GetRunTag() const {
  const char* out;
  ORT_THROW_ON_ERROR(g_api->RunOptionsGetRunTag(p_, &out));
  return out;
}

inline RunOptions& RunOptions::EnableTerminate() {
  ORT_THROW_ON_ERROR(g_api->RunOptionsEnableTerminate(p_));
  return *this;
}

inline RunOptions& RunOptions::DisableTerminate() {
  ORT_THROW_ON_ERROR(g_api->RunOptionsDisableTerminate(p_));
  return *this;
}

inline SessionOptions::SessionOptions() {
  ORT_THROW_ON_ERROR(g_api->CreateSessionOptions(&p_));
}

inline SessionOptions SessionOptions::Clone() const {
  OrtSessionOptions* out;
  ORT_THROW_ON_ERROR(g_api->CloneSessionOptions(p_, &out));
  return SessionOptions{out};
}

inline SessionOptions& SessionOptions::SetThreadPoolSize(int session_thread_pool_size) {
  ORT_THROW_ON_ERROR(g_api->SetSessionThreadPoolSize(p_, session_thread_pool_size));
  return *this;
}

inline SessionOptions& SessionOptions::SetGraphOptimizationLevel(GraphOptimizationLevel graph_optimization_level) {
  ORT_THROW_ON_ERROR(g_api->SetSessionGraphOptimizationLevel(p_, graph_optimization_level));
  return *this;
}

inline SessionOptions& SessionOptions::SetOptimizedModelFilePath(const ORTCHAR_T* optimized_model_filepath) {
  ORT_THROW_ON_ERROR(g_api->SetOptimizedModelFilePath(p_, optimized_model_filepath));
  return *this;
}

inline SessionOptions& SessionOptions::EnableProfiling(const ORTCHAR_T* profile_file_prefix) {
  ORT_THROW_ON_ERROR(g_api->EnableProfiling(p_, profile_file_prefix));
  return *this;
}

inline SessionOptions& SessionOptions::DisableProfiling() {
  ORT_THROW_ON_ERROR(g_api->DisableProfiling(p_));
  return *this;
}

inline SessionOptions& SessionOptions::EnableMemPattern() {
  ORT_THROW_ON_ERROR(g_api->EnableMemPattern(p_));
  return *this;
}

inline SessionOptions& SessionOptions::DisableMemPattern() {
  ORT_THROW_ON_ERROR(g_api->DisableMemPattern(p_));
  return *this;
}

inline SessionOptions& SessionOptions::EnableCpuMemArena() {
  ORT_THROW_ON_ERROR(g_api->EnableCpuMemArena(p_));
  return *this;
}

inline SessionOptions& SessionOptions::DisableCpuMemArena() {
  ORT_THROW_ON_ERROR(g_api->DisableCpuMemArena(p_));
  return *this;
}

inline SessionOptions& SessionOptions::EnableSequentialExecution() {
  ORT_THROW_ON_ERROR(g_api->EnableSequentialExecution(p_));
  return *this;
}

inline SessionOptions& SessionOptions::DisableSequentialExecution() {
  ORT_THROW_ON_ERROR(g_api->DisableSequentialExecution(p_));
  return *this;
}

inline SessionOptions& SessionOptions::SetLogId(const char* logid) {
  ORT_THROW_ON_ERROR(g_api->SetSessionLogId(p_, logid));
  return *this;
}
inline SessionOptions& SessionOptions::Add(OrtCustomOpDomain* custom_op_domain) {
  ORT_THROW_ON_ERROR(g_api->AddCustomOpDomain(p_, custom_op_domain));
  return *this;
}

inline Session::Session(Env& env, const ORTCHAR_T* model_path, const SessionOptions& options) {
  ORT_THROW_ON_ERROR(g_api->CreateSession(env, model_path, options, &p_));
}

inline Session::Session(Env& env, const void* model_data, size_t model_data_length, const SessionOptions& options) {
  ORT_THROW_ON_ERROR(g_api->CreateSessionFromArray(env, model_data, model_data_length, options, &p_));
}

inline std::vector<Value> Session::Run(const RunOptions& run_options, const char* const* input_names, Value* input_values, size_t input_count,
                                       const char* const* output_names, size_t output_names_count) {
  std::vector<Ort::Value> output_values;
  for (size_t i = 0; i < output_names_count; i++)
    output_values.emplace_back(nullptr);
  Run(run_options, input_names, input_values, input_count, output_names, output_values.data(), output_names_count);
  return output_values;
}

inline void Session::Run(const RunOptions& run_options, const char* const* input_names, Value* input_values, size_t input_count,
                         const char* const* output_names, Value* output_values, size_t output_count) {
  static_assert(sizeof(Value) == sizeof(OrtValue*), "Value is really just an array of OrtValue* in memory, so we can reinterpret_cast safely");
  auto ort_input_values = reinterpret_cast<OrtValue**>(input_values);
  auto ort_output_values = reinterpret_cast<OrtValue**>(output_values);
  ORT_THROW_ON_ERROR(g_api->Run(p_, run_options, input_names, ort_input_values, input_count, output_names, output_count, ort_output_values));
}

inline size_t Session::GetInputCount() const {
  size_t out;
  ORT_THROW_ON_ERROR(g_api->SessionGetInputCount(p_, &out));
  return out;
}

inline size_t Session::GetOutputCount() const {
  size_t out;
  ORT_THROW_ON_ERROR(g_api->SessionGetOutputCount(p_, &out));
  return out;
}

inline char* Session::GetInputName(size_t index, OrtAllocator* allocator) const {
  char* out;
  ORT_THROW_ON_ERROR(g_api->SessionGetInputName(p_, index, allocator, &out));
  return out;
}

inline char* Session::GetOutputName(size_t index, OrtAllocator* allocator) const {
  char* out;
  ORT_THROW_ON_ERROR(g_api->SessionGetOutputName(p_, index, allocator, &out));
  return out;
}

inline TypeInfo Session::GetInputTypeInfo(size_t index) const {
  OrtTypeInfo* out;
  ORT_THROW_ON_ERROR(g_api->SessionGetInputTypeInfo(p_, index, &out));
  return TypeInfo{out};
}

inline TypeInfo Session::GetOutputTypeInfo(size_t index) const {
  OrtTypeInfo* out;
  ORT_THROW_ON_ERROR(g_api->SessionGetOutputTypeInfo(p_, index, &out));
  return TypeInfo{out};
}

inline ONNXTensorElementDataType TensorTypeAndShapeInfo::GetElementType() const {
  ONNXTensorElementDataType out;
  ORT_THROW_ON_ERROR(g_api->GetTensorElementType(p_, &out));
  return out;
}

inline size_t TensorTypeAndShapeInfo::GetElementCount() const {
  size_t out;
  ORT_THROW_ON_ERROR(g_api->GetTensorShapeElementCount(p_, &out));
  return static_cast<size_t>(out);
}

inline size_t TensorTypeAndShapeInfo::GetDimensionsCount() const {
  size_t out;
  ORT_THROW_ON_ERROR(g_api->GetDimensionsCount(p_, &out));
  return out;
}

inline void TensorTypeAndShapeInfo::GetDimensions(int64_t* values, size_t values_count) const {
  ORT_THROW_ON_ERROR(g_api->GetDimensions(p_, values, values_count));
}

inline std::vector<int64_t> TensorTypeAndShapeInfo::GetShape() const {
  std::vector<int64_t> out(GetDimensionsCount(), 0);
  GetDimensions(out.data(), out.size());
  return out;
}

inline Unowned<TensorTypeAndShapeInfo> TypeInfo::GetTensorTypeAndShapeInfo() const {
  const OrtTensorTypeAndShapeInfo* out;
  ORT_THROW_ON_ERROR(g_api->CastTypeInfoToTensorInfo(p_, &out));
  return Unowned<TensorTypeAndShapeInfo>{const_cast<OrtTensorTypeAndShapeInfo*>(out)};
}

inline ONNXType TypeInfo::GetONNXType() const {
  ONNXType out;
  ORT_THROW_ON_ERROR(g_api->GetOnnxTypeFromTypeInfo(p_, &out));
  return out;
}

template <typename T>
inline Value Value::CreateTensor(const OrtAllocatorInfo* info, T* p_data, size_t p_data_element_count, const int64_t* shape, size_t shape_len) {
  return CreateTensor(info, p_data, p_data_element_count * sizeof(T), shape, shape_len, TypeToTensorType<T>::type);
}

inline Value Value::CreateTensor(const OrtAllocatorInfo* info, void* p_data, size_t p_data_byte_count, const int64_t* shape, size_t shape_len,
                                 ONNXTensorElementDataType type) {
  OrtValue* out;
  ORT_THROW_ON_ERROR(g_api->CreateTensorWithDataAsOrtValue(info, p_data, p_data_byte_count, shape, shape_len, type, &out));
  return Value{out};
}

template <typename T>
inline Value Value::CreateTensor(OrtAllocator* allocator, const int64_t* shape, size_t shape_len) {
  return CreateTensor(allocator, shape, shape_len, TypeToTensorType<T>::type);
}

inline Value Value::CreateTensor(OrtAllocator* allocator, const int64_t* shape, size_t shape_len, ONNXTensorElementDataType type) {
  OrtValue* out;
  ORT_THROW_ON_ERROR(g_api->CreateTensorAsOrtValue(allocator, shape, shape_len, type, &out));
  return Value{out};
}

inline Value Value::CreateMap(Value& keys, Value& values) {
  OrtValue* out;
  OrtValue* inputs[2] = {keys, values};
  ORT_THROW_ON_ERROR(g_api->CreateValue(inputs, 2, ONNX_TYPE_MAP, &out));
  return Value{out};
}

inline Value Value::CreateSequence(std::vector<Value>& values) {
  OrtValue* out;
  std::vector<OrtValue*> values_ort{values.data(), values.data() + values.size()};
  ORT_THROW_ON_ERROR(g_api->CreateValue(values_ort.data(), values_ort.size(), ONNX_TYPE_SEQUENCE, &out));
  return Value{out};
}

inline bool Value::IsTensor() const {
  int out;
  ORT_THROW_ON_ERROR(g_api->IsTensor(p_, &out));
  return out != 0;
}

inline size_t Value::GetCount() const {
  size_t out;
  ORT_THROW_ON_ERROR(g_api->GetValueCount(p_, &out));
  return out;
}

inline Value Value::GetValue(int index, OrtAllocator* allocator) const {
  OrtValue* out;
  ORT_THROW_ON_ERROR(g_api->GetValue(p_, index, allocator, &out));
  return Value{out};
}

inline size_t Value::GetStringTensorDataLength() const {
  size_t out;
  ORT_THROW_ON_ERROR(g_api->GetStringTensorDataLength(p_, &out));
  return out;
}

inline void Value::GetStringTensorContent(void* buffer, size_t buffer_length, size_t* offsets, size_t offsets_count) const {
  ORT_THROW_ON_ERROR(g_api->GetStringTensorContent(p_, buffer, buffer_length, offsets, offsets_count));
}

template <typename T>
T* Value::GetTensorMutableData() {
  T* out;
  ORT_THROW_ON_ERROR(g_api->GetTensorMutableData(p_, (void**)&out));
  return out;
}

inline TypeInfo Value::GetTypeInfo() const {
  OrtTypeInfo* output;
  ORT_THROW_ON_ERROR(g_api->GetTypeInfo(p_, &output));
  return TypeInfo{output};
}

inline TensorTypeAndShapeInfo Value::GetTensorTypeAndShapeInfo() const {
  OrtTensorTypeAndShapeInfo* output;
  ORT_THROW_ON_ERROR(g_api->GetTensorTypeAndShape(p_, &output));
  return TensorTypeAndShapeInfo{output};
}

//
// Custom OP API Inlines
//
template <>
inline float CustomOpApi::KernelInfoGetAttribute<float>(_In_ const OrtKernelInfo* info, _In_ const char* name) {
  float out;
  ORT_THROW_ON_ERROR(api_.KernelInfoGetAttribute_float(info, name, &out));
  return out;
}

template <>
inline int64_t CustomOpApi::KernelInfoGetAttribute<int64_t>(_In_ const OrtKernelInfo* info, _In_ const char* name) {
  int64_t out;
  ORT_THROW_ON_ERROR(api_.KernelInfoGetAttribute_int64(info, name, &out));
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
    ORT_THROW_ON_ERROR(api_.KernelInfoGetAttribute_string(info, name, &out[0], &size));
    out.resize(size - 1);  // remove the terminating character '\0'
  } else {
    ORT_THROW_ON_ERROR(status);
  }
  return out;
}

inline OrtTensorTypeAndShapeInfo* CustomOpApi::GetTensorTypeAndShape(_In_ const OrtValue* value) {
  OrtTensorTypeAndShapeInfo* out;
  ORT_THROW_ON_ERROR(api_.GetTensorTypeAndShape(value, &out));
  return out;
}

inline size_t CustomOpApi::GetTensorShapeElementCount(_In_ const OrtTensorTypeAndShapeInfo* info) {
  size_t out;
  ORT_THROW_ON_ERROR(api_.GetTensorShapeElementCount(info, &out));
  return out;
}

inline ONNXTensorElementDataType CustomOpApi::GetTensorElementType(const OrtTensorTypeAndShapeInfo* info) {
  ONNXTensorElementDataType out;
  ORT_THROW_ON_ERROR(api_.GetTensorElementType(info, &out));
  return out;
}

inline size_t CustomOpApi::GetDimensionsCount(_In_ const OrtTensorTypeAndShapeInfo* info) {
  size_t out;
  ORT_THROW_ON_ERROR(api_.GetDimensionsCount(info, &out));
  return out;
}

inline void CustomOpApi::GetDimensions(_In_ const OrtTensorTypeAndShapeInfo* info, _Out_ int64_t* dim_values, size_t dim_values_length) {
  ORT_THROW_ON_ERROR(api_.GetDimensions(info, dim_values, dim_values_length));
}

inline void CustomOpApi::SetDimensions(OrtTensorTypeAndShapeInfo* info, _In_ const int64_t* dim_values, size_t dim_count) {
  ORT_THROW_ON_ERROR(api_.SetDimensions(info, dim_values, dim_count));
}

template <typename T>
inline T* CustomOpApi::GetTensorMutableData(_Inout_ OrtValue* value) {
  T* data;
  ORT_THROW_ON_ERROR(api_.GetTensorMutableData(value, reinterpret_cast<void**>(&data)));
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
  ORT_THROW_ON_ERROR(api_.KernelContext_GetInputCount(context, &out));
  return out;
}

inline const OrtValue* CustomOpApi::KernelContext_GetInput(const OrtKernelContext* context, _In_ size_t index) {
  const OrtValue* out;
  ORT_THROW_ON_ERROR(api_.KernelContext_GetInput(context, index, &out));
  return out;
}

inline size_t CustomOpApi::KernelContext_GetOutputCount(const OrtKernelContext* context) {
  size_t out;
  ORT_THROW_ON_ERROR(api_.KernelContext_GetOutputCount(context, &out));
  return out;
}

inline OrtValue* CustomOpApi::KernelContext_GetOutput(OrtKernelContext* context, _In_ size_t index, _In_ const int64_t* dim_values, size_t dim_count) {
  OrtValue* out;
  ORT_THROW_ON_ERROR(api_.KernelContext_GetOutput(context, index, dim_values, dim_count, &out));
  return out;
}

}  // namespace Ort
