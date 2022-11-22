// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Do not include this file directly. Please include "onnxruntime_cxx_api.h" instead.
// If interested in trying out features of the new experimental C++ API, include "experimental_onnxruntime_cxx_api.h" instead.
//
// These are the inline implementations of the C++ header APIs. They're in this separate file as to not clutter
// the main C++ file with implementation details.

namespace Ort {

#ifndef ORT_CXX_API_THROW
inline void ThrowOnError(OrtStatus* ort_status) {
  if (ort_status) {
    throw Exception(std::unique_ptr<OrtStatus>{ort_status});
  }
}
#else
ORT_CXX_API_THROW
#endif

inline OrtErrorCode Exception::GetOrtErrorCode() const { return ort_status_->GetErrorCode(); }
inline const char* Exception::what() const noexcept { return api->GetErrorMessage(ort_status_.get()); }

struct StandardAllocator : OrtAllocator
{
  StandardAllocator() : OrtAllocator{}
  {
    version = ORT_API_VERSION;
    OrtAllocator::Alloc = [](OrtAllocator* this_, size_t size) -> void* { return new std::byte[size]; };
    OrtAllocator::Free = [](OrtAllocator* this_, void *p) { delete p; };
  }
};

inline StandardAllocator standard_allocator;

// Used on methods that return a std::string. This implements an OrtAllocator that Allocates memory directly
// from a std::string.
struct StringAllocator : OrtAllocator
{
  StringAllocator() : OrtAllocator{}
  {
    version = ORT_API_VERSION;
    OrtAllocator::Alloc = [](OrtAllocator* this_, size_t size) { return static_cast<StringAllocator*>(this_)->Alloc(size); };
  }

  void* Alloc(size_t size)
  {
    string_.resize(size);
    return string_.data();
  }

  operator std::string && ()
  {
    string_.resize(string_.size() - 1); // Remove the trailing null
    return std::move(string_);
  }

  char* out;

private:
  std::string string_;
};

// This template converts a C++ type into it's ONNXTensorElementDataType
template <typename T>
struct TypeToTensorType;
template <>
struct TypeToTensorType<float> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };
template <>
struct TypeToTensorType<Float16_t> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16; };
template <>
struct TypeToTensorType<BFloat16_t> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16; };
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

inline std::vector<std::string> GetAvailableProviders() {
  int len;
  char** providers;
  ThrowOnError(api->GetAvailableProviders(&providers, &len));
  std::vector<std::string> available_providers(providers, providers + len);
  ThrowOnError(api->ReleaseAvailableProviders(providers, len));
  return available_providers;
}

inline void* Allocator::Alloc(size_t size) {
  void* out;
  ThrowOnError(api->AllocatorAlloc(this, size, &out));
  return out;
}

inline void Allocator::Free(void* p) {
  ThrowOnError(api->AllocatorFree(this, p));
}

inline const OrtMemoryInfo& Allocator::GetInfo() const {
  const OrtMemoryInfo* out;
  ThrowOnError(api->AllocatorGetInfo(this, &out));
  return *out;
}

inline Allocator& Allocator::GetWithDefaultOptions() {
  OrtAllocator* p;
  ThrowOnError(api->GetAllocatorWithDefaultOptions(&p));
  return *static_cast<Allocator*>(p);
}

inline std::unique_ptr<Allocator> Allocator::Create(const OrtSession& sess, const OrtMemoryInfo& mem_info) {
  OrtAllocator* p;
  ThrowOnError(api->CreateAllocator(&sess, &mem_info, &p));
  return std::unique_ptr<Allocator>{static_cast<Ort::Allocator*>(p)};
}

} // namespace Ort

inline std::unique_ptr<OrtStatus> OrtStatus::Create(OrtErrorCode code, const std::string& what) {
  return std::unique_ptr<OrtStatus>{Ort::api->CreateStatus(code, what.c_str())};
}

inline std::string OrtStatus::GetErrorMessage() const {
  std::string message(Ort::api->GetErrorMessage(this));
  return message;
}

inline OrtErrorCode OrtStatus::GetErrorCode() const {
  return Ort::api->GetErrorCode(this);
}

inline std::string OrtMemoryInfo::GetAllocatorName() const {
  const char* name = nullptr;
  Ort::ThrowOnError(Ort::api->MemoryInfoGetName(this, &name));
  return std::string(name);
}

inline OrtAllocatorType OrtMemoryInfo::GetAllocatorType() const {
  OrtAllocatorType type;
  Ort::ThrowOnError(Ort::api->MemoryInfoGetType(this, &type));
  return type;
}

inline int OrtMemoryInfo::GetDeviceId() const {
  int id = 0;
  Ort::ThrowOnError(Ort::api->MemoryInfoGetId(this, &id));
  return id;
}

inline OrtMemoryInfoDeviceType OrtMemoryInfo::GetDeviceType() const {
  OrtMemoryInfoDeviceType type;
  Ort::api->MemoryInfoGetDeviceType(this, &type);
  return type;
}

inline OrtMemType OrtMemoryInfo::GetMemoryType() const {
  OrtMemType type;
  Ort::ThrowOnError(Ort::api->MemoryInfoGetMemType(this, &type));
  return type;
}

inline bool OrtMemoryInfo::operator==(const OrtMemoryInfo& o) const {
  int comp_result = 0;
  Ort::ThrowOnError(Ort::api->CompareMemoryInfo(this, &o, &comp_result));
  return comp_result == 0;
}

inline std::unique_ptr<OrtMemoryInfo> OrtMemoryInfo::CreateCpu(OrtAllocatorType type, OrtMemType mem_type) {
  OrtMemoryInfo* p;
  Ort::ThrowOnError(Ort::api->CreateCpuMemoryInfo(type, mem_type, &p));
  return std::unique_ptr<OrtMemoryInfo>{p};
}

inline std::unique_ptr<OrtMemoryInfo> OrtMemoryInfo::Create(const char* name, OrtAllocatorType type, int id, OrtMemType mem_type) {
  OrtMemoryInfo *p;
  Ort::ThrowOnError(Ort::api->CreateMemoryInfo(name, type, id, mem_type, &p));
  return std::unique_ptr<OrtMemoryInfo>{p};
}

inline std::unique_ptr<OrtIoBinding> OrtIoBinding::Create(OrtSession& session) {
  OrtIoBinding *p;
  Ort::ThrowOnError(Ort::api->CreateIoBinding(&session, &p));
  return std::unique_ptr<OrtIoBinding>{p};
}

inline std::vector<std::string> OrtIoBinding::GetOutputNames() const {
  char* buffer{};
  size_t* lengths{};
  size_t count{};
  Ort::ThrowOnError(Ort::api->GetBoundOutputNames(this, &Ort::standard_allocator, &buffer, &lengths, &count));

  std::unique_ptr<size_t> lengths_owned{ lengths };
  std::unique_ptr<char> buffer_owned{ buffer };

  std::vector<std::string> result;
  for (size_t i = 0; i < count; ++i) {
    auto sz = *lengths;
    result.emplace_back(buffer, sz);
    buffer += sz;
    ++lengths;
  }
  return result;
}

inline std::vector<std::unique_ptr<OrtValue>> OrtIoBinding::GetOutputValues() const {
  size_t owned = 0;
  size_t output_count = 0;
  OrtValue** output_buffer{};
  Ort::ThrowOnError(Ort::api->GetBoundOutputValues(this, &Ort::standard_allocator, &output_buffer, &output_count));
  std::unique_ptr<OrtValue*> owned_output_buffer{ output_buffer };

  try {
    std::vector<std::unique_ptr<OrtValue>> result;
    for (size_t i = 0; i < output_count; ++i) {
      result.emplace_back(output_buffer[i]);
      ++owned;
    }
    return result;
  }
  catch (...) { // delete any untransferred OrtValues
    while (owned < output_count)
      delete output_buffer[owned++];
    throw;
  }
}
  
inline void OrtIoBinding::BindInput(const char* name, const OrtValue& value) {
  Ort::ThrowOnError(Ort::api->BindInput(this, name, &value));
}

inline void OrtIoBinding::BindOutput(const char* name, const OrtValue& value) {
  Ort::ThrowOnError(Ort::api->BindOutput(this, name, &value));
}

inline void OrtIoBinding::BindOutput(const char* name, const OrtMemoryInfo& mem_info) {
  Ort::ThrowOnError(Ort::api->BindOutputToDevice(this, name, &mem_info));
}

inline void OrtIoBinding::ClearBoundInputs() {
  Ort::api->ClearBoundInputs(this);
}

inline void OrtIoBinding::ClearBoundOutputs() {
  Ort::api->ClearBoundOutputs(this);
}

inline void OrtIoBinding::SynchronizeInputs() {
  Ort::ThrowOnError(Ort::api->SynchronizeBoundInputs(this));
}

inline void OrtIoBinding::SynchronizeOutputs() {
  Ort::ThrowOnError(Ort::api->SynchronizeBoundOutputs(this));
}

inline std::unique_ptr<OrtArenaCfg> OrtArenaCfg::Create(size_t max_mem, int arena_extend_strategy, int initial_chunk_size_bytes, int max_dead_bytes_per_chunk) {
  OrtArenaCfg *p;
  Ort::ThrowOnError(Ort::api->CreateArenaCfg(max_mem, arena_extend_strategy, initial_chunk_size_bytes, max_dead_bytes_per_chunk, &p));
  return std::unique_ptr<OrtArenaCfg>{p};
}

inline void OrtCommonEnvInit(OrtEnv& v, _In_ const char* logid)
{
  if (strcmp(logid, "onnxruntime-node") == 0) {
    Ort::ThrowOnError(Ort::api->SetLanguageProjection(&v, OrtLanguageProjection::ORT_PROJECTION_NODEJS));
  }
  else {
    Ort::ThrowOnError(Ort::api->SetLanguageProjection(&v, OrtLanguageProjection::ORT_PROJECTION_CPLUSPLUS));
  }
}

inline std::unique_ptr<OrtEnv> OrtEnv::Create(OrtLoggingLevel logging_level, _In_ const char* logid) {
  OrtEnv *p;
  Ort::ThrowOnError(Ort::api->CreateEnv(logging_level, logid, &p));
  OrtCommonEnvInit(*p, logid);
  return std::unique_ptr<OrtEnv>(p);
}

inline std::unique_ptr<OrtEnv> OrtEnv::Create(OrtLoggingLevel logging_level, const char* logid, OrtLoggingFunction logging_function, void* logger_param) {
  OrtEnv *p;
  Ort::ThrowOnError(Ort::api->CreateEnvWithCustomLogger(logging_function, logger_param, logging_level, logid, &p));
  OrtCommonEnvInit(*p, logid);
  return std::unique_ptr<OrtEnv>(p);
}

inline std::unique_ptr<OrtEnv> OrtEnv::Create(const OrtThreadingOptions* tp_options, OrtLoggingLevel logging_level, _In_ const char* logid) {
  OrtEnv *p;
  Ort::ThrowOnError(Ort::api->CreateEnvWithGlobalThreadPools(logging_level, logid, tp_options, &p));
  OrtCommonEnvInit(*p, logid);
  return std::unique_ptr<OrtEnv>(p);
}

inline std::unique_ptr<OrtEnv> OrtEnv::Create(const OrtThreadingOptions* tp_options, OrtLoggingFunction logging_function, void* logger_param,
                OrtLoggingLevel logging_level, _In_ const char* logid) {
  OrtEnv *p;
  Ort::ThrowOnError(Ort::api->CreateEnvWithCustomLoggerAndGlobalThreadPools(logging_function, logger_param, logging_level, logid, tp_options, &p));
  OrtCommonEnvInit(*p, logid);
  return std::unique_ptr<OrtEnv>(p);
}

inline OrtEnv& OrtEnv::EnableTelemetryEvents() {
  Ort::ThrowOnError(Ort::api->EnableTelemetryEvents(this));
  return *this;
}

inline OrtEnv& OrtEnv::DisableTelemetryEvents() {
  Ort::ThrowOnError(Ort::api->DisableTelemetryEvents(this));
  return *this;
}

inline OrtEnv& OrtEnv::CreateAndRegisterAllocator(const OrtMemoryInfo& mem_info, const OrtArenaCfg& arena_cfg) {
  Ort::ThrowOnError(Ort::api->CreateAndRegisterAllocator(this, &mem_info, &arena_cfg));
  return *this;
}

inline std::unique_ptr<OrtThreadingOptions> OrtThreadingOptions::Create() {
  OrtThreadingOptions* p;
  Ort::ThrowOnError(Ort::api->CreateThreadingOptions(&p));
  return std::unique_ptr<OrtThreadingOptions>{p};
}

inline void OrtThreadingOptions::SetGlobalIntraOpNumThreads(int intra_op_num_threads) {
  Ort::ThrowOnError(Ort::api->SetGlobalIntraOpNumThreads(this, intra_op_num_threads));
}

inline void OrtThreadingOptions::SetGlobalInterOpNumThreads(int inter_op_num_threads) {
  Ort::ThrowOnError(Ort::api->SetGlobalInterOpNumThreads(this, inter_op_num_threads));
}

inline void OrtThreadingOptions::SetGlobalSpinControl(int allow_spinning) {
  Ort::ThrowOnError(Ort::api->SetGlobalSpinControl(this, allow_spinning));
}

inline void OrtThreadingOptions::SetGlobalDenormalAsZero() {
  Ort::ThrowOnError(Ort::api->SetGlobalDenormalAsZero(this));
}

inline void OrtThreadingOptions::SetGlobalCustomCreateThreadFn(OrtCustomCreateThreadFn ort_custom_create_thread_fn) {
  Ort::ThrowOnError(Ort::api->SetGlobalCustomCreateThreadFn(this, ort_custom_create_thread_fn));
}

inline void OrtThreadingOptions::SetGlobalCustomThreadCreationOptions(void* ort_custom_thread_creation_options) {
  Ort::ThrowOnError(Ort::api->SetGlobalCustomThreadCreationOptions(this, ort_custom_thread_creation_options));
}

inline void OrtThreadingOptions::SetGlobalCustomJoinThreadFn(OrtCustomJoinThreadFn ort_custom_join_thread_fn) {
  Ort::ThrowOnError(Ort::api->SetGlobalCustomJoinThreadFn(this, ort_custom_join_thread_fn));
}

inline std::unique_ptr<OrtCustomOpDomain> OrtCustomOpDomain::Create(const char* domain) {
  OrtCustomOpDomain *p;
  Ort::ThrowOnError(Ort::api->CreateCustomOpDomain(domain, &p));
  return std::unique_ptr<OrtCustomOpDomain>{p};
}

inline void OrtCustomOpDomain::Add(const OrtCustomOp& op) {
  Ort::ThrowOnError(Ort::api->CustomOpDomain_Add(this, &op));
}

inline std::unique_ptr<OrtRunOptions> OrtRunOptions::Create() {
  OrtRunOptions *p;
  Ort::ThrowOnError(Ort::api->CreateRunOptions(&p));
  return std::unique_ptr<OrtRunOptions>{p};
}

inline OrtRunOptions& OrtRunOptions::SetRunLogVerbosityLevel(int level) {
  Ort::ThrowOnError(Ort::api->RunOptionsSetRunLogVerbosityLevel(this, level));
  return *this;
}

inline OrtRunOptions& OrtRunOptions::SetRunLogSeverityLevel(int level) {
  Ort::ThrowOnError(Ort::api->RunOptionsSetRunLogSeverityLevel(this, level));
  return *this;
}

inline int OrtRunOptions::GetRunLogVerbosityLevel() const {
  int out;
  Ort::ThrowOnError(Ort::api->RunOptionsGetRunLogVerbosityLevel(this, &out));
  return out;
}

inline int OrtRunOptions::GetRunLogSeverityLevel() const {
  int out;
  Ort::ThrowOnError(Ort::api->RunOptionsGetRunLogSeverityLevel(this, &out));
  return out;
}

inline OrtRunOptions& OrtRunOptions::SetRunTag(const char* run_tag) {
  Ort::ThrowOnError(Ort::api->RunOptionsSetRunTag(this, run_tag));
  return *this;
}

inline const char* OrtRunOptions::GetRunTag() const {
  const char* out;
  Ort::ThrowOnError(Ort::api->RunOptionsGetRunTag(this, &out));
  return out;
}

inline OrtRunOptions& OrtRunOptions::AddConfigEntry(const char* config_key, const char* config_value) {
  Ort::ThrowOnError(Ort::api->AddRunConfigEntry(this, config_key, config_value));
  return *this;
}

inline OrtRunOptions& OrtRunOptions::SetTerminate() {
  Ort::ThrowOnError(Ort::api->RunOptionsSetTerminate(this));
  return *this;
}

inline OrtRunOptions& OrtRunOptions::UnsetTerminate() {
  Ort::ThrowOnError(Ort::api->RunOptionsUnsetTerminate(this));
  return *this;
}

inline std::unique_ptr<OrtSessionOptions> OrtSessionOptions::Create() {
  OrtSessionOptions* p;
  Ort::ThrowOnError(Ort::api->CreateSessionOptions(&p));
  return std::unique_ptr<OrtSessionOptions>{p};
}

inline std::unique_ptr<OrtSessionOptions> OrtSessionOptions::Clone() const {
  OrtSessionOptions* out;
  Ort::ThrowOnError(Ort::api->CloneSessionOptions(this, &out));
  return std::unique_ptr<OrtSessionOptions>{out};
}

inline OrtSessionOptions& OrtSessionOptions::SetIntraOpNumThreads(int intra_op_num_threads) {
  Ort::ThrowOnError(Ort::api->SetIntraOpNumThreads(this, intra_op_num_threads));
  return *this;
}

inline OrtSessionOptions& OrtSessionOptions::SetInterOpNumThreads(int inter_op_num_threads) {
  Ort::ThrowOnError(Ort::api->SetInterOpNumThreads(this, inter_op_num_threads));
  return *this;
}

inline OrtSessionOptions& OrtSessionOptions::SetGraphOptimizationLevel(GraphOptimizationLevel graph_optimization_level) {
  Ort::ThrowOnError(Ort::api->SetSessionGraphOptimizationLevel(this, graph_optimization_level));
  return *this;
}

inline OrtSessionOptions& OrtSessionOptions::SetOptimizedModelFilePath(const ORTCHAR_T* optimized_model_filepath) {
  Ort::ThrowOnError(Ort::api->SetOptimizedModelFilePath(this, optimized_model_filepath));
  return *this;
}

inline OrtSessionOptions& OrtSessionOptions::EnableProfiling(const ORTCHAR_T* profile_file_prefix) {
  Ort::ThrowOnError(Ort::api->EnableProfiling(this, profile_file_prefix));
  return *this;
}

inline OrtSessionOptions& OrtSessionOptions::DisableProfiling() {
  Ort::ThrowOnError(Ort::api->DisableProfiling(this));
  return *this;
}

inline OrtSessionOptions& OrtSessionOptions::EnableOrtCustomOps() {
  Ort::ThrowOnError(Ort::api->EnableOrtCustomOps(this));
  return *this;
}

inline OrtSessionOptions& OrtSessionOptions::EnableMemPattern() {
  Ort::ThrowOnError(Ort::api->EnableMemPattern(this));
  return *this;
}

inline OrtSessionOptions& OrtSessionOptions::DisableMemPattern() {
  Ort::ThrowOnError(Ort::api->DisableMemPattern(this));
  return *this;
}

inline OrtSessionOptions& OrtSessionOptions::EnableCpuMemArena() {
  Ort::ThrowOnError(Ort::api->EnableCpuMemArena(this));
  return *this;
}

inline OrtSessionOptions& OrtSessionOptions::DisableCpuMemArena() {
  Ort::ThrowOnError(Ort::api->DisableCpuMemArena(this));
  return *this;
}

inline OrtSessionOptions& OrtSessionOptions::SetExecutionMode(ExecutionMode execution_mode) {
  Ort::ThrowOnError(Ort::api->SetSessionExecutionMode(this, execution_mode));
  return *this;
}

inline OrtSessionOptions& OrtSessionOptions::SetLogId(const char* logid) {
  Ort::ThrowOnError(Ort::api->SetSessionLogId(this, logid));
  return *this;
}

inline OrtSessionOptions& OrtSessionOptions::SetLogSeverityLevel(int level) {
  Ort::ThrowOnError(Ort::api->SetSessionLogSeverityLevel(this, level));
  return *this;
}

inline OrtSessionOptions& OrtSessionOptions::Add(OrtCustomOpDomain& custom_op_domain) {
  Ort::ThrowOnError(Ort::api->AddCustomOpDomain(this, &custom_op_domain));
  return *this;
}

inline OrtSessionOptions& OrtSessionOptions::AddConfigEntry(const char* config_key, const char* config_value) {
  Ort::ThrowOnError(Ort::api->AddSessionConfigEntry(this, config_key, config_value));
  return *this;
}

inline OrtSessionOptions& OrtSessionOptions::AddInitializer(const char* name, const OrtValue& ort_val) {
  Ort::ThrowOnError(Ort::api->AddInitializer(this, name, &ort_val));
  return *this;
}

inline OrtSessionOptions& OrtSessionOptions::DisablePerSessionThreads() {
  Ort::ThrowOnError(Ort::api->DisablePerSessionThreads(this));
  return *this;
}

inline OrtSessionOptions& OrtSessionOptions::AddExternalInitializers(const std::vector<std::string>& names,
                                                                     const std::vector<std::unique_ptr<OrtValue>>& ort_values) {
  const size_t inputs_num = names.size();
  if (inputs_num != ort_values.size()) {
    Ort::ThrowOnError(OrtStatus::Create(ORT_INVALID_ARGUMENT, "Expecting names and ort_values to have the same length").get());
  }
  std::vector<const char*> names_ptr;
  std::vector<const OrtValue*> ort_values_ptrs;
  names_ptr.reserve(inputs_num);
  ort_values_ptrs.reserve(inputs_num);
  for (size_t i = 0; i < inputs_num; ++i) {
    names_ptr.push_back(names[i].c_str());
    ort_values_ptrs.push_back(ort_values[i].get());
  }
  Ort::ThrowOnError(Ort::api->AddExternalInitializers(this, names_ptr.data(), ort_values_ptrs.data(), inputs_num));
  return *this;
}

inline OrtSessionOptions& OrtSessionOptions::AppendExecutionProvider_CUDA(const OrtCUDAProviderOptions& provider_options) {
  Ort::ThrowOnError(Ort::api->SessionOptionsAppendExecutionProvider_CUDA(this, &provider_options));
  return *this;
}

inline OrtSessionOptions& OrtSessionOptions::AppendExecutionProvider_CUDA_V2(const OrtCUDAProviderOptionsV2& provider_options) {
  Ort::ThrowOnError(Ort::api->SessionOptionsAppendExecutionProvider_CUDA_V2(this, &provider_options));
  return *this;
}

inline OrtSessionOptions& OrtSessionOptions::AppendExecutionProvider_ROCM(const OrtROCMProviderOptions& provider_options) {
  Ort::ThrowOnError(Ort::api->SessionOptionsAppendExecutionProvider_ROCM(this, &provider_options));
  return *this;
}

inline OrtSessionOptions& OrtSessionOptions::AppendExecutionProvider_TensorRT(const OrtTensorRTProviderOptions& provider_options) {
  Ort::ThrowOnError(Ort::api->SessionOptionsAppendExecutionProvider_TensorRT(this, &provider_options));
  return *this;
}

inline OrtSessionOptions& OrtSessionOptions::AppendExecutionProvider_TensorRT_V2(const OrtTensorRTProviderOptionsV2& provider_options) {
  Ort::ThrowOnError(Ort::api->SessionOptionsAppendExecutionProvider_TensorRT_V2(this, &provider_options));
  return *this;
}

inline OrtSessionOptions& OrtSessionOptions::AppendExecutionProvider_MIGraphX(const OrtMIGraphXProviderOptions& provider_options) {
  Ort::ThrowOnError(Ort::api->SessionOptionsAppendExecutionProvider_MIGraphX(this, &provider_options));
  return *this;
}

inline OrtSessionOptions& OrtSessionOptions::AppendExecutionProvider_CANN(const OrtCANNProviderOptions& provider_options) {
  Ort::ThrowOnError(Ort::api->SessionOptionsAppendExecutionProvider_CANN(this, &provider_options));
  return *this;
}

inline OrtSessionOptions& OrtSessionOptions::AppendExecutionProvider(
    const std::string& provider_name,
    const std::unordered_map<std::string, std::string>& provider_options) {
  auto num_entries = provider_options.size();
  std::vector<const char*> keys, values;
  if (num_entries > 0) {
    keys.reserve(num_entries);
    values.reserve(num_entries);

    for (const auto& entry : provider_options) {
      keys.push_back(entry.first.c_str());
      values.push_back(entry.second.c_str());
    }
  }

  Ort::ThrowOnError(Ort::api->SessionOptionsAppendExecutionProvider(this, provider_name.c_str(),
                                                              keys.data(), values.data(), num_entries));

  return *this;
}

inline OrtSessionOptions& OrtSessionOptions::SetCustomCreateThreadFn(OrtCustomCreateThreadFn ort_custom_create_thread_fn) {
  Ort::ThrowOnError(Ort::api->SessionOptionsSetCustomCreateThreadFn(this, ort_custom_create_thread_fn));
  return *this;
}

inline OrtSessionOptions& OrtSessionOptions::SetCustomThreadCreationOptions(void* ort_custom_thread_creation_options) {
  Ort::ThrowOnError(Ort::api->SessionOptionsSetCustomThreadCreationOptions(this, ort_custom_thread_creation_options));
  return *this;
}

inline OrtSessionOptions& OrtSessionOptions::SetCustomJoinThreadFn(OrtCustomJoinThreadFn ort_custom_join_thread_fn) {
  Ort::ThrowOnError(Ort::api->SessionOptionsSetCustomJoinThreadFn(this, ort_custom_join_thread_fn));
  return *this;
}

inline OrtSessionOptions& OrtSessionOptions::AppendExecutionProvider_OpenVINO(const OrtOpenVINOProviderOptions& provider_options) {
  Ort::ThrowOnError(Ort::api->SessionOptionsAppendExecutionProvider_OpenVINO(this, &provider_options));
  return *this;
}

/// Session
inline std::unique_ptr<OrtSession> OrtSession::Create(OrtEnv& env, const ORTCHAR_T* model_path, const OrtSessionOptions* options) {
  OrtSession *p;
  Ort::ThrowOnError(Ort::api->CreateSession(&env, model_path, options, &p));
  return std::unique_ptr<OrtSession>(p);
}

inline std::unique_ptr<OrtSession> OrtSession::Create(OrtEnv& env, const ORTCHAR_T* model_path, const OrtSessionOptions* options,
  OrtPrepackedWeightsContainer& prepacked_weights_container) {
  OrtSession* p;
  Ort::ThrowOnError(Ort::api->CreateSessionWithPrepackedWeightsContainer(&env, model_path, options, &prepacked_weights_container, &p));
  return std::unique_ptr<OrtSession>(p);
}

inline std::unique_ptr<OrtSession> OrtSession::Create(OrtEnv& env, const void* model_data, size_t model_data_length, const OrtSessionOptions* options) {
  OrtSession* p;
  Ort::ThrowOnError(Ort::api->CreateSessionFromArray(&env, model_data, model_data_length, options, &p));
  return std::unique_ptr<OrtSession>(p);
}

inline std::unique_ptr<OrtSession> OrtSession::Create(OrtEnv& env, const void* model_data, size_t model_data_length,
  const OrtSessionOptions* options, OrtPrepackedWeightsContainer& prepacked_weights_container) {
  OrtSession* p;
  Ort::ThrowOnError(Ort::api->CreateSessionFromArrayWithPrepackedWeightsContainer(&env, model_data, model_data_length, options,
    &prepacked_weights_container, &p));
  return std::unique_ptr<OrtSession>(p);
}

inline size_t OrtSession::GetInputCount() const {
  size_t out;
  Ort::ThrowOnError(Ort::api->SessionGetInputCount(this, &out));
  return out;
}

inline size_t OrtSession::GetOutputCount() const {
  size_t out;
  Ort::ThrowOnError(Ort::api->SessionGetOutputCount(this, &out));
  return out;
}

inline size_t OrtSession::GetOverridableInitializerCount() const {
  size_t out;
  Ort::ThrowOnError(Ort::api->SessionGetOverridableInitializerCount(this, &out));
  return out;
}

inline std::string OrtSession::GetInputName(size_t index) const {
  Ort::StringAllocator string_allocator;
  Ort::ThrowOnError(Ort::api->SessionGetInputName(this, index, &string_allocator, &string_allocator.out));
  return string_allocator;
}

inline std::vector<std::string> OrtSession::GetInputNames() const {
  std::vector<std::string> out;
  for (size_t i = 0, count = GetInputCount(); i < count; i++) {
    out.emplace_back(GetInputName(i));
  }
  return out;
}

inline std::string OrtSession::GetOutputName(size_t index) const {
  Ort::StringAllocator string_allocator;
  Ort::ThrowOnError(Ort::api->SessionGetOutputName(this, index, &string_allocator, &string_allocator.out));
  return string_allocator;
}

inline std::vector<std::string> OrtSession::GetOutputNames() const {
  std::vector<std::string> out;
  for (size_t i = 0, count = GetOutputCount(); i < count; i++) {
    out.emplace_back(GetOutputName(i));
  }
  return out;
}

inline std::string OrtSession::GetOverridableInitializerName(size_t index) const {
  Ort::StringAllocator string_allocator;
  Ort::ThrowOnError(Ort::api->SessionGetOverridableInitializerName(this, index, &string_allocator, &string_allocator.out));
  return string_allocator;
}

inline std::vector<std::string> OrtSession::GetOverridableInitializerNames() const {
  std::vector<std::string> out;
  for (size_t i = 0, count = GetOverridableInitializerCount(); i < count; i++) {
    out.emplace_back(GetOverridableInitializerName(i));
  }
  return out;
}

inline std::string OrtSession::EndProfiling() {
  Ort::StringAllocator string_allocator;
  Ort::ThrowOnError(Ort::api->SessionEndProfiling(this, &string_allocator, &string_allocator.out));
  return string_allocator;
}

inline uint64_t OrtSession::GetProfilingStartTimeNs() const {
  uint64_t out;
  Ort::ThrowOnError(Ort::api->SessionGetProfilingStartTimeNs(this, &out));
  return out;
}

inline std::unique_ptr<OrtModelMetadata> OrtSession::GetModelMetadata() const {
  OrtModelMetadata* out;
  Ort::ThrowOnError(Ort::api->SessionGetModelMetadata(this, &out));
  return std::unique_ptr<OrtModelMetadata>(out);
}

inline std::unique_ptr<OrtTypeInfo> OrtSession::GetInputTypeInfo(size_t index) const {
  OrtTypeInfo* out;
  Ort::ThrowOnError(Ort::api->SessionGetInputTypeInfo(this, index, &out));
  return std::unique_ptr<OrtTypeInfo>(out);
}

inline std::unique_ptr<OrtTypeInfo> OrtSession::GetOutputTypeInfo(size_t index) const {
  OrtTypeInfo* out;
  Ort::ThrowOnError(Ort::api->SessionGetOutputTypeInfo(this, index, &out));
  return std::unique_ptr<OrtTypeInfo>(out);
}

inline std::unique_ptr<OrtTypeInfo> OrtSession::GetOverridableInitializerTypeInfo(size_t index) const {
  OrtTypeInfo* out;
  Ort::ThrowOnError(Ort::api->SessionGetOverridableInitializerTypeInfo(this, index, &out));
  return std::unique_ptr<OrtTypeInfo>(out);
}

inline std::vector<std::unique_ptr<OrtValue>> OrtSession::Run(const OrtRunOptions* run_options, const char* const* input_names, const OrtValue* const* input_values, size_t input_count,
                                              const char* const* output_names, size_t output_count) {
  std::vector<OrtValue*> output_values(output_count);
  std::vector<std::unique_ptr<OrtValue>> results(output_count); // Allocate before Run() so that if it fails, it fails before we have unowned OrtValue*s in output_values
  Run(run_options, input_names, input_values, input_count, output_names, output_values.data(), output_count);

  for(auto i=0;i<output_count;i++)
    results[i]=std::unique_ptr<OrtValue>{output_values[i]};

  return results;
}

inline void OrtSession::Run(const OrtRunOptions* run_options, const char* const* input_names, const OrtValue* const* input_values, size_t input_count,
                                const char* const* output_names, OrtValue** output_values, size_t output_count) {
  Ort::ThrowOnError(Ort::api->Run(this, run_options, input_names, input_values, input_count, output_names, output_count, output_values));
}

inline void OrtSession::Run(const OrtRunOptions* run_options, const OrtIoBinding& io_binding) {
  Ort::ThrowOnError(Ort::api->RunWithBinding(this, run_options, &io_binding));
}

inline std::string OrtModelMetadata::GetProducerName() const {
  Ort::StringAllocator string_allocator;
  Ort::ThrowOnError(Ort::api->ModelMetadataGetProducerName(this, &string_allocator, &string_allocator.out));
  return string_allocator;
}

inline std::string OrtModelMetadata::GetGraphName() const {
  Ort::StringAllocator string_allocator;
  Ort::ThrowOnError(Ort::api->ModelMetadataGetGraphName(this, &string_allocator, &string_allocator.out));
  return string_allocator;
}

inline std::string OrtModelMetadata::GetDomain() const {
  Ort::StringAllocator string_allocator;
  Ort::ThrowOnError(Ort::api->ModelMetadataGetDomain(this, &string_allocator, &string_allocator.out));
  return string_allocator;
}

inline std::string OrtModelMetadata::GetDescription() const {
  Ort::StringAllocator string_allocator;
  Ort::ThrowOnError(Ort::api->ModelMetadataGetDescription(this, &string_allocator, &string_allocator.out));
  return string_allocator;
}

inline std::string OrtModelMetadata::GetGraphDescription() const {
  Ort::StringAllocator string_allocator;
  Ort::ThrowOnError(Ort::api->ModelMetadataGetGraphDescription(this, &string_allocator, &string_allocator.out));
  return string_allocator;
}

inline std::string OrtModelMetadata::LookupCustomMetadataMap(const char* key) const {
  Ort::StringAllocator string_allocator;
  Ort::ThrowOnError(Ort::api->ModelMetadataLookupCustomMetadataMap(this, &string_allocator, key, &string_allocator.out));
  return string_allocator;
}

inline std::vector<std::string> OrtModelMetadata::GetCustomMetadataMapKeysAllocated() const {
  char** out;
  int64_t num_keys = 0;
  Ort::ThrowOnError(Ort::api->ModelMetadataGetCustomMetadataMapKeys(this, &Ort::standard_allocator, &out, &num_keys));
  if (num_keys <= 0) {
    return {};
  }

  // Own the returned memory so it will be freed
  auto strings_deletor = [num_keys](char** out) { for(int64_t i = 0; i < num_keys; ++i) delete out[i]; };
  std::unique_ptr<char*, decltype(strings_deletor)> strings_guard(out, strings_deletor);

  std::vector<std::string> result;
  for (int64_t i = 0; i < num_keys; ++i)
    result.push_back(out[i]);
  return result;
}

inline int64_t OrtModelMetadata::GetVersion() const {
  int64_t out;
  Ort::ThrowOnError(Ort::api->ModelMetadataGetVersion(this, &out));
  return out;
}

inline ONNXTensorElementDataType OrtTensorTypeAndShapeInfo::GetElementType() const {
  ONNXTensorElementDataType out;
  Ort::ThrowOnError(Ort::api->GetTensorElementType(this, &out));
  return out;
}

inline size_t OrtTensorTypeAndShapeInfo::GetElementCount() const {
  size_t out;
  Ort::ThrowOnError(Ort::api->GetTensorShapeElementCount(this, &out));
  return static_cast<size_t>(out);
}

inline size_t GetDimensionsCount(const OrtTensorTypeAndShapeInfo *p) {
  size_t out;
  Ort::ThrowOnError(Ort::api->GetDimensionsCount(p, &out));
  return out;
}

inline std::vector<const char*> OrtTensorTypeAndShapeInfo::GetSymbolicDimensions() const {
  std::vector<const char*> out(GetDimensionsCount(this), nullptr);
  Ort::ThrowOnError(Ort::api->GetSymbolicDimensions(this, out.data(), out.size()));
  return out;
}

inline std::vector<int64_t> OrtTensorTypeAndShapeInfo::GetShape() const {
  std::vector<int64_t> out(GetDimensionsCount(this), 0);
  Ort::ThrowOnError(Ort::api->GetDimensions(this, out.data(), out.size()));
  return out;
}

inline std::unique_ptr<OrtTypeInfo> OrtSequenceTypeInfo::GetSequenceElementType() const {
  OrtTypeInfo* output;
  Ort::ThrowOnError(Ort::api->GetSequenceElementType(this, &output));
  return std::unique_ptr<OrtTypeInfo>{output};
}

inline ONNXTensorElementDataType OrtMapTypeInfo::GetMapKeyType() const {
  ONNXTensorElementDataType out;
  Ort::ThrowOnError(Ort::api->GetMapKeyType(this, &out));
  return out;
}

inline std::unique_ptr<OrtTypeInfo> OrtMapTypeInfo::GetMapValueType() const {
  OrtTypeInfo* output;
  Ort::ThrowOnError(Ort::api->GetMapValueType(this, &output));
  return std::unique_ptr<OrtTypeInfo>{output};
}

inline const OrtTensorTypeAndShapeInfo& OrtTypeInfo::GetTensorTypeAndShapeInfo() const {
  const OrtTensorTypeAndShapeInfo* out;
  Ort::ThrowOnError(Ort::api->CastTypeInfoToTensorInfo(this, &out));
  return *out;
}

inline const OrtSequenceTypeInfo& OrtTypeInfo::GetSequenceTypeInfo() const {
  const OrtSequenceTypeInfo* out;
  Ort::ThrowOnError(Ort::api->CastTypeInfoToSequenceTypeInfo(this, &out));
  return *out;
}

inline const OrtMapTypeInfo& OrtTypeInfo::GetMapTypeInfo() const {
  const OrtMapTypeInfo* out;
  Ort::ThrowOnError(Ort::api->CastTypeInfoToMapTypeInfo(this, &out));
  return *out;
}

inline ONNXType OrtTypeInfo::GetONNXType() const {
  ONNXType out;
  Ort::ThrowOnError(Ort::api->GetOnnxTypeFromTypeInfo(this, &out));
  return out;
}

template <typename T>
inline void OrtValue::GetOpaqueData(const char* domain, const char* type_name, T& out) const {
  Ort::ThrowOnError(Ort::api->GetOpaqueValue(domain, type_name, this, &out, sizeof(T)));
}

inline bool OrtValue::IsTensor() const {
  int out;
  Ort::ThrowOnError(Ort::api->IsTensor(this, &out));
  return out != 0;
}

inline bool OrtValue::HasValue() const {
  int out;
  Ort::ThrowOnError(Ort::api->HasValue(this, &out));
  return out != 0;
}

inline size_t OrtValue::GetCount() const {
  size_t out;
  Ort::ThrowOnError(Ort::api->GetValueCount(this, &out));
  return out;
}

inline std::unique_ptr<OrtValue> OrtValue::GetValue(int index) const {
  OrtValue* out;
  Ort::ThrowOnError(Ort::api->GetValue(this, index, &Ort::standard_allocator, &out));
  return std::unique_ptr<OrtValue>{out};
}

inline size_t OrtValue::GetStringTensorDataLength() const {
  size_t out;
  Ort::ThrowOnError(Ort::api->GetStringTensorDataLength(this, &out));
  return out;
}

inline size_t OrtValue::GetStringTensorElementLength(size_t element_index) const {
  size_t out;
  Ort::ThrowOnError(Ort::api->GetStringTensorElementLength(this, element_index, &out));
  return out;
}

template <typename T>
inline const T* OrtValue::GetTensorData() const {
  T* out;
  Ort::ThrowOnError(Ort::api->GetTensorMutableData(const_cast<OrtValue*>(this), (void**)&out));
  return out;
}

inline const void* OrtValue::GetTensorRawData() const {
  void* out;
  Ort::ThrowOnError(Ort::api->GetTensorMutableData(const_cast<OrtValue*>(this), &out));
  return out;
}

inline std::unique_ptr<OrtTypeInfo> OrtValue::GetTypeInfo() const {
  OrtTypeInfo* output;
  Ort::ThrowOnError(Ort::api->GetTypeInfo(this, &output));
  return std::unique_ptr<OrtTypeInfo>{output};
}

inline std::unique_ptr<OrtTensorTypeAndShapeInfo> OrtValue::GetTensorTypeAndShapeInfo() const {
  OrtTensorTypeAndShapeInfo* output;
  Ort::ThrowOnError(Ort::api->GetTensorTypeAndShape(this, &output));
  return std::unique_ptr<OrtTensorTypeAndShapeInfo>{output};
}

inline const OrtMemoryInfo& OrtValue::GetTensorMemoryInfo() const {
  const OrtMemoryInfo* mem_info;
  Ort::ThrowOnError(Ort::api->GetTensorMemoryInfo(this, &mem_info));
  return *mem_info;
}

inline void OrtValue::GetStringTensorElement(size_t buffer_length, size_t element_index, void* buffer) const {
  Ort::ThrowOnError(Ort::api->GetStringTensorElement(this, buffer_length, element_index, buffer));
}

inline void OrtValue::GetStringTensorContent(void* buffer, size_t buffer_length, size_t* offsets, size_t offsets_count) const {
  Ort::ThrowOnError(Ort::api->GetStringTensorContent(this, buffer, buffer_length, offsets, offsets_count));
}

#if !defined(DISABLE_SPARSE_TENSORS)
inline OrtSparseFormat OrtValue::GetSparseFormat() const {
  OrtSparseFormat format;
  Ort::ThrowOnError(Ort::api->GetSparseTensorFormat(this, &format));
  return format;
}

inline std::unique_ptr<OrtTensorTypeAndShapeInfo> OrtValue::GetSparseTensorValuesTypeAndShapeInfo() const {
  OrtTensorTypeAndShapeInfo* output;
  Ort::ThrowOnError(Ort::api->GetSparseTensorValuesTypeAndShape(this, &output));
  return std::unique_ptr<OrtTensorTypeAndShapeInfo>{output};
}

inline std::unique_ptr<OrtTensorTypeAndShapeInfo> OrtValue::GetSparseTensorIndicesTypeShapeInfo(OrtSparseIndicesFormat indices_format) const {
  OrtTensorTypeAndShapeInfo* output;
  Ort::ThrowOnError(Ort::api->GetSparseTensorIndicesTypeShape(this, indices_format, &output));
  return std::unique_ptr<OrtTensorTypeAndShapeInfo>{output};
}

template <typename T>
inline const T* OrtValue::GetSparseTensorIndicesData(OrtSparseIndicesFormat indices_format, size_t& num_indices) const {
  const void* out;
  Ort::ThrowOnError(Ort::api->GetSparseTensorIndices(this, indices_format, &num_indices, &out));
  return reinterpret_cast<const T*>(out);
}

inline bool OrtValue::IsSparseTensor() const {
  int out;
  Ort::ThrowOnError(Ort::api->IsSparseTensor(this, &out));
  return out != 0;
}

template <typename T>
inline const T* OrtValue::GetSparseTensorValues() const {
  const void* out;
  Ort::ThrowOnError(Ort::api->GetSparseTensorValues(this, &out));
  return reinterpret_cast<const T*>(out);
}

#endif

void OrtValue::FillStringTensor(const char* const* s, size_t s_len) {
  Ort::ThrowOnError(Ort::api->FillStringTensor(this, s, s_len));
}

void OrtValue::FillStringTensorElement(const char* s, size_t index) {
  Ort::ThrowOnError(Ort::api->FillStringTensorElement(this, s, index));
}

void* OrtValue::GetTensorMutableRawData() {
  void* out;
  Ort::ThrowOnError(Ort::api->GetTensorMutableData(this, &out));
  return out;
}

template <typename T>
T* OrtValue::GetTensorMutableData() {
  T* out;
  Ort::ThrowOnError(Ort::api->GetTensorMutableData(this, (void**)&out));
  return out;
}

template <typename T>
T& OrtValue::At(const std::vector<int64_t>& location) {
  static_assert(!std::is_same<T, std::string>::value, "this api does not support std::string");
  T* out;
  Ort::ThrowOnError(Ort::api->TensorAt(this, location.data(), location.size(), (void**)&out));
  return *out;
}

#if !defined(DISABLE_SPARSE_TENSORS)
void OrtValue::UseCooIndices(int64_t* indices_data, size_t indices_num) {
  Ort::ThrowOnError(Ort::api->UseCooIndices(this, indices_data, indices_num));
}

void OrtValue::UseCsrIndices(int64_t* inner_data, size_t inner_num, int64_t* outer_data, size_t outer_num) {
  Ort::ThrowOnError(Ort::api->UseCsrIndices(this, inner_data, inner_num, outer_data, outer_num));
}

void OrtValue::UseBlockSparseIndices(const OrtShape& indices_shape, int32_t* indices_data) {
  Ort::ThrowOnError(Ort::api->UseBlockSparseIndices(this, indices_shape.shape, indices_shape.shape_len, indices_data));
}

void OrtValue::FillSparseTensorCoo(const OrtMemoryInfo& mem_info, const OrtSparseValuesParam& values_param,
                                   const int64_t* indices_data, size_t indices_num) {
  Ort::ThrowOnError(Ort::api->FillSparseTensorCoo(this, &mem_info, values_param.values_shape,
                                            values_param.values_shape_len, values_param.data.p_data,
                                            indices_data, indices_num));
}

void OrtValue::FillSparseTensorCsr(const OrtMemoryInfo& data_mem_info,
                                   const OrtSparseValuesParam& values,
                                   const int64_t* inner_indices_data, size_t inner_indices_num,
                                   const int64_t* outer_indices_data, size_t outer_indices_num) {
  Ort::ThrowOnError(Ort::api->FillSparseTensorCsr(this, &data_mem_info, values.values_shape, values.values_shape_len, values.data.p_data,
                                            inner_indices_data, inner_indices_num,
                                            outer_indices_data, outer_indices_num));
}

void OrtValue::FillSparseTensorBlockSparse(const OrtMemoryInfo& data_mem_info,
                                           const OrtSparseValuesParam& values,
                                           const OrtShape& indices_shape,
                                           const int32_t* indices_data) {
  Ort::ThrowOnError(Ort::api->FillSparseTensorBlockSparse(this, &data_mem_info, values.values_shape, values.values_shape_len, values.data.p_data,
                                                    indices_shape.shape, indices_shape.shape_len,
                                                    indices_data));
}

#endif  // !defined(DISABLE_SPARSE_TENSORS)

template <typename T>
inline std::unique_ptr<OrtValue> OrtValue::CreateTensor(const OrtMemoryInfo& info, T* p_data, size_t p_data_element_count, const int64_t* shape, size_t shape_len) {
  return CreateTensor(info, p_data, p_data_element_count * sizeof(T), shape, shape_len, Ort::TypeToTensorType<T>::type);
}

inline std::unique_ptr<OrtValue> OrtValue::CreateTensor(const OrtMemoryInfo& info, void* p_data, size_t p_data_byte_count, const int64_t* shape, size_t shape_len,
                                 ONNXTensorElementDataType type) {
  OrtValue* out;
  Ort::ThrowOnError(Ort::api->CreateTensorWithDataAsOrtValue(&info, p_data, p_data_byte_count, shape, shape_len, type, &out));
  return std::unique_ptr<OrtValue>{out};
}

template <typename T>
inline std::unique_ptr<OrtValue> OrtValue::CreateTensor(OrtAllocator& allocator, const int64_t* shape, size_t shape_len) {
  return CreateTensor(&allocator, shape, shape_len, Ort::TypeToTensorType<T>::type);
}

inline std::unique_ptr<OrtValue> OrtValue::CreateTensor(OrtAllocator& allocator, const int64_t* shape, size_t shape_len, ONNXTensorElementDataType type) {
  OrtValue* out;
  Ort::ThrowOnError(Ort::api->CreateTensorAsOrtValue(&allocator, shape, shape_len, type, &out));
  return std::unique_ptr<OrtValue>{out};
}

#if !defined(DISABLE_SPARSE_TENSORS)

template <typename T>
inline std::unique_ptr<OrtValue> OrtValue::CreateSparseTensor(const OrtMemoryInfo& info, T* p_data, const OrtShape& dense_shape,
                                                              const OrtShape& values_shape) {
  return CreateSparseTensor(info, p_data, dense_shape, values_shape, Ort::TypeToTensorType<T>::type);
}

inline std::unique_ptr<OrtValue> OrtValue::CreateSparseTensor(const OrtMemoryInfo& info, void* p_data, const OrtShape& dense_shape,
                                       const OrtShape& values_shape, ONNXTensorElementDataType type) {
  OrtValue* out;
  Ort::ThrowOnError(Ort::api->CreateSparseTensorWithValuesAsOrtValue(&info, p_data, dense_shape.shape, dense_shape.shape_len,
                                                               values_shape.shape, values_shape.shape_len, type, &out));
  return std::unique_ptr<OrtValue>{out};
}

template <typename T>
inline std::unique_ptr<OrtValue> OrtValue::CreateSparseTensor(OrtAllocator* allocator, const OrtShape& dense_shape) {
  return CreateSparseTensor(allocator, dense_shape, Ort::TypeToTensorType<T>::type);
}

inline std::unique_ptr<OrtValue> OrtValue::CreateSparseTensor(OrtAllocator* allocator, const OrtShape& dense_shape,
                                       ONNXTensorElementDataType type) {
  OrtValue* out;
  Ort::ThrowOnError(Ort::api->CreateSparseTensorAsOrtValue(allocator, dense_shape.shape, dense_shape.shape_len, type, &out));
  return std::unique_ptr<OrtValue>{out};
}
#endif  // !defined(DISABLE_SPARSE_TENSORS)

inline std::unique_ptr<OrtValue> OrtValue::CreateMap(OrtValue& keys, OrtValue& values) {
  OrtValue* out;
  OrtValue* inputs[2] = {&keys, &values};
  Ort::ThrowOnError(Ort::api->CreateValue(inputs, 2, ONNX_TYPE_MAP, &out));
  return std::unique_ptr<OrtValue>{out};
}

inline std::unique_ptr<OrtValue> OrtValue::CreateSequence(const OrtValue* const* values, size_t count) {
  OrtValue* out;

  Ort::ThrowOnError(Ort::api->CreateValue(values, count, ONNX_TYPE_SEQUENCE, &out));
  return std::unique_ptr<OrtValue>{out};
}

template <typename T>
inline std::unique_ptr<OrtValue> OrtValue::CreateOpaque(const char* domain, const char* type_name, const T& data_container) {
  OrtValue* out;
  Ort::ThrowOnError(Ort::api->CreateOpaqueValue(domain, type_name, &data_container, sizeof(T), &out));
  return std::unique_ptr<OrtValue>{out};
}

//
// Custom OP Inlines
//
inline size_t OrtKernelContext::GetInputCount() const {
  size_t out;
  Ort::ThrowOnError(Ort::api->KernelContext_GetInputCount(this, &out));
  return out;
}

inline size_t OrtKernelContext::GetOutputCount() const {
  size_t out;
  Ort::ThrowOnError(Ort::api->KernelContext_GetOutputCount(this, &out));
  return out;
}

inline const OrtValue* OrtKernelContext::GetInput(size_t index) const {
  const OrtValue* out;
  Ort::ThrowOnError(Ort::api->KernelContext_GetInput(this, index, &out));
  return out;
}

inline OrtValue* OrtKernelContext::GetOutput(size_t index, const int64_t* dim_values, size_t dim_count) {
  OrtValue* out;
  Ort::ThrowOnError(Ort::api->KernelContext_GetOutput(this, index, dim_values, dim_count, &out));
  return out;
}

inline OrtValue* OrtKernelContext::GetOutput(size_t index, const std::vector<int64_t>& dims) {
  OrtValue* out;
  Ort::ThrowOnError(Ort::api->KernelContext_GetOutput(this, index, dims.data(), dims.size(), &out));
  return out;
}

inline void* OrtKernelContext::GetGPUComputeStream() const {
  void* out;
  Ort::ThrowOnError(Ort::api->KernelContext_GetGPUComputeStream(this, &out));
  return out;
}

inline std::unique_ptr<OrtOpAttr> OrtOpAttr::Create(const char* name, const void* data, int len, OrtOpAttrType type) {
  OrtOpAttr *p;
  Ort::ThrowOnError(Ort::api->CreateOpAttr(name, data, len, type, &p));
  return std::unique_ptr<OrtOpAttr>{p};
}

inline std::unique_ptr<OrtKernelInfo> OrtKernelInfo::Clone() const {
  OrtKernelInfo* p;
  Ort::ThrowOnError(Ort::api->CopyKernelInfo(this, &p));
  return std::unique_ptr<OrtKernelInfo>{p};
}

inline void OrtKernelInfo::GetAttr(const char* name, float& out) {
  Ort::ThrowOnError(Ort::api->KernelInfoGetAttribute_float(this, name, &out));
}

inline void OrtKernelInfo::GetAttr(const char* name, int64_t& out) {
  Ort::ThrowOnError(Ort::api->KernelInfoGetAttribute_int64(this, name, &out));
}

inline void OrtKernelInfo::GetAttr(const char* name, std::string& result) {
  size_t size = 0;
  // Feed nullptr for the data buffer to query the true size of the string attribute
  Ort::ThrowOnError(Ort::api->KernelInfoGetAttribute_string(this, name, nullptr, &size));

  std::string out;
  out.resize(size);
  Ort::ThrowOnError(Ort::api->KernelInfoGetAttribute_string(this, name, &out[0], &size));
  out.resize(size - 1);  // remove the terminating character '\0'
  out.swap(result);
}

inline void OrtKernelInfo::GetAttrs(const char* name, std::vector<float>& result) {
  size_t size = 0;
  // Feed nullptr for the data buffer to query the true size of the attribute
  Ort::ThrowOnError(Ort::api->KernelInfoGetAttributeArray_float(this, name, nullptr, &size));

  std::vector<float> out;
  out.resize(size);
  Ort::ThrowOnError(Ort::api->KernelInfoGetAttributeArray_float(this, name, out.data(), &size));
  out.swap(result);
}

inline void OrtKernelInfo::GetAttrs(const char* name, std::vector<int64_t>& result) {
  size_t size = 0;

  // Feed nullptr for the data buffer to query the true size of the attribute
  Ort::ThrowOnError(Ort::api->KernelInfoGetAttributeArray_int64(this, name, nullptr, &size));

  std::vector<int64_t> out;
  out.resize(size);
  Ort::ThrowOnError(Ort::api->KernelInfoGetAttributeArray_int64(this, name, out.data(), &size));
  out.swap(result);
}

inline std::unique_ptr<OrtOp> OrtOp::Create(const OrtKernelInfo* info, const char* op_name, const char* domain, int version,
                     const char** type_constraint_names,
                     const ONNXTensorElementDataType* type_constraint_values,
                     size_t type_constraint_count,
                     const OrtOpAttr* const* attr_values, size_t attr_count,
                     size_t input_count, size_t output_count) {
  OrtOp* p;
  Ort::ThrowOnError(Ort::api->CreateOp(info, op_name, domain, version, type_constraint_names, type_constraint_values,
                                      static_cast<int>(type_constraint_count),
                                      attr_values,
                                      static_cast<int>(attr_count),
                                      static_cast<int>(input_count),
                                      static_cast<int>(output_count), &p));
  return std::unique_ptr<OrtOp>{p};
}

inline void OrtOp::Invoke(const OrtKernelContext* context,
                       const OrtValue* const* input_values,
                       size_t input_count,
                       OrtValue* const* output_values,
                       size_t output_count) {
  Ort::ThrowOnError(Ort::api->InvokeOp(context, this, input_values, static_cast<int>(input_count),
                                      output_values, static_cast<int>(output_count)));
}
