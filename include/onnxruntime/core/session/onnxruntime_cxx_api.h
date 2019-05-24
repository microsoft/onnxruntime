// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "onnxruntime_c_api.h"
#include <cstddef>
#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace Ort {

using std::nullptr_t;

struct Exception : std::exception {
  Exception(std::string&& string, OrtErrorCode code) : message_{std::move(string)}, code_{code} {}

  OrtErrorCode GetOrtErrorCode() const { return code_; }
  const char* what() const noexcept override { return message_.c_str(); }

 private:
  std::string message_;
  OrtErrorCode code_;
};

#define ORT_THROW_ON_ERROR(expr)                                        \
  if (OrtStatus* onnx_status = (expr)) {                                \
    std::string ort_error_message = OrtGetErrorMessage(onnx_status);    \
    OrtErrorCode ort_error_code = OrtGetErrorCode(onnx_status);         \
    OrtReleaseStatus(onnx_status);                                      \
    throw Ort::Exception(std::move(ort_error_message), ort_error_code); \
  }

#define ORT_DEFINE_RELEASE(NAME) \
  inline void OrtRelease(Ort##NAME* ptr) { OrtRelease##NAME(ptr); }

ORT_DEFINE_RELEASE(Allocator);
ORT_DEFINE_RELEASE(AllocatorInfo);
ORT_DEFINE_RELEASE(CustomOpDomain);
ORT_DEFINE_RELEASE(Env);
ORT_DEFINE_RELEASE(RunOptions);
ORT_DEFINE_RELEASE(Session);
ORT_DEFINE_RELEASE(SessionOptions);
ORT_DEFINE_RELEASE(TensorTypeAndShapeInfo);
ORT_DEFINE_RELEASE(TypeInfo);
ORT_DEFINE_RELEASE(Value);

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

template <typename T>
struct Base {
  Base() = default;
  Base(T* p) : p_{p} {
    if (!p) throw Ort::Exception("Allocation failure", ORT_FAIL);
  }
  ~Base() { OrtRelease(p_); }

  operator T*() { return p_; }
  operator const T*() const { return p_; }

  T* release() {
    T* p = p_;
    p_ = nullptr;
    return p;
  }

 protected:
  Base(const Base&) = delete;
  Base(Base&& v) : p_{v.p_} { v.p_ = nullptr; }
  void operator=(Base&& v) {
    OrtRelease(p_);
    p_ = v.p_;
    v.p_ = nullptr;
  }

  T* p_{};

  template <typename>
  friend struct Unowned;
};

template <typename T>
struct Unowned : T {
  Unowned(decltype(T::p_) p) : T{p} {}
  Unowned(Unowned&& v) : T{v.p_} {}
  ~Unowned() { this->p_ = nullptr; }
};

struct Allocator;
struct AllocatorInfo;
struct Env;
struct TypeInfo;
struct Value;

struct Env : Base<OrtEnv> {
  Env(nullptr_t) {}
  Env(OrtLoggingLevel default_warning_level, _In_ const char* logid);
  Env(OrtLoggingLevel default_warning_level, const char* logid, OrtLoggingFunction logging_function, void* logger_param);
  explicit Env(OrtEnv* p) : Base<OrtEnv>{p} {}
};

struct CustomOpDomain : Base<OrtCustomOpDomain> {
  explicit CustomOpDomain(nullptr_t) {}
  explicit CustomOpDomain(const char* domain);

  void Add(OrtCustomOp* op);
};

struct RunOptions : Base<OrtRunOptions> {
  RunOptions(nullptr_t) {}
  RunOptions();

  RunOptions& SetRunLogVerbosityLevel(unsigned int);
  unsigned int GetRunLogVerbosityLevel() const;

  RunOptions& SetRunTag(const char* run_tag);
  const char* GetRunTag() const;

  RunOptions& SetTerminate(bool flag);
};

struct SessionOptions : Base<OrtSessionOptions> {
  explicit SessionOptions(nullptr_t) {}
  SessionOptions();
  explicit SessionOptions(OrtSessionOptions* p) : Base<OrtSessionOptions>{p} {}

  SessionOptions Clone() const;

  SessionOptions& SetThreadPoolSize(int session_thread_pool_size);
  SessionOptions& SetGraphOptimizationLevel(uint32_t graph_optimization_level);

  SessionOptions& EnableCpuMemArena();
  SessionOptions& DisableCpuMemArena();

  SessionOptions& EnableProfiling(const ORTCHAR_T* profile_file_prefix);
  SessionOptions& DisableProfiling();

  SessionOptions& EnableMemPattern();
  SessionOptions& DisableMemPattern();

  SessionOptions& EnableSequentialExecution();
  SessionOptions& DisableSequentialExecution();

  SessionOptions& SetLogId(const char* logid);

  SessionOptions& Add(OrtCustomOpDomain* custom_op_domain);
};

struct Session : Base<OrtSession> {
  explicit Session(nullptr_t) {}
  Session(Env& env, const ORTCHAR_T* model_path, const SessionOptions& options);
  Session(Env& env, const void* model_data, size_t model_data_length, const SessionOptions& options);

  // Run that will allocate the output values
  std::vector<Value> Run(const RunOptions& run_options, const char* const* input_names, Value* input_values, size_t input_count,
                         const char* const* output_names, size_t output_count);
  // Run for when there is a list of prealloated outputs
  void Run(const RunOptions& run_options, const char* const* input_names, Value* input_values, size_t input_count,
           const char* const* output_names, Value* output_values, size_t output_count);

  size_t GetInputCount() const;
  size_t GetOutputCount() const;

  char* GetInputName(size_t index, OrtAllocator* allocator) const;
  char* GetOutputName(size_t index, OrtAllocator* allocator) const;

  TypeInfo GetInputTypeInfo(size_t index) const;
  TypeInfo GetOutputTypeInfo(size_t index) const;
};

struct TensorTypeAndShapeInfo : Base<OrtTensorTypeAndShapeInfo> {
  explicit TensorTypeAndShapeInfo(nullptr_t) {}
  explicit TensorTypeAndShapeInfo(OrtTensorTypeAndShapeInfo* p) : Base<OrtTensorTypeAndShapeInfo>{p} {}

  ONNXTensorElementDataType GetElementType() const;
  size_t GetElementCount() const;

  size_t GetDimensionsCount() const;
  void GetDimensions(int64_t* values, size_t values_count) const;
  std::vector<int64_t> GetShape() const;
};

struct TypeInfo : Base<OrtTypeInfo> {
  explicit TypeInfo(nullptr_t) {}
  explicit TypeInfo(OrtTypeInfo* p) : Base<OrtTypeInfo>{p} {}

  Unowned<TensorTypeAndShapeInfo> GetTensorTypeAndShapeInfo() const;
  ONNXType GetONNXType() const;
};

struct Value : Base<OrtValue> {
  template <typename T>
  static Value CreateTensor(const OrtAllocatorInfo* info, T* p_data, size_t p_data_element_count, const int64_t* shape, size_t shape_len);
  static Value CreateTensor(const OrtAllocatorInfo* info, void* p_data, size_t p_data_byte_count, const int64_t* shape, size_t shape_len,
                            ONNXTensorElementDataType type);
  template <typename T>
  static Value CreateTensor(OrtAllocator* allocator, const int64_t* shape, size_t shape_len);
  static Value CreateTensor(OrtAllocator* allocator, const int64_t* shape, size_t shape_len, ONNXTensorElementDataType type);

  static Value CreateMap(Value& keys, Value& values);
  static Value CreateSequence(std::vector<Value>& values);

  explicit Value(nullptr_t) {}
  explicit Value(OrtValue* p) : Base<OrtValue>{p} {}

  bool IsTensor() const;
  size_t GetCount() const;  // If a non tensor, returns 2 for map and N for sequence, where N is the number of elements
  Value GetValue(int index, OrtAllocator* allocator) const;

  size_t GetStringTensorDataLength() const;
  void GetStringTensorContent(void* buffer, size_t buffer_length, size_t* offsets, size_t offsets_count) const;

  template <typename T>
  T* GetTensorMutableData();

  TypeInfo GetTypeInfo() const;
  TensorTypeAndShapeInfo GetTensorTypeAndShapeInfo() const;
};

struct Allocator : Base<OrtAllocator> {
  static Allocator CreateDefault();

  explicit Allocator(nullptr_t) {}
  explicit Allocator(OrtAllocator* p) : Base<OrtAllocator>{p} {}

  void* Alloc(size_t size);
  void Free(void* p);

  const OrtAllocatorInfo* GetInfo() const;
};

struct AllocatorInfo : Base<OrtAllocatorInfo> {
  static AllocatorInfo CreateCpu(OrtAllocatorType type, OrtMemType mem_type1);

  explicit AllocatorInfo(nullptr_t) {}
  AllocatorInfo(const char* name, OrtAllocatorType type, int id, OrtMemType mem_type);

  explicit AllocatorInfo(OrtAllocatorInfo* p) : Base<OrtAllocatorInfo>{p} {}
};

}  // namespace Ort

namespace Ort {

inline Allocator Allocator::CreateDefault() {
  OrtAllocator* p;
  ORT_THROW_ON_ERROR(OrtCreateDefaultAllocator(&p));
  return Allocator(p);
}

inline void* Allocator::Alloc(size_t size) {
  return OrtAllocatorAlloc(p_, size);
}

inline void Allocator::Free(void* p) {
  OrtAllocatorFree(p_, p);
}

inline const OrtAllocatorInfo* Allocator::GetInfo() const {
  return OrtAllocatorGetInfo(p_);
}

inline AllocatorInfo AllocatorInfo::CreateCpu(OrtAllocatorType type, OrtMemType mem_type) {
  OrtAllocatorInfo* p;
  ORT_THROW_ON_ERROR(OrtCreateCpuAllocatorInfo(type, mem_type, &p));
  return AllocatorInfo(p);
}

inline AllocatorInfo::AllocatorInfo(const char* name, OrtAllocatorType type, int id, OrtMemType mem_type) {
  ORT_THROW_ON_ERROR(OrtCreateAllocatorInfo(name, type, id, mem_type, &p_));
}

inline Env::Env(OrtLoggingLevel default_warning_level, _In_ const char* logid) {
  ORT_THROW_ON_ERROR(OrtCreateEnv(default_warning_level, logid, &p_));
}

inline Env::Env(OrtLoggingLevel default_warning_level, const char* logid, OrtLoggingFunction logging_function, void* logger_param) {
  ORT_THROW_ON_ERROR(OrtCreateEnvWithCustomLogger(logging_function, logger_param, default_warning_level, logid, &p_));
}

inline CustomOpDomain::CustomOpDomain(const char* domain)
    : Base<OrtCustomOpDomain>{OrtCreateCustomOpDomain(domain)} {
}

inline void CustomOpDomain::Add(OrtCustomOp* op) {
  ORT_THROW_ON_ERROR(OrtCustomOpDomain_Add(p_, op));
}

inline RunOptions::RunOptions() : Base<OrtRunOptions>{OrtCreateRunOptions()} {}

inline RunOptions& RunOptions::SetRunLogVerbosityLevel(unsigned int level) {
  ORT_THROW_ON_ERROR(OrtRunOptionsSetRunLogVerbosityLevel(p_, level));
  return *this;
}

inline unsigned int RunOptions::GetRunLogVerbosityLevel() const {
  return OrtRunOptionsGetRunLogVerbosityLevel(p_);
}

inline RunOptions& RunOptions::SetRunTag(const char* run_tag) {
  ORT_THROW_ON_ERROR(OrtRunOptionsSetRunTag(p_, run_tag));
  return *this;
}

inline const char* RunOptions::GetRunTag() const {
  return OrtRunOptionsGetRunTag(p_);
}

inline RunOptions& RunOptions::SetTerminate(bool flag) {
  OrtRunOptionsSetTerminate(p_, flag ? 1 : 0);
  return *this;
}

inline SessionOptions::SessionOptions() : Base<OrtSessionOptions>{OrtCreateSessionOptions()} {
}

inline SessionOptions SessionOptions::Clone() const {
  return SessionOptions{OrtCloneSessionOptions(p_)};
}

inline SessionOptions& SessionOptions::SetThreadPoolSize(int session_thread_pool_size) {
  if (OrtSetSessionThreadPoolSize(p_, session_thread_pool_size) == -1)
    throw Exception("Error calling SessionOptions::SetThreadPoolSize", ORT_FAIL);
  return *this;
}

inline SessionOptions& SessionOptions::SetGraphOptimizationLevel(uint32_t graph_optimization_level) {
  if (OrtSetSessionGraphOptimizationLevel(p_, graph_optimization_level) == -1)
    throw Exception("Error calling SessionOptions::SetGraphOptimizationLevel", ORT_FAIL);
  return *this;
}

inline SessionOptions& SessionOptions::EnableProfiling(const ORTCHAR_T* profile_file_prefix) {
  OrtEnableProfiling(p_, profile_file_prefix);
  return *this;
}

inline SessionOptions& SessionOptions::DisableProfiling() {
  OrtDisableProfiling(p_);
  return *this;
}

inline SessionOptions& SessionOptions::EnableMemPattern() {
  OrtEnableMemPattern(p_);
  return *this;
}

inline SessionOptions& SessionOptions::DisableMemPattern() {
  OrtDisableMemPattern(p_);
  return *this;
}

inline SessionOptions& SessionOptions::EnableCpuMemArena() {
  OrtEnableCpuMemArena(p_);
  return *this;
}

inline SessionOptions& SessionOptions::DisableCpuMemArena() {
  OrtDisableCpuMemArena(p_);
  return *this;
}

inline SessionOptions& SessionOptions::EnableSequentialExecution() {
  OrtEnableSequentialExecution(p_);
  return *this;
}

inline SessionOptions& SessionOptions::DisableSequentialExecution() {
  OrtDisableSequentialExecution(p_);
  return *this;
}

inline SessionOptions& SessionOptions::SetLogId(const char* logid) {
  OrtSetSessionLogId(p_, logid);
  return *this;
}
inline SessionOptions& SessionOptions::Add(OrtCustomOpDomain* custom_op_domain) {
  ORT_THROW_ON_ERROR(OrtAddCustomOpDomain(p_, custom_op_domain));
  return *this;
}

inline Session::Session(Env& env, const ORTCHAR_T* model_path, const SessionOptions& options) {
  ORT_THROW_ON_ERROR(OrtCreateSession(env, model_path, options, &p_));
}

inline Session::Session(Env& env, const void* model_data, size_t model_data_length, const SessionOptions& options) {
  ORT_THROW_ON_ERROR(OrtCreateSessionFromArray(env, model_data, model_data_length, options, &p_));
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
  ORT_THROW_ON_ERROR(OrtRun(p_, run_options, input_names, ort_input_values, input_count, output_names, output_count, ort_output_values));
}

inline size_t Session::GetInputCount() const {
  size_t out;
  ORT_THROW_ON_ERROR(OrtSessionGetInputCount(p_, &out));
  return out;
}

inline size_t Session::GetOutputCount() const {
  size_t out;
  ORT_THROW_ON_ERROR(OrtSessionGetOutputCount(p_, &out));
  return out;
}

inline char* Session::GetInputName(size_t index, OrtAllocator* allocator) const {
  char* out;
  ORT_THROW_ON_ERROR(OrtSessionGetInputName(p_, index, allocator, &out));
  return out;
}

inline char* Session::GetOutputName(size_t index, OrtAllocator* allocator) const {
  char* out;
  ORT_THROW_ON_ERROR(OrtSessionGetOutputName(p_, index, allocator, &out));
  return out;
}

inline TypeInfo Session::GetInputTypeInfo(size_t index) const {
  OrtTypeInfo* out;
  ORT_THROW_ON_ERROR(OrtSessionGetInputTypeInfo(p_, index, &out));
  return TypeInfo{out};
}

inline TypeInfo Session::GetOutputTypeInfo(size_t index) const {
  OrtTypeInfo* out;
  ORT_THROW_ON_ERROR(OrtSessionGetOutputTypeInfo(p_, index, &out));
  return TypeInfo{out};
}

inline ONNXTensorElementDataType TensorTypeAndShapeInfo::GetElementType() const {
  return OrtGetTensorElementType(p_);
}

inline size_t TensorTypeAndShapeInfo::GetElementCount() const {
  return static_cast<size_t>(OrtGetTensorShapeElementCount(p_));
}

inline size_t TensorTypeAndShapeInfo::GetDimensionsCount() const {
  return OrtGetDimensionsCount(p_);
}

inline void TensorTypeAndShapeInfo::GetDimensions(int64_t* values, size_t values_count) const {
  OrtGetDimensions(p_, values, values_count);
}

inline std::vector<int64_t> TensorTypeAndShapeInfo::GetShape() const {
  std::vector<int64_t> out(GetDimensionsCount(), 0);
  GetDimensions(out.data(), out.size());
  return out;
}

inline Unowned<TensorTypeAndShapeInfo> TypeInfo::GetTensorTypeAndShapeInfo() const {
  return Unowned<TensorTypeAndShapeInfo>{const_cast<OrtTensorTypeAndShapeInfo*>(OrtCastTypeInfoToTensorInfo(p_))};
}

inline ONNXType TypeInfo::GetONNXType() const {
  return OrtOnnxTypeFromTypeInfo(p_);
}

template <typename T>
inline Value Value::CreateTensor(const OrtAllocatorInfo* info, T* p_data, size_t p_data_element_count, const int64_t* shape, size_t shape_len) {
  return CreateTensor(info, p_data, p_data_element_count * sizeof(T), shape, shape_len, TypeToTensorType<T>::type);
}

inline Value Value::CreateTensor(const OrtAllocatorInfo* info, void* p_data, size_t p_data_byte_count, const int64_t* shape, size_t shape_len,
                                 ONNXTensorElementDataType type) {
  OrtValue* out;
  ORT_THROW_ON_ERROR(OrtCreateTensorWithDataAsOrtValue(info, p_data, p_data_byte_count, shape, shape_len, type, &out));
  return Value{out};
}

template <typename T>
inline Value Value::CreateTensor(OrtAllocator* allocator, const int64_t* shape, size_t shape_len) {
  return CreateTensor(allocator, shape, shape_len, TypeToTensorType<T>::type);
}

inline Value Value::CreateTensor(OrtAllocator* allocator, const int64_t* shape, size_t shape_len, ONNXTensorElementDataType type) {
  OrtValue* out;
  ORT_THROW_ON_ERROR(OrtCreateTensorAsOrtValue(allocator, shape, shape_len, type, &out));
  return Value{out};
}

ORT_API_STATUS(OrtCreateTensorAsOrtValue, _Inout_ OrtAllocator* allocator,
               _In_ const int64_t* shape, size_t shape_len, ONNXTensorElementDataType type,
               _Out_ OrtValue** out);

inline Value Value::CreateMap(Value& keys, Value& values) {
  OrtValue* out;
  OrtValue* inputs[2] = {keys, values};
  ORT_THROW_ON_ERROR(OrtCreateValue(inputs, 2, ONNX_TYPE_MAP, &out));
  return Value{out};
}

inline Value Value::CreateSequence(std::vector<Value>& values) {
  OrtValue* out;
  std::vector<OrtValue*> values_ort{values.data(), values.data() + values.size()};
  ORT_THROW_ON_ERROR(OrtCreateValue(values_ort.data(), values_ort.size(), ONNX_TYPE_SEQUENCE, &out));
  return Value{out};
}

inline bool Value::IsTensor() const {
  return OrtIsTensor(p_) != 0;
}

inline size_t Value::GetCount() const {
  size_t out;
  ORT_THROW_ON_ERROR(OrtGetValueCount(p_, &out));
  return out;
}

inline Value Value::GetValue(int index, OrtAllocator* allocator) const {
  OrtValue* out;
  ORT_THROW_ON_ERROR(OrtGetValue(p_, index, allocator, &out));
  return Value{out};
}

inline size_t Value::GetStringTensorDataLength() const {
  size_t out;
  ORT_THROW_ON_ERROR(OrtGetStringTensorDataLength(p_, &out));
  return out;
}

inline void Value::GetStringTensorContent(void* buffer, size_t buffer_length, size_t* offsets, size_t offsets_count) const {
  ORT_THROW_ON_ERROR(OrtGetStringTensorContent(p_, buffer, buffer_length, offsets, offsets_count));
}

template <typename T>
T* Value::GetTensorMutableData() {
  T* out;
  ORT_THROW_ON_ERROR(OrtGetTensorMutableData(p_, (void**)&out));
  return out;
}

inline TypeInfo Value::GetTypeInfo() const {
  OrtTypeInfo* output;
  ORT_THROW_ON_ERROR(OrtGetTypeInfo(p_, &output));
  return TypeInfo{output};
}

inline TensorTypeAndShapeInfo Value::GetTensorTypeAndShapeInfo() const {
  OrtTensorTypeAndShapeInfo* output;
  ORT_THROW_ON_ERROR(OrtGetTensorTypeAndShape(p_, &output));
  return TensorTypeAndShapeInfo{output};
}

}  // namespace Ort

namespace Ort {
struct CustomOpApi {
  CustomOpApi(const OrtCustomOpApi& api) : api_(api) {}

  template <typename T>
  T KernelInfoGetAttribute(_In_ const OrtKernelInfo* info, _In_ const char* name);

  OrtTensorTypeAndShapeInfo* GetTensorTypeAndShape(_In_ const OrtValue* value) {
    OrtTensorTypeAndShapeInfo* out;
    ORT_THROW_ON_ERROR(api_.GetTensorTypeAndShape(value, &out));
    return out;
  }

  int64_t GetTensorShapeElementCount(_In_ const OrtTensorTypeAndShapeInfo* info) {
    return api_.GetTensorShapeElementCount(info);
  }

  ONNXTensorElementDataType GetTensorElementType(const OrtTensorTypeAndShapeInfo* info) {
    return api_.GetTensorElementType(info);
  }

  size_t GetDimensionCount(_In_ const OrtTensorTypeAndShapeInfo* info) {
    return api_.GetDimensionCount(info);
  }

  void GetDimensions(_In_ const OrtTensorTypeAndShapeInfo* info, _Out_ int64_t* dim_values, size_t dim_values_length) {
    api_.GetDimensions(info, dim_values, dim_values_length);
  }

  void SetDimensions(OrtTensorTypeAndShapeInfo* info, _In_ const int64_t* dim_values, size_t dim_count) {
    api_.SetDimensions(info, dim_values, dim_count);
  }

  template <typename T>
  T* GetTensorMutableData(_Inout_ OrtValue* value) {
    T* data;
    ORT_THROW_ON_ERROR(api_.GetTensorMutableData(value, reinterpret_cast<void**>(&data)));
    return data;
  }

  template <typename T>
  const T* GetTensorData(_Inout_ const OrtValue* value) {
    return GetTensorMutableData<T>(const_cast<OrtValue*>(value));
  }

  std::vector<int64_t> GetTensorShape(const OrtTensorTypeAndShapeInfo* info) {
    std::vector<int64_t> output(GetDimensionCount(info));
    GetDimensions(info, output.data(), output.size());
    return output;
  }

  void ReleaseTensorTypeAndShapeInfo(OrtTensorTypeAndShapeInfo* input) {
    api_.ReleaseTensorTypeAndShapeInfo(input);
  }

  size_t KernelContext_GetInputCount(const OrtKernelContext* context) {
    return api_.KernelContext_GetInputCount(context);
  }

  const OrtValue* KernelContext_GetInput(const OrtKernelContext* context, _In_ size_t index) {
    return api_.KernelContext_GetInput(context, index);
  }

  size_t KernelContext_GetOutputCount(const OrtKernelContext* context) {
    return api_.KernelContext_GetOutputCount(context);
  }

  OrtValue* KernelContext_GetOutput(OrtKernelContext* context, _In_ size_t index, _In_ const int64_t* dim_values, size_t dim_count) {
    return api_.KernelContext_GetOutput(context, index, dim_values, dim_count);
  }

 private:
  const OrtCustomOpApi& api_;
};

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

template <typename TOp, typename TKernel>
struct CustomOpBase : OrtCustomOp {
  CustomOpBase() {
    OrtCustomOp::version = ORT_API_VERSION;
    OrtCustomOp::CreateKernel = [](OrtCustomOp* this_, const OrtCustomOpApi* api, const OrtKernelInfo* info) { return static_cast<TOp*>(this_)->CreateKernel(*api, info); };
    OrtCustomOp::GetName = [](OrtCustomOp* this_) { return static_cast<TOp*>(this_)->GetName(); };

    OrtCustomOp::GetInputTypeCount = [](OrtCustomOp* this_) { return static_cast<TOp*>(this_)->GetInputTypeCount(); };
    OrtCustomOp::GetInputType = [](OrtCustomOp* this_, size_t index) { return static_cast<TOp*>(this_)->GetInputType(index); };

    OrtCustomOp::GetOutputTypeCount = [](OrtCustomOp* this_) { return static_cast<TOp*>(this_)->GetOutputTypeCount(); };
    OrtCustomOp::GetOutputType = [](OrtCustomOp* this_, size_t index) { return static_cast<TOp*>(this_)->GetOutputType(index); };

    OrtCustomOp::KernelCompute = [](void* op_kernel, OrtKernelContext* context) { static_cast<TKernel*>(op_kernel)->Compute(context); };
    OrtCustomOp::KernelDestroy = [](void* op_kernel) { delete static_cast<TKernel*>(op_kernel); };
  }
};

}  // namespace Ort

#undef ORT_REDIRECT_SIMPLE_FUNCTION_CALL
