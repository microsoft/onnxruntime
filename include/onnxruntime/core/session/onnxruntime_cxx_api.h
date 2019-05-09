// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "onnxruntime_c_api.h"
#include <cstddef>
#include <array>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define ORT_REDIRECT_SIMPLE_FUNCTION_CALL(NAME) \
  decltype(Ort##NAME(value.get())) NAME() {     \
    return Ort##NAME(value.get());              \
  }

#define ORT_DEFINE_DELETER(NAME)      \
  template <>                         \
  struct default_delete<Ort##NAME> {  \
    void operator()(Ort##NAME* ptr) { \
      OrtRelease##NAME(ptr);          \
    }                                 \
  };

namespace std {
ORT_DEFINE_DELETER(Allocator);
ORT_DEFINE_DELETER(TypeInfo);
ORT_DEFINE_DELETER(RunOptions);
ORT_DEFINE_DELETER(SessionOptions);
ORT_DEFINE_DELETER(TensorTypeAndShapeInfo);
}  // namespace std

namespace Ort {

using std::nullptr_t;

struct Exception : std::exception {
  Exception(std::string&& string, OrtErrorCode code) : message_{std::move(string)}, code_{code} {}

  OrtErrorCode GetOrtErrorCode() const { return code_; }
  const char* what() const noexcept { return message_.c_str(); }

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
  inline void Release(Ort##NAME* ptr) { OrtRelease##NAME(ptr); }

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
struct Base {
  Base() = default;
  Base(T* p) : p_{p} {}
  ~Base() { Release(p_); }

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
    Release(p_);
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

struct TypeInfo;
struct Value;

struct Env : Base<OrtEnv> {
  Env(nullptr_t) {}
  Env(OrtLoggingLevel default_warning_level, _In_ const char* logid);
};

struct CustomOpDomain : Base<OrtCustomOpDomain> {
  CustomOpDomain(nullptr_t) {}
  CustomOpDomain(const char* domain);

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
  SessionOptions(nullptr_t) {}
  SessionOptions();
  explicit SessionOptions(OrtSessionOptions* p) : Base<OrtSessionOptions>{p} {}

  SessionOptions clone() const;

  SessionOptions& SetThreadPoolSize(int session_thread_pool_size);
  SessionOptions& SetGraphOptimizationLevel(uint32_t graph_optimization_level);

  SessionOptions& EnableCpuMemArena();
  SessionOptions& DisableCpuMemArena();

  SessionOptions& EnableSequentialExecution();
  SessionOptions& DisableSequentialExecution();

  SessionOptions& SetLogId(const char* logid);

  SessionOptions& Add(OrtCustomOpDomain* custom_op_domain);
};

struct Session : Base<OrtSession> {
  Session(nullptr_t) {}
  Session(OrtEnv* env, const ORTCHAR_T* model_path, const OrtSessionOptions* options);

  template <unsigned InputCount>
  Value Run(OrtRunOptions* run_options, const char* const* input_names, Value (&input)[InputCount],
            const char* const* output_names, size_t output_names_len);

  size_t GetInputCount() const;
  size_t GetOutputCount() const;

  char* GetInputName(size_t index, OrtAllocator* allocator) const;
  char* GetOutputName(size_t index, OrtAllocator* allocator) const;

  TypeInfo GetInputTypeInfo(size_t index) const;
  TypeInfo GetOutputTypeInfo(size_t index) const;
};

struct TensorTypeAndShapeInfo : Base<OrtTensorTypeAndShapeInfo> {
  TensorTypeAndShapeInfo(nullptr_t) {}
  explicit TensorTypeAndShapeInfo(OrtTensorTypeAndShapeInfo* p) : Base<OrtTensorTypeAndShapeInfo>{p} {}

  ONNXTensorElementDataType GetElementType() const;

  size_t GetDimensionsCount() const;
  void GetDimensions(int64_t* values, size_t values_count) const;
  std::vector<int64_t> GetShape() const;
};

struct TypeInfo : Base<OrtTypeInfo> {
  TypeInfo(nullptr_t) {}
  explicit TypeInfo(OrtTypeInfo* p) : Base<OrtTypeInfo>{p} {}

  Unowned<TensorTypeAndShapeInfo> GetTensorTypeAndShapeInfo() const;
};

struct Value : Base<OrtValue> {
  static Value CreateTensor(const OrtAllocatorInfo* info, void* p_data, size_t p_data_len, const int64_t* shape, size_t shape_len,
                            ONNXTensorElementDataType type);

  Value(nullptr_t) {}
  explicit Value(OrtValue* p) : Base<OrtValue>{p} {}

  bool IsTensor() const;
  template <typename T>
  T* GetTensorMutableData();

  TensorTypeAndShapeInfo GetTensorTypeAndShapeInfo() const;
};

struct Allocator : Base<OrtAllocator> {
  static Allocator Create_Default();

  Allocator(nullptr_t) {}
  explicit Allocator(OrtAllocator* p) : Base<OrtAllocator>{p} {}

  void* Alloc(size_t size);
  void Free(void* p);

  const OrtAllocatorInfo* GetInfo() const;
};

struct AllocatorInfo : Base<OrtAllocatorInfo> {
  static AllocatorInfo Create_Cpu(OrtAllocatorType type, OrtMemType mem_type1);

  AllocatorInfo(nullptr_t) {}
  explicit AllocatorInfo(OrtAllocatorInfo* p) : Base<OrtAllocatorInfo>{p} {}
};
}  // namespace Ort

namespace Ort {
inline Allocator Allocator::Create_Default() {
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

inline AllocatorInfo AllocatorInfo::Create_Cpu(OrtAllocatorType type, OrtMemType mem_type) {
  OrtAllocatorInfo* p;
  ORT_THROW_ON_ERROR(OrtCreateCpuAllocatorInfo(type, mem_type, &p));
  return AllocatorInfo(p);
}

inline Env::Env(OrtLoggingLevel default_warning_level, _In_ const char* logid) {
  ORT_THROW_ON_ERROR(OrtCreateEnv(default_warning_level, logid, &p_));
}

inline CustomOpDomain::CustomOpDomain(const char* domain)
    : Base<OrtCustomOpDomain>{OrtCreateCustomOpDomain(domain)} {
}

inline void CustomOpDomain::Add(OrtCustomOp* op) {
  ORT_THROW_ON_ERROR(OrtCustomOpDomain_Add(p_, op));
}

inline RunOptions::RunOptions() : Base<OrtRunOptions>{
                                      OrtCreateRunOptions()} {}

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

inline SessionOptions SessionOptions::clone() const {
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

inline Session::Session(OrtEnv* env, const ORTCHAR_T* model_path, const OrtSessionOptions* options) {
  ORT_THROW_ON_ERROR(OrtCreateSession(env, model_path, options, &p_));
}

template <unsigned InputCount>
inline Value Session::Run(OrtRunOptions* run_options, const char* const* input_names, Value (&inputs)[InputCount],
                          const char* const* output_names, size_t output_names_len) {
  std::array<OrtValue*, InputCount> internal_inputs;
  std::copy_n(inputs, InputCount, internal_inputs.data());

  OrtValue* out{};
  ORT_THROW_ON_ERROR(OrtRun(p_, run_options, input_names, internal_inputs.data(), InputCount, output_names, output_names_len, &out));
  return Value{out};
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

inline size_t TensorTypeAndShapeInfo::GetDimensionsCount() const {
  return OrtGetNumOfDimensions(p_);
}

inline void TensorTypeAndShapeInfo::GetDimensions(int64_t* values, size_t values_count) const {
  OrtGetDimensions(p_, values, values_count);
}

inline std::vector<int64_t> TensorTypeAndShapeInfo::GetShape() const {
  std::vector<int64_t> output;
  output.resize(GetDimensionsCount());
  GetDimensions(output.data(), output.size());
  return output;
}

inline Unowned<TensorTypeAndShapeInfo> TypeInfo::GetTensorTypeAndShapeInfo() const {
  return Unowned<TensorTypeAndShapeInfo>{const_cast<OrtTensorTypeAndShapeInfo*>(OrtCastTypeInfoToTensorInfo(p_))};
}

inline Value Value::CreateTensor(const OrtAllocatorInfo* info, void* p_data, size_t p_data_len, const int64_t* shape, size_t shape_len,
                                 ONNXTensorElementDataType type) {
  OrtValue* out;
  ORT_THROW_ON_ERROR(OrtCreateTensorWithDataAsOrtValue(info, p_data, p_data_len, shape, shape_len, type, &out));
  return Value(out);
}

inline bool Value::IsTensor() const {
  return OrtIsTensor(p_) != 0;
}

template <typename T>
T* Value::GetTensorMutableData() {
  T* output;
  ORT_THROW_ON_ERROR(OrtGetTensorMutableData(p_, (void**)&output));
  return output;
}

inline TensorTypeAndShapeInfo Value::GetTensorTypeAndShapeInfo() const {
  OrtTensorTypeAndShapeInfo* output;
  ORT_THROW_ON_ERROR(OrtGetTensorShapeAndType(p_, &output));
  return TensorTypeAndShapeInfo{output};
}

}  // namespace Ort

// Deprecated: Will be removed once all dependencies of it are removed
#if 1
namespace onnxruntime {

class SessionOptionsWrapper {
 private:
  std::unique_ptr<OrtSessionOptions> value;
  OrtEnv* env_;
  SessionOptionsWrapper(_In_ OrtEnv* env, OrtSessionOptions* p) : value(p), env_(env){};

 public:
  operator OrtSessionOptions*() { return value.get(); }

  //TODO: for the input arg, should we call addref here?
  SessionOptionsWrapper(_In_ OrtEnv* env) : value(OrtCreateSessionOptions()), env_(env){};
  ORT_REDIRECT_SIMPLE_FUNCTION_CALL(EnableSequentialExecution)
  ORT_REDIRECT_SIMPLE_FUNCTION_CALL(DisableSequentialExecution)
  ORT_REDIRECT_SIMPLE_FUNCTION_CALL(DisableProfiling)
  ORT_REDIRECT_SIMPLE_FUNCTION_CALL(EnableMemPattern)
  ORT_REDIRECT_SIMPLE_FUNCTION_CALL(DisableMemPattern)
  ORT_REDIRECT_SIMPLE_FUNCTION_CALL(EnableCpuMemArena)
  ORT_REDIRECT_SIMPLE_FUNCTION_CALL(DisableCpuMemArena)
  void EnableProfiling(_In_ const ORTCHAR_T* profile_file_prefix) {
    OrtEnableProfiling(value.get(), profile_file_prefix);
  }

  void SetSessionLogId(const char* logid) {
    OrtSetSessionLogId(value.get(), logid);
  }
  void SetSessionLogVerbosityLevel(uint32_t session_log_verbosity_level) {
    OrtSetSessionLogVerbosityLevel(value.get(), session_log_verbosity_level);
  }
  int SetSessionGraphOptimizationLevel(uint32_t graph_optimization_level) {
    return OrtSetSessionGraphOptimizationLevel(value.get(), graph_optimization_level);
  }
  void SetSessionThreadPoolSize(int session_thread_pool_size) {
    OrtSetSessionThreadPoolSize(value.get(), session_thread_pool_size);
  }

  SessionOptionsWrapper clone() const {
    OrtSessionOptions* p = OrtCloneSessionOptions(value.get());
    return SessionOptionsWrapper(env_, p);
  }

  OrtSession* OrtCreateSession(_In_ const ORTCHAR_T* model_path) {
    OrtSession* ret = nullptr;
    ORT_THROW_ON_ERROR(::OrtCreateSession(env_, model_path, value.get(), &ret));
    return ret;
  }
};

inline OrtValue* OrtCreateTensorAsOrtValue(_Inout_ OrtAllocator* env, const std::vector<int64_t>& shape, ONNXTensorElementDataType type) {
  OrtValue* ret;
  ORT_THROW_ON_ERROR(::OrtCreateTensorAsOrtValue(env, shape.data(), shape.size(), type, &ret));
  return ret;
}

inline OrtValue* OrtCreateTensorWithDataAsOrtValue(_In_ const OrtAllocatorInfo* info, _In_ void* p_data, size_t p_data_len, const std::vector<int64_t>& shape, ONNXTensorElementDataType type) {
  OrtValue* ret;
  ORT_THROW_ON_ERROR(::OrtCreateTensorWithDataAsOrtValue(info, p_data, p_data_len, shape.data(), shape.size(), type, &ret));
  return ret;
}

inline std::vector<int64_t> GetTensorShape(const OrtTensorTypeAndShapeInfo* info) {
  size_t dims = OrtGetNumOfDimensions(info);
  std::vector<int64_t> ret(dims);
  OrtGetDimensions(info, ret.data(), ret.size());
  return ret;
}

}  // namespace onnxruntime
#endif

namespace Ort {
struct CustomOpApi {
  CustomOpApi(const OrtCustomOpApi& api) : api_(api) {}

  template <typename T>
  T KernelInfoGetAttribute(_In_ const OrtKernelInfo* info, _In_ const char* name);

  OrtTensorTypeAndShapeInfo* GetTensorShapeAndType(_In_ const OrtValue* value) {
    OrtTensorTypeAndShapeInfo* out;
    ORT_THROW_ON_ERROR(api_.GetTensorShapeAndType(value, &out));
    return out;
  }

  int64_t GetTensorShapeElementCount(_In_ const OrtTensorTypeAndShapeInfo* info) {
    return api_.GetTensorShapeElementCount(info);
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

  void ReleaseTensorTypeAndShapeInfo(OrtTensorTypeAndShapeInfo* input) {
    api_.ReleaseTensorTypeAndShapeInfo(input);
  }

  const OrtValue* KernelContext_GetInput(const OrtKernelContext* context, _In_ size_t index) {
    return api_.KernelContext_GetInput(context, index);
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

    OrtCustomOp::KernelGetOutputShape = [](void* op_kernel, OrtKernelContext* context, size_t output_index, OrtTensorTypeAndShapeInfo* output) { static_cast<TKernel*>(op_kernel)->GetOutputShape(context, output_index, output); };
    OrtCustomOp::KernelCompute = [](void* op_kernel, OrtKernelContext* context) { static_cast<TKernel*>(op_kernel)->Compute(context); };
    OrtCustomOp::KernelDestroy = [](void* op_kernel) { delete static_cast<TKernel*>(op_kernel); };
  }
};

}  // namespace Ort

#undef ORT_REDIRECT_SIMPLE_FUNCTION_CALL
