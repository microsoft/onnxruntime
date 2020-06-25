// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Summary: The Ort C++ API is a header only wrapper around the Ort C API.
//
// The C++ API simplifies usage by returning values directly instead of error codes, throwing exceptions on errors
// and automatically releasing resources in the destructors.
//
// Each of the C++ wrapper classes holds only a pointer to the C internal object. Treat them like smart pointers.
// To create an empty object, pass 'nullptr' to the constructor (for example, Env e{nullptr};).
//
// Only move assignment between objects is allowed, there are no copy constructors. Some objects have explicit 'Clone'
// methods for this purpose.

#pragma once
#include "onnxruntime_c_api.h"
#include <cstddef>
#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#include <type_traits>

namespace Ort {

// All C++ methods that can fail will throw an exception of this type
struct Exception : std::exception {
  Exception(std::string&& string, OrtErrorCode code) : message_{std::move(string)}, code_{code} {}

  OrtErrorCode GetOrtErrorCode() const { return code_; }
  const char* what() const noexcept override { return message_.c_str(); }

 private:
  std::string message_;
  OrtErrorCode code_;
};

// This is used internally by the C++ API. This class holds the global variable that points to the OrtApi, it's in a template so that we can define a global variable in a header and make
// it transparent to the users of the API.
template <typename T>
struct Global {
  static const OrtApi& api_;
};

#ifdef EXCLUDE_REFERENCE_TO_ORT_DLL
OrtApi stub_api;
template <typename T>
const OrtApi& Global<T>::api_ = stub_api;
#else
template <typename T>
const OrtApi& Global<T>::api_ = *OrtGetApiBase()->GetApi(ORT_API_VERSION);
#endif

// This returns a reference to the OrtApi interface in use, in case someone wants to use the C API functions
inline const OrtApi& GetApi() { return Global<void>::api_; }

// This is a C++ wrapper for GetAvailableProviders() C API and returns
// a vector of strings representing the available execution providers.
std::vector<std::string> GetAvailableProviders();

// This is used internally by the C++ API. This macro is to make it easy to generate overloaded methods for all of the various OrtRelease* functions for every Ort* type
// This can't be done in the C API since C doesn't have function overloading.
#define ORT_DEFINE_RELEASE(NAME) \
  inline void OrtRelease(Ort##NAME* ptr) { Global<void>::api_.Release##NAME(ptr); }

ORT_DEFINE_RELEASE(MemoryInfo);
ORT_DEFINE_RELEASE(CustomOpDomain);
ORT_DEFINE_RELEASE(Env);
ORT_DEFINE_RELEASE(RunOptions);
ORT_DEFINE_RELEASE(Session);
ORT_DEFINE_RELEASE(SessionOptions);
ORT_DEFINE_RELEASE(TensorTypeAndShapeInfo);
ORT_DEFINE_RELEASE(TypeInfo);
ORT_DEFINE_RELEASE(Value);
ORT_DEFINE_RELEASE(ModelMetadata);
ORT_DEFINE_RELEASE(ThreadingOptions);

// This is used internally by the C++ API. This is the common base class used by the wrapper objects.
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
  Base& operator=(const Base&) = delete;
  Base(Base&& v) noexcept : p_{v.p_} { v.p_ = nullptr; }
  void operator=(Base&& v) noexcept {
    OrtRelease(p_);
    p_ = v.p_;
    v.p_ = nullptr;
  }

  T* p_{};

  template <typename>
  friend struct Unowned;  // This friend line is needed to keep the centos C++ compiler from giving an error
};

template <typename T>
struct Unowned : T {
  Unowned(decltype(T::p_) p) : T{p} {}
  Unowned(Unowned&& v) : T{v.p_} {}
  ~Unowned() { this->p_ = nullptr; }
};

struct AllocatorWithDefaultOptions;
struct MemoryInfo;
struct Env;
struct TypeInfo;
struct Value;
struct ModelMetadata;

struct Env : Base<OrtEnv> {
  Env(std::nullptr_t) {}
  Env(OrtLoggingLevel default_logging_level = ORT_LOGGING_LEVEL_WARNING, _In_ const char* logid = "");
  Env(const OrtThreadingOptions* tp_options, OrtLoggingLevel default_logging_level = ORT_LOGGING_LEVEL_WARNING, _In_ const char* logid = "");
  Env(OrtLoggingLevel default_logging_level, const char* logid, OrtLoggingFunction logging_function, void* logger_param);
  explicit Env(OrtEnv* p) : Base<OrtEnv>{p} {}

  Env& EnableTelemetryEvents();
  Env& DisableTelemetryEvents();

  static const OrtApi* s_api;
};

struct CustomOpDomain : Base<OrtCustomOpDomain> {
  explicit CustomOpDomain(std::nullptr_t) {}
  explicit CustomOpDomain(const char* domain);

  void Add(OrtCustomOp* op);
};

struct RunOptions : Base<OrtRunOptions> {
  RunOptions(std::nullptr_t) {}
  RunOptions();

  RunOptions& SetRunLogVerbosityLevel(int);
  int GetRunLogVerbosityLevel() const;

  RunOptions& SetRunLogSeverityLevel(int);
  int GetRunLogSeverityLevel() const;

  RunOptions& SetRunTag(const char* run_tag);
  const char* GetRunTag() const;

  // terminate ALL currently executing Session::Run calls that were made using this RunOptions instance
  RunOptions& SetTerminate();
  // unset the terminate flag so this RunOptions instance can be used in a new Session::Run call
  RunOptions& UnsetTerminate();
};

struct SessionOptions : Base<OrtSessionOptions> {
  explicit SessionOptions(std::nullptr_t) {}
  SessionOptions();
  explicit SessionOptions(OrtSessionOptions* p) : Base<OrtSessionOptions>{p} {}

  SessionOptions Clone() const;

  SessionOptions& SetIntraOpNumThreads(int intra_op_num_threads);
  SessionOptions& SetInterOpNumThreads(int inter_op_num_threads);
  SessionOptions& SetGraphOptimizationLevel(GraphOptimizationLevel graph_optimization_level);

  SessionOptions& EnableCpuMemArena();
  SessionOptions& DisableCpuMemArena();

  SessionOptions& SetOptimizedModelFilePath(const ORTCHAR_T* optimized_model_file);

  SessionOptions& EnableProfiling(const ORTCHAR_T* profile_file_prefix);
  SessionOptions& DisableProfiling();

  SessionOptions& EnableMemPattern();
  SessionOptions& DisableMemPattern();

  SessionOptions& SetExecutionMode(ExecutionMode execution_mode);

  SessionOptions& SetLogId(const char* logid);

  SessionOptions& Add(OrtCustomOpDomain* custom_op_domain);

  SessionOptions& DisablePerSessionThreads();
};

struct ModelMetadata : Base<OrtModelMetadata> {
  explicit ModelMetadata(std::nullptr_t) {}
  explicit ModelMetadata(OrtModelMetadata* p) : Base<OrtModelMetadata>{p} {}

  char* GetProducerName(OrtAllocator* allocator) const;
  char* GetGraphName(OrtAllocator* allocator) const;
  char* GetDomain(OrtAllocator* allocator) const;
  char* GetDescription(OrtAllocator* allocator) const;
  char** GetCustomMetadataMapKeys(OrtAllocator* allocator, _Out_ int64_t& num_keys) const;
  char* LookupCustomMetadataMap(const char* key, OrtAllocator* allocator) const;
  int64_t GetVersion() const;
};

struct Session : Base<OrtSession> {
  explicit Session(std::nullptr_t) {}
  Session(Env& env, const ORTCHAR_T* model_path, const SessionOptions& options);
  Session(Env& env, const void* model_data, size_t model_data_length, const SessionOptions& options);

  // Run that will allocate the output values
  std::vector<Value> Run(const RunOptions& run_options, const char* const* input_names, const Value* input_values, size_t input_count,
                         const char* const* output_names, size_t output_count);
  // Run for when there is a list of prealloated outputs
  void Run(const RunOptions& run_options, const char* const* input_names, const Value* input_values, size_t input_count,
           const char* const* output_names, Value* output_values, size_t output_count);

  size_t GetInputCount() const;
  size_t GetOutputCount() const;
  size_t GetOverridableInitializerCount() const;

  char* GetInputName(size_t index, OrtAllocator* allocator) const;
  char* GetOutputName(size_t index, OrtAllocator* allocator) const;
  char* GetOverridableInitializerName(size_t index, OrtAllocator* allocator) const;
  char* EndProfiling(OrtAllocator* allocator) const;
  ModelMetadata GetModelMetadata() const;

  TypeInfo GetInputTypeInfo(size_t index) const;
  TypeInfo GetOutputTypeInfo(size_t index) const;
  TypeInfo GetOverridableInitializerTypeInfo(size_t index) const;
};

struct TensorTypeAndShapeInfo : Base<OrtTensorTypeAndShapeInfo> {
  explicit TensorTypeAndShapeInfo(std::nullptr_t) {}
  explicit TensorTypeAndShapeInfo(OrtTensorTypeAndShapeInfo* p) : Base<OrtTensorTypeAndShapeInfo>{p} {}

  ONNXTensorElementDataType GetElementType() const;
  size_t GetElementCount() const;

  size_t GetDimensionsCount() const;
  void GetDimensions(int64_t* values, size_t values_count) const;
  void GetSymbolicDimensions(const char** values, size_t values_count) const;

  std::vector<int64_t> GetShape() const;
};

struct TypeInfo : Base<OrtTypeInfo> {
  explicit TypeInfo(std::nullptr_t) {}
  explicit TypeInfo(OrtTypeInfo* p) : Base<OrtTypeInfo>{p} {}

  Unowned<TensorTypeAndShapeInfo> GetTensorTypeAndShapeInfo() const;
  ONNXType GetONNXType() const;
};

struct Value : Base<OrtValue> {
  template <typename T>
  static Value CreateTensor(const OrtMemoryInfo* info, T* p_data, size_t p_data_element_count, const int64_t* shape, size_t shape_len);
  static Value CreateTensor(const OrtMemoryInfo* info, void* p_data, size_t p_data_byte_count, const int64_t* shape, size_t shape_len,
                            ONNXTensorElementDataType type);
  template <typename T>
  static Value CreateTensor(OrtAllocator* allocator, const int64_t* shape, size_t shape_len);
  static Value CreateTensor(OrtAllocator* allocator, const int64_t* shape, size_t shape_len, ONNXTensorElementDataType type);

  static Value CreateMap(Value& keys, Value& values);
  static Value CreateSequence(std::vector<Value>& values);

  template <typename T>
  static Value CreateOpaque(const char* domain, const char* type_name, const T&);

  template <typename T>
  void GetOpaqueData(const char* domain, const char* type_name, T&);

  explicit Value(std::nullptr_t) {}
  explicit Value(OrtValue* p) : Base<OrtValue>{p} {}
  Value(Value&&) = default;
  Value& operator=(Value&&) = default;

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

struct AllocatorWithDefaultOptions {
  AllocatorWithDefaultOptions();

  operator OrtAllocator*() { return p_; }
  operator const OrtAllocator*() const { return p_; }

  void* Alloc(size_t size);
  void Free(void* p);

  const OrtMemoryInfo* GetInfo() const;

 private:
  OrtAllocator* p_{};
};

struct MemoryInfo : Base<OrtMemoryInfo> {
  static MemoryInfo CreateCpu(OrtAllocatorType type, OrtMemType mem_type1);

  explicit MemoryInfo(std::nullptr_t) {}
  MemoryInfo(const char* name, OrtAllocatorType type, int id, OrtMemType mem_type);

  explicit MemoryInfo(OrtMemoryInfo* p) : Base<OrtMemoryInfo>{p} {}
};

//
// Custom OPs (only needed to implement custom OPs)
//

struct CustomOpApi {
  CustomOpApi(const OrtApi& api) : api_(api) {}

  template <typename T>  // T is only implemented for float, int64_t, and string
  T KernelInfoGetAttribute(_In_ const OrtKernelInfo* info, _In_ const char* name);

  OrtTensorTypeAndShapeInfo* GetTensorTypeAndShape(_In_ const OrtValue* value);
  size_t GetTensorShapeElementCount(_In_ const OrtTensorTypeAndShapeInfo* info);
  ONNXTensorElementDataType GetTensorElementType(const OrtTensorTypeAndShapeInfo* info);
  size_t GetDimensionsCount(_In_ const OrtTensorTypeAndShapeInfo* info);
  void GetDimensions(_In_ const OrtTensorTypeAndShapeInfo* info, _Out_ int64_t* dim_values, size_t dim_values_length);
  void SetDimensions(OrtTensorTypeAndShapeInfo* info, _In_ const int64_t* dim_values, size_t dim_count);

  template <typename T>
  T* GetTensorMutableData(_Inout_ OrtValue* value);
  template <typename T>
  const T* GetTensorData(_Inout_ const OrtValue* value);

  std::vector<int64_t> GetTensorShape(const OrtTensorTypeAndShapeInfo* info);
  void ReleaseTensorTypeAndShapeInfo(OrtTensorTypeAndShapeInfo* input);
  size_t KernelContext_GetInputCount(const OrtKernelContext* context);
  const OrtValue* KernelContext_GetInput(const OrtKernelContext* context, _In_ size_t index);
  size_t KernelContext_GetOutputCount(const OrtKernelContext* context);
  OrtValue* KernelContext_GetOutput(OrtKernelContext* context, _In_ size_t index, _In_ const int64_t* dim_values, size_t dim_count);

  void ThrowOnError(OrtStatus* result);

 private:
  const OrtApi& api_;
};

template <typename TOp, typename TKernel>
struct CustomOpBase : OrtCustomOp {
  CustomOpBase() {
    OrtCustomOp::version = ORT_API_VERSION;
    OrtCustomOp::CreateKernel = [](OrtCustomOp* this_, const OrtApi* api, const OrtKernelInfo* info) { return static_cast<TOp*>(this_)->CreateKernel(*api, info); };
    OrtCustomOp::GetName = [](OrtCustomOp* this_) { return static_cast<TOp*>(this_)->GetName(); };

    OrtCustomOp::GetExecutionProviderType = [](OrtCustomOp* this_) { return static_cast<TOp*>(this_)->GetExecutionProviderType(); };

    OrtCustomOp::GetInputTypeCount = [](OrtCustomOp* this_) { return static_cast<TOp*>(this_)->GetInputTypeCount(); };
    OrtCustomOp::GetInputType = [](OrtCustomOp* this_, size_t index) { return static_cast<TOp*>(this_)->GetInputType(index); };

    OrtCustomOp::GetOutputTypeCount = [](OrtCustomOp* this_) { return static_cast<TOp*>(this_)->GetOutputTypeCount(); };
    OrtCustomOp::GetOutputType = [](OrtCustomOp* this_, size_t index) { return static_cast<TOp*>(this_)->GetOutputType(index); };

    OrtCustomOp::KernelCompute = [](void* op_kernel, OrtKernelContext* context) { static_cast<TKernel*>(op_kernel)->Compute(context); };
    OrtCustomOp::KernelDestroy = [](void* op_kernel) { delete static_cast<TKernel*>(op_kernel); };
  }

  // Default implementation of GetExecutionProviderType that returns nullptr to default to the CPU provider
  const char* GetExecutionProviderType() const { return nullptr; }
};

}  // namespace Ort

#include "onnxruntime_cxx_inline.h"
