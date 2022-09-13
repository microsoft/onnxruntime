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
#include <unordered_map>
#include <utility>
#include <type_traits>

#ifdef ORT_NO_EXCEPTIONS
#include <iostream>
#endif

/** \brief All C++ Onnxruntime APIs are defined inside this namespace
 *
 */
namespace Ort {

/** \brief All C++ methods that can fail will throw an exception of this type
 *
 * If <tt>ORT_NO_EXCEPTIONS</tt> is defined, then any error will result in a call to abort()
 */
struct Exception : std::exception {
  Exception(std::string&& string, OrtErrorCode code) : message_{std::move(string)}, code_{code} {}

  OrtErrorCode GetOrtErrorCode() const { return code_; }
  const char* what() const noexcept override { return message_.c_str(); }

 private:
  std::string message_;
  OrtErrorCode code_;
};

#ifdef ORT_NO_EXCEPTIONS
// The #ifndef is for the very special case where the user of this library wants to define their own way of handling errors.
// NOTE: This header expects control flow to not continue after calling ORT_CXX_API_THROW
#ifndef ORT_CXX_API_THROW
#define ORT_CXX_API_THROW(string, code)       \
  do {                                        \
    std::cerr << Ort::Exception(string, code) \
                     .what()                  \
              << std::endl;                   \
    abort();                                  \
  } while (false)
#endif
#else
#define ORT_CXX_API_THROW(string, code) \
  throw Ort::Exception(string, code)
#endif

// This is used internally by the C++ API. This class holds the global variable that points to the OrtApi, it's in a template so that we can define a global variable in a header and make
// it transparent to the users of the API.
template <typename T>
struct Global {
  static const OrtApi* api_;
};

// If macro ORT_API_MANUAL_INIT is defined, no static initialization will be performed. Instead, user must call InitApi() before using it.
template <typename T>
#ifdef ORT_API_MANUAL_INIT
const OrtApi* Global<T>::api_{};
inline void InitApi() { Global<void>::api_ = OrtGetApiBase()->GetApi(ORT_API_VERSION); }
#else
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
// "Global initializer calls a non-constexpr function." Therefore you can't use ORT APIs in the other global initializers.
// Please define ORT_API_MANUAL_INIT if it conerns you.
#pragma warning(disable : 26426)
#endif
const OrtApi* Global<T>::api_ = OrtGetApiBase()->GetApi(ORT_API_VERSION);
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif
#endif

/// This returns a reference to the OrtApi interface in use
inline const OrtApi& GetApi() { return *Global<void>::api_; }

/// This is a C++ wrapper for OrtApi::GetAvailableProviders() and returns a vector of strings representing the available execution providers.
std::vector<std::string> GetAvailableProviders();

// This is used internally by the C++ API. This macro is to make it easy to generate overloaded methods for all of the various OrtRelease* functions for every Ort* type
// This can't be done in the C API since C doesn't have function overloading.
#define ORT_DEFINE_RELEASE(NAME) \
  inline void OrtRelease(Ort##NAME* ptr) { GetApi().Release##NAME(ptr); }

ORT_DEFINE_RELEASE(Allocator);
ORT_DEFINE_RELEASE(MemoryInfo);
ORT_DEFINE_RELEASE(CustomOpDomain);
ORT_DEFINE_RELEASE(Env);
ORT_DEFINE_RELEASE(RunOptions);
ORT_DEFINE_RELEASE(Session);
ORT_DEFINE_RELEASE(SessionOptions);
ORT_DEFINE_RELEASE(TensorTypeAndShapeInfo);
ORT_DEFINE_RELEASE(SequenceTypeInfo);
ORT_DEFINE_RELEASE(MapTypeInfo);
ORT_DEFINE_RELEASE(TypeInfo);
ORT_DEFINE_RELEASE(Value);
ORT_DEFINE_RELEASE(ModelMetadata);
ORT_DEFINE_RELEASE(ThreadingOptions);
ORT_DEFINE_RELEASE(IoBinding);
ORT_DEFINE_RELEASE(ArenaCfg);
ORT_DEFINE_RELEASE(Status);
ORT_DEFINE_RELEASE(OpAttr);
ORT_DEFINE_RELEASE(Op);
ORT_DEFINE_RELEASE(KernelInfo);

#undef ORT_DEFINE_RELEASE

/** \brief IEEE 754 half-precision floating point data type
 * \details It is necessary for type dispatching to make use of C++ API
 * The type is implicitly convertible to/from uint16_t.
 * The size of the structure should align with uint16_t and one can freely cast
 * uint16_t buffers to/from Ort::Float16_t to feed and retrieve data.
 *
 * Generally, you can feed any of your types as float16/blfoat16 data to create a tensor
 * on top of it, providing it can form a continuous buffer with 16-bit elements with no padding.
 * And you can also feed a array of uint16_t elements directly. For example,
 *
 * \code{.unparsed}
 * uint16_t values[] = { 15360, 16384, 16896, 17408, 17664};
 * constexpr size_t values_length = sizeof(values) / sizeof(values[0]);
 * std::vector<int64_t> dims = {values_length};  // one dimensional example
 * Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
 * // Note we are passing bytes count in this api, not number of elements -> sizeof(values)
 * auto float16_tensor = Ort::Value::CreateTensor(info, values, sizeof(values),
 *                                                dims.data(), dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
 * \endcode
 *
 * Here is another example, a little bit more elaborate. Let's assume that you use your own float16 type and you want to use
 * a templated version of the API above so the type is automatically set based on your type. You will need to supply an extra
 * template specialization.
 *
 * \code{.unparsed}
 * namespace yours { struct half {}; } // assume this is your type, define this:
 * namespace Ort {
 * template<>
 * struct TypeToTensorType<yours::half> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16; };
 * } //namespace Ort
 *
 * std::vector<yours::half> values;
 * std::vector<int64_t> dims = {values.size()}; // one dimensional example
 * Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
 * // Here we are passing element count -> values.size()
 * auto float16_tensor = Ort::Value::CreateTensor<yours::half>(info, values.data(), values.size(), dims.data(), dims.size());
 *
 *  \endcode
 */
struct Float16_t {
  uint16_t value;
  constexpr Float16_t() noexcept : value(0) {}
  constexpr Float16_t(uint16_t v) noexcept : value(v) {}
  constexpr operator uint16_t() const noexcept { return value; }
  constexpr bool operator==(const Float16_t& rhs) const noexcept { return value == rhs.value; };
  constexpr bool operator!=(const Float16_t& rhs) const noexcept { return value != rhs.value; };
};

static_assert(sizeof(Float16_t) == sizeof(uint16_t), "Sizes must match");

/** \brief bfloat16 (Brain Floating Point) data type
 * \details It is necessary for type dispatching to make use of C++ API
 * The type is implicitly convertible to/from uint16_t.
 * The size of the structure should align with uint16_t and one can freely cast
 * uint16_t buffers to/from Ort::BFloat16_t to feed and retrieve data.
 *
 * See also code examples for Float16_t above.
 */
struct BFloat16_t {
  uint16_t value;
  constexpr BFloat16_t() noexcept : value(0) {}
  constexpr BFloat16_t(uint16_t v) noexcept : value(v) {}
  constexpr operator uint16_t() const noexcept { return value; }
  constexpr bool operator==(const BFloat16_t& rhs) const noexcept { return value == rhs.value; };
  constexpr bool operator!=(const BFloat16_t& rhs) const noexcept { return value != rhs.value; };
};

static_assert(sizeof(BFloat16_t) == sizeof(uint16_t), "Sizes must match");

/** \brief Used internally by the C++ API. C++ wrapper types inherit from this
 *
 * This is a zero cost abstraction to wrap the C API objects and delete them on destruction.
 * There is a secondary class 'Unowned<T>' that is used to prevent deletion on destruction (Used for return types that are
 * not owned by the caller)
 *
 */
template <typename T>
struct Base {
  using contained_type = T;

  Base() = default;
  Base(T* p) : p_{p} {
    if (!p)
      ORT_CXX_API_THROW("Allocation failure", ORT_FAIL);
  }
  ~Base() { OrtRelease(p_); }

  operator T*() { return p_; }
  operator const T*() const { return p_; }

  /// \brief Releases ownership of the contained pointer
  T* release() {
    T* p = p_;
    p_ = nullptr;
    return p;
  }

  Base(const Base&) = delete;
  Base& operator=(const Base&) = delete;

 protected:
  Base(Base&& v) noexcept : p_{v.p_} { v.p_ = nullptr; }
  void operator=(Base&& v) noexcept {
    OrtRelease(p_);
    p_ = v.release();
  }

  T* p_{};

  template <typename>
  friend struct Unowned;  // This friend line is needed to keep the centos C++ compiler from giving an error
};

/** \brief Wraps an object that inherits from Ort::Base and stops it from deleting the contained pointer on destruction
 *  This is useful when you want to wrap a native C object pointer so you can use it with the C++ API, but you do not
 *  want the C++ API to delete the pointer on destruction. Wrap it to Ort::Unowned<Ort::XXX> for non-const pointers and
 *  Ort::Unowned<const Ort::XXX> const pointers.
 * 
 * This has the effect of making it not own the memory held by Ort::Base.
 */
template <typename T>
struct Unowned : T {
  using Type = typename T::contained_type;
  Unowned(Type* p) : T{p} {}
  Unowned(const Type* p) : T{const_cast<Type*>(p)} {}
  Unowned(Unowned&& v) : T{v.p_} { v.p_ = nullptr; }
  Unowned& operator=(Unowned&&) = delete;
  ~Unowned() { this->release(); }
  template <typename R = T>
  std::enable_if_t<std::is_const<R>::value, R>& operator*() const { return *this; }
  template <typename R = T>
  std::enable_if_t<!std::is_const<R>::value, R>& operator*() { return *this; }
};

struct AllocatorWithDefaultOptions;
struct MemoryInfo;
struct Env;
struct TypeInfo;
struct Value;
struct ModelMetadata;

namespace detail {
// Light functor to release memory with OrtAllocator
struct AllocatedFree {
  OrtAllocator* allocator_;
  explicit AllocatedFree(OrtAllocator* allocator)
      : allocator_(allocator) {}
  void operator()(void* ptr) const {
    if (ptr) allocator_->Free(allocator_, ptr);
  }
};
}  // namespace detail

/** \brief unique_ptr typedef used to own strings allocated by OrtAllocators
 *  and release them at the end of the scope. The lifespan of the given allocator
 *  must eclipse the lifespan of AllocatedStringPtr instance
 */
using AllocatedStringPtr = std::unique_ptr<char, detail::AllocatedFree>;

/** \brief The Status that holds ownership of OrtStatus received from C API
 *  Use it to safely destroy OrtStatus* returned from the C API. Use appropriate
 *  constructors to construct an instance of a Status object from exceptions.
 */
struct Status : Base<OrtStatus> {
  explicit Status(std::nullptr_t) {}      ///< Create an empty object, must be assigned a valid one to be used
  explicit Status(OrtStatus* status);     ///< Takes ownership of OrtStatus instance returned from the C API. Must be non-null
  explicit Status(const std::exception&); ///< Create status instance out of exception
  std::string GetErrorMessage() const;
  OrtErrorCode GetErrorCode() const;
};

/** \brief The Env (Environment)
 *
 * The Env holds the logging state used by all other objects.
 * <b>Note:</b> One Env must be created before using any other Onnxruntime functionality
 */
struct Env : Base<OrtEnv> {
  explicit Env(std::nullptr_t) {}  ///< Create an empty Env object, must be assigned a valid one to be used

  /// \brief Wraps OrtApi::CreateEnv
  Env(OrtLoggingLevel logging_level = ORT_LOGGING_LEVEL_WARNING, _In_ const char* logid = "");

  /// \brief Wraps OrtApi::CreateEnvWithCustomLogger
  Env(OrtLoggingLevel logging_level, const char* logid, OrtLoggingFunction logging_function, void* logger_param);

  /// \brief Wraps OrtApi::CreateEnvWithGlobalThreadPools
  Env(const OrtThreadingOptions* tp_options, OrtLoggingLevel logging_level = ORT_LOGGING_LEVEL_WARNING, _In_ const char* logid = "");

  /// \brief Wraps OrtApi::CreateEnvWithCustomLoggerAndGlobalThreadPools
  Env(const OrtThreadingOptions* tp_options, OrtLoggingFunction logging_function, void* logger_param,
      OrtLoggingLevel logging_level = ORT_LOGGING_LEVEL_WARNING, _In_ const char* logid = "");

  /// \brief C Interop Helper
  explicit Env(OrtEnv* p) : Base<OrtEnv>{p} {}

  Env& EnableTelemetryEvents();   ///< Wraps OrtApi::EnableTelemetryEvents
  Env& DisableTelemetryEvents();  ///< Wraps OrtApi::DisableTelemetryEvents

  Env& CreateAndRegisterAllocator(const OrtMemoryInfo* mem_info, const OrtArenaCfg* arena_cfg);  ///< Wraps OrtApi::CreateAndRegisterAllocator
};

/** \brief Custom Op Domain
 *
 */
struct CustomOpDomain : Base<OrtCustomOpDomain> {
  explicit CustomOpDomain(std::nullptr_t) {}  ///< Create an empty CustomOpDomain object, must be assigned a valid one to be used

  /// \brief Wraps OrtApi::CreateCustomOpDomain
  explicit CustomOpDomain(const char* domain);

  // This does not take ownership of the op, simply registers it.
  void Add(const OrtCustomOp* op);  ///< Wraps CustomOpDomain_Add
};

struct RunOptions : Base<OrtRunOptions> {
  explicit RunOptions(std::nullptr_t) {}  ///< Create an empty RunOptions object, must be assigned a valid one to be used
  RunOptions();                           ///< Wraps OrtApi::CreateRunOptions

  RunOptions& SetRunLogVerbosityLevel(int);  ///< Wraps OrtApi::RunOptionsSetRunLogVerbosityLevel
  int GetRunLogVerbosityLevel() const;       ///< Wraps OrtApi::RunOptionsGetRunLogVerbosityLevel

  RunOptions& SetRunLogSeverityLevel(int);  ///< Wraps OrtApi::RunOptionsSetRunLogSeverityLevel
  int GetRunLogSeverityLevel() const;       ///< Wraps OrtApi::RunOptionsGetRunLogSeverityLevel

  RunOptions& SetRunTag(const char* run_tag);  ///< wraps OrtApi::RunOptionsSetRunTag
  const char* GetRunTag() const;               ///< Wraps OrtApi::RunOptionsGetRunTag

  RunOptions& AddConfigEntry(const char* config_key, const char* config_value);  ///< Wraps OrtApi::AddRunConfigEntry

  /** \brief Terminates all currently executing Session::Run calls that were made using this RunOptions instance
   *
   * If a currently executing session needs to be force terminated, this can be called from another thread to force it to fail with an error
   * Wraps OrtApi::RunOptionsSetTerminate
   */
  RunOptions& SetTerminate();

  /** \brief Clears the terminate flag so this RunOptions instance can be used in a new Session::Run call without it instantly terminating
   *
   * Wraps OrtApi::RunOptionsUnsetTerminate
   */
  RunOptions& UnsetTerminate();
};

/** \brief Options object used when creating a new Session object
 *
 * Wraps ::OrtSessionOptions object and methods
 */
struct SessionOptions : Base<OrtSessionOptions> {
  explicit SessionOptions(std::nullptr_t) {}                                     ///< Create an empty SessionOptions object, must be assigned a valid one to be used
  SessionOptions();                                                              ///< Wraps OrtApi::CreateSessionOptions
  explicit SessionOptions(OrtSessionOptions* p) : Base<OrtSessionOptions>{p} {}  ///< Used for interop with the C API

  SessionOptions Clone() const;  ///< Creates and returns a copy of this SessionOptions object. Wraps OrtApi::CloneSessionOptions

  SessionOptions& SetIntraOpNumThreads(int intra_op_num_threads);                              ///< Wraps OrtApi::SetIntraOpNumThreads
  SessionOptions& SetInterOpNumThreads(int inter_op_num_threads);                              ///< Wraps OrtApi::SetInterOpNumThreads
  SessionOptions& SetGraphOptimizationLevel(GraphOptimizationLevel graph_optimization_level);  ///< Wraps OrtApi::SetSessionGraphOptimizationLevel

  SessionOptions& EnableCpuMemArena();   ///< Wraps OrtApi::EnableCpuMemArena
  SessionOptions& DisableCpuMemArena();  ///< Wraps OrtApi::DisableCpuMemArena

  SessionOptions& SetOptimizedModelFilePath(const ORTCHAR_T* optimized_model_file);  ///< Wraps OrtApi::SetOptimizedModelFilePath

  SessionOptions& EnableProfiling(const ORTCHAR_T* profile_file_prefix);  ///< Wraps OrtApi::EnableProfiling
  SessionOptions& DisableProfiling();                                     ///< Wraps OrtApi::DisableProfiling

  SessionOptions& EnableOrtCustomOps();  ///< Wraps OrtApi::EnableOrtCustomOps

  SessionOptions& EnableMemPattern();   ///< Wraps OrtApi::EnableMemPattern
  SessionOptions& DisableMemPattern();  ///< Wraps OrtApi::DisableMemPattern

  SessionOptions& SetExecutionMode(ExecutionMode execution_mode);  ///< Wraps OrtApi::SetSessionExecutionMode

  SessionOptions& SetLogId(const char* logid);     ///< Wraps OrtApi::SetSessionLogId
  SessionOptions& SetLogSeverityLevel(int level);  ///< Wraps OrtApi::SetSessionLogSeverityLevel

  SessionOptions& Add(OrtCustomOpDomain* custom_op_domain);  ///< Wraps OrtApi::AddCustomOpDomain

  SessionOptions& DisablePerSessionThreads();  ///< Wraps OrtApi::DisablePerSessionThreads

  SessionOptions& AddConfigEntry(const char* config_key, const char* config_value);                                      ///< Wraps OrtApi::AddSessionConfigEntry
  SessionOptions& AddInitializer(const char* name, const OrtValue* ort_val);                                             ///< Wraps OrtApi::AddInitializer
  SessionOptions& AddExternalInitializers(const std::vector<std::string>& names, const std::vector<Value>& ort_values);  ///< Wraps OrtApi::AddExternalInitializers

  SessionOptions& AppendExecutionProvider_CUDA(const OrtCUDAProviderOptions& provider_options);               ///< Wraps OrtApi::SessionOptionsAppendExecutionProvider_CUDA
  SessionOptions& AppendExecutionProvider_CUDA_V2(const OrtCUDAProviderOptionsV2& provider_options);          ///< Wraps OrtApi::SessionOptionsAppendExecutionProvider_CUDA_V2
  SessionOptions& AppendExecutionProvider_ROCM(const OrtROCMProviderOptions& provider_options);               ///< Wraps OrtApi::SessionOptionsAppendExecutionProvider_ROCM
  SessionOptions& AppendExecutionProvider_OpenVINO(const OrtOpenVINOProviderOptions& provider_options);       ///< Wraps OrtApi::SessionOptionsAppendExecutionProvider_OpenVINO
  SessionOptions& AppendExecutionProvider_TensorRT(const OrtTensorRTProviderOptions& provider_options);       ///< Wraps OrtApi::SessionOptionsAppendExecutionProvider_TensorRT
  SessionOptions& AppendExecutionProvider_TensorRT_V2(const OrtTensorRTProviderOptionsV2& provider_options);  ///< Wraps OrtApi::SessionOptionsAppendExecutionProvider_TensorRT
  SessionOptions& AppendExecutionProvider_MIGraphX(const OrtMIGraphXProviderOptions& provider_options);       ///< Wraps OrtApi::SessionOptionsAppendExecutionProvider_MIGraphX
  /// Wraps OrtApi::SessionOptionsAppendExecutionProvider. Currently supports SNPE and XNNPACK.
  SessionOptions& AppendExecutionProvider(const std::string& provider_name,
                                          const std::unordered_map<std::string, std::string>& provider_options = {});

  SessionOptions& SetCustomCreateThreadFn(OrtCustomCreateThreadFn ort_custom_create_thread_fn);  ///< Wraps OrtApi::SessionOptionsSetCustomCreateThreadFn
  SessionOptions& SetCustomThreadCreationOptions(void* ort_custom_thread_creation_options);      ///< Wraps OrtApi::SessionOptionsSetCustomThreadCreationOptions
  SessionOptions& SetCustomJoinThreadFn(OrtCustomJoinThreadFn ort_custom_join_thread_fn);        ///< Wraps OrtApi::SessionOptionsSetCustomJoinThreadFn
};

/** \brief Wrapper around ::OrtModelMetadata
 *
 */
struct ModelMetadata : Base<OrtModelMetadata> {
  explicit ModelMetadata(std::nullptr_t) {}                                   ///< Create an empty ModelMetadata object, must be assigned a valid one to be used
  explicit ModelMetadata(OrtModelMetadata* p) : Base<OrtModelMetadata>{p} {}  ///< Used for interop with the C API

  /** \brief Returns a copy of the producer name.
   *
   * \param allocator to allocate memory for the copy of the name returned
   * \return a instance of smart pointer that would deallocate the buffer when out of scope.
   *  The OrtAllocator instances must be valid at the point of memory release.
   */
  AllocatedStringPtr GetProducerNameAllocated(OrtAllocator* allocator) const;  ///< Wraps OrtApi::ModelMetadataGetProducerName

  /** \brief Returns a copy of the graph name.
   *
   * \param allocator to allocate memory for the copy of the name returned
   * \return a instance of smart pointer that would deallocate the buffer when out of scope.
   *  The OrtAllocator instances must be valid at the point of memory release.
   */
  AllocatedStringPtr GetGraphNameAllocated(OrtAllocator* allocator) const;  ///< Wraps OrtApi::ModelMetadataGetGraphName

  /** \brief Returns a copy of the domain name.
   *
   * \param allocator to allocate memory for the copy of the name returned
   * \return a instance of smart pointer that would deallocate the buffer when out of scope.
   *  The OrtAllocator instances must be valid at the point of memory release.
   */
  AllocatedStringPtr GetDomainAllocated(OrtAllocator* allocator) const;  ///< Wraps OrtApi::ModelMetadataGetDomain

  /** \brief Returns a copy of the description.
   *
   * \param allocator to allocate memory for the copy of the string returned
   * \return a instance of smart pointer that would deallocate the buffer when out of scope.
   *  The OrtAllocator instances must be valid at the point of memory release.
   */
  AllocatedStringPtr GetDescriptionAllocated(OrtAllocator* allocator) const;  ///< Wraps OrtApi::ModelMetadataGetDescription

  /** \brief Returns a copy of the graph description.
   *
   * \param allocator to allocate memory for the copy of the string returned
   * \return a instance of smart pointer that would deallocate the buffer when out of scope.
   *  The OrtAllocator instances must be valid at the point of memory release.
   */
  AllocatedStringPtr GetGraphDescriptionAllocated(OrtAllocator* allocator) const;  ///< Wraps OrtApi::ModelMetadataGetGraphDescription

  /** \brief Returns a vector of copies of the custom metadata keys.
   *
   * \param allocator to allocate memory for the copy of the string returned
   * \return a instance std::vector of smart pointers that would deallocate the buffers when out of scope.
   *  The OrtAllocator instance must be valid at the point of memory release.
   */
  std::vector<AllocatedStringPtr> GetCustomMetadataMapKeysAllocated(OrtAllocator* allocator) const;  ///< Wraps OrtApi::ModelMetadataGetCustomMetadataMapKeys

  /** \brief Looks up a value by a key in the Custom Metadata map
   *
   * \param key zero terminated string key to lookup
   * \param allocator to allocate memory for the copy of the string returned
   * \return a instance of smart pointer that would deallocate the buffer when out of scope.
   *  maybe nullptr if key is not found.
   *
   *  The OrtAllocator instances must be valid at the point of memory release.
   */
  AllocatedStringPtr LookupCustomMetadataMapAllocated(const char* key, OrtAllocator* allocator) const;  ///< Wraps OrtApi::ModelMetadataLookupCustomMetadataMap

  int64_t GetVersion() const;  ///< Wraps OrtApi::ModelMetadataGetVersion
};

/** \brief Wrapper around ::OrtSession
 *
 */
struct Session : Base<OrtSession> {
  explicit Session(std::nullptr_t) {}                                                                                                        ///< Create an empty Session object, must be assigned a valid one to be used
  Session(Env& env, const ORTCHAR_T* model_path, const SessionOptions& options);                                                             ///< Wraps OrtApi::CreateSession
  Session(Env& env, const ORTCHAR_T* model_path, const SessionOptions& options, OrtPrepackedWeightsContainer* prepacked_weights_container);  ///< Wraps OrtApi::CreateSessionWithPrepackedWeightsContainer
  Session(Env& env, const void* model_data, size_t model_data_length, const SessionOptions& options);                                        ///< Wraps OrtApi::CreateSessionFromArray
  Session(Env& env, const void* model_data, size_t model_data_length, const SessionOptions& options,
          OrtPrepackedWeightsContainer* prepacked_weights_container);  ///< Wraps OrtApi::CreateSessionFromArrayWithPrepackedWeightsContainer

  /** \brief Run the model returning results in an Ort allocated vector.
   *
   * Wraps OrtApi::Run
   *
   * The caller provides a list of inputs and a list of the desired outputs to return.
   *
   * See the output logs for more information on warnings/errors that occur while processing the model.
   * Common errors are.. (TODO)
   *
   * \param[in] run_options
   * \param[in] input_names Array of null terminated strings of length input_count that is the list of input names
   * \param[in] input_values Array of Value objects of length input_count that is the list of input values
   * \param[in] input_count Number of inputs (the size of the input_names & input_values arrays)
   * \param[in] output_names Array of C style strings of length output_count that is the list of output names
   * \param[in] output_count Number of outputs (the size of the output_names array)
   * \return A std::vector of Value objects that directly maps to the output_count (eg. output_name[0] is the first entry of the returned vector)
   */
  std::vector<Value> Run(const RunOptions& run_options, const char* const* input_names, const Value* input_values, size_t input_count,
                         const char* const* output_names, size_t output_count);

  /** \brief Run the model returning results in user provided outputs
   * Same as Run(const RunOptions&, const char* const*, const Value*, size_t,const char* const*, size_t)
   */
  void Run(const RunOptions& run_options, const char* const* input_names, const Value* input_values, size_t input_count,
           const char* const* output_names, Value* output_values, size_t output_count);

  void Run(const RunOptions& run_options, const struct IoBinding&);  ///< Wraps OrtApi::RunWithBinding

  size_t GetInputCount() const;                   ///< Returns the number of model inputs
  size_t GetOutputCount() const;                  ///< Returns the number of model outputs
  size_t GetOverridableInitializerCount() const;  ///< Returns the number of inputs that have defaults that can be overridden

  /** \brief Returns a copy of input name at the specified index.
   *
   * \param index must less than the value returned by GetInputCount()
   * \param allocator to allocate memory for the copy of the name returned
   * \return a instance of smart pointer that would deallocate the buffer when out of scope.
   *  The OrtAllocator instances must be valid at the point of memory release.
   */
  AllocatedStringPtr GetInputNameAllocated(size_t index, OrtAllocator* allocator) const;

  /** \brief Returns a copy of output name at then specified index.
   *
   * \param index must less than the value returned by GetOutputCount()
   * \param allocator to allocate memory for the copy of the name returned
   * \return a instance of smart pointer that would deallocate the buffer when out of scope.
   *  The OrtAllocator instances must be valid at the point of memory release.
   */
  AllocatedStringPtr GetOutputNameAllocated(size_t index, OrtAllocator* allocator) const;

  /** \brief Returns a copy of the overridable initializer name at then specified index.
   *
   * \param index must less than the value returned by GetOverridableInitializerCount()
   * \param allocator to allocate memory for the copy of the name returned
   * \return a instance of smart pointer that would deallocate the buffer when out of scope.
   *  The OrtAllocator instances must be valid at the point of memory release.
   */
  AllocatedStringPtr GetOverridableInitializerNameAllocated(size_t index, OrtAllocator* allocator) const;  ///< Wraps OrtApi::SessionGetOverridableInitializerName

  /** \brief Returns a copy of the profiling file name.
   *
   * \param allocator to allocate memory for the copy of the string returned
   * \return a instance of smart pointer that would deallocate the buffer when out of scope.
   *  The OrtAllocator instances must be valid at the point of memory release.
   */
  AllocatedStringPtr EndProfilingAllocated(OrtAllocator* allocator) const;  ///< Wraps OrtApi::SessionEndProfiling
  uint64_t GetProfilingStartTimeNs() const;                                 ///< Wraps OrtApi::SessionGetProfilingStartTimeNs
  ModelMetadata GetModelMetadata() const;                                   ///< Wraps OrtApi::SessionGetModelMetadata

  TypeInfo GetInputTypeInfo(size_t index) const;                   ///< Wraps OrtApi::SessionGetInputTypeInfo
  TypeInfo GetOutputTypeInfo(size_t index) const;                  ///< Wraps OrtApi::SessionGetOutputTypeInfo
  TypeInfo GetOverridableInitializerTypeInfo(size_t index) const;  ///< Wraps OrtApi::SessionGetOverridableInitializerTypeInfo
};

/** \brief Wrapper around ::OrtTensorTypeAndShapeInfo
 *
 */
struct TensorTypeAndShapeInfo : Base<OrtTensorTypeAndShapeInfo> {
  explicit TensorTypeAndShapeInfo(std::nullptr_t) {}                                                     ///< Create an empty TensorTypeAndShapeInfo object, must be assigned a valid one to be used
  explicit TensorTypeAndShapeInfo(OrtTensorTypeAndShapeInfo* p) : Base<OrtTensorTypeAndShapeInfo>{p} {}  ///< Used for interop with the C API

  ONNXTensorElementDataType GetElementType() const;  ///< Wraps OrtApi::GetTensorElementType
  size_t GetElementCount() const;                    ///< Wraps OrtApi::GetTensorShapeElementCount

  size_t GetDimensionsCount() const;  ///< Wraps OrtApi::GetDimensionsCount

  /** \deprecated use GetShape() returning std::vector
   * [[deprecated]]
   * This interface is unsafe to use
   */
  [[deprecated("use GetShape()")]]
  void GetDimensions(int64_t* values, size_t values_count) const;  ///< Wraps OrtApi::GetDimensions

  void GetSymbolicDimensions(const char** values, size_t values_count) const;  ///< Wraps OrtApi::GetSymbolicDimensions

  std::vector<int64_t> GetShape() const;  ///< Uses GetDimensionsCount & GetDimensions to return a std::vector of the shape
};

/** \brief Wrapper around ::OrtSequenceTypeInfo
 *
 */
struct SequenceTypeInfo : Base<OrtSequenceTypeInfo> {
  explicit SequenceTypeInfo(std::nullptr_t) {}                                         ///< Create an empty SequenceTypeInfo object, must be assigned a valid one to be used
  explicit SequenceTypeInfo(OrtSequenceTypeInfo* p) : Base<OrtSequenceTypeInfo>{p} {}  ///< Used for interop with the C API

  TypeInfo GetSequenceElementType() const;  ///< Wraps OrtApi::GetSequenceElementType
};

/** \brief Wrapper around ::OrtMapTypeInfo
 *
 */
struct MapTypeInfo : Base<OrtMapTypeInfo> {
  explicit MapTypeInfo(std::nullptr_t) {}                               ///< Create an empty MapTypeInfo object, must be assigned a valid one to be used
  explicit MapTypeInfo(OrtMapTypeInfo* p) : Base<OrtMapTypeInfo>{p} {}  ///< Used for interop with the C API

  ONNXTensorElementDataType GetMapKeyType() const;  ///< Wraps OrtApi::GetMapKeyType
  TypeInfo GetMapValueType() const;                 ///< Wraps OrtApi::GetMapValueType
};

struct TypeInfo : Base<OrtTypeInfo> {
  explicit TypeInfo(std::nullptr_t) {}                         ///< Create an empty TypeInfo object, must be assigned a valid one to be used
  explicit TypeInfo(OrtTypeInfo* p) : Base<OrtTypeInfo>{p} {}  ///< C API Interop

  Unowned<TensorTypeAndShapeInfo> GetTensorTypeAndShapeInfo() const;  ///< Wraps OrtApi::CastTypeInfoToTensorInfo
  Unowned<SequenceTypeInfo> GetSequenceTypeInfo() const;              ///< Wraps OrtApi::CastTypeInfoToSequenceTypeInfo
  Unowned<MapTypeInfo> GetMapTypeInfo() const;                        ///< Wraps OrtApi::CastTypeInfoToMapTypeInfo

  ONNXType GetONNXType() const;
};

struct Value : Base<OrtValue> {
  // This structure is used to feed  sparse tensor values
  // information for use with FillSparseTensor<Format>() API
  // if the data type for the sparse tensor values is numeric
  // use data.p_data, otherwise, use data.str pointer to feed
  // values. data.str is an array of const char* that are zero terminated.
  // number of strings in the array must match shape size.
  // For fully sparse tensors use shape {0} and set p_data/str
  // to nullptr.
  struct OrtSparseValuesParam {
    const int64_t* values_shape;
    size_t values_shape_len;
    union {
      const void* p_data;
      const char** str;
    } data;
  };

  // Provides a way to pass shape in a single
  // argument
  struct Shape {
    const int64_t* shape;
    size_t shape_len;
  };

  /** \brief Creates a tensor with a user supplied buffer. Wraps OrtApi::CreateTensorWithDataAsOrtValue.
   * \tparam T The numeric datatype. This API is not suitable for strings.
   * \param info Memory description of where the p_data buffer resides (CPU vs GPU etc).
   * \param p_data Pointer to the data buffer.
   * \param p_data_element_count The number of elements in the data buffer.
   * \param shape Pointer to the tensor shape dimensions.
   * \param shape_len The number of tensor shape dimensions.
   */
  template <typename T>
  static Value CreateTensor(const OrtMemoryInfo* info, T* p_data, size_t p_data_element_count, const int64_t* shape, size_t shape_len);

  /** \brief Creates a tensor with a user supplied buffer. Wraps OrtApi::CreateTensorWithDataAsOrtValue.
   * \param info Memory description of where the p_data buffer resides (CPU vs GPU etc).
   * \param p_data Pointer to the data buffer.
   * \param p_data_byte_count The number of bytes in the data buffer.
   * \param shape Pointer to the tensor shape dimensions.
   * \param shape_len The number of tensor shape dimensions.
   * \param type The data type.
   */
  static Value CreateTensor(const OrtMemoryInfo* info, void* p_data, size_t p_data_byte_count, const int64_t* shape, size_t shape_len,
                            ONNXTensorElementDataType type);

#if !defined(DISABLE_SPARSE_TENSORS)
  /// <summary>
  /// This is a simple forwarding method to the other overload that helps deducing
  /// data type enum value from the type of the buffer.
  /// </summary>
  /// <typeparam name="T">numeric datatype. This API is not suitable for strings.</typeparam>
  /// <param name="info">Memory description where the user buffers reside (CPU vs GPU etc)</param>
  /// <param name="p_data">pointer to the user supplied buffer, use nullptr for fully sparse tensors</param>
  /// <param name="dense_shape">a would be dense shape of the tensor</param>
  /// <param name="values_shape">non zero values shape. Use a single 0 shape for fully sparse tensors.</param>
  /// <returns></returns>
  template <typename T>
  static Value CreateSparseTensor(const OrtMemoryInfo* info, T* p_data, const Shape& dense_shape,
                                  const Shape& values_shape);

  /// <summary>
  /// Creates an OrtValue instance containing SparseTensor. This constructs
  /// a sparse tensor that makes use of user allocated buffers. It does not make copies
  /// of the user provided data and does not modify it. The lifespan of user provided buffers should
  /// eclipse the life span of the resulting OrtValue. This call constructs an instance that only contain
  /// a pointer to non-zero values. To fully populate the sparse tensor call Use<Format>Indices() API below
  /// to supply a sparse format specific indices.
  /// This API is not suitable for string data. Use CreateSparseTensor() with allocator specified so strings
  /// can be properly copied into the allocated buffer.
  /// </summary>
  /// <param name="info">Memory description where the user buffers reside (CPU vs GPU etc)</param>
  /// <param name="p_data">pointer to the user supplied buffer, use nullptr for fully sparse tensors</param>
  /// <param name="dense_shape">a would be dense shape of the tensor</param>
  /// <param name="values_shape">non zero values shape. Use a single 0 shape for fully sparse tensors.</param>
  /// <param name="type">data type</param>
  /// <returns>Ort::Value instance containing SparseTensor</returns>
  static Value CreateSparseTensor(const OrtMemoryInfo* info, void* p_data, const Shape& dense_shape,
                                  const Shape& values_shape, ONNXTensorElementDataType type);

  /// <summary>
  /// Supplies COO format specific indices and marks the contained sparse tensor as being a COO format tensor.
  /// Values are supplied with a CreateSparseTensor() API. The supplied indices are not copied and the user
  /// allocated buffers lifespan must eclipse that of the OrtValue.
  /// The location of the indices is assumed to be the same as specified by OrtMemoryInfo argument at the creation time.
  /// </summary>
  /// <param name="indices_data">pointer to the user allocated buffer with indices. Use nullptr for fully sparse tensors.</param>
  /// <param name="indices_num">number of indices entries. Use 0 for fully sparse tensors</param>
  void UseCooIndices(int64_t* indices_data, size_t indices_num);

  /// <summary>
  /// Supplies CSR format specific indices and marks the contained sparse tensor as being a CSR format tensor.
  /// Values are supplied with a CreateSparseTensor() API. The supplied indices are not copied and the user
  /// allocated buffers lifespan must eclipse that of the OrtValue.
  /// The location of the indices is assumed to be the same as specified by OrtMemoryInfo argument at the creation time.
  /// </summary>
  /// <param name="inner_data">pointer to the user allocated buffer with inner indices or nullptr for fully sparse tensors</param>
  /// <param name="inner_num">number of csr inner indices or 0 for fully sparse tensors</param>
  /// <param name="outer_data">pointer to the user allocated buffer with outer indices or nullptr for fully sparse tensors</param>
  /// <param name="outer_num">number of csr outer indices or 0 for fully sparse tensors</param>
  void UseCsrIndices(int64_t* inner_data, size_t inner_num, int64_t* outer_data, size_t outer_num);

  /// <summary>
  /// Supplies BlockSparse format specific indices and marks the contained sparse tensor as being a BlockSparse format tensor.
  /// Values are supplied with a CreateSparseTensor() API. The supplied indices are not copied and the user
  /// allocated buffers lifespan must eclipse that of the OrtValue.
  /// The location of the indices is assumed to be the same as specified by OrtMemoryInfo argument at the creation time.
  /// </summary>
  /// <param name="indices_shape">indices shape or a {0} for fully sparse</param>
  /// <param name="indices_data">user allocated buffer with indices or nullptr for fully spare tensors</param>
  void UseBlockSparseIndices(const Shape& indices_shape, int32_t* indices_data);

#endif  // !defined(DISABLE_SPARSE_TENSORS)

  /** \brief Creates a tensor using a supplied OrtAllocator. Wraps OrtApi::CreateTensorAsOrtValue.
   * \tparam T The numeric datatype. This API is not suitable for strings.
   * \param allocator The allocator to use.
   * \param shape Pointer to the tensor shape dimensions.
   * \param shape_len The number of tensor shape dimensions.
   */
  template <typename T>
  static Value CreateTensor(OrtAllocator* allocator, const int64_t* shape, size_t shape_len);

  /** \brief Creates a tensor using a supplied OrtAllocator. Wraps OrtApi::CreateTensorAsOrtValue.
   * \param allocator The allocator to use.
   * \param shape Pointer to the tensor shape dimensions.
   * \param shape_len The number of tensor shape dimensions.
   * \param type The data type.
   */
  static Value CreateTensor(OrtAllocator* allocator, const int64_t* shape, size_t shape_len, ONNXTensorElementDataType type);

#if !defined(DISABLE_SPARSE_TENSORS)
  /// <summary>
  /// This is a simple forwarding method the below CreateSparseTensor.
  /// This helps to specify data type enum in terms of C++ data type.
  /// Use CreateSparseTensor<T>
  /// </summary>
  /// <typeparam name="T">numeric data type only. String data enum must be specified explicitly.</typeparam>
  /// <param name="allocator">allocator to use</param>
  /// <param name="dense_shape">a would be dense shape of the tensor</param>
  /// <returns>Ort::Value</returns>
  template <typename T>
  static Value CreateSparseTensor(OrtAllocator* allocator, const Shape& dense_shape);

  /// <summary>
  /// Creates an instance of OrtValue containing sparse tensor. The created instance has no data.
  /// The data must be supplied by on of the FillSparseTensor<Format>() methods that take both non-zero values
  /// and indices. The data will be copied into a buffer that would be allocated using the supplied allocator.
  /// Use this API to create OrtValues that contain sparse tensors with all supported data types including
  /// strings.
  /// </summary>
  /// <param name="allocator">allocator to use. The allocator lifespan must eclipse that of the resulting OrtValue</param>
  /// <param name="dense_shape">a would be dense shape of the tensor</param>
  /// <param name="type">data type</param>
  /// <returns>an instance of Ort::Value</returns>
  static Value CreateSparseTensor(OrtAllocator* allocator, const Shape& dense_shape, ONNXTensorElementDataType type);

  /// <summary>
  /// The API will allocate memory using the allocator instance supplied to the CreateSparseTensor() API
  /// and copy the values and COO indices into it. If data_mem_info specifies that the data is located
  /// at difference device than the allocator, a X-device copy will be performed if possible.
  /// </summary>
  /// <param name="data_mem_info">specified buffer memory description</param>
  /// <param name="values_param">values buffer information.</param>
  /// <param name="indices_data">coo indices buffer or nullptr for fully sparse data</param>
  /// <param name="indices_num">number of COO indices or 0 for fully sparse data</param>
  void FillSparseTensorCoo(const OrtMemoryInfo* data_mem_info, const OrtSparseValuesParam& values_param,
                           const int64_t* indices_data, size_t indices_num);

  /// <summary>
  /// The API will allocate memory using the allocator instance supplied to the CreateSparseTensor() API
  /// and copy the values and CSR indices into it. If data_mem_info specifies that the data is located
  /// at difference device than the allocator, a X-device copy will be performed if possible.
  /// </summary>
  /// <param name="data_mem_info">specified buffer memory description</param>
  /// <param name="values">values buffer information</param>
  /// <param name="inner_indices_data">csr inner indices pointer or nullptr for fully sparse tensors</param>
  /// <param name="inner_indices_num">number of csr inner indices or 0 for fully sparse tensors</param>
  /// <param name="outer_indices_data">pointer to csr indices data or nullptr for fully sparse tensors</param>
  /// <param name="outer_indices_num">number of csr outer indices or 0</param>
  void FillSparseTensorCsr(const OrtMemoryInfo* data_mem_info,
                           const OrtSparseValuesParam& values,
                           const int64_t* inner_indices_data, size_t inner_indices_num,
                           const int64_t* outer_indices_data, size_t outer_indices_num);

  /// <summary>
  /// The API will allocate memory using the allocator instance supplied to the CreateSparseTensor() API
  /// and copy the values and BlockSparse indices into it. If data_mem_info specifies that the data is located
  /// at difference device than the allocator, a X-device copy will be performed if possible.
  /// </summary>
  /// <param name="data_mem_info">specified buffer memory description</param>
  /// <param name="values">values buffer information</param>
  /// <param name="indices_shape">indices shape. use {0} for fully sparse tensors</param>
  /// <param name="indices_data">pointer to indices data or nullptr for fully sparse tensors</param>
  void FillSparseTensorBlockSparse(const OrtMemoryInfo* data_mem_info,
                                   const OrtSparseValuesParam& values,
                                   const Shape& indices_shape,
                                   const int32_t* indices_data);

  /// <summary>
  /// The API returns the sparse data format this OrtValue holds in a sparse tensor.
  /// If the sparse tensor was not fully constructed, i.e. Use*() or Fill*() API were not used
  /// the value returned is ORT_SPARSE_UNDEFINED.
  /// </summary>
  /// <returns>Format enum</returns>
  OrtSparseFormat GetSparseFormat() const;

  /// <summary>
  /// The API returns type and shape information for stored non-zero values of the
  /// sparse tensor. Use GetSparseTensorValues() to obtain values buffer pointer.
  /// </summary>
  /// <returns>TensorTypeAndShapeInfo values information</returns>
  TensorTypeAndShapeInfo GetSparseTensorValuesTypeAndShapeInfo() const;

  /// <summary>
  /// The API returns type and shape information for the specified indices. Each supported
  /// indices have their own enum values even if a give format has more than one kind of indices.
  /// Use GetSparseTensorIndicesData() to obtain pointer to indices buffer.
  /// </summary>
  /// <param name="format">enum requested</param>
  /// <returns>type and shape information</returns>
  TensorTypeAndShapeInfo GetSparseTensorIndicesTypeShapeInfo(OrtSparseIndicesFormat format) const;

  /// <summary>
  /// The API retrieves a pointer to the internal indices buffer. The API merely performs
  /// a convenience data type casting on the return type pointer. Make sure you are requesting
  /// the right type, use GetSparseTensorIndicesTypeShapeInfo();
  /// </summary>
  /// <typeparam name="T">type to cast to</typeparam>
  /// <param name="indices_format">requested indices kind</param>
  /// <param name="num_indices">number of indices entries</param>
  /// <returns>Pinter to the internal sparse tensor buffer containing indices. Do not free this pointer.</returns>
  template <typename T>
  const T* GetSparseTensorIndicesData(OrtSparseIndicesFormat indices_format, size_t& num_indices) const;

#endif  // !defined(DISABLE_SPARSE_TENSORS)

  static Value CreateMap(Value& keys, Value& values);       ///< Wraps OrtApi::CreateValue
  static Value CreateSequence(std::vector<Value>& values);  ///< Wraps OrtApi::CreateValue

  template <typename T>
  static Value CreateOpaque(const char* domain, const char* type_name, const T&);  ///< Wraps OrtApi::CreateOpaqueValue

  template <typename T>
  void GetOpaqueData(const char* domain, const char* type_name, T&) const;  ///< Wraps OrtApi::GetOpaqueValue

  explicit Value(std::nullptr_t) {}                   ///< Create an empty Value object, must be assigned a valid one to be used
  explicit Value(OrtValue* p) : Base<OrtValue>{p} {}  ///< Used for interop with the C API
  Value(Value&&) = default;
  Value& operator=(Value&&) = default;

  bool IsTensor() const;  ///< Returns true if Value is a tensor, false for other types like map/sequence/etc
  bool HasValue() const;  /// < Return true if OrtValue contains data and returns false if the OrtValue is a None

#if !defined(DISABLE_SPARSE_TENSORS)
  /// <summary>
  /// Returns true if the OrtValue contains a sparse tensor
  /// </summary>
  /// <returns></returns>
  bool IsSparseTensor() const;
#endif

  size_t GetCount() const;  // If a non tensor, returns 2 for map and N for sequence, where N is the number of elements
  Value GetValue(int index, OrtAllocator* allocator) const;

  /// <summary>
  /// This API returns a full length of string data contained within either a tensor or a sparse Tensor.
  /// For sparse tensor it returns a full length of stored non-empty strings (values). The API is useful
  /// for allocating necessary memory and calling GetStringTensorContent().
  /// </summary>
  /// <returns>total length of UTF-8 encoded bytes contained. No zero terminators counted.</returns>
  size_t GetStringTensorDataLength() const;

  /// <summary>
  /// The API copies all of the UTF-8 encoded string data contained within a tensor or a sparse tensor
  /// into a supplied buffer. Use GetStringTensorDataLength() to find out the length of the buffer to allocate.
  /// The user must also allocate offsets buffer with the number of entries equal to that of the contained
  /// strings.
  ///
  /// Strings are always assumed to be on CPU, no X-device copy.
  /// </summary>
  /// <param name="buffer">user allocated buffer</param>
  /// <param name="buffer_length">length in bytes of the allocated buffer</param>
  /// <param name="offsets">a pointer to the offsets user allocated buffer</param>
  /// <param name="offsets_count">count of offsets, must be equal to the number of strings contained.
  ///   that can be obtained from the shape of the tensor or from GetSparseTensorValuesTypeAndShapeInfo()
  ///   for sparse tensors</param>
  void GetStringTensorContent(void* buffer, size_t buffer_length, size_t* offsets, size_t offsets_count) const;

  /// <summary>
  /// Returns a non-const typed pointer to an OrtValue/Tensor contained buffer
  /// No type checking is performed, the caller must ensure the type matches the tensor type.
  /// </summary>
  /// <typeparam name="T"></typeparam>
  /// <returns>non-const pointer to data, no copies made</returns>
  template <typename T>
  T* GetTensorMutableData();  ///< Wraps OrtApi::GetTensorMutableData

  /// <summary>
  /// Returns a const typed pointer to the tensor contained data.
  /// No type checking is performed, the caller must ensure the type matches the tensor type.
  /// </summary>
  /// <typeparam name="T"></typeparam>
  /// <returns>const pointer to data, no copies made</returns>
  template <typename T>
  const T* GetTensorData() const;  ///< Wraps OrtApi::GetTensorMutableData

  /// <summary>
  /// Returns a non-typed pointer to a tensor contained data.
  /// </summary>
  /// <returns>const pointer to data, no copies made</returns>
  const void* GetTensorRawData() const;
  
  /// <summary>
  /// Returns a non-typed con-const pointer to a tensor contained data.
  /// </summary>
  /// <returns>pointer to data, no copies made</returns>
  void* GetTensorMutableRawData();

#if !defined(DISABLE_SPARSE_TENSORS)
  /// <summary>
  /// The API returns a pointer to an internal buffer of the sparse tensor
  /// containing non-zero values. The API merely does casting. Make sure you
  /// are requesting the right data type by calling GetSparseTensorValuesTypeAndShapeInfo()
  /// first.
  /// </summary>
  /// <typeparam name="T">numeric data types only. Use GetStringTensor*() to retrieve strings.</typeparam>
  /// <returns>a pointer to the internal values buffer. Do not free this pointer.</returns>
  template <typename T>
  const T* GetSparseTensorValues() const;
#endif

  template <typename T>
  T& At(const std::vector<int64_t>& location);

  /// <summary>
  /// The API returns type information for data contained in a tensor. For sparse
  /// tensors it returns type information for contained non-zero values.
  /// It returns dense shape for sparse tensors.
  /// </summary>
  /// <returns>TypeInfo</returns>
  TypeInfo GetTypeInfo() const;

  /// <summary>
  /// The API returns type information for data contained in a tensor. For sparse
  /// tensors it returns type information for contained non-zero values.
  /// It returns dense shape for sparse tensors.
  /// </summary>
  /// <returns>TensorTypeAndShapeInfo</returns>
  TensorTypeAndShapeInfo GetTensorTypeAndShapeInfo() const;

  /// <summary>
  /// This API returns information about the memory allocation used to hold data.
  /// MemoryInfo is owned by the underlying OrtValue and must not be released.
  /// </summary>
  /// <returns>Non owning instance of MemoryInfo</returns>
  Unowned<const MemoryInfo> GetTensorMemoryInfo() const;

  /// <summary>
  /// The API returns a byte length of UTF-8 encoded string element
  /// contained in either a tensor or a spare tensor values.
  /// </summary>
  /// <param name="element_index"></param>
  /// <returns>byte length for the specified string element</returns>
  size_t GetStringTensorElementLength(size_t element_index) const;

  /// <summary>
  /// The API copies UTF-8 encoded bytes for the requested string element
  /// contained within a tensor or a sparse tensor into a provided buffer.
  /// Use GetStringTensorElementLength() to obtain the length of the buffer to allocate.
  /// </summary>
  /// <param name="buffer_length"></param>
  /// <param name="element_index"></param>
  /// <param name="buffer"></param>
  void GetStringTensorElement(size_t buffer_length, size_t element_index, void* buffer) const;

  void FillStringTensor(const char* const* s, size_t s_len);
  void FillStringTensorElement(const char* s, size_t index);
};

/// <summary>
/// Represents native memory allocation coming from one of the
/// OrtAllocators registered with OnnxRuntime.
/// Use it to wrap an allocation made by an allocator
/// so it can be automatically released when no longer needed.
/// </summary>
struct MemoryAllocation {
  MemoryAllocation(OrtAllocator* allocator, void* p, size_t size);
  ~MemoryAllocation();
  MemoryAllocation(const MemoryAllocation&) = delete;
  MemoryAllocation& operator=(const MemoryAllocation&) = delete;
  MemoryAllocation(MemoryAllocation&&) noexcept;
  MemoryAllocation& operator=(MemoryAllocation&&) noexcept;

  void* get() { return p_; }
  size_t size() const { return size_; }

 private:
  OrtAllocator* allocator_;
  void* p_;
  size_t size_;
};

struct AllocatorWithDefaultOptions {
  AllocatorWithDefaultOptions();

  operator OrtAllocator*() { return p_; }
  operator const OrtAllocator*() const { return p_; }

  void* Alloc(size_t size);
  // The return value will own the allocation
  MemoryAllocation GetAllocation(size_t size);
  void Free(void* p);

  const OrtMemoryInfo* GetInfo() const;

 private:
  OrtAllocator* p_{};
};

struct MemoryInfo : Base<OrtMemoryInfo> {
  static MemoryInfo CreateCpu(OrtAllocatorType type, OrtMemType mem_type1);

  explicit MemoryInfo(std::nullptr_t) {} // No instance is created
  explicit MemoryInfo(OrtMemoryInfo* p) : Base<OrtMemoryInfo>{p} {}  ///< Used for interop with the C API
  MemoryInfo(const char* name, OrtAllocatorType type, int id, OrtMemType mem_type);

  std::string GetAllocatorName() const;
  OrtAllocatorType GetAllocatorType() const;
  int GetDeviceId() const;
  OrtMemoryInfoDeviceType GetDeviceType() const;
  OrtMemType GetMemoryType() const;

  bool operator==(const MemoryInfo& o) const;
};

struct Allocator : public Base<OrtAllocator> {
  Allocator(const Session& session, const MemoryInfo&);

  void* Alloc(size_t size) const;
  // The return value will own the allocation
  MemoryAllocation GetAllocation(size_t size);
  void Free(void* p) const;
  Unowned<const MemoryInfo> GetInfo() const;
};

struct IoBinding : public Base<OrtIoBinding> {
  explicit IoBinding(Session& session);
  void BindInput(const char* name, const Value&);
  void BindOutput(const char* name, const Value&);
  void BindOutput(const char* name, const MemoryInfo&);
  std::vector<std::string> GetOutputNames() const;
  std::vector<std::string> GetOutputNames(Allocator&) const;
  std::vector<Value> GetOutputValues() const;
  std::vector<Value> GetOutputValues(Allocator&) const;
  void ClearBoundInputs();
  void ClearBoundOutputs();
  void SynchronizeInputs();
  void SynchronizeOutputs();

 private:
  std::vector<std::string> GetOutputNamesHelper(OrtAllocator*) const;
  std::vector<Value> GetOutputValuesHelper(OrtAllocator*) const;
};

/*! \struct Ort::ArenaCfg
 * \brief it is a structure that represents the configuration of an arena based allocator
 * \details Please see docs/C_API.md for details
 */
struct ArenaCfg : Base<OrtArenaCfg> {
  explicit ArenaCfg(std::nullptr_t) {}  ///< Create an empty ArenaCfg object, must be assigned a valid one to be used
  /**
   * Wraps OrtApi::CreateArenaCfg
   * \param max_mem - use 0 to allow ORT to choose the default
   * \param arena_extend_strategy -  use -1 to allow ORT to choose the default, 0 = kNextPowerOfTwo, 1 = kSameAsRequested
   * \param initial_chunk_size_bytes - use -1 to allow ORT to choose the default
   * \param max_dead_bytes_per_chunk - use -1 to allow ORT to choose the default
   * See docs/C_API.md for details on what the following parameters mean and how to choose these values
   */
  ArenaCfg(size_t max_mem, int arena_extend_strategy, int initial_chunk_size_bytes, int max_dead_bytes_per_chunk);
};

//
// Custom OPs (only needed to implement custom OPs)
//

/// <summary>
/// This struct provides life time management for custom op attribute
/// </summary>
struct OpAttr : Base<OrtOpAttr> {
  OpAttr(const char* name, const void* data, int len, OrtOpAttrType type);
};

/// <summary>
/// This class wraps a raw pointer OrtKernelContext* that is being passed
/// to the custom kernel Compute() method. Use it to safely access context
/// attributes, input and output parameters with exception safety guarantees.
/// See usage example in onnxruntime/test/testdata/custom_op_library/custom_op_library.cc
/// </summary>
struct KernelContext {
  explicit KernelContext(OrtKernelContext* context);
  size_t GetInputCount() const;
  size_t GetOutputCount() const;
  Unowned<const Value> GetInput(size_t index) const;
  Unowned<Value> GetOutput(size_t index, const int64_t* dim_values, size_t dim_count) const;
  Unowned<Value> GetOutput(size_t index, const std::vector<int64_t>& dims) const;
  void* GetGPUComputeStream() const;

 private:
  OrtKernelContext* ctx_;
};

/// <summary>
/// This struct owns the OrtKernInfo* pointer when a copy is made.
/// For convenient wrapping of OrtKernelInfo* passed to kernel constructor
/// and query attributes, warp the pointer with Ort::Unowned<KernelInfo> instance
/// so it does not destroy the pointer the kernel does not own.
/// </summary>
struct KernelInfo : Base<OrtKernelInfo> {
  explicit KernelInfo(std::nullptr_t) {} ///< Create an empty instance to initialize later
  explicit KernelInfo(OrtKernelInfo* info); ///< Take ownership of the instance
  KernelInfo Copy() const;
  
  template <typename T>  // T is only implemented for float, int64_t, and string
  T GetAttribute(const char* name) const;

  template <typename T>  // T is only implemented for std::vector<float>, std::vector<int64_t>
  std::vector<T>  GetAttributes(const char* name) const;
};

/// <summary>
/// Create and own custom defined operation.
/// </summary>
struct Op : Base<OrtOp> {
  explicit Op(std::nullptr_t) {}  ///< Create an empty Operator object, must be assigned a valid one to be used
  
  explicit Op(OrtOp*);  ///< Take ownership of the OrtOp
  
  static Op Create(const OrtKernelInfo* info, const char* op_name, const char* domain,
                  int version, const char** type_constraint_names,
                  const ONNXTensorElementDataType* type_constraint_values,
                  size_t type_constraint_count,
                  const OpAttr* attr_values,
                  size_t attr_count,
                  size_t input_count, size_t output_count);

  void Invoke(const OrtKernelContext* context,
              const Value* input_values,
              size_t input_count,
              Value* output_values,
              size_t output_count);

  // For easier refactoring
  void Invoke(const OrtKernelContext* context,
              const OrtValue* const* input_values,
              size_t input_count,
              OrtValue* const* output_values,
              size_t output_count);
};

/// <summary>
/// This entire structure is deprecated, but we not marking
/// it as a whole yet since we want to preserve for the next release.
/// </summary>
struct CustomOpApi {
  CustomOpApi(const OrtApi& api) : api_(api) {}

  /** \deprecated use Ort::Value::GetTensorTypeAndShape()
   * [[deprecated]]
   * This interface produces a pointer that must be released. Not exception safe.
   */
  [[deprecated("use Ort::Value::GetTensorTypeAndShape()")]]
  OrtTensorTypeAndShapeInfo* GetTensorTypeAndShape(_In_ const OrtValue* value);

  /** \deprecated use Ort::TensorTypeAndShapeInfo::GetElementCount()
   * [[deprecated]]
   * This interface is redundant.
   */
  [[deprecated("use Ort::TensorTypeAndShapeInfo::GetElementCount()")]]
  size_t GetTensorShapeElementCount(_In_ const OrtTensorTypeAndShapeInfo* info);

  /** \deprecated use Ort::TensorTypeAndShapeInfo::GetElementType()
   * [[deprecated]]
   * This interface is redundant.
   */
  [[deprecated("use Ort::TensorTypeAndShapeInfo::GetElementType()")]]
  ONNXTensorElementDataType GetTensorElementType(const OrtTensorTypeAndShapeInfo* info);

  /** \deprecated use Ort::TensorTypeAndShapeInfo::GetDimensionsCount()
   * [[deprecated]]
   * This interface is redundant.
   */
  [[deprecated("use Ort::TensorTypeAndShapeInfo::GetDimensionsCount()")]]
  size_t GetDimensionsCount(_In_ const OrtTensorTypeAndShapeInfo* info);

  /** \deprecated use Ort::TensorTypeAndShapeInfo::GetShape()
   * [[deprecated]]
   * This interface is redundant.
   */
  [[deprecated("use Ort::TensorTypeAndShapeInfo::GetShape()")]]
  void GetDimensions(_In_ const OrtTensorTypeAndShapeInfo* info, _Out_ int64_t* dim_values, size_t dim_values_length);

  /** \deprecated
   * [[deprecated]]
   * This interface sets dimensions to TensorTypeAndShapeInfo, but has no effect on the OrtValue.
   */
  [[deprecated("Do not use")]]
  void SetDimensions(OrtTensorTypeAndShapeInfo* info, _In_ const int64_t* dim_values, size_t dim_count);

  /** \deprecated use Ort::Value::GetTensorMutableData()
   * [[deprecated]]
   * This interface is redundant.
   */
  template <typename T>
  [[deprecated("use Ort::Value::GetTensorMutableData()")]]
  T* GetTensorMutableData(_Inout_ OrtValue* value);

  /** \deprecated use Ort::Value::GetTensorData()
   * [[deprecated]]
   * This interface is redundant.
   */
  template <typename T>
  [[deprecated("use Ort::Value::GetTensorData()")]]
  const T* GetTensorData(_Inout_ const OrtValue* value);

  /** \deprecated use Ort::Value::GetTensorMemoryInfo()
   * [[deprecated]]
   * This interface is redundant.
   */
  [[deprecated("use Ort::Value::GetTensorMemoryInfo()")]]
  const OrtMemoryInfo* GetTensorMemoryInfo(_In_ const OrtValue* value);

  /** \deprecated use Ort::TensorTypeAndShapeInfo::GetShape()
   * [[deprecated]]
   * This interface is redundant.
   */
  [[deprecated("use Ort::TensorTypeAndShapeInfo::GetShape()")]]
  std::vector<int64_t> GetTensorShape(const OrtTensorTypeAndShapeInfo* info);

  /** \deprecated use TensorTypeAndShapeInfo instances for automatic ownership.
   * [[deprecated]]
   * This interface is not exception safe.
   */
  [[deprecated("use TensorTypeAndShapeInfo")]]
  void ReleaseTensorTypeAndShapeInfo(OrtTensorTypeAndShapeInfo* input);

  /** \deprecated use Ort::KernelContext::GetInputCount
   * [[deprecated]]
   * This interface is redundant.
   */
  [[deprecated("use Ort::KernelContext::GetInputCount")]]
  size_t KernelContext_GetInputCount(const OrtKernelContext* context);

  /** \deprecated use Ort::KernelContext::GetInput
   * [[deprecated]]
   * This interface is redundant.
   */
  [[deprecated("use Ort::KernelContext::GetInput")]]
  const OrtValue* KernelContext_GetInput(const OrtKernelContext* context, _In_ size_t index);

  /** \deprecated use Ort::KernelContext::GetOutputCount
   * [[deprecated]]
   * This interface is redundant.
   */
  [[deprecated("use Ort::KernelContext::GetOutputCount")]]
  size_t KernelContext_GetOutputCount(const OrtKernelContext* context);

  /** \deprecated use Ort::KernelContext::GetOutput
   * [[deprecated]]
   * This interface is redundant.
   */
  [[deprecated("use Ort::KernelContext::GetOutput")]]
  OrtValue* KernelContext_GetOutput(OrtKernelContext* context, _In_ size_t index, _In_ const int64_t* dim_values, size_t dim_count);

  /** \deprecated use Ort::KernelContext::GetGPUComputeStream
   * [[deprecated]]
   * This interface is redundant.
   */
  [[deprecated("use Ort::KernelContext::GetGPUComputeStream")]]
  void* KernelContext_GetGPUComputeStream(const OrtKernelContext* context);

  /** \deprecated use Ort::ThrowOnError()
   * [[deprecated]]
   * This interface is redundant.
   */
  [[deprecated("use Ort::ThrowOnError()")]]
  void ThrowOnError(OrtStatus* result);

  /** \deprecated use Ort::OpAttr
   * [[deprecated]]
   * This interface is not exception safe.
   */
  [[deprecated("use Ort::OpAttr")]]
  OrtOpAttr* CreateOpAttr(_In_ const char* name,
                          _In_ const void* data,
                          _In_ int len,
                          _In_ OrtOpAttrType type);

  /** \deprecated use Ort::OpAttr
   * [[deprecated]]
   * This interface is not exception safe.
   */
  [[deprecated("use Ort::OpAttr")]]
  void ReleaseOpAttr(_Frees_ptr_opt_ OrtOpAttr* op_attr);

  /** \deprecated use Ort::Op
   * [[deprecated]]
   * This interface is not exception safe.
   */
  [[deprecated("use Ort::Op")]]
  OrtOp* CreateOp(_In_ const OrtKernelInfo* info,
                  _In_ const char* op_name,
                  _In_ const char* domain,
                  _In_ int version,
                  _In_opt_ const char** type_constraint_names,
                  _In_opt_ const ONNXTensorElementDataType* type_constraint_values,
                  _In_opt_ int type_constraint_count,
                  _In_opt_ const OrtOpAttr* const* attr_values,
                  _In_opt_ int attr_count,
                  _In_ int input_count,
                  _In_ int output_count);

  /** \deprecated use Ort::Op::Invoke
   * [[deprecated]]
   * This interface is redundant
   */
  [[deprecated("use Ort::Op::Invoke")]]
  void InvokeOp(_In_ const OrtKernelContext* context,
                _In_ const OrtOp* ort_op,
                _In_ const OrtValue* const* input_values,
                _In_ int input_count,
                _Inout_ OrtValue* const* output_values,
                _In_ int output_count);

  /** \deprecated use Ort::Op for automatic lifespan management.
   * [[deprecated]]
   * This interface is not exception safe.
   */
  [[deprecated("use Ort::Op")]]
  void ReleaseOp(_Frees_ptr_opt_ OrtOp* ort_op);

  /** \deprecated use Ort::KernelInfo for automatic lifespan management or for
   * querying attributes
   * [[deprecated]]
   * This interface is redundant
   */
  template <typename T>  // T is only implemented for std::vector<float>, std::vector<int64_t>, float, int64_t, and string
  [[deprecated("use Ort::KernelInfo::GetAttribute")]]
  T KernelInfoGetAttribute(_In_ const OrtKernelInfo* info, _In_ const char* name);

  /** \deprecated use Ort::KernelInfo::Copy
   * querying attributes
   * [[deprecated]]
   * This interface is not exception safe
   */
  [[deprecated("use Ort::KernelInfo::Copy")]]
  OrtKernelInfo* CopyKernelInfo(_In_ const OrtKernelInfo* info);

  /** \deprecated use Ort::KernelInfo for lifespan management
   * querying attributes
   * [[deprecated]]
   * This interface is not exception safe
   */
  [[deprecated("use Ort::KernelInfo")]]
  void ReleaseKernelInfo(_Frees_ptr_opt_ OrtKernelInfo* info_copy);

 private:
  const OrtApi& api_;
};

template <typename TOp, typename TKernel>
struct CustomOpBase : OrtCustomOp {
  CustomOpBase() {
    OrtCustomOp::version = ORT_API_VERSION;
    OrtCustomOp::CreateKernel = [](const OrtCustomOp* this_, const OrtApi* api, const OrtKernelInfo* info) { return static_cast<const TOp*>(this_)->CreateKernel(*api, info); };
    OrtCustomOp::GetName = [](const OrtCustomOp* this_) { return static_cast<const TOp*>(this_)->GetName(); };

    OrtCustomOp::GetExecutionProviderType = [](const OrtCustomOp* this_) { return static_cast<const TOp*>(this_)->GetExecutionProviderType(); };

    OrtCustomOp::GetInputTypeCount = [](const OrtCustomOp* this_) { return static_cast<const TOp*>(this_)->GetInputTypeCount(); };
    OrtCustomOp::GetInputType = [](const OrtCustomOp* this_, size_t index) { return static_cast<const TOp*>(this_)->GetInputType(index); };

    OrtCustomOp::GetOutputTypeCount = [](const OrtCustomOp* this_) { return static_cast<const TOp*>(this_)->GetOutputTypeCount(); };
    OrtCustomOp::GetOutputType = [](const OrtCustomOp* this_, size_t index) { return static_cast<const TOp*>(this_)->GetOutputType(index); };

    OrtCustomOp::KernelCompute = [](void* op_kernel, OrtKernelContext* context) { static_cast<TKernel*>(op_kernel)->Compute(context); };
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
#pragma warning(disable : 26409)
#endif
    OrtCustomOp::KernelDestroy = [](void* op_kernel) { delete static_cast<TKernel*>(op_kernel); };
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif
    OrtCustomOp::GetInputCharacteristic = [](const OrtCustomOp* this_, size_t index) { return static_cast<const TOp*>(this_)->GetInputCharacteristic(index); };
    OrtCustomOp::GetOutputCharacteristic = [](const OrtCustomOp* this_, size_t index) { return static_cast<const TOp*>(this_)->GetOutputCharacteristic(index); };
  }

  // Default implementation of GetExecutionProviderType that returns nullptr to default to the CPU provider
  const char* GetExecutionProviderType() const { return nullptr; }

  // Default implementations of GetInputCharacteristic() and GetOutputCharacteristic() below
  // (inputs and outputs are required by default)
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t /*index*/) const {
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  }

  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t /*index*/) const {
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  }
};

}  // namespace Ort

#include "onnxruntime_cxx_inline.h"
