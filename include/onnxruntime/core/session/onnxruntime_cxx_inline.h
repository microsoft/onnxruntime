// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Do not include this file directly. Please include "onnxruntime_cxx_api.h" instead.
// If interested in trying out features of the new experimental C++ API, include "experimental_onnxruntime_cxx_api.h" instead.
//
// These are the inline implementations of the C++ header APIs. They're in this separate file as to not clutter
// the main C++ file with implementation details.

#include <algorithm>
#include <functional>
#include <iterator>
#include <string>
#include <type_traits>
#include <vector>

// Convert OrtStatus to Ort::Status and return
// instead of throwing
#define ORT_CXX_RETURN_ON_API_FAIL(expression) \
  {                                            \
    auto ort_status = (expression);            \
    if (ort_status) {                          \
      return Ort::Status(ort_status);          \
    }                                          \
  }

#ifdef __cpp_if_constexpr
#define ORT_CXX_IF_CONSTEXPR if constexpr
#else
#define ORT_CXX_IF_CONSTEXPR if
#endif

namespace Ort {

namespace detail {
inline void ThrowStatus(const Status& st) {
  std::string error_message = st.GetErrorMessage();
  OrtErrorCode error_code = st.GetErrorCode();
  ORT_CXX_API_THROW(std::move(error_message), error_code);
}
}  // namespace detail

inline void ThrowOnError(OrtStatus* ort_status) {
  if (ort_status) {
    Ort::Status st(ort_status);
    detail::ThrowStatus(st);
  }
}

inline void ThrowOnError(const Status& st) {
  if (st) {
    detail::ThrowStatus(st);
  }
}

inline Status::Status(OrtStatus* status) noexcept : detail::Base<OrtStatus>{status} {
}

inline Status::Status(const std::exception& e) {
  p_ = GetApi().CreateStatus(ORT_FAIL, e.what());
}

inline Status::Status(const Exception& e) {
  p_ = GetApi().CreateStatus(e.GetOrtErrorCode(), e.what());
}

inline Status::Status(const char* message, OrtErrorCode code) {
  p_ = GetApi().CreateStatus(code, message);
}

inline std::string Status::GetErrorMessage() const {
  std::string message(GetApi().GetErrorMessage(p_));
  return message;
}

inline OrtErrorCode Status::GetErrorCode() const {
  return GetApi().GetErrorCode(p_);
}

inline bool Status::IsOK() const noexcept {
  return (p_ == nullptr);
}

// This template converts a C++ type into it's ONNXTensorElementDataType
template <typename T>
struct TypeToTensorType;
template <>
struct TypeToTensorType<float> {
  static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
};
template <>
struct TypeToTensorType<Float16_t> {
  static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
};
template <>
struct TypeToTensorType<BFloat16_t> {
  static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
};
template <>
struct TypeToTensorType<double> {
  static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
};
template <>
struct TypeToTensorType<int8_t> {
  static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
};
template <>
struct TypeToTensorType<int16_t> {
  static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
};
template <>
struct TypeToTensorType<int32_t> {
  static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
};
template <>
struct TypeToTensorType<int64_t> {
  static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
};
template <>
struct TypeToTensorType<uint8_t> {
  static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
};
template <>
struct TypeToTensorType<uint16_t> {
  static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
};
template <>
struct TypeToTensorType<uint32_t> {
  static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
};
template <>
struct TypeToTensorType<uint64_t> {
  static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
};
template <>
struct TypeToTensorType<bool> {
  static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
};

template <>
struct TypeToTensorType<Float8E4M3FN_t> {
  static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN;
};
template <>
struct TypeToTensorType<Float8E4M3FNUZ_t> {
  static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ;
};
template <>
struct TypeToTensorType<Float8E5M2_t> {
  static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2;
};
template <>
struct TypeToTensorType<Float8E5M2FNUZ_t> {
  static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ;
};

inline bool BFloat16_t::operator==(const BFloat16_t& rhs) const noexcept {
  if (IsNaN() || rhs.IsNaN()) {
    // IEEE defines that NaN is not equal to anything, including itself.
    return false;
  }
  return val == rhs.val;
}

inline bool BFloat16_t::operator<(const BFloat16_t& rhs) const noexcept {
  if (IsNaN() || rhs.IsNaN()) {
    // IEEE defines that NaN is unordered with respect to everything, including itself.
    return false;
  }

  const bool left_is_negative = IsNegative();
  if (left_is_negative != rhs.IsNegative()) {
    // When the signs of left and right differ, we know that left is less than right if it is
    // the negative value. The exception to this is if both values are zero, in which case IEEE
    // says they should be equal, even if the signs differ.
    return left_is_negative && !AreZero(*this, rhs);
  }
  return (val != rhs.val) && ((val < rhs.val) ^ left_is_negative);
}

inline MemoryAllocation::MemoryAllocation(OrtAllocator* allocator, void* p, size_t size)
    : allocator_(allocator), p_(p), size_(size) {
}

inline MemoryAllocation::~MemoryAllocation() {
  if (p_ != nullptr) {
    // We do not throw out of destructor
    auto ret = GetApi().AllocatorFree(allocator_, p_);
    static_cast<void>(ret);
  }
}

inline MemoryAllocation::MemoryAllocation(MemoryAllocation&& o) noexcept : allocator_(nullptr), p_(nullptr), size_(0) {
  *this = std::move(o);
}

inline MemoryAllocation& MemoryAllocation::operator=(MemoryAllocation&& o) noexcept {
  OrtAllocator* alloc = nullptr;
  void* p = nullptr;
  size_t sz = 0;

  // Swap out this
  std::swap(alloc, allocator_);
  std::swap(p, p_);
  std::swap(sz, size_);

  // Swap with incoming
  std::swap(allocator_, o.allocator_);
  std::swap(p_, o.p_);
  std::swap(size_, o.size_);

  // Destroy this instance if needed
  MemoryAllocation this_alloc(alloc, p, sz);
  return *this;
}

namespace detail {

template <typename T>
inline void* AllocatorImpl<T>::Alloc(size_t size) {
  void* out;
  ThrowOnError(GetApi().AllocatorAlloc(this->p_, size, &out));
  return out;
}

template <typename T>
inline MemoryAllocation AllocatorImpl<T>::GetAllocation(size_t size) {
  void* out;
  ThrowOnError(GetApi().AllocatorAlloc(this->p_, size, &out));
  MemoryAllocation result(this->p_, out, size);
  return result;
}

template <typename T>
inline void AllocatorImpl<T>::Free(void* p) {
  ThrowOnError(GetApi().AllocatorFree(this->p_, p));
}

template <typename T>
inline ConstMemoryInfo AllocatorImpl<T>::GetInfo() const {
  const OrtMemoryInfo* out;
  ThrowOnError(GetApi().AllocatorGetInfo(this->p_, &out));
  return ConstMemoryInfo{out};
}

template <typename T>
inline KeyValuePairs AllocatorImpl<T>::GetStats() const {
  OrtKeyValuePairs* out;
  ThrowOnError(GetApi().AllocatorGetStats(this->p_, &out));
  return KeyValuePairs(out);
}
}  // namespace detail

inline AllocatorWithDefaultOptions::AllocatorWithDefaultOptions() {
  ThrowOnError(GetApi().GetAllocatorWithDefaultOptions(&this->p_));
}

inline Allocator::Allocator(const Session& sess, const OrtMemoryInfo* mem_info) {
  ThrowOnError(GetApi().CreateAllocator(sess, mem_info, &this->p_));
}

namespace detail {

template <typename T>
inline std::string MemoryInfoImpl<T>::GetAllocatorName() const {
  const char* name = nullptr;
  ThrowOnError(GetApi().MemoryInfoGetName(this->p_, &name));
  return std::string(name);
}

template <typename T>
inline OrtAllocatorType MemoryInfoImpl<T>::GetAllocatorType() const {
  OrtAllocatorType type;
  ThrowOnError(GetApi().MemoryInfoGetType(this->p_, &type));
  return type;
}

template <typename T>
inline int MemoryInfoImpl<T>::GetDeviceId() const {
  int id = 0;
  ThrowOnError(GetApi().MemoryInfoGetId(this->p_, &id));
  return id;
}

template <typename T>
inline OrtMemoryInfoDeviceType MemoryInfoImpl<T>::GetDeviceType() const {
  OrtMemoryInfoDeviceType type;
  GetApi().MemoryInfoGetDeviceType(this->p_, &type);
  return type;
}

template <typename T>
inline OrtMemType MemoryInfoImpl<T>::GetMemoryType() const {
  OrtMemType type;
  ThrowOnError(GetApi().MemoryInfoGetMemType(this->p_, &type));
  return type;
}

template <typename T>
inline OrtDeviceMemoryType MemoryInfoImpl<T>::GetDeviceMemoryType() const {
  return GetApi().MemoryInfoGetDeviceMemType(this->p_);
}

template <typename T>
inline uint32_t MemoryInfoImpl<T>::GetVendorId() const {
  return GetApi().MemoryInfoGetVendorId(this->p_);
}

template <typename T>
template <typename U>
inline bool MemoryInfoImpl<T>::operator==(const MemoryInfoImpl<U>& o) const {
  int comp_result = 0;
  ThrowOnError(Ort::GetApi().CompareMemoryInfo(this->p_, o, &comp_result));
  return comp_result == 0;
}

}  // namespace detail

inline MemoryInfo MemoryInfo::CreateCpu(OrtAllocatorType type, OrtMemType mem_type) {
  OrtMemoryInfo* p;
  ThrowOnError(GetApi().CreateCpuMemoryInfo(type, mem_type, &p));
  return MemoryInfo(p);
}

inline MemoryInfo::MemoryInfo(const char* name, OrtAllocatorType type, int id, OrtMemType mem_type) {
  ThrowOnError(GetApi().CreateMemoryInfo(name, type, id, mem_type, &this->p_));
}

inline MemoryInfo::MemoryInfo(const char* name, OrtMemoryInfoDeviceType device_type, uint32_t vendor_id, uint32_t device_id,
                              OrtDeviceMemoryType mem_type, size_t alignment, OrtAllocatorType allocator_type) {
  ThrowOnError(GetApi().CreateMemoryInfo_V2(name, device_type, vendor_id, device_id, mem_type, alignment,
                                            allocator_type, &this->p_));
}

namespace detail {
template <typename T>
inline std::vector<std::string> ConstIoBindingImpl<T>::GetOutputNames() const {
  AllocatorWithDefaultOptions allocator;
  return binding_utils::GetOutputNamesHelper(this->p_, allocator);
}

template <typename T>
inline std::vector<std::string> ConstIoBindingImpl<T>::GetOutputNames(OrtAllocator* allocator) const {
  return binding_utils::GetOutputNamesHelper(this->p_, allocator);
}

template <typename T>
inline std::vector<Value> ConstIoBindingImpl<T>::GetOutputValues() const {
  AllocatorWithDefaultOptions allocator;
  return binding_utils::GetOutputValuesHelper(this->p_, allocator);
}

template <typename T>
inline std::vector<Value> ConstIoBindingImpl<T>::GetOutputValues(OrtAllocator* allocator) const {
  return binding_utils::GetOutputValuesHelper(this->p_, allocator);
}

template <typename T>
inline void IoBindingImpl<T>::BindInput(const char* name, const Value& value) {
  ThrowOnError(GetApi().BindInput(this->p_, name, value));
}

template <typename T>
inline void IoBindingImpl<T>::BindOutput(const char* name, const Value& value) {
  ThrowOnError(GetApi().BindOutput(this->p_, name, value));
}

template <typename T>
inline void IoBindingImpl<T>::BindOutput(const char* name, const OrtMemoryInfo* mem_info) {
  ThrowOnError(GetApi().BindOutputToDevice(this->p_, name, mem_info));
}

template <typename T>
inline void IoBindingImpl<T>::ClearBoundInputs() {
  GetApi().ClearBoundInputs(this->p_);
}

template <typename T>
inline void IoBindingImpl<T>::ClearBoundOutputs() {
  GetApi().ClearBoundOutputs(this->p_);
}

template <typename T>
inline void IoBindingImpl<T>::SynchronizeInputs() {
  ThrowOnError(GetApi().SynchronizeBoundInputs(this->p_));
}

template <typename T>
inline void IoBindingImpl<T>::SynchronizeOutputs() {
  ThrowOnError(GetApi().SynchronizeBoundOutputs(this->p_));
}

namespace binding_utils {
inline std::vector<std::string> GetOutputNamesHelper(const OrtIoBinding* binding, OrtAllocator* allocator) {
  std::vector<std::string> result;
  auto free_fn = detail::AllocatedFree(allocator);
  using Ptr = std::unique_ptr<void, decltype(free_fn)>;

  char* buffer = nullptr;
  size_t* lengths = nullptr;
  size_t count = 0;
  ThrowOnError(GetApi().GetBoundOutputNames(binding, allocator, &buffer, &lengths, &count));

  if (count == 0) {
    return result;
  }

  Ptr buffer_g(buffer, free_fn);
  Ptr lengths_g(lengths, free_fn);

  result.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    auto sz = *lengths;
    result.emplace_back(buffer, sz);
    buffer += sz;
    ++lengths;
  }
  return result;
}

inline std::vector<Value> GetOutputValuesHelper(const OrtIoBinding* binding, OrtAllocator* allocator) {
  std::vector<Value> result;
  size_t output_count = 0;

  OrtValue** output_buffer = nullptr;
  ThrowOnError(GetApi().GetBoundOutputValues(binding, allocator, &output_buffer, &output_count));
  if (output_count == 0) {
    return result;
  }

  std::unique_ptr<void, AllocatedFree> buffer_g(output_buffer, AllocatedFree(allocator));

  result.reserve(output_count);
  for (size_t i = 0; i < output_count; ++i) {
    result.emplace_back(output_buffer[i]);
  }
  return result;
}

}  // namespace binding_utils
}  // namespace detail

inline IoBinding::IoBinding(Session& session) {
  ThrowOnError(GetApi().CreateIoBinding(session, &this->p_));
}

inline ArenaCfg::ArenaCfg(size_t max_mem, int arena_extend_strategy, int initial_chunk_size_bytes, int max_dead_bytes_per_chunk) {
  ThrowOnError(GetApi().CreateArenaCfg(max_mem, arena_extend_strategy, initial_chunk_size_bytes, max_dead_bytes_per_chunk, &p_));
}

inline ArenaCfg::ArenaCfg(const std::unordered_map<std::string, size_t>& arena_config) {
  std::vector<const char*> keys;
  std::vector<size_t> values;
  keys.reserve(arena_config.size());
  values.reserve(arena_config.size());
  for (const auto& kv : arena_config) {
    keys.push_back(kv.first.c_str());
    values.push_back(kv.second);
  }
  ThrowOnError(GetApi().CreateArenaCfgV2(keys.data(), values.data(), arena_config.size(), &p_));
}

inline ThreadingOptions::ThreadingOptions() {
  ThrowOnError(GetApi().CreateThreadingOptions(&p_));
}

inline ThreadingOptions& ThreadingOptions::SetGlobalIntraOpNumThreads(int intra_op_num_threads) {
  ThrowOnError(GetApi().SetGlobalIntraOpNumThreads(p_, intra_op_num_threads));
  return *this;
}

inline ThreadingOptions& ThreadingOptions::SetGlobalInterOpNumThreads(int inter_op_num_threads) {
  ThrowOnError(GetApi().SetGlobalInterOpNumThreads(p_, inter_op_num_threads));
  return *this;
}

inline ThreadingOptions& ThreadingOptions::SetGlobalSpinControl(int allow_spinning) {
  ThrowOnError(GetApi().SetGlobalSpinControl(p_, allow_spinning));
  return *this;
}

inline ThreadingOptions& ThreadingOptions::SetGlobalDenormalAsZero() {
  ThrowOnError(GetApi().SetGlobalDenormalAsZero(p_));
  return *this;
}

inline ThreadingOptions& ThreadingOptions::SetGlobalCustomCreateThreadFn(OrtCustomCreateThreadFn ort_custom_create_thread_fn) {
  ThrowOnError(GetApi().SetGlobalCustomCreateThreadFn(p_, ort_custom_create_thread_fn));
  return *this;
}

inline ThreadingOptions& ThreadingOptions::SetGlobalCustomThreadCreationOptions(void* ort_custom_thread_creation_options) {
  ThrowOnError(GetApi().SetGlobalCustomThreadCreationOptions(p_, ort_custom_thread_creation_options));
  return *this;
}

inline ThreadingOptions& ThreadingOptions::SetGlobalCustomJoinThreadFn(OrtCustomJoinThreadFn ort_custom_join_thread_fn) {
  ThrowOnError(GetApi().SetGlobalCustomJoinThreadFn(p_, ort_custom_join_thread_fn));
  return *this;
}

inline TensorRTProviderOptions::TensorRTProviderOptions() {
  ThrowOnError(GetApi().CreateTensorRTProviderOptions(&this->p_));
}

inline void TensorRTProviderOptions::Update(const std::unordered_map<std::string, std::string>& options) {
  std::vector<const char*> keys;
  std::vector<const char*> values;
  keys.reserve(options.size());
  values.reserve(options.size());
  for (const auto& kv : options) {
    keys.push_back(kv.first.c_str());
    values.push_back(kv.second.c_str());
  }
  ThrowOnError(GetApi().UpdateTensorRTProviderOptions(p_, keys.data(), values.data(), options.size()));
}

inline void TensorRTProviderOptions::UpdateWithValue(const char* key, void* value) {
  ThrowOnError(GetApi().UpdateTensorRTProviderOptionsWithValue(p_, key, value));
}

inline void* TensorRTProviderOptions::GetOptionByName(const char* name) const {
  void* value = nullptr;
  ThrowOnError(GetApi().GetTensorRTProviderOptionsByName(p_, name, &value));
  return value;
}

inline std::string TensorRTProviderOptions::GetTensorRTProviderOptionsAsString() const {
  AllocatorWithDefaultOptions allocator;
  char* options_str = nullptr;
  ThrowOnError(GetApi().GetTensorRTProviderOptionsAsString(p_, allocator, &options_str));
  std::unique_ptr<void, detail::AllocatedFree> options_str_g(options_str, detail::AllocatedFree(allocator));
  return std::string(options_str);
}

inline CUDAProviderOptions::CUDAProviderOptions() {
  ThrowOnError(GetApi().CreateCUDAProviderOptions(&this->p_));
}

inline void CUDAProviderOptions::Update(const std::unordered_map<std::string, std::string>& options) {
  std::vector<const char*> keys;
  std::vector<const char*> values;
  keys.reserve(options.size());
  values.reserve(options.size());
  for (const auto& kv : options) {
    keys.push_back(kv.first.c_str());
    values.push_back(kv.second.c_str());
  }
  ThrowOnError(GetApi().UpdateCUDAProviderOptions(p_, keys.data(), values.data(), options.size()));
}

inline std::string CUDAProviderOptions::GetCUDAProviderOptionsAsString() const {
  AllocatorWithDefaultOptions allocator;
  char* options_str = nullptr;
  ThrowOnError(GetApi().GetCUDAProviderOptionsAsString(p_, allocator, &options_str));
  std::unique_ptr<void, detail::AllocatedFree> options_str_g(options_str, detail::AllocatedFree(allocator));
  return std::string(options_str);
}

inline void CUDAProviderOptions::UpdateWithValue(const char* key, void* value) {
  ThrowOnError(GetApi().UpdateCUDAProviderOptionsWithValue(p_, key, value));
}

inline void* CUDAProviderOptions::GetOptionByName(const char* name) const {
  void* value = nullptr;
  ThrowOnError(GetApi().GetCUDAProviderOptionsByName(p_, name, &value));
  return value;
}

inline PrepackedWeightsContainer::PrepackedWeightsContainer() {
  ThrowOnError(GetApi().CreatePrepackedWeightsContainer(&this->p_));
}

namespace detail {

template <typename T>
inline const std::basic_string<ORTCHAR_T> ConstExternalInitializerInfoImpl<T>::GetFilePath() const {
  return GetApi().ExternalInitializerInfo_GetFilePath(this->p_);
}

template <typename T>
inline int64_t ConstExternalInitializerInfoImpl<T>::GetFileOffset() const {
  return GetApi().ExternalInitializerInfo_GetFileOffset(this->p_);
}

template <typename T>
inline size_t ConstExternalInitializerInfoImpl<T>::GetByteSize() const {
  return GetApi().ExternalInitializerInfo_GetByteSize(this->p_);
}
}  // namespace detail

inline ExternalInitializerInfo::ExternalInitializerInfo(const ORTCHAR_T* filepath, int64_t file_offset,
                                                        size_t byte_size) {
  ThrowOnError(GetApi().CreateExternalInitializerInfo(filepath, file_offset, byte_size, &this->p_));
}

inline Status ExternalInitializerInfo::Create(const ORTCHAR_T* filepath, int64_t file_offset, size_t byte_size,
                                              /*out*/ ExternalInitializerInfo& out) {
  OrtExternalInitializerInfo* info = nullptr;
  OrtStatus* status = GetApi().CreateExternalInitializerInfo(filepath, file_offset, byte_size, &info);
  if (status != nullptr) {
    return Status{status};
  }

  out = ExternalInitializerInfo(info);

  return Status{nullptr};
}

namespace detail {
template <typename T>
inline const char* KeyValuePairsImpl<T>::GetValue(const char* key) const {
  return GetApi().GetKeyValue(this->p_, key);
}

template <typename T>
inline std::unordered_map<std::string, std::string> KeyValuePairsImpl<T>::GetKeyValuePairs() const {
  std::unordered_map<std::string, std::string> out;

  size_t num_pairs = 0;
  const char* const* keys = nullptr;
  const char* const* values = nullptr;
  GetApi().GetKeyValuePairs(this->p_, &keys, &values, &num_pairs);
  if (num_pairs > 0) {
    out.reserve(num_pairs);
    for (size_t i = 0; i < num_pairs; ++i) {
      out.emplace(keys[i], values[i]);
    }
  }

  return out;
}

template <typename T>
inline void KeyValuePairsImpl<T>::GetKeyValuePairs(std::vector<const char*>& keys,
                                                   std::vector<const char*>& values) const {
  keys.clear();
  values.clear();

  size_t num_pairs = 0;
  const char* const* keys_ptr = nullptr;
  const char* const* values_ptr = nullptr;
  GetApi().GetKeyValuePairs(this->p_, &keys_ptr, &values_ptr, &num_pairs);
  if (num_pairs > 0) {
    keys.resize(num_pairs);
    values.resize(num_pairs);
    std::copy(keys_ptr, keys_ptr + num_pairs, keys.begin());
    std::copy(values_ptr, values_ptr + num_pairs, values.begin());
  }
}
}  // namespace detail

inline KeyValuePairs::KeyValuePairs() {
  GetApi().CreateKeyValuePairs(&p_);
}

inline KeyValuePairs::KeyValuePairs(const std::unordered_map<std::string, std::string>& kv_pairs) {
  GetApi().CreateKeyValuePairs(&p_);
  for (const auto& kv : kv_pairs) {
    GetApi().AddKeyValuePair(this->p_, kv.first.c_str(), kv.second.c_str());
  }
}

inline void KeyValuePairs::Add(const char* key, const char* value) {
  GetApi().AddKeyValuePair(this->p_, key, value);
}

inline void KeyValuePairs::Remove(const char* key) {
  GetApi().RemoveKeyValuePair(this->p_, key);
}

namespace detail {
template <typename T>
inline void* SyncStreamImpl<T>::GetHandle() {
  return GetApi().SyncStream_GetHandle(this->p_);
}
}  // namespace detail

namespace detail {
template <typename T>
inline OrtHardwareDeviceType HardwareDeviceImpl<T>::Type() const {
  return GetApi().HardwareDevice_Type(this->p_);
}

template <typename T>
inline uint32_t HardwareDeviceImpl<T>::VendorId() const {
  return GetApi().HardwareDevice_VendorId(this->p_);
}

template <typename T>
inline uint32_t HardwareDeviceImpl<T>::DeviceId() const {
  return GetApi().HardwareDevice_DeviceId(this->p_);
}

template <typename T>
inline const char* HardwareDeviceImpl<T>::Vendor() const {
  return GetApi().HardwareDevice_Vendor(this->p_);
}

template <typename T>
inline ConstKeyValuePairs HardwareDeviceImpl<T>::Metadata() const {
  return ConstKeyValuePairs{GetApi().HardwareDevice_Metadata(this->p_)};
}

template <typename T>
inline const char* EpDeviceImpl<T>::EpName() const {
  return GetApi().EpDevice_EpName(this->p_);
}

template <typename T>
inline const char* EpDeviceImpl<T>::EpVendor() const {
  return GetApi().EpDevice_EpVendor(this->p_);
}

template <typename T>
inline ConstKeyValuePairs EpDeviceImpl<T>::EpMetadata() const {
  return ConstKeyValuePairs(GetApi().EpDevice_EpMetadata(this->p_));
}

template <typename T>
inline ConstKeyValuePairs EpDeviceImpl<T>::EpOptions() const {
  return ConstKeyValuePairs(GetApi().EpDevice_EpOptions(this->p_));
}

template <typename T>
inline ConstHardwareDevice EpDeviceImpl<T>::Device() const {
  return ConstHardwareDevice(GetApi().EpDevice_Device(this->p_));
}

template <typename T>
inline ConstMemoryInfo EpDeviceImpl<T>::GetMemoryInfo(OrtDeviceMemoryType memory_type) const {
  const auto* mem_info = GetApi().EpDevice_MemoryInfo(this->p_, memory_type);
  return ConstMemoryInfo{mem_info};
}

template <typename T>
inline SyncStream EpDeviceImpl<T>::CreateSyncStream(ConstKeyValuePairs stream_options) const {
  OrtSyncStream* stream = nullptr;
  ThrowOnError(GetApi().CreateSyncStreamForEpDevice(this->p_, stream_options, &stream));
  return SyncStream{stream};
}
}  // namespace detail

inline EpDevice::EpDevice(OrtEpFactory& ep_factory, ConstHardwareDevice& hardware_device,
                          ConstKeyValuePairs ep_metadata, ConstKeyValuePairs ep_options) {
  ThrowOnError(GetEpApi().CreateEpDevice(&ep_factory, hardware_device, ep_metadata, ep_options, &p_));
}

inline Env::Env(OrtLoggingLevel logging_level, _In_ const char* logid) {
  ThrowOnError(GetApi().CreateEnv(logging_level, logid, &p_));
  if (strcmp(logid, "onnxruntime-node") == 0) {
    ThrowOnError(GetApi().SetLanguageProjection(p_, OrtLanguageProjection::ORT_PROJECTION_NODEJS));
  } else {
    ThrowOnError(GetApi().SetLanguageProjection(p_, OrtLanguageProjection::ORT_PROJECTION_CPLUSPLUS));
  }
}

inline Env::Env(OrtLoggingLevel logging_level, const char* logid, OrtLoggingFunction logging_function, void* logger_param) {
  ThrowOnError(GetApi().CreateEnvWithCustomLogger(logging_function, logger_param, logging_level, logid, &p_));
  if (strcmp(logid, "onnxruntime-node") == 0) {
    ThrowOnError(GetApi().SetLanguageProjection(p_, OrtLanguageProjection::ORT_PROJECTION_NODEJS));
  } else {
    ThrowOnError(GetApi().SetLanguageProjection(p_, OrtLanguageProjection::ORT_PROJECTION_CPLUSPLUS));
  }
}

inline Env::Env(const OrtThreadingOptions* tp_options, OrtLoggingLevel logging_level, _In_ const char* logid) {
  ThrowOnError(GetApi().CreateEnvWithGlobalThreadPools(logging_level, logid, tp_options, &p_));
  if (strcmp(logid, "onnxruntime-node") == 0) {
    ThrowOnError(GetApi().SetLanguageProjection(p_, OrtLanguageProjection::ORT_PROJECTION_NODEJS));
  } else {
    ThrowOnError(GetApi().SetLanguageProjection(p_, OrtLanguageProjection::ORT_PROJECTION_CPLUSPLUS));
  }
}

inline Env::Env(const OrtThreadingOptions* tp_options, OrtLoggingFunction logging_function, void* logger_param,
                OrtLoggingLevel logging_level, _In_ const char* logid) {
  ThrowOnError(GetApi().CreateEnvWithCustomLoggerAndGlobalThreadPools(logging_function, logger_param, logging_level, logid, tp_options, &p_));
  if (strcmp(logid, "onnxruntime-node") == 0) {
    ThrowOnError(GetApi().SetLanguageProjection(p_, OrtLanguageProjection::ORT_PROJECTION_NODEJS));
  } else {
    ThrowOnError(GetApi().SetLanguageProjection(p_, OrtLanguageProjection::ORT_PROJECTION_CPLUSPLUS));
  }
}

inline Env& Env::EnableTelemetryEvents() {
  ThrowOnError(GetApi().EnableTelemetryEvents(p_));
  return *this;
}

inline Env& Env::DisableTelemetryEvents() {
  ThrowOnError(GetApi().DisableTelemetryEvents(p_));
  return *this;
}

inline Env& Env::UpdateEnvWithCustomLogLevel(OrtLoggingLevel log_severity_level) {
  ThrowOnError(GetApi().UpdateEnvWithCustomLogLevel(p_, log_severity_level));
  return *this;
}

inline Env& Env::CreateAndRegisterAllocator(const OrtMemoryInfo* mem_info, const OrtArenaCfg* arena_cfg) {
  ThrowOnError(GetApi().CreateAndRegisterAllocator(p_, mem_info, arena_cfg));
  return *this;
}

inline Env& Env::CreateAndRegisterAllocatorV2(const std::string& provider_type, const OrtMemoryInfo* mem_info, const std::unordered_map<std::string, std::string>& options, const OrtArenaCfg* arena_cfg) {
  std::vector<const char*> keys, values;
  auto num_entries = options.size();
  if (num_entries > 0) {
    keys.reserve(num_entries);
    values.reserve(num_entries);
    for (const auto& entry : options) {
      keys.push_back(entry.first.c_str());
      values.push_back(entry.second.c_str());
    }
  }
  ThrowOnError(GetApi().CreateAndRegisterAllocatorV2(p_, provider_type.c_str(), mem_info, arena_cfg, keys.data(), values.data(), num_entries));
  return *this;
}

inline Env& Env::RegisterAllocator(OrtAllocator* allocator) {
  ThrowOnError(GetApi().RegisterAllocator(p_, allocator));
  return *this;
}

inline Env& Env::UnregisterAllocator(const OrtMemoryInfo* mem_info) {
  ThrowOnError(GetApi().UnregisterAllocator(p_, mem_info));
  return *this;
}

inline Env& Env::RegisterExecutionProviderLibrary(const char* registration_name,
                                                  const std::basic_string<ORTCHAR_T>& path) {
  ThrowOnError(GetApi().RegisterExecutionProviderLibrary(p_, registration_name, path.c_str()));
  return *this;
}

inline Env& Env::UnregisterExecutionProviderLibrary(const char* registration_name) {
  ThrowOnError(GetApi().UnregisterExecutionProviderLibrary(p_, registration_name));
  return *this;
}

inline std::vector<ConstEpDevice> Env::GetEpDevices() const {
  size_t num_devices = 0;
  const OrtEpDevice* const* device_ptrs = nullptr;
  ThrowOnError(GetApi().GetEpDevices(p_, &device_ptrs, &num_devices));

  std::vector<ConstEpDevice> devices;
  if (num_devices > 0) {
    devices.reserve(num_devices);
    for (size_t i = 0; i < num_devices; ++i) {
      devices.emplace_back(device_ptrs[i]);
    }
  }

  return devices;
}

inline Status Env::CopyTensors(const std::vector<Value>& src_tensors,
                               const std::vector<Value>& dst_tensors,
                               OrtSyncStream* stream) const {
  if (src_tensors.size() != dst_tensors.size()) {
    return Status("Source and destination tensor vectors must have the same size", ORT_INVALID_ARGUMENT);
  }
  if (src_tensors.empty()) {
    return Status(nullptr);
  }

  const OrtValue* const* src_tensors_ptr = reinterpret_cast<const OrtValue* const*>(src_tensors.data());
  OrtValue* const* dst_tensors_ptr = reinterpret_cast<OrtValue* const*>(dst_tensors.data());
  OrtStatus* status = GetApi().CopyTensors(p_, src_tensors_ptr, dst_tensors_ptr, stream, src_tensors.size());
  return Status(status);
}

inline UnownedAllocator Env::CreateSharedAllocator(const OrtEpDevice* ep_device, OrtDeviceMemoryType mem_type,
                                                   OrtAllocatorType allocator_type,
                                                   const OrtKeyValuePairs* allocator_options) {
  OrtAllocator* p;
  ThrowOnError(GetApi().CreateSharedAllocator(p_, ep_device, mem_type, allocator_type, allocator_options, &p));
  return UnownedAllocator{p};
}

inline UnownedAllocator Env::GetSharedAllocator(const OrtMemoryInfo* mem_info) {
  OrtAllocator* p;
  ThrowOnError(GetApi().GetSharedAllocator(p_, mem_info, &p));
  return UnownedAllocator{p};
}

inline void Env::ReleaseSharedAllocator(const OrtEpDevice* ep_device,
                                        OrtDeviceMemoryType mem_type) {
  ThrowOnError(GetApi().ReleaseSharedAllocator(p_, ep_device, mem_type));
}

inline CustomOpDomain::CustomOpDomain(const char* domain) {
  ThrowOnError(GetApi().CreateCustomOpDomain(domain, &p_));
}

inline void CustomOpDomain::Add(const OrtCustomOp* op) {
  ThrowOnError(GetApi().CustomOpDomain_Add(p_, op));
}

inline OrtCompiledModelCompatibility GetModelCompatibilityForEpDevices(
    const std::vector<ConstEpDevice>& ep_devices,
    const char* compatibility_info) {
  if (ep_devices.empty()) {
    ORT_CXX_API_THROW("ep_devices is empty", ORT_INVALID_ARGUMENT);
  }

  std::vector<const OrtEpDevice*> ptrs;
  ptrs.reserve(ep_devices.size());
  for (const auto& d : ep_devices) ptrs.push_back(d);

  OrtCompiledModelCompatibility status = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
  ThrowOnError(GetApi().GetModelCompatibilityForEpDevices(
      reinterpret_cast<const OrtEpDevice* const*>(ptrs.data()),
      ptrs.size(),
      compatibility_info,
      &status));
  return status;
}

inline LoraAdapter LoraAdapter::CreateLoraAdapter(const std::basic_string<ORTCHAR_T>& adapter_path,
                                                  OrtAllocator* allocator) {
  OrtLoraAdapter* p;
  ThrowOnError(GetApi().CreateLoraAdapter(adapter_path.c_str(), allocator, &p));
  return LoraAdapter{p};
}

inline LoraAdapter LoraAdapter::CreateLoraAdapterFromArray(const void* bytes, size_t num_bytes,
                                                           OrtAllocator* allocator) {
  OrtLoraAdapter* p;
  ThrowOnError(GetApi().CreateLoraAdapterFromArray(bytes, num_bytes, allocator, &p));
  return LoraAdapter{p};
}

inline RunOptions::RunOptions() {
  ThrowOnError(GetApi().CreateRunOptions(&p_));
}

inline RunOptions& RunOptions::SetRunLogVerbosityLevel(int level) {
  ThrowOnError(GetApi().RunOptionsSetRunLogVerbosityLevel(p_, level));
  return *this;
}

inline RunOptions& RunOptions::SetRunLogSeverityLevel(int level) {
  ThrowOnError(GetApi().RunOptionsSetRunLogSeverityLevel(p_, level));
  return *this;
}

inline int RunOptions::GetRunLogVerbosityLevel() const {
  int out;
  ThrowOnError(GetApi().RunOptionsGetRunLogVerbosityLevel(p_, &out));
  return out;
}

inline int RunOptions::GetRunLogSeverityLevel() const {
  int out;
  ThrowOnError(GetApi().RunOptionsGetRunLogSeverityLevel(p_, &out));
  return out;
}

inline RunOptions& RunOptions::SetRunTag(const char* run_tag) {
  ThrowOnError(GetApi().RunOptionsSetRunTag(p_, run_tag));
  return *this;
}

inline const char* RunOptions::GetRunTag() const {
  const char* out;
  ThrowOnError(GetApi().RunOptionsGetRunTag(p_, &out));
  return out;
}

inline RunOptions& RunOptions::AddConfigEntry(const char* config_key, const char* config_value) {
  ThrowOnError(GetApi().AddRunConfigEntry(p_, config_key, config_value));
  return *this;
}

inline const char* RunOptions::GetConfigEntry(const char* config_key) {
  return GetApi().GetRunConfigEntry(p_, config_key);
}

inline RunOptions& RunOptions::SetTerminate() {
  ThrowOnError(GetApi().RunOptionsSetTerminate(p_));
  return *this;
}

inline RunOptions& RunOptions::UnsetTerminate() {
  ThrowOnError(GetApi().RunOptionsUnsetTerminate(p_));
  return *this;
}

inline RunOptions& RunOptions::AddActiveLoraAdapter(const LoraAdapter& adapter) {
  ThrowOnError(GetApi().RunOptionsAddActiveLoraAdapter(p_, adapter));
  return *this;
}

inline ModelCompilationOptions::ModelCompilationOptions(const Env& env, const SessionOptions& session_options) {
  ThrowOnError(GetCompileApi().CreateModelCompilationOptionsFromSessionOptions(env, session_options, &this->p_));
}

inline ModelCompilationOptions::ModelCompilationOptions(const Env& env, ConstSessionOptions session_options) {
  ThrowOnError(GetCompileApi().CreateModelCompilationOptionsFromSessionOptions(env, session_options, &this->p_));
}

inline Status CompileModel(const Env& env, const ModelCompilationOptions& model_compilation_options) {
  return Ort::Status(GetCompileApi().CompileModel(env, model_compilation_options));
}

inline ModelCompilationOptions& ModelCompilationOptions::SetInputModelPath(
    const ORTCHAR_T* input_model_path) {
  Ort::ThrowOnError(GetCompileApi().ModelCompilationOptions_SetInputModelPath(this->p_, input_model_path));
  return *this;
}

inline ModelCompilationOptions& ModelCompilationOptions::SetInputModelFromBuffer(
    const void* input_model_data, size_t input_model_data_size) {
  Ort::ThrowOnError(GetCompileApi().ModelCompilationOptions_SetInputModelFromBuffer(this->p_, input_model_data,
                                                                                    input_model_data_size));
  return *this;
}

inline ModelCompilationOptions& ModelCompilationOptions::SetOutputModelPath(
    const ORTCHAR_T* output_model_path) {
  Ort::ThrowOnError(GetCompileApi().ModelCompilationOptions_SetOutputModelPath(this->p_, output_model_path));
  return *this;
}

inline ModelCompilationOptions& ModelCompilationOptions::SetEpContextBinaryInformation(
    const ORTCHAR_T* output_directory, const ORTCHAR_T* model_name) {
  Ort::ThrowOnError(GetCompileApi().ModelCompilationOptions_SetEpContextBinaryInformation(
      this->p_,
      output_directory,
      model_name));
  return *this;
}

inline ModelCompilationOptions& ModelCompilationOptions::SetOutputModelExternalInitializersFile(
    const ORTCHAR_T* file_path, size_t initializer_size_threshold) {
  Ort::ThrowOnError(GetCompileApi().ModelCompilationOptions_SetOutputModelExternalInitializersFile(
      this->p_,
      file_path,
      initializer_size_threshold));
  return *this;
}

inline ModelCompilationOptions&
ModelCompilationOptions::SetOutputModelGetInitializerLocationFunc(
    OrtGetInitializerLocationFunc get_initializer_location_func, void* state) {
  Ort::ThrowOnError(GetCompileApi().ModelCompilationOptions_SetOutputModelGetInitializerLocationFunc(
      this->p_,
      get_initializer_location_func,
      state));
  return *this;
}

inline ModelCompilationOptions& ModelCompilationOptions::SetOutputModelBuffer(
    OrtAllocator* allocator, void** output_model_buffer_ptr, size_t* output_model_buffer_size_ptr) {
  Ort::ThrowOnError(GetCompileApi().ModelCompilationOptions_SetOutputModelBuffer(this->p_, allocator,
                                                                                 output_model_buffer_ptr,
                                                                                 output_model_buffer_size_ptr));
  return *this;
}

inline ModelCompilationOptions& ModelCompilationOptions::SetOutputModelWriteFunc(OrtWriteBufferFunc write_func,
                                                                                 void* state) {
  Ort::ThrowOnError(GetCompileApi().ModelCompilationOptions_SetOutputModelWriteFunc(this->p_, write_func, state));
  return *this;
}

inline ModelCompilationOptions& ModelCompilationOptions::SetEpContextEmbedMode(
    bool embed_ep_context_in_model) {
  Ort::ThrowOnError(GetCompileApi().ModelCompilationOptions_SetEpContextEmbedMode(
      this->p_,
      embed_ep_context_in_model));
  return *this;
}

inline ModelCompilationOptions& ModelCompilationOptions::SetFlags(uint32_t flags) {
  Ort::ThrowOnError(GetCompileApi().ModelCompilationOptions_SetFlags(this->p_, flags));
  return *this;
}

inline ModelCompilationOptions& ModelCompilationOptions::SetGraphOptimizationLevel(
    GraphOptimizationLevel graph_optimization_level) {
  Ort::ThrowOnError(GetCompileApi().ModelCompilationOptions_SetGraphOptimizationLevel(this->p_,
                                                                                      graph_optimization_level));
  return *this;
}

namespace detail {

template <typename T>
inline Ort::SessionOptions ConstSessionOptionsImpl<T>::Clone() const {
  OrtSessionOptions* out;
  ThrowOnError(GetApi().CloneSessionOptions(this->p_, &out));
  return SessionOptions{out};
}

template <typename T>
inline std::string ConstSessionOptionsImpl<T>::GetConfigEntry(const char* config_key) const {
  size_t size = 0;
  // Feed nullptr for the data buffer to query the true size of the string value
  Ort::ThrowOnError(GetApi().GetSessionConfigEntry(this->p_, config_key, nullptr, &size));

  std::string out;
  out.resize(size);
  Ort::ThrowOnError(GetApi().GetSessionConfigEntry(this->p_, config_key, &out[0], &size));
  out.resize(size - 1);  // remove the terminating character '\0'

  return out;
}

template <typename T>
inline bool ConstSessionOptionsImpl<T>::HasConfigEntry(const char* config_key) const {
  int out = 0;
  Ort::ThrowOnError(GetApi().HasSessionConfigEntry(this->p_, config_key, &out));
  return static_cast<bool>(out);
}

template <typename T>
inline std::string ConstSessionOptionsImpl<T>::GetConfigEntryOrDefault(const char* config_key,
                                                                       const std::string& def) const {
  if (!this->HasConfigEntry(config_key)) {
    return def;
  }

  return this->GetConfigEntry(config_key);
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::SetIntraOpNumThreads(int intra_op_num_threads) {
  ThrowOnError(GetApi().SetIntraOpNumThreads(this->p_, intra_op_num_threads));
  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::SetInterOpNumThreads(int inter_op_num_threads) {
  ThrowOnError(GetApi().SetInterOpNumThreads(this->p_, inter_op_num_threads));
  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::SetGraphOptimizationLevel(GraphOptimizationLevel graph_optimization_level) {
  ThrowOnError(GetApi().SetSessionGraphOptimizationLevel(this->p_, graph_optimization_level));
  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::SetDeterministicCompute(bool value) {
  ThrowOnError(GetApi().SetDeterministicCompute(this->p_, value));
  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::SetOptimizedModelFilePath(const ORTCHAR_T* optimized_model_filepath) {
  ThrowOnError(GetApi().SetOptimizedModelFilePath(this->p_, optimized_model_filepath));
  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::EnableProfiling(const ORTCHAR_T* profile_file_prefix) {
  ThrowOnError(GetApi().EnableProfiling(this->p_, profile_file_prefix));
  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::DisableProfiling() {
  ThrowOnError(GetApi().DisableProfiling(this->p_));
  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::EnableOrtCustomOps() {
  ThrowOnError(GetApi().EnableOrtCustomOps(this->p_));
  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::EnableMemPattern() {
  ThrowOnError(GetApi().EnableMemPattern(this->p_));
  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::DisableMemPattern() {
  ThrowOnError(GetApi().DisableMemPattern(this->p_));
  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::EnableCpuMemArena() {
  ThrowOnError(GetApi().EnableCpuMemArena(this->p_));
  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::DisableCpuMemArena() {
  ThrowOnError(GetApi().DisableCpuMemArena(this->p_));
  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::SetExecutionMode(ExecutionMode execution_mode) {
  ThrowOnError(GetApi().SetSessionExecutionMode(this->p_, execution_mode));
  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::SetLoadCancellationFlag(bool value) {
  ThrowOnError(GetApi().SessionOptionsSetLoadCancellationFlag(this->p_, value));
  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::SetLogId(const char* logid) {
  ThrowOnError(GetApi().SetSessionLogId(this->p_, logid));
  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::SetLogSeverityLevel(int level) {
  ThrowOnError(GetApi().SetSessionLogSeverityLevel(this->p_, level));
  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::Add(OrtCustomOpDomain* custom_op_domain) {
  ThrowOnError(GetApi().AddCustomOpDomain(this->p_, custom_op_domain));
  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::AddConfigEntry(const char* config_key, const char* config_value) {
  ThrowOnError(GetApi().AddSessionConfigEntry(this->p_, config_key, config_value));
  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::AddInitializer(const char* name, const OrtValue* ort_val) {
  ThrowOnError(GetApi().AddInitializer(this->p_, name, ort_val));
  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::DisablePerSessionThreads() {
  ThrowOnError(GetApi().DisablePerSessionThreads(this->p_));
  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::AddExternalInitializers(const std::vector<std::string>& names,
                                                                             const std::vector<Value>& ort_values) {
  const size_t inputs_num = names.size();
  if (inputs_num != ort_values.size()) {
    ORT_CXX_API_THROW("Expecting names and ort_values to have the same length", ORT_INVALID_ARGUMENT);
  }
  std::vector<const char*> names_ptr;
  std::vector<const OrtValue*> ort_values_ptrs;
  names_ptr.reserve(inputs_num);
  ort_values_ptrs.reserve(inputs_num);
  for (size_t i = 0; i < inputs_num; ++i) {
    names_ptr.push_back(names[i].c_str());
    ort_values_ptrs.push_back(ort_values[i]);
  }
  ThrowOnError(GetApi().AddExternalInitializers(this->p_, names_ptr.data(), ort_values_ptrs.data(), inputs_num));
  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::AddExternalInitializersFromFilesInMemory(const std::vector<std::basic_string<ORTCHAR_T>>& file_names,
                                                                                              const std::vector<char*>& buffer_array,
                                                                                              const std::vector<size_t>& file_lengths) {
  const size_t inputs_num = file_names.size();
  if (inputs_num != buffer_array.size()) {
    ORT_CXX_API_THROW("Expecting names and buffer_array to have the same length", ORT_INVALID_ARGUMENT);
  }
  if (inputs_num != file_lengths.size()) {
    ORT_CXX_API_THROW("Expecting names and file_lengths to have the same length", ORT_INVALID_ARGUMENT);
  }
  std::vector<const ORTCHAR_T*> names_ptr;
  names_ptr.reserve(inputs_num);
  for (size_t i = 0; i < inputs_num; ++i) {
    names_ptr.push_back(file_names[i].c_str());
  }
  ThrowOnError(GetApi().AddExternalInitializersFromFilesInMemory(this->p_, names_ptr.data(), buffer_array.data(),
                                                                 file_lengths.data(), inputs_num));
  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::AppendExecutionProvider_CPU(int use_arena) {
  ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CPU(this->p_, use_arena));
  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::AppendExecutionProvider_CUDA(const OrtCUDAProviderOptions& provider_options) {
  ThrowOnError(GetApi().SessionOptionsAppendExecutionProvider_CUDA(this->p_, &provider_options));
  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::AppendExecutionProvider_CUDA_V2(const OrtCUDAProviderOptionsV2& provider_options) {
  ThrowOnError(GetApi().SessionOptionsAppendExecutionProvider_CUDA_V2(this->p_, &provider_options));
  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::AppendExecutionProvider_ROCM(const OrtROCMProviderOptions& provider_options) {
  ThrowOnError(GetApi().SessionOptionsAppendExecutionProvider_ROCM(this->p_, &provider_options));
  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::AppendExecutionProvider_TensorRT(const OrtTensorRTProviderOptions& provider_options) {
  ThrowOnError(GetApi().SessionOptionsAppendExecutionProvider_TensorRT(this->p_, &provider_options));
  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::AppendExecutionProvider_TensorRT_V2(const OrtTensorRTProviderOptionsV2& provider_options) {
  ThrowOnError(GetApi().SessionOptionsAppendExecutionProvider_TensorRT_V2(this->p_, &provider_options));
  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::AppendExecutionProvider_MIGraphX(const OrtMIGraphXProviderOptions& provider_options) {
  ThrowOnError(GetApi().SessionOptionsAppendExecutionProvider_MIGraphX(this->p_, &provider_options));
  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::AppendExecutionProvider_CANN(const OrtCANNProviderOptions& provider_options) {
  ThrowOnError(GetApi().SessionOptionsAppendExecutionProvider_CANN(this->p_, &provider_options));
  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::AppendExecutionProvider_Dnnl(const OrtDnnlProviderOptions& provider_options) {
  ThrowOnError(GetApi().SessionOptionsAppendExecutionProvider_Dnnl(this->p_, &provider_options));
  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::AppendExecutionProvider(
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

  ThrowOnError(GetApi().SessionOptionsAppendExecutionProvider(this->p_, provider_name.c_str(),
                                                              keys.data(), values.data(), num_entries));

  return *this;
}

namespace {
template <typename T>
void SessionOptionsAppendEP(detail::SessionOptionsImpl<T>& session_options,
                            Env& env, const std::vector<ConstEpDevice>& ep_devices,
                            const std::vector<const char*>& ep_options_keys,
                            const std::vector<const char*>& ep_options_values) {
  std::vector<const OrtEpDevice*> ep_devices_ptrs;
  ep_devices_ptrs.reserve(ep_devices.size());
  for (const auto& ep_device : ep_devices) {
    ep_devices_ptrs.push_back(ep_device);
  }

  ThrowOnError(GetApi().SessionOptionsAppendExecutionProvider_V2(
      session_options, env, ep_devices_ptrs.data(), ep_devices_ptrs.size(),
      ep_options_keys.data(), ep_options_values.data(), ep_options_keys.size()));
}
}  // namespace

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::AppendExecutionProvider_V2(
    Env& env, const std::vector<ConstEpDevice>& ep_devices, const KeyValuePairs& ep_options) {
  std::vector<const char*> ep_options_keys, ep_options_values;
  ep_options.GetKeyValuePairs(ep_options_keys, ep_options_values);

  SessionOptionsAppendEP(*this, env, ep_devices, ep_options_keys, ep_options_values);

  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::AppendExecutionProvider_V2(
    Env& env, const std::vector<ConstEpDevice>& ep_devices,
    const std::unordered_map<std::string, std::string>& ep_options) {
  std::vector<const char*> ep_options_keys, ep_options_values;
  ep_options_keys.reserve(ep_options.size());
  ep_options_values.reserve(ep_options.size());

  for (const auto& [key, value] : ep_options) {
    ep_options_keys.push_back(key.c_str());
    ep_options_values.push_back(value.c_str());
  }

  SessionOptionsAppendEP(*this, env, ep_devices, ep_options_keys, ep_options_values);

  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::SetEpSelectionPolicy(OrtExecutionProviderDevicePolicy policy) {
  ThrowOnError(GetApi().SessionOptionsSetEpSelectionPolicy(this->p_, policy));
  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::SetEpSelectionPolicy(EpSelectionDelegate delegate, void* state) {
  ThrowOnError(GetApi().SessionOptionsSetEpSelectionPolicyDelegate(this->p_, delegate, state));
  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::SetCustomCreateThreadFn(OrtCustomCreateThreadFn ort_custom_create_thread_fn) {
  ThrowOnError(GetApi().SessionOptionsSetCustomCreateThreadFn(this->p_, ort_custom_create_thread_fn));
  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::SetCustomThreadCreationOptions(void* ort_custom_thread_creation_options) {
  ThrowOnError(GetApi().SessionOptionsSetCustomThreadCreationOptions(this->p_, ort_custom_thread_creation_options));
  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::SetCustomJoinThreadFn(OrtCustomJoinThreadFn ort_custom_join_thread_fn) {
  ThrowOnError(GetApi().SessionOptionsSetCustomJoinThreadFn(this->p_, ort_custom_join_thread_fn));
  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::AppendExecutionProvider_OpenVINO(const OrtOpenVINOProviderOptions& provider_options) {
  ThrowOnError(GetApi().SessionOptionsAppendExecutionProvider_OpenVINO(this->p_, &provider_options));
  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::AppendExecutionProvider_OpenVINO_V2(const std::unordered_map<std::string, std::string>& provider_options) {
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

  ThrowOnError(GetApi().SessionOptionsAppendExecutionProvider_OpenVINO_V2(this->p_,
                                                                          keys.data(), values.data(), num_entries));

  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::AppendExecutionProvider_VitisAI(const std::unordered_map<std::string, std::string>& provider_options) {
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

  ThrowOnError(GetApi().SessionOptionsAppendExecutionProvider_VitisAI(this->p_, keys.data(), values.data(), num_entries));

  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::RegisterCustomOpsLibrary(const ORTCHAR_T* library_name,
                                                                              const CustomOpConfigs& custom_op_configs) {
  // Add custom op config entries before registering the custom op library. Otherwise, the config entries _may_ be ignored by
  // the custom op library.
  for (const auto& config_iter : custom_op_configs.GetFlattenedConfigs()) {
    AddConfigEntry(config_iter.first.c_str(), config_iter.second.c_str());
  }

  ThrowOnError(GetApi().RegisterCustomOpsLibrary_V2(this->p_, library_name));
  return *this;
}

template <typename T>
inline SessionOptionsImpl<T>& SessionOptionsImpl<T>::RegisterCustomOpsUsingFunction(const char* registration_function_name) {
  ThrowOnError(GetApi().RegisterCustomOpsUsingFunction(this->p_, registration_function_name));
  return *this;
}

/// Session
template <typename T>
inline size_t ConstSessionImpl<T>::GetInputCount() const {
  size_t out;
  ThrowOnError(GetApi().SessionGetInputCount(this->p_, &out));
  return out;
}

template <typename T>
inline size_t ConstSessionImpl<T>::GetOutputCount() const {
  size_t out;
  ThrowOnError(GetApi().SessionGetOutputCount(this->p_, &out));
  return out;
}

template <typename T>
inline size_t ConstSessionImpl<T>::GetOverridableInitializerCount() const {
  size_t out;
  ThrowOnError(GetApi().SessionGetOverridableInitializerCount(this->p_, &out));
  return out;
}

template <typename T>
inline std::vector<std::string> ConstSessionImpl<T>::GetInputNames() const {
  AllocatorWithDefaultOptions allocator;

  auto num_inputs = GetInputCount();
  std::vector<std::string> input_names;
  input_names.reserve(num_inputs);

  for (size_t i = 0; i < num_inputs; ++i) {
    char* name;
    ThrowOnError(GetApi().SessionGetInputName(this->p_, i, allocator, &name));
    input_names.emplace_back(name);
    allocator.Free(name);
  }

  return input_names;
}

template <typename T>
inline std::vector<std::string> ConstSessionImpl<T>::GetOutputNames() const {
  AllocatorWithDefaultOptions allocator;

  auto num_inputs = GetOutputCount();
  std::vector<std::string> output_names;
  output_names.reserve(num_inputs);

  for (size_t i = 0; i < num_inputs; ++i) {
    char* name;
    ThrowOnError(GetApi().SessionGetOutputName(this->p_, i, allocator, &name));
    output_names.emplace_back(name);
    allocator.Free(name);
  }

  return output_names;
}

template <typename T>
inline std::vector<std::string> ConstSessionImpl<T>::GetOverridableInitializerNames() const {
  AllocatorWithDefaultOptions allocator;

  auto num_initializers = GetOverridableInitializerCount();
  std::vector<std::string> initializer_names;
  initializer_names.reserve(num_initializers);

  for (size_t i = 0; i < num_initializers; ++i) {
    char* name;
    ThrowOnError(GetApi().SessionGetOverridableInitializerName(this->p_, i, allocator, &name));
    initializer_names.emplace_back(name);
  }

  return initializer_names;
}

template <typename T>
inline std::vector<ConstMemoryInfo> ConstSessionImpl<T>::GetMemoryInfoForInputs() const {
  static_assert(sizeof(ConstMemoryInfo) == sizeof(OrtMemoryInfo*),
                "ConstMemoryInfo must be compatible with OrtMemoryInfo*");

  auto num_inputs = GetInputCount();
  std::vector<ConstMemoryInfo> mem_infos;
  if (num_inputs > 0) {
    mem_infos.resize(num_inputs);

    ThrowOnError(GetApi().SessionGetMemoryInfoForInputs(this->p_,
                                                        reinterpret_cast<const OrtMemoryInfo**>(mem_infos.data()),
                                                        num_inputs));
  }

  return mem_infos;
}

template <typename T>
inline std::vector<ConstMemoryInfo> ConstSessionImpl<T>::GetMemoryInfoForOutputs() const {
  static_assert(sizeof(ConstMemoryInfo) == sizeof(OrtMemoryInfo*),
                "ConstMemoryInfo must be compatible with OrtMemoryInfo*");

  auto num_outputs = GetOutputCount();
  std::vector<ConstMemoryInfo> mem_infos;
  if (num_outputs > 0) {
    mem_infos.resize(num_outputs);

    ThrowOnError(GetApi().SessionGetMemoryInfoForOutputs(this->p_,
                                                         reinterpret_cast<const OrtMemoryInfo**>(mem_infos.data()),
                                                         num_outputs));
  }
  return mem_infos;
}

template <typename T>
inline AllocatedStringPtr ConstSessionImpl<T>::GetInputNameAllocated(size_t index, OrtAllocator* allocator) const {
  char* out;
  ThrowOnError(GetApi().SessionGetInputName(this->p_, index, allocator, &out));
  return AllocatedStringPtr(out, detail::AllocatedFree(allocator));
}

template <typename T>
inline AllocatedStringPtr ConstSessionImpl<T>::GetOutputNameAllocated(size_t index, OrtAllocator* allocator) const {
  char* out;
  ThrowOnError(GetApi().SessionGetOutputName(this->p_, index, allocator, &out));
  return AllocatedStringPtr(out, detail::AllocatedFree(allocator));
}

template <typename T>
inline AllocatedStringPtr ConstSessionImpl<T>::GetOverridableInitializerNameAllocated(size_t index, OrtAllocator* allocator) const {
  char* out;
  ThrowOnError(GetApi().SessionGetOverridableInitializerName(this->p_, index, allocator, &out));
  return AllocatedStringPtr(out, detail::AllocatedFree(allocator));
}

template <typename T>
inline std::vector<ConstEpDevice> ConstSessionImpl<T>::GetEpDeviceForInputs() const {
  auto num_inputs = GetInputCount();
  std::vector<ConstEpDevice> input_devices;
  if (num_inputs > 0) {
    input_devices.resize(num_inputs);
    ThrowOnError(GetApi().SessionGetEpDeviceForInputs(this->p_,
                                                      reinterpret_cast<const OrtEpDevice**>(input_devices.data()),
                                                      num_inputs));
  }
  return input_devices;
}

template <typename T>
inline uint64_t ConstSessionImpl<T>::GetProfilingStartTimeNs() const {
  uint64_t out;
  ThrowOnError(GetApi().SessionGetProfilingStartTimeNs(this->p_, &out));
  return out;
}

template <typename T>
inline ModelMetadata ConstSessionImpl<T>::GetModelMetadata() const {
  OrtModelMetadata* out;
  ThrowOnError(GetApi().SessionGetModelMetadata(this->p_, &out));
  return ModelMetadata{out};
}

template <typename T>
inline TypeInfo ConstSessionImpl<T>::GetInputTypeInfo(size_t index) const {
  OrtTypeInfo* out;
  ThrowOnError(GetApi().SessionGetInputTypeInfo(this->p_, index, &out));
  return TypeInfo{out};
}

template <typename T>
inline TypeInfo ConstSessionImpl<T>::GetOutputTypeInfo(size_t index) const {
  OrtTypeInfo* out;
  ThrowOnError(GetApi().SessionGetOutputTypeInfo(this->p_, index, &out));
  return TypeInfo{out};
}

template <typename T>
inline TypeInfo ConstSessionImpl<T>::GetOverridableInitializerTypeInfo(size_t index) const {
  OrtTypeInfo* out;
  ThrowOnError(GetApi().SessionGetOverridableInitializerTypeInfo(this->p_, index, &out));
  return TypeInfo{out};
}

#if !defined(ORT_MINIMAL_BUILD)
template <typename T>
inline int ConstSessionImpl<T>::GetOpset(const std::string& domain) const {
  int opset;
  ThrowOnError(GetModelEditorApi().SessionGetOpsetForDomain(this->p_, domain.c_str(), &opset));
  return opset;
}
#endif  // !defined(ORT_MINIMAL_BUILD)

template <typename T>
std::vector<ValueInfo> ConstSessionImpl<T>::GetInputs() const {
  const std::vector<std::string> input_names = GetInputNames();

  std::vector<ValueInfo> inputs;
  inputs.reserve(input_names.size());

  for (size_t i = 0; i < input_names.size(); ++i) {
    auto type_info = GetInputTypeInfo(i);
    inputs.emplace_back(ValueInfo{input_names[i], type_info.GetConst()});
  }

  return inputs;
}

template <typename T>
std::vector<ValueInfo> ConstSessionImpl<T>::GetOutputs() const {
  const std::vector<std::string> output_names = GetOutputNames();

  std::vector<ValueInfo> outputs;
  outputs.reserve(output_names.size());

  for (size_t i = 0; i < output_names.size(); ++i) {
    auto type_info = GetOutputTypeInfo(i);
    outputs.emplace_back(ValueInfo{output_names[i], type_info.GetConst()});
  }

  return outputs;
}

template <typename T>
inline std::vector<Value> SessionImpl<T>::Run(const RunOptions& run_options, const char* const* input_names, const Value* input_values, size_t input_count,
                                              const char* const* output_names, size_t output_count) {
  std::vector<Value> output_values;
  output_values.reserve(output_count);
  for (size_t i = 0; i < output_count; i++)
    output_values.emplace_back(nullptr);
  Run(run_options, input_names, input_values, input_count, output_names, output_values.data(), output_count);
  return output_values;
}

template <typename T>
inline void SessionImpl<T>::Run(const RunOptions& run_options, const char* const* input_names, const Value* input_values, size_t input_count,
                                const char* const* output_names, Value* output_values, size_t output_count) {
  static_assert(sizeof(Value) == sizeof(OrtValue*), "Value is really just an array of OrtValue* in memory, so we can reinterpret_cast safely");
  auto ort_input_values = reinterpret_cast<const OrtValue* const*>(input_values);
  auto ort_output_values = reinterpret_cast<OrtValue**>(output_values);
  ThrowOnError(GetApi().Run(this->p_, run_options, input_names, ort_input_values, input_count, output_names, output_count, ort_output_values));
}

template <typename T>
inline void SessionImpl<T>::Run(const RunOptions& run_options, const IoBinding& io_binding) {
  ThrowOnError(GetApi().RunWithBinding(this->p_, run_options, io_binding));
}

template <typename T>
inline void SessionImpl<T>::RunAsync(const RunOptions& run_options, const char* const* input_names, const Value* input_values, size_t input_count,
                                     const char* const* output_names, Value* output_values, size_t output_count, RunAsyncCallbackFn callback, void* user_data) {
  auto ort_input_values = reinterpret_cast<const OrtValue* const*>(input_values);
  auto ort_output_values = reinterpret_cast<OrtValue**>(output_values);
  ThrowOnError(GetApi().RunAsync(this->p_, run_options, input_names,
                                 ort_input_values, input_count, output_names, output_count,
                                 ort_output_values, callback, user_data));
}

template <typename T>
inline AllocatedStringPtr SessionImpl<T>::EndProfilingAllocated(OrtAllocator* allocator) {
  char* out = nullptr;
  ThrowOnError(GetApi().SessionEndProfiling(this->p_, allocator, &out));
  return AllocatedStringPtr(out, detail::AllocatedFree(allocator));
}

template <typename T>
inline void SessionImpl<T>::SetEpDynamicOptions(const char* const* keys, const char* const* values, size_t kv_len) {
  ThrowOnError(GetApi().SetEpDynamicOptions(this->p_, keys, values, kv_len));
}

#if !defined(ORT_MINIMAL_BUILD)
template <typename T>
inline void SessionImpl<T>::FinalizeModelEditorSession(const Model& model, const SessionOptions& options,
                                                       OrtPrepackedWeightsContainer* prepacked_weights_container) {
  ThrowOnError(GetModelEditorApi().ApplyModelToModelEditorSession(this->p_, model));
  ThrowOnError(GetModelEditorApi().FinalizeModelEditorSession(this->p_, options, prepacked_weights_container));
}
#endif  // #if !defined(ORT_MINIMAL_BUILD)

}  // namespace detail

inline SessionOptions::SessionOptions() {
  ThrowOnError(GetApi().CreateSessionOptions(&this->p_));
}

/// CustomOpConfigs
inline std::string detail::MakeCustomOpConfigEntryKey(const char* custom_op_name, const char* config) {
  std::string config_key = "custom_op.";

  config_key += custom_op_name;
  config_key += ".";
  config_key += config;

  return config_key;
}

inline CustomOpConfigs& CustomOpConfigs::AddConfig(const char* custom_op_name, const char* config_key, const char* config_value) {
  const std::string full_flat_key = detail::MakeCustomOpConfigEntryKey(custom_op_name, config_key);
  flat_configs_[full_flat_key] = config_value;
  return *this;
}

inline const std::unordered_map<std::string, std::string>& CustomOpConfigs::GetFlattenedConfigs() const {
  return flat_configs_;
}

inline Session::Session(const Env& env, const ORTCHAR_T* model_path, const SessionOptions& options) {
  ThrowOnError(GetApi().CreateSession(env, model_path, options, &this->p_));
}

inline Session::Session(const Env& env, const ORTCHAR_T* model_path, const SessionOptions& options,
                        OrtPrepackedWeightsContainer* prepacked_weights_container) {
  ThrowOnError(GetApi().CreateSessionWithPrepackedWeightsContainer(env, model_path, options, prepacked_weights_container, &this->p_));
}

inline Session::Session(const Env& env, const void* model_data, size_t model_data_length, const SessionOptions& options) {
  ThrowOnError(GetApi().CreateSessionFromArray(env, model_data, model_data_length, options, &this->p_));
}

inline Session::Session(const Env& env, const void* model_data, size_t model_data_length,
                        const SessionOptions& options, OrtPrepackedWeightsContainer* prepacked_weights_container) {
  ThrowOnError(GetApi().CreateSessionFromArrayWithPrepackedWeightsContainer(env, model_data, model_data_length, options,
                                                                            prepacked_weights_container, &this->p_));
}

#if !defined(ORT_MINIMAL_BUILD)
inline Session::Session(const Env& env, const Model& model, const SessionOptions& options) {
  ThrowOnError(GetModelEditorApi().CreateSessionFromModel(env, model, options, &this->p_));
}

// static
inline Session Session::CreateModelEditorSession(const Env& env, const ORTCHAR_T* model_path,
                                                 const SessionOptions& options) {
  OrtSession* session = nullptr;
  ThrowOnError(GetModelEditorApi().CreateModelEditorSession(env, model_path, options, &session));
  return Session(session);
}

// static
inline Session Session::CreateModelEditorSession(const Env& env, const void* model_data, size_t model_data_length,
                                                 const SessionOptions& options) {
  OrtSession* session = nullptr;
  ThrowOnError(GetModelEditorApi().CreateModelEditorSessionFromArray(env, model_data, model_data_length, options,
                                                                     &session));
  return Session(session);
}

void FinalizeModelEditorSession(const Model& model, const SessionOptions& options,
                                OrtPrepackedWeightsContainer* prepacked_weights_container);
#endif  // #if !defined(ORT_MINIMAL_BUILD)

inline AllocatedStringPtr ModelMetadata::GetProducerNameAllocated(OrtAllocator* allocator) const {
  char* out;
  ThrowOnError(GetApi().ModelMetadataGetProducerName(p_, allocator, &out));
  return AllocatedStringPtr(out, detail::AllocatedFree(allocator));
}

inline AllocatedStringPtr ModelMetadata::GetGraphNameAllocated(OrtAllocator* allocator) const {
  char* out;
  ThrowOnError(GetApi().ModelMetadataGetGraphName(p_, allocator, &out));
  return AllocatedStringPtr(out, detail::AllocatedFree(allocator));
}

inline AllocatedStringPtr ModelMetadata::GetDomainAllocated(OrtAllocator* allocator) const {
  char* out;
  ThrowOnError(GetApi().ModelMetadataGetDomain(p_, allocator, &out));
  return AllocatedStringPtr(out, detail::AllocatedFree(allocator));
}

inline AllocatedStringPtr Ort::ModelMetadata::GetDescriptionAllocated(OrtAllocator* allocator) const {
  char* out;
  ThrowOnError(GetApi().ModelMetadataGetDescription(p_, allocator, &out));
  return AllocatedStringPtr(out, detail::AllocatedFree(allocator));
}

inline AllocatedStringPtr ModelMetadata::GetGraphDescriptionAllocated(OrtAllocator* allocator) const {
  char* out;
  ThrowOnError(GetApi().ModelMetadataGetGraphDescription(p_, allocator, &out));
  return AllocatedStringPtr(out, detail::AllocatedFree(allocator));
}

inline AllocatedStringPtr ModelMetadata::LookupCustomMetadataMapAllocated(const char* key, OrtAllocator* allocator) const {
  char* out;
  ThrowOnError(GetApi().ModelMetadataLookupCustomMetadataMap(p_, allocator, key, &out));
  return AllocatedStringPtr(out, detail::AllocatedFree(allocator));
}

inline std::vector<AllocatedStringPtr> ModelMetadata::GetCustomMetadataMapKeysAllocated(OrtAllocator* allocator) const {
  auto deletor = detail::AllocatedFree(allocator);
  std::vector<AllocatedStringPtr> result;

  char** out = nullptr;
  int64_t num_keys = 0;
  ThrowOnError(GetApi().ModelMetadataGetCustomMetadataMapKeys(p_, allocator, &out, &num_keys));
  if (num_keys <= 0) {
    return result;
  }

  // array of pointers will be freed
  std::unique_ptr<void, decltype(deletor)> array_guard(out, deletor);
  // reserve may throw
  auto strings_deletor = [&deletor, num_keys](char** out) { for(int64_t i = 0; i < num_keys; ++i) deletor(out[i]); };
  std::unique_ptr<char*, decltype(strings_deletor)> strings_guard(out, strings_deletor);
  result.reserve(static_cast<size_t>(num_keys));
  strings_guard.release();
  for (int64_t i = 0; i < num_keys; ++i) {
    result.push_back(AllocatedStringPtr(out[i], deletor));
  }

  return result;
}

inline int64_t ModelMetadata::GetVersion() const {
  int64_t out;
  ThrowOnError(GetApi().ModelMetadataGetVersion(p_, &out));
  return out;
}

inline TensorTypeAndShapeInfo::TensorTypeAndShapeInfo(ONNXTensorElementDataType element_type,
                                                      const std::vector<int64_t>& dims,
                                                      const std::vector<std::string>* symbolic_dims) {
  ThrowOnError(GetApi().CreateTensorTypeAndShapeInfo(&p_));
  ThrowOnError(GetApi().SetTensorElementType(p_, element_type));
  ThrowOnError(GetApi().SetDimensions(p_, dims.data(), dims.size()));

  if (symbolic_dims) {
    std::vector<const char*> symbolic_dims_cstr;
    symbolic_dims_cstr.reserve(symbolic_dims->size());
    std::transform(symbolic_dims->begin(), symbolic_dims->end(), std::back_inserter(symbolic_dims_cstr),
                   [](const std::string& s) { return s.c_str(); });
    ThrowOnError(GetApi().SetSymbolicDimensions(p_, symbolic_dims_cstr.data(), symbolic_dims_cstr.size()));
  }
}

#if !defined(ORT_MINIMAL_BUILD)
// static
inline TypeInfo TypeInfo::CreateTensorInfo(ConstTensorTypeAndShapeInfo tensor_type_and_shape_info) {
  OrtTypeInfo* output = nullptr;
  ThrowOnError(GetModelEditorApi().CreateTensorTypeInfo(tensor_type_and_shape_info, &output));
  return TypeInfo{output};
}

// static
inline TypeInfo TypeInfo::CreateSparseTensorInfo(ConstTensorTypeAndShapeInfo sparse_tensor_type_and_shape_info) {
  OrtTypeInfo* output = nullptr;
  ThrowOnError(GetModelEditorApi().CreateSparseTensorTypeInfo(sparse_tensor_type_and_shape_info, &output));
  return TypeInfo{output};
}

// static
inline TypeInfo TypeInfo::CreateSequenceTypeInfo(ConstTypeInfo sequence_type) {
  OrtTypeInfo* output;
  ThrowOnError(GetModelEditorApi().CreateSequenceTypeInfo(sequence_type, &output));
  return TypeInfo{output};
}

// static
inline TypeInfo TypeInfo::CreateMapTypeInfo(ONNXTensorElementDataType key_type, ConstTypeInfo value_type) {
  OrtTypeInfo* output;
  ThrowOnError(GetModelEditorApi().CreateMapTypeInfo(key_type, value_type, &output));
  return TypeInfo{output};
}

// static
inline TypeInfo TypeInfo::CreateOptionalTypeInfo(ConstTypeInfo contained_type) {
  OrtTypeInfo* output;
  ThrowOnError(GetModelEditorApi().CreateOptionalTypeInfo(contained_type, &output));
  return TypeInfo{output};
}
#endif  // #if !defined(ORT_MINIMAL_BUILD)

namespace detail {

template <typename T>
inline ONNXTensorElementDataType TensorTypeAndShapeInfoImpl<T>::GetElementType() const {
  ONNXTensorElementDataType out;
  ThrowOnError(GetApi().GetTensorElementType(this->p_, &out));
  return out;
}

template <typename T>
inline size_t TensorTypeAndShapeInfoImpl<T>::GetElementCount() const {
  size_t out;
  ThrowOnError(GetApi().GetTensorShapeElementCount(this->p_, &out));
  return static_cast<size_t>(out);
}

template <typename T>
inline size_t TensorTypeAndShapeInfoImpl<T>::GetDimensionsCount() const {
  size_t out;
  ThrowOnError(GetApi().GetDimensionsCount(this->p_, &out));
  return out;
}

template <typename T>
inline void TensorTypeAndShapeInfoImpl<T>::GetDimensions(int64_t* values, size_t values_count) const {
  ThrowOnError(GetApi().GetDimensions(this->p_, values, values_count));
}

template <typename T>
inline void TensorTypeAndShapeInfoImpl<T>::GetSymbolicDimensions(const char** values, size_t values_count) const {
  ThrowOnError(GetApi().GetSymbolicDimensions(this->p_, values, values_count));
}

template <typename T>
inline std::vector<const char*> TensorTypeAndShapeInfoImpl<T>::GetSymbolicDimensions() const {
  std::vector<const char*> out(GetDimensionsCount(), nullptr);
  ThrowOnError(GetApi().GetSymbolicDimensions(this->p_, out.data(), out.size()));
  return out;
}

template <typename T>
inline std::vector<int64_t> TensorTypeAndShapeInfoImpl<T>::GetShape() const {
  std::vector<int64_t> out(GetDimensionsCount(), -1);
  ThrowOnError(GetApi().GetDimensions(this->p_, out.data(), out.size()));
  return out;
}

template <typename T>
inline ConstTensorTypeAndShapeInfo TypeInfoImpl<T>::GetTensorTypeAndShapeInfo() const {
  const OrtTensorTypeAndShapeInfo* out;
  ThrowOnError(GetApi().CastTypeInfoToTensorInfo(this->p_, &out));
  return ConstTensorTypeAndShapeInfo{out};
}

template <typename T>
inline ConstSequenceTypeInfo TypeInfoImpl<T>::GetSequenceTypeInfo() const {
  const OrtSequenceTypeInfo* out;
  ThrowOnError(GetApi().CastTypeInfoToSequenceTypeInfo(this->p_, &out));
  return ConstSequenceTypeInfo{out};
}

template <typename T>
inline ConstMapTypeInfo TypeInfoImpl<T>::GetMapTypeInfo() const {
  const OrtMapTypeInfo* out;
  ThrowOnError(GetApi().CastTypeInfoToMapTypeInfo(this->p_, &out));
  return ConstMapTypeInfo{out};
}

template <typename T>
inline ONNXType TypeInfoImpl<T>::GetONNXType() const {
  ONNXType out;
  ThrowOnError(GetApi().GetOnnxTypeFromTypeInfo(this->p_, &out));
  return out;
}

template <typename T>
inline TypeInfo SequenceTypeInfoImpl<T>::GetSequenceElementType() const {
  OrtTypeInfo* output;
  ThrowOnError(GetApi().GetSequenceElementType(this->p_, &output));
  return TypeInfo{output};
}

template <typename T>
inline TypeInfo OptionalTypeInfoImpl<T>::GetOptionalElementType() const {
  OrtTypeInfo* info;
  ThrowOnError(GetApi().GetOptionalContainedTypeInfo(this->p_, &info));
  return TypeInfo{info};
}

template <typename T>
inline ONNXTensorElementDataType MapTypeInfoImpl<T>::GetMapKeyType() const {
  ONNXTensorElementDataType out;
  ThrowOnError(GetApi().GetMapKeyType(this->p_, &out));
  return out;
}

template <typename T>
inline TypeInfo MapTypeInfoImpl<T>::GetMapValueType() const {
  OrtTypeInfo* output;
  ThrowOnError(GetApi().GetMapValueType(this->p_, &output));
  return TypeInfo{output};
}

template <typename T>
inline ConstOptionalTypeInfo TypeInfoImpl<T>::GetOptionalTypeInfo() const {
  const OrtOptionalTypeInfo* info;
  ThrowOnError(GetApi().CastTypeInfoToOptionalTypeInfo(this->p_, &info));
  return ConstOptionalTypeInfo{info};
}

}  // namespace detail

namespace detail {

template <typename T>
template <typename R>
inline void ConstValueImpl<T>::GetOpaqueData(const char* domain, const char* type_name, R& out) const {
  ThrowOnError(GetApi().GetOpaqueValue(domain, type_name, this->p_, &out, sizeof(R)));
}

template <typename T>
inline bool ConstValueImpl<T>::IsTensor() const {
  int out;
  ThrowOnError(GetApi().IsTensor(this->p_, &out));
  return out != 0;
}

template <typename T>
inline bool ConstValueImpl<T>::HasValue() const {
  int out;
  ThrowOnError(GetApi().HasValue(this->p_, &out));
  return out != 0;
}

template <typename T>
inline size_t ConstValueImpl<T>::GetCount() const {
  size_t out;
  ThrowOnError(GetApi().GetValueCount(this->p_, &out));
  return out;
}

template <typename T>
inline Value ConstValueImpl<T>::GetValue(int index, OrtAllocator* allocator) const {
  OrtValue* out;
  ThrowOnError(GetApi().GetValue(this->p_, index, allocator, &out));
  return Value{out};
}

template <typename T>
inline size_t ConstValueImpl<T>::GetStringTensorDataLength() const {
  size_t out;
  ThrowOnError(GetApi().GetStringTensorDataLength(this->p_, &out));
  return out;
}

template <typename T>
inline size_t ConstValueImpl<T>::GetStringTensorElementLength(size_t element_index) const {
  size_t out;
  ThrowOnError(GetApi().GetStringTensorElementLength(this->p_, element_index, &out));
  return out;
}

template <typename T>
inline size_t ConstValueImpl<T>::GetTensorSizeInBytes() const {
  size_t out;
  ThrowOnError(GetApi().GetTensorSizeInBytes(this->p_, &out));
  return out;
}

template <typename T>
template <typename R>
inline const R* ConstValueImpl<T>::GetTensorData() const {
  const R* out;
  ThrowOnError(GetApi().GetTensorData(this->p_, reinterpret_cast<const void**>(&out)));
  return out;
}

template <typename T>
inline const void* ConstValueImpl<T>::GetTensorRawData() const {
  const void* out;
  ThrowOnError(GetApi().GetTensorData(this->p_, &out));
  return out;
}

template <typename T>
inline TypeInfo ConstValueImpl<T>::GetTypeInfo() const {
  OrtTypeInfo* output;
  ThrowOnError(GetApi().GetTypeInfo(this->p_, &output));
  return TypeInfo{output};
}

template <typename T>
inline TensorTypeAndShapeInfo ConstValueImpl<T>::GetTensorTypeAndShapeInfo() const {
  OrtTensorTypeAndShapeInfo* output;
  ThrowOnError(GetApi().GetTensorTypeAndShape(this->p_, &output));
  return TensorTypeAndShapeInfo{output};
}

template <typename T>
inline ConstMemoryInfo ConstValueImpl<T>::GetTensorMemoryInfo() const {
  const OrtMemoryInfo* mem_info;
  ThrowOnError(GetApi().GetTensorMemoryInfo(this->p_, &mem_info));
  return ConstMemoryInfo(mem_info);
}

template <typename T>
inline void ConstValueImpl<T>::GetStringTensorElement(size_t buffer_length, size_t element_index, void* buffer) const {
  ThrowOnError(GetApi().GetStringTensorElement(this->p_, buffer_length, element_index, buffer));
}

template <typename T>
inline std::string ConstValueImpl<T>::GetStringTensorElement(size_t element_index) const {
  size_t buffer_length;
  ThrowOnError(GetApi().GetStringTensorElementLength(this->p_, element_index, &buffer_length));

  std::string s;
  s.resize(buffer_length);
  ThrowOnError(GetApi().GetStringTensorElement(this->p_, buffer_length, element_index, &s[0]));
  return s;
}

template <typename T>
inline void ConstValueImpl<T>::GetStringTensorContent(void* buffer, size_t buffer_length, size_t* offsets, size_t offsets_count) const {
  ThrowOnError(GetApi().GetStringTensorContent(this->p_, buffer, buffer_length, offsets, offsets_count));
}

#if !defined(DISABLE_SPARSE_TENSORS)
template <typename T>
inline OrtSparseFormat ConstValueImpl<T>::GetSparseFormat() const {
  OrtSparseFormat format;
  ThrowOnError(GetApi().GetSparseTensorFormat(this->p_, &format));
  return format;
}

template <typename T>
inline TensorTypeAndShapeInfo ConstValueImpl<T>::GetSparseTensorValuesTypeAndShapeInfo() const {
  OrtTensorTypeAndShapeInfo* output;
  ThrowOnError(GetApi().GetSparseTensorValuesTypeAndShape(this->p_, &output));
  return TensorTypeAndShapeInfo{output};
}

template <typename T>
inline TensorTypeAndShapeInfo ConstValueImpl<T>::GetSparseTensorIndicesTypeShapeInfo(OrtSparseIndicesFormat indices_format) const {
  OrtTensorTypeAndShapeInfo* output;
  ThrowOnError(GetApi().GetSparseTensorIndicesTypeShape(this->p_, indices_format, &output));
  return TensorTypeAndShapeInfo{output};
}

template <typename T>
template <typename R>
inline const R* ConstValueImpl<T>::GetSparseTensorIndicesData(OrtSparseIndicesFormat indices_format, size_t& num_indices) const {
  const void* out;
  ThrowOnError(GetApi().GetSparseTensorIndices(this->p_, indices_format, &num_indices, &out));
  return reinterpret_cast<const R*>(out);
}

template <typename T>
inline bool ConstValueImpl<T>::IsSparseTensor() const {
  int out;
  ThrowOnError(GetApi().IsSparseTensor(this->p_, &out));
  return out != 0;
}

template <typename T>
template <typename R>
inline const R* ConstValueImpl<T>::GetSparseTensorValues() const {
  const void* out;
  ThrowOnError(GetApi().GetSparseTensorValues(this->p_, &out));
  return reinterpret_cast<const R*>(out);
}

#endif

template <typename T>
void ValueImpl<T>::FillStringTensor(const char* const* s, size_t s_len) {
  ThrowOnError(GetApi().FillStringTensor(this->p_, s, s_len));
}

template <typename T>
void ValueImpl<T>::FillStringTensorElement(const char* s, size_t index) {
  ThrowOnError(GetApi().FillStringTensorElement(this->p_, s, index));
}

template <typename T>
inline char* ValueImpl<T>::GetResizedStringTensorElementBuffer(size_t index, size_t buffer_length) {
  char* result;
  ThrowOnError(GetApi().GetResizedStringTensorElementBuffer(this->p_, index, buffer_length, &result));
  return result;
}

template <typename T>
void* ValueImpl<T>::GetTensorMutableRawData() {
  void* out;
  ThrowOnError(GetApi().GetTensorMutableData(this->p_, &out));
  return out;
}

template <typename T>
template <typename R>
R* ValueImpl<T>::GetTensorMutableData() {
  R* out;
  ThrowOnError(GetApi().GetTensorMutableData(this->p_, (void**)&out));
  return out;
}

template <typename T>
template <typename R>
R& ValueImpl<T>::At(const std::vector<int64_t>& location) {
  static_assert(!std::is_same<T, std::string>::value, "this api does not support std::string");
  R* out;
  ThrowOnError(GetApi().TensorAt(this->p_, location.data(), location.size(), (void**)&out));
  return *out;
}

#if !defined(DISABLE_SPARSE_TENSORS)
template <typename T>
void ValueImpl<T>::UseCooIndices(int64_t* indices_data, size_t indices_num) {
  ThrowOnError(GetApi().UseCooIndices(this->p_, indices_data, indices_num));
}

template <typename T>
void ValueImpl<T>::UseCsrIndices(int64_t* inner_data, size_t inner_num, int64_t* outer_data, size_t outer_num) {
  ThrowOnError(GetApi().UseCsrIndices(this->p_, inner_data, inner_num, outer_data, outer_num));
}

template <typename T>
void ValueImpl<T>::UseBlockSparseIndices(const Shape& indices_shape, int32_t* indices_data) {
  ThrowOnError(GetApi().UseBlockSparseIndices(this->p_, indices_shape.shape, indices_shape.shape_len, indices_data));
}

template <typename T>
void ValueImpl<T>::FillSparseTensorCoo(const OrtMemoryInfo* mem_info, const OrtSparseValuesParam& values_param,
                                       const int64_t* indices_data, size_t indices_num) {
  ThrowOnError(GetApi().FillSparseTensorCoo(this->p_, mem_info, values_param.values_shape,
                                            values_param.values_shape_len, values_param.data.p_data,
                                            indices_data, indices_num));
}

template <typename T>
void ValueImpl<T>::FillSparseTensorCsr(const OrtMemoryInfo* data_mem_info,
                                       const OrtSparseValuesParam& values,
                                       const int64_t* inner_indices_data, size_t inner_indices_num,
                                       const int64_t* outer_indices_data, size_t outer_indices_num) {
  ThrowOnError(GetApi().FillSparseTensorCsr(this->p_, data_mem_info, values.values_shape, values.values_shape_len, values.data.p_data,
                                            inner_indices_data, inner_indices_num,
                                            outer_indices_data, outer_indices_num));
}

template <typename T>
void ValueImpl<T>::FillSparseTensorBlockSparse(const OrtMemoryInfo* data_mem_info,
                                               const OrtSparseValuesParam& values,
                                               const Shape& indices_shape,
                                               const int32_t* indices_data) {
  ThrowOnError(GetApi().FillSparseTensorBlockSparse(this->p_, data_mem_info, values.values_shape, values.values_shape_len, values.data.p_data,
                                                    indices_shape.shape, indices_shape.shape_len,
                                                    indices_data));
}

#endif  // !defined(DISABLE_SPARSE_TENSORS)

}  // namespace detail

template <typename T>
inline Value Value::CreateTensor(const OrtMemoryInfo* info, T* p_data, size_t p_data_element_count,
                                 const int64_t* shape, size_t shape_len) {
  return CreateTensor(info, p_data, p_data_element_count * sizeof(T), shape, shape_len, TypeToTensorType<T>::type);
}

inline Value Value::CreateTensor(const OrtMemoryInfo* info, void* p_data, size_t p_data_byte_count,
                                 const int64_t* shape, size_t shape_len,
                                 ONNXTensorElementDataType type) {
  OrtValue* out;
  ThrowOnError(GetApi().CreateTensorWithDataAsOrtValue(info, p_data, p_data_byte_count, shape, shape_len, type, &out));
  return Value{out};
}

inline Value Value::CreateTensor(OrtAllocator* deleter, void* p_data, size_t p_data_byte_count,
                                 const int64_t* shape, size_t shape_len,
                                 ONNXTensorElementDataType type) {
  OrtValue* out;
  ThrowOnError(GetApi().CreateTensorWithDataAndDeleterAsOrtValue(deleter, p_data, p_data_byte_count,
                                                                 shape, shape_len, type, &out));
  return Value{out};
}

template <typename T>
inline Value Value::CreateTensor(OrtAllocator* allocator, const int64_t* shape, size_t shape_len) {
  return CreateTensor(allocator, shape, shape_len, TypeToTensorType<T>::type);
}

inline Value Value::CreateTensor(OrtAllocator* allocator, const int64_t* shape, size_t shape_len,
                                 ONNXTensorElementDataType type) {
  OrtValue* out;
  ThrowOnError(GetApi().CreateTensorAsOrtValue(allocator, shape, shape_len, type, &out));
  return Value{out};
}

#if !defined(DISABLE_SPARSE_TENSORS)

template <typename T>
inline Value Value::CreateSparseTensor(const OrtMemoryInfo* info, T* p_data, const Shape& dense_shape,
                                       const Shape& values_shape) {
  return CreateSparseTensor(info, p_data, dense_shape, values_shape, TypeToTensorType<T>::type);
}

inline Value Value::CreateSparseTensor(const OrtMemoryInfo* info, void* p_data, const Shape& dense_shape,
                                       const Shape& values_shape, ONNXTensorElementDataType type) {
  OrtValue* out;
  ThrowOnError(GetApi().CreateSparseTensorWithValuesAsOrtValue(info, p_data, dense_shape.shape, dense_shape.shape_len,
                                                               values_shape.shape, values_shape.shape_len, type,
                                                               &out));
  return Value{out};
}

template <typename T>
inline Value Value::CreateSparseTensor(OrtAllocator* allocator, const Shape& dense_shape) {
  return CreateSparseTensor(allocator, dense_shape, TypeToTensorType<T>::type);
}

inline Value Value::CreateSparseTensor(OrtAllocator* allocator, const Shape& dense_shape,
                                       ONNXTensorElementDataType type) {
  OrtValue* out;
  ThrowOnError(GetApi().CreateSparseTensorAsOrtValue(allocator, dense_shape.shape, dense_shape.shape_len, type, &out));
  return Value{out};
}
#endif  // !defined(DISABLE_SPARSE_TENSORS)

inline Value Value::CreateMap(const Value& keys, const Value& values) {
  OrtValue* out;
  const OrtValue* inputs[2] = {keys, values};
  ThrowOnError(GetApi().CreateValue(inputs, 2, ONNX_TYPE_MAP, &out));
  return Value{out};
}

inline Value Value::CreateSequence(const std::vector<Value>& values) {
  OrtValue* out;
  std::vector<const OrtValue*> values_ort{values.data(), values.data() + values.size()};
  ThrowOnError(GetApi().CreateValue(values_ort.data(), values_ort.size(), ONNX_TYPE_SEQUENCE, &out));
  return Value{out};
}

template <typename T>
inline Value Value::CreateOpaque(const char* domain, const char* type_name, const T& data_container) {
  OrtValue* out;
  ThrowOnError(GetApi().CreateOpaqueValue(domain, type_name, &data_container, sizeof(T), &out));
  return Value{out};
}

//
// Custom OP Inlines
//
inline Logger::Logger(const OrtLogger* logger) : logger_(logger) {
  Ort::ThrowOnError(GetApi().Logger_GetLoggingSeverityLevel(this->logger_, &this->cached_severity_level_));
}

inline OrtLoggingLevel Logger::GetLoggingSeverityLevel() const noexcept {
  return cached_severity_level_;
}

inline Status Logger::LogMessage(OrtLoggingLevel log_severity_level, const ORTCHAR_T* file_path, int line_number,
                                 const char* func_name, const char* message) const noexcept {
  OrtStatus* status = GetApi().Logger_LogMessage(logger_, log_severity_level, message, file_path, line_number,
                                                 func_name);
  return Status{status};
}

// Disable warnings about the format string not being a literal (-Wformat-nonliteral and -Wformat-security)
// for gcc and clang. The alternative is to use actual C-style variadic parameters and apply
// __attribute__(format(printf...)), which does not work with variadic templates.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
#pragma GCC diagnostic ignored "-Wformat-security"
#elif defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-nonliteral"
#pragma clang diagnostic ignored "-Wformat-security"
#endif
template <typename... Args>
inline Status Logger::LogFormattedMessage(OrtLoggingLevel log_severity_level, const ORTCHAR_T* file_path,
                                          int line_number, const char* func_name, const char* format,
                                          Args&&... args) const noexcept {
  int msg_len = std::snprintf(nullptr, 0U, format, std::forward<Args>(args)...);

  if (msg_len < 0) {  // Formatting error
    return Status("Failed to log message due to formatting error", OrtErrorCode::ORT_FAIL);
  }

  OrtStatus* status = nullptr;
  const size_t buffer_size = static_cast<size_t>(msg_len) + 1U;

  constexpr size_t kStackBufferSize = 1024;

  if (buffer_size < kStackBufferSize) {
    char buffer[kStackBufferSize];
    snprintf(buffer, kStackBufferSize, format, std::forward<Args>(args)...);
    status = GetApi().Logger_LogMessage(logger_, log_severity_level, buffer, file_path, line_number, func_name);
  } else {
    // std::make_unique is only supported starting at C++14.
#if (__cplusplus >= 201402L) || (_MSC_VER >= 1900)
    auto buffer = std::make_unique<char[]>(buffer_size);
#else
    std::unique_ptr<char[]> buffer(new char[buffer_size]);
#endif
    std::snprintf(buffer.get(), buffer_size, format, std::forward<Args>(args)...);
    status = GetApi().Logger_LogMessage(logger_, log_severity_level, buffer.get(), file_path, line_number, func_name);
  }

  return Status{status};
}
// Re-enable -Wformat-nonliteral and -Wformat-security
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(__clang__)
#pragma clang diagnostic pop
#endif

inline KernelContext::KernelContext(OrtKernelContext* context) : ctx_(context) {
}

inline size_t KernelContext::GetInputCount() const {
  size_t out = 0;
  Ort::ThrowOnError(GetApi().KernelContext_GetInputCount(ctx_, &out));
  return out;
}

inline size_t KernelContext::GetOutputCount() const {
  size_t out = 0;
  Ort::ThrowOnError(GetApi().KernelContext_GetOutputCount(ctx_, &out));
  return out;
}

inline ConstValue KernelContext::GetInput(size_t index) const {
  const OrtValue* out = nullptr;
  Ort::ThrowOnError(GetApi().KernelContext_GetInput(ctx_, index, &out));
  return ConstValue{out};
}

inline UnownedValue KernelContext::GetOutput(size_t index, const int64_t* dim_values, size_t dim_count) const {
  OrtValue* out = nullptr;
  Ort::ThrowOnError(GetApi().KernelContext_GetOutput(ctx_, index, dim_values, dim_count, &out));
  return UnownedValue(out);
}

inline UnownedValue KernelContext::GetOutput(size_t index, const std::vector<int64_t>& dims) const {
  OrtValue* out = nullptr;
  Ort::ThrowOnError(GetApi().KernelContext_GetOutput(ctx_, index, dims.data(), dims.size(), &out));
  return UnownedValue(out);
}

inline void* KernelContext::GetGPUComputeStream() const {
  void* out = nullptr;
  Ort::ThrowOnError(GetApi().KernelContext_GetGPUComputeStream(ctx_, &out));
  return out;
}

inline OrtAllocator* KernelContext::GetAllocator(const OrtMemoryInfo& memory_info) const {
  OrtAllocator* out = nullptr;
  Ort::ThrowOnError(GetApi().KernelContext_GetAllocator(ctx_, &memory_info, &out));
  return out;
}

inline Logger KernelContext::GetLogger() const {
  const OrtLogger* out = nullptr;
  ThrowOnError(GetApi().KernelContext_GetLogger(this->ctx_, &out));
  return Logger{out};
}

inline void KernelContext::ParallelFor(void (*fn)(void*, size_t), size_t total, size_t num_batch, void* usr_data) const {
  ThrowOnError(GetApi().KernelContext_ParallelFor(ctx_, fn, total, num_batch, usr_data));
}

namespace detail {

template <typename T>
constexpr OrtOpAttrType TypeToAttrType();

template <>
inline constexpr OrtOpAttrType TypeToAttrType<int64_t>() {
  return OrtOpAttrType::ORT_OP_ATTR_INT;
}

template <>
inline constexpr OrtOpAttrType TypeToAttrType<float>() {
  return OrtOpAttrType::ORT_OP_ATTR_FLOAT;
}

template <typename T>
inline constexpr OrtOpAttrType TypeToAttrsType();

template <>
inline constexpr OrtOpAttrType TypeToAttrsType<int64_t>() {
  return OrtOpAttrType::ORT_OP_ATTR_INTS;
}

template <>
inline constexpr OrtOpAttrType TypeToAttrsType<float>() {
  return OrtOpAttrType::ORT_OP_ATTR_FLOATS;
}

inline Status CheckAttrType(const OrtOpAttr* attr, OrtOpAttrType requested_type) {
  OrtOpAttrType type;
  Ort::Status status(GetApi().OpAttr_GetType(attr, &type));
  if (!status.IsOK()) return status;
  if (requested_type != type) {
    std::string msg = "Attribute type mismatch: expected " + std::to_string(requested_type) +
                      ", but got " + std::to_string(type);
    return Ort::Status(msg.c_str(), OrtErrorCode::ORT_INVALID_ARGUMENT);
  }
  return Ort::Status{};
}

inline size_t GetDataSize(const OrtOpAttr* attr, OrtOpAttrType attr_type) {
  size_t result{};
  // Ignore the status here because we check the data type so the error should only be about
  // the size
  [[maybe_unused]] Status status{GetApi().ReadOpAttr(attr, attr_type, nullptr, 0, &result)};
  return result;
}

template <typename T>
Ort::Status GetNumericValue(const OrtOpAttr* attr, T& out) {
  static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
  size_t size{};
  return Ort::Status{GetApi().ReadOpAttr(attr, TypeToAttrType<T>(), &out, sizeof(out), &size)};
}

template <typename T>
struct GetValueImpl {
  static Status GetValue(const OrtOpAttr* attr, T& out) {
    return GetNumericValue<T>(attr, out);
  }
  static Status GetValues(const OrtOpAttr* attr, std::vector<T>& out) {
    // Api deficiency when it comes to value arrays. It is not possible
    // to tell if the error is due to the type mismatch or the size
    // so we check the type first, and then ignore the status of the size check
    constexpr auto deduced_type = TypeToAttrsType<T>();
    auto status = CheckAttrType(attr, deduced_type);
    if (!status.IsOK()) return status;
    auto size = GetDataSize(attr, deduced_type);
    std::vector<T> result;
    if (size > 0) {
      result.resize(size / sizeof(T));
      status = Status{GetApi().ReadOpAttr(
          attr, deduced_type, result.data(), size, &size)};
      if (!status.IsOK()) return status;
    }
    out.swap(result);
    return status;
  }
};

// Create GetValueImpl specializations for std::string
template <>
struct GetValueImpl<std::string> {
  static Status GetValue(const OrtOpAttr* attr, std::string& out) {
    // Api deficiency when it comes to value arrays. It is not possible
    // to tell if the error is due to the type mismatch or the size
    // so we check the type first, and then ignore the status of the size check
    auto status = CheckAttrType(attr, OrtOpAttrType::ORT_OP_ATTR_STRING);
    if (!status.IsOK()) return status;
    auto size = GetDataSize(attr, OrtOpAttrType::ORT_OP_ATTR_STRING);
    std::string result;
    if (size > 0) {
      result.resize(size);
      // some compilers in use do not support std::string::data() non-const
      auto* buffer = &result[0];
      status = Status{GetApi().ReadOpAttr(
          attr, OrtOpAttrType::ORT_OP_ATTR_STRING, buffer, size, &size)};
      if (!status.IsOK()) return status;
    }
    out.swap(result);
    return status;
  }
  static Status GetValues(const OrtOpAttr* attr, std::vector<std::string>& out) {
    auto status = CheckAttrType(attr, OrtOpAttrType::ORT_OP_ATTR_STRINGS);
    if (!status.IsOK()) return status;

    std::vector<std::string> result;
    size_t total_buffer_size = GetDataSize(attr, OrtOpAttrType::ORT_OP_ATTR_STRINGS);
    if (total_buffer_size > 0) {
      // Create a temporary buffer to hold the string data
      std::vector<char> buffer(total_buffer_size);
      status = Status{GetApi().ReadOpAttr(attr, OrtOpAttrType::ORT_OP_ATTR_STRINGS, buffer.data(),
                                          total_buffer_size, &total_buffer_size)};
      if (!status.IsOK()) return status;

      const char* data = buffer.data();
      const char* end = data + total_buffer_size;
      while (data < end) {
        result.emplace_back(data);
        data += result.back().size() + 1;  // Move past the null terminator
      }
    }
    out.swap(result);
    return status;
  }
};

template <typename T>
template <typename R>
inline Status ConstOpAttrImpl<T>::GetValue(R& out) const {
  return GetValueImpl<R>::GetValue(this->p_, out);
}

template <typename T>
template <typename R>
inline Status ConstOpAttrImpl<T>::GetValueArray(std::vector<R>& out) const {
  return GetValueImpl<R>::GetValues(this->p_, out);
}

template <typename T>
inline Status ConstOpAttrImpl<T>::GetTensorAttributeAsOrtValue(Value& out) const {
  OrtValue* tensor_value = nullptr;
  auto status = Status(GetApi().OpAttr_GetTensorAttributeAsOrtValue(this->p_, &tensor_value));
  if (!status.IsOK()) return status;
  out = Value{tensor_value};
  return status;
}

template <typename T>
inline std::string ConstOpAttrImpl<T>::GetName() const {
  const char* name = nullptr;
  ThrowOnError(GetApi().OpAttr_GetName(this->p_, &name));
  if (name != nullptr) {
    return name;
  }
  return {};
}

template <typename T>
inline OrtOpAttrType ConstOpAttrImpl<T>::GetType() const {
  OrtOpAttrType type;
  ThrowOnError(GetApi().OpAttr_GetType(this->p_, &type));
  return type;
}
}  // namespace detail

inline OpAttr::OpAttr(const char* name, const void* data, int len, OrtOpAttrType type) {
  Ort::ThrowOnError(GetApi().CreateOpAttr(name, data, len, type, &p_));
}

namespace detail {
template <typename T>
inline KernelInfo KernelInfoImpl<T>::Copy() const {
  OrtKernelInfo* info_copy = nullptr;
  Ort::ThrowOnError(GetApi().CopyKernelInfo(this->p_, &info_copy));
  return KernelInfo{info_copy};
}

template <typename T>
inline size_t KernelInfoImpl<T>::GetInputCount() const {
  size_t out = 0;
  ThrowOnError(GetApi().KernelInfo_GetInputCount(this->p_, &out));
  return out;
}

template <typename T>
inline size_t KernelInfoImpl<T>::GetOutputCount() const {
  size_t out = 0;
  ThrowOnError(GetApi().KernelInfo_GetOutputCount(this->p_, &out));
  return out;
}

template <typename T>
inline std::string KernelInfoImpl<T>::GetInputName(size_t index) const {
  size_t size = 0;

  // Feed nullptr for the data buffer to query the true size of the string value
  Ort::ThrowOnError(GetApi().KernelInfo_GetInputName(this->p_, index, nullptr, &size));

  std::string out;
  out.resize(size);
  Ort::ThrowOnError(GetApi().KernelInfo_GetInputName(this->p_, index, &out[0], &size));
  out.resize(size - 1);  // remove the terminating character '\0'

  return out;
}

template <typename T>
inline std::string KernelInfoImpl<T>::GetOutputName(size_t index) const {
  size_t size = 0;

  // Feed nullptr for the data buffer to query the true size of the string value
  Ort::ThrowOnError(GetApi().KernelInfo_GetOutputName(this->p_, index, nullptr, &size));

  std::string out;
  out.resize(size);
  Ort::ThrowOnError(GetApi().KernelInfo_GetOutputName(this->p_, index, &out[0], &size));
  out.resize(size - 1);  // remove the terminating character '\0'

  return out;
}

template <typename T>
inline TypeInfo KernelInfoImpl<T>::GetInputTypeInfo(size_t index) const {
  OrtTypeInfo* out = nullptr;
  ThrowOnError(GetApi().KernelInfo_GetInputTypeInfo(this->p_, index, &out));
  return TypeInfo{out};
}

template <typename T>
inline TypeInfo KernelInfoImpl<T>::GetOutputTypeInfo(size_t index) const {
  OrtTypeInfo* out = nullptr;
  ThrowOnError(GetApi().KernelInfo_GetOutputTypeInfo(this->p_, index, &out));
  return TypeInfo{out};
}

template <typename T>
inline Value KernelInfoImpl<T>::GetTensorAttribute(const char* name, OrtAllocator* allocator) const {
  OrtValue* out = nullptr;
  ThrowOnError(GetApi().KernelInfoGetAttribute_tensor(this->p_, name, allocator, &out));
  return Value{out};
}

template <typename T>
inline ConstValue KernelInfoImpl<T>::GetTensorConstantInput(size_t index, int* is_constant) const {
  const OrtValue* out = nullptr;
  ThrowOnError(GetApi().KernelInfoGetConstantInput_tensor(this->p_, index, is_constant, &out));
  return ConstValue{out};
}

template <typename T>
inline std::string KernelInfoImpl<T>::GetNodeName() const {
  size_t size = 0;

  // Feed nullptr for the data buffer to query the true size of the string value
  Ort::ThrowOnError(GetApi().KernelInfo_GetNodeName(this->p_, nullptr, &size));

  std::string out;
  out.resize(size);
  Ort::ThrowOnError(GetApi().KernelInfo_GetNodeName(this->p_, &out[0], &size));
  out.resize(size - 1);  // remove the terminating character '\0'

  return out;
}

template <typename T>
inline Logger KernelInfoImpl<T>::GetLogger() const {
  const OrtLogger* out = nullptr;
  ThrowOnError(GetApi().KernelInfo_GetLogger(this->p_, &out));
  return Logger{out};
}

inline void attr_utils::GetAttr(const OrtKernelInfo* p, const char* name, float& out) {
  Ort::ThrowOnError(GetApi().KernelInfoGetAttribute_float(p, name, &out));
}

inline void attr_utils::GetAttr(const OrtKernelInfo* p, const char* name, int64_t& out) {
  Ort::ThrowOnError(GetApi().KernelInfoGetAttribute_int64(p, name, &out));
}

inline void attr_utils::GetAttr(const OrtKernelInfo* p, const char* name, std::string& result) {
  size_t size = 0;
  // Feed nullptr for the data buffer to query the true size of the string attribute
  Ort::ThrowOnError(GetApi().KernelInfoGetAttribute_string(p, name, nullptr, &size));

  std::string out;
  out.resize(size);
  Ort::ThrowOnError(GetApi().KernelInfoGetAttribute_string(p, name, &out[0], &size));
  out.resize(size - 1);  // remove the terminating character '\0'
  out.swap(result);
}

inline void attr_utils::GetAttrs(const OrtKernelInfo* p, const char* name, std::vector<float>& result) {
  size_t size = 0;
  // Feed nullptr for the data buffer to query the true size of the attribute
  Ort::ThrowOnError(GetApi().KernelInfoGetAttributeArray_float(p, name, nullptr, &size));

  std::vector<float> out;
  out.resize(size);
  Ort::ThrowOnError(GetApi().KernelInfoGetAttributeArray_float(p, name, out.data(), &size));
  out.swap(result);
}

inline void attr_utils::GetAttrs(const OrtKernelInfo* p, const char* name, std::vector<int64_t>& result) {
  size_t size = 0;

  // Feed nullptr for the data buffer to query the true size of the attribute
  Ort::ThrowOnError(GetApi().KernelInfoGetAttributeArray_int64(p, name, nullptr, &size));

  std::vector<int64_t> out;
  out.resize(size);
  Ort::ThrowOnError(GetApi().KernelInfoGetAttributeArray_int64(p, name, out.data(), &size));
  out.swap(result);
}
}  // namespace detail

inline KernelInfo::KernelInfo(OrtKernelInfo* info) : detail::KernelInfoImpl<OrtKernelInfo>{info} {}

inline Op::Op(OrtOp* p) : detail::Base<OrtOp>(p) {}

inline Op Op::Create(const OrtKernelInfo* info, const char* op_name, const char* domain, int version,
                     const char** type_constraint_names,
                     const ONNXTensorElementDataType* type_constraint_values,
                     size_t type_constraint_count,
                     const OpAttr* attr_values, size_t attr_count,
                     size_t input_count, size_t output_count) {
  static_assert(sizeof(OpAttr) == sizeof(OrtOpAttr*),
                "OpAttr's is expected to be just an array of OrtOpAttr in memory so we can reinterpret safely");
  auto attr_input_values = reinterpret_cast<const OrtOpAttr* const*>(attr_values);
  OrtOp* op;
  Ort::ThrowOnError(GetApi().CreateOp(info, op_name, domain, version, type_constraint_names, type_constraint_values,
                                      static_cast<int>(type_constraint_count),
                                      attr_input_values,
                                      static_cast<int>(attr_count),
                                      static_cast<int>(input_count),
                                      static_cast<int>(output_count), &op));
  return Op{op};
}

inline void Op::Invoke(const OrtKernelContext* context,
                       const Value* input_values,
                       size_t input_count,
                       Value* output_values,
                       size_t output_count) {
  static_assert(sizeof(Value) == sizeof(OrtValue*),
                "Value is really just an array of OrtValue* in memory, so we can reinterpret_cast safely");
  auto ort_input_values = reinterpret_cast<const OrtValue* const*>(input_values);
  auto ort_output_values = reinterpret_cast<OrtValue**>(output_values);
  Ort::ThrowOnError(GetApi().InvokeOp(context, p_, ort_input_values, static_cast<int>(input_count),
                                      ort_output_values, static_cast<int>(output_count)));
}

inline void Op::Invoke(const OrtKernelContext* context,
                       const OrtValue* const* input_values,
                       size_t input_count,
                       OrtValue* const* output_values,
                       size_t output_count) {
  Ort::ThrowOnError(GetApi().InvokeOp(context, p_, input_values, static_cast<int>(input_count),
                                      output_values, static_cast<int>(output_count)));
}

inline std::string GetVersionString() {
  return OrtGetApiBase()->GetVersionString();
}

inline std::string GetBuildInfoString() {
  return GetApi().GetBuildInfoString();
}

inline std::vector<std::string> GetAvailableProviders() {
  char** providers;
  int len;

  auto release_fn = [&len](char** providers) {
    // This should always return nullptr.
    ThrowOnError(GetApi().ReleaseAvailableProviders(providers, len));
  };

  ThrowOnError(GetApi().GetAvailableProviders(&providers, &len));
  std::unique_ptr<char*, decltype(release_fn)> guard(providers, release_fn);
  std::vector<std::string> available_providers;
  available_providers.reserve(static_cast<size_t>(len));
  for (int i = 0; i < len; ++i) {
    available_providers.emplace_back(providers[i]);
  }
  return available_providers;
}

template <typename TOp, typename TKernel, bool WithStatus>
void CustomOpBase<TOp, TKernel, WithStatus>::GetSessionConfigs(std::unordered_map<std::string, std::string>& out,
                                                               ConstSessionOptions options) const {
  const TOp* derived = static_cast<const TOp*>(this);
  std::vector<std::string> keys = derived->GetSessionConfigKeys();

  out.reserve(keys.size());

  std::string config_entry_key = detail::MakeCustomOpConfigEntryKey(derived->GetName(), "");
  const size_t prefix_size = config_entry_key.length();

  for (const auto& key : keys) {
    config_entry_key.resize(prefix_size);
    config_entry_key.append(key);
    out[key] = options.GetConfigEntryOrDefault(config_entry_key.c_str(), "");
  }
}

inline ShapeInferContext::ShapeInferContext(const OrtApi* ort_api,
                                            OrtShapeInferContext* ctx) : ort_api_(ort_api), ctx_(ctx) {
  size_t input_count = 0;
  Ort::ThrowOnError(ort_api_->ShapeInferContext_GetInputCount(ctx_, &input_count));
  for (size_t ith_input = 0; ith_input < input_count; ++ith_input) {
    OrtTensorTypeAndShapeInfo* info{};
    Ort::ThrowOnError(ort_api_->ShapeInferContext_GetInputTypeShape(ctx, ith_input, &info));
    TensorTypeAndShapeInfo type_shape_info(info);
    auto integer_shape = type_shape_info.GetShape();
    std::vector<const char*> symbolic_shape(integer_shape.size(), {});
    if (!integer_shape.empty()) {
      type_shape_info.GetSymbolicDimensions(&symbolic_shape[0], integer_shape.size());
    }
    Shape shape;
    for (size_t ith = 0; ith < integer_shape.size(); ++ith) {
      if (symbolic_shape[ith] && std::string{symbolic_shape[ith]}.size() > 0) {
        shape.emplace_back(symbolic_shape[ith]);
      } else {
        shape.emplace_back(integer_shape[ith]);
      }
    }
    input_shapes_.push_back(std::move(shape));
    type_shape_info.release();
  }
}

inline Status ShapeInferContext::SetOutputShape(size_t indice, const Shape& shape, ONNXTensorElementDataType type) {
  OrtTensorTypeAndShapeInfo* info = {};
  ORT_CXX_RETURN_ON_API_FAIL(ort_api_->CreateTensorTypeAndShapeInfo(&info));
  ORT_CXX_RETURN_ON_API_FAIL(ort_api_->SetTensorElementType(info, type));

  using InfoPtr = std::unique_ptr<OrtTensorTypeAndShapeInfo, std::function<void(OrtTensorTypeAndShapeInfo*)>>;

  InfoPtr info_ptr(info, [this](OrtTensorTypeAndShapeInfo* obj) {
    ort_api_->ReleaseTensorTypeAndShapeInfo(obj);
  });

  std::vector<int64_t> integer_dims;
  std::vector<const char*> symbolic_dims;

  for (const auto dim : shape) {
    if (dim.IsInt()) {
      integer_dims.push_back(dim.AsInt());
      symbolic_dims.push_back("");
    } else {
      if (!dim.AsSym() || std::string{dim.AsSym()}.empty()) {
        ORT_CXX_API_THROW("Symbolic dim must not be an empty string", ORT_INVALID_ARGUMENT);
      }
      integer_dims.push_back(SymbolicInteger::INVALID_INT_DIM);
      symbolic_dims.push_back(dim.AsSym());
    }
  }

  ORT_CXX_RETURN_ON_API_FAIL(ort_api_->SetDimensions(info, integer_dims.data(), integer_dims.size()));
  ORT_CXX_RETURN_ON_API_FAIL(ort_api_->SetSymbolicDimensions(info, symbolic_dims.data(), symbolic_dims.size()));
  ORT_CXX_RETURN_ON_API_FAIL(ort_api_->ShapeInferContext_SetOutputTypeShape(ctx_, indice, info));
  return Status{nullptr};
}

inline int64_t ShapeInferContext::GetAttrInt(const char* attr_name) {
  auto attr = GetAttrHdl(attr_name);
  int64_t value;
  Status status = attr.GetValue<int64_t>(value);
  if (!status.IsOK()) {
    ORT_CXX_API_THROW("Getting int attribute failed: " + status.GetErrorMessage(), status.GetErrorCode());
  }
  return value;
}

inline ShapeInferContext::Ints ShapeInferContext::GetAttrInts(const char* attr_name) {
  auto attr = GetAttrHdl(attr_name);
  ShapeInferContext::Ints result;
  auto status = attr.GetValueArray<int64_t>(result);
  if (!status.IsOK()) {
    ORT_CXX_API_THROW("Getting ints attribute failed: " + status.GetErrorMessage(), status.GetErrorCode());
  }
  return result;
}

inline float ShapeInferContext::GetAttrFloat(const char* attr_name) {
  auto attr = GetAttrHdl(attr_name);
  float value;
  Status status = attr.GetValue<float>(value);
  if (!status.IsOK()) {
    ORT_CXX_API_THROW("Getting float attribute failed: " + status.GetErrorMessage(), status.GetErrorCode());
  }
  return value;
}

inline ShapeInferContext::Floats ShapeInferContext::GetAttrFloats(const char* attr_name) {
  auto attr = GetAttrHdl(attr_name);
  ShapeInferContext::Floats result;
  auto status = attr.GetValueArray<float>(result);
  if (!status.IsOK()) {
    ORT_CXX_API_THROW("Getting floats attribute failed: " + status.GetErrorMessage(), status.GetErrorCode());
  }
  return result;
}

inline std::string ShapeInferContext::GetAttrString(const char* attr_name) {
  auto attr = GetAttrHdl(attr_name);
  std::string value;
  Status status = attr.GetValue<std::string>(value);
  if (!status.IsOK()) {
    ORT_CXX_API_THROW("Getting string attribute failed: " + status.GetErrorMessage(), status.GetErrorCode());
  }
  return value;
}

inline ShapeInferContext::Strings ShapeInferContext::GetAttrStrings(const char* attr_name) {
  auto attr = GetAttrHdl(attr_name);
  ShapeInferContext::Strings result;
  auto status = attr.GetValueArray<std::string>(result);
  if (!status.IsOK()) {
    ORT_CXX_API_THROW("Getting strings attribute failed: " + status.GetErrorMessage(), status.GetErrorCode());
  }
  return result;
}

inline ConstOpAttr ShapeInferContext::GetAttrHdl(const char* attr_name) const {
  const OrtOpAttr* attr_hdl = {};
  Ort::ThrowOnError(ort_api_->ShapeInferContext_GetAttribute(ctx_, attr_name, &attr_hdl));
  return ConstOpAttr{attr_hdl};
}

namespace detail {
inline std::vector<const char*> StringsToCharPtrs(const std::vector<std::string>& strings) {
  std::vector<const char*> ptrs;
  ptrs.reserve(strings.size());
  std::transform(strings.begin(), strings.end(), std::back_inserter(ptrs),
                 [](const std::string& s) { return s.c_str(); });

  return ptrs;
}
}  // namespace detail

namespace detail {
template <typename T>
inline size_t ConstNodeImpl<T>::GetId() const {
  size_t id;
  ThrowOnError(GetApi().Node_GetId(this->p_, &id));
  return id;
}

template <typename T>
inline std::string ConstNodeImpl<T>::GetName() const {
  const char* name;
  ThrowOnError(GetApi().Node_GetName(this->p_, &name));
  return std::string(name);
}

template <typename T>
inline std::string ConstNodeImpl<T>::GetOperatorType() const {
  const char* type;
  ThrowOnError(GetApi().Node_GetOperatorType(this->p_, &type));
  return std::string(type);
}

template <typename T>
inline std::string ConstNodeImpl<T>::GetDomain() const {
  const char* domain;
  ThrowOnError(GetApi().Node_GetDomain(this->p_, &domain));
  return std::string(domain);
}

template <typename T>
inline int ConstNodeImpl<T>::GetSinceVersion() const {
  int since_version;
  ThrowOnError(GetApi().Node_GetSinceVersion(this->p_, &since_version));
  return since_version;
}

template <typename T>
inline std::vector<ConstValueInfo> ConstNodeImpl<T>::GetInputs() const {
  static_assert(sizeof(const OrtValueInfo*) == sizeof(ConstValueInfo));
  size_t num_vi;
  ThrowOnError(GetApi().Node_GetNumInputs(this->p_, &num_vi));
  std::vector<ConstValueInfo> result;
  if (num_vi > 0) {
    result.resize(num_vi);
    ThrowOnError(GetApi().Node_GetInputs(this->p_, reinterpret_cast<const OrtValueInfo**>(result.data()), num_vi));
  }
  return result;
}

template <typename T>
inline std::vector<ConstValueInfo> ConstNodeImpl<T>::GetOutputs() const {
  static_assert(sizeof(const OrtValueInfo*) == sizeof(ConstValueInfo));
  size_t num_vi;
  ThrowOnError(GetApi().Node_GetNumOutputs(this->p_, &num_vi));
  std::vector<ConstValueInfo> result;
  if (num_vi > 0) {
    result.resize(num_vi);
    ThrowOnError(GetApi().Node_GetOutputs(this->p_, reinterpret_cast<const OrtValueInfo**>(result.data()), num_vi));
  }
  return result;
}

template <typename T>
inline std::vector<ConstValueInfo> ConstNodeImpl<T>::GetImplicitInputs() const {
  static_assert(sizeof(const OrtValueInfo*) == sizeof(ConstValueInfo));
  size_t num_vi;
  ThrowOnError(GetApi().Node_GetNumImplicitInputs(this->p_, &num_vi));
  std::vector<ConstValueInfo> result;
  if (num_vi > 0) {
    result.resize(num_vi);
    ThrowOnError(GetApi().Node_GetImplicitInputs(this->p_, reinterpret_cast<const OrtValueInfo**>(result.data()),
                                                 num_vi));
  }
  return result;
}

template <typename T>
inline std::vector<ConstOpAttr> ConstNodeImpl<T>::GetAttributes() const {
  static_assert(sizeof(const OrtOpAttr*) == sizeof(ConstOpAttr), "Must be the same size");
  size_t num_attrs;
  ThrowOnError(GetApi().Node_GetNumAttributes(this->p_, &num_attrs));
  std::vector<ConstOpAttr> attrs;
  if (num_attrs > 0) {
    attrs.resize(num_attrs);
    ThrowOnError(GetApi().Node_GetAttributes(this->p_, reinterpret_cast<const OrtOpAttr**>(attrs.data()), num_attrs));
  }
  return attrs;
}

template <typename T>
inline Status ConstNodeImpl<T>::GetAttributeByName(const std::string& name, ConstOpAttr& out) const {
  const OrtOpAttr* attr = nullptr;
  auto status = Status(GetApi().Node_GetAttributeByName(this->p_, name.c_str(), &attr));
  out = ConstOpAttr{attr};
  return status;
}

template <typename T>
inline std::vector<AttrNameSubgraph> ConstNodeImpl<T>::GetSubgraphs() const {
  size_t num_graphs;
  ThrowOnError(GetApi().Node_GetNumSubgraphs(this->p_, &num_graphs));
  std::vector<AttrNameSubgraph> result;
  if (num_graphs > 0) {
    std::vector<const OrtGraph*> sub_graphs(num_graphs);
    std::vector<const char*> attr_names(num_graphs);
    ThrowOnError(GetApi().Node_GetSubgraphs(this->p_, sub_graphs.data(), num_graphs, attr_names.data()));
    result.reserve(num_graphs);
    for (size_t i = 0; i < num_graphs; ++i) {
      result.push_back({std::string(attr_names[i]), ConstGraph{sub_graphs[i]}});
    }
  }
  return result;
}

template <typename T>
inline ConstGraph ConstNodeImpl<T>::GetGraph() const {
  const OrtGraph* graph;
  ThrowOnError(GetApi().Node_GetGraph(this->p_, &graph));
  return ConstGraph{graph};
}

template <typename T>
inline std::string ConstNodeImpl<T>::GetEpName() const {
  const char* name;
  ThrowOnError(GetApi().Node_GetEpName(this->p_, &name));
  return std::string(name);
}

}  // namespace detail

#if !defined(ORT_MINIMAL_BUILD)
// static
inline void Node::Init(const std::string& operator_name, const std::string& operator_domain,
                       const std::string& node_name,
                       const std::vector<std::string>& input_names,
                       const std::vector<std::string>& output_names,
                       std::vector<OpAttr>& attributes,
                       OrtNode*& node) {
  auto inputs = detail::StringsToCharPtrs(input_names);
  auto outputs = detail::StringsToCharPtrs(output_names);

  std::vector<OrtOpAttr*> attributes_ptrs;
  attributes_ptrs.reserve(attributes.size());
  std::transform(attributes.begin(), attributes.end(), std::back_inserter(attributes_ptrs),
                 [](OpAttr& attr) -> OrtOpAttr* { return attr; });

  ThrowOnError(GetModelEditorApi().CreateNode(operator_name.c_str(), operator_domain.c_str(), node_name.c_str(),
                                              inputs.data(), inputs.size(),
                                              outputs.data(), outputs.size(),
                                              attributes_ptrs.data(), attributes_ptrs.size(),
                                              &node));

  // Node now owns the attributes
  std::for_each(attributes.begin(), attributes.end(), [](OpAttr& attr) { attr.release(); });
}

inline Node::Node(const std::string& operator_name, const std::string& operator_domain,
                  const std::string& node_name,
                  const std::vector<std::string>& input_names,
                  const std::vector<std::string>& output_names,
                  std::vector<OpAttr>& attributes) {
  Init(operator_name, operator_domain, node_name, input_names, output_names, attributes, p_);
}

inline Node::Node(const std::string& operator_name, const std::string& operator_domain,
                  const std::string& node_name,
                  const std::vector<std::string>& input_names,
                  const std::vector<std::string>& output_names) {
  std::vector<OpAttr> empty_attributes;
  Init(operator_name, operator_domain, node_name, input_names, output_names, empty_attributes, p_);
}
inline ValueInfo::ValueInfo(const std::string& name, const ConstTypeInfo& type_info) {
  ThrowOnError(GetModelEditorApi().CreateValueInfo(name.c_str(), type_info, &p_));
}
#endif  // !defined(ORT_MINIMAL_BUILD)

namespace detail {
template <typename T>
inline std::string ConstValueInfoImpl<T>::GetName() const {
  const char* p = nullptr;
  ThrowOnError(GetApi().GetValueInfoName(this->p_, &p));
  return std::string(p);
}

template <typename T>
inline ConstTypeInfo ConstValueInfoImpl<T>::TypeInfo() const {
  const OrtTypeInfo* type_info = nullptr;
  ThrowOnError(GetApi().GetValueInfoTypeInfo(this->p_, &type_info));
  return ConstTypeInfo{type_info};
}

template <typename T>
inline ValueInfoConsumerProducerInfo ConstValueInfoImpl<T>::GetProducerNode() const {
  ValueInfoConsumerProducerInfo info;
  const OrtNode* producer;
  size_t index;
  ThrowOnError(GetApi().ValueInfo_GetValueProducer(this->p_, &producer, &index));
  info.node = ConstNode(producer);
  info.index = static_cast<int64_t>(index);
  return info;
}

template <typename T>
inline std::vector<ValueInfoConsumerProducerInfo> ConstValueInfoImpl<T>::GetConsumers() const {
  size_t num = 0;
  ThrowOnError(GetApi().ValueInfo_GetValueNumConsumers(this->p_, &num));
  std::vector<ValueInfoConsumerProducerInfo> out;
  if (num > 0) {
    std::vector<const OrtNode*> nodes(num);
    std::vector<int64_t> indices(num);
    ThrowOnError(GetApi().ValueInfo_GetValueConsumers(this->p_, nodes.data(), indices.data(), num));
    out.reserve(num);
    for (size_t i = 0; i < num; ++i) {
      out.push_back({ConstNode{nodes[i]}, indices[i]});
    }
  }
  return out;
}

template <typename T>
inline Status ConstValueInfoImpl<T>::GetInitializer(ConstValue& value) const {
  const OrtValue* out = nullptr;
  auto status = Status(GetApi().ValueInfo_GetInitializerValue(this->p_, &out));
  if (!status.IsOK()) return status;
  value = ConstValue{out};
  return status;
}

template <typename T>
inline Status ConstValueInfoImpl<T>::GetExternalInitializerInfo(ExternalInitializerInfo& info) const {
  OrtExternalInitializerInfo* out = nullptr;
  auto status = Status(GetApi().ValueInfo_GetExternalInitializerInfo(this->p_, &out));
  if (!status.IsOK()) return status;
  info = ExternalInitializerInfo{out};
  return status;
}

template <typename T>
inline bool ConstValueInfoImpl<T>::IsRequiredGraphInput() const {
  bool out = false;
  ThrowOnError(GetApi().ValueInfo_IsRequiredGraphInput(this->p_, &out));
  return out;
}

template <typename T>
inline bool ConstValueInfoImpl<T>::IsOptionalGraphInput() const {
  bool out = false;
  ThrowOnError(GetApi().ValueInfo_IsOptionalGraphInput(this->p_, &out));
  return out;
}

template <typename T>
inline bool ConstValueInfoImpl<T>::IsGraphOutput() const {
  bool out = false;
  ThrowOnError(GetApi().ValueInfo_IsGraphOutput(this->p_, &out));
  return out;
}

template <typename T>
inline bool ConstValueInfoImpl<T>::IsConstantInitializer() const {
  bool out = false;
  ThrowOnError(GetApi().ValueInfo_IsConstantInitializer(this->p_, &out));
  return out;
}

template <typename T>
inline bool ConstValueInfoImpl<T>::IsFromOuterScope() const {
  bool out = false;
  ThrowOnError(GetApi().ValueInfo_IsFromOuterScope(this->p_, &out));
  return out;
}

template <typename T>
inline ModelMetadata ConstGraphImpl<T>::GetModelMetadata() const {
  OrtModelMetadata* out;
  ThrowOnError(GetApi().Graph_GetModelMetadata(this->p_, &out));
  return ModelMetadata{out};
}

template <typename T>
inline std::string ConstGraphImpl<T>::GetName() const {
  const char* name;
  ThrowOnError(GetApi().Graph_GetName(this->p_, &name));
  return std::string(name);
}

template <typename T>
inline std::basic_string<ORTCHAR_T> ConstGraphImpl<T>::GetModelPath() const {
  const ORTCHAR_T* path;
  ThrowOnError(GetApi().Graph_GetModelPath(this->p_, &path));
  return std::basic_string<ORTCHAR_T>(path);
}

template <typename T>
inline int64_t ConstGraphImpl<T>::GetOnnxIRVersion() const {
  int64_t version;
  ThrowOnError(GetApi().Graph_GetOnnxIRVersion(this->p_, &version));
  return version;
}

template <typename T>
inline std::vector<OperatorSet> ConstGraphImpl<T>::GetOperatorSets() const {
  size_t num_opsets;
  ThrowOnError(GetApi().Graph_GetNumOperatorSets(this->p_, &num_opsets));
  std::vector<OperatorSet> result;
  if (num_opsets > 0) {
    std::vector<const char*> domains;
    std::vector<int64_t> versions;
    domains.resize(num_opsets);
    versions.resize(num_opsets);
    ThrowOnError(GetApi().Graph_GetOperatorSets(this->p_, domains.data(), versions.data(), num_opsets));
    result.reserve(num_opsets);
    for (size_t i = 0; i < num_opsets; ++i) {
      result.push_back({domains[i], versions[i]});
    }
  }
  return result;
}

template <typename T>
inline std::vector<ConstValueInfo> ConstGraphImpl<T>::GetInputs() const {
  static_assert(sizeof(const OrtValueInfo*) == sizeof(ConstValueInfo));
  size_t num_vi;
  ThrowOnError(GetApi().Graph_GetNumInputs(this->p_, &num_vi));
  std::vector<ConstValueInfo> result;
  if (num_vi > 0) {
    result.resize(num_vi);
    ThrowOnError(GetApi().Graph_GetInputs(this->p_, reinterpret_cast<const OrtValueInfo**>(result.data()), num_vi));
  }
  return result;
}

template <typename T>
inline std::vector<ConstValueInfo> ConstGraphImpl<T>::GetOutputs() const {
  static_assert(sizeof(const OrtValueInfo*) == sizeof(ConstValueInfo));
  size_t num_vi;
  ThrowOnError(GetApi().Graph_GetNumOutputs(this->p_, &num_vi));
  std::vector<ConstValueInfo> result;
  if (num_vi > 0) {
    result.resize(num_vi);
    ThrowOnError(GetApi().Graph_GetOutputs(this->p_, reinterpret_cast<const OrtValueInfo**>(result.data()), num_vi));
  }
  return result;
}

template <typename T>
inline std::vector<ConstValueInfo> ConstGraphImpl<T>::GetInitializers() const {
  static_assert(sizeof(const OrtValueInfo*) == sizeof(ConstValueInfo));
  size_t num_vi;
  ThrowOnError(GetApi().Graph_GetNumInitializers(this->p_, &num_vi));
  std::vector<ConstValueInfo> result;
  if (num_vi > 0) {
    result.resize(num_vi);
    ThrowOnError(GetApi().Graph_GetInitializers(this->p_, reinterpret_cast<const OrtValueInfo**>(result.data()),
                                                num_vi));
  }
  return result;
}

template <typename T>
inline std::vector<ConstNode> ConstGraphImpl<T>::GetNodes() const {
  static_assert(sizeof(const OrtNode*) == sizeof(ConstNode));
  size_t num_nodes;
  ThrowOnError(GetApi().Graph_GetNumNodes(this->p_, &num_nodes));
  std::vector<ConstNode> result;
  if (num_nodes > 0) {
    result.resize(num_nodes);
    ThrowOnError(GetApi().Graph_GetNodes(this->p_, reinterpret_cast<const OrtNode**>(result.data()), num_nodes));
  }
  return result;
}

template <typename T>
inline ConstNode ConstGraphImpl<T>::GetParentNode() const {
  const OrtNode* parent;
  ThrowOnError(GetApi().Graph_GetParentNode(this->p_, &parent));
  return ConstNode{parent};
}

template <typename T>
inline Graph ConstGraphImpl<T>::GetGraphView(const std::vector<ConstNode>& nodes) const {
  OrtGraph* graph_viewer;
  std::vector<const OrtNode*> inputs_ptrs;
  inputs_ptrs.reserve(nodes.size());
  std::transform(nodes.begin(), nodes.end(), std::back_inserter(inputs_ptrs),
                 [](ConstNode n) -> const OrtNode* { return n; });
  ThrowOnError(GetApi().Graph_GetGraphView(this->p_, inputs_ptrs.data(),
                                           nodes.size(), &graph_viewer));
  return Graph{graph_viewer};
}

#if !defined(ORT_MINIMAL_BUILD)
template <typename T>
inline void GraphImpl<T>::SetInputs(std::vector<ValueInfo>& inputs) {
  std::vector<OrtValueInfo*> inputs_ptrs;
  inputs_ptrs.reserve(inputs.size());
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(inputs_ptrs),
                 [](ValueInfo& vi) -> OrtValueInfo* { return vi; });

  ThrowOnError(GetModelEditorApi().SetGraphInputs(this->p_, inputs_ptrs.data(), inputs_ptrs.size()));

  // Graph now owns the inputs
  std::for_each(inputs.begin(), inputs.end(), [](ValueInfo& vi) { vi.release(); });
}

template <typename T>
inline void GraphImpl<T>::SetOutputs(std::vector<ValueInfo>& outputs) {
  std::vector<OrtValueInfo*> outputs_ptrs;
  outputs_ptrs.reserve(outputs.size());
  std::transform(outputs.begin(), outputs.end(), std::back_inserter(outputs_ptrs),
                 [](ValueInfo& vi) -> OrtValueInfo* { return vi; });

  ThrowOnError(GetModelEditorApi().SetGraphOutputs(this->p_, outputs_ptrs.data(), outputs_ptrs.size()));

  // Graph now owns the outputs
  std::for_each(outputs.begin(), outputs.end(), [](ValueInfo& vi) { vi.release(); });
}

template <typename T>
inline void GraphImpl<T>::AddInitializer(const std::string& name, Value& initializer, bool data_is_external) {
  // Graph takes ownership of `initializer`
  // On error the ownership is not transferred.
  ThrowOnError(GetModelEditorApi().AddInitializerToGraph(this->p_, name.c_str(), initializer, data_is_external));
  initializer.release();
}

template <typename T>
inline void GraphImpl<T>::AddNode(Node& node) {
  // Graph takes ownership of `node`
  ThrowOnError(GetModelEditorApi().AddNodeToGraph(this->p_, node.release()));
}

template <typename T>
inline void ModelImpl<T>::AddGraph(Graph& graph) {
  // Model takes ownership of `graph`
  ThrowOnError(GetModelEditorApi().AddGraphToModel(this->p_, graph.release()));
}
#endif  // !defined(ORT_MINIMAL_BUILD)

}  // namespace detail

#if !defined(ORT_MINIMAL_BUILD)
inline Graph::Graph() {
  ThrowOnError(GetModelEditorApi().CreateGraph(&p_));
}

inline Model::Model(const std::vector<DomainOpsetPair>& opsets) {
  std::vector<const char*> domains;
  std::vector<int> versions;
  domains.reserve(opsets.size());
  versions.reserve(opsets.size());

  for (const auto& pair : opsets) {
    domains.push_back(pair.first.c_str());
    versions.push_back(pair.second);
  }

  ThrowOnError(GetModelEditorApi().CreateModel(domains.data(), versions.data(), opsets.size(), &p_));
}
#endif

}  // namespace Ort
