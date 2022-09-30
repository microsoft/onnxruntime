// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include "opwrapper_provider_factory.h"
#include "onnxruntime_cxx_api.h"

/** \brief All C++ ONNXRuntime APIs in onnxruntime_cxx_api.h are defined inside the Ort:: namespace.
 *
 */
namespace Ort {

/** \brief All C++ OpWrapper execution provider APIs are defined inside the Ort::OpWrapper:: namespace.
 *
 */
namespace OpWrapper {

namespace detail {
/** \brief Internal function that gets the OrtOpWrapperApi from OrtApi's GetExecutionProviderApi().
 *
 */
inline const OrtOpWrapperApi* GetOrtOpWrapperApi() {
  const OrtApi& ort_api = Ort::GetApi();
  const void* opwrapper_api = nullptr;

  Ort::ThrowOnError(ort_api.GetExecutionProviderApi("OpWrapper", ORT_API_VERSION, &opwrapper_api));

  return reinterpret_cast<const OrtOpWrapperApi*>(opwrapper_api);
}
}  // namespace detail

// This class holds the global variable that points to the OrtOpWrapperApi. It's in a template so that we can define a
// global variable in a header and make it transparent to the users of the API. Note that Ort::Global holds the
// global OrtAPi, and Ort::OpWrapper::Global holds the global OrtOpWrapperApi.
template <typename T>
struct Global {
  static const OrtOpWrapperApi* api_;
};

// Return's a reference to the global OrtOpWrapperApi object. User must call InitApi() first if ORT_API_MANUAL_INIT
// is defined. Note that Ort::GetApi() returns OrtApi&, and Ort::OpWrapper::GetApi() returns OrtOpWrapperApi&.
inline const OrtOpWrapperApi& GetApi() { return *Global<void>::api_; }

// If macro ORT_API_MANUAL_INIT is defined, no static initialization will be performed. Instead, user must
// call Ort::OpWrapper::InitApi() before using the global OrtOpWrapperApi object.
#ifdef ORT_API_MANUAL_INIT
template <typename T>
const OrtOpWrapperApi* Global<T>::api_{};

inline void InitApi() {
  if (!Ort::ApiIsInit()) {
    Ort::InitApi();
  }

  Global<void>::api_ = detail::GetOrtOpWrapperApi();
}

inline void InitApi(const OrtApi* api) {
  Ort::InitApi(api);
  Global<void>::api_ = detail::GetOrtOpWrapperApi();
}
#else
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
// "Global initializer calls a non-constexpr function." Therefore you can't use ORT APIs in the other global
// initializers. Please define ORT_API_MANUAL_INIT if it concerns you.
#pragma warning(disable : 26426)
#endif
template <typename T>
const OrtOpWrapperApi* Global<T>::api_ = detail::GetOrtOpWrapperApi();
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif
#endif

struct ProviderOptions : Ort::Base<OrtOpWrapperProviderOptions> {
  explicit ProviderOptions(std::nullptr_t) {}
  ProviderOptions();  // Wraps OrtOpWrapperApi::CreateProviderOptions.

  // Wraps OrtOpWrapperApi::CreateProviderOptions and OrtOpWrapperApi::ProviderOptions_Update.
  explicit ProviderOptions(const std::unordered_map<std::string, std::string>& options);
  explicit ProviderOptions(OrtOpWrapperProviderOptions* options);

  [[nodiscard]] bool HasOption(const char* key) const;
  [[nodiscard]] std::string_view GetOption(const char* key) const;
  void UpdateOptions(const std::unordered_map<std::string, std::string>& options);
  [[nodiscard]] std::unordered_map<std::string_view, std::string_view> ToMap(OrtAllocator* allocator) const;

  static ProviderOptions FromKernelInfo(const OrtKernelInfo* kernel_info, const char* op_name);
};

Ort::SessionOptions& AppendExecutionProvider(Ort::SessionOptions& session_options,
                                             const std::unordered_map<std::string, ProviderOptions>& provider_options);
Ort::SessionOptions& AppendExecutionProvider(Ort::SessionOptions& session_options,
                                             const char* op_name,
                                             const ProviderOptions& op_options);
}  // namespace OpWrapper

// Defines OrtRelease(OrtXXX* ptr) functions used by Ort::Base<> to release the underlying OrtXXX resource.
#define ORT_OPWRAPPER_DEFINE_RELEASE(NAME) \
  inline void OrtRelease(Ort##NAME* ptr) { Ort::OpWrapper::GetApi().Release##NAME(ptr); }

ORT_OPWRAPPER_DEFINE_RELEASE(OpWrapperProviderOptions)
}  // namespace Ort

// Implementation details below

namespace Ort {
namespace OpWrapper {
inline ProviderOptions::ProviderOptions(OrtOpWrapperProviderOptions* options)
    : Base<OrtOpWrapperProviderOptions>{options} {
}

inline ProviderOptions::ProviderOptions() {
  Ort::ThrowOnError(GetApi().CreateProviderOptions(&p_));
}

inline ProviderOptions::ProviderOptions(const std::unordered_map<std::string, std::string>& options) {
  Ort::ThrowOnError(GetApi().CreateProviderOptions(&p_));
  UpdateOptions(options);
}

inline bool ProviderOptions::HasOption(const char* key) const {
  int exists = 0;
  Ort::ThrowOnError(GetApi().ProviderOptions_HasOption(p_, key, &exists));
  return exists != 0;
}

inline std::string_view ProviderOptions::GetOption(const char* key) const {
  const char* out = nullptr;
  size_t len = 0;
  Ort::ThrowOnError(GetApi().ProviderOptions_GetOption(p_, key, &out, &len));
  return std::string_view(out, len);
}

inline void ProviderOptions::UpdateOptions(const std::unordered_map<std::string, std::string>& options) {
  const size_t num_options = options.size();
  std::vector<const char*> keys;
  std::vector<const char*> vals;

  keys.reserve(num_options);
  vals.reserve(num_options);

  for (const auto& it : options) {
    keys.push_back(it.first.c_str());
    vals.push_back(it.second.c_str());
  }

  Ort::ThrowOnError(GetApi().ProviderOptions_Update(p_, keys.data(), vals.data(), num_options));
}

inline std::unordered_map<std::string_view, std::string_view> ProviderOptions::ToMap(OrtAllocator* allocator) const {
  std::unordered_map<std::string_view, std::string_view> map;
  size_t num_options = 0;
  const char** keys = nullptr;
  const char** vals = nullptr;
  size_t* key_lens = nullptr;
  size_t* val_lens = nullptr;

  ThrowOnError(GetApi().ProviderOptions_Serialize(p_, allocator, &keys, &key_lens, &vals, &val_lens, &num_options));

  if (num_options != 0) {

    // TODO: Use std::span<> when C++20 is supported by ORT. For now, disable warning when indexing C-style arrays.
    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    for (size_t i = 0; i < num_options; ++i) {
      std::string_view key(keys[i], key_lens[i]);
      std::string_view val(vals[i], val_lens[i]);
      map.emplace(key, val);
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

    allocator->Free(allocator, keys);
    allocator->Free(allocator, vals);
    allocator->Free(allocator, key_lens);
    allocator->Free(allocator, val_lens);
  }

  return map;
}

inline ProviderOptions ProviderOptions::FromKernelInfo(const OrtKernelInfo* kernel_info, const char* op_name) {
  OrtOpWrapperProviderOptions* options = nullptr;
  Ort::ThrowOnError(GetApi().KernelInfo_GetProviderOptions(kernel_info, op_name, &options));
  return ProviderOptions(options);
}

inline Ort::SessionOptions& AppendExecutionProvider(Ort::SessionOptions& session_options,
                                                    const std::unordered_map<std::string, ProviderOptions>& provider_options) {
  const size_t num_ops = provider_options.size();
  std::vector<const char*> op_names;
  std::vector<const OrtOpWrapperProviderOptions*> op_options;

  op_names.reserve(num_ops);
  op_options.reserve(num_ops);

  for (const auto& it : provider_options) {
    op_names.push_back(it.first.c_str());
    op_options.push_back(it.second);
  }

  Ort::ThrowOnError(GetApi().SessionOptionsAppendExecutionProvider(session_options, op_names.data(),
                                                                   op_options.data(), num_ops));

  return session_options;
}

inline Ort::SessionOptions& AppendExecutionProvider(Ort::SessionOptions& session_options,
                                                    const char* op_name,
                                                    const ProviderOptions& op_options) {
  constexpr size_t num_ops = 1;
  std::array<const OrtOpWrapperProviderOptions*, num_ops> ops_options{op_options};

  Ort::ThrowOnError(GetApi().SessionOptionsAppendExecutionProvider(session_options, &op_name, ops_options.data(),
                                                                   num_ops));
  return session_options;
}
}  // namespace OpWrapper
}  // namespace Ort
