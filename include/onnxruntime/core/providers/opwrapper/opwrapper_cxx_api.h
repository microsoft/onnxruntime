// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
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
const OrtOpWrapperApi* GetOrtOpWrapperApi() {
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


struct OpWrapperProviderOptions : Ort::Base<OrtOpWrapperProviderOptions> {
  explicit OpWrapperProviderOptions(std::nullptr_t) {}
  explicit OpWrapperProviderOptions(OrtOpWrapperProviderOptions* options);

  size_t HasOption(const char* key) const;
  std::string GetOption(const char* key, size_t value_size = 0) const;
  std::unordered_map<std::string, std::string> ToMap() const;

  static OpWrapperProviderOptions FromKernelInfo(Unowned<const KernelInfo> kernel_info, const char* op_name);
};
}  // namespace OpWrapper

// Defines OrtRelease(OrtXXX* ptr) functions used by Ort::Base<> to release the underlying OrtXXX resource.
#define ORT_OPWRAPPER_DEFINE_RELEASE(NAME) \
  inline void OrtRelease(Ort##NAME* ptr) { Ort::OpWrapper::GetApi().Release##NAME(ptr); }

ORT_OPWRAPPER_DEFINE_RELEASE(OpWrapperProviderOptions)
}  // namespace Ort

#include "opwrapper_cxx_inline.h"
