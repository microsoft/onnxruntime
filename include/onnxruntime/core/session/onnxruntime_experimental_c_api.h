// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Experimental C API consumer header.
//
// This header provides typedefs, name constants, and (for C++) typed inline accessors for experimental ORT functions.
// It should be used together with the experimental header lookup function `OrtApi::GetExperimentalFunction()`.
//
// This header contains code generated from onnxruntime_experimental_c_api.inc, which defines the list of experimental
// API functions.
//
// IMPORTANT: Experimental functions are NOT part of the stable ABI. They may be added, changed, or removed between
// releases without notice. A function's availability should always be checked at runtime (the lookup returns nullptr
// if the function is not present).
//
// C usage:
//   OrtExperimental_OrtApi_ExperimentalApiTest_SinceV28_Fn fn =
//       (OrtExperimental_OrtApi_ExperimentalApiTest_SinceV28_Fn)api->GetExperimentalFunction(
//           kOrtExperimental_OrtApi_ExperimentalApiTest_SinceV28_FnName);
//   if (fn) {
//     OrtStatusPtr status = fn(&result);
//   }
//
// C++ usage:
//   if (auto* fn = Ort::Experimental::Get_OrtApi_ExperimentalApiTest_SinceV28_Fn(api)) {
//     Ort::Status status(fn(&result));
//   }

#pragma once

#include "onnxruntime_c_api.h"

//
// Auxiliary declarations
//
// Declarations of auxiliary types or typedefs required by experimental APIs go here.
//
// For example, if an experimental API uses a new type, OrtExperimentalType, we would declare it in this file:
//   ORT_RUNTIME_CLASS(ExperimentalType);
//

ORT_RUNTIME_CLASS(ModelPackageOptions);
ORT_RUNTIME_CLASS(ModelPackageContext);
ORT_RUNTIME_CLASS(ModelPackageComponentContext);

// Opaque handle holding the EPContext callbacks and opaque state extracted from an OrtSessionOptions instance. Used by
// the experimental OrtEpApi_* EPContext data functions. Create via OrtEpApi_SessionOptions_GetEpContextConfig and
// release with OrtEpApi_ReleaseEpContextConfig.
ORT_RUNTIME_CLASS(EpContextConfig);

/** \brief Function called to write named binary data.
 *
 * This callback is currently used for EPContext binary data, but its contract is intentionally generic so future APIs
 * can reuse it for other named data payloads. The callback is called synchronously by the component that receives it.
 * ORT does not own or retain buffer after the callback returns. ORT does not serialize invocations made by different
 * EP instances or worker threads.
 *
 * Each callback invocation represents one complete write operation for name. The callback signature does not
 * provide an offset, sequence number, or final-chunk marker, so the component invoking the callback must define any
 * chunked ordering and completion contract with the application. Current EPContext use should prefer a single callback
 * invocation per EPContext binary unless chunking semantics are documented by the EP.
 *
 * The application's implementation can process the data in any way (e.g., encrypt and store, upload to cloud storage,
 * or compress) before persisting it.
 *
 * \param[in] state Opaque pointer holding the user's state. ORT does not own or manage this pointer. The application
 *                  must keep it valid for the duration required by the API that accepted the callback and must provide
 *                  any synchronization required if it can be used concurrently.
 * \param[in] name The file name or logical data identifier as a null-terminated UTF-8 string.
 * \param[in] buffer The buffer containing data to write.
 * \param[in] buffer_num_bytes The size of the buffer in bytes.
 *
 * \return OrtStatus* Write status. Return nullptr on success.
 *                    On failure, use CreateStatus to provide error info with an appropriate OrtErrorCode
 *                    (e.g., ORT_FAIL); ORT propagates the returned code. ORT will release the OrtStatus* if not null.
 */
typedef OrtStatus*(ORT_API_CALL* OrtWriteNamedBufferFunc)(_In_ void* state,
                                                          _In_ const char* name,
                                                          _In_ const void* buffer,
                                                          _In_ size_t buffer_num_bytes);

/** \brief Function called to read named binary data.
 *
 * This callback is currently used for EPContext binary data, but its contract is intentionally generic so future APIs
 * can reuse it for other named data payloads. The application reads, processes (e.g., decrypts, decompresses,
 * downloads), and returns the requested data. ORT provides an allocator so the application can allocate the output
 * buffer directly. The callback is called synchronously by the component that receives it. ORT does not serialize
 * invocations made by different EP instances or worker threads.
 *
 * \param[in] state Opaque pointer holding the user's state. ORT does not own or manage this pointer. The application
 *                  must keep it valid for the duration required by the API that accepted the callback and must provide
 *                  any synchronization required if it can be used concurrently.
 * \param[in] name The file name or logical data identifier to read as a null-terminated UTF-8 string.
 * \param[in] allocator ORT-provided allocator. The application must use this to allocate the output buffer.
 * \param[out] buffer Set by the implementation to the allocated buffer containing the output data.
 * \param[out] data_size Set by the implementation to the size of the output data in bytes.
 *
 * \return OrtStatus* Read status. Return nullptr on success.
 *                    On failure, use CreateStatus to provide error info with an appropriate OrtErrorCode
 *                    (e.g., ORT_FAIL); ORT propagates the returned code. ORT will release the OrtStatus* if not null.
 */
typedef OrtStatus*(ORT_API_CALL* OrtReadNamedBufferFunc)(_In_ void* state,
                                                         _In_ const char* name,
                                                         _In_ OrtAllocator* allocator,
                                                         _Outptr_ void** buffer,
                                                         _Out_ size_t* data_size);

//
// C: function pointer typedefs and name constants
//

// For each ORT_EXPERIMENTAL_API(VER, RET, NAME, ...) entry in the .inc file, this generates:
//
//   // Function pointer typedef:
//   typedef RET(ORT_API_CALL* OrtExperimental_<NAME>_SinceV<VER>_Fn)(...) NO_EXCEPTION;
//
//   // Name constant for lookup:
//   static const char* const kOrtExperimental_<NAME>_SinceV<VER>_FnName = "<NAME>_SinceV<VER>";
//
// Example: ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtApi_ExperimentalApiTest, _Out_ int64_t* out) produces:
//   typedef OrtStatusPtr(ORT_API_CALL* OrtExperimental_OrtApi_ExperimentalApiTest_SinceV28_Fn)(
//       _Out_ int64_t* out) NO_EXCEPTION;
//   static const char* const kOrtExperimental_OrtApi_ExperimentalApiTest_SinceV28_FnName =
//       "OrtApi_ExperimentalApiTest_SinceV28";
#define ORT_EXPERIMENTAL_API(VER, RET, NAME, ...)                                                 \
  typedef RET(ORT_API_CALL* OrtExperimental_##NAME##_SinceV##VER##_Fn)(__VA_ARGS__) NO_EXCEPTION; \
  static const char* const kOrtExperimental_##NAME##_SinceV##VER##_FnName = #NAME "_SinceV" #VER;

#include "onnxruntime_experimental_c_api.inc"

#undef ORT_EXPERIMENTAL_API

//
// C++: typed inline accessors
//

#ifdef __cplusplus

namespace Ort {
namespace Experimental {

// For each .inc entry, this generates a typed accessor in Ort::Experimental:
//
//   inline OrtExperimental_<NAME>_SinceV<VER>_Fn Get_<NAME>_SinceV<VER>_Fn(const OrtApi* api);
//
// Example: ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtApi_ExperimentalApiTest, ...) produces:
//   inline OrtExperimental_OrtApi_ExperimentalApiTest_SinceV28_Fn
//   Get_OrtApi_ExperimentalApiTest_SinceV28_Fn(const OrtApi* api) {
//     return reinterpret_cast<OrtExperimental_OrtApi_ExperimentalApiTest_SinceV28_Fn>(
//         api->GetExperimentalFunction(kOrtExperimental_OrtApi_ExperimentalApiTest_SinceV28_FnName));
//   }
#define ORT_EXPERIMENTAL_API(VER, RET, NAME, ...)                                      \
  inline OrtExperimental_##NAME##_SinceV##VER##_Fn Get_##NAME##_SinceV##VER##_Fn(      \
      const OrtApi* api) {                                                             \
    return reinterpret_cast<OrtExperimental_##NAME##_SinceV##VER##_Fn>(                \
        api->GetExperimentalFunction(kOrtExperimental_##NAME##_SinceV##VER##_FnName)); \
  }

#include "onnxruntime_experimental_c_api.inc"

#undef ORT_EXPERIMENTAL_API

//
// C++ wrapper types for the experimental APIs
//

// Move-only RAII owner for an OrtEpContextConfig handle obtained from OrtEpApi_SessionOptions_GetEpContextConfig.
// Releases the handle via OrtEpApi_ReleaseEpContextConfig when destroyed.
//
// Note: intentionally lightweight and exception-free so this header (which only depends on the C API) can be included
// by ORT-internal code without pulling in onnxruntime_cxx_api.h.
class EpContextConfig {
 public:
  EpContextConfig() noexcept = default;

  // Takes ownership of `config`. `api` is used to release it; both may be null (an empty wrapper).
  EpContextConfig(const OrtApi* api, OrtEpContextConfig* config) noexcept : api_{api}, config_{config} {}

  EpContextConfig(EpContextConfig&& other) noexcept : api_{other.api_}, config_{other.config_} {
    other.api_ = nullptr;
    other.config_ = nullptr;
  }

  EpContextConfig& operator=(EpContextConfig&& other) noexcept {
    if (this != &other) {
      reset();
      api_ = other.api_;
      config_ = other.config_;
      other.api_ = nullptr;
      other.config_ = nullptr;
    }
    return *this;
  }

  EpContextConfig(const EpContextConfig&) = delete;
  EpContextConfig& operator=(const EpContextConfig&) = delete;

  ~EpContextConfig() { reset(); }

  OrtEpContextConfig* get() const noexcept { return config_; }
  explicit operator bool() const noexcept { return config_ != nullptr; }

  // Relinquishes ownership of the handle without releasing it.
  OrtEpContextConfig* release() noexcept {
    OrtEpContextConfig* released = config_;
    api_ = nullptr;
    config_ = nullptr;
    return released;
  }

  // Releases any owned handle and resets to empty.
  void reset() noexcept {
    if (api_ != nullptr && config_ != nullptr) {
      if (auto* release_fn = Get_OrtEpApi_ReleaseEpContextConfig_SinceV28_Fn(api_)) {
        release_fn(config_);
      }
    }
    api_ = nullptr;
    config_ = nullptr;
  }

 private:
  const OrtApi* api_ = nullptr;
  OrtEpContextConfig* config_ = nullptr;
};

}  // namespace Experimental
}  // namespace Ort

#endif  // __cplusplus
