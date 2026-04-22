// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/model_package_api.h"

#include "core/common/common.h"
#include "core/framework/error_code_helper.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ort_apis.h"
#include "core/session/ort_env.h"

#if !defined(ORT_MINIMAL_BUILD)
#include "core/session/model_package/model_package_context.h"
#include "core/session/model_package/model_package_options.h"
#endif

using namespace onnxruntime;

#define RETURN_NOT_IMPL_IN_MINIMAL_BUILD()                                                  \
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED,                                         \
                               "Model package API is not supported in this build")

ORT_API(void, OrtModelPackageAPI::ReleaseModelPackageOptions,
        _Frees_ptr_opt_ OrtModelPackageOptions* options) {
#if !defined(ORT_MINIMAL_BUILD)
  delete reinterpret_cast<onnxruntime::ModelPackageOptions*>(options);
#else
  ORT_UNUSED_PARAMETER(options);
#endif
}

ORT_API_STATUS_IMPL(OrtModelPackageAPI::CreateModelPackageOptionsFromSessionOptions,
                    _In_ const OrtEnv* env,
                    _In_ const OrtSessionOptions* session_options,
                    _Outptr_ OrtModelPackageOptions** out) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  if (env == nullptr || session_options == nullptr || out == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "env, session_options, and out must be non-null");
  }
  auto options = std::make_unique<onnxruntime::ModelPackageOptions>(*env, *session_options);
  ORT_API_RETURN_IF_STATUS_NOT_OK(options->ResolveEpSelection());
  *out = reinterpret_cast<OrtModelPackageOptions*>(options.release());
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(env);
  ORT_UNUSED_PARAMETER(session_options);
  ORT_UNUSED_PARAMETER(out);
  RETURN_NOT_IMPL_IN_MINIMAL_BUILD();
#endif
  API_IMPL_END
}

ORT_API(void, OrtModelPackageAPI::ReleaseModelPackageContext,
        _Frees_ptr_opt_ OrtModelPackageContext* ctx) {
#if !defined(ORT_MINIMAL_BUILD)
  delete reinterpret_cast<onnxruntime::ModelPackageContext*>(ctx);
#else
  ORT_UNUSED_PARAMETER(ctx);
#endif
}

ORT_API_STATUS_IMPL(OrtModelPackageAPI::CreateModelPackageContext,
                    _In_ const OrtEnv* env,
                    _In_ const ORTCHAR_T* package_root,
                    _In_ const OrtModelPackageOptions* options,
                    _Outptr_ OrtModelPackageContext** out) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  if (env == nullptr || package_root == nullptr || options == nullptr || out == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "env, package_root, options, and out must be non-null");
  }
  const auto& cxx_options = *reinterpret_cast<const onnxruntime::ModelPackageOptions*>(options);
  auto ctx = std::make_unique<onnxruntime::ModelPackageContext>(std::filesystem::path{package_root}, cxx_options);
  ORT_API_RETURN_IF_STATUS_NOT_OK(ctx->ResolveVariants());
  *out = reinterpret_cast<OrtModelPackageContext*>(ctx.release());
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(env);
  ORT_UNUSED_PARAMETER(package_root);
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(out);
  RETURN_NOT_IMPL_IN_MINIMAL_BUILD();
#endif
  API_IMPL_END
}

// ---------- Query APIs (skeletons) ----------------------------------------
// These forward to ModelPackageContext accessors. Fill in once the
// descriptor parser exposes per-component grouping + file identifiers.

ORT_API_STATUS_IMPL(OrtModelPackageAPI::ModelPackageContext_GetComponentModelCount,
                    _In_ const OrtModelPackageContext* ctx,
                    _Out_ size_t* out_count) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  if (ctx == nullptr || out_count == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "ctx and out_count must be non-null");
  }
  *out_count = reinterpret_cast<const onnxruntime::ModelPackageContext*>(ctx)->GetComponentModelCount();
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ctx);
  ORT_UNUSED_PARAMETER(out_count);
  RETURN_NOT_IMPL_IN_MINIMAL_BUILD();
#endif
  API_IMPL_END
}

// ModelPackageContext_GetComponentModelName, GetSelectedVariantInfo,
// GetSelectedVariantFileCount, GetSelectedVariantFileIdentifier all follow the
// same pattern: validate args, forward to the ModelPackageContext method,
// map onnxruntime::Status to OrtStatus*. (Omitted here for brevity — straight
// mechanical expansions of the pattern above.)

// ---------- CreateSession --------------------------------------------------

ORT_API_STATUS_IMPL(CreateSession,
                    _In_ const OrtEnv* env,
                    _In_ const OrtModelPackageContext* ctx,
                    _In_ const char* component_name,
                    _In_opt_ const char* file_identifier,
                    _In_opt_ const OrtSessionOptions* session_options,
                    _Outptr_ OrtSession** session) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  if (env == nullptr || ctx == nullptr || component_name == nullptr || session == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "env, ctx, component_name, and session must be non-null");
  }
  const auto& cxx_ctx = *reinterpret_cast<const onnxruntime::ModelPackageContext*>(ctx);

  // 1) Resolve the physical file to load.
  std::filesystem::path model_path;
  ORT_API_RETURN_IF_STATUS_NOT_OK(
      cxx_ctx.ResolveSelectedVariantFile(component_name, file_identifier, model_path));

  // 2) Build the OrtSessionOptions to use, per precedence rules in the doc:
  //    - session_options == nullptr: start from the one captured on ctx.Options()
  //      and merge variant-specific session + provider options from the package metadata.
  //    - session_options != nullptr: use as-is, skip metadata merge.
  OrtSessionOptions effective{};
  if (session_options == nullptr) {
    effective = cxx_ctx.Options()->SessionOptions();
    // TODO: merge variant-specific session and provider options declared in the
    // package metadata (component_name / variant / file_identifier) onto `effective`.
  } else {
    effective = *session_options;
    // Advanced path: do NOT merge variant metadata-supplied options.
  }

  // 3) Delegate to the standard session creation path.
  return OrtApis::CreateSession(env, model_path.c_str(), &effective, session);
#else
  ORT_UNUSED_PARAMETER(env);
  ORT_UNUSED_PARAMETER(ctx);
  ORT_UNUSED_PARAMETER(component_name);
  ORT_UNUSED_PARAMETER(file_identifier);
  ORT_UNUSED_PARAMETER(session_options);
  ORT_UNUSED_PARAMETER(session);
  RETURN_NOT_IMPL_IN_MINIMAL_BUILD();
#endif
  API_IMPL_END
}

// ---------- API table ------------------------------------------------------

static constexpr OrtModelPackageApi ort_model_package_api = {
    // Options
    &OrtModelPackageAPI::ReleaseModelPackageOptions,
    &OrtModelPackageAPI::CreateModelPackageOptionsFromSessionOptions,

    // Context
    &OrtModelPackageAPI::ReleaseModelPackageContext,
    &OrtModelPackageAPI::CreateModelPackageContext,

    // Query
    &OrtModelPackageAPI::ModelPackageContext_GetComponentModelCount,
    &OrtModelPackageAPI::ModelPackageContext_GetComponentModelName,
    &OrtModelPackageAPI::ModelPackageContext_GetSelectedVariantFileCount,
    &OrtModelPackageAPI::ModelPackageContext_GetSelectedVariantFileIdentifier,

    // Session
    &OrtModelPackageAPI::CreateSession,

    // End of Version X - DO NOT MODIFY ABOVE
};

static_assert(offsetof(OrtModelPackageApi, CreateSession) / sizeof(void*) == 8,
              "Size of initial OrtModelPackageApi cannot change");

ORT_API(const OrtModelPackageApi*, OrtModelPackageAPI::GetModelPackageApi) {
  return &ort_model_package_api;
}