// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/model_package_api.h"

#include "core/common/common.h"
#include "core/framework/error_code_helper.h"
#include "core/session/inference_session.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ort_apis.h"
#include "core/session/ort_env.h"

#if !defined(ORT_MINIMAL_BUILD)
#include "core/session/model_package/model_package_context.h"
#include "core/session/model_package/model_package_options.h"
#include "core/session/utils.h"
#endif

namespace {
void BuildCStringArray(gsl::span<const std::string> in,
                       std::vector<const char*>& cache,
                       const char* const** out,
                       size_t* out_count) {
  cache.clear();
  cache.reserve(in.size());
  for (const auto& s : in) {
    cache.push_back(s.c_str());
  }

  *out_count = in.size();
  *out = cache.empty() ? nullptr : cache.data();
}
}  // namespace

using namespace onnxruntime;

#define RETURN_NOT_IMPL_IN_MINIMAL_BUILD()          \
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, \
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
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "env, session_options and out must be non-null");
  }

  auto options = std::make_unique<onnxruntime::ModelPackageOptions>(env->GetEnvironment(), *session_options);
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
                    _In_ const ORTCHAR_T* package_root,
                    _Outptr_ OrtModelPackageContext** out) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)

  if (package_root == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "package_root must be non-null");
  }

  if (out == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "out must be non-null");
  }

  auto ctx = std::make_unique<onnxruntime::ModelPackageContext>(std::filesystem::path{package_root});

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

ORT_API_STATUS_IMPL(OrtModelPackageAPI::ModelPackageContext_GetComponentModelNames,
                    _In_ const OrtModelPackageContext* ctx,
                    _Outptr_result_buffer_maybenull_(*out_count) const char* const** out_names,
                    _Out_ size_t* out_count) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  if (ctx == nullptr || out_names == nullptr || out_count == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "ctx, out_names, and out_count must be non-null");
  }

  gsl::span<const std::string> names;
  ORT_API_RETURN_IF_STATUS_NOT_OK(
      reinterpret_cast<const onnxruntime::ModelPackageContext*>(ctx)->GetComponentModelNames(names));

  static thread_local std::vector<const char*> name_ptrs;
  BuildCStringArray(names, name_ptrs, out_names, out_count);
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ctx);
  ORT_UNUSED_PARAMETER(out_names);
  ORT_UNUSED_PARAMETER(out_count);
  RETURN_NOT_IMPL_IN_MINIMAL_BUILD();
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelPackageAPI::ModelPackageContext_GetModelVariantCount,
                    _In_ const OrtModelPackageContext* ctx,
                    _In_ const char* component_name,
                    _Out_ size_t* out_count) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  if (ctx == nullptr || component_name == nullptr || out_count == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "ctx, component_name, and out_count must be non-null");
  }

  ORT_API_RETURN_IF_STATUS_NOT_OK(
      reinterpret_cast<const onnxruntime::ModelPackageContext*>(ctx)->GetModelVariantCount(component_name, *out_count));
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ctx);
  ORT_UNUSED_PARAMETER(component_name);
  ORT_UNUSED_PARAMETER(out_count);
  RETURN_NOT_IMPL_IN_MINIMAL_BUILD();
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelPackageAPI::ModelPackageContext_GetModelVariantNames,
                    _In_ const OrtModelPackageContext* ctx,
                    _In_ const char* component_name,
                    _Outptr_result_buffer_maybenull_(*out_count) const char* const** out_variant_names,
                    _Out_ size_t* out_count) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  if (ctx == nullptr || component_name == nullptr || out_variant_names == nullptr || out_count == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "ctx, component_name, out_variant_names, and out_count must be non-null");
  }

  gsl::span<const std::string> variant_names;
  ORT_API_RETURN_IF_STATUS_NOT_OK(
      reinterpret_cast<const onnxruntime::ModelPackageContext*>(ctx)->GetModelVariantNames(component_name, variant_names));

  static thread_local std::vector<const char*> variant_name_ptrs;
  BuildCStringArray(variant_names, variant_name_ptrs, out_variant_names, out_count);
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ctx);
  ORT_UNUSED_PARAMETER(component_name);
  ORT_UNUSED_PARAMETER(out_variant_names);
  ORT_UNUSED_PARAMETER(out_count);
  RETURN_NOT_IMPL_IN_MINIMAL_BUILD();
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelPackageAPI::ModelPackageContext_GetFileCount,
                    _In_ const OrtModelPackageContext* ctx,
                    _In_ const char* component_name,
                    _In_ const char* variant_name,
                    _Out_ size_t* out_count) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  if (ctx == nullptr || component_name == nullptr || variant_name == nullptr || out_count == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "ctx, component_name, variant_name, and out_count must be non-null");
  }

  ORT_API_RETURN_IF_STATUS_NOT_OK(
      reinterpret_cast<const onnxruntime::ModelPackageContext*>(ctx)->GetFileCount(component_name, variant_name, *out_count));
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ctx);
  ORT_UNUSED_PARAMETER(component_name);
  ORT_UNUSED_PARAMETER(variant_name);
  ORT_UNUSED_PARAMETER(out_count);
  RETURN_NOT_IMPL_IN_MINIMAL_BUILD();
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelPackageAPI::ModelPackageContext_GetFileIdentifiers,
                    _In_ const OrtModelPackageContext* ctx,
                    _In_ const char* component_name,
                    _In_ const char* variant_name,
                    _Outptr_result_buffer_maybenull_(*out_count) const char* const** out_file_identifiers,
                    _Out_ size_t* out_count) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  if (ctx == nullptr || component_name == nullptr || variant_name == nullptr ||
      out_file_identifiers == nullptr || out_count == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "ctx, component_name, variant_name, out_file_identifiers, and out_count must be non-null");
  }

  gsl::span<const std::string> file_ids;
  ORT_API_RETURN_IF_STATUS_NOT_OK(
      reinterpret_cast<const onnxruntime::ModelPackageContext*>(ctx)->GetFileIdentifiers(component_name, variant_name, file_ids));

  static thread_local std::vector<const char*> file_id_ptrs;
  BuildCStringArray(file_ids, file_id_ptrs, out_file_identifiers, out_count);
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ctx);
  ORT_UNUSED_PARAMETER(component_name);
  ORT_UNUSED_PARAMETER(variant_name);
  ORT_UNUSED_PARAMETER(out_file_identifiers);
  ORT_UNUSED_PARAMETER(out_count);
  RETURN_NOT_IMPL_IN_MINIMAL_BUILD();
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelPackageAPI::ModelPackageContext_GetFilePath,
                    _In_ const OrtModelPackageContext* ctx,
                    _In_ const char* component_name,
                    _In_ const char* variant_name,
                    _In_opt_ const char* file_identifier,
                    _Outptr_ const ORTCHAR_T** out_path) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  if (ctx == nullptr || component_name == nullptr || variant_name == nullptr || out_path == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "ctx, component_name, variant_name, and out_path must be non-null");
  }

  std::filesystem::path path;
  ORT_API_RETURN_IF_STATUS_NOT_OK(
      reinterpret_cast<const onnxruntime::ModelPackageContext*>(ctx)->GetFilePath(component_name, variant_name, file_identifier, path));

  static thread_local std::filesystem::path path_cache;
  path_cache = std::move(path);
  *out_path = path_cache.c_str();
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ctx);
  ORT_UNUSED_PARAMETER(component_name);
  ORT_UNUSED_PARAMETER(variant_name);
  ORT_UNUSED_PARAMETER(file_identifier);
  ORT_UNUSED_PARAMETER(out_path);
  RETURN_NOT_IMPL_IN_MINIMAL_BUILD();
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelPackageAPI::ResolveVariant,
                    _Inout_ OrtModelPackageContext* ctx,
                    _In_ const OrtModelPackageOptions* options) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  if (ctx == nullptr || options == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "ctx and options must be non-null");
  }

  auto* cxx_ctx = reinterpret_cast<onnxruntime::ModelPackageContext*>(ctx);
  const auto* cxx_options = reinterpret_cast<const onnxruntime::ModelPackageOptions*>(options);

  ORT_API_RETURN_IF_STATUS_NOT_OK(cxx_ctx->ResolveVariant(cxx_options));
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ctx);
  ORT_UNUSED_PARAMETER(options);
  RETURN_NOT_IMPL_IN_MINIMAL_BUILD();
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelPackageAPI::ModelPackageContext_GetSelectedVariantFileCount,
                    _In_ const OrtModelPackageContext* ctx,
                    _In_ const char* component_name,
                    _Out_ size_t* out_count) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  if (ctx == nullptr || component_name == nullptr || out_count == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "ctx, component_name, and out_count must be non-null");
  }

  gsl::span<const std::string> file_ids;
  ORT_API_RETURN_IF_STATUS_NOT_OK(
      reinterpret_cast<const onnxruntime::ModelPackageContext*>(ctx)->GetSelectedVariantFileIdentifiers(component_name, file_ids));

  *out_count = file_ids.size();
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ctx);
  ORT_UNUSED_PARAMETER(component_name);
  ORT_UNUSED_PARAMETER(out_count);
  RETURN_NOT_IMPL_IN_MINIMAL_BUILD();
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelPackageAPI::ModelPackageContext_GetSelectedVariantFileIdentifier,
                    _In_ const OrtModelPackageContext* ctx,
                    _In_ const char* component_name,
                    _In_ size_t index,
                    _Outptr_ const char** out_file_identifier) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  if (ctx == nullptr || component_name == nullptr || out_file_identifier == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "ctx, component_name, and out_file_identifier must be non-null");
  }

  gsl::span<const std::string> file_ids;
  ORT_API_RETURN_IF_STATUS_NOT_OK(
      reinterpret_cast<const onnxruntime::ModelPackageContext*>(ctx)->GetSelectedVariantFileIdentifiers(component_name, file_ids));

  if (index >= file_ids.size()) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "index out of range for selected variant files");
  }

  *out_file_identifier = file_ids[index].c_str();
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ctx);
  ORT_UNUSED_PARAMETER(component_name);
  ORT_UNUSED_PARAMETER(index);
  ORT_UNUSED_PARAMETER(out_file_identifier);
  RETURN_NOT_IMPL_IN_MINIMAL_BUILD();
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelPackageAPI::CreateSession,
                    _In_ const OrtEnv* env,
                    _In_ OrtModelPackageContext* ctx,
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

  auto& mp_ctx = *reinterpret_cast<onnxruntime::ModelPackageContext*>(ctx);
  const auto* mp_ctx_options = mp_ctx.Options();
  if (mp_ctx_options == nullptr) {
    return OrtApis::CreateStatus(ORT_FAIL, "ModelPackageContext has no associated options.");
  }

  // 1) Resolve the physical file to load (variant was already selected by CreateModelPackageContext).
  std::filesystem::path selected_model_path;
  ORT_API_RETURN_IF_STATUS_NOT_OK(
      mp_ctx.ResolveSelectedVariantFilePath(component_name, file_identifier, selected_model_path));

  // 2) Pick the OrtSessionOptions per precedence rules:
  //    - session_options == nullptr (default path): use the options captured on the context,
  //      and merge variant-specific session + provider options from the package metadata.
  //    - session_options != nullptr (advanced path): use caller-supplied as-is, no metadata merge.
  const OrtSessionOptions* effective_options = nullptr;
  std::optional<OrtSessionOptions> effective_options_storage;

  if (session_options == nullptr) {
    // Important: use copy-constructor, not assignment (operator= is not implemented).
    effective_options_storage.emplace(mp_ctx_options->SessionOptions());

    // Merge variant/file session options into config options.
    gsl::span<const std::string> session_option_keys;
    gsl::span<const std::string> session_option_values;
    ORT_API_RETURN_IF_STATUS_NOT_OK(
        mp_ctx.GetSelectedVariantFileSessionOptions(component_name, file_identifier,
                                                    session_option_keys, session_option_values));

    ORT_API_RETURN_IF(session_option_keys.size() != session_option_values.size(),
                      ORT_FAIL, "Session option keys/values size mismatch.");

    for (size_t i = 0; i < session_option_keys.size(); ++i) {
      ORT_API_RETURN_IF_STATUS_NOT_OK(
          effective_options_storage->value.config_options.AddConfigEntry(
              session_option_keys[i].c_str(), session_option_values[i].c_str()));
    }

    // Merge variant/file provider options as flat key/value entries for the selected EP devices.
    gsl::span<const std::string> provider_option_keys;
    gsl::span<const std::string> provider_option_values;
    ORT_API_RETURN_IF_STATUS_NOT_OK(
        mp_ctx.GetSelectedVariantFileProviderOptions(component_name, file_identifier,
                                                     provider_option_keys, provider_option_values));

    ORT_API_RETURN_IF(provider_option_keys.size() != provider_option_values.size(),
                      ORT_FAIL, "Provider option keys/values size mismatch.");

    if (!provider_option_keys.empty()) {
      std::vector<const char*> provider_option_key_ptrs;
      std::vector<const char*> provider_option_value_ptrs;
      provider_option_key_ptrs.reserve(provider_option_keys.size());
      provider_option_value_ptrs.reserve(provider_option_values.size());

      for (size_t i = 0; i < provider_option_keys.size(); ++i) {
        provider_option_key_ptrs.push_back(provider_option_keys[i].c_str());
        provider_option_value_ptrs.push_back(provider_option_values[i].c_str());
      }

      ORT_API_RETURN_IF_STATUS_NOT_OK(onnxruntime::AddEpOptionsToSessionOptions(
          gsl::span<const OrtEpDevice* const>(mp_ctx.DevicesSelected().data(), mp_ctx.DevicesSelected().size()),
          gsl::span<const char* const>(provider_option_key_ptrs.data(), provider_option_key_ptrs.size()),
          gsl::span<const char* const>(provider_option_value_ptrs.data(), provider_option_value_ptrs.size()),
          effective_options_storage->value));
    }

    effective_options = &*effective_options_storage;
  } else {
    effective_options = session_options;
  }

  // 3) Create session with the resolved file and effective session options.
  std::unique_ptr<onnxruntime::InferenceSession> sess;
  ORT_API_RETURN_IF_ERROR(onnxruntime::CreateSessionForResolvedModelPackage(
      effective_options,
      env->GetEnvironment(),
      selected_model_path,
      mp_ctx,
      sess));

  // 4) Initialize.
  ORT_API_RETURN_IF_ERROR(InitializeSession(effective_options, *sess));

  *session = reinterpret_cast<OrtSession*>(sess.release());
  return nullptr;
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

ORT_API_STATUS_IMPL(OrtModelPackageAPI::ModelPackageGetFileSessionOptions,
                    _In_ const OrtModelPackageContext* context,
                    _In_ const char* component_name,
                    _In_opt_ const char* file_identifier,
                    _Outptr_result_buffer_maybenull_(*num_entries) const char* const** option_keys,
                    _Outptr_result_buffer_maybenull_(*num_entries) const char* const** option_values,
                    _Out_ size_t* num_entries) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  if (context == nullptr || component_name == nullptr || option_keys == nullptr || option_values == nullptr || num_entries == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid null argument.");
  }

  gsl::span<const std::string> keys;
  gsl::span<const std::string> values;
  ORT_API_RETURN_IF_STATUS_NOT_OK(
      reinterpret_cast<const onnxruntime::ModelPackageContext*>(context)->GetSelectedVariantFileSessionOptions(
          component_name, file_identifier, keys, values));

  ORT_API_RETURN_IF(keys.size() != values.size(), ORT_FAIL, "Session options keys/values size mismatch.");
  *num_entries = keys.size();

  if (*num_entries == 0) {
    *option_keys = nullptr;
    *option_values = nullptr;
  } else {
    static thread_local std::vector<const char*> key_ptrs;
    static thread_local std::vector<const char*> value_ptrs;

    key_ptrs.clear();
    value_ptrs.clear();
    key_ptrs.reserve(keys.size());
    value_ptrs.reserve(values.size());

    for (size_t i = 0; i < keys.size(); ++i) {
      key_ptrs.push_back(keys[i].c_str());
      value_ptrs.push_back(values[i].c_str());
    }

    *option_keys = key_ptrs.data();
    *option_values = value_ptrs.data();
  }

  return nullptr;
#else
  ORT_UNUSED_PARAMETER(context);
  ORT_UNUSED_PARAMETER(component_name);
  ORT_UNUSED_PARAMETER(file_identifier);
  ORT_UNUSED_PARAMETER(option_keys);
  ORT_UNUSED_PARAMETER(option_values);
  ORT_UNUSED_PARAMETER(num_entries);
  RETURN_NOT_IMPL_IN_MINIMAL_BUILD();
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelPackageAPI::ModelPackageGetFileProviderOptions,
                    _In_ const OrtModelPackageContext* context,
                    _In_ const char* component_name,
                    _In_opt_ const char* file_identifier,
                    _Outptr_result_buffer_maybenull_(*num_entries) const char* const** option_keys,
                    _Outptr_result_buffer_maybenull_(*num_entries) const char* const** option_values,
                    _Out_ size_t* num_entries) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  if (context == nullptr || component_name == nullptr || option_keys == nullptr || option_values == nullptr || num_entries == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid null argument.");
  }

  gsl::span<const std::string> keys;
  gsl::span<const std::string> values;
  ORT_API_RETURN_IF_STATUS_NOT_OK(
      reinterpret_cast<const onnxruntime::ModelPackageContext*>(context)->GetSelectedVariantFileProviderOptions(
          component_name, file_identifier, keys, values));

  ORT_API_RETURN_IF(keys.size() != values.size(), ORT_FAIL, "Provider options keys/values size mismatch.");
  *num_entries = keys.size();

  if (*num_entries == 0) {
    *option_keys = nullptr;
    *option_values = nullptr;
  } else {
    static thread_local std::vector<const char*> key_ptrs;
    static thread_local std::vector<const char*> value_ptrs;

    key_ptrs.clear();
    value_ptrs.clear();
    key_ptrs.reserve(keys.size());
    value_ptrs.reserve(values.size());

    for (size_t i = 0; i < keys.size(); ++i) {
      key_ptrs.push_back(keys[i].c_str());
      value_ptrs.push_back(values[i].c_str());
    }

    *option_keys = key_ptrs.data();
    *option_values = value_ptrs.data();
  }

  return nullptr;
#else
  ORT_UNUSED_PARAMETER(context);
  ORT_UNUSED_PARAMETER(component_name);
  ORT_UNUSED_PARAMETER(file_identifier);
  ORT_UNUSED_PARAMETER(option_keys);
  ORT_UNUSED_PARAMETER(option_values);
  ORT_UNUSED_PARAMETER(num_entries);
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

    // Generic metadata queries
    &OrtModelPackageAPI::ModelPackageContext_GetComponentModelCount,
    &OrtModelPackageAPI::ModelPackageContext_GetComponentModelNames,
    &OrtModelPackageAPI::ModelPackageContext_GetModelVariantCount,
    &OrtModelPackageAPI::ModelPackageContext_GetModelVariantNames,
    &OrtModelPackageAPI::ModelPackageContext_GetFileCount,
    &OrtModelPackageAPI::ModelPackageContext_GetFileIdentifiers,
    &OrtModelPackageAPI::ModelPackageContext_GetFilePath,
    &OrtModelPackageAPI::ModelPackageGetFileSessionOptions,
    &OrtModelPackageAPI::ModelPackageGetFileProviderOptions,

    // Variant selection
    &OrtModelPackageAPI::ResolveVariant,
    &OrtModelPackageAPI::ModelPackageContext_GetSelectedVariantFileCount,
    &OrtModelPackageAPI::ModelPackageContext_GetSelectedVariantFileIdentifier,

    // Session
    &OrtModelPackageAPI::CreateSession,

    // End of Version 1.26 - DO NOT MODIFY ABOVE
};

static_assert(offsetof(OrtModelPackageApi, CreateSession) / sizeof(void*) == 16,
              "Size of initial OrtModelPackageApi cannot change");

ORT_API(const OrtModelPackageApi*, OrtModelPackageAPI::GetModelPackageApi) {
  return &ort_model_package_api;
}