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
#endif

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
  ORT_UNUSED_PARAMETER(package_root);
  ORT_UNUSED_PARAMETER(out);
  RETURN_NOT_IMPL_IN_MINIMAL_BUILD();
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelPackageAPI::ModelPackage_GetComponentCount,
                    _In_ const OrtModelPackageContext* ctx,
                    _Out_ size_t* out_count) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  if (ctx == nullptr || out_count == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "ctx and out_count must be non-null");
  }
  *out_count = reinterpret_cast<const onnxruntime::ModelPackageContext*>(ctx)->GetComponentCount();
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ctx);
  ORT_UNUSED_PARAMETER(out_count);
  RETURN_NOT_IMPL_IN_MINIMAL_BUILD();
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelPackageAPI::ModelPackage_GetComponentNames,
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
      reinterpret_cast<const onnxruntime::ModelPackageContext*>(ctx)->GetComponentNames(names));

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

ORT_API_STATUS_IMPL(OrtModelPackageAPI::ModelPackage_GetVariantCount,
                    _In_ const OrtModelPackageContext* ctx,
                    _In_ const char* component_name,
                    _Out_ size_t* out_count) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  if (ctx == nullptr || component_name == nullptr || out_count == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "ctx, component_name, and out_count must be non-null");
  }

  ORT_API_RETURN_IF_STATUS_NOT_OK(
      reinterpret_cast<const onnxruntime::ModelPackageContext*>(ctx)->GetVariantCount(component_name, *out_count));
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ctx);
  ORT_UNUSED_PARAMETER(component_name);
  ORT_UNUSED_PARAMETER(out_count);
  RETURN_NOT_IMPL_IN_MINIMAL_BUILD();
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelPackageAPI::ModelPackage_GetVariantNames,
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
      reinterpret_cast<const onnxruntime::ModelPackageContext*>(ctx)->GetVariantNames(component_name, variant_names));

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

ORT_API_STATUS_IMPL(OrtModelPackageAPI::SelectComponent,
                    _In_ const OrtModelPackageContext* context,
                    _In_ const char* component_name,
                    _In_ const OrtModelPackageOptions* options,
                    _Outptr_ OrtModelPackageComponentContext** out) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  if (context == nullptr || component_name == nullptr || options == nullptr || out == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "context, component_name, options, and out must be non-null");
  }

  const auto* cxx_ctx = reinterpret_cast<const onnxruntime::ModelPackageContext*>(context);
  const auto* cxx_options = reinterpret_cast<const onnxruntime::ModelPackageOptions*>(options);

  const auto& package_info = cxx_ctx->GetModelPackageInfo();
  const ComponentInfo* component_info = nullptr;
  for (const auto& component : package_info.components) {
    if (component.component_name == component_name) {
      component_info = &component;
      break;
    }
  }

  if (component_info == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Component model not found.");
  }

  auto cix = std::make_unique<onnxruntime::ModelPackageComponentContext>(
      component_name, *component_info, cxx_options);

  ORT_API_RETURN_IF_STATUS_NOT_OK(cix->ResolveVariant());

  *out = reinterpret_cast<OrtModelPackageComponentContext*>(cix.release());
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(context);
  ORT_UNUSED_PARAMETER(component_name);
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(out);
  RETURN_NOT_IMPL_IN_MINIMAL_BUILD();
#endif
  API_IMPL_END
}

ORT_API(void, OrtModelPackageAPI::ReleaseModelPackageComponentContext,
        _Frees_ptr_opt_ OrtModelPackageComponentContext* cix) {
#if !defined(ORT_MINIMAL_BUILD)
  delete reinterpret_cast<onnxruntime::ModelPackageComponentContext*>(cix);
#else
  ORT_UNUSED_PARAMETER(cix);
#endif
}

ORT_API_STATUS_IMPL(OrtModelPackageAPI::ModelPackageComponent_GetSelectedVariantFolderPath,
                    _In_ const OrtModelPackageComponentContext* ctx,
                    _Outptr_ const ORTCHAR_T** folder_path) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  if (ctx == nullptr || folder_path == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "ctx and folder_path must be non-null");
  }

  const auto* cxx_ctx = reinterpret_cast<const onnxruntime::ModelPackageComponentContext*>(ctx);

  const std::filesystem::path* folder = nullptr;
  ORT_API_RETURN_IF_STATUS_NOT_OK(cxx_ctx->GetSelectedVariantFolderPath(folder));
  ORT_API_RETURN_IF(folder == nullptr, ORT_FAIL, "Selected variant folder path is null.");

  *folder_path = folder->c_str();
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ctx);
  ORT_UNUSED_PARAMETER(folder_path);
  RETURN_NOT_IMPL_IN_MINIMAL_BUILD();
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelPackageAPI::ModelPackageComponent_GetSelectedVariantFileCount,
                    _In_ const OrtModelPackageComponentContext* ctx,
                    _Out_ size_t* num_files) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  if (ctx == nullptr || num_files == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "ctx and num_files must be non-null");
  }

  const auto* cxx_ctx = reinterpret_cast<const onnxruntime::ModelPackageComponentContext*>(ctx);

  gsl::span<const std::filesystem::path> file_paths;
  ORT_API_RETURN_IF_STATUS_NOT_OK(cxx_ctx->GetSelectedVariantFilePaths(file_paths));

  *num_files = file_paths.size();
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ctx);
  ORT_UNUSED_PARAMETER(num_files);
  RETURN_NOT_IMPL_IN_MINIMAL_BUILD();
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelPackageAPI::ModelPackageComponent_GetSelectedVariantFilePath,
                    _In_ const OrtModelPackageComponentContext* ctx,
                    _In_ size_t file_idx,
                    _Outptr_ const ORTCHAR_T** out_path) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  if (ctx == nullptr || out_path == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "ctx and out_path must be non-null");
  }

  const auto* cxx_ctx = reinterpret_cast<const onnxruntime::ModelPackageComponentContext*>(ctx);

  gsl::span<const std::filesystem::path> file_paths;
  ORT_API_RETURN_IF_STATUS_NOT_OK(cxx_ctx->GetSelectedVariantFilePaths(file_paths));

  if (file_idx >= file_paths.size()) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "file_idx out of range");
  }

  // Pointer lifetime is owned by ModelPackageComponentContext cache.
  *out_path = file_paths[file_idx].c_str();
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ctx);
  ORT_UNUSED_PARAMETER(file_idx);
  ORT_UNUSED_PARAMETER(out_path);
  RETURN_NOT_IMPL_IN_MINIMAL_BUILD();
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelPackageAPI::ModelPackageComponent_GetSelectedVariantFileSessionOptions,
                    _In_ const OrtModelPackageComponentContext* ctx,
                    _In_ size_t file_idx,
                    _Outptr_result_buffer_maybenull_(*num_entries) const char* const** option_keys,
                    _Outptr_result_buffer_maybenull_(*num_entries) const char* const** option_values,
                    _Out_ size_t* num_entries) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  if (ctx == nullptr || option_keys == nullptr || option_values == nullptr || num_entries == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid null argument.");
  }

  gsl::span<const std::string> keys;
  gsl::span<const std::string> values;
  ORT_API_RETURN_IF_STATUS_NOT_OK(
      reinterpret_cast<const onnxruntime::ModelPackageComponentContext*>(ctx)->GetSelectedVariantFileSessionOptions(
          file_idx, keys, values));

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
  ORT_UNUSED_PARAMETER(ctx);
  ORT_UNUSED_PARAMETER(file_idx);
  ORT_UNUSED_PARAMETER(option_keys);
  ORT_UNUSED_PARAMETER(option_values);
  ORT_UNUSED_PARAMETER(num_entries);
  RETURN_NOT_IMPL_IN_MINIMAL_BUILD();
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelPackageAPI::ModelPackageComponent_GetSelectedVariantFileProviderOptions,
                    _In_ const OrtModelPackageComponentContext* ctx,
                    _In_ size_t file_idx,
                    _Outptr_result_buffer_maybenull_(*num_entries) const char* const** option_keys,
                    _Outptr_result_buffer_maybenull_(*num_entries) const char* const** option_values,
                    _Out_ size_t* num_entries) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  if (ctx == nullptr || option_keys == nullptr || option_values == nullptr || num_entries == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid null argument.");
  }

  gsl::span<const std::string> keys;
  gsl::span<const std::string> values;
  ORT_API_RETURN_IF_STATUS_NOT_OK(
      reinterpret_cast<const onnxruntime::ModelPackageComponentContext*>(ctx)->GetSelectedVariantFileProviderOptions(
          file_idx, keys, values));

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
  ORT_UNUSED_PARAMETER(ctx);
  ORT_UNUSED_PARAMETER(file_idx);
  ORT_UNUSED_PARAMETER(option_keys);
  ORT_UNUSED_PARAMETER(option_values);
  ORT_UNUSED_PARAMETER(num_entries);
  RETURN_NOT_IMPL_IN_MINIMAL_BUILD();
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelPackageAPI::CreateSession,
                    _In_ const OrtEnv* env,
                    _In_ OrtModelPackageComponentContext* ctx,
                    _In_opt_ const OrtSessionOptions* session_options,
                    _Outptr_ OrtSession** session) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  if (env == nullptr || ctx == nullptr || session == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "env, ctx, and session must be non-null");
  }

  auto& mp_ctx = *reinterpret_cast<onnxruntime::ModelPackageComponentContext*>(ctx);
  const auto* mp_ctx_options = mp_ctx.Options();
  if (mp_ctx_options == nullptr) {
    return OrtApis::CreateStatus(ORT_FAIL, "ModelPackageContext has no associated options.");
  }

  // 1) Get the selected variant model file path.
  //    Note: This API only supports single-file variants. For multi-file variants, the caller
  //          should use ModelPackageComponent_GetSelectedVariantFilePath to get individual file paths
  //          and create sessions accordingly.
  std::filesystem::path selected_file_path;
  ORT_API_RETURN_IF_STATUS_NOT_OK(mp_ctx.GetSelectedVariantFilePath(selected_file_path));

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
        mp_ctx.GetSelectedVariantFileSessionOptions(0, session_option_keys, session_option_values));

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
        mp_ctx.GetSelectedVariantFileProviderOptions(0, provider_option_keys, provider_option_values));

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
  ORT_API_RETURN_IF_ERROR(onnxruntime::CreateSessionForModelPackage(
      effective_options,
      env->GetEnvironment(),
      selected_file_path,
      mp_ctx,
      sess));

  // 4) Initialize.
  ORT_API_RETURN_IF_ERROR(InitializeSession(effective_options, *sess));

  *session = reinterpret_cast<OrtSession*>(sess.release());
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(env);
  ORT_UNUSED_PARAMETER(ctx);
  ORT_UNUSED_PARAMETER(session_options);
  ORT_UNUSED_PARAMETER(session);
  RETURN_NOT_IMPL_IN_MINIMAL_BUILD();
#endif
  API_IMPL_END
}

// ---------- API table ------------------------------------------------------

ORT_API_STATUS_IMPL(OrtModelPackageAPI::ModelPackage_GetVariantEpCompatibilityCount,
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
      reinterpret_cast<const onnxruntime::ModelPackageContext*>(ctx)->GetVariantEpCompatibilityCount(
          component_name, variant_name, *out_count));
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

ORT_API_STATUS_IMPL(OrtModelPackageAPI::ModelPackage_GetVariantEpCompatibility,
                    _In_ const OrtModelPackageContext* ctx,
                    _In_ const char* component_name,
                    _In_ const char* variant_name,
                    _In_ size_t ep_idx,
                    _Outptr_result_maybenull_ const char** out_ep,
                    _Outptr_result_maybenull_ const char** out_device,
                    _Outptr_result_maybenull_ const char** out_compatibility_string) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  if (ctx == nullptr || component_name == nullptr || variant_name == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "ctx, component_name, and variant_name must be non-null");
  }

  const onnxruntime::VariantEpCompatibilityInfo* info = nullptr;
  ORT_API_RETURN_IF_STATUS_NOT_OK(
      reinterpret_cast<const onnxruntime::ModelPackageContext*>(ctx)->GetVariantEpCompatibilityInfo(
          component_name, variant_name, ep_idx, info));

  if (out_ep != nullptr) {
    *out_ep = (info->ep.has_value()) ? info->ep->c_str() : nullptr;
  }
  if (out_device != nullptr) {
    *out_device = (info->device.has_value()) ? info->device->c_str() : nullptr;
  }
  if (out_compatibility_string != nullptr) {
    *out_compatibility_string = (info->compatibility_string.has_value())
                                    ? info->compatibility_string->c_str()
                                    : nullptr;
  }
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ctx);
  ORT_UNUSED_PARAMETER(component_name);
  ORT_UNUSED_PARAMETER(variant_name);
  ORT_UNUSED_PARAMETER(ep_idx);
  ORT_UNUSED_PARAMETER(out_ep);
  ORT_UNUSED_PARAMETER(out_device);
  ORT_UNUSED_PARAMETER(out_compatibility_string);
  RETURN_NOT_IMPL_IN_MINIMAL_BUILD();
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelPackageAPI::ModelPackageComponent_GetSelectedVariantConsumerMetadata,
                    _In_ const OrtModelPackageComponentContext* ctx,
                    _Outptr_ const char** out_json_str) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  if (ctx == nullptr || out_json_str == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "ctx and out_json_str must be non-null");
  }

  const std::string* s = nullptr;
  ORT_API_RETURN_IF_STATUS_NOT_OK(
      reinterpret_cast<const onnxruntime::ModelPackageComponentContext*>(ctx)
          ->GetSelectedVariantConsumerMetadata(s));
  ORT_API_RETURN_IF(s == nullptr, ORT_FAIL, "Consumer metadata accessor returned null.");

  *out_json_str = s->c_str();
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ctx);
  ORT_UNUSED_PARAMETER(out_json_str);
  RETURN_NOT_IMPL_IN_MINIMAL_BUILD();
#endif
  API_IMPL_END
}

// ---------- API table dispatch ---------------------------------------------

static constexpr OrtModelPackageApi ort_model_package_api = {
    // Options
    &OrtModelPackageAPI::CreateModelPackageOptionsFromSessionOptions,
    &OrtModelPackageAPI::ReleaseModelPackageOptions,

    // Context
    &OrtModelPackageAPI::CreateModelPackageContext,
    &OrtModelPackageAPI::ReleaseModelPackageContext,

    // Generic metadata queries
    &OrtModelPackageAPI::ModelPackage_GetComponentCount,
    &OrtModelPackageAPI::ModelPackage_GetComponentNames,
    &OrtModelPackageAPI::ModelPackage_GetVariantCount,
    &OrtModelPackageAPI::ModelPackage_GetVariantNames,

    // Variant selection and queries
    &OrtModelPackageAPI::SelectComponent,
    &OrtModelPackageAPI::ReleaseModelPackageComponentContext,
    &OrtModelPackageAPI::ModelPackageComponent_GetSelectedVariantFolderPath,
    &OrtModelPackageAPI::ModelPackageComponent_GetSelectedVariantFileCount,
    &OrtModelPackageAPI::ModelPackageComponent_GetSelectedVariantFilePath,
    &OrtModelPackageAPI::ModelPackageComponent_GetSelectedVariantFileSessionOptions,
    &OrtModelPackageAPI::ModelPackageComponent_GetSelectedVariantFileProviderOptions,

    // Session
    &OrtModelPackageAPI::CreateSession,

    // Pre-selection EP compatibility traversal (added after the initial 1.27 slots; appended to
    // keep existing offsets stable).
    &OrtModelPackageAPI::ModelPackage_GetVariantEpCompatibilityCount,
    &OrtModelPackageAPI::ModelPackage_GetVariantEpCompatibility,

    // Post-selection variant queries.
    &OrtModelPackageAPI::ModelPackageComponent_GetSelectedVariantConsumerMetadata,

    // End of Version 1.27 - DO NOT MODIFY ABOVE
};

static_assert(offsetof(OrtModelPackageApi, CreateSession) / sizeof(void*) == 15,
              "Size of initial OrtModelPackageApi cannot change");

ORT_API(const OrtModelPackageApi*, OrtModelPackageAPI::GetModelPackageApi) {
  return &ort_model_package_api;
}