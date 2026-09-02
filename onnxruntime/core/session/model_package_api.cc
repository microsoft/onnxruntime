// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_experimental_c_api.h"

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

using namespace onnxruntime;

#define RETURN_NOT_IMPL_IN_MINIMAL_BUILD()          \
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, \
                               "Model package API is not supported in this build")

namespace OrtExperimentalApis {

ORT_API(void, OrtModelPackageApi_ReleaseModelPackageOptions_SinceV28,
        _Frees_ptr_opt_ OrtModelPackageOptions* options) {
#if !defined(ORT_MINIMAL_BUILD)
  delete reinterpret_cast<onnxruntime::ModelPackageOptions*>(options);
#else
  ORT_UNUSED_PARAMETER(options);
#endif
}

ORT_API_STATUS_IMPL(OrtModelPackageApi_CreateModelPackageOptionsFromSessionOptions_SinceV28,
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

ORT_API(void, OrtModelPackageApi_ReleaseModelPackageContext_SinceV28,
        _Frees_ptr_opt_ OrtModelPackageContext* ctx) {
#if !defined(ORT_MINIMAL_BUILD)
  delete reinterpret_cast<onnxruntime::ModelPackageContext*>(ctx);
#else
  ORT_UNUSED_PARAMETER(ctx);
#endif
}

ORT_API_STATUS_IMPL(OrtModelPackageApi_CreateModelPackageContext_SinceV28,
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

ORT_API_STATUS_IMPL(OrtModelPackageApi_ModelPackage_GetComponentCount_SinceV28,
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

ORT_API_STATUS_IMPL(OrtModelPackageApi_ModelPackage_GetComponentNames_SinceV28,
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

  const char* const* ptrs = nullptr;
  size_t count = 0;
  reinterpret_cast<const onnxruntime::ModelPackageContext*>(ctx)->GetComponentNamePtrs(ptrs, count);
  *out_names = ptrs;
  *out_count = count;
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ctx);
  ORT_UNUSED_PARAMETER(out_names);
  ORT_UNUSED_PARAMETER(out_count);
  RETURN_NOT_IMPL_IN_MINIMAL_BUILD();
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelPackageApi_ModelPackage_GetVariantCount_SinceV28,
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

ORT_API_STATUS_IMPL(OrtModelPackageApi_ModelPackage_GetVariantNames_SinceV28,
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

  const char* const* ptrs = nullptr;
  size_t count = 0;
  reinterpret_cast<const onnxruntime::ModelPackageContext*>(ctx)->GetVariantNamePtrs(component_name, ptrs, count);
  *out_variant_names = ptrs;
  *out_count = count;
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

ORT_API_STATUS_IMPL(OrtModelPackageApi_SelectComponent_SinceV28,
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
      component_name, *component_info, *cxx_options);

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

ORT_API(void, OrtModelPackageApi_ReleaseModelPackageComponentContext_SinceV28,
        _Frees_ptr_opt_ OrtModelPackageComponentContext* cix) {
#if !defined(ORT_MINIMAL_BUILD)
  delete reinterpret_cast<onnxruntime::ModelPackageComponentContext*>(cix);
#else
  ORT_UNUSED_PARAMETER(cix);
#endif
}

ORT_API_STATUS_IMPL(OrtModelPackageApi_ModelPackageComponent_GetSelectedVariantFolderPath_SinceV28,
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

ORT_API_STATUS_IMPL(OrtModelPackageApi_CreateSession_SinceV28,
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

  // 1) Get the selected variant model file path.
  //    ORT only supports a single ONNX file per variant folder.
  std::filesystem::path selected_file_path;
  ORT_API_RETURN_IF_STATUS_NOT_OK(mp_ctx.GetSelectedVariantFilePath(selected_file_path));

  // 2) Pick the OrtSessionOptions per precedence rules:
  //    - session_options == nullptr (default path): start from a clean OrtSessionOptions,
  //      and merge variant-specific session + provider options from the package metadata.
  //    - session_options != nullptr (advanced path): use caller-supplied as-is, no metadata merge.
  const OrtSessionOptions* effective_options = nullptr;
  std::optional<OrtSessionOptions> effective_options_storage;

  if (session_options == nullptr) {
    // Start from a clean session options. The only EP-related state comes from
    // the variant metadata (session options and provider options), not from the
    // original session options used to create the package options.
    effective_options_storage.emplace();

    // Merge variant file session options into config options.
    gsl::span<const std::string> session_option_keys;
    gsl::span<const std::string> session_option_values;
    ORT_API_RETURN_IF_STATUS_NOT_OK(
        mp_ctx.GetSelectedVariantFileSessionOptions(session_option_keys, session_option_values));

    ORT_API_RETURN_IF(session_option_keys.size() != session_option_values.size(),
                      ORT_FAIL, "Session option keys/values size mismatch.");

    for (size_t i = 0; i < session_option_keys.size(); ++i) {
      OrtStatus* st = OrtApis::AddSessionConfigEntry(&*effective_options_storage,
                                                     session_option_keys[i].c_str(),
                                                     session_option_values[i].c_str());
      if (st != nullptr) {
        return st;
      }
    }

    // Merge variant file provider options as flat key/value entries for the selected EP devices.
    gsl::span<const std::string> provider_option_keys;
    gsl::span<const std::string> provider_option_values;
    ORT_API_RETURN_IF_STATUS_NOT_OK(
        mp_ctx.GetSelectedVariantFileProviderOptions(provider_option_keys, provider_option_values));

    ORT_API_RETURN_IF(provider_option_keys.size() != provider_option_values.size(),
                      ORT_FAIL, "Provider option keys/values size mismatch.");

    if (!provider_option_keys.empty()) {
      // Use ep_devices from the captured EP info for applying provider options.
      // DevicesSelected() is only populated for the policy path, but ep_infos[0].ep_devices
      // is populated for both factory and policy paths.
      const auto& ep_infos = mp_ctx.EpInfos();
      if (!ep_infos.empty() && !ep_infos[0].ep_devices.empty()) {
        std::vector<const char*> provider_option_key_ptrs;
        std::vector<const char*> provider_option_value_ptrs;
        provider_option_key_ptrs.reserve(provider_option_keys.size());
        provider_option_value_ptrs.reserve(provider_option_values.size());

        for (size_t i = 0; i < provider_option_keys.size(); ++i) {
          provider_option_key_ptrs.push_back(provider_option_keys[i].c_str());
          provider_option_value_ptrs.push_back(provider_option_values[i].c_str());
        }

        ORT_API_RETURN_IF_STATUS_NOT_OK(onnxruntime::AddEpOptionsToSessionOptions(
            gsl::span<const OrtEpDevice* const>(ep_infos[0].ep_devices.data(), ep_infos[0].ep_devices.size()),
            gsl::span<const char* const>(provider_option_key_ptrs.data(), provider_option_key_ptrs.size()),
            gsl::span<const char* const>(provider_option_value_ptrs.data(), provider_option_value_ptrs.size()),
            effective_options_storage->value));
      }
    }

    effective_options = &*effective_options_storage;
  } else {
    // Advanced path: use the caller-supplied options. Still carry over the variant's path-valued
    // session options (e.g. the external initializers folder the model needs to load), but only
    // for keys the caller did not set, so an explicit user value wins.
    gsl::span<const std::string> session_option_keys;
    gsl::span<const std::string> session_option_values;
    ORT_API_RETURN_IF_STATUS_NOT_OK(
        mp_ctx.GetSelectedVariantFileSessionOptions(session_option_keys, session_option_values));
    ORT_API_RETURN_IF(session_option_keys.size() != session_option_values.size(),
                      ORT_FAIL, "Session option keys/values size mismatch.");

    effective_options_storage.emplace(*session_options);
    const auto& existing = effective_options_storage->value.config_options.GetConfigOptionsMap();
    for (size_t i = 0; i < session_option_keys.size(); ++i) {
      if (!onnxruntime::IsModelPackagePathSessionOption(session_option_keys[i]) ||
          existing.count(session_option_keys[i]) != 0) {
        continue;
      }
      OrtStatus* st = OrtApis::AddSessionConfigEntry(&*effective_options_storage,
                                                     session_option_keys[i].c_str(),
                                                     session_option_values[i].c_str());
      if (st != nullptr) {
        return st;
      }
    }
    effective_options = &*effective_options_storage;
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

ORT_API_STATUS_IMPL(OrtModelPackageApi_ModelPackage_GetVariantEpName_SinceV28,
                    _In_ const OrtModelPackageContext* ctx,
                    _In_ const char* component_name,
                    _In_ const char* variant_name,
                    _Outptr_result_maybenull_ const char** out_ep) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  if (ctx == nullptr || component_name == nullptr || variant_name == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "ctx, component_name, and variant_name must be non-null");
  }

  const onnxruntime::VariantEpCompatibilityInfo* info = nullptr;
  auto status = reinterpret_cast<const onnxruntime::ModelPackageContext*>(ctx)->GetVariantEpCompatibility(
      component_name, variant_name, info);
  if (!status.IsOK()) {
    if (out_ep != nullptr) *out_ep = nullptr;
    return onnxruntime::ToOrtStatus(status);
  }

  if (out_ep != nullptr) {
    *out_ep = (info != nullptr && info->ep.has_value()) ? info->ep->c_str() : nullptr;
  }
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ctx);
  ORT_UNUSED_PARAMETER(component_name);
  ORT_UNUSED_PARAMETER(variant_name);
  ORT_UNUSED_PARAMETER(out_ep);
  RETURN_NOT_IMPL_IN_MINIMAL_BUILD();
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelPackageApi_ModelPackage_ResolveStringRef_SinceV28,
                    _In_ const OrtModelPackageContext* ctx,
                    _In_opt_ const char* base_dir,
                    _In_ const char* input,
                    _In_ int must_exist,
                    _Outptr_ const char** out_path) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  if (ctx == nullptr || input == nullptr || out_path == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "ctx, input, and out_path must be non-null");
  }
  *out_path = nullptr;

  const char* resolved = nullptr;
  auto status = reinterpret_cast<const onnxruntime::ModelPackageContext*>(ctx)->ResolveStringRef(
      base_dir != nullptr ? std::string(base_dir) : std::string{}, std::string(input),
      must_exist != 0, resolved);
  if (!status.IsOK()) {
    return onnxruntime::ToOrtStatus(status);
  }
  *out_path = resolved;
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ctx);
  ORT_UNUSED_PARAMETER(base_dir);
  ORT_UNUSED_PARAMETER(input);
  ORT_UNUSED_PARAMETER(must_exist);
  ORT_UNUSED_PARAMETER(out_path);
  RETURN_NOT_IMPL_IN_MINIMAL_BUILD();
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelPackageApi_ModelPackage_GetSchemaVersion_SinceV28,
                    _In_ const OrtModelPackageContext* ctx,
                    _Out_ int64_t* out_version) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  if (ctx == nullptr || out_version == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "ctx and out_version must be non-null");
  }

  const auto& package_info = reinterpret_cast<const onnxruntime::ModelPackageContext*>(ctx)->GetModelPackageInfo();
  *out_version = package_info.schema_version;
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ctx);
  ORT_UNUSED_PARAMETER(out_version);
  RETURN_NOT_IMPL_IN_MINIMAL_BUILD();
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelPackageApi_ModelPackageComponent_GetSelectedVariantName_SinceV28,
                    _In_ const OrtModelPackageComponentContext* ctx,
                    _Outptr_ const char** out_name) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  if (ctx == nullptr || out_name == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "ctx and out_name must be non-null");
  }

  const auto* mp_ctx = reinterpret_cast<const onnxruntime::ModelPackageComponentContext*>(ctx);
  const std::string* name = nullptr;
  ORT_API_RETURN_IF_STATUS_NOT_OK(mp_ctx->GetSelectedVariantName(name));
  ORT_API_RETURN_IF(name == nullptr, ORT_FAIL, "Selected variant name is null.");

  *out_name = name->c_str();
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ctx);
  ORT_UNUSED_PARAMETER(out_name);
  RETURN_NOT_IMPL_IN_MINIMAL_BUILD();
#endif
  API_IMPL_END
}

}  // namespace OrtExperimentalApis
