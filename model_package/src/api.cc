// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "model_package_api.h"
#include "model_package_internal.h"
#include "parser.h"

#include <new>
#include <string>

// ─────────────────────────────────────────────────────────────────────────────
// Status implementation
// ─────────────────────────────────────────────────────────────────────────────

struct ModelPackageStatus {
  std::string message;
};

static ModelPackageStatus* MakeError(std::string msg) {
  return new (std::nothrow) ModelPackageStatus{std::move(msg)};
}

// ─────────────────────────────────────────────────────────────────────────────
// Context is the public opaque type wrapping ContextImpl
// ─────────────────────────────────────────────────────────────────────────────

struct ModelPackageContext {
  model_package::ContextImpl impl;
};

// ─────────────────────────────────────────────────────────────────────────────
// ContextImpl lookup helpers
// ─────────────────────────────────────────────────────────────────────────────

namespace model_package {

const Component* ContextImpl::FindComponent(const char* name) const {
  for (const auto& c : package_info.components) {
    if (c.name == name) return &c;
  }
  return nullptr;
}

const Variant* ContextImpl::FindVariant(const char* component_name, const char* variant_name) const {
  const auto* comp = FindComponent(component_name);
  if (!comp) return nullptr;
  for (const auto& v : comp->variants) {
    if (v.name == variant_name) return &v;
  }
  return nullptr;
}

}  // namespace model_package

// ─────────────────────────────────────────────────────────────────────────────
// Validation macro
// ─────────────────────────────────────────────────────────────────────────────

#define RETURN_IF_NULL(ptr, param_name)                                       \
  do {                                                                        \
    if ((ptr) == nullptr)                                                     \
      return MakeError(std::string(param_name) + " must not be null.");       \
  } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// C API implementation
// ─────────────────────────────────────────────────────────────────────────────

extern "C" {

void ModelPackage_ReleaseStatus(ModelPackageStatus* status) {
  delete status;
}

const char* ModelPackage_GetErrorMessage(const ModelPackageStatus* status) {
  if (status == nullptr) return nullptr;
  return status->message.c_str();
}

ModelPackageStatus* ModelPackage_CreateContext(
    const char* package_root_path,
    ModelPackageContext** out_context) {
  RETURN_IF_NULL(package_root_path, "package_root_path");
  RETURN_IF_NULL(out_context, "out_context");

  *out_context = nullptr;

  auto ctx = std::make_unique<ModelPackageContext>();
  std::string error;

  if (!model_package::ParsePackage(
          std::filesystem::path(std::string(package_root_path)),
          ctx->impl.package_info, error)) {
    return MakeError(std::move(error));
  }

  // Build component names cache.
  ctx->impl.component_names_cache.clear();
  for (const auto& c : ctx->impl.package_info.components) {
    ctx->impl.component_names_cache.push_back(c.name);
  }

  // Build variant names cache.
  for (const auto& c : ctx->impl.package_info.components) {
    auto& names = ctx->impl.variant_names_cache[c.name];
    names.clear();
    for (const auto& v : c.variants) {
      names.push_back(v.name);
    }
  }

  *out_context = ctx.release();
  return nullptr;
}

void ModelPackage_ReleaseContext(ModelPackageContext* context) {
  delete context;
}

ModelPackageStatus* ModelPackage_GetSchemaVersion(
    const ModelPackageContext* context,
    int64_t* out_version) {
  RETURN_IF_NULL(context, "context");
  RETURN_IF_NULL(out_version, "out_version");
  *out_version = context->impl.package_info.schema_version;
  return nullptr;
}

ModelPackageStatus* ModelPackage_GetComponentCount(
    const ModelPackageContext* context,
    size_t* out_count) {
  RETURN_IF_NULL(context, "context");
  RETURN_IF_NULL(out_count, "out_count");
  *out_count = context->impl.package_info.components.size();
  return nullptr;
}

ModelPackageStatus* ModelPackage_GetComponentName(
    const ModelPackageContext* context,
    size_t component_idx,
    const char** out_name) {
  RETURN_IF_NULL(context, "context");
  RETURN_IF_NULL(out_name, "out_name");

  if (component_idx >= context->impl.component_names_cache.size()) {
    return MakeError("component_idx out of range: " + std::to_string(component_idx));
  }

  *out_name = context->impl.component_names_cache[component_idx].c_str();
  return nullptr;
}

ModelPackageStatus* ModelPackage_GetVariantCount(
    const ModelPackageContext* context,
    const char* component_name,
    size_t* out_count) {
  RETURN_IF_NULL(context, "context");
  RETURN_IF_NULL(component_name, "component_name");
  RETURN_IF_NULL(out_count, "out_count");

  const auto* comp = context->impl.FindComponent(component_name);
  if (!comp) {
    return MakeError(std::string("Component not found: '") + component_name + "'.");
  }

  *out_count = comp->variants.size();
  return nullptr;
}

ModelPackageStatus* ModelPackage_GetVariantName(
    const ModelPackageContext* context,
    const char* component_name,
    size_t variant_idx,
    const char** out_name) {
  RETURN_IF_NULL(context, "context");
  RETURN_IF_NULL(component_name, "component_name");
  RETURN_IF_NULL(out_name, "out_name");

  auto it = context->impl.variant_names_cache.find(component_name);
  if (it == context->impl.variant_names_cache.end()) {
    return MakeError(std::string("Component not found: '") + component_name + "'.");
  }

  if (variant_idx >= it->second.size()) {
    return MakeError("variant_idx out of range: " + std::to_string(variant_idx));
  }

  *out_name = it->second[variant_idx].c_str();
  return nullptr;
}

ModelPackageStatus* ModelPackage_GetVariantFolderPath(
    const ModelPackageContext* context,
    const char* component_name,
    const char* variant_name,
    const char** out_path) {
  RETURN_IF_NULL(context, "context");
  RETURN_IF_NULL(component_name, "component_name");
  RETURN_IF_NULL(variant_name, "variant_name");
  RETURN_IF_NULL(out_path, "out_path");

  const auto* variant = context->impl.FindVariant(component_name, variant_name);
  if (!variant) {
    return MakeError(std::string("Variant '") + variant_name + "' not found in component '" +
                     component_name + "'.");
  }

  // Cache the path string for stable pointer.
  std::string cache_key = std::string(component_name) + "/" + variant_name;
  auto& cached = const_cast<ModelPackageContext*>(context)->impl.folder_path_strings_cache[cache_key];
  if (cached.empty()) {
    cached = variant->folder_path.string();
  }
  *out_path = cached.c_str();
  return nullptr;
}

ModelPackageStatus* ModelPackage_GetVariantEpCompatibilityCount(
    const ModelPackageContext* context,
    const char* component_name,
    const char* variant_name,
    size_t* out_count) {
  RETURN_IF_NULL(context, "context");
  RETURN_IF_NULL(component_name, "component_name");
  RETURN_IF_NULL(variant_name, "variant_name");
  RETURN_IF_NULL(out_count, "out_count");

  const auto* variant = context->impl.FindVariant(component_name, variant_name);
  if (!variant) {
    return MakeError(std::string("Variant '") + variant_name + "' not found in component '" +
                     component_name + "'.");
  }

  *out_count = variant->ep_compatibility.size();
  return nullptr;
}

ModelPackageStatus* ModelPackage_GetVariantEpCompatibility(
    const ModelPackageContext* context,
    const char* component_name,
    const char* variant_name,
    size_t ep_idx,
    const char** out_ep,
    const char** out_device,
    const char** out_compatibility_string) {
  RETURN_IF_NULL(context, "context");
  RETURN_IF_NULL(component_name, "component_name");
  RETURN_IF_NULL(variant_name, "variant_name");

  const auto* variant = context->impl.FindVariant(component_name, variant_name);
  if (!variant) {
    return MakeError(std::string("Variant '") + variant_name + "' not found in component '" +
                     component_name + "'.");
  }

  if (ep_idx >= variant->ep_compatibility.size()) {
    return MakeError("ep_idx out of range: " + std::to_string(ep_idx));
  }

  const auto& ec = variant->ep_compatibility[ep_idx];
  if (out_ep) *out_ep = ec.ep.has_value() ? ec.ep->c_str() : nullptr;
  if (out_device) *out_device = ec.device.has_value() ? ec.device->c_str() : nullptr;
  if (out_compatibility_string) {
    *out_compatibility_string = ec.compatibility_string.has_value()
                                    ? ec.compatibility_string->c_str()
                                    : nullptr;
  }
  return nullptr;
}

ModelPackageStatus* ModelPackage_GetVariantFileCount(
    const ModelPackageContext* context,
    const char* component_name,
    const char* variant_name,
    size_t* out_count) {
  RETURN_IF_NULL(context, "context");
  RETURN_IF_NULL(component_name, "component_name");
  RETURN_IF_NULL(variant_name, "variant_name");
  RETURN_IF_NULL(out_count, "out_count");

  const auto* variant = context->impl.FindVariant(component_name, variant_name);
  if (!variant) {
    return MakeError(std::string("Variant '") + variant_name + "' not found in component '" +
                     component_name + "'.");
  }

  *out_count = variant->files.size();
  return nullptr;
}

ModelPackageStatus* ModelPackage_GetVariantFilePath(
    const ModelPackageContext* context,
    const char* component_name,
    const char* variant_name,
    size_t file_idx,
    const char** out_path) {
  RETURN_IF_NULL(context, "context");
  RETURN_IF_NULL(component_name, "component_name");
  RETURN_IF_NULL(variant_name, "variant_name");
  RETURN_IF_NULL(out_path, "out_path");

  const auto* variant = context->impl.FindVariant(component_name, variant_name);
  if (!variant) {
    return MakeError(std::string("Variant '") + variant_name + "' not found in component '" +
                     component_name + "'.");
  }

  if (file_idx >= variant->files.size()) {
    return MakeError("file_idx out of range: " + std::to_string(file_idx));
  }

  // Cache the path string for stable pointer.
  auto cache_key = model_package::ContextImpl::MakeCacheKey(component_name, variant_name, file_idx);
  auto& cached = const_cast<ModelPackageContext*>(context)->impl.file_path_strings_cache[cache_key];
  if (cached.empty()) {
    cached = variant->files[file_idx].resolved_path.string();
  }
  *out_path = cached.c_str();
  return nullptr;
}

// Helper to build/retrieve options cache
static ModelPackageStatus* GetOptionsFromMap(
    const ModelPackageContext* context,
    const char* component_name,
    const char* variant_name,
    size_t file_idx,
    bool is_session_options,
    const char* const** out_keys,
    const char* const** out_values,
    size_t* out_count) {
  RETURN_IF_NULL(context, "context");
  RETURN_IF_NULL(component_name, "component_name");
  RETURN_IF_NULL(variant_name, "variant_name");
  RETURN_IF_NULL(out_keys, "out_keys");
  RETURN_IF_NULL(out_values, "out_values");
  RETURN_IF_NULL(out_count, "out_count");

  const auto* variant = context->impl.FindVariant(component_name, variant_name);
  if (!variant) {
    return MakeError(std::string("Variant '") + variant_name + "' not found in component '" +
                     component_name + "'.");
  }

  if (file_idx >= variant->files.size()) {
    return MakeError("file_idx out of range: " + std::to_string(file_idx));
  }

  const auto& opts = is_session_options
                         ? variant->files[file_idx].session_options
                         : variant->files[file_idx].provider_options;

  if (!opts.has_value() || opts->empty()) {
    *out_keys = nullptr;
    *out_values = nullptr;
    *out_count = 0;
    return nullptr;
  }

  auto cache_key = model_package::ContextImpl::MakeCacheKey(component_name, variant_name, file_idx);
  auto& cache_map = is_session_options
                        ? const_cast<ModelPackageContext*>(context)->impl.session_options_cache
                        : const_cast<ModelPackageContext*>(context)->impl.provider_options_cache;

  auto it = cache_map.find(cache_key);
  if (it == cache_map.end()) {
    model_package::ContextImpl::OptionsCache cache{};
    cache.keys.reserve(opts->size());
    cache.values.reserve(opts->size());
    for (const auto& kv : *opts) {
      cache.keys.push_back(kv.first);
      cache.values.push_back(kv.second);
    }
    cache.key_ptrs.reserve(cache.keys.size());
    cache.value_ptrs.reserve(cache.values.size());
    for (size_t i = 0; i < cache.keys.size(); ++i) {
      cache.key_ptrs.push_back(cache.keys[i].c_str());
      cache.value_ptrs.push_back(cache.values[i].c_str());
    }
    it = cache_map.emplace(std::move(cache_key), std::move(cache)).first;
  }

  *out_keys = it->second.key_ptrs.data();
  *out_values = it->second.value_ptrs.data();
  *out_count = it->second.key_ptrs.size();
  return nullptr;
}

ModelPackageStatus* ModelPackage_GetVariantFileSessionOptions(
    const ModelPackageContext* context,
    const char* component_name,
    const char* variant_name,
    size_t file_idx,
    const char* const** out_keys,
    const char* const** out_values,
    size_t* out_count) {
  return GetOptionsFromMap(context, component_name, variant_name, file_idx,
                           /*is_session_options=*/true, out_keys, out_values, out_count);
}

ModelPackageStatus* ModelPackage_GetVariantFileProviderOptions(
    const ModelPackageContext* context,
    const char* component_name,
    const char* variant_name,
    size_t file_idx,
    const char* const** out_keys,
    const char* const** out_values,
    size_t* out_count) {
  return GetOptionsFromMap(context, component_name, variant_name, file_idx,
                           /*is_session_options=*/false, out_keys, out_values, out_count);
}

ModelPackageStatus* ModelPackage_GetVariantConsumerMetadata(
    const ModelPackageContext* context,
    const char* component_name,
    const char* variant_name,
    const char** out_json) {
  RETURN_IF_NULL(context, "context");
  RETURN_IF_NULL(component_name, "component_name");
  RETURN_IF_NULL(variant_name, "variant_name");
  RETURN_IF_NULL(out_json, "out_json");

  const auto* variant = context->impl.FindVariant(component_name, variant_name);
  if (!variant) {
    return MakeError(std::string("Variant '") + variant_name + "' not found in component '" +
                     component_name + "'.");
  }

  if (!variant->consumer_metadata_json.has_value()) {
    *out_json = nullptr;
  } else {
    *out_json = variant->consumer_metadata_json->c_str();
  }
  return nullptr;
}

}  // extern "C"
