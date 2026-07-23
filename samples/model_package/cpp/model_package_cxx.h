// ---------------------------------------------------------------------------
// C++ convenience wrappers for the (experimental) ONNX Runtime model-package API.
//
// These wrappers give the model-package API the same look and feel as the rest of
// the ORT C++ API (Ort::Env, Ort::SessionOptions, Ort::Session): RAII objects and
// member functions, no raw function-pointer juggling.
//
// Today the underlying C functions are *experimental* and are resolved by name via
// OrtApi::GetExperimentalFunction(). When they are promoted to the stable OrtApi,
// only the private ApiTable below changes (it will call ort.ModelPackageXxx()
// directly); the Ort::ModelPackage / Ort::ModelPackageComponent classes and all
// calling code stay exactly the same.
//
// Usage:
//   Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "app"};
//   Ort::ModelPackage pkg{ORT_TSTR("path/to/package")};
//   for (const std::string& name : pkg.ComponentNames()) {
//     Ort::SessionOptions so;
//     so.AppendExecutionProvider_V2(env, {device}, {});          // or leave empty for CPU
//     Ort::ModelPackageComponent comp = pkg.SelectComponent(env, name, so);
//     Ort::Session session = comp.CreateSession(env);
//     // ... run inference on `session` ...
//   }
// ---------------------------------------------------------------------------
#pragma once

#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"
#include "onnxruntime_experimental_c_api.h"

namespace Ort {

namespace detail {

// Loads and caches the experimental OrtModelPackageApi_* function pointers.
// When these are promoted to the stable OrtApi this whole struct collapses to
// direct `Ort::GetApi().ModelPackageXxx` accessors.
struct ModelPackageApiTable {
  OrtExperimental_OrtModelPackageApi_CreateModelPackageOptionsFromSessionOptions_SinceV28_Fn create_options;
  OrtExperimental_OrtModelPackageApi_ReleaseModelPackageOptions_SinceV28_Fn release_options;
  OrtExperimental_OrtModelPackageApi_CreateModelPackageContext_SinceV28_Fn create_context;
  OrtExperimental_OrtModelPackageApi_ReleaseModelPackageContext_SinceV28_Fn release_context;
  OrtExperimental_OrtModelPackageApi_ModelPackage_GetComponentNames_SinceV28_Fn get_component_names;
  OrtExperimental_OrtModelPackageApi_SelectComponent_SinceV28_Fn select_component;
  OrtExperimental_OrtModelPackageApi_ReleaseModelPackageComponentContext_SinceV28_Fn release_component;
  OrtExperimental_OrtModelPackageApi_ModelPackageComponent_GetSelectedVariantName_SinceV28_Fn get_variant_name;
  OrtExperimental_OrtModelPackageApi_CreateSession_SinceV28_Fn create_session;

  static const ModelPackageApiTable& Get() {
    static const ModelPackageApiTable table;  // thread-safe, loaded once
    return table;
  }

 private:
  template <typename Fn>
  static Fn Load(const char* name) {
    auto* fn = reinterpret_cast<Fn>(GetApi().GetExperimentalFunction(name));
    if (fn == nullptr) {
      throw Ort::Exception(std::string("Model package API is not available in this ONNX Runtime build "
                                       "(missing experimental function: ") +
                               name + "). Requires onnxruntime >= 1.28.",
                           ORT_FAIL);
    }
    return fn;
  }

  ModelPackageApiTable()
      : create_options(Load<decltype(create_options)>(
            kOrtExperimental_OrtModelPackageApi_CreateModelPackageOptionsFromSessionOptions_SinceV28_FnName)),
        release_options(Load<decltype(release_options)>(
            kOrtExperimental_OrtModelPackageApi_ReleaseModelPackageOptions_SinceV28_FnName)),
        create_context(Load<decltype(create_context)>(
            kOrtExperimental_OrtModelPackageApi_CreateModelPackageContext_SinceV28_FnName)),
        release_context(Load<decltype(release_context)>(
            kOrtExperimental_OrtModelPackageApi_ReleaseModelPackageContext_SinceV28_FnName)),
        get_component_names(Load<decltype(get_component_names)>(
            kOrtExperimental_OrtModelPackageApi_ModelPackage_GetComponentNames_SinceV28_FnName)),
        select_component(Load<decltype(select_component)>(
            kOrtExperimental_OrtModelPackageApi_SelectComponent_SinceV28_FnName)),
        release_component(Load<decltype(release_component)>(
            kOrtExperimental_OrtModelPackageApi_ReleaseModelPackageComponentContext_SinceV28_FnName)),
        get_variant_name(Load<decltype(get_variant_name)>(
            kOrtExperimental_OrtModelPackageApi_ModelPackageComponent_GetSelectedVariantName_SinceV28_FnName)),
        create_session(Load<decltype(create_session)>(
            kOrtExperimental_OrtModelPackageApi_CreateSession_SinceV28_FnName)) {}
};

}  // namespace detail

/// \brief A selected component: the winning variant plus a factory for its session.
///
/// Returned by ModelPackage::SelectComponent. Move-only; releases the underlying
/// context on destruction.
class ModelPackageComponent {
 public:
  explicit ModelPackageComponent(std::nullptr_t) {}
  explicit ModelPackageComponent(OrtModelPackageComponentContext* p) : p_{p} {}

  ModelPackageComponent(ModelPackageComponent&& o) noexcept : p_{o.p_} { o.p_ = nullptr; }
  ModelPackageComponent& operator=(ModelPackageComponent&& o) noexcept {
    if (this != &o) {
      Reset();
      p_ = o.p_;
      o.p_ = nullptr;
    }
    return *this;
  }
  ModelPackageComponent(const ModelPackageComponent&) = delete;
  ModelPackageComponent& operator=(const ModelPackageComponent&) = delete;
  ~ModelPackageComponent() { Reset(); }

  /// \brief Name of the variant that was selected for the configured EPs.
  std::string SelectedVariantName() const {
    const char* name = nullptr;
    Ort::ThrowOnError(detail::ModelPackageApiTable::Get().get_variant_name(p_, &name));
    return name ? std::string(name) : std::string();
  }

  /// \brief Create a session for the selected variant.
  ///
  /// ORT merges the variant's session/provider options declared in the package
  /// (e.g. ep.share_ep_contexts, the external-initializers folder, ep.context_file_path).
  Ort::Session CreateSession(const Ort::Env& env) const {
    OrtSession* session = nullptr;
    Ort::ThrowOnError(detail::ModelPackageApiTable::Get().create_session(
        env, p_, /*session_options*/ nullptr, &session));
    return Ort::Session{session};
  }

 private:
  void Reset() {
    if (p_ != nullptr) {
      detail::ModelPackageApiTable::Get().release_component(p_);
      p_ = nullptr;
    }
  }
  OrtModelPackageComponentContext* p_ = nullptr;
};

/// \brief An opened model package: a directory of components, each with per-EP variants.
///
/// Move-only; releases the underlying context on destruction.
class ModelPackage {
 public:
  /// \brief Open (parse) the package at `package_root`.
  explicit ModelPackage(const std::basic_string<ORTCHAR_T>& package_root) {
    Ort::ThrowOnError(detail::ModelPackageApiTable::Get().create_context(package_root.c_str(), &p_));
  }

  ModelPackage(ModelPackage&& o) noexcept : p_{o.p_} { o.p_ = nullptr; }
  ModelPackage& operator=(ModelPackage&& o) noexcept {
    if (this != &o) {
      Reset();
      p_ = o.p_;
      o.p_ = nullptr;
    }
    return *this;
  }
  ModelPackage(const ModelPackage&) = delete;
  ModelPackage& operator=(const ModelPackage&) = delete;
  ~ModelPackage() { Reset(); }

  /// \brief Names of the components in this package.
  std::vector<std::string> ComponentNames() const {
    const char* const* names = nullptr;
    size_t count = 0;
    Ort::ThrowOnError(detail::ModelPackageApiTable::Get().get_component_names(p_, &names, &count));
    std::vector<std::string> result;
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) result.emplace_back(names[i]);
    return result;
  }

  /// \brief Select the best variant of `component_name` for the EPs configured on
  ///        `session_options` (append EP devices, or leave empty to select the CPU
  ///        variant). Throws if no compatible variant is found.
  ModelPackageComponent SelectComponent(const Ort::Env& env, const std::string& component_name,
                                        const Ort::SessionOptions& session_options) const {
    const auto& api = detail::ModelPackageApiTable::Get();
    OrtModelPackageOptions* options = nullptr;
    Ort::ThrowOnError(api.create_options(env, session_options, &options));
    OrtModelPackageComponentContext* component = nullptr;
    OrtStatus* status = api.select_component(p_, component_name.c_str(), options, &component);
    api.release_options(options);
    Ort::ThrowOnError(status);
    return ModelPackageComponent{component};
  }

 private:
  void Reset() {
    if (p_ != nullptr) {
      detail::ModelPackageApiTable::Get().release_context(p_);
      p_ = nullptr;
    }
  }
  OrtModelPackageContext* p_ = nullptr;
};

}  // namespace Ort
