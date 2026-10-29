// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include "python/onnxruntime_pybind_state_common.h"
#include "python/onnxruntime_pybind_exceptions.h"

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <pybind11/stl.h>

#include "core/common/common.h"
#include "core/common/path_string.h"
#include "core/framework/error_code_helper.h"
#include "model_package.h"

namespace onnxruntime::python {
namespace py = pybind11;

namespace {

// Translated to FileNotFoundError/OSError; a C++ type is safe to throw with the GIL released.
struct PackageOsError : std::runtime_error {
  PackageOsError(const char* message, bool not_found) : std::runtime_error(message), not_found_(not_found) {}
  bool not_found_;
};

void CheckPackageStatus(ModelPackageStatus* status) {
  if (status != nullptr) {
    std::unique_ptr<ModelPackageStatus, decltype(&ModelPackageStatus_Release)> owned_status{
        status, ModelPackageStatus_Release};
    const char* message = ModelPackageStatus_Message(owned_status.get());
    switch (ModelPackageStatus_Code(owned_status.get())) {
      case MODEL_PACKAGE_ERR_NOT_FOUND:
      case MODEL_PACKAGE_ERR_ASSET_MISSING:
        throw PackageOsError(message, true);
      case MODEL_PACKAGE_ERR_IO:
        throw PackageOsError(message, false);
      case MODEL_PACKAGE_ERR_SCHEMA:
      case MODEL_PACKAGE_ERR_VERSION:
      case MODEL_PACKAGE_ERR_PATH_CONFINEMENT:
      case MODEL_PACKAGE_ERR_ASSET_HASH_MISMATCH:
      case MODEL_PACKAGE_ERR_INVALID_ARG:
        throw std::invalid_argument(message);
      default:
        throw std::runtime_error(message);
    }
  }
}

const char* OptionalString(const std::optional<std::string>& value) {
  return value ? value->c_str() : nullptr;
}

py::object OptionalText(const char* value) {
  return value ? py::object(py::str(value)) : py::none();
}

// Copies the C info tree, whose pointers are invalidated by the next package mutation.
py::dict PackageInfoToDict(const ModelPackageInfo& info) {
  py::list components;
  for (size_t c = 0; c < info.num_components; ++c) {
    const ModelComponentInfo& component = info.components[c];
    py::list variants;
    for (size_t v = 0; v < component.num_variants; ++v) {
      const ModelVariantInfo& variant = component.variants[v];
      py::dict executor_info;
      for (size_t e = 0; e < variant.num_executor_infos; ++e) {
        executor_info[py::str(variant.executor_infos[e].namespace_key)] = py::str(variant.executor_infos[e].json);
      }
      py::dict variant_dict;
      variant_dict["name"] = OptionalText(variant.name);
      variant_dict["variant_directory"] = OptionalText(variant.variant_directory);
      variant_dict["ep"] = OptionalText(variant.ep);
      variant_dict["device"] = OptionalText(variant.device);
      variant_dict["compatibility_string"] = OptionalText(variant.compatibility_string);
      variant_dict["additional_metadata_json"] = OptionalText(variant.additional_metadata_json);
      variant_dict["executor_info"] = executor_info;
      variants.append(variant_dict);
    }
    py::dict component_dict;
    component_dict["name"] = OptionalText(component.name);
    component_dict["additional_metadata_json"] = OptionalText(component.additional_metadata_json);
    component_dict["variants"] = variants;
    components.append(component_dict);
  }

  py::list shared_assets;
  for (size_t a = 0; a < info.num_shared_assets; ++a) {
    py::dict asset;
    asset["uri"] = OptionalText(info.shared_assets[a].uri);
    asset["resolved_path"] = OptionalText(info.shared_assets[a].resolved_path);
    shared_assets.append(asset);
  }

  py::dict result;
  result["schema_version_major"] = info.schema_version_major;
  result["schema_version_minor"] = info.schema_version_minor;
  result["package_name"] = OptionalText(info.package_name);
  result["package_version"] = OptionalText(info.package_version);
  result["description"] = OptionalText(info.description);
  result["layout"] = OptionalText(info.layout);
  result["additional_metadata_json"] = OptionalText(info.additional_metadata_json);
  result["components"] = components;
  result["shared_assets"] = shared_assets;
  return result;
}

class PyModelPackage {
 public:
  using Handle = std::unique_ptr<ModelPackage, decltype(&ModelPackage_Close)>;

  explicit PyModelPackage(Handle package) : package_(std::move(package)) {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(PyModelPackage);

  ModelPackage* Get() const {
    if (!package_) {
      throw py::value_error("Model package is closed.");
    }
    return package_.get();
  }

  void Close() { package_.reset(); }

 private:
  Handle package_;
};

}  // namespace

void addModelPackageMethods(py::module& m) {
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const PackageOsError& e) {
      PyErr_SetString(e.not_found_ ? PyExc_FileNotFoundError : PyExc_OSError, e.what());
    }
  });

  py::enum_<ModelPackageWriteMode>(m, "ModelPackageWriteMode")
      .value("PRESERVE", MODEL_PACKAGE_WRITE_PRESERVE)
      .value("DENSE", MODEL_PACKAGE_WRITE_DENSE);

  py::class_<PyModelPackage>(m, "ModelPackage",
                            "Author model packages using the native model package library. JSON arguments are strings.")
      .def(py::init([]() {
        ModelPackage* raw = nullptr;
        CheckPackageStatus(ModelPackage_New(&raw));
        return std::make_unique<PyModelPackage>(PyModelPackage::Handle{raw, ModelPackage_Close});
      }))
      .def_static("open", [](const std::string& root, bool allow_external_paths,
                             bool follow_symlinks, bool strict_unknown_fields) {
        ModelPackageOpenOptions options{allow_external_paths, follow_symlinks, strict_unknown_fields};
        ModelPackage* raw = nullptr;
        CheckPackageStatus(ModelPackage_Open(root.c_str(), &options, &raw));
        return std::make_unique<PyModelPackage>(PyModelPackage::Handle{raw, ModelPackage_Close});
      }, py::arg("package_root"), py::arg("allow_external_paths") = false,
         py::arg("follow_symlinks") = true, py::arg("strict_unknown_fields") = true)
      .def("close", &PyModelPackage::Close)
      .def("__enter__", [](PyModelPackage& self) -> PyModelPackage& {
        self.Get();
        return self;
      }, py::return_value_policy::reference_internal)
      .def("__exit__", [](PyModelPackage& self, const py::object&, const py::object&, const py::object&) {
        self.Close();
      })
      .def("set_metadata", [](PyModelPackage& self, const std::optional<std::string>& name,
                              const std::optional<std::string>& version,
                              const std::optional<std::string>& description) {
        CheckPackageStatus(ModelPackage_SetMetadata(self.Get(), OptionalString(name), OptionalString(version),
                                                   OptionalString(description)));
      }, py::arg("name") = std::nullopt, py::arg("version") = std::nullopt,
         py::arg("description") = std::nullopt)
      .def("set_layout", [](PyModelPackage& self, const std::string& layout) {
        CheckPackageStatus(ModelPackage_SetLayout(self.Get(), layout.c_str()));
      }, py::arg("layout"))
      .def("set_component_inline", [](PyModelPackage& self, const std::string& name, const std::string& json) {
        CheckPackageStatus(ModelPackage_SetComponentInline(self.Get(), name.c_str(), json.c_str()));
      }, py::arg("name"), py::arg("component_json"))
      .def("set_component_external", [](PyModelPackage& self, const std::string& name, const std::string& path) {
        CheckPackageStatus(ModelPackage_SetComponentExternal(self.Get(), name.c_str(), path.c_str()));
      }, py::arg("name"), py::arg("path"))
      .def("remove_component", [](PyModelPackage& self, const std::string& name) {
        CheckPackageStatus(ModelPackage_RemoveComponent(self.Get(), name.c_str()));
      }, py::arg("name"))
      .def("set_variant", [](PyModelPackage& self, const std::string& component, const std::string& variant,
                             const std::string& json) {
        CheckPackageStatus(ModelPackage_SetVariant(self.Get(), component.c_str(), variant.c_str(), json.c_str()));
      }, py::arg("component_name"), py::arg("variant_name"), py::arg("variant_json"))
      .def("remove_variant", [](PyModelPackage& self, const std::string& component, const std::string& variant) {
        CheckPackageStatus(ModelPackage_RemoveVariant(self.Get(), component.c_str(), variant.c_str()));
      }, py::arg("component_name"), py::arg("variant_name"))
      .def("set_variant_executor_info_inline", [](PyModelPackage& self, const std::string& component,
                                                  const std::string& variant, const std::string& ns,
                                                  const std::string& json) {
        CheckPackageStatus(ModelPackage_SetVariantExecutorInfoInline(
            self.Get(), component.c_str(), variant.c_str(), ns.c_str(), json.c_str()));
      }, py::arg("component_name"), py::arg("variant_name"), py::arg("namespace_key"), py::arg("info_json"))
      .def("set_variant_executor_info_external", [](PyModelPackage& self, const std::string& component,
                                                    const std::string& variant, const std::string& ns,
                                                    const std::string& path) {
        CheckPackageStatus(ModelPackage_SetVariantExecutorInfoExternal(
            self.Get(), component.c_str(), variant.c_str(), ns.c_str(), path.c_str()));
      }, py::arg("component_name"), py::arg("variant_name"), py::arg("namespace_key"), py::arg("path"))
      .def("remove_variant_executor_info", [](PyModelPackage& self, const std::string& component,
                                              const std::string& variant, const std::string& ns) {
        CheckPackageStatus(ModelPackage_RemoveVariantExecutorInfo(
            self.Get(), component.c_str(), variant.c_str(), ns.c_str()));
      }, py::arg("component_name"), py::arg("variant_name"), py::arg("namespace_key"))
      .def("set_additional_metadata_json", [](PyModelPackage& self, const std::string& scope,
                                              const std::optional<std::string>& component,
                                              const std::optional<std::string>& variant,
                                              const std::optional<std::string>& json) {
        CheckPackageStatus(ModelPackage_SetAdditionalMetadataJson(
            self.Get(), scope.c_str(), OptionalString(component), OptionalString(variant), OptionalString(json)));
      }, py::arg("scope"), py::arg("component_name") = std::nullopt,
         py::arg("variant_name") = std::nullopt, py::arg("json") = std::nullopt)
      .def("add_shared_asset", [](PyModelPackage& self, const std::string& source,
                                  const std::optional<std::string>& expected_uri, bool copy_in) {
        const char* uri = nullptr;
        CheckPackageStatus(ModelPackage_AddSharedAsset(
            self.Get(), source.c_str(), OptionalString(expected_uri), copy_in, &uri));
        return std::string{uri};
      }, py::arg("source_dir"), py::arg("expected_uri") = std::nullopt, py::arg("copy_in") = true)
      .def("remove_shared_asset", [](PyModelPackage& self, const std::string& uri) {
        CheckPackageStatus(ModelPackage_RemoveSharedAsset(self.Get(), uri.c_str()));
      }, py::arg("uri"))
      .def("get_info", [](const PyModelPackage& self) {
        return PackageInfoToDict(*ModelPackage_Info(self.Get()));
      }, "Return package metadata, components, variants and shared assets as a dict.")
      .def("get_component_json", [](const PyModelPackage& self, const std::string& component) {
        const char* json = nullptr;
        CheckPackageStatus(ModelPackage_GetComponentJson(self.Get(), component.c_str(), &json));
        return std::string{json};
      }, py::arg("component_name"))
      .def("get_variant_json", [](const PyModelPackage& self, const std::string& component,
                                 const std::string& variant) {
        const char* json = nullptr;
        CheckPackageStatus(ModelPackage_GetVariantJson(self.Get(), component.c_str(), variant.c_str(), &json));
        return std::string{json};
      }, py::arg("component_name"), py::arg("variant_name"))
      .def("resolve_asset_uri", [](const PyModelPackage& self, const std::string& uri) {
        const char* path = nullptr;
        CheckPackageStatus(ModelPackage_ResolveAssetUri(self.Get(), uri.c_str(), &path));
        return std::string{path};
      }, py::arg("uri"))
      .def("resolve_string_ref", [](const PyModelPackage& self, const std::string& ref,
                                   const std::optional<std::string>& base, bool must_exist) {
        const char* path = nullptr;
        CheckPackageStatus(ModelPackage_ResolveStringRef(
            self.Get(), OptionalString(base), ref.c_str(), must_exist, &path));
        return std::string{path};
      }, py::arg("string_ref"), py::arg("base_dir") = std::nullopt, py::arg("must_exist") = false)
      .def_static("compute_directory_hash", [](const std::string& source) {
        const char* uri = nullptr;
        CheckPackageStatus(ModelPackage_ComputeDirectoryHash(source.c_str(), &uri));
        return std::string{uri};
      }, py::arg("source_dir"))
      .def("commit", [](PyModelPackage& self, const std::optional<std::string>& destination,
                        ModelPackageWriteMode mode) {
        py::gil_scoped_release release;
        CheckPackageStatus(ModelPackage_Commit(self.Get(), OptionalString(destination), mode));
      }, py::arg("dest_root") = std::nullopt, py::arg("mode") = MODEL_PACKAGE_WRITE_PRESERVE,
         "Commit in place, or to an empty destination. Shared asset sources must remain readable until commit.")
      .def("prune", [](PyModelPackage& self) {
        CheckPackageStatus(ModelPackage_Prune(self.Get()));
      });

  py::class_<Ort::ModelPackageOptions>(m, "ModelPackageOptions",
                                      "Execution provider configuration used for package variant selection.")
      .def(py::init([](const PySessionOptions& options) {
        auto snapshot = CreatePySessionOptionsSnapshot(options);
        OrtModelPackageOptions* result = nullptr;
        OrtPybindThrowIfError(ToStatusAndRelease(
            Ort::GetModelPackageApi().CreateModelPackageOptionsFromSessionOptions(
                GetOrtEnv(), &snapshot.options, &result)));
        return Ort::ModelPackageOptions{result};
      }), py::arg("sess_options"));

  py::class_<Ort::ModelPackageComponentContext>(m, "ModelPackageComponentContext",
                                               "A selected component variant from an existing model package.")
      .def("get_selected_variant_name", &Ort::ModelPackageComponentContext::GetSelectedVariantName)
      .def("get_selected_variant_folder_path", [](const Ort::ModelPackageComponentContext& self) {
        return PathToUTF8String(self.GetSelectedVariantFolderPath());
      })
      .def("_create_session", [](Ort::ModelPackageComponentContext& self, const PySessionOptions* options) {
        auto snapshot = options ? CreatePySessionOptionsSnapshot(*options) : PySessionOptionsSnapshot{};
        OrtSession* result = nullptr;
        {
          py::gil_scoped_release release;
          OrtPybindThrowIfError(ToStatusAndRelease(Ort::GetModelPackageApi().CreateSession(
            GetOrtEnv(), self, options ? &snapshot.options : nullptr, &result)));
        }
        // The stable C API's OrtSession handle owns an internal InferenceSession.
        std::unique_ptr<InferenceSession> session{reinterpret_cast<InferenceSession*>(result)};
        return std::make_unique<PyInferenceSession>(std::move(session), snapshot);
      }, py::arg("sess_options") = static_cast<const PySessionOptions*>(nullptr),
           "Create an initialized InferenceSession. None applies the package's session/provider options; "
           "explicit options override them, while missing package path options are still added.");

  py::class_<Ort::ModelPackageContext>(m, "ModelPackageContext", "Open and inspect an existing model package.")
      .def(py::init([](const std::string& root) {
        return Ort::ModelPackageContext{ToPathString(root).c_str()};
      }), py::arg("package_root"))
      .def("get_schema_version", &Ort::ModelPackageContext::GetSchemaVersion)
      .def("get_component_count", &Ort::ModelPackageContext::GetComponentCount)
      .def("get_component_names", &Ort::ModelPackageContext::GetComponentNames)
      .def("get_variant_count", &Ort::ModelPackageContext::GetVariantCount, py::arg("component_name"))
      .def("get_variant_names", &Ort::ModelPackageContext::GetVariantNames, py::arg("component_name"))
      .def("get_variant_ep_name", [](const Ort::ModelPackageContext& self, const std::string& component,
                                     const std::string& variant) -> std::optional<std::string> {
        const char* ep = self.GetVariantEpName(component.c_str(), variant.c_str());
        return ep ? std::optional<std::string>{ep} : std::nullopt;
      }, py::arg("component_name"), py::arg("variant_name"))
      .def("resolve_string_ref", [](const Ort::ModelPackageContext& self, const std::string& ref,
                                    const std::optional<std::string>& base, bool must_exist) {
        return self.ResolveStringRef(base ? base->c_str() : nullptr, ref.c_str(), must_exist);
      }, py::arg("string_ref"), py::arg("base_dir") = std::nullopt, py::arg("must_exist") = false)
      .def("select_component", [](const Ort::ModelPackageContext& self, const std::string& component_name,
                                  const Ort::ModelPackageOptions& options) {
        py::gil_scoped_release release;
        return self.SelectComponent(component_name.c_str(), options);
      }, py::arg("component_name"), py::arg("options"));
}

}  // namespace onnxruntime::python

#endif  // !defined(ORT_MINIMAL_BUILD)
