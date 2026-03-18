// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <fstream>

#include "core/session/model_package_context.h"

namespace onnxruntime {
namespace {

}  // namespace

Status ModelPackageManifestParser::ParseManifest(const std::filesystem::path& package_root,
                                                   /*out*/ std::vector<EpContextVariantInfo>& components) {
  components.clear();
  const auto manifest_path = package_root / kModelPackageManifestFileName;
  if (!std::filesystem::exists(manifest_path)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "No manifest.json found at ", manifest_path.string());
  }

  std::ifstream f(manifest_path, std::ios::binary);
  if (!f) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to open manifest.json at ", manifest_path.string());
  }

  ORT_TRY {
    json doc = json::parse(f);
    if (!doc.is_object() || !doc.contains(kComponentsKey) || !doc[kComponentsKey].is_array()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "The \"components\" field in the manifest.json is missing or not an array");
    }

    for (const auto& comp : doc[kComponentsKey]) {
      if (!comp.is_object() || !comp.contains(kVariantNameKey) || !comp[kVariantNameKey].is_string()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "The \"variant_name\" field in a component is missing or not a string");
      }

      if (!comp.is_object() || !comp.contains(kFileKey) || !comp[kFileKey].is_string()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "The \"file\" field in a component is missing or not a string");
      }

      EpContextVariantInfo c;

      // variant name
      std::string variant_name = comp[kVariantNameKey].get<std::string>();
      c.metadata.Add(kVariantNameKey, variant_name);

      // Build model path: package_root / "models" / variant_name / file
      std::filesystem::path model_dir = package_root / "models" / variant_name;
      c.model_path = model_dir / comp[kFileKey].get<std::string>();

      if (comp.contains(kConstraintsKey) && comp[kConstraintsKey].is_object()) {
        const auto& cons = comp[kConstraintsKey];
        if (cons.contains(kEpKey) && cons[kEpKey].is_string()) c.ep = cons[kEpKey].get<std::string>();
        if (cons.contains(kDeviceKey) && cons[kDeviceKey].is_string()) c.device = cons[kDeviceKey].get<std::string>();
        if (cons.contains(kArchitectureKey) && cons[kArchitectureKey].is_string()) {
          c.architecture = cons[kArchitectureKey].get<std::string>();
        }
      }

      components.push_back(std::move(c));
    }
  }
  ORT_CATCH(const std::exception& ex) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "manifest.json is not valid JSON: ", ex.what());
  }

  for (const auto& c : components) {
    LOGS(logger_, INFO) << "manifest component: file='" << c.model_path.string()
                        << "' ep='" << c.ep << "' device='" << c.device
                        << "' arch='" << c.architecture << "'";
  }

  return Status::OK();
}

Status ModelPackageContext::SelectComponent(const Environment& env,
                                            gsl::span<EpContextVariantInfo> components,
                                            gsl::span<const OrtEpDevice* const> ep_devices,
                                            std::optional<std::filesystem::path>& selected_component_path) {
  
  

  return Status::OK();
}


}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
