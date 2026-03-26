// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "core/session/onnxruntime_cxx_api.h"
#include "test/util/include/asserts.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {
namespace {

// ------------------------------------------------------------------
// Helpers to build a test model package on disk
// ------------------------------------------------------------------

std::filesystem::path CreateManifestJson(const std::filesystem::path& package_root,
                                         std::string_view manifest_json) {
  std::filesystem::path manifest_path = package_root / "manifest.json";
  std::filesystem::create_directories(package_root);

  std::ofstream os(manifest_path, std::ios::binary);
  os << manifest_json;
  return manifest_path;
}

std::filesystem::path CreateModelPackage(
    const std::filesystem::path& package_root,
    std::string_view manifest_json,
    std::string_view component_model_name,
    std::string_view variant_name_1,
    std::string_view variant_name_2,
    const std::filesystem::path& source_model_1,
    const std::filesystem::path& source_model_2) {
  std::error_code ec;
  std::filesystem::remove_all(package_root, ec);
  std::filesystem::create_directories(package_root);

  CreateManifestJson(package_root, manifest_json);

  const auto models_root = package_root / "models" / component_model_name;
  const auto variant1_dir = models_root / variant_name_1;
  const auto variant2_dir = models_root / variant_name_2;

  std::filesystem::create_directories(variant1_dir);
  std::filesystem::create_directories(variant2_dir);

  const auto variant1_model = variant1_dir / source_model_1.filename();
  const auto variant2_model = variant2_dir / source_model_2.filename();

  std::filesystem::copy_file(source_model_1, variant1_model, std::filesystem::copy_options::overwrite_existing, ec);
  std::filesystem::copy_file(source_model_2, variant2_model, std::filesystem::copy_options::overwrite_existing, ec);
  return package_root;
}

std::filesystem::path CreateComponentModelMetadata(const std::filesystem::path& component_root,
                                                   std::string_view metadata_json) {
  std::filesystem::create_directories(component_root);

  const std::filesystem::path metadata_path = component_root / "metadata.json";
  std::ofstream os(metadata_path, std::ios::binary);
  os << metadata_json;
  return metadata_path;
}

std::filesystem::path CreateComponentModelsWithMetadata(
    const std::filesystem::path& package_root,
    std::string_view component_model_name,
    std::string_view metadata_json) {
  const auto component_root = package_root / "models" / component_model_name;
  std::error_code ec;

  // Ensure component root and metadata.json exist
  CreateComponentModelMetadata(component_root, metadata_json);

  return component_root;
}

}  // namespace

// ------------------------------------------------------------------
// OpenVINO EP Model Package Tests
// ------------------------------------------------------------------

// Test that model package correctly selects NPU variant when device_type is set to CPU
TEST(OpenVINOModelPackageTest, LoadModelPackage_CPU_DeviceConstraint) {
  // Build model package on disk with metadata.json containing two variants:
  // - variant_cpu: constrained to device "cpu"
  // - variant_gpu: constrained to device "gpu"
  const auto package_root = std::filesystem::temp_directory_path() / "ov_model_package_test_cpu";

  constexpr std::string_view manifest_json = R"({
    "name": "openvino_test_model",
    "component_models": {
      "model_1": {
      }
    }
  })";

  // Create model package with two different model files
  // For this test, we'll use the same model file but in real scenarios these would be different
  CreateModelPackage(package_root, manifest_json,
                     "model_1", "variant_cpu", "variant_gpu",
                     std::filesystem::path{"testdata/mul_1.onnx"},
                     std::filesystem::path{"testdata/mul_16.onnx"});

  // Create metadata.json with device constraints
  constexpr std::string_view metadata_json = R"({
    "model_variants": {
      "variant_cpu": {
        "file": "mul_1.onnx",
        "constraints": {
          "ep": "OpenVINOExecutionProvider",
          "device": "cpu"
        }
      },
      "variant_gpu": {
        "file": "mul_16.onnx",
        "constraints": {
          "ep": "OpenVINOExecutionProvider",
          "device": "gpu"
        }
      }
    }
  })";

  CreateComponentModelsWithMetadata(package_root, "model_1", metadata_json);

  // Prepare session options with OpenVINO EP configured for NPU
  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> ov_options;
  ov_options["device_type"] = "CPU";

  session_options.AppendExecutionProvider_OpenVINO_V2(ov_options);

  // Create session from package root (directory)
  // ORT should pick the variant_cpu model, i.e. mul_1.onnx since the device constraint matches "cpu"
  // If variant_gpu was selected and loaded, i.e. mul_16.onnx, session initialization would fail with
  // error "Error No Op registered for Mul16".
  Ort::Session session(*ort_env, package_root.c_str(), session_options);

  // Cleanup
  std::error_code ec;
  std::filesystem::remove_all(package_root, ec);
}

}  // namespace test
}  // namespace onnxruntime