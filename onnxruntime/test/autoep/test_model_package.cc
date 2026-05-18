// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"

#include "core/session/model_package/model_package_context.h"
#include "core/session/model_package/model_package_descriptor_parser.h"
#include "core/session/abi_devices.h"
#include "test/autoep/test_autoep_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/api_asserts.h"

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

std::string MakeVariantJson(std::string_view filename) {
  std::ostringstream oss;
  oss << R"({
    "files": [
      {
        "filename": ")"
      << filename << R"("
      }
    ]
  })";
  return oss.str();
}

void CreateVariantDescriptor(const std::filesystem::path& package_root,
                             std::string_view component_name,
                             std::string_view variant_name,
                             std::string_view variant_json) {
  const auto variant_root = package_root / "models" / std::string(component_name) / std::string(variant_name);
  std::filesystem::create_directories(variant_root);

  std::ofstream os(variant_root / "variant.json", std::ios::binary);
  os << variant_json;
}

std::filesystem::path CreateModelPackage(
    const std::filesystem::path& package_root,
    std::string_view manifest_json,
    std::string_view component_name,
    std::string_view variant_name_1,
    std::string_view variant_name_2,
    const std::filesystem::path& source_model_1,
    const std::filesystem::path& source_model_2) {
  std::error_code ec;
  std::filesystem::remove_all(package_root, ec);
  std::filesystem::create_directories(package_root);

  CreateManifestJson(package_root, manifest_json);

  const auto models_root = package_root / "models" / std::string(component_name);
  const auto variant1_dir = models_root / std::string(variant_name_1);
  const auto variant2_dir = models_root / std::string(variant_name_2);

  std::filesystem::create_directories(variant1_dir);
  std::filesystem::create_directories(variant2_dir);

  const auto variant1_model = variant1_dir / source_model_1.filename();
  const auto variant2_model = variant2_dir / source_model_2.filename();

  std::filesystem::copy_file(source_model_1, variant1_model, std::filesystem::copy_options::overwrite_existing, ec);
  std::filesystem::copy_file(source_model_2, variant2_model, std::filesystem::copy_options::overwrite_existing, ec);

  CreateVariantDescriptor(package_root, component_name, variant_name_1,
                          MakeVariantJson(source_model_1.filename().string()));
  CreateVariantDescriptor(package_root, component_name, variant_name_2,
                          MakeVariantJson(source_model_2.filename().string()));

  return package_root;
}

std::filesystem::path CreateComponentModelMetadata(
    const std::filesystem::path& package_root,
    std::string_view component_name,
    std::string_view metadata_json) {
  const auto component_root = package_root / "models" / std::string(component_name);

  std::filesystem::create_directories(component_root);

  const std::filesystem::path metadata_path = component_root / "metadata.json";
  std::ofstream os(metadata_path, std::ios::binary);
  os << metadata_json;

  return component_root;
}

std::string MakeManifestJson(std::string_view component_name) {
  std::ostringstream oss;
  oss << R"({
    "schema_version": 1,
    "components": [")"
      << component_name << R"("]
  })";
  return oss.str();
}

std::string MakeMetadataJsonTwoVariants(std::string_view component_name,
                                        std::string_view variant_name_1,
                                        std::string_view variant_ep_1,
                                        std::string_view variant_device_1,
                                        std::string_view variant_compatibility_string_1,
                                        std::string_view variant_name_2,
                                        std::string_view variant_ep_2,
                                        std::string_view variant_device_2,
                                        std::string_view variant_compatibility_string_2) {
  std::ostringstream oss;
  oss << R"({
    "component_name": ")"
      << component_name << R"(",
    "variants": {
      ")"
      << variant_name_1 << R"(": {
        "ep_compatibility": [{
          "ep": ")"
      << variant_ep_1 << R"(",
          "device": ")"
      << variant_device_1 << R"(",
          "compatibility_string": ")"
      << variant_compatibility_string_1 << R"("
        }]
      },
      ")"
      << variant_name_2 << R"(": {
        "ep_compatibility": [{
          "ep": ")"
      << variant_ep_2 << R"(",
          "device": ")"
      << variant_device_2 << R"(",
          "compatibility_string": ")"
      << variant_compatibility_string_2 << R"("
        }]
      }
    }
  })";
  return oss.str();
}

std::filesystem::path CreateModelPackageApiTestPackage(bool multi_file_variant = false) {
  const auto package_root = std::filesystem::temp_directory_path() / "ort_model_package_api_test";
  std::error_code ec;
  std::filesystem::remove_all(package_root, ec);

  constexpr std::string_view manifest_json = R"({
    "schema_version": 1,
    "components": ["model_1"]
  })";

  CreateModelPackage(package_root, manifest_json,
                     "model_1", "variant_1", "variant_2",
                     std::filesystem::path{"testdata/mul_1.onnx"}, std::filesystem::path{"testdata/mul_16.onnx"});

  constexpr std::string_view metadata_json = R"({
    "component_name": "model_1",
    "variants": {
      "variant_1": {
        "ep_compatibility": [{
          "ep": "example_ep",
          "device": "cpu",
          "compatibility_string": "example_ep;version=0.1.0;ort_api_version=25;hardware_architecture=arch1"
        }]
      },
      "variant_2": {
        "ep_compatibility": [{
          "ep": "example_ep",
          "device": "npu",
          "compatibility_string": "example_ep;version=0.1.0;ort_api_version=25;hardware_architecture=arch2"
        }]
      }
    }
  })";

  CreateComponentModelMetadata(package_root, "model_1", metadata_json);

  if (!multi_file_variant) {
    std::ofstream os(package_root / "models" / "model_1" / "variant_1" / "variant.json", std::ios::binary);
    os << R"({
      "files": [
        {
          "filename": "mul_1.onnx",
          "session_options": {
            "session.disable_prepacking": "1",
            "session.intra_op.allow_spinning": "0"
          },
          "provider_options": {
            "backend_path": "example_backend",
            "enable_htp": "1"
          }
        }
      ]
    })";
  } else {
    const auto variant1_dir = package_root / "models" / "model_1" / "variant_1";
    std::filesystem::copy_file(variant1_dir / "mul_1.onnx",
                               variant1_dir / "mul_1_stage2.onnx",
                               std::filesystem::copy_options::overwrite_existing,
                               ec);

    std::ofstream os(package_root / "models" / "model_1" / "variant_1" / "variant.json", std::ios::binary);
    os << R"({
      "files": [
        {
          "filename": "mul_1.onnx",
          "session_options": {
            "session.disable_prepacking": "1",
            "session.intra_op.allow_spinning": "0"
          },
          "provider_options": {
            "backend_path": "example_backend",
            "enable_htp": "1"
          }
        },
        {
          "filename": "mul_1_stage2.onnx",
          "session_options": {
            "session.disable_prepacking": "0"
          },
          "provider_options": {
            "backend_path": "example_backend_stage2"
          }
        }
      ]
    })";
  }

  {
    std::ofstream os(package_root / "models" / "model_1" / "variant_2" / "variant.json", std::ios::binary);
    os << R"({
      "files": [
        {
          "filename": "mul_16.onnx"
        }
      ]
    })";
  }

  return package_root;
}

}  // namespace

// ------------------------------------------------------------------
// Model Package API tests
// ------------------------------------------------------------------
TEST(ModelPackageApiTest, PackageContextQueries) {
  const auto package_root = CreateModelPackageApiTestPackage();

  const OrtModelPackageApi* pkg_api = Ort::GetApi().GetModelPackageApi();
  ASSERT_NE(pkg_api, nullptr);

  auto context_deleter = [pkg_api](OrtModelPackageContext* p) {
    if (p) pkg_api->ReleaseModelPackageContext(p);
  };
  std::unique_ptr<OrtModelPackageContext, decltype(context_deleter)> model_pkg_context(nullptr, context_deleter);

  OrtModelPackageContext* raw_context = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api->CreateModelPackageContext(package_root.c_str(), &raw_context));
  model_pkg_context.reset(raw_context);

  // Query: component count + names
  size_t component_count = 0;
  ASSERT_ORTSTATUS_OK(pkg_api->ModelPackage_GetComponentCount(model_pkg_context.get(), &component_count));
  ASSERT_EQ(component_count, 1u);

  const char* const* component_names = nullptr;
  size_t component_name_count = 0;
  ASSERT_ORTSTATUS_OK(pkg_api->ModelPackage_GetComponentNames(
      model_pkg_context.get(), &component_names, &component_name_count));
  ASSERT_EQ(component_name_count, 1u);
  ASSERT_NE(component_names, nullptr);
  ASSERT_NE(component_names[0], nullptr);
  EXPECT_STREQ(component_names[0], "model_1");

  // Query: variant count + names
  size_t variant_count = 0;
  ASSERT_ORTSTATUS_OK(pkg_api->ModelPackage_GetVariantCount(
      model_pkg_context.get(), "model_1", &variant_count));
  ASSERT_EQ(variant_count, 2u);

  const char* const* variant_names = nullptr;
  size_t variant_name_count = 0;
  ASSERT_ORTSTATUS_OK(pkg_api->ModelPackage_GetVariantNames(
      model_pkg_context.get(), "model_1", &variant_names, &variant_name_count));
  ASSERT_EQ(variant_name_count, 2u);

  std::unordered_set<std::string> variant_name_set;
  for (size_t i = 0; i < variant_name_count; ++i) {
    ASSERT_NE(variant_names[i], nullptr);
    variant_name_set.insert(variant_names[i]);
  }
  EXPECT_EQ(variant_name_set.count("variant_1"), 1u);
  EXPECT_EQ(variant_name_set.count("variant_2"), 1u);

  std::error_code ec;
  std::filesystem::remove_all(package_root, ec);
}

TEST(ModelPackageApiTest, SingleFileVariantInComponent_SelectComponentAndCreateSession) {
  const auto package_root = CreateModelPackageApiTestPackage();

  RegisteredEpDeviceUniquePtr example_ep;
  ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_info, example_ep));
  Ort::ConstEpDevice plugin_ep_device(example_ep.get());

  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> ep_options;
  session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);

  const OrtModelPackageApi* pkg_api = Ort::GetApi().GetModelPackageApi();
  ASSERT_NE(pkg_api, nullptr);

  auto options_deleter = [pkg_api](OrtModelPackageOptions* p) {
    if (p) pkg_api->ReleaseModelPackageOptions(p);
  };
  auto context_deleter = [pkg_api](OrtModelPackageContext* p) {
    if (p) pkg_api->ReleaseModelPackageContext(p);
  };
  auto component_context_deleter = [pkg_api](OrtModelPackageComponentContext* p) {
    if (p) pkg_api->ReleaseModelPackageComponentContext(p);
  };

  std::unique_ptr<OrtModelPackageOptions, decltype(options_deleter)> model_pkg_options(nullptr, options_deleter);
  std::unique_ptr<OrtModelPackageContext, decltype(context_deleter)> model_pkg_context(nullptr, context_deleter);
  std::unique_ptr<OrtModelPackageComponentContext, decltype(component_context_deleter)> component_context(nullptr, component_context_deleter);

  OrtModelPackageOptions* raw_options = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api->CreateModelPackageOptionsFromSessionOptions(*ort_env, session_options, &raw_options));
  model_pkg_options.reset(raw_options);

  OrtModelPackageContext* raw_context = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api->CreateModelPackageContext(package_root.c_str(), &raw_context));
  model_pkg_context.reset(raw_context);

  OrtModelPackageComponentContext* raw_component_context = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api->SelectComponent(model_pkg_context.get(),
                                               "model_1",
                                               model_pkg_options.get(),
                                               &raw_component_context));
  component_context.reset(raw_component_context);

  size_t selected_file_count = 0;
  ASSERT_ORTSTATUS_OK(pkg_api->ModelPackageComponent_GetSelectedVariantFileCount(component_context.get(),
                                                                                 &selected_file_count));
  ASSERT_EQ(selected_file_count, 1u);

  const ORTCHAR_T* selected_file_path = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api->ModelPackageComponent_GetSelectedVariantFilePath(component_context.get(),
                                                                                0,
                                                                                &selected_file_path));
  ASSERT_NE(selected_file_path, nullptr);

  // Validate file session options from selected component context.
  const char* const* session_option_keys = nullptr;
  const char* const* session_option_values = nullptr;
  size_t session_option_count = 0;
  ASSERT_ORTSTATUS_OK(pkg_api->ModelPackageComponent_GetSelectedVariantFileSessionOptions(
      component_context.get(),
      0,
      &session_option_keys,
      &session_option_values,
      &session_option_count));

  ASSERT_EQ(session_option_count, 2u);
  ASSERT_NE(session_option_keys, nullptr);
  ASSERT_NE(session_option_values, nullptr);

  std::unordered_map<std::string, std::string> session_options_from_api;
  for (size_t i = 0; i < session_option_count; ++i) {
    ASSERT_NE(session_option_keys[i], nullptr);
    ASSERT_NE(session_option_values[i], nullptr);
    session_options_from_api.emplace(session_option_keys[i], session_option_values[i]);
  }

  EXPECT_EQ(session_options_from_api.at("session.disable_prepacking"), "1");
  EXPECT_EQ(session_options_from_api.at("session.intra_op.allow_spinning"), "0");

  // Validate file provider options from selected component context.
  const char* const* provider_option_keys = nullptr;
  const char* const* provider_option_values = nullptr;
  size_t provider_option_count = 0;
  ASSERT_ORTSTATUS_OK(pkg_api->ModelPackageComponent_GetSelectedVariantFileProviderOptions(
      component_context.get(),
      0,
      &provider_option_keys,
      &provider_option_values,
      &provider_option_count));

  ASSERT_EQ(provider_option_count, 2u);
  ASSERT_NE(provider_option_keys, nullptr);
  ASSERT_NE(provider_option_values, nullptr);

  std::unordered_map<std::string, std::string> provider_options_from_api;
  for (size_t i = 0; i < provider_option_count; ++i) {
    ASSERT_NE(provider_option_keys[i], nullptr);
    ASSERT_NE(provider_option_values[i], nullptr);
    provider_options_from_api.emplace(provider_option_keys[i], provider_option_values[i]);
  }

  EXPECT_EQ(provider_options_from_api.at("backend_path"), "example_backend");
  EXPECT_EQ(provider_options_from_api.at("enable_htp"), "1");

  OrtSession* raw_session = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api->CreateSession(*ort_env,
                                             component_context.get(),
                                             session_options,
                                             &raw_session));
  Ort::Session session(raw_session);

  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::vector<int64_t> shape = {3, 2};
  std::vector<float> input_data = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  Ort::Value input = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_data.size(),
                                                     shape.data(), shape.size());
  const char* input_names[] = {"X"};
  const char* output_names[] = {"Y"};
  std::vector<Ort::Value> inputs;
  inputs.push_back(std::move(input));

  auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), inputs.size(), output_names, 1);
  ASSERT_EQ(outputs.size(), 1u);
  const float* out = outputs[0].GetTensorData<float>();
  gsl::span<const float> out_span(out, input_data.size());
  EXPECT_THAT(out_span, ::testing::ElementsAre(1.f, 4.f, 9.f, 16.f, 25.f, 36.f));

  std::error_code ec;
  std::filesystem::remove_all(package_root, ec);
}

TEST(ModelPackageApiTest, MultiFileVariantInComponent_SelectComponentAndCreateSession) {
  const auto package_root = CreateModelPackageApiTestPackage(/*multi_file_variant*/ true);

  RegisteredEpDeviceUniquePtr example_ep;
  ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_info, example_ep));
  Ort::ConstEpDevice plugin_ep_device(example_ep.get());

  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> ep_options;
  session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);

  const OrtModelPackageApi* pkg_api = Ort::GetApi().GetModelPackageApi();
  ASSERT_NE(pkg_api, nullptr);

  auto options_deleter = [pkg_api](OrtModelPackageOptions* p) {
    if (p) pkg_api->ReleaseModelPackageOptions(p);
  };
  auto context_deleter = [pkg_api](OrtModelPackageContext* p) {
    if (p) pkg_api->ReleaseModelPackageContext(p);
  };
  auto component_context_deleter = [pkg_api](OrtModelPackageComponentContext* p) {
    if (p) pkg_api->ReleaseModelPackageComponentContext(p);
  };

  std::unique_ptr<OrtModelPackageOptions, decltype(options_deleter)> model_pkg_options(nullptr, options_deleter);
  std::unique_ptr<OrtModelPackageContext, decltype(context_deleter)> model_pkg_context(nullptr, context_deleter);
  std::unique_ptr<OrtModelPackageComponentContext, decltype(component_context_deleter)> component_context(nullptr, component_context_deleter);

  OrtModelPackageOptions* raw_options = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api->CreateModelPackageOptionsFromSessionOptions(*ort_env, session_options, &raw_options));
  model_pkg_options.reset(raw_options);

  OrtModelPackageContext* raw_context = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api->CreateModelPackageContext(package_root.c_str(), &raw_context));
  model_pkg_context.reset(raw_context);

  OrtModelPackageComponentContext* raw_component_context = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api->SelectComponent(model_pkg_context.get(),
                                               "model_1",
                                               model_pkg_options.get(),
                                               &raw_component_context));
  component_context.reset(raw_component_context);

  size_t file_count = 0;
  ASSERT_ORTSTATUS_OK(pkg_api->ModelPackageComponent_GetSelectedVariantFileCount(component_context.get(), &file_count));
  ASSERT_GT(file_count, 1u);

  const ORTCHAR_T* folder = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api->ModelPackageComponent_GetSelectedVariantFolderPath(component_context.get(), &folder));
  ASSERT_NE(folder, nullptr);

  const auto variant_json_path = std::filesystem::path(folder) / "variant.json";
  ASSERT_TRUE(std::filesystem::exists(variant_json_path));

  std::ifstream vf(variant_json_path, std::ios::binary);
  ASSERT_TRUE(vf.good());

  json vm = json::parse(vf);
  ASSERT_TRUE(vm.contains("files"));
  ASSERT_TRUE(vm["files"].is_array());
  ASSERT_EQ(vm["files"].size(), file_count);

  for (size_t i = 0; i < file_count; ++i) {
    const auto& file_entry = vm["files"][i];
    ASSERT_TRUE(file_entry.contains("filename"));
    ASSERT_TRUE(file_entry["filename"].is_string());

    const std::string expected_filename = file_entry["filename"].get<std::string>();

    const ORTCHAR_T* file_path = nullptr;
    ASSERT_ORTSTATUS_OK(pkg_api->ModelPackageComponent_GetSelectedVariantFilePath(component_context.get(), i, &file_path));
    ASSERT_NE(file_path, nullptr);
    EXPECT_EQ(std::filesystem::path(file_path).filename().string(), expected_filename);

    const char* const* session_keys = nullptr;
    const char* const* session_values = nullptr;
    size_t session_options_count = 0;
    ASSERT_ORTSTATUS_OK(pkg_api->ModelPackageComponent_GetSelectedVariantFileSessionOptions(
        component_context.get(), i, &session_keys, &session_values, &session_options_count));

    const size_t expected_session_options_count =
        (file_entry.contains("session_options") && file_entry["session_options"].is_object())
            ? file_entry["session_options"].size()
            : 0u;
    EXPECT_EQ(session_options_count, expected_session_options_count);

    const char* const* provider_keys = nullptr;
    const char* const* provider_values = nullptr;
    size_t provider_options_count = 0;
    ASSERT_ORTSTATUS_OK(pkg_api->ModelPackageComponent_GetSelectedVariantFileProviderOptions(
        component_context.get(), i, &provider_keys, &provider_values, &provider_options_count));

    const size_t expected_provider_options_count =
        (file_entry.contains("provider_options") && file_entry["provider_options"].is_object())
            ? file_entry["provider_options"].size()
            : 0u;
    EXPECT_EQ(provider_options_count, expected_provider_options_count);

    // Build per-file session options and create a stage session from the resolved file path.
    Ort::SessionOptions stage_session_options;

    // Apply per-file session options (as config entries).
    for (size_t k = 0; k < session_options_count; ++k) {
      ASSERT_NE(session_keys[k], nullptr);
      ASSERT_NE(session_values[k], nullptr);
      ASSERT_NO_THROW(stage_session_options.AddConfigEntry(session_keys[k], session_values[k]));
    }

    // Apply per-file provider options.
    std::unordered_map<std::string, std::string> stage_provider_options;
    for (size_t k = 0; k < provider_options_count; ++k) {
      ASSERT_NE(provider_keys[k], nullptr);
      ASSERT_NE(provider_values[k], nullptr);
      stage_provider_options.emplace(provider_keys[k], provider_values[k]);
    }

    stage_session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, stage_provider_options);

    // Create per-file stage session.
    Ort::Session stage_session(*ort_env, file_path, stage_session_options);

    // Smoke-run to ensure created session is functional.
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    std::vector<int64_t> shape = {3, 2};
    std::vector<float> input_data = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    Ort::Value input = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_data.size(),
                                                       shape.data(), shape.size());
    const char* input_names[] = {"X"};
    const char* output_names[] = {"Y"};
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(input));

    auto outputs = stage_session.Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), inputs.size(), output_names, 1);
    ASSERT_EQ(outputs.size(), 1u);
  }
}

TEST(ModelPackageTest, LoadModelPackageAndRunInference_PluginEp_AppendV2) {
  // Test Case 1:
  // package_root is a model package directory which contains a manifest.json.
  // This model package only contains one component model and it has a metadata.json.
  // ORT should parse the manifest and the metadata.json to get model variants' constraints.
  // ORT selects most suitable model variant based on constraints and then loads it to run inference successfully.
  {
    // Build model package on disk
    const auto package_root = std::filesystem::temp_directory_path() / "ort_model_package_test";
    CreateModelPackage(package_root, MakeManifestJson("model_1"),
                       "model_1", "variant_1", "variant_2",
                       std::filesystem::path{"testdata/mul_1.onnx"}, std::filesystem::path{"testdata/mul_16.onnx"});

    const std::string metadata_json = MakeMetadataJsonTwoVariants(
        "model_1",
        "variant_1", "example_ep", "cpu", "",
        "variant_2", "example_ep", "npu", "");

    CreateComponentModelMetadata(package_root,
                                 "model_1",
                                 metadata_json);

    // Register example EP and get its device
    RegisteredEpDeviceUniquePtr example_ep;
    ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_info, example_ep));
    Ort::ConstEpDevice plugin_ep_device(example_ep.get());

    // Prepare session options with ExampleEP appended
    Ort::SessionOptions session_options;
    std::unordered_map<std::string, std::string> ep_options;
    session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);

    // Create session from package root (directory)
    // ORT should pick the variant_1 model since the constraints match the example EP device (device "cpu" matches)
    Ort::Session session(*ort_env, package_root.c_str(), session_options);

    // Prepare input X
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    std::vector<int64_t> shape = {3, 2};
    std::vector<float> input_data = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    Ort::Value input = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_data.size(),
                                                       shape.data(), shape.size());
    const char* input_names[] = {"X"};
    const char* output_names[] = {"Y"};
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(input));

    // Run
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), inputs.size(),
                               output_names, 1);
    ASSERT_EQ(outputs.size(), 1u);
    const float* out = outputs[0].GetTensorData<float>();
    gsl::span<const float> out_span(out, input_data.size());
    EXPECT_THAT(out_span, ::testing::ElementsAre(1.f, 4.f, 9.f, 16.f, 25.f, 36.f));

    // Cleanup
    std::error_code ec;
    std::filesystem::remove_all(package_root, ec);
  }

  // Test Case 2:
  // package_root is a component model directory which contains a metadata.json.
  // ORT should parse metadata.json to get model variants' constraints.
  // ORT selects most suitable model variant based on constraints and then loads it to run inference successfully.
  {
    // Build model package on disk
    const auto package_root = std::filesystem::temp_directory_path() / "ort_model_package_test";

    CreateModelPackage(package_root, MakeManifestJson("model_1"),
                       "model_1", "variant_1", "variant_2",
                       std::filesystem::path{"testdata/mul_1.onnx"}, std::filesystem::path{"testdata/mul_16.onnx"});

    const std::string metadata_json = MakeMetadataJsonTwoVariants(
        "model_1",
        "variant_1", "example_ep", "cpu", "",
        "variant_2", "example_ep", "npu", "");

    const auto component_model_root = CreateComponentModelMetadata(package_root,
                                                                   "model_1",
                                                                   metadata_json);

    // Register example EP and get its device
    RegisteredEpDeviceUniquePtr example_ep;
    ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_info, example_ep));
    Ort::ConstEpDevice plugin_ep_device(example_ep.get());

    // Prepare session options with ExampleEP appended
    Ort::SessionOptions session_options;
    std::unordered_map<std::string, std::string> ep_options;
    session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);

    // Create session from component model root (directory)
    // ORT should pick the variant_1 model since the constraints match the example EP device (device "cpu" matches)
    Ort::Session session(*ort_env, component_model_root.c_str(), session_options);

    // Prepare input X
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    std::vector<int64_t> shape = {3, 2};
    std::vector<float> input_data = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    Ort::Value input = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_data.size(),
                                                       shape.data(), shape.size());
    const char* input_names[] = {"X"};
    const char* output_names[] = {"Y"};
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(input));

    // Run
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), inputs.size(),
                               output_names, 1);
    ASSERT_EQ(outputs.size(), 1u);
    const float* out = outputs[0].GetTensorData<float>();
    gsl::span<const float> out_span(out, input_data.size());
    EXPECT_THAT(out_span, ::testing::ElementsAre(1.f, 4.f, 9.f, 16.f, 25.f, 36.f));

    // Cleanup
    std::error_code ec;
    std::filesystem::remove_all(package_root, ec);
  }
}

TEST(ModelPackageTest, LoadModelPackageAndRunInference_PreferCpu) {
  // Build model package on disk
  const auto package_root = std::filesystem::temp_directory_path() / "ort_model_package_test";

  CreateModelPackage(package_root, MakeManifestJson("model_1"),
                     "model_1", "variant_1", "variant_2",
                     std::filesystem::path{"testdata/mul_1.onnx"}, std::filesystem::path{"testdata/mul_16.onnx"});

  const std::string metadata_json = MakeMetadataJsonTwoVariants(
      "model_1",
      "variant_1", "example_ep", "cpu", "",
      "variant_2", "example_ep", "npu", "");

  CreateComponentModelMetadata(package_root,
                               "model_1",
                               metadata_json);

  // Register example EP and get its device
  RegisteredEpDeviceUniquePtr example_ep;
  ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_info, example_ep));
  Ort::ConstEpDevice plugin_ep_device(example_ep.get());

  // Prepare session options with ExampleEP appended
  Ort::SessionOptions session_options;
  session_options.SetEpSelectionPolicy(OrtExecutionProviderDevicePolicy_PREFER_CPU);

  // Create session from package root (directory)
  // ORT should pick the variant_1 model since the constraints match the example EP device (device "cpu" matches)
  Ort::Session session(*ort_env, package_root.c_str(), session_options);

  // Prepare input X
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::vector<int64_t> shape = {3, 2};
  std::vector<float> input_data = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  Ort::Value input = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_data.size(),
                                                     shape.data(), shape.size());
  const char* input_names[] = {"X"};
  const char* output_names[] = {"Y"};
  std::vector<Ort::Value> inputs;
  inputs.push_back(std::move(input));

  // Run
  auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), inputs.size(),
                             output_names, 1);
  ASSERT_EQ(outputs.size(), 1u);
  const float* out = outputs[0].GetTensorData<float>();
  gsl::span<const float> out_span(out, input_data.size());
  EXPECT_THAT(out_span, ::testing::ElementsAre(1.f, 4.f, 9.f, 16.f, 25.f, 36.f));

  // Cleanup
  std::error_code ec;
  std::filesystem::remove_all(package_root, ec);
}

TEST(ModelPackageTest, CheckCompiledModelCompatibilityInfo) {
  RegisteredEpDeviceUniquePtr example_ep;
  ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_info, example_ep));
  Ort::ConstEpDevice plugin_ep_device(example_ep.get());

  const ORTCHAR_T* input_model_file = ORT_TSTR("testdata/mul_1.onnx");
  const ORTCHAR_T* output_model_file = ORT_TSTR("plugin_ep_compat_test.onnx");
  std::filesystem::remove(output_model_file);

  // Compile the model
  {
    Ort::SessionOptions session_options;
    std::unordered_map<std::string, std::string> ep_options;
    session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);

    Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
    compile_options.SetFlags(OrtCompileApiFlags_ERROR_IF_NO_NODES_COMPILED);
    compile_options.SetInputModelPath(input_model_file);
    compile_options.SetOutputModelPath(output_model_file);

    ASSERT_CXX_ORTSTATUS_OK(Ort::CompileModel(*ort_env, compile_options));
    ASSERT_TRUE(std::filesystem::exists(output_model_file));
  }

  // Build model package on disk
  const auto package_root = std::filesystem::temp_directory_path() / "ort_model_package_test";

  CreateModelPackage(package_root, MakeManifestJson("model_1"),
                     "model_1", "variant_2", "variant_1",
                     std::filesystem::path{"testdata/mul_16.onnx"}, std::filesystem::path{"plugin_ep_compat_test.onnx"});

  // Build compat strings dynamically against current ORT_API_VERSION so the EP's ORT-version check
  // doesn't short-circuit to PREFER_RECOMPILATION for both variants (which would make hardware_architecture
  // irrelevant and the variant ranking collapse to a tie). With matching ORT versions, the arch differentiates:
  // arch1 -> OPTIMAL, arch2 -> PREFER_RECOMPILATION; variant_1 must win.
  const std::string ort_api_version_str = std::to_string(ORT_API_VERSION);
  const std::string compat_arch2 =
      "example_ep;version=0.1.0;ort_api_version=" + ort_api_version_str + ";hardware_architecture=arch2";
  const std::string compat_arch1 =
      "example_ep;version=0.1.0;ort_api_version=" + ort_api_version_str + ";hardware_architecture=arch1";
  const std::string metadata_json = MakeMetadataJsonTwoVariants(
      "model_1",
      "variant_2", "example_ep", "cpu", compat_arch2.c_str(),
      "variant_1", "example_ep", "cpu", compat_arch1.c_str());

  CreateComponentModelMetadata(package_root,
                               "model_1",
                               metadata_json);

  // Prepare session options with ExampleEP appended
  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> ep_options;
  session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);

  // Create session from package root (directory)
  // ORT should pick the variant_1 model since it has OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL for the example EP,
  // while variant_2 is only OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION.
  // If variant_2 was selected and loaded, i.e. mul_16.onnx, session initialization would fail with error "Error No Op registered for Mul16".
  Ort::Session session(*ort_env, package_root.c_str(), session_options);

  // Cleanup
  std::error_code ec;
  std::filesystem::remove_all(package_root, ec);
}

TEST(ModelPackageTest, LoadModelPackageAndRunInference_DiscoverComponentsFromModelsFolder) {
  // manifest.json without "components"; discovery should scan models/* with metadata.json.
  const auto package_root = std::filesystem::temp_directory_path() / "ort_model_package_discover_test";
  constexpr std::string_view manifest_json = R"({
    "schema_version": 1,
    "model_name": "test_model"
  })";

  CreateModelPackage(package_root, manifest_json,
                     "model_1", "variant_1", "variant_2",
                     std::filesystem::path{"testdata/mul_1.onnx"}, std::filesystem::path{"testdata/mul_16.onnx"});

  // Prepare component model with metadata and variants
  const std::string component_name = "model_1";
  const std::string metadata_json = MakeMetadataJsonTwoVariants(
      "model_1",
      "variant_1", "example_ep", "cpu", "",
      "variant_2", "example_ep", "npu", "");

  // Create metadata.json under models/model_1
  const auto component_root = CreateComponentModelMetadata(package_root,
                                                           component_name,
                                                           metadata_json);

  // Add another component folder without metadata to ensure it's ignored
  std::filesystem::create_directories(package_root / "models" / "ignored_component");

  // Register example EP and get its device
  RegisteredEpDeviceUniquePtr example_ep;
  ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_info, example_ep));
  Ort::ConstEpDevice plugin_ep_device(example_ep.get());

  // Prepare session options with ExampleEP appended
  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> ep_options;
  session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);

  // Create session from package root (directory). Discovery should find model_1 via metadata.json,
  // then pick variant_1 (device cpu) matching the example EP device.
  // If variant_2 was selected and loaded, i.e. mul_16.onnx, session initialization would fail with error "Error No Op registered for Mul16".
  Ort::Session session(*ort_env, package_root.c_str(), session_options);

  // Prepare input X
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::vector<int64_t> shape = {3, 2};
  std::vector<float> input_data = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  Ort::Value input = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_data.size(),
                                                     shape.data(), shape.size());
  const char* input_names[] = {"X"};
  const char* output_names[] = {"Y"};
  std::vector<Ort::Value> inputs;
  inputs.push_back(std::move(input));

  // Run
  auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), inputs.size(),
                             output_names, 1);
  ASSERT_EQ(outputs.size(), 1u);
  const float* out = outputs[0].GetTensorData<float>();
  gsl::span<const float> out_span(out, input_data.size());
  EXPECT_THAT(out_span, ::testing::ElementsAre(1.f, 4.f, 9.f, 16.f, 25.f, 36.f));

  // Cleanup
  std::error_code ec;
  std::filesystem::remove_all(package_root, ec);
}

TEST(ModelPackageTest, ParseVariantsFromRoot_PackageRootDirectory) {
  const auto package_root = std::filesystem::temp_directory_path() / "ort_model_package_parse_from_package_root";
  std::error_code ec;
  std::filesystem::remove_all(package_root, ec);

  // package_root is a model package directory (has manifest.json).
  constexpr std::string_view manifest_json = R"({
    "schema_version": 1,
    "components": ["model_1"]
  })";

  CreateModelPackage(package_root, manifest_json,
                     "model_1", "variant_1", "variant_2",
                     std::filesystem::path{"testdata/mul_1.onnx"}, std::filesystem::path{"testdata/mul_16.onnx"});

  constexpr std::string_view metadata_json = R"({
    "component_name": "model_1",
    "variants": {
      "variant_1": {
        "ep_compatibility": [{
          "ep": "example_ep",
          "device": "cpu"
        }]
      },
      "variant_2": {
        "ep_compatibility": [{
          "ep": "example_ep",
          "device": "npu"
        }]
      }
    }
  })";

  CreateComponentModelMetadata(package_root, "model_1", metadata_json);

  // New schema: per-variant descriptor in variant.json
  {
    std::ofstream os(package_root / "models" / "model_1" / "variant_1" / "variant.json", std::ios::binary);
    os << R"({
      "files": [
        { "filename": "mul_1.onnx" }
      ]
    })";
  }
  {
    std::ofstream os(package_root / "models" / "model_1" / "variant_2" / "variant.json", std::ios::binary);
    os << R"({
      "files": [
        { "filename": "mul_16.onnx" }
      ]
    })";
  }

  ModelPackageDescriptorParser parser(logging::LoggingManager::DefaultLogger());
  std::vector<VariantInfo> variants;
  auto status = parser.ParseVariantsFromRoot(package_root, variants);

  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  ASSERT_EQ(variants.size(), 2u);

  std::unordered_map<std::string, const VariantInfo*> by_file;
  for (const auto& v : variants) {
    ASSERT_EQ(v.files.size(), 1u);
    by_file.emplace(v.files[0].model_file_path.filename().string(), &v);
  }

  ASSERT_EQ(by_file.count("mul_1.onnx"), 1u);
  ASSERT_EQ(by_file.count("mul_16.onnx"), 1u);

  const auto* v1 = by_file.at("mul_1.onnx");
  ASSERT_FALSE(v1->ep_compatibility.empty());
  EXPECT_EQ(v1->ep_compatibility[0].ep.value_or(""), "example_ep");
  EXPECT_EQ(v1->ep_compatibility[0].device.value_or(""), "cpu");

  const auto* v2 = by_file.at("mul_16.onnx");
  ASSERT_FALSE(v2->ep_compatibility.empty());
  EXPECT_EQ(v2->ep_compatibility[0].ep.value_or(""), "example_ep");
  EXPECT_EQ(v2->ep_compatibility[0].device.value_or(""), "npu");

  std::filesystem::remove_all(package_root, ec);
}

TEST(ModelPackageTest, ParseVariantsFromRoot_ComponentModelDirectory) {
  const auto component_root = std::filesystem::temp_directory_path() / "ort_model_package_parse_from_component_root";
  std::error_code ec;
  std::filesystem::remove_all(component_root, ec);
  std::filesystem::create_directories(component_root);

  // package_root is a component model directory (has metadata.json, no manifest.json).
  const auto variant_dir = component_root / "variant_1";
  std::filesystem::create_directories(variant_dir);
  std::filesystem::copy_file("testdata/mul_1.onnx", variant_dir / "mul_1.onnx",
                             std::filesystem::copy_options::overwrite_existing, ec);

  constexpr std::string_view metadata_json = R"({
    "component_name": "model_1",
    "variants": {
      "variant_1": {
        "ep_compatibility": [{
          "ep": "example_ep",
          "device": "cpu"
        }]
      }
    }
  })";

  {
    std::ofstream os(component_root / "metadata.json", std::ios::binary);
    os << metadata_json;
  }

  {
    std::ofstream os(variant_dir / "variant.json", std::ios::binary);
    os << R"({
      "files": [
        { "filename": "mul_1.onnx" }
      ]
    })";
  }

  ModelPackageDescriptorParser parser(logging::LoggingManager::DefaultLogger());
  std::vector<VariantInfo> variants;
  auto status = parser.ParseVariantsFromRoot(component_root, variants);

  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  ASSERT_EQ(variants.size(), 1u);
  ASSERT_EQ(variants[0].files.size(), 1u);
  EXPECT_EQ(variants[0].files[0].model_file_path.filename().string(), "mul_1.onnx");

  ASSERT_FALSE(variants[0].ep_compatibility.empty());
  EXPECT_EQ(variants[0].ep_compatibility[0].ep.value_or(""), "example_ep");
  EXPECT_EQ(variants[0].ep_compatibility[0].device.value_or(""), "cpu");

  std::filesystem::remove_all(component_root, ec);
}

// ------------------------------------------------------------------
// Tests for descriptor parser: enforced "ep" field in ep_compatibility entries.
// ------------------------------------------------------------------
namespace {

// Make a single-component, single-variant package on disk where metadata.json is written
// directly at the package root (the "single-component metadata flow" of the parser).
// In this flow ep_compatibility schema validation errors are propagated, instead of being
// swallowed by the manifest-driven discovery path which falls back to "Missing metadata variants".
// Returns the package_root.
std::filesystem::path MakeSingleComponentPackageWithMetadata(std::string_view subdir,
                                                             std::string_view metadata_json,
                                                             std::string_view variant_json = R"({"files":[{"filename":"mul_1.onnx"}]})") {
  const auto package_root = std::filesystem::temp_directory_path() / std::string(subdir);
  std::error_code ec;
  std::filesystem::remove_all(package_root, ec);
  std::filesystem::create_directories(package_root);

  // Write metadata.json directly under package_root (no manifest, no models/ subdir).
  {
    std::ofstream os(package_root / "metadata.json", std::ios::binary);
    os << metadata_json;
  }

  // Variants live directly under package_root for the single-component flow.
  const auto variant_dir = package_root / "variant_1";
  std::filesystem::create_directories(variant_dir);
  std::filesystem::copy_file("testdata/mul_1.onnx", variant_dir / "mul_1.onnx",
                             std::filesystem::copy_options::overwrite_existing, ec);

  std::ofstream os(variant_dir / "variant.json", std::ios::binary);
  os << variant_json;

  return package_root;
}

}  // namespace

TEST(ModelPackageTest, ParserRejects_EpCompatibilityMissingEp) {
  // The "ep" field is required in every ep_compatibility entry.
  // Omitting it must yield a parse error (not silently accept a wildcard / portable variant).
  constexpr std::string_view metadata_json = R"({
    "component_name": "model_1",
    "variants": {
      "variant_1": {
        "ep_compatibility": [{
          "device": "cpu",
          "compatibility_string": "anything"
        }]
      }
    }
  })";
  const auto package_root = MakeSingleComponentPackageWithMetadata(
      "ort_model_package_parser_missing_ep", metadata_json);

  ModelPackageDescriptorParser parser(logging::LoggingManager::DefaultLogger());
  std::vector<VariantInfo> variants;
  auto status = parser.ParseVariantsFromRoot(package_root, variants);

  EXPECT_FALSE(status.IsOK());
  EXPECT_NE(status.ErrorMessage().find("ep"), std::string::npos) << status.ErrorMessage();

  std::error_code ec;
  std::filesystem::remove_all(package_root, ec);
}

TEST(ModelPackageTest, ParserRejects_EpCompatibilityNullEp) {
  constexpr std::string_view metadata_json = R"({
    "component_name": "model_1",
    "variants": {
      "variant_1": {
        "ep_compatibility": [{
          "ep": null,
          "device": "cpu"
        }]
      }
    }
  })";
  const auto package_root = MakeSingleComponentPackageWithMetadata(
      "ort_model_package_parser_null_ep", metadata_json);

  ModelPackageDescriptorParser parser(logging::LoggingManager::DefaultLogger());
  std::vector<VariantInfo> variants;
  auto status = parser.ParseVariantsFromRoot(package_root, variants);

  EXPECT_FALSE(status.IsOK());
  EXPECT_NE(status.ErrorMessage().find("ep"), std::string::npos) << status.ErrorMessage();

  std::error_code ec;
  std::filesystem::remove_all(package_root, ec);
}

TEST(ModelPackageTest, ParserRejects_EpCompatibilityEmptyEp) {
  constexpr std::string_view metadata_json = R"({
    "component_name": "model_1",
    "variants": {
      "variant_1": {
        "ep_compatibility": [{
          "ep": "",
          "device": "cpu"
        }]
      }
    }
  })";
  const auto package_root = MakeSingleComponentPackageWithMetadata(
      "ort_model_package_parser_empty_ep", metadata_json);

  ModelPackageDescriptorParser parser(logging::LoggingManager::DefaultLogger());
  std::vector<VariantInfo> variants;
  auto status = parser.ParseVariantsFromRoot(package_root, variants);

  EXPECT_FALSE(status.IsOK());
  EXPECT_NE(status.ErrorMessage().find("ep"), std::string::npos) << status.ErrorMessage();

  std::error_code ec;
  std::filesystem::remove_all(package_root, ec);
}

// ------------------------------------------------------------------
// Tests for new pre-selection EP-compat traversal accessors.
// ------------------------------------------------------------------
TEST(ModelPackageApiTest, GetVariantEpCompatibility_ReturnsAllEntries) {
  const auto package_root = std::filesystem::temp_directory_path() / "ort_mp_pre_selection_ep_compat";
  std::error_code ec;
  std::filesystem::remove_all(package_root, ec);

  CreateManifestJson(package_root, MakeManifestJson("model_1"));

  const auto variant1_dir = package_root / "models" / "model_1" / "variant_1";
  const auto variant2_dir = package_root / "models" / "model_1" / "variant_2";
  std::filesystem::create_directories(variant1_dir);
  std::filesystem::create_directories(variant2_dir);
  std::filesystem::copy_file("testdata/mul_1.onnx", variant1_dir / "mul_1.onnx",
                             std::filesystem::copy_options::overwrite_existing, ec);
  std::filesystem::copy_file("testdata/mul_1.onnx", variant2_dir / "mul_1.onnx",
                             std::filesystem::copy_options::overwrite_existing, ec);

  // variant_1 has two ep_compatibility entries (one with compatibility_string omitted).
  // variant_2 has one entry with all fields populated.
  constexpr std::string_view metadata_json = R"({
    "component_name": "model_1",
    "variants": {
      "variant_1": {
        "ep_compatibility": [
          { "ep": "example_ep", "device": "cpu" },
          { "ep": "other_ep",   "device": "gpu", "compatibility_string": "compat_a" }
        ]
      },
      "variant_2": {
        "ep_compatibility": [
          { "ep": "example_ep", "device": "npu", "compatibility_string": "compat_b" }
        ]
      }
    }
  })";
  CreateComponentModelMetadata(package_root, "model_1", metadata_json);

  for (const auto& d : {variant1_dir, variant2_dir}) {
    std::ofstream os(d / "variant.json", std::ios::binary);
    os << R"({"files":[{"filename":"mul_1.onnx"}]})";
  }

  const OrtModelPackageApi* pkg_api = Ort::GetApi().GetModelPackageApi();
  ASSERT_NE(pkg_api, nullptr);

  auto context_deleter = [pkg_api](OrtModelPackageContext* p) {
    if (p) pkg_api->ReleaseModelPackageContext(p);
  };
  std::unique_ptr<OrtModelPackageContext, decltype(context_deleter)> ctx(nullptr, context_deleter);
  OrtModelPackageContext* raw_ctx = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api->CreateModelPackageContext(package_root.c_str(), &raw_ctx));
  ctx.reset(raw_ctx);

  // variant_1: 2 entries
  size_t v1_count = 0;
  ASSERT_ORTSTATUS_OK(pkg_api->ModelPackage_GetVariantEpCompatibilityCount(
      ctx.get(), "model_1", "variant_1", &v1_count));
  ASSERT_EQ(v1_count, 2u);

  // Aggregate the entries in a set since underlying storage may not preserve declaration order.
  struct Entry {
    std::string ep, device, compat;
  };
  std::vector<Entry> v1_entries;
  for (size_t i = 0; i < v1_count; ++i) {
    const char* ep = nullptr;
    const char* dev = nullptr;
    const char* compat = nullptr;
    ASSERT_ORTSTATUS_OK(pkg_api->ModelPackage_GetVariantEpCompatibility(
        ctx.get(), "model_1", "variant_1", i, &ep, &dev, &compat));
    v1_entries.push_back({ep ? ep : "", dev ? dev : "", compat ? compat : ""});
  }

  auto find_ep = [&](const std::string& ep_name) -> const Entry* {
    for (const auto& e : v1_entries) {
      if (e.ep == ep_name) return &e;
    }
    return nullptr;
  };
  const auto* example = find_ep("example_ep");
  const auto* other = find_ep("other_ep");
  ASSERT_NE(example, nullptr);
  ASSERT_NE(other, nullptr);
  EXPECT_EQ(example->device, "cpu");
  EXPECT_EQ(example->compat, "");  // omitted -> empty / NULL
  EXPECT_EQ(other->device, "gpu");
  EXPECT_EQ(other->compat, "compat_a");

  // variant_2: 1 entry
  size_t v2_count = 0;
  ASSERT_ORTSTATUS_OK(pkg_api->ModelPackage_GetVariantEpCompatibilityCount(
      ctx.get(), "model_1", "variant_2", &v2_count));
  ASSERT_EQ(v2_count, 1u);

  const char* ep = nullptr;
  const char* dev = nullptr;
  const char* compat = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api->ModelPackage_GetVariantEpCompatibility(
      ctx.get(), "model_1", "variant_2", 0, &ep, &dev, &compat));
  ASSERT_NE(ep, nullptr);
  EXPECT_STREQ(ep, "example_ep");
  ASSERT_NE(dev, nullptr);
  EXPECT_STREQ(dev, "npu");
  ASSERT_NE(compat, nullptr);
  EXPECT_STREQ(compat, "compat_b");

  // Optional out-parameters: callers can pass NULL for fields they don't need.
  ASSERT_ORTSTATUS_OK(pkg_api->ModelPackage_GetVariantEpCompatibility(
      ctx.get(), "model_1", "variant_2", 0, nullptr, nullptr, nullptr));

  std::filesystem::remove_all(package_root, ec);
}

TEST(ModelPackageApiTest, GetVariantEpCompatibility_OutOfRangeIsError) {
  const auto package_root = MakeSingleComponentPackageWithMetadata(
      "ort_mp_ep_compat_oor",
      R"({
        "component_name": "model_1",
        "variants": {
          "variant_1": {
            "ep_compatibility": [{ "ep": "example_ep", "device": "cpu" }]
          }
        }
      })");

  const OrtModelPackageApi* pkg_api = Ort::GetApi().GetModelPackageApi();
  auto context_deleter = [pkg_api](OrtModelPackageContext* p) {
    if (p) pkg_api->ReleaseModelPackageContext(p);
  };
  std::unique_ptr<OrtModelPackageContext, decltype(context_deleter)> ctx(nullptr, context_deleter);
  OrtModelPackageContext* raw_ctx = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api->CreateModelPackageContext(package_root.c_str(), &raw_ctx));
  ctx.reset(raw_ctx);

  const char* ep = nullptr;
  const char* dev = nullptr;
  const char* compat = nullptr;
  OrtStatus* st = pkg_api->ModelPackage_GetVariantEpCompatibility(
      ctx.get(), "model_1", "variant_1", /*ep_idx*/ 5, &ep, &dev, &compat);
  EXPECT_NE(st, nullptr);
  if (st != nullptr) Ort::GetApi().ReleaseStatus(st);

  std::error_code ec;
  std::filesystem::remove_all(package_root, ec);
}

// ------------------------------------------------------------------
// Test for the consumer_metadata accessor on a selected variant.
// ------------------------------------------------------------------
TEST(ModelPackageApiTest, GetSelectedVariantConsumerMetadata) {
  const auto package_root = std::filesystem::temp_directory_path() / "ort_mp_consumer_metadata";
  std::error_code ec;
  std::filesystem::remove_all(package_root, ec);

  CreateManifestJson(package_root, MakeManifestJson("model_1"));

  const auto variant_dir = package_root / "models" / "model_1" / "variant_1";
  std::filesystem::create_directories(variant_dir);
  std::filesystem::copy_file("testdata/mul_1.onnx", variant_dir / "mul_1.onnx",
                             std::filesystem::copy_options::overwrite_existing, ec);

  constexpr std::string_view metadata_json = R"({
    "component_name": "model_1",
    "variants": {
      "variant_1": {
        "ep_compatibility": [{ "ep": "example_ep", "device": "cpu" }]
      }
    }
  })";
  CreateComponentModelMetadata(package_root, "model_1", metadata_json);

  // variant.json has a non-trivial consumer_metadata sub-object.
  {
    std::ofstream os(variant_dir / "variant.json", std::ios::binary);
    os << R"({
      "files": [{ "filename": "mul_1.onnx" }],
      "consumer_metadata": {
        "framework": "onnxruntime-genai",
        "tokens": { "bos": 1, "eos": 2 }
      }
    })";
  }

  RegisteredEpDeviceUniquePtr example_ep;
  ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_info, example_ep));
  Ort::ConstEpDevice plugin_ep_device(example_ep.get());

  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> ep_options;
  session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);

  const OrtModelPackageApi* pkg_api = Ort::GetApi().GetModelPackageApi();
  ASSERT_NE(pkg_api, nullptr);

  auto options_deleter = [pkg_api](OrtModelPackageOptions* p) { if (p) pkg_api->ReleaseModelPackageOptions(p); };
  auto context_deleter = [pkg_api](OrtModelPackageContext* p) { if (p) pkg_api->ReleaseModelPackageContext(p); };
  auto component_context_deleter = [pkg_api](OrtModelPackageComponentContext* p) {
    if (p) pkg_api->ReleaseModelPackageComponentContext(p);
  };
  std::unique_ptr<OrtModelPackageOptions, decltype(options_deleter)> mp_opts(nullptr, options_deleter);
  std::unique_ptr<OrtModelPackageContext, decltype(context_deleter)> ctx(nullptr, context_deleter);
  std::unique_ptr<OrtModelPackageComponentContext, decltype(component_context_deleter)> comp_ctx(nullptr, component_context_deleter);

  OrtModelPackageOptions* raw_mp_opts = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api->CreateModelPackageOptionsFromSessionOptions(*ort_env, session_options, &raw_mp_opts));
  mp_opts.reset(raw_mp_opts);

  OrtModelPackageContext* raw_ctx = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api->CreateModelPackageContext(package_root.c_str(), &raw_ctx));
  ctx.reset(raw_ctx);

  OrtModelPackageComponentContext* raw_comp_ctx = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api->SelectComponent(ctx.get(), "model_1", mp_opts.get(), &raw_comp_ctx));
  comp_ctx.reset(raw_comp_ctx);

  const char* json_str = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api->ModelPackageComponent_GetSelectedVariantConsumerMetadata(comp_ctx.get(), &json_str));
  ASSERT_NE(json_str, nullptr);

  // The returned blob must be a parseable JSON object with the expected fields.
  json parsed = json::parse(json_str);
  ASSERT_TRUE(parsed.is_object());
  EXPECT_EQ(parsed.value("framework", ""), "onnxruntime-genai");
  ASSERT_TRUE(parsed.contains("tokens"));
  EXPECT_EQ(parsed["tokens"].value("bos", 0), 1);
  EXPECT_EQ(parsed["tokens"].value("eos", 0), 2);

  std::filesystem::remove_all(package_root, ec);
}

// ------------------------------------------------------------------
// Test: variant selector tie-break is deterministic across repeated invocations.
// Two variants advertise compatibility for the same EP/device and EP returns the same
// validation score for both -- selection must be stable.
// ------------------------------------------------------------------
TEST(ModelPackageTest, VariantSelector_TieBreakIsDeterministic) {
  // Both variants point at the *same* model file (mul_1.onnx) so whichever wins works at runtime.
  // They advertise identical EP/device pairs and empty compatibility_string so the EP returns the
  // same score (NOT_APPLICABLE) for both -- a tie. The fix in commit 27217da484 guarantees that
  // ties resolve deterministically, i.e., selection is stable across repeated runs.
  RegisteredEpDeviceUniquePtr example_ep;
  ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_info, example_ep));
  Ort::ConstEpDevice plugin_ep_device(example_ep.get());

  std::string first_selected_filename;

  for (int iter = 0; iter < 5; ++iter) {
    const auto package_root = std::filesystem::temp_directory_path() / "ort_mp_tie_break";
    std::error_code ec;
    std::filesystem::remove_all(package_root, ec);

    CreateModelPackage(package_root, MakeManifestJson("model_1"),
                       "model_1", "variant_a", "variant_b",
                       std::filesystem::path{"testdata/mul_1.onnx"},
                       std::filesystem::path{"testdata/mul_1.onnx"});

    const std::string metadata_json = MakeMetadataJsonTwoVariants(
        "model_1",
        "variant_a", "example_ep", "cpu", "",
        "variant_b", "example_ep", "cpu", "");
    CreateComponentModelMetadata(package_root, "model_1", metadata_json);

    Ort::SessionOptions session_options;
    std::unordered_map<std::string, std::string> ep_options;
    session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);

    const OrtModelPackageApi* pkg_api = Ort::GetApi().GetModelPackageApi();
    ASSERT_NE(pkg_api, nullptr);

    auto options_deleter = [pkg_api](OrtModelPackageOptions* p) { if (p) pkg_api->ReleaseModelPackageOptions(p); };
    auto context_deleter = [pkg_api](OrtModelPackageContext* p) { if (p) pkg_api->ReleaseModelPackageContext(p); };
    auto component_context_deleter = [pkg_api](OrtModelPackageComponentContext* p) {
      if (p) pkg_api->ReleaseModelPackageComponentContext(p);
    };
    std::unique_ptr<OrtModelPackageOptions, decltype(options_deleter)> mp_opts(nullptr, options_deleter);
    std::unique_ptr<OrtModelPackageContext, decltype(context_deleter)> ctx(nullptr, context_deleter);
    std::unique_ptr<OrtModelPackageComponentContext, decltype(component_context_deleter)> comp_ctx(nullptr, component_context_deleter);

    OrtModelPackageOptions* raw_mp_opts = nullptr;
    ASSERT_ORTSTATUS_OK(pkg_api->CreateModelPackageOptionsFromSessionOptions(*ort_env, session_options, &raw_mp_opts));
    mp_opts.reset(raw_mp_opts);

    OrtModelPackageContext* raw_ctx = nullptr;
    ASSERT_ORTSTATUS_OK(pkg_api->CreateModelPackageContext(package_root.c_str(), &raw_ctx));
    ctx.reset(raw_ctx);

    OrtModelPackageComponentContext* raw_comp_ctx = nullptr;
    ASSERT_ORTSTATUS_OK(pkg_api->SelectComponent(ctx.get(), "model_1", mp_opts.get(), &raw_comp_ctx));
    comp_ctx.reset(raw_comp_ctx);

    const ORTCHAR_T* selected_path = nullptr;
    ASSERT_ORTSTATUS_OK(pkg_api->ModelPackageComponent_GetSelectedVariantFilePath(comp_ctx.get(), 0, &selected_path));
    ASSERT_NE(selected_path, nullptr);

    // Path looks like .../models/model_1/<variant_x>/mul_1.onnx -- the parent dir name is the variant.
    const auto selected_variant_dir = std::filesystem::path(selected_path).parent_path().filename().string();
    ASSERT_TRUE(selected_variant_dir == "variant_a" || selected_variant_dir == "variant_b")
        << "unexpected variant dir: " << selected_variant_dir;

    if (iter == 0) {
      first_selected_filename = selected_variant_dir;
    } else {
      EXPECT_EQ(selected_variant_dir, first_selected_filename)
          << "tie-break selection drifted across runs (iter " << iter << ")";
    }

    std::filesystem::remove_all(package_root, ec);
  }
}

// ------------------------------------------------------------------
// Test: a variant's per-file `session_options` flow through OrtApis::AddSessionOption.
// We verify this by feeding a *known* typed key (intra_op_num_threads) a non-integer value:
// pre-change behavior would silently stuff it into AddConfigEntry and succeed; post-change
// behavior parses it via the typed dispatcher and fails CreateSession with a parse error.
// ------------------------------------------------------------------
TEST(ModelPackageTest, VariantSessionOptions_DispatchedThroughAddSessionOption) {
  const auto package_root = std::filesystem::temp_directory_path() / "ort_mp_session_options_dispatch";
  std::error_code ec;
  std::filesystem::remove_all(package_root, ec);

  CreateManifestJson(package_root, MakeManifestJson("model_1"));

  const auto variant_dir = package_root / "models" / "model_1" / "variant_1";
  std::filesystem::create_directories(variant_dir);
  std::filesystem::copy_file("testdata/mul_1.onnx", variant_dir / "mul_1.onnx",
                             std::filesystem::copy_options::overwrite_existing, ec);

  constexpr std::string_view metadata_json = R"({
    "component_name": "model_1",
    "variants": {
      "variant_1": {
        "ep_compatibility": [{ "ep": "example_ep", "device": "cpu" }]
      }
    }
  })";
  CreateComponentModelMetadata(package_root, "model_1", metadata_json);

  // Per-file session_options assigns a typed key (intra_op_num_threads) a value that is not a
  // valid integer. Routing this through AddSessionOption (the new behavior) must reject it.
  {
    std::ofstream os(variant_dir / "variant.json", std::ios::binary);
    os << R"({
      "files": [{
        "filename": "mul_1.onnx",
        "session_options": {
          "intra_op_num_threads": "not_an_int"
        }
      }]
    })";
  }

  RegisteredEpDeviceUniquePtr example_ep;
  ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_info, example_ep));
  Ort::ConstEpDevice plugin_ep_device(example_ep.get());

  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> ep_options;
  session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);

  const OrtModelPackageApi* pkg_api = Ort::GetApi().GetModelPackageApi();
  ASSERT_NE(pkg_api, nullptr);

  auto options_deleter = [pkg_api](OrtModelPackageOptions* p) { if (p) pkg_api->ReleaseModelPackageOptions(p); };
  auto context_deleter = [pkg_api](OrtModelPackageContext* p) { if (p) pkg_api->ReleaseModelPackageContext(p); };
  auto component_context_deleter = [pkg_api](OrtModelPackageComponentContext* p) {
    if (p) pkg_api->ReleaseModelPackageComponentContext(p);
  };
  std::unique_ptr<OrtModelPackageOptions, decltype(options_deleter)> mp_opts(nullptr, options_deleter);
  std::unique_ptr<OrtModelPackageContext, decltype(context_deleter)> ctx(nullptr, context_deleter);
  std::unique_ptr<OrtModelPackageComponentContext, decltype(component_context_deleter)> comp_ctx(nullptr, component_context_deleter);

  OrtModelPackageOptions* raw_mp_opts = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api->CreateModelPackageOptionsFromSessionOptions(*ort_env, session_options, &raw_mp_opts));
  mp_opts.reset(raw_mp_opts);

  OrtModelPackageContext* raw_ctx = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api->CreateModelPackageContext(package_root.c_str(), &raw_ctx));
  ctx.reset(raw_ctx);

  OrtModelPackageComponentContext* raw_comp_ctx = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api->SelectComponent(ctx.get(), "model_1", mp_opts.get(), &raw_comp_ctx));
  comp_ctx.reset(raw_comp_ctx);

  // CreateSession iterates the per-file session_options and dispatches each through AddSessionOption.
  // The bad int value must surface as an error from this call.
  // Pass nullptr for session_options so the metadata-merge path runs (it is skipped when the caller
  // supplies their own session_options).
  OrtSession* raw_session = nullptr;
  OrtStatus* st = pkg_api->CreateSession(*ort_env, comp_ctx.get(), /*session_options=*/nullptr, &raw_session);
  ASSERT_NE(st, nullptr) << "CreateSession unexpectedly succeeded with malformed intra_op_num_threads";
  const std::string err_msg = Ort::GetApi().GetErrorMessage(st);
  Ort::GetApi().ReleaseStatus(st);
  if (raw_session != nullptr) {
    Ort::GetApi().ReleaseSession(raw_session);
  }

  // Message should mention either AddSessionOption or the typed-int parse failure.
  const bool mentions_dispatch =
      err_msg.find("AddSessionOption") != std::string::npos ||
      err_msg.find("base-10 int32") != std::string::npos ||
      err_msg.find("intra_op_num_threads") != std::string::npos;
  EXPECT_TRUE(mentions_dispatch) << "error did not mention typed dispatch: " << err_msg;

  std::filesystem::remove_all(package_root, ec);
}

}  // namespace test
}  // namespace onnxruntime
