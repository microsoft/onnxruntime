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
#include "core/session/onnxruntime_experimental_cxx_api.h"
#include "core/session/abi_devices.h"
#include "test/autoep/test_autoep_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/api_asserts.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {
namespace {

// Typed function pointers for every OrtModelPackageApi_* experimental entry,
// resolved once via the experimental name-based lookup.
struct ModelPackageFns {
  OrtExperimental_OrtModelPackageApi_CreateModelPackageOptionsFromSessionOptions_SinceV28_Fn
      CreateModelPackageOptionsFromSessionOptions{nullptr};
  OrtExperimental_OrtModelPackageApi_ReleaseModelPackageOptions_SinceV28_Fn
      ReleaseModelPackageOptions{nullptr};
  OrtExperimental_OrtModelPackageApi_CreateModelPackageContext_SinceV28_Fn
      CreateModelPackageContext{nullptr};
  OrtExperimental_OrtModelPackageApi_ReleaseModelPackageContext_SinceV28_Fn
      ReleaseModelPackageContext{nullptr};
  OrtExperimental_OrtModelPackageApi_ModelPackage_GetSchemaVersion_SinceV28_Fn
      ModelPackage_GetSchemaVersion{nullptr};
  OrtExperimental_OrtModelPackageApi_ModelPackage_GetComponentCount_SinceV28_Fn
      ModelPackage_GetComponentCount{nullptr};
  OrtExperimental_OrtModelPackageApi_ModelPackage_GetComponentNames_SinceV28_Fn
      ModelPackage_GetComponentNames{nullptr};
  OrtExperimental_OrtModelPackageApi_ModelPackage_GetVariantCount_SinceV28_Fn
      ModelPackage_GetVariantCount{nullptr};
  OrtExperimental_OrtModelPackageApi_ModelPackage_GetVariantNames_SinceV28_Fn
      ModelPackage_GetVariantNames{nullptr};
  OrtExperimental_OrtModelPackageApi_ModelPackage_GetVariantEpName_SinceV28_Fn
      ModelPackage_GetVariantEpName{nullptr};
  OrtExperimental_OrtModelPackageApi_SelectComponent_SinceV28_Fn
      SelectComponent{nullptr};
  OrtExperimental_OrtModelPackageApi_ReleaseModelPackageComponentContext_SinceV28_Fn
      ReleaseModelPackageComponentContext{nullptr};
  OrtExperimental_OrtModelPackageApi_ModelPackageComponent_GetSelectedVariantName_SinceV28_Fn
      ModelPackageComponent_GetSelectedVariantName{nullptr};
  OrtExperimental_OrtModelPackageApi_ModelPackageComponent_GetSelectedVariantFolderPath_SinceV28_Fn
      ModelPackageComponent_GetSelectedVariantFolderPath{nullptr};
  OrtExperimental_OrtModelPackageApi_CreateSession_SinceV28_Fn
      CreateSession{nullptr};
};

inline const ModelPackageFns& GetModelPackageFns() {
  static const ModelPackageFns fns = []() {
    const OrtApi* api = &Ort::GetApi();
    namespace Exp = Ort::Experimental;
    ModelPackageFns f;
    f.CreateModelPackageOptionsFromSessionOptions =
        Exp::Get_OrtModelPackageApi_CreateModelPackageOptionsFromSessionOptions_SinceV28_FnOrThrow(api);
    f.ReleaseModelPackageOptions =
        Exp::Get_OrtModelPackageApi_ReleaseModelPackageOptions_SinceV28_FnOrThrow(api);
    f.CreateModelPackageContext =
        Exp::Get_OrtModelPackageApi_CreateModelPackageContext_SinceV28_FnOrThrow(api);
    f.ReleaseModelPackageContext =
        Exp::Get_OrtModelPackageApi_ReleaseModelPackageContext_SinceV28_FnOrThrow(api);
    f.ModelPackage_GetSchemaVersion =
        Exp::Get_OrtModelPackageApi_ModelPackage_GetSchemaVersion_SinceV28_FnOrThrow(api);
    f.ModelPackage_GetComponentCount =
        Exp::Get_OrtModelPackageApi_ModelPackage_GetComponentCount_SinceV28_FnOrThrow(api);
    f.ModelPackage_GetComponentNames =
        Exp::Get_OrtModelPackageApi_ModelPackage_GetComponentNames_SinceV28_FnOrThrow(api);
    f.ModelPackage_GetVariantCount =
        Exp::Get_OrtModelPackageApi_ModelPackage_GetVariantCount_SinceV28_FnOrThrow(api);
    f.ModelPackage_GetVariantNames =
        Exp::Get_OrtModelPackageApi_ModelPackage_GetVariantNames_SinceV28_FnOrThrow(api);
    f.ModelPackage_GetVariantEpName =
        Exp::Get_OrtModelPackageApi_ModelPackage_GetVariantEpName_SinceV28_FnOrThrow(api);
    f.SelectComponent =
        Exp::Get_OrtModelPackageApi_SelectComponent_SinceV28_FnOrThrow(api);
    f.ReleaseModelPackageComponentContext =
        Exp::Get_OrtModelPackageApi_ReleaseModelPackageComponentContext_SinceV28_FnOrThrow(api);
    f.ModelPackageComponent_GetSelectedVariantName =
        Exp::Get_OrtModelPackageApi_ModelPackageComponent_GetSelectedVariantName_SinceV28_FnOrThrow(api);
    f.ModelPackageComponent_GetSelectedVariantFolderPath =
        Exp::Get_OrtModelPackageApi_ModelPackageComponent_GetSelectedVariantFolderPath_SinceV28_FnOrThrow(api);
    f.CreateSession =
        Exp::Get_OrtModelPackageApi_CreateSession_SinceV28_FnOrThrow(api);
    return f;
  }();
  return fns;
}
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
    "filename": ")"
      << filename << R"("
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
        "ep": ")"
      << variant_ep_1 << R"(",
        "device": ")"
      << variant_device_1 << R"(",
        "compatibility_string": ")"
      << variant_compatibility_string_1 << R"("
      },
      ")"
      << variant_name_2 << R"(": {
        "ep": ")"
      << variant_ep_2 << R"(",
        "device": ")"
      << variant_device_2 << R"(",
        "compatibility_string": ")"
      << variant_compatibility_string_2 << R"("
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
        "ep": "example_ep",
        "device": "cpu",
        "compatibility_string": "example_ep;version=0.1.0;ort_api_version=25;hardware_architecture=arch1"
      },
      "variant_2": {
        "ep": "example_ep",
        "device": "npu",
        "compatibility_string": "example_ep;version=0.1.0;ort_api_version=25;hardware_architecture=arch2"
      }
    }
  })";

  CreateComponentModelMetadata(package_root, "model_1", metadata_json);

  if (!multi_file_variant) {
    std::ofstream os(package_root / "models" / "model_1" / "variant_1" / "variant.json", std::ios::binary);
    os << R"({
      "filename": "mul_1.onnx",
      "session_options": {
        "session.disable_prepacking": "1",
        "session.intra_op.allow_spinning": "0"
      },
      "provider_options": {
        "backend_path": "example_backend",
        "enable_htp": "1"
      }
    })";
  } else {
    // Multi-file variants are no longer supported. For backward-compat testing,
    // just write a single-file variant.json.
    std::ofstream os(package_root / "models" / "model_1" / "variant_1" / "variant.json", std::ios::binary);
    os << R"({
      "filename": "mul_1.onnx",
      "session_options": {
        "session.disable_prepacking": "1",
        "session.intra_op.allow_spinning": "0"
      },
      "provider_options": {
        "backend_path": "example_backend",
        "enable_htp": "1"
      }
    })";
  }

  {
    std::ofstream os(package_root / "models" / "model_1" / "variant_2" / "variant.json", std::ios::binary);
    os << R"({
      "filename": "mul_16.onnx"
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

  const auto& pkg_api = GetModelPackageFns();

  auto context_deleter = [&pkg_api](OrtModelPackageContext* p) {
    if (p) pkg_api.ReleaseModelPackageContext(p);
  };
  std::unique_ptr<OrtModelPackageContext, decltype(context_deleter)> model_pkg_context(nullptr, context_deleter);

  OrtModelPackageContext* raw_context = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api.CreateModelPackageContext(package_root.c_str(), &raw_context));
  model_pkg_context.reset(raw_context);

  // Query: component count + names
  size_t component_count = 0;
  ASSERT_ORTSTATUS_OK(pkg_api.ModelPackage_GetComponentCount(model_pkg_context.get(), &component_count));
  ASSERT_EQ(component_count, 1u);

  const char* const* component_names = nullptr;
  size_t component_name_count = 0;
  ASSERT_ORTSTATUS_OK(pkg_api.ModelPackage_GetComponentNames(
      model_pkg_context.get(), &component_names, &component_name_count));
  ASSERT_EQ(component_name_count, 1u);
  ASSERT_NE(component_names, nullptr);
  ASSERT_NE(component_names[0], nullptr);
  EXPECT_STREQ(component_names[0], "model_1");

  // Query: variant count + names
  size_t variant_count = 0;
  ASSERT_ORTSTATUS_OK(pkg_api.ModelPackage_GetVariantCount(
      model_pkg_context.get(), "model_1", &variant_count));
  ASSERT_EQ(variant_count, 2u);

  const char* const* variant_names = nullptr;
  size_t variant_name_count = 0;
  ASSERT_ORTSTATUS_OK(pkg_api.ModelPackage_GetVariantNames(
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

  const auto& pkg_api = GetModelPackageFns();

  auto options_deleter = [&pkg_api](OrtModelPackageOptions* p) {
    if (p) pkg_api.ReleaseModelPackageOptions(p);
  };
  auto context_deleter = [&pkg_api](OrtModelPackageContext* p) {
    if (p) pkg_api.ReleaseModelPackageContext(p);
  };
  auto component_context_deleter = [&pkg_api](OrtModelPackageComponentContext* p) {
    if (p) pkg_api.ReleaseModelPackageComponentContext(p);
  };

  std::unique_ptr<OrtModelPackageOptions, decltype(options_deleter)> model_pkg_options(nullptr, options_deleter);
  std::unique_ptr<OrtModelPackageContext, decltype(context_deleter)> model_pkg_context(nullptr, context_deleter);
  std::unique_ptr<OrtModelPackageComponentContext, decltype(component_context_deleter)> component_context(nullptr, component_context_deleter);

  OrtModelPackageOptions* raw_options = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api.CreateModelPackageOptionsFromSessionOptions(*ort_env, session_options, &raw_options));
  model_pkg_options.reset(raw_options);

  OrtModelPackageContext* raw_context = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api.CreateModelPackageContext(package_root.c_str(), &raw_context));
  model_pkg_context.reset(raw_context);

  OrtModelPackageComponentContext* raw_component_context = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api.SelectComponent(model_pkg_context.get(),
                                              "model_1",
                                              model_pkg_options.get(),
                                              &raw_component_context));
  component_context.reset(raw_component_context);

  OrtSession* raw_session = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api.CreateSession(*ort_env,
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

    // Embed the EPContext binary data inside the compiled model so the model is self-contained.
    // This test copies only the compiled .onnx into the model package, so it must not rely on a
    // separate sidecar EPContext data file (which non-embedded mode would produce).
    compile_options.SetEpContextEmbedMode(true);

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
        "ep": "example_ep",
        "device": "cpu"
      },
      "variant_2": {
        "ep": "example_ep",
        "device": "npu"
      }
    }
  })";

  CreateComponentModelMetadata(package_root, "model_1", metadata_json);

  // New schema: per-variant descriptor in variant.json
  {
    std::ofstream os(package_root / "models" / "model_1" / "variant_1" / "variant.json", std::ios::binary);
    os << R"({ "filename": "mul_1.onnx" })";
  }
  {
    std::ofstream os(package_root / "models" / "model_1" / "variant_2" / "variant.json", std::ios::binary);
    os << R"({ "filename": "mul_16.onnx" })";
  }

  ModelPackageContext ctx(package_root);
  const auto& variants = ctx.GetVariantInfos();

  ASSERT_EQ(variants.size(), 2u);

  std::unordered_map<std::string, const VariantInfo*> by_file;
  for (const auto& v : variants) {
    ASSERT_TRUE(v.file.has_value());
    by_file.emplace(v.file->model_file_path.filename().string(), &v);
  }

  ASSERT_EQ(by_file.count("mul_1.onnx"), 1u);
  ASSERT_EQ(by_file.count("mul_16.onnx"), 1u);

  const auto* v1 = by_file.at("mul_1.onnx");
  EXPECT_EQ(v1->ep_compatibility.ep.value_or(""), "example_ep");
  EXPECT_EQ(v1->ep_compatibility.device.value_or(""), "cpu");

  const auto* v2 = by_file.at("mul_16.onnx");
  EXPECT_EQ(v2->ep_compatibility.ep.value_or(""), "example_ep");
  EXPECT_EQ(v2->ep_compatibility.device.value_or(""), "npu");

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
        "ep": "example_ep",
        "device": "cpu"
      }
    }
  })";

  {
    std::ofstream os(component_root / "metadata.json", std::ios::binary);
    os << metadata_json;
  }

  {
    std::ofstream os(variant_dir / "variant.json", std::ios::binary);
    os << R"({ "filename": "mul_1.onnx" })";
  }

  ModelPackageContext ctx(component_root);
  const auto& variants = ctx.GetVariantInfos();

  ASSERT_EQ(variants.size(), 1u);
  ASSERT_TRUE(variants[0].file.has_value());
  EXPECT_EQ(variants[0].file->model_file_path.filename().string(), "mul_1.onnx");

  EXPECT_EQ(variants[0].ep_compatibility.ep.value_or(""), "example_ep");
  EXPECT_EQ(variants[0].ep_compatibility.device.value_or(""), "cpu");

  std::filesystem::remove_all(component_root, ec);
}

// ------------------------------------------------------------------
// Tests for descriptor parser: enforced "ep" field in variant EP metadata.
// ------------------------------------------------------------------
namespace {

// Make a single-component, single-variant package on disk where metadata.json is written
// directly at the package root (the "single-component metadata flow" of the parser).
// In this flow variant EP metadata schema validation errors are propagated, instead of being
// swallowed by the manifest-driven discovery path which falls back to "Missing metadata variants".
// Returns the package_root.
std::filesystem::path MakeSingleComponentPackageWithMetadata(std::string_view subdir,
                                                             std::string_view metadata_json,
                                                             std::string_view variant_json = R"({"filename":"mul_1.onnx"})") {
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
  // The "ep" field is required in every variant descriptor.
  // Omitting it must yield a parse error (not silently accept a wildcard / portable variant).
  constexpr std::string_view metadata_json = R"({
    "component_name": "model_1",
    "variants": {
      "variant_1": {
        "device": "cpu",
        "compatibility_string": "anything"
      }
    }
  })";
  const auto package_root = MakeSingleComponentPackageWithMetadata(
      "ort_model_package_parser_missing_ep", metadata_json);

  try {
    ModelPackageContext ctx(package_root);
    FAIL() << "Expected exception for missing 'ep' field";
  } catch (const std::exception& e) {
    EXPECT_NE(std::string(e.what()).find("ep"), std::string::npos) << e.what();
  }

  std::error_code ec;
  std::filesystem::remove_all(package_root, ec);
}

TEST(ModelPackageTest, ParserRejects_EpCompatibilityNullEp) {
  constexpr std::string_view metadata_json = R"({
    "component_name": "model_1",
    "variants": {
      "variant_1": {
        "ep": null,
        "device": "cpu"
      }
    }
  })";
  const auto package_root = MakeSingleComponentPackageWithMetadata(
      "ort_model_package_parser_null_ep", metadata_json);

  try {
    ModelPackageContext ctx(package_root);
    FAIL() << "Expected exception for null 'ep' field";
  } catch (const std::exception& e) {
    EXPECT_NE(std::string(e.what()).find("ep"), std::string::npos) << e.what();
  }

  std::error_code ec;
  std::filesystem::remove_all(package_root, ec);
}

TEST(ModelPackageTest, ParserRejects_EpCompatibilityEmptyEp) {
  constexpr std::string_view metadata_json = R"({
    "component_name": "model_1",
    "variants": {
      "variant_1": {
        "ep": "",
        "device": "cpu"
      }
    }
  })";
  const auto package_root = MakeSingleComponentPackageWithMetadata(
      "ort_model_package_parser_empty_ep", metadata_json);

  try {
    ModelPackageContext ctx(package_root);
    FAIL() << "Expected exception for empty 'ep' field";
  } catch (const std::exception& e) {
    EXPECT_NE(std::string(e.what()).find("ep"), std::string::npos) << e.what();
  }

  std::error_code ec;
  std::filesystem::remove_all(package_root, ec);
}

// ------------------------------------------------------------------
// Tests for new pre-selection EP-compat traversal accessors.
// ------------------------------------------------------------------
TEST(ModelPackageApiTest, GetVariantEpName_ReturnsSingleEp) {
  const auto package_root = std::filesystem::temp_directory_path() / "ort_mp_pre_selection_ep_name";
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

  // Each variant declares a single EP.
  constexpr std::string_view metadata_json = R"({
    "component_name": "model_1",
    "variants": {
      "variant_1": {
        "ep": "example_ep",
        "device": "cpu"
      },
      "variant_2": {
        "ep": "other_ep",
        "device": "npu"
      }
    }
  })";
  CreateComponentModelMetadata(package_root, "model_1", metadata_json);

  for (const auto& d : {variant1_dir, variant2_dir}) {
    std::ofstream os(d / "variant.json", std::ios::binary);
    os << R"({"filename":"mul_1.onnx"})";
  }

  const auto& pkg_api = GetModelPackageFns();

  auto context_deleter = [&pkg_api](OrtModelPackageContext* p) {
    if (p) pkg_api.ReleaseModelPackageContext(p);
  };
  std::unique_ptr<OrtModelPackageContext, decltype(context_deleter)> ctx(nullptr, context_deleter);
  OrtModelPackageContext* raw_ctx = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api.CreateModelPackageContext(package_root.c_str(), &raw_ctx));
  ctx.reset(raw_ctx);

  // variant_1 targets example_ep
  const char* ep1 = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api.ModelPackage_GetVariantEpName(
      ctx.get(), "model_1", "variant_1", &ep1));
  ASSERT_NE(ep1, nullptr);
  EXPECT_STREQ(ep1, "example_ep");

  // variant_2 targets other_ep
  const char* ep2 = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api.ModelPackage_GetVariantEpName(
      ctx.get(), "model_1", "variant_2", &ep2));
  ASSERT_NE(ep2, nullptr);
  EXPECT_STREQ(ep2, "other_ep");

  // Optional out-parameter: callers can pass NULL.
  ASSERT_ORTSTATUS_OK(pkg_api.ModelPackage_GetVariantEpName(
      ctx.get(), "model_1", "variant_1", nullptr));

  std::filesystem::remove_all(package_root, ec);
}

// ------------------------------------------------------------------
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

    const auto& pkg_api = GetModelPackageFns();

    auto options_deleter = [&pkg_api](OrtModelPackageOptions* p) { if (p) pkg_api.ReleaseModelPackageOptions(p); };
    auto context_deleter = [&pkg_api](OrtModelPackageContext* p) { if (p) pkg_api.ReleaseModelPackageContext(p); };
    auto component_context_deleter = [&pkg_api](OrtModelPackageComponentContext* p) {
      if (p) pkg_api.ReleaseModelPackageComponentContext(p);
    };
    std::unique_ptr<OrtModelPackageOptions, decltype(options_deleter)> mp_opts(nullptr, options_deleter);
    std::unique_ptr<OrtModelPackageContext, decltype(context_deleter)> ctx(nullptr, context_deleter);
    std::unique_ptr<OrtModelPackageComponentContext, decltype(component_context_deleter)> comp_ctx(nullptr, component_context_deleter);

    OrtModelPackageOptions* raw_mp_opts = nullptr;
    ASSERT_ORTSTATUS_OK(pkg_api.CreateModelPackageOptionsFromSessionOptions(*ort_env, session_options, &raw_mp_opts));
    mp_opts.reset(raw_mp_opts);

    OrtModelPackageContext* raw_ctx = nullptr;
    ASSERT_ORTSTATUS_OK(pkg_api.CreateModelPackageContext(package_root.c_str(), &raw_ctx));
    ctx.reset(raw_ctx);

    OrtModelPackageComponentContext* raw_comp_ctx = nullptr;
    ASSERT_ORTSTATUS_OK(pkg_api.SelectComponent(ctx.get(), "model_1", mp_opts.get(), &raw_comp_ctx));
    comp_ctx.reset(raw_comp_ctx);

    const ORTCHAR_T* selected_folder = nullptr;
    ASSERT_ORTSTATUS_OK(pkg_api.ModelPackageComponent_GetSelectedVariantFolderPath(comp_ctx.get(), &selected_folder));
    ASSERT_NE(selected_folder, nullptr);

    // Path looks like .../models/model_1/<variant_x> -- the folder name is the variant.
    const auto selected_variant_dir = std::filesystem::path(selected_folder).filename().string();
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
// Test: a variant's per-file `session_options` flow through OrtApis::AddSessionConfigEntry.
// We verify this by feeding a *known* typed key (session.intra_op_num_threads) a non-integer value:
// pre-change behavior would silently stuff it into AddConfigEntry and succeed; post-change
// behavior parses it via the typed dispatcher and fails CreateSession with a parse error.
// ------------------------------------------------------------------
TEST(ModelPackageTest, VariantSessionOptions_DispatchedThroughAddSessionConfigEntry) {
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
        "ep": "example_ep", "device": "cpu"
      }
    }
  })";
  CreateComponentModelMetadata(package_root, "model_1", metadata_json);

  // Per-file session_options assigns a typed key (session.intra_op_num_threads) a value that is not a
  // valid integer. Routing this through OrtApis::AddSessionConfigEntry (the new behavior) must reject it.
  {
    std::ofstream os(variant_dir / "variant.json", std::ios::binary);
    os << R"({
      "filename": "mul_1.onnx",
      "session_options": {
        "session.intra_op_num_threads": "not_an_int"
      }
    })";
  }

  RegisteredEpDeviceUniquePtr example_ep;
  ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_info, example_ep));
  Ort::ConstEpDevice plugin_ep_device(example_ep.get());

  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> ep_options;
  session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);

  const auto& pkg_api = GetModelPackageFns();

  auto options_deleter = [&pkg_api](OrtModelPackageOptions* p) { if (p) pkg_api.ReleaseModelPackageOptions(p); };
  auto context_deleter = [&pkg_api](OrtModelPackageContext* p) { if (p) pkg_api.ReleaseModelPackageContext(p); };
  auto component_context_deleter = [&pkg_api](OrtModelPackageComponentContext* p) {
    if (p) pkg_api.ReleaseModelPackageComponentContext(p);
  };
  std::unique_ptr<OrtModelPackageOptions, decltype(options_deleter)> mp_opts(nullptr, options_deleter);
  std::unique_ptr<OrtModelPackageContext, decltype(context_deleter)> ctx(nullptr, context_deleter);
  std::unique_ptr<OrtModelPackageComponentContext, decltype(component_context_deleter)> comp_ctx(nullptr, component_context_deleter);

  OrtModelPackageOptions* raw_mp_opts = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api.CreateModelPackageOptionsFromSessionOptions(*ort_env, session_options, &raw_mp_opts));
  mp_opts.reset(raw_mp_opts);

  OrtModelPackageContext* raw_ctx = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api.CreateModelPackageContext(package_root.c_str(), &raw_ctx));
  ctx.reset(raw_ctx);

  OrtModelPackageComponentContext* raw_comp_ctx = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api.SelectComponent(ctx.get(), "model_1", mp_opts.get(), &raw_comp_ctx));
  comp_ctx.reset(raw_comp_ctx);

  // CreateSession iterates the per-file session_options and dispatches each through OrtApis::AddSessionConfigEntry.
  // The bad int value must surface as an error from this call.
  // Pass nullptr for session_options so the metadata-merge path runs (it is skipped when the caller
  // supplies their own session_options).
  OrtSession* raw_session = nullptr;
  OrtStatus* st = pkg_api.CreateSession(*ort_env, comp_ctx.get(), /*session_options=*/nullptr, &raw_session);
  // Clean up session first to avoid leaks if assertion fails.
  if (raw_session != nullptr) {
    Ort::GetApi().ReleaseSession(raw_session);
    raw_session = nullptr;
  }
  ASSERT_NE(st, nullptr) << "CreateSession unexpectedly succeeded with malformed session.intra_op_num_threads";
  const std::string err_msg = Ort::GetApi().GetErrorMessage(st);
  Ort::GetApi().ReleaseStatus(st);

  // Message should mention either AddSessionConfigEntry or the typed-int parse failure.
  const bool mentions_dispatch =
      err_msg.find("AddSessionConfigEntry") != std::string::npos ||
      err_msg.find("base-10 int32") != std::string::npos ||
      err_msg.find("intra_op_num_threads") != std::string::npos;
  EXPECT_TRUE(mentions_dispatch) << "error did not mention typed dispatch: " << err_msg;

  std::filesystem::remove_all(package_root, ec);
}

// ------------------------------------------------------------------
// Test: GetSelectedVariantFolderPath returns correct path even when variant.json is absent.
// ------------------------------------------------------------------
TEST(ModelPackageApiTest, FolderPath_ReturnsCorrectPath_WhenVariantJsonAbsent) {
  const auto package_root = std::filesystem::temp_directory_path() / "ort_mp_folder_path_no_variant_json";
  std::error_code ec;
  std::filesystem::remove_all(package_root, ec);
  std::filesystem::create_directories(package_root);

  CreateManifestJson(package_root, MakeManifestJson("model_1"));

  const auto variant_dir = package_root / "models" / "model_1" / "variant_1";
  std::filesystem::create_directories(variant_dir);

  // Copy a model file but do NOT create variant.json
  std::filesystem::copy_file("testdata/mul_1.onnx", variant_dir / "mul_1.onnx",
                             std::filesystem::copy_options::overwrite_existing, ec);

  constexpr std::string_view metadata_json = R"({
    "component_name": "model_1",
    "variants": {
      "variant_1": {
        "ep": "example_ep",
        "device": "cpu"
      }
    }
  })";
  CreateComponentModelMetadata(package_root, "model_1", metadata_json);

  RegisteredEpDeviceUniquePtr example_ep;
  ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_info, example_ep));
  Ort::ConstEpDevice plugin_ep_device(example_ep.get());

  Ort::SessionOptions so;
  std::unordered_map<std::string, std::string> ep_options;
  so.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);

  const auto& pkg_api = GetModelPackageFns();

  OrtModelPackageOptions* raw_mp_opts = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api.CreateModelPackageOptionsFromSessionOptions(*ort_env, so, &raw_mp_opts));
  auto options_deleter = [&pkg_api](OrtModelPackageOptions* p) { if (p) pkg_api.ReleaseModelPackageOptions(p); };
  std::unique_ptr<OrtModelPackageOptions, decltype(options_deleter)> mp_opts(raw_mp_opts, options_deleter);

  OrtModelPackageContext* raw_ctx = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api.CreateModelPackageContext(package_root.c_str(), &raw_ctx));
  auto context_deleter = [&pkg_api](OrtModelPackageContext* p) { if (p) pkg_api.ReleaseModelPackageContext(p); };
  std::unique_ptr<OrtModelPackageContext, decltype(context_deleter)> ctx(raw_ctx, context_deleter);

  OrtModelPackageComponentContext* raw_comp_ctx = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api.SelectComponent(ctx.get(), "model_1", mp_opts.get(), &raw_comp_ctx));
  auto component_context_deleter = [&pkg_api](OrtModelPackageComponentContext* p) {
    if (p) pkg_api.ReleaseModelPackageComponentContext(p);
  };
  std::unique_ptr<OrtModelPackageComponentContext, decltype(component_context_deleter)> comp_ctx(raw_comp_ctx, component_context_deleter);

  // GetSelectedVariantFolderPath should return the variant directory even without variant.json.
  const ORTCHAR_T* selected_folder = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api.ModelPackageComponent_GetSelectedVariantFolderPath(comp_ctx.get(), &selected_folder));
  ASSERT_NE(selected_folder, nullptr);

  const auto result_path = std::filesystem::path(selected_folder);
  EXPECT_FALSE(result_path.empty());
  EXPECT_EQ(result_path.filename().string(), "variant_1");

  std::filesystem::remove_all(package_root, ec);
}

}  // namespace test
}  // namespace onnxruntime
