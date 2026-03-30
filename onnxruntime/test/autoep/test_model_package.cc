// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
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

std::filesystem::path CreateComponentModelMetadata(
    const std::filesystem::path& package_root,
    std::string_view component_model_name,
    std::string_view metadata_json) {
  const auto component_root = package_root / "models" / component_model_name;
  std::error_code ec;

  // Ensure component root and metadata.json exist
  std::filesystem::create_directories(component_root);

  const std::filesystem::path metadata_path = component_root / "metadata.json";
  std::ofstream os(metadata_path, std::ios::binary);
  os << metadata_json;

  return component_root;
}

}  // namespace

// ------------------------------------------------------------------
// Model package end-to-end test
// ------------------------------------------------------------------
TEST(ModelPackageTest, LoadModelPackageAndRunInference_PluginEp_AppendV2) {
  // Parse manifest alone to get model variants' constraints.
  // ORT selects most suitable model variant based on constraints and then loads it to run inference successfully.
  {
    // Build model package on disk
    const auto package_root = std::filesystem::temp_directory_path() / "ort_model_package_test";
    constexpr std::string_view manifest_json = R"({
    "model_name": "test_model",
    "component_models": {
      "model_1": {
        "model_variants": {
          "variant_1" : {
             "file": "mul_1.onnx",
             "constraints": {
               "ep": "example_ep",
               "device": "cpu",
               "architecture": "arch1"
             }
          },
          "variant_2" : {
            "file": "mul_16.onnx",
            "constraints": {
              "ep": "example_ep",
              "device": "npu",
              "architecture": "arch2"
            }
          }
        }
      }
    }
    })";

    CreateModelPackage(package_root, manifest_json,
                       "model_1", "variant_1", "variant_2",
                       std::filesystem::path{"testdata/mul_1.onnx"}, std::filesystem::path{"testdata/mul_16.onnx"});

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

  // Parse manifest and component model's metadata.json to get model variants' constraints.
  // ORT selects most suitable model variant based on constraints and then loads it to run inference successfully.
  {
    // Build model package on disk
    const auto package_root = std::filesystem::temp_directory_path() / "ort_model_package_test";
    constexpr std::string_view manifest_json = R"({
    "model_name": "test_model",
    "component_models": {
      "model_1": {
      }
    }
    })";

    CreateModelPackage(package_root, manifest_json,
                       "model_1", "variant_1", "variant_2",
                       std::filesystem::path{"testdata/mul_1.onnx"}, std::filesystem::path{"testdata/mul_16.onnx"});

    constexpr std::string_view metadata_json = R"({
      "model_component_name": "model_1",
      "model_variants": {
        "variant_1": {
          "file": "mul_1.onnx",
          "constraints": {
            "ep": "example_ep",
            "device": "cpu",
            "architecture": "arch1"
          }
        },
        "variant_2": {
          "file": "mul_16.onnx",
          "constraints": {
            "ep": "example_ep",
            "device": "npu",
            "architecture": "arch2"
          }
        }
      }
    })";

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
}

TEST(ModelPackageTest, LoadModelPackageAndRunInference_PreferCpu) {
  // Build model package on disk
  const auto package_root = std::filesystem::temp_directory_path() / "ort_model_package_test";
  constexpr std::string_view manifest_json = R"({
    "model_name": "test_model",
    "component_models": {
      "model_1": {
      }
    }
    })";

  CreateModelPackage(package_root, manifest_json,
                     "model_1", "variant_1", "variant_2",
                     std::filesystem::path{"testdata/mul_1.onnx"}, std::filesystem::path{"testdata/mul_16.onnx"});

  constexpr std::string_view metadata_json = R"({
      "model_component_name": "model_1",
      "model_variants": {
        "variant_1": {
          "file": "mul_1.onnx",
          "constraints": {
            "ep": "example_ep",
            "device": "cpu",
            "architecture": "arch1"
          }
        },
        "variant_2": {
          "file": "mul_16.onnx",
          "constraints": {
            "ep": "example_ep",
            "device": "npu",
            "architecture": "arch2"
          }
        }
      }
    })";

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
  constexpr std::string_view manifest_json = R"({
    "model_name": "test_model",
    "component_models": {
      "model_1": {
      }
    }
    })";

  CreateModelPackage(package_root, manifest_json,
                     "model_1", "variant_2", "variant_1",
                     std::filesystem::path{"testdata/mul_16.onnx"}, std::filesystem::path{"plugin_ep_compat_test.onnx"});

  constexpr std::string_view metadata_json = R"({
      "model_component_name": "model_1",
      "model_variants": {
        "variant_2": {
          "file": "mul_16.onnx",
          "constraints": {
            "ep": "example_ep",
            "device": "cpu",
            "architecture": "arch2",
            "ep_compatibility_info": "example_ep;version=0.1.0;ort_api_version=25;hardware_architecture=arch2"
          }
        },
        "variant_1": {
          "file": "plugin_ep_compat_test.onnx",
          "constraints": {
            "ep": "example_ep",
            "device": "cpu",
            "architecture": "arch1",
            "ep_compatibility_info": "example_ep;version=0.1.0;ort_api_version=25;hardware_architecture=arch1"
          }
        }
      }
    })";

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
  // manifest.json without "component_models"; discovery should scan models/* with metadata.json.
  const auto package_root = std::filesystem::temp_directory_path() / "ort_model_package_discover_test";
  constexpr std::string_view manifest_json = R"({
    "model_name": "test_model"
  })";

  CreateModelPackage(package_root, manifest_json,
                     "model_1", "variant_1", "variant_2",
                     std::filesystem::path{"testdata/mul_1.onnx"}, std::filesystem::path{"testdata/mul_16.onnx"});

  // Prepare component model with metadata and variants
  const std::string component_model_name = "model_1";
  constexpr std::string_view metadata_json = R"({
      "model_component_name": "model_1",
      "model_variants": {
        "variant_1": {
          "file": "mul_1.onnx",
          "constraints": {
            "ep": "example_ep",
            "device": "cpu",
            "architecture": "arch1"
          }
        },
        "variant_2": {
          "file": "mul_16.onnx",
          "constraints": {
            "ep": "example_ep",
            "device": "npu",
            "architecture": "arch2"
          }
        }
      }
    })";

  // Create metadata.json under models/model_1
  const auto component_root = CreateComponentModelMetadata(package_root,
                                                           component_model_name,
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

TEST(ModelPackageTest, ParseVariantFileResolution) {
  const auto package_root = std::filesystem::temp_directory_path() / "ort_model_package_parse_variant";
  std::error_code ec;

  // Base manifest with a single component/variant entry; we'll override per subcase.
  auto write_manifest = [&](std::string_view manifest_json) {
    CreateManifestJson(package_root, manifest_json);
  };

  // Subcase 1: "file" points to a directory containing exactly one .onnx file.
  {
    std::filesystem::remove_all(package_root, ec);
    constexpr std::string_view manifest_json = R"({
      "model_name": "test_model",
      "component_models": {
        "model_1": {
          "model_variants": {
            "variant_dir": {
              "file": "subdir",
              "constraints": {}
            }
          }
        }
      }
    })";
    write_manifest(manifest_json);

    const auto variant_dir = package_root / "models" / "model_1" / "variant_dir";
    const auto subdir = variant_dir / "subdir";
    std::filesystem::create_directories(subdir);
    std::filesystem::copy_file("testdata/mul_1.onnx", subdir / "only.onnx",
                               std::filesystem::copy_options::overwrite_existing, ec);

    ModelPackageDescriptorParser parser(logging::LoggingManager::DefaultLogger());
    std::vector<ModelVariantInfo> variants;
    auto status = parser.ParseManifestAndMetadata(package_root, variants);
    ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
    ASSERT_EQ(variants.size(), 1u);
    EXPECT_EQ(variants[0].model_path.filename().string(), "only.onnx");
  }

  // Subcase 2: No "file" field; discover the single .onnx in the variant directory.
  {
    std::filesystem::remove_all(package_root, ec);
    constexpr std::string_view manifest_json = R"({
      "model_name": "test_model",
      "component_models": {
        "model_1": {
          "model_variants": {
            "variant_autodiscover": {
              "constraints": {}
            }
          }
        }
      }
    })";
    write_manifest(manifest_json);

    const auto variant_dir = package_root / "models" / "model_1" / "variant_autodiscover";
    std::filesystem::create_directories(variant_dir);
    std::filesystem::copy_file("testdata/mul_1.onnx", variant_dir / "auto.onnx",
                               std::filesystem::copy_options::overwrite_existing, ec);

    ModelPackageDescriptorParser parser(logging::LoggingManager::DefaultLogger());
    std::vector<ModelVariantInfo> variants;
    auto status = parser.ParseManifestAndMetadata(package_root, variants);
    ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
    ASSERT_EQ(variants.size(), 1u);
    EXPECT_EQ(variants[0].model_path.filename().string(), "auto.onnx");
  }

  // Subcase 3: Multiple .onnx files -> error.
  {
    std::filesystem::remove_all(package_root, ec);
    constexpr std::string_view manifest_json = R"({
      "model_name": "test_model",
      "component_models": {
        "model_1": {
          "model_variants": {
            "variant_multi": {
              "constraints": {}
            }
          }
        }
      }
    })";
    write_manifest(manifest_json);

    const auto variant_dir = package_root / "models" / "model_1" / "variant_multi";
    std::filesystem::create_directories(variant_dir);
    std::filesystem::copy_file("testdata/mul_1.onnx", variant_dir / "a.onnx",
                               std::filesystem::copy_options::overwrite_existing, ec);
    std::filesystem::copy_file("testdata/mul_16.onnx", variant_dir / "b.onnx",
                               std::filesystem::copy_options::overwrite_existing, ec);

    ModelPackageDescriptorParser parser(logging::LoggingManager::DefaultLogger());
    std::vector<ModelVariantInfo> variants;
    auto status = parser.ParseManifestAndMetadata(package_root, variants);
    ASSERT_FALSE(status.IsOK());
    EXPECT_THAT(status.ErrorMessage(),
                ::testing::HasSubstr("Multiple ONNX model files found under"));
  }

  std::filesystem::remove_all(package_root, ec);
}

}  // namespace test
}  // namespace onnxruntime