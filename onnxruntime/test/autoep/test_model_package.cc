// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "core/session/model_package_context.h"
#include "core/session/abi_devices.h"
#include "test/autoep/test_autoep_utils.h"

#include "test/util/include/asserts.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {
namespace {

/*
OrtHardwareDevice MakeHardwareDevice(std::string vendor,
                                     OrtHardwareDeviceType type,
                                     uint32_t vendor_id,
                                     uint32_t device_id,
                                     std::map<std::string, std::string> metadata_entries = {}) {
  OrtHardwareDevice hd{};
  hd.type = type;
  hd.vendor_id = vendor_id;
  hd.device_id = device_id;
  hd.vendor = std::move(vendor);
  hd.metadata.CopyFromMap(std::move(metadata_entries));
  return hd;
}

SelectionEpInfo MakeSelectionEpInfo(const std::string& ep_name,
                                    const OrtHardwareDevice* hw) {
  SelectionEpInfo info{};
  info.ep_name = ep_name;
  info.ep_factory = nullptr;
  info.hardware_devices.push_back(hw);
  info.ep_metadata.push_back(nullptr);
  return info;
}

EpContextVariantInfo MakeComponent(const std::string& ep,
                                   const std::string& device,
                                   const std::string& arch,
                                   const std::filesystem::path& path) {
  EpContextVariantInfo c{};
  c.ep = ep;
  c.device = device;
  c.architecture = arch;
  c.model_path = path;
  return c;
}
*/

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
    const std::filesystem::path& source_model_1,
    const std::filesystem::path& source_model_2) {
  std::error_code ec;
  std::filesystem::remove_all(package_root, ec);
  std::filesystem::create_directories(package_root);

  CreateManifestJson(package_root, manifest_json);

  const auto models_root = package_root / "models" / "test_model";
  const auto variant1_dir = models_root / "variant_1";
  const auto variant2_dir = models_root / "variant_2";

  std::filesystem::create_directories(variant1_dir);
  std::filesystem::create_directories(variant2_dir);

  const auto variant1_model = variant1_dir / source_model_1.filename();
  const auto variant2_model = variant2_dir / source_model_2.filename();

  std::filesystem::copy_file(source_model_1, variant1_model, std::filesystem::copy_options::overwrite_existing, ec);
  std::filesystem::copy_file(source_model_2, variant2_model, std::filesystem::copy_options::overwrite_existing, ec);
  return package_root;
}

}  // namespace

// ------------------------------------------------------------------
// Model package end-to-end test
// ------------------------------------------------------------------

TEST(ModelPackageTest, LoadModelPackageAndRunInference_Basic) {
  // Build model package on disk
  const auto package_root = std::filesystem::temp_directory_path() / "ort_model_package_test";
  constexpr std::string_view manifest_json = R"({
    "name": "test_model",
    "components": [
      {
        "variant_name": "variant_1",
        "file": "mul_1.onnx",
        "constraints": {
          "ep": "example_ep",
          "device": "cpu",
          "architecture": "arch1"
        }
      },
      {
        "variant_name": "variant_2",
        "file": "mul_16.onnx",
        "constraints": {
          "ep": "example_ep",
          "device": "npu",
          "architecture": "arch2"
        }
      }
    ]
  })";

  CreateModelPackage(package_root, manifest_json,
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

}  // namespace test
}  // namespace onnxruntime