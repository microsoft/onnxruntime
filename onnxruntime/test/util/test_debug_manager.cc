// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <cstring>
#include <fstream>

#include "core/graph/onnx_protobuf.h"
#include "core/framework/tensorprotoutils.h"

#include "test/util/include/test/test_environment.h"
#include "test/util/include/test_debug_manager.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Helper function to get an environment variable value
// Returns the value of the environment variable, or empty string if not set
std::string GetEnvVar(const char* name) {
#ifdef _WIN32
  // Use the secure _dupenv_s on Windows
  char* value = nullptr;
  size_t len = 0;
  errno_t err = _dupenv_s(&value, &len, name);

  // Handle error or not found case
  if (err != 0 || value == nullptr) {
    return std::string();
  }

  // Get the value as string
  std::string result(value);
  // free the memory allocated by _dupenv_s
  free(value);
  return result;
#else
  // Use getenv on non-Windows platforms
  const char* value = std::getenv(name);

  // Handle not found case
  if (value == nullptr) {
    return std::string();
  }

  return std::string(value);
#endif
}

// Helper function to check if an environment variable is set
bool IsEnvVarSet(const char* name) {
  std::string value = GetEnvVar(name);
  return !value.empty() && value != "0";
}

void UnitTestDebugManager::ParseEnvironmentVars() {
  const auto& logger = DefaultLoggingManager().DefaultLogger();
  if (IsEnvVarSet("ORT_UNIT_TEST_DUMP_ARTIFACTS")) {
    dump_artifacts_ = true;
    LOGS(logger, INFO) << "Enabled artifacts dumping.";
  }

  if (IsEnvVarSet("ORT_UNIT_TEST_ARTIFACTS_DIR")) {
    output_dir_ = GetEnvVar("ORT_UNIT_TEST_ARTIFACTS_DIR");
    LOGS(logger, INFO) << "Set output directory to: " << *output_dir_;
  }
}

std::filesystem::path UnitTestDebugManager::CreateTestcaseDir() {
  std::string test_suite_name = ::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name();
  std::string test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
  std::filesystem::path output_dir = GetOutputDir() / (test_suite_name + "_" + test_name);
  std::error_code ec;
  std::filesystem::create_directories(output_dir, ec);
  // Ignore error if directory already exists
  if (ec && !std::filesystem::exists(output_dir)) {
    ORT_THROW("Failed to create test case directory: " + output_dir.string() + " - " + ec.message());
  }

  return output_dir;
}

// Helper to save tensor data to a raw binary file
void UnitTestDebugManager::SaveTensorToFile(const std::string& tensor_name,
                                            const OrtValue& ort_val,
                                            const std::filesystem::path& dir) {
  const auto& logger = DefaultLoggingManager().DefaultLogger();
  if (!ort_val.IsTensor()) {
    LOGS(logger, WARNING) << "Cannot save non-tensor value to file: " << tensor_name;
    return;
  }
  const auto& tensor = ort_val.Get<onnxruntime::Tensor>();
  const size_t size_in_bytes = tensor.SizeInBytes();

  std::string sanitized_name = tensor_name;
  // Replace characters that might be problematic in filename
  std::replace(sanitized_name.begin(), sanitized_name.end(), '/', '_');
  std::replace(sanitized_name.begin(), sanitized_name.end(), '\\', '_');
  std::replace(sanitized_name.begin(), sanitized_name.end(), ':', '_');
  // Replace periods with underscores (handles path traversal sequences)
  std::replace(sanitized_name.begin(), sanitized_name.end(), '.', '_');

  // Serialize to .pb file
  std::filesystem::path filename = dir / (sanitized_name + ".pb");
  ONNX_NAMESPACE::TensorProto tensor_proto = utils::TensorToTensorProto(tensor, tensor_name);
  std::ofstream file(filename, std::ios::binary);
  ORT_ENFORCE(file.is_open(), "Failed to open file for writing: " + filename.string());
  ORT_ENFORCE(tensor_proto.SerializeToOstream(&file), "Failed to serialize tensor to file: " + filename.string());
  file.close();
  LOGS(logger, INFO) << "Saved tensor '" << tensor_name << "' to file: " << filename
                     << " (" << size_in_bytes << " bytes)";
}

// Helper to save all tensors in a map to raw files
void UnitTestDebugManager::SaveFeedsToFiles(const NameMLValMap& feeds, const std::filesystem::path& dir) {
  // Create a folder to save raw files
  std::filesystem::path raw_dir = dir / "inputs";
  std::error_code ec;
  std::filesystem::create_directories(raw_dir, ec);
  ORT_ENFORCE(!ec, "Failed to create inputs directory: " + raw_dir.string() + " - " + ec.message());
  for (const auto& kv : feeds) {
    SaveTensorToFile(kv.first, kv.second, raw_dir);
  }
}

void UnitTestDebugManager::SaveFetchesToFiles(const std::vector<std::string>& names,
                                              const std::vector<OrtValue>& values,
                                              const std::filesystem::path& dir,
                                              const std::string& provider_type) {
  // Validate that names and values have the same size
  ORT_ENFORCE(names.size() == values.size(),
              "Mismatch between names and values size: " +
                  std::to_string(names.size()) + " vs " + std::to_string(values.size()));
  // Create a folder to save raw files
  std::filesystem::path raw_dir = dir / (provider_type + "_" + "outputs");
  std::error_code ec;
  std::filesystem::create_directories(raw_dir, ec);
  ORT_ENFORCE(!ec, "Failed to create inputs directory: " + raw_dir.string() + " - " + ec.message());
  for (size_t i = 0; i < values.size(); ++i) {
    SaveTensorToFile(names[i], values[i], raw_dir);
  }
}

void UnitTestDebugManager::DumpArtifacts(const NameMLValMap& feeds,
                                         const std::vector<std::string>& output_names,
                                         const std::vector<std::pair<std::string, std::vector<OrtValue>>>& labeled_outputs) {
  // Dump inputs
  std::filesystem::path testcase_dir = CreateTestcaseDir();
  SaveFeedsToFiles(feeds, testcase_dir);
  // Dump outputs for each labeled case
  for (const auto& [label, values] : labeled_outputs) {
    SaveFetchesToFiles(output_names, values, testcase_dir, label);
  }
}

}  // namespace test
}  // namespace onnxruntime
