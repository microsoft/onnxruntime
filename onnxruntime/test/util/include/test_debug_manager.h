// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <filesystem>
#include <string>
#include <optional>
#include <vector>

#include "core/framework/ort_value.h"
#include "core/framework/framework_common.h"

namespace onnxruntime {
namespace test {

class UnitTestDebugManager {
 public:
  static UnitTestDebugManager& GetInstance() {
    static UnitTestDebugManager instance;
    return instance;
  }

  // Dump settings
  bool ShouldDumpArtifacts() const { return dump_artifacts_; }

  // Output directory for raw files
  std::filesystem::path GetOutputDir() const {
    std::filesystem::path output_dir_path(output_dir_.value_or("."));
    return output_dir_path;
  }

  // Create a directory to save artifacts
  std::filesystem::path CreateTestcaseDir();

  // Save tensor data to a raw binary file
  void SaveTensorToFile(const std::string& name,
                        const OrtValue& ort_val,
                        const std::filesystem::path& dir);

  // Save all input tensors in a map to raw files
  void SaveFeedsToFiles(const NameMLValMap& feeds, const std::filesystem::path& dir);

  // Save all output tensors into raw files
  void SaveFetchesToFiles(const std::vector<std::string>& names,
                          const std::vector<OrtValue>& values,
                          const std::filesystem::path& dir,
                          const std::string& provider_type);

  // Dump artifacts for model
  void DumpArtifacts(const NameMLValMap& feeds,
                     const std::vector<std::string>& output_names,
                     const std::vector<std::pair<std::string, std::vector<OrtValue>>>& labeled_outputs);

  // Delete copy and move constructors/operators
  UnitTestDebugManager(const UnitTestDebugManager&) = delete;
  UnitTestDebugManager& operator=(const UnitTestDebugManager&) = delete;
  UnitTestDebugManager(UnitTestDebugManager&&) = delete;
  UnitTestDebugManager& operator=(UnitTestDebugManager&&) = delete;

 private:
  UnitTestDebugManager() {
    ParseEnvironmentVars();
  }

  void ParseEnvironmentVars();

  // Dump flags
  bool dump_artifacts_ = false;

  // Output directory
  std::optional<std::string> output_dir_;
};

}  // namespace test
}  // namespace onnxruntime
