// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <map>
#include <string>

namespace onnxruntime {
namespace codegen {

// use log level warning as default to make sure logs are outputted
#define CODEGEN_SETTINGS_LOG_LEVEL WARNING

// This stores codegen settings to control dumps, execution preference, etc.
// CodeGenSettings could come from command line options or environment variables
// Or could come from a static variables in source code
class CodeGenSettings {
 public:
  // generic built-in options
  constexpr static const char* kDumpAllOptions = "dump_all_options";
  constexpr static const char* kCodeGenDumpModule = "codegen_dump_module";      // dump tvm module
  constexpr static const char* kCodeGenDumpLower = "codegen_dump_lower";        // dump lowered func
  constexpr static const char* kCodeGenDumpSchedule = "codegen_dump_schedule";  // dump scheduler

  void InsertOptions(const std::map<std::string, std::string>& options);
  void DumpOptions() const;
  std::string GetOptionValue(const std::string& key) const;
  bool HasOption(const std::string& key) const;
  bool OptionMatches(const std::string& key, const std::string& value) const;
  void Clear();
  static CodeGenSettings& Instance();

 private:
  CodeGenSettings();

  std::map<std::string, std::string> options_;
};

}  // namespace codegen
}  // namespace onnxruntime
