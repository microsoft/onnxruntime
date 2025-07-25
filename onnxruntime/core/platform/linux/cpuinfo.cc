// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/linux/cpuinfo.h"

#include <fstream>
#include <iostream>  // TODO for debugging - remove later
#include <map>
#include <string_view>

#include "core/common/string_utils.h"
#include "core/common/parse_string.h"

namespace onnxruntime {

namespace {
using KeyValuePairs = std::map<std::string, std::string, std::less<>>;

Status GetValue(const KeyValuePairs& key_value_pairs, std::string_view key,
                std::string_view& value) {
  auto it = key_value_pairs.find(key);
  ORT_RETURN_IF(it == key_value_pairs.end(), "Failed to find key: ", key);
  value = it->second;
  return Status::OK();
}
}  // namespace

Status ParseCpuInfoFile(const std::string& cpu_info_file, CpuInfo& cpu_info_out) {
  std::ifstream in{cpu_info_file};

  ORT_RETURN_IF_NOT(in, "Failed to open file: ", cpu_info_file);

  CpuInfo cpu_info{};
  KeyValuePairs key_value_pairs{};

  auto add_processor_info = [&]() -> Status {
    if (!key_value_pairs.empty()) {
      std::string_view value{};
      CpuInfoFileProcessorInfo processor_info{};

      ORT_RETURN_IF_ERROR(GetValue(key_value_pairs, "processor", value));
      ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale<size_t>(value, processor_info.processor));

      ORT_RETURN_IF_ERROR(GetValue(key_value_pairs, "vendor", value));
      processor_info.vendor_id = std::string{value};

      cpu_info.emplace_back(std::move(processor_info));

      key_value_pairs.clear();
    }
    return Status::OK();
  };

  for (std::string line{}; std::getline(in, line);) {
    std::cerr << "/proc/cpuinfo line: " << line << "\n";
    line = utils::TrimString(line);

    if (line.empty()) {
      ORT_RETURN_IF_ERROR(add_processor_info());
      continue;
    }

    auto parts = utils::SplitString(line, ":");
    ORT_RETURN_IF_NOT(parts.size() == 2, "Unexpected format. Line: '", line, "'");

    key_value_pairs.emplace(utils::TrimString(parts[0]), utils::TrimString(parts[1]));
  }

  ORT_RETURN_IF_ERROR(add_processor_info());

  cpu_info_out = std::move(cpu_info);
  return Status::OK();
}

}  // namespace onnxruntime
