// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/linux/cpuinfo.h"

#include <fstream>
#include <map>
#include <string_view>

#include "core/common/string_utils.h"
#include "core/common/parse_string.h"

namespace onnxruntime {

namespace {
using KeyValuePairs = std::map<std::string, std::string, std::less<>>;

bool TryGetValue(const KeyValuePairs& key_value_pairs, std::string_view key, std::string& value) {
  auto it = key_value_pairs.find(key);
  if (it == key_value_pairs.end()) {
    return false;
  }

  value = it->second;
  return true;
}
}  // namespace

Status ParseCpuInfoFile(const std::string& cpu_info_file, CpuInfo& cpu_info_out) {
  std::ifstream in{cpu_info_file};

  ORT_RETURN_IF_NOT(in, "Failed to open file: ", cpu_info_file);

  CpuInfo cpu_info{};
  KeyValuePairs key_value_pairs{};

  auto add_processor_info = [&]() -> Status {
    if (!key_value_pairs.empty()) {
      CpuInfoFileProcessorInfo processor_info{};

      {
        std::string processor_str{};
        ORT_RETURN_IF_NOT(TryGetValue(key_value_pairs, "processor", processor_str), "Failed to get processor value.");
        ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale<size_t>(processor_str, processor_info.processor));
      }

      // Try to get a vendor name.
      // This approach doesn't always work, e.g., for ARM processors.
      if (!TryGetValue(key_value_pairs, "vendor", processor_info.vendor_id)) {
        // TODO try something else?
      }

      cpu_info.emplace_back(std::move(processor_info));

      key_value_pairs.clear();
    }
    return Status::OK();
  };

  for (std::string line{}; std::getline(in, line);) {
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
