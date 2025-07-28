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

std::string ArmCpuImplementerIdToVendorName(uint32_t implementer_id) {
  // ARM CPU implementer ids are copied from here:
  // https://github.com/torvalds/linux/blob/038d61fd642278bab63ee8ef722c50d10ab01e8f/arch/arm64/include/asm/cputype.h#L54-L64
  // https://github.com/torvalds/linux/blob/038d61fd642278bab63ee8ef722c50d10ab01e8f/arch/arm/include/asm/cputype.h#L65-L68

  switch (implementer_id) {
    case 0x41:
      return "ARM";
    case 0x42:
      return "Broadcom";
    case 0x44:
      return "DEC";
    case 0x43:
      return "Cavium";
    case 0x46:
      return "Fujitsu";
    case 0x48:
      return "HiSilicon";
    case 0x4E:
      return "Nvidia";
    case 0x50:
      return "APM";
    case 0x51:
      return "Qualcomm";
    case 0x61:
      return "Apple";
    case 0x69:
      return "Intel";
    case 0x6D:
      return "Microsoft";
    case 0xC0:
      return "Ampere";

    default:
      return "unknown";
  }
}
}  // namespace

Status ParseCpuInfoFile(const std::string& cpu_info_file, std::vector<CpuInfoFileProcessorInfo>& cpu_infos_out) {
  std::ifstream in{cpu_info_file};

  ORT_RETURN_IF_NOT(in, "Failed to open file: ", cpu_info_file);

  std::vector<CpuInfoFileProcessorInfo> cpu_infos{};
  KeyValuePairs key_value_pairs{};

  auto add_processor_info = [&]() -> Status {
    if (!key_value_pairs.empty()) {
      CpuInfoFileProcessorInfo processor_info{};

      {
        std::string processor_str{};
        ORT_RETURN_IF_NOT(TryGetValue(key_value_pairs, "processor", processor_str), "Failed to get processor value.");
        ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale<size_t>(processor_str, processor_info.processor));
      }

      // Try to get a vendor string.
      if (std::string vendor_id;
          TryGetValue(key_value_pairs, "vendor_id", vendor_id)) {
        processor_info.vendor = std::move(vendor_id);
      } else if (std::string implementer_id_str;
                 TryGetValue(key_value_pairs, "CPU implementer", implementer_id_str)) {
        const auto implementer_id = ParseStringWithClassicLocale<uint32_t>(implementer_id_str);
        processor_info.vendor = ArmCpuImplementerIdToVendorName(implementer_id);
      } else {
        processor_info.vendor = "unknown";
      }

      cpu_infos.emplace_back(std::move(processor_info));

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

  cpu_infos_out = std::move(cpu_infos);
  return Status::OK();
}

}  // namespace onnxruntime
