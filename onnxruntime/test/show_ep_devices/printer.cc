// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/show_ep_devices/printer.h"

#include <string_view>

#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace onnxruntime::show_ep_devices {

namespace {

constexpr int kIndentWidth = 2;

std::string_view DeviceTypeAsStr(OrtHardwareDeviceType device_type) {
  switch (device_type) {
    case OrtHardwareDeviceType_CPU:
      return "CPU";
    case OrtHardwareDeviceType_GPU:
      return "GPU";
    case OrtHardwareDeviceType_NPU:
      return "NPU";
    default:
      return "unknown";
  }
}

void PrintEpDeviceInfoAsJson(const std::vector<Ort::ConstEpDevice>& ep_devices,
                             std::ostream& output_stream) {
  auto ep_device_as_json = [](Ort::ConstEpDevice ep_device) -> json {
    const auto device = ep_device.Device();
    json j = {
        {"device", {
                       {"id", device.DeviceId()},
                       {"type", DeviceTypeAsStr(device.Type())},
                       {"vendor", device.Vendor()},
                       {"vendor_id", device.VendorId()},
                       {"metadata", device.Metadata().GetKeyValuePairs()},
                   }},
        {"ep_name", ep_device.EpName()},
        {"ep_vendor", ep_device.EpVendor()},
        {"ep_options", ep_device.EpOptions().GetKeyValuePairs()},
        {"ep_metadata", ep_device.EpMetadata().GetKeyValuePairs()},
    };
    return j;
  };

  auto ep_devices_json = json::array();

  for (const auto& ep_device : ep_devices) {
    ep_devices_json.push_back(ep_device_as_json(ep_device));
  }

  output_stream << ep_devices_json.dump(kIndentWidth) << "\n";
}

void PrintEpDeviceInfoAsText(const std::vector<Ort::ConstEpDevice>& ep_devices,
                             std::ostream& output_stream) {
  auto print_kvp_entries = [](Ort::ConstKeyValuePairs ort_key_value_pairs, size_t indent_level,
                              std::ostream& output_stream) {
    const auto& kvps = ort_key_value_pairs.GetKeyValuePairs();
    const auto ordered_kvps = std::map<std::string, std::string>(kvps.begin(), kvps.end());
    const auto indent = std::string(indent_level * kIndentWidth, ' ');
    for (const auto& [key, value] : ordered_kvps) {
      output_stream << indent << key << ": " << value << "\n";
    }
  };

  for (size_t i = 0; i < ep_devices.size(); ++i) {
    const auto& ep_device = ep_devices[i];
    const auto device = ep_device.Device();
    output_stream << "===== EP Device " << i << " =====\n"
                  << "EP Name: " << ep_device.EpName() << "\n"
                  << "EP Vendor: " << ep_device.EpVendor() << "\n"
                  << "EP Metadata:\n";
    print_kvp_entries(ep_device.EpMetadata(), 1, output_stream);
    output_stream << "Device:\n"
                  << "  Type: " << DeviceTypeAsStr(device.Type()) << "\n"
                  << "  ID: " << device.DeviceId() << "\n"
                  << "  Vendor: " << device.Vendor() << "\n"
                  << "  Metadata:\n";
    print_kvp_entries(device.Metadata(), 2, output_stream);
    output_stream << "\n";
  }
}

}  // namespace

std::optional<OutputFormat> ParseOutputFormat(std::string_view output_format_str) {
  if (output_format_str == "txt") {
    return OutputFormat::txt;
  }
  if (output_format_str == "json") {
    return OutputFormat::json;
  }
  return std::nullopt;
}

void PrintEpDeviceInfo(const std::vector<Ort::ConstEpDevice>& ep_devices, OutputFormat output_format,
                       std::ostream& output_stream) {
  switch (output_format) {
    case OutputFormat::json: {
      PrintEpDeviceInfoAsJson(ep_devices, output_stream);
      return;
    }
    case OutputFormat::txt: {
      PrintEpDeviceInfoAsText(ep_devices, output_stream);
      return;
    }
  }
}

}  // namespace onnxruntime::show_ep_devices
