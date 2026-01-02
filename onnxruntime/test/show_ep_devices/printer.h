// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <optional>
#include <ostream>
#include <string_view>
#include <vector>

#include "onnxruntime_cxx_api.h"

namespace onnxruntime::show_ep_devices {

enum class OutputFormat {
  txt,
  json,
};

std::optional<OutputFormat> ParseOutputFormat(std::string_view output_format_str);

void PrintEpDeviceInfo(const std::vector<Ort::ConstEpDevice>& ep_devices,
                       OutputFormat output_format,
                       std::ostream& output_stream);

}  // namespace onnxruntime::show_ep_devices
