// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>

namespace onnxruntime {
namespace {

const char* DiagnosticGpuProfilePath() {
  return std::getenv("ORT_WEBGPU_DIAGNOSTIC_GPU_PROFILE");
}

void WriteDiagnosticGpuProfile(const void* session, const profiling::Events& events,
                               const wgpu::AdapterInfo& adapter) {
  const char* path = DiagnosticGpuProfilePath();
  ORT_ENFORCE(path != nullptr && *path, "Missing diagnostic GPU profile path.");
  std::ofstream output(path, std::ios::app);
  ORT_ENFORCE(output.is_open(), "Cannot open diagnostic GPU profile: ", path);
  std::ostringstream session_id;
  session_id << session;
  output << "{\"session\":" << std::quoted(session_id.str())
         << ",\"adapter_vendor\":" << std::quoted(std::string(static_cast<std::string_view>(adapter.vendor)))
         << ",\"adapter_device\":" << std::quoted(std::string(static_cast<std::string_view>(adapter.device)))
         << ",\"kernels\":[";
  bool first = true;
  for (const auto& event : events) {
    if (!first) {
      output << ',';
    }
    first = false;
    output << "{\"name\":" << std::quoted(event.name)
           << ",\"dur\":" << event.dur << ",\"ts\":" << event.ts
           << ",\"shapes\":" << std::quoted(event.args.at("shapes"))
           << ",\"cache_key\":" << std::quoted(event.args.at("cache_key")) << '}';
  }
  output << "]}\n";
  output.flush();
  ORT_ENFORCE(output.good(), "Cannot write diagnostic GPU profile: ", path);
}

}  // namespace
}  // namespace onnxruntime
