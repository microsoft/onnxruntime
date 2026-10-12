// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <span>
#include <string>
#include <mutex>

#include "core/common/inlined_containers.h"
#include "core/providers/webgpu/program.h"

namespace onnxruntime {
namespace webgpu {

std::string CalculateProgramCacheKey(const ProgramBase& program,
                                     std::span<uint32_t> inputs_segments,
                                     std::span<uint32_t> outputs_segments);

// Opt-in diagnostics only. Never expose the pointer-bearing binary key.
// Entries live with one ProgramManager; identifiers are unique across managers.
class ShaderDiagnosticIds final {
 public:
  ShaderDiagnosticIds() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ShaderDiagnosticIds);

  std::string Format(const std::string& key) const;

 private:
  mutable std::mutex mutex_;
  mutable InlinedHashMap<std::string, uint64_t> ids_;
};

}  // namespace webgpu
}  // namespace onnxruntime
