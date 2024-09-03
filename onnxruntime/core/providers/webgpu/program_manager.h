// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

#include <string>
#include <unordered_map>

#include <webgpu/webgpu_cpp.h>

#include "core/common/common.h"

#include "core/providers/webgpu/program.h"

namespace onnxruntime {
class Tensor;

namespace webgpu {

class ProgramArtifact {
 public:
  ProgramArtifact(const ProgramBase& program, wgpu::ComputePipeline&& compute_pipeline, std::vector<int>&& shape_uniform_ranks);

  const std::string name;
  const wgpu::ComputePipeline compute_pipeline;
  const std::vector<int> shape_uniform_ranks;

  ProgramArtifact(ProgramArtifact&&) = default;
  ProgramArtifact& operator=(ProgramArtifact&&) = default;

 private:
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(ProgramArtifact);
};

class ProgramManager {
 public:
  ProgramManager(const wgpu::Device& device, const wgpu::Limits& limits) : device_(device), limits_(limits) {}

  Status NormalizeDispatchGroupSize(uint32_t& x, uint32_t& y, uint32_t& z) const;

  Status Build(const ProgramBase& program,
               const ProgramMetadata& metadata,
#ifndef NDEBUG  // if debug build
               const std::string& program_key,
#endif
               uint32_t normalized_dispatch_x,
               uint32_t normalized_dispatch_y,
               uint32_t normalized_dispatch_z,
               wgpu::ComputePipeline& compute_pipeline,
               std::vector<int>& shape_uniform_ranks) const;
  const ProgramArtifact* Get(const std::string& key) const;
  const ProgramArtifact* Set(const std::string& key, ProgramArtifact&& program);

 private:
  std::unordered_map<std::string, ProgramArtifact> programs_;
  const wgpu::Device& device_;
  const wgpu::Limits& limits_;
};

}  // namespace webgpu
}  // namespace onnxruntime
