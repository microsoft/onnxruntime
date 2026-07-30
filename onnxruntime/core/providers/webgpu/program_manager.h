// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>

#include "core/providers/webgpu/webgpu_external_header.h"

#include "core/common/common.h"

#include "core/providers/webgpu/program.h"

namespace onnxruntime {
class Tensor;

namespace webgpu {
class WebGpuContext;

class ProgramArtifact {
 public:
  ProgramArtifact(const ProgramBase& program, wgpu::ComputePipeline&& compute_pipeline, std::vector<int>&& shape_uniform_ranks);

  const std::string name;
  const wgpu::ComputePipeline compute_pipeline;
  const std::vector<int> shape_uniform_ranks;

  ProgramArtifact(ProgramArtifact&&) = default;
  ProgramArtifact& operator=(ProgramArtifact&&) = delete;

 private:
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(ProgramArtifact);
};

class ProgramManager {
 public:
  ProgramManager(WebGpuContext& webgpu_context);

  Status NormalizeDispatchGroupSize(uint32_t& x, uint32_t& y, uint32_t& z) const;
  Status CalculateSegmentsForInputsAndOutputs(const ProgramBase& program, std::vector<uint32_t>& inputs_segments, std::vector<uint32_t>& outputs_segments) const;

  Status Build(const ProgramBase& program,
               const ProgramMetadata& metadata,
               const std::span<uint32_t> inputs_segments,
               const std::span<uint32_t> outputs_segments,
               const std::string& program_key,
               uint32_t normalized_dispatch_x,
               uint32_t normalized_dispatch_y,
               uint32_t normalized_dispatch_z,
               wgpu::ComputePipeline& compute_pipeline,
               std::vector<int>& shape_uniform_ranks) const;
  // Pipeline cache lookup / insert. These are the only members shared across sessions, so both
  // are serialized. Build() is deliberately not: callers invoke it between Get() and Set(), which
  // keeps shader compilation - by far the expensive part - outside the lock. Two sessions racing
  // on the same key therefore compile it twice; Set() keeps the first artifact and the loser is
  // discarded, so the only cost is the duplicated compile.
  //
  // The pointer stays valid after the lock is released: this map is insert-only and
  // std::unordered_map keeps element references stable across rehash.
  const ProgramArtifact* Get(const std::string& key) const;
  const ProgramArtifact* Set(const std::string& key, ProgramArtifact&& program);

 private:
  mutable std::mutex programs_mutex_;
  std::unordered_map<std::string, ProgramArtifact> programs_;
  WebGpuContext& webgpu_context_;

  std::function<void(std::string_view)> shader_dump_fn_;
};

}  // namespace webgpu
}  // namespace onnxruntime
