// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <memory>
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

// Callback state for asynchronous pipeline creation. `pipeline` refers to caller-owned storage,
// so both this object and that storage must remain alive until the future completes.
struct PipelineCallbackContext {
  wgpu::ComputePipeline& pipeline;
  Status status;
};

class ProgramArtifact {
 public:
  ProgramArtifact(std::string program_name, wgpu::ComputePipeline&& compute_pipeline, std::vector<int>&& shape_uniform_ranks);

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

  // Starts building a compute pipeline for `program` and returns immediately. The caller must keep
  // both `callback_context` and `compute_pipeline` alive until `future` completes.
  Status Build(const ProgramBase& program,
               const ProgramMetadata& metadata,
               const std::span<uint32_t> inputs_segments,
               const std::span<uint32_t> outputs_segments,
               const std::string& program_key,
               uint32_t normalized_dispatch_x,
               uint32_t normalized_dispatch_y,
               uint32_t normalized_dispatch_z,
               wgpu::ComputePipeline& compute_pipeline,
               std::vector<int>& shape_uniform_ranks,
               wgpu::Future& future,
               std::unique_ptr<PipelineCallbackContext>& callback_context) const;
  const ProgramArtifact* Get(const std::string& key) const;
  const ProgramArtifact* Set(const std::string& key, ProgramArtifact&& program);

 private:
  std::unordered_map<std::string, ProgramArtifact> programs_;
  WebGpuContext& webgpu_context_;

  std::function<void(std::string_view)> shader_dump_fn_;
};

}  // namespace webgpu
}  // namespace onnxruntime
