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

// Holds the target compute pipeline and status written by the asynchronous
// CreateComputePipelineAsync callback (used by deferred-dispatch). When a pipeline is built
// asynchronously, an instance of this must outlive the Build() call until the corresponding
// future is waited on.
struct PipelineCallbackContext {
  wgpu::ComputePipeline& pipeline;
  Status status;
};

class ProgramArtifact {
 public:
  ProgramArtifact(const ProgramBase& program, wgpu::ComputePipeline&& compute_pipeline, std::vector<int>&& shape_uniform_ranks);
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

  // Build a compute pipeline for the given program.
  //
  // By default (out_future == nullptr) this issues CreateComputePipelineAsync and waits
  // synchronously for completion, returning the ready pipeline in `compute_pipeline`.
  //
  // When `out_future` is provided (asynchronous mode, used by deferred-dispatch), the asynchronous
  // pipeline creation is issued but NOT waited on: the future is returned via `*out_future`
  // and ownership of the callback context (which the async callback writes the resulting
  // pipeline and status into) is returned via `*out_ctx`. The caller must keep `*out_ctx`
  // and the `compute_pipeline` storage alive until the future is waited on.
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
               wgpu::Future* out_future = nullptr,
               std::unique_ptr<PipelineCallbackContext>* out_ctx = nullptr) const;
  const ProgramArtifact* Get(const std::string& key) const;
  const ProgramArtifact* Set(const std::string& key, ProgramArtifact&& program);

 private:
  std::unordered_map<std::string, ProgramArtifact> programs_;
  WebGpuContext& webgpu_context_;

  std::function<void(std::string_view)> shader_dump_fn_;
};

}  // namespace webgpu
}  // namespace onnxruntime
